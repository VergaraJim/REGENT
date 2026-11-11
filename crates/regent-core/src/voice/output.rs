//! The voice output service (spec §5.9): spoken replies, sentence by sentence.
//!
//! A reply streams in as text pieces. [`Speaker::push`] feeds them to a
//! [`SentenceSplitter`]; each complete sentence is synthesized by the synth
//! thread and handed to the playback thread, which queues it on an [`AudioOut`].
//! Both the synthesizer and the audio output are traits, so the whole service is
//! tested without hardware.
//!
//! Two threads do the work:
//!
//! - the synth thread owns the splitter, the loaded synthesizer and the lazy
//!   voice selection;
//! - the playback thread owns the audio output, tracks the shared `pending`
//!   counter and publishes [`SpeakerEvent::Speaking`] transitions.
//!
//! A shared generation counter makes [`Speaker::stop`] immediate: everything
//! produced or queued for an older generation is discarded, and the playback
//! thread stops the audio output as soon as it sees the stop message.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::speech_text::SentenceSplitter;

/// How long the microphone stays paused after playback ends.
pub const MIC_RESUME_DELAY_MS: u64 = 400;

/// Shown when the audio output cannot be opened or play a sentence.
pub const AUDIO_FAILED: &str = "Regent couldn't play sound. Check your speakers or headphones.";

/// Turns one sentence into mono samples in `[-1, 1]` and their sample rate.
pub trait Synthesizer: Send {
    /// Synthesizes `sentence`.
    fn synthesize(&mut self, sentence: &str) -> Result<(Vec<f32>, u32), String>;
}

/// Loads a synthesizer for a voice file.
pub type SynthFactory = Box<dyn Fn(&Path) -> Result<Box<dyn Synthesizer>, String> + Send + Sync>;

/// The audio device the playback thread writes to.
pub trait AudioOut: Send {
    /// Queues `samples` (mono, `sample_rate` Hz); it must not block long.
    fn play(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), String>;
    /// Drops every queued sample and stops playing now.
    fn stop(&mut self);
    /// Whether nothing is left to play.
    fn is_idle(&self) -> bool;
}

/// Opens the audio output.
pub type AudioOutFactory = Box<dyn Fn() -> Result<Box<dyn AudioOut>, String> + Send + Sync>;

/// Everything the speaker tells the outside world.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeakerEvent {
    /// Speech started (`true`) or finished (`false`).
    Speaking(bool),
    /// Something went wrong that the user should see.
    Error(String),
}

/// Receives [`SpeakerEvent`]s; called from the service threads.
pub type SpeakerSink = Box<dyn Fn(SpeakerEvent) + Send + Sync>;

/// What [`Speaker::spawn`] needs to run.
pub struct SpeakerDeps {
    /// Loads the synthesizer for a voice file.
    pub synth_factory: SynthFactory,
    /// Opens the audio output.
    pub audio_factory: AudioOutFactory,
    /// Receives speaking and error events.
    pub sink: SpeakerSink,
}

/// State shared by the handle and both service threads.
struct Shared {
    generation: AtomicU64,
    pending: AtomicU64,
    speaking: AtomicBool,
    sink: SpeakerSink,
}

impl Shared {
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    fn bump_generation(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn pending(&self) -> u64 {
        self.pending.load(Ordering::SeqCst)
    }

    fn add_pending(&self) {
        self.pending.fetch_add(1, Ordering::SeqCst);
    }

    fn drop_pending(&self) {
        let _ = self
            .pending
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |value| {
                Some(value.saturating_sub(1))
            });
    }

    fn is_speaking(&self) -> bool {
        self.speaking.load(Ordering::SeqCst)
    }

    fn emit(&self, event: SpeakerEvent) {
        (self.sink)(event);
    }
}

/// What the handle asks the synth thread to do.
enum SynthMessage {
    /// Use `voice` from now on (`None` disables speech).
    SetVoice(Option<PathBuf>),
    /// A piece of the current reply.
    Push {
        generation: u64,
        text: String,
    },
    /// The reply is complete: flush the splitter.
    EndReply {
        generation: u64,
    },
    /// Drop queued text and the splitter.
    Stop,
    Shutdown,
}

/// What the synth and handle ask the playback thread to do.
enum PlaybackMessage {
    /// A synthesized sentence to queue on the audio output.
    Sentence {
        generation: u64,
        samples: Vec<f32>,
        sample_rate: u32,
    },
    /// Reset the per-voice audio error flag.
    ResetError,
    /// Stop the audio output as soon as possible.
    Stop,
    Shutdown,
}

/// The synth thread: owns the splitter and the lazily loaded synthesizer.
struct SynthWorker {
    shared: Arc<Shared>,
    factory: SynthFactory,
    playback_tx: Sender<PlaybackMessage>,
    rx: Receiver<SynthMessage>,
    splitter: SentenceSplitter,
    synthesizer: Option<Box<dyn Synthesizer>>,
    voice: Option<PathBuf>,
    /// A failed load suppresses pushes until the next `set_voice`.
    load_failed: bool,
    /// A synthesis error is reported at most once per reply.
    reply_error_sent: bool,
    /// Set by `end_reply`; the next push starts a fresh reply.
    awaiting_new_reply: bool,
}

impl SynthWorker {
    fn run(mut self) {
        while let Ok(message) = self.rx.recv() {
            match message {
                SynthMessage::Shutdown => return,
                SynthMessage::Stop => self.splitter.clear(),
                SynthMessage::SetVoice(voice) => self.set_voice(voice),
                SynthMessage::Push { generation, text } => self.push(generation, &text),
                SynthMessage::EndReply { generation } => self.end_reply(generation),
            }
        }
    }

    fn set_voice(&mut self, voice: Option<PathBuf>) {
        if voice != self.voice {
            self.voice = voice;
            self.synthesizer = None;
        }
        self.load_failed = false;
        self.reply_error_sent = false;
        self.awaiting_new_reply = false;
        if self.voice.is_none() {
            self.splitter.clear();
        }
    }

    fn push(&mut self, generation: u64, text: &str) {
        if generation != self.shared.generation() || self.voice.is_none() || self.load_failed {
            return;
        }
        if self.awaiting_new_reply {
            self.awaiting_new_reply = false;
            self.reply_error_sent = false;
        }
        for sentence in self.splitter.push(text) {
            self.synthesize_sentence(sentence, generation);
        }
    }

    fn end_reply(&mut self, generation: u64) {
        if generation != self.shared.generation() {
            return;
        }
        if self.voice.is_none() || self.load_failed {
            self.splitter.clear();
            self.awaiting_new_reply = true;
            return;
        }
        for sentence in self.splitter.finish() {
            self.synthesize_sentence(sentence, generation);
        }
        self.awaiting_new_reply = true;
    }

    /// Synthesizes one sentence and hands it to the playback thread.
    fn synthesize_sentence(&mut self, sentence: String, generation: u64) {
        if self.load_failed || self.voice.is_none() {
            return;
        }
        if generation != self.shared.generation() {
            return;
        }
        if self.synthesizer.is_none() {
            let Some(path) = self.voice.clone() else {
                return;
            };
            match (self.factory)(&path) {
                Ok(synth) => self.synthesizer = Some(synth),
                Err(message) => {
                    self.load_failed = true;
                    self.shared.emit(SpeakerEvent::Error(message));
                    return;
                }
            }
        }
        if generation != self.shared.generation() {
            return;
        }

        self.shared.add_pending();
        let result = self
            .synthesizer
            .as_mut()
            .expect("the synthesizer was just loaded")
            .synthesize(&sentence);

        if generation != self.shared.generation() {
            self.shared.drop_pending();
            return;
        }
        match result {
            Ok((samples, sample_rate)) => {
                let sent = self
                    .playback_tx
                    .send(PlaybackMessage::Sentence {
                        generation,
                        samples,
                        sample_rate,
                    })
                    .is_ok();
                if !sent {
                    self.shared.drop_pending();
                }
            }
            Err(message) => {
                self.shared.drop_pending();
                if !self.reply_error_sent {
                    self.reply_error_sent = true;
                    self.shared.emit(SpeakerEvent::Error(message));
                }
            }
        }
    }
}

/// The playback thread: owns the audio output and publishes speaking events.
fn playback_loop(shared: Arc<Shared>, rx: Receiver<PlaybackMessage>, factory: AudioOutFactory) {
    let mut audio: Option<Box<dyn AudioOut>> = None;
    let mut error_reported = false;
    let mut idle_since: Option<Instant> = None;

    loop {
        match rx.recv_timeout(Duration::from_millis(20)) {
            Ok(PlaybackMessage::Sentence {
                generation,
                samples,
                sample_rate,
            }) => {
                if generation != shared.generation() {
                    shared.drop_pending();
                    continue;
                }
                play_sentence(
                    &shared,
                    &factory,
                    &mut audio,
                    &mut error_reported,
                    &samples,
                    sample_rate,
                );
                shared.drop_pending();
            }
            Ok(PlaybackMessage::ResetError) => error_reported = false,
            Ok(PlaybackMessage::Stop) => {
                if let Some(out) = audio.as_mut() {
                    out.stop();
                }
                idle_since = None;
            }
            Ok(PlaybackMessage::Shutdown) => return,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
        update_speaking(&shared, audio.as_deref(), &mut idle_since);
    }
}

/// Opens the audio output on demand and queues one sentence.
fn play_sentence(
    shared: &Arc<Shared>,
    factory: &AudioOutFactory,
    audio: &mut Option<Box<dyn AudioOut>>,
    error_reported: &mut bool,
    samples: &[f32],
    sample_rate: u32,
) {
    if audio.is_none() {
        // Always try again after a failure, even within the same reply: a
        // re-plugged device works without restarting Regent. Only the message
        // is reported once.
        match factory() {
            Ok(out) => *audio = Some(out),
            Err(_) => {
                if !*error_reported {
                    *error_reported = true;
                    shared.emit(SpeakerEvent::Error(AUDIO_FAILED.to_string()));
                }
                return;
            }
        }
    }
    let Some(out) = audio.as_mut() else {
        return;
    };
    if out.play(samples, sample_rate).is_err() {
        *audio = None;
        if !*error_reported {
            *error_reported = true;
            shared.emit(SpeakerEvent::Error(AUDIO_FAILED.to_string()));
        }
    }
}

/// Busy means a sentence is pending or the output still has audio.
fn update_speaking(
    shared: &Shared,
    audio: Option<&dyn AudioOut>,
    idle_since: &mut Option<Instant>,
) {
    let busy = shared.pending() > 0 || audio.is_some_and(|out| !out.is_idle());
    if busy {
        *idle_since = None;
        if !shared.is_speaking() {
            shared.speaking.store(true, Ordering::SeqCst);
            shared.emit(SpeakerEvent::Speaking(true));
        }
        return;
    }
    if !shared.is_speaking() {
        *idle_since = None;
        return;
    }
    let since = *idle_since.get_or_insert_with(Instant::now);
    if since.elapsed() >= Duration::from_millis(MIC_RESUME_DELAY_MS) {
        shared.speaking.store(false, Ordering::SeqCst);
        *idle_since = None;
        shared.emit(SpeakerEvent::Speaking(false));
    }
}

/// The voice output service.
pub struct Speaker {
    shared: Arc<Shared>,
    synth_tx: Sender<SynthMessage>,
    playback_tx: Sender<PlaybackMessage>,
    synth: Option<JoinHandle<()>>,
    playback: Option<JoinHandle<()>>,
}

impl Speaker {
    /// Starts the synth and playback threads.
    pub fn spawn(deps: SpeakerDeps) -> Speaker {
        let SpeakerDeps {
            synth_factory,
            audio_factory,
            sink,
        } = deps;
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            pending: AtomicU64::new(0),
            speaking: AtomicBool::new(false),
            sink,
        });
        let (synth_tx, synth_rx) = mpsc::channel();
        let (playback_tx, playback_rx) = mpsc::channel();

        let playback = {
            let shared = Arc::clone(&shared);
            thread::spawn(move || playback_loop(shared, playback_rx, audio_factory))
        };

        let synth = {
            let worker = SynthWorker {
                shared: Arc::clone(&shared),
                factory: synth_factory,
                playback_tx: playback_tx.clone(),
                rx: synth_rx,
                splitter: SentenceSplitter::new(),
                synthesizer: None,
                voice: None,
                load_failed: false,
                reply_error_sent: false,
                awaiting_new_reply: false,
            };
            thread::spawn(move || worker.run())
        };

        Speaker {
            shared,
            synth_tx,
            playback_tx,
            synth: Some(synth),
            playback: Some(playback),
        }
    }

    /// Selects the voice file; `None` stops speech and ignores further text.
    pub fn set_voice(&self, voice: Option<PathBuf>) {
        if voice.is_none() {
            self.stop();
        }
        let _ = self.playback_tx.send(PlaybackMessage::ResetError);
        let _ = self.synth_tx.send(SynthMessage::SetVoice(voice));
    }

    /// Feeds one piece of the current reply.
    pub fn push(&self, text: &str) {
        let generation = self.shared.generation();
        let _ = self.synth_tx.send(SynthMessage::Push {
            generation,
            text: text.to_string(),
        });
    }

    /// Marks the current reply complete so the splitter flushes.
    pub fn end_reply(&self) {
        let generation = self.shared.generation();
        let _ = self.synth_tx.send(SynthMessage::EndReply { generation });
    }

    /// Cuts speech now and discards everything queued.
    pub fn stop(&self) {
        self.shared.bump_generation();
        self.shared.pending.store(0, Ordering::SeqCst);
        if self.shared.speaking.swap(false, Ordering::SeqCst) {
            self.shared.emit(SpeakerEvent::Speaking(false));
        }
        let _ = self.playback_tx.send(PlaybackMessage::Stop);
        let _ = self.synth_tx.send(SynthMessage::Stop);
    }

    /// Whether audio is currently playing or about to play.
    pub fn is_speaking(&self) -> bool {
        self.shared.is_speaking()
    }
}

impl Drop for Speaker {
    fn drop(&mut self) {
        self.stop();
        let _ = self.synth_tx.send(SynthMessage::Shutdown);
        let _ = self.playback_tx.send(PlaybackMessage::Shutdown);
        if let Some(synth) = self.synth.take() {
            let _ = synth.join();
        }
        if let Some(playback) = self.playback.take() {
            let _ = playback.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Public fakes
// ---------------------------------------------------------------------------

/// What [`FakeSynth`] records across all loaded instances.
struct FakeSynthState {
    sentences: Vec<String>,
    load_error: Option<String>,
    fail_on: Option<String>,
}

/// A synthesizer that records sentences and returns silence-like samples.
pub struct FakeSynth;

impl FakeSynth {
    /// Creates a factory and a handle to inspect and steer it.
    pub fn factory() -> (SynthFactory, FakeSynthHandle) {
        let state = Arc::new(Mutex::new(FakeSynthState {
            sentences: Vec::new(),
            load_error: None,
            fail_on: None,
        }));
        let factory_state = Arc::clone(&state);
        let factory: SynthFactory = Box::new(move |_path| {
            let load_error = factory_state.lock().unwrap().load_error.clone();
            match load_error {
                Some(message) => Err(message),
                None => Ok(Box::new(FakeSynthInstance {
                    state: Arc::clone(&factory_state),
                }) as Box<dyn Synthesizer>),
            }
        });
        (factory, FakeSynthHandle { state })
    }
}

/// Controls and inspects a [`FakeSynth`].
#[derive(Clone)]
pub struct FakeSynthHandle {
    state: Arc<Mutex<FakeSynthState>>,
}

impl FakeSynthHandle {
    /// Every sentence passed to a synthesizer, in order.
    pub fn sentences(&self) -> Vec<String> {
        self.state.lock().unwrap().sentences.clone()
    }

    /// Makes the next factory call fail with `message`.
    pub fn set_load_error(&self, message: Option<String>) {
        self.state.lock().unwrap().load_error = message;
    }

    /// Makes synthesizing exactly `sentence` fail.
    pub fn fail_on(&self, sentence: Option<String>) {
        self.state.lock().unwrap().fail_on = sentence;
    }
}

struct FakeSynthInstance {
    state: Arc<Mutex<FakeSynthState>>,
}

impl Synthesizer for FakeSynthInstance {
    fn synthesize(&mut self, sentence: &str) -> Result<(Vec<f32>, u32), String> {
        let mut state = self.state.lock().unwrap();
        state.sentences.push(sentence.to_string());
        if state.fail_on.as_deref() == Some(sentence) {
            return Err(format!("could not synthesize {sentence:?}"));
        }
        // 160 samples per character: 10 ms of 16 kHz audio.
        let samples = vec![0.1; 160 * sentence.chars().count()];
        Ok((samples, 16_000))
    }
}

/// What [`FakeAudioOut`] records across all opened instances.
struct FakeAudioState {
    busy_until: Instant,
    plays: Vec<(usize, u32)>,
    stops: usize,
    opens: usize,
    /// How many of the next plays must fail as if the device had gone away.
    fail_plays: usize,
    open_error: Option<String>,
    last_end: Option<Instant>,
}

/// An audio output that simulates real-time playback.
pub struct FakeAudioOut;

impl FakeAudioOut {
    /// Creates a factory and a handle to inspect and steer it.
    pub fn factory() -> (AudioOutFactory, FakeAudioHandle) {
        let state = Arc::new(Mutex::new(FakeAudioState {
            busy_until: Instant::now(),
            plays: Vec::new(),
            stops: 0,
            opens: 0,
            fail_plays: 0,
            open_error: None,
            last_end: None,
        }));
        let factory_state = Arc::clone(&state);
        let factory: AudioOutFactory = Box::new(move || {
            let mut state = factory_state.lock().unwrap();
            if let Some(message) = state.open_error.clone() {
                return Err(message);
            }
            state.opens += 1;
            drop(state);
            Ok(Box::new(FakeAudioOutInstance {
                state: Arc::clone(&factory_state),
            }) as Box<dyn AudioOut>)
        });
        (factory, FakeAudioHandle { state })
    }
}

/// Controls and inspects a [`FakeAudioOut`].
#[derive(Clone)]
pub struct FakeAudioHandle {
    state: Arc<Mutex<FakeAudioState>>,
}

impl FakeAudioHandle {
    /// `(sample count, sample rate)` for every queued play.
    pub fn plays(&self) -> Vec<(usize, u32)> {
        self.state.lock().unwrap().plays.clone()
    }

    /// How many times the output was stopped.
    pub fn stops(&self) -> usize {
        self.state.lock().unwrap().stops
    }

    /// Makes the next open fail with `message`.
    pub fn set_open_error(&self, message: Option<String>) {
        self.state.lock().unwrap().open_error = message;
    }

    /// How many times the audio output was opened.
    pub fn opens(&self) -> usize {
        self.state.lock().unwrap().opens
    }

    /// Makes the next play fail, as if the output device had gone away.
    pub fn fail_next_play(&self) {
        self.state.lock().unwrap().fail_plays = 1;
    }

    /// When the audio queued so far is expected to finish, if anything played.
    pub fn last_audio_end(&self) -> Option<Instant> {
        self.state.lock().unwrap().last_end
    }
}

struct FakeAudioOutInstance {
    state: Arc<Mutex<FakeAudioState>>,
}

impl AudioOut for FakeAudioOutInstance {
    fn play(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), String> {
        let mut state = self.state.lock().unwrap();
        if state.fail_plays > 0 {
            state.fail_plays -= 1;
            return Err(AUDIO_FAILED.to_string());
        }
        let now = Instant::now();
        let start = if now >= state.busy_until {
            now
        } else {
            state.busy_until
        };
        let seconds = samples.len() as f64 / f64::from(sample_rate.max(1));
        state.busy_until = start + Duration::from_secs_f64(seconds);
        state.last_end = Some(state.busy_until);
        state.plays.push((samples.len(), sample_rate));
        Ok(())
    }

    fn stop(&mut self) {
        let mut state = self.state.lock().unwrap();
        state.busy_until = Instant::now();
        state.stops += 1;
    }

    fn is_idle(&self) -> bool {
        Instant::now() >= self.state.lock().unwrap().busy_until
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    fn voice_path() -> PathBuf {
        PathBuf::from("/voices/test.onnx")
    }

    fn wait_until<T>(mut probe: impl FnMut() -> Option<T>) -> T {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(value) = probe() {
                return value;
            }
            assert!(Instant::now() < deadline, "timed out waiting for a result");
            thread::sleep(Duration::from_millis(10));
        }
    }

    struct Harness {
        speaker: Speaker,
        synth: FakeSynthHandle,
        audio: FakeAudioHandle,
        events: Arc<StdMutex<Vec<(SpeakerEvent, Instant)>>>,
    }

    impl Harness {
        fn new() -> Harness {
            let (synth_factory, synth) = FakeSynth::factory();
            let (audio_factory, audio) = FakeAudioOut::factory();
            let events = Arc::new(StdMutex::new(Vec::new()));
            let sink_events = Arc::clone(&events);
            let sink: SpeakerSink =
                Box::new(move |event| sink_events.lock().unwrap().push((event, Instant::now())));
            let speaker = Speaker::spawn(SpeakerDeps {
                synth_factory,
                audio_factory,
                sink,
            });
            Harness {
                speaker,
                synth,
                audio,
                events,
            }
        }

        fn events(&self) -> Vec<(SpeakerEvent, Instant)> {
            self.events.lock().unwrap().clone()
        }

        fn speaking_events(&self) -> Vec<bool> {
            self.events()
                .into_iter()
                .filter_map(|(event, _)| match event {
                    SpeakerEvent::Speaking(value) => Some(value),
                    SpeakerEvent::Error(_) => None,
                })
                .collect()
        }

        fn errors(&self) -> Vec<String> {
            self.events()
                .into_iter()
                .filter_map(|(event, _)| match event {
                    SpeakerEvent::Error(message) => Some(message),
                    SpeakerEvent::Speaking(_) => None,
                })
                .collect()
        }
    }

    #[test]
    fn sentences_are_synthesized_and_played_in_order() {
        let harness = Harness::new();
        harness.speaker.set_voice(Some(voice_path()));
        harness.speaker.push("Hello there. How are");
        harness.speaker.push(" you? Fine");
        harness.speaker.end_reply();

        wait_until(|| (harness.synth.sentences().len() >= 3).then_some(()));
        assert_eq!(
            harness.synth.sentences(),
            ["Hello there.", "How are you?", "Fine"]
        );
        wait_until(|| (harness.audio.plays().len() >= 3).then_some(()));
        assert_eq!(harness.audio.plays().len(), 3);
    }

    #[test]
    fn markdown_and_code_fences_are_cleaned() {
        let harness = Harness::new();
        harness.speaker.set_voice(Some(voice_path()));
        harness.speaker.push("# Title\n```\ncode\n```\nDone.");
        harness.speaker.end_reply();

        wait_until(|| (harness.synth.sentences().len() >= 2).then_some(()));
        assert_eq!(harness.synth.sentences(), ["Title", "Done."]);
    }

    #[test]
    fn speaking_ends_after_the_resume_delay() {
        let harness = Harness::new();
        harness.speaker.set_voice(Some(voice_path()));
        harness.speaker.push("Hello there. Bye.");
        harness.speaker.end_reply();

        wait_until(|| (harness.audio.plays().len() >= 2).then_some(()));
        let last_end = wait_until(|| harness.audio.last_audio_end());
        wait_until(|| harness.speaking_events().contains(&false).then_some(()));

        assert!(!harness.speaker.is_speaking());
        assert_eq!(harness.speaking_events().first(), Some(&true));
        let (_, false_at) = harness
            .events()
            .into_iter()
            .find(|(event, _)| *event == SpeakerEvent::Speaking(false))
            .expect("a Speaking(false) event");
        assert!(
            false_at.duration_since(last_end) >= Duration::from_millis(380),
            "false arrived {:?} after the audio ended",
            false_at.duration_since(last_end)
        );
    }

    #[test]
    fn stop_cuts_playback_and_the_next_reply_works() {
        let harness = Harness::new();
        harness.speaker.set_voice(Some(voice_path()));
        for index in 0..10 {
            harness.speaker.push(&format!("Sentence {index}."));
        }
        harness.speaker.end_reply();

        wait_until(|| (!harness.audio.plays().is_empty()).then_some(()));
        harness.speaker.stop();
        let plays_at_stop = harness.audio.plays().len();
        thread::sleep(Duration::from_millis(300));
        assert_eq!(
            harness.audio.plays().len(),
            plays_at_stop,
            "no new plays after stop"
        );
        assert!(harness.audio.stops() >= 1);
        wait_until(|| harness.speaking_events().contains(&false).then_some(()));

        harness.speaker.push("A fresh reply.");
        harness.speaker.end_reply();
        wait_until(|| {
            harness
                .synth
                .sentences()
                .iter()
                .any(|sentence| sentence == "A fresh reply.")
                .then_some(())
        });
    }

    #[test]
    fn without_a_voice_nothing_happens() {
        let harness = Harness::new();
        harness.speaker.push("Hello there.");
        harness.speaker.end_reply();
        thread::sleep(Duration::from_millis(200));

        assert!(harness.synth.sentences().is_empty());
        assert!(harness.audio.plays().is_empty());
        assert!(harness.events().is_empty());
    }

    #[test]
    fn a_synth_load_error_is_reported_once() {
        let harness = Harness::new();
        harness.synth.set_load_error(Some("bad voice".to_string()));
        harness.speaker.set_voice(Some(voice_path()));
        harness.speaker.push("Hello there.");
        harness.speaker.end_reply();

        wait_until(|| (!harness.errors().is_empty()).then_some(()));
        thread::sleep(Duration::from_millis(200));
        assert_eq!(harness.errors(), ["bad voice"]);
        assert!(harness.audio.plays().is_empty());
    }

    #[test]
    fn an_audio_open_error_is_reported_once() {
        let harness = Harness::new();
        harness.audio.set_open_error(Some("no device".to_string()));
        harness.speaker.set_voice(Some(voice_path()));
        harness.speaker.push("Hello there.");
        harness.speaker.end_reply();

        wait_until(|| (!harness.errors().is_empty()).then_some(()));
        thread::sleep(Duration::from_millis(200));
        assert_eq!(harness.errors(), [AUDIO_FAILED.to_string()]);
        assert!(harness.audio.plays().is_empty());
    }

    #[test]
    fn a_failed_play_reopens_the_device_for_the_next_reply() {
        let harness = Harness::new();
        harness.audio.fail_next_play();
        harness.speaker.set_voice(Some(voice_path()));
        harness.speaker.push("Hello there.");
        harness.speaker.end_reply();

        wait_until(|| (!harness.errors().is_empty()).then_some(()));
        assert_eq!(harness.errors(), [AUDIO_FAILED.to_string()]);
        assert!(harness.audio.plays().is_empty());
        assert_eq!(harness.audio.opens(), 1);

        harness.speaker.push("A fresh reply.");
        harness.speaker.end_reply();

        wait_until(|| (!harness.audio.plays().is_empty()).then_some(()));
        assert_eq!(harness.audio.opens(), 2, "the device was opened again");
    }

    #[test]
    fn a_failed_sentence_is_skipped_and_reported_once() {
        let harness = Harness::new();
        harness.synth.fail_on(Some("Bad one.".to_string()));
        harness.speaker.set_voice(Some(voice_path()));
        harness.speaker.push("First good. Bad one. Last good.");
        harness.speaker.end_reply();

        wait_until(|| (harness.synth.sentences().len() >= 3).then_some(()));
        wait_until(|| (harness.audio.plays().len() >= 2).then_some(()));
        thread::sleep(Duration::from_millis(200));

        assert_eq!(harness.errors().len(), 1);
        assert_eq!(harness.audio.plays().len(), 2);
    }
}
