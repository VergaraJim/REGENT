//! The always-listening voice input service (spec §5.8).
//!
//! Audio flows from the microphone callbacks through an energy VAD; each
//! finished utterance is queued for the speech thread, transcribed, run
//! through the post-filter and published as a transcript. Status, level and
//! transcript events go to a single sink.
//!
//! Two threads do the work: a control thread owns the microphone and handles
//! commands, and a speech thread owns the loaded engine and drains the
//! utterance queue (one in flight plus up to [`MAX_QUEUED_UTTERANCES`]
//! waiting).

use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use serde::Serialize;

use super::engine::{MicCallbacks, MicFactory, MicStream, SttConfig, SttEngine, SttFactory};
use super::post_filter::clean_transcript;
use super::vad::{EnergyVad, Utterance, VoiceDetector};

/// The most utterances that may wait for the speech thread.
pub const MAX_QUEUED_UTTERANCES: usize = 3;

/// The number of input samples between two level reports: about 15 a second
/// at 16 kHz.
pub const LEVEL_INTERVAL_SAMPLES: usize = 1067;

/// Shown when [`VoiceInput::start`] is given a model file that is not there.
const MISSING_MODEL_MESSAGE: &str =
    "The speech model file couldn't be found. Choose it again in Settings.";

/// The status error left behind by a failed transcription.
const STT_FAILED_MESSAGE: &str = "Speech recognition failed for the last thing you said.";

/// Returned by [`VoiceInput::set_listening`] before the engine is loaded.
const NOT_READY_MESSAGE: &str = "Voice input isn't ready yet.";

/// Where the voice service is in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VoiceState {
    /// Stopped: no model, no microphone.
    Off,
    /// A model is being loaded.
    Loading,
    /// A model is loaded; the microphone may be opened.
    Ready,
    /// Something failed; `error` says what.
    Error,
}

/// The state the UI renders.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VoiceStatus {
    /// The lifecycle state.
    pub state: VoiceState,
    /// Whether the microphone is open.
    pub listening: bool,
    /// Whether samples are being discarded.
    pub paused: bool,
    /// What went wrong, when something did.
    pub error: Option<String>,
}

impl VoiceStatus {
    /// The state of a service that was never started.
    fn off() -> VoiceStatus {
        VoiceStatus {
            state: VoiceState::Off,
            listening: false,
            paused: false,
            error: None,
        }
    }
}

/// Everything the voice service tells the outside world.
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceEvent {
    /// The status changed.
    Status(VoiceStatus),
    /// The microphone level while speech is in progress.
    Level(f32),
    /// A cleaned transcript worth acting on.
    Transcript(String),
}

/// Receives [`VoiceEvent`]s; called from the service threads.
pub type VoiceSink = Box<dyn Fn(VoiceEvent) + Send + Sync>;

/// What [`VoiceInput::spawn`] needs to run.
pub struct VoiceInputDeps {
    /// Opens the microphone.
    pub mic_factory: MicFactory,
    /// Loads the speech engine.
    pub stt_factory: SttFactory,
    /// Receives status, level and transcript events.
    pub sink: VoiceSink,
}

/// The shared state and the event sink.
struct Shared {
    inner: Mutex<Inner>,
    sink: VoiceSink,
}

struct Inner {
    generation: u64,
    status: VoiceStatus,
    vad_reset: bool,
}

impl Shared {
    fn status(&self) -> VoiceStatus {
        self.inner.lock().unwrap().status.clone()
    }

    fn generation(&self) -> u64 {
        self.inner.lock().unwrap().generation
    }

    fn bump_generation(&self) -> u64 {
        let mut inner = self.inner.lock().unwrap();
        inner.generation += 1;
        inner.vad_reset = false;
        inner.generation
    }

    fn request_vad_reset(&self) {
        self.inner.lock().unwrap().vad_reset = true;
    }

    /// Takes a pending VAD reset request and reports whether input is paused.
    fn take_vad_reset(&self) -> (bool, bool) {
        let mut inner = self.inner.lock().unwrap();
        let reset = std::mem::take(&mut inner.vad_reset);
        (reset, inner.status.paused)
    }

    /// Stores `status` and publishes it when it differs from the snapshot.
    fn publish(&self, status: VoiceStatus) {
        let changed = {
            let mut inner = self.inner.lock().unwrap();
            if inner.status == status {
                None
            } else {
                inner.status = status.clone();
                Some(status)
            }
        };
        if let Some(status) = changed {
            (self.sink)(VoiceEvent::Status(status));
        }
    }

    /// Publishes `status` unless a newer start or stop superseded `generation`.
    fn publish_if_current(&self, generation: u64, status: VoiceStatus) {
        let changed = {
            let mut inner = self.inner.lock().unwrap();
            if inner.generation != generation || inner.status == status {
                None
            } else {
                inner.status = status.clone();
                Some(status)
            }
        };
        if let Some(status) = changed {
            (self.sink)(VoiceEvent::Status(status));
        }
    }

    fn emit(&self, event: VoiceEvent) {
        (self.sink)(event);
    }
}

/// One utterance waiting for the speech thread.
struct QueuedUtterance {
    generation: u64,
    utterance: Utterance,
}

/// What the control thread asks the speech thread to do.
enum SttCommand {
    /// Load an engine for `generation` from `config`.
    Load { generation: u64, config: SttConfig },
    /// Drop the engine and everything queued.
    Clear,
}

/// One item the speech thread takes off its queue.
enum SttWork {
    Command(SttCommand),
    Utterance(QueuedUtterance),
    Shutdown,
}

struct SttQueueInner {
    commands: VecDeque<SttCommand>,
    utterances: VecDeque<QueuedUtterance>,
    /// Utterances handed to the speech thread that it has not finished yet,
    /// including the one in flight, so the backlog is bounded even before the
    /// thread picks the first one up.
    unfinished: usize,
    shutdown: bool,
}

/// The speech thread's inbox: control commands and utterances.
struct SttQueue {
    inner: Mutex<SttQueueInner>,
    ready: Condvar,
}

impl SttQueue {
    fn new() -> SttQueue {
        SttQueue {
            inner: Mutex::new(SttQueueInner {
                commands: VecDeque::new(),
                utterances: VecDeque::new(),
                unfinished: 0,
                shutdown: false,
            }),
            ready: Condvar::new(),
        }
    }

    fn push_command(&self, command: SttCommand) {
        let mut inner = self.inner.lock().unwrap();
        inner.commands.push_back(command);
        self.ready.notify_all();
    }

    /// Queues `queued` unless one utterance is in flight and
    /// [`MAX_QUEUED_UTTERANCES`] more are waiting.
    fn push_utterance(&self, queued: QueuedUtterance) {
        let mut inner = self.inner.lock().unwrap();
        if inner.unfinished > MAX_QUEUED_UTTERANCES {
            return;
        }
        inner.unfinished += 1;
        inner.utterances.push_back(queued);
        self.ready.notify_all();
    }

    /// Records that the speech thread finished (or dropped) one utterance.
    fn finish_utterance(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.unfinished = inner.unfinished.saturating_sub(1);
    }

    /// Drops everything waiting, as [`VoiceInput::stop`] requires.
    fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.utterances.clear();
        inner.unfinished = 0;
    }

    fn shutdown(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.shutdown = true;
        self.ready.notify_all();
    }

    fn take(&self) -> SttWork {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if inner.shutdown {
                return SttWork::Shutdown;
            }
            if let Some(command) = inner.commands.pop_front() {
                return SttWork::Command(command);
            }
            if let Some(utterance) = inner.utterances.pop_front() {
                return SttWork::Utterance(utterance);
            }
            inner = self.ready.wait(inner).unwrap();
        }
    }
}

/// What the outside asks the control thread to do.
enum Command {
    Start(Option<SttConfig>),
    Stop,
    SetListening(bool, Sender<Result<(), String>>),
    SetPaused(bool),
    Loaded { generation: u64 },
    MicError(String),
    Shutdown,
}

/// The control thread: owns the microphone, serialises every status change.
struct Control {
    shared: Arc<Shared>,
    queue: Arc<SttQueue>,
    command_rx: Receiver<Command>,
    command_tx: Sender<Command>,
    mic_factory: MicFactory,
    mic: Option<Box<dyn MicStream>>,
}

impl Control {
    fn run(mut self) {
        while let Ok(command) = self.command_rx.recv() {
            if !self.handle(command) {
                return;
            }
        }
        self.shutdown();
    }

    /// Handles one command; returns false when the thread should exit.
    fn handle(&mut self, command: Command) -> bool {
        match command {
            Command::Start(config) => self.start(config),
            Command::Stop => self.stop(),
            Command::SetListening(on, reply) => {
                let _ = reply.send(self.set_listening(on));
            }
            Command::SetPaused(paused) => self.set_paused(paused),
            Command::Loaded { generation } => {
                let _ = self.loaded(generation);
            }
            Command::MicError(message) => self.mic_error(message),
            Command::Shutdown => {
                self.shutdown();
                return false;
            }
        }
        true
    }

    fn start(&mut self, config: Option<SttConfig>) {
        self.stop();
        let Some(config) = config else {
            return;
        };
        let generation = self.shared.bump_generation();
        if !config.model_path.exists() {
            self.shared.publish(VoiceStatus {
                state: VoiceState::Error,
                listening: false,
                paused: false,
                error: Some(MISSING_MODEL_MESSAGE.to_string()),
            });
            return;
        }
        self.shared.publish(VoiceStatus {
            state: VoiceState::Loading,
            listening: false,
            paused: false,
            error: None,
        });
        self.queue
            .push_command(SttCommand::Load { generation, config });
    }

    fn stop(&mut self) {
        self.mic = None;
        self.shared.bump_generation();
        self.queue.push_command(SttCommand::Clear);
        self.shared.publish(VoiceStatus::off());
    }

    /// Opens the microphone once the engine has loaded for `generation`.
    fn loaded(&mut self, generation: u64) -> Result<(), String> {
        if self.shared.generation() != generation {
            return Ok(());
        }
        self.open_mic()
    }

    fn set_listening(&mut self, on: bool) -> Result<(), String> {
        if !on {
            if self.mic.take().is_some() {
                let mut status = self.shared.status();
                status.listening = false;
                self.shared.publish(status);
            }
            return Ok(());
        }
        if self.shared.status().state != VoiceState::Ready {
            return Err(NOT_READY_MESSAGE.to_string());
        }
        if self.mic.is_some() {
            return Ok(());
        }
        self.open_mic()
    }

    fn open_mic(&mut self) -> Result<(), String> {
        let generation = self.shared.generation();
        let callbacks = self.callbacks(generation);
        match (self.mic_factory)(callbacks) {
            Ok(stream) => {
                self.mic = Some(stream);
                let mut status = self.shared.status();
                status.state = VoiceState::Ready;
                status.listening = true;
                status.error = None;
                self.shared.publish(status);
                Ok(())
            }
            Err(message) => {
                let mut status = self.shared.status();
                // The engine did load, so the service stays Ready; only the
                // microphone is missing.
                status.state = VoiceState::Ready;
                status.listening = false;
                status.error = Some(message.clone());
                self.shared.publish(status);
                Err(message)
            }
        }
    }

    fn mic_error(&mut self, message: String) {
        self.mic = None;
        let mut status = self.shared.status();
        status.listening = false;
        status.error = Some(message);
        self.shared.publish(status);
    }

    fn set_paused(&mut self, paused: bool) {
        let mut status = self.shared.status();
        if status.paused == paused {
            return;
        }
        status.paused = paused;
        if paused {
            self.shared.request_vad_reset();
        }
        self.shared.publish(status);
    }

    fn shutdown(&mut self) {
        self.mic = None;
        self.queue.shutdown();
    }

    fn callbacks(&self, generation: u64) -> MicCallbacks {
        let mut pipeline = Pipeline::new(
            Arc::clone(&self.shared),
            Arc::clone(&self.queue),
            generation,
        );
        let command_tx = self.command_tx.clone();
        MicCallbacks {
            on_samples: Box::new(move |samples| pipeline.push(samples)),
            on_error: Box::new(move |message| {
                let _ = command_tx.send(Command::MicError(message));
            }),
        }
    }
}

/// The audio-thread half of the pipeline: VAD, levels and the STT queue.
struct Pipeline {
    shared: Arc<Shared>,
    queue: Arc<SttQueue>,
    generation: u64,
    vad: EnergyVad,
    since_level: usize,
    was_active: bool,
}

impl Pipeline {
    fn new(shared: Arc<Shared>, queue: Arc<SttQueue>, generation: u64) -> Pipeline {
        Pipeline {
            shared,
            queue,
            generation,
            vad: EnergyVad::new(),
            since_level: 0,
            was_active: false,
        }
    }

    fn push(&mut self, samples: &[f32]) {
        let (reset, paused) = self.shared.take_vad_reset();
        if reset {
            if self.was_active || self.vad.is_active() {
                self.shared.emit(VoiceEvent::Level(0.0));
            }
            self.vad.reset();
            self.was_active = false;
            self.since_level = 0;
        }
        if paused {
            return;
        }

        let mut utterances = Vec::new();
        self.vad.push(samples, &mut utterances);

        // Levels first, so the closing 0.0 is out before the transcript the
        // speech thread may publish for the utterance that just ended.
        self.since_level += samples.len();
        if self.vad.is_active() {
            if self.since_level >= LEVEL_INTERVAL_SAMPLES {
                self.since_level = 0;
                self.shared.emit(VoiceEvent::Level(self.vad.last_level()));
            }
            self.was_active = true;
        } else if self.was_active {
            self.was_active = false;
            self.shared.emit(VoiceEvent::Level(0.0));
        }

        for utterance in utterances {
            self.queue.push_utterance(QueuedUtterance {
                generation: self.generation,
                utterance,
            });
        }
    }
}

/// The speech thread: owns the engine and consumes the queue.
fn stt_loop(
    shared: Arc<Shared>,
    queue: Arc<SttQueue>,
    factory: SttFactory,
    command_tx: Sender<Command>,
) {
    let mut engine: Option<Box<dyn SttEngine>> = None;
    loop {
        match queue.take() {
            SttWork::Shutdown => return,
            SttWork::Command(SttCommand::Clear) => {
                queue.clear();
                engine = None;
            }
            SttWork::Command(SttCommand::Load { generation, config }) => match factory(&config) {
                Ok(loaded) => {
                    if shared.generation() == generation {
                        engine = Some(loaded);
                        let _ = command_tx.send(Command::Loaded { generation });
                    }
                }
                Err(message) => {
                    shared.publish_if_current(
                        generation,
                        VoiceStatus {
                            state: VoiceState::Error,
                            listening: false,
                            paused: false,
                            error: Some(message),
                        },
                    );
                }
            },
            SttWork::Utterance(queued) => {
                transcribe_utterance(&shared, &mut engine, queued);
                queue.finish_utterance();
            }
        }
    }
}

/// Transcribes one utterance and publishes the outcome.
fn transcribe_utterance(
    shared: &Arc<Shared>,
    engine: &mut Option<Box<dyn SttEngine>>,
    queued: QueuedUtterance,
) {
    if queued.generation != shared.generation() {
        return;
    }
    let Some(engine) = engine.as_mut() else {
        return;
    };
    let Utterance {
        samples,
        rms,
        duration_ms,
    } = &queued.utterance;
    match engine.transcribe(samples) {
        Ok(text) => {
            if let Some(transcript) = clean_transcript(&text, *duration_ms, *rms) {
                shared.emit(VoiceEvent::Transcript(transcript));
                let mut status = shared.status();
                if status.error.as_deref() == Some(STT_FAILED_MESSAGE) {
                    status.error = None;
                    shared.publish(status);
                }
            }
        }
        Err(_) => {
            let mut status = shared.status();
            status.error = Some(STT_FAILED_MESSAGE.to_string());
            shared.publish(status);
        }
    }
}

/// The always-listening voice input service.
pub struct VoiceInput {
    shared: Arc<Shared>,
    command_tx: Sender<Command>,
    control: Option<JoinHandle<()>>,
    speech: Option<JoinHandle<()>>,
}

impl VoiceInput {
    /// Starts the service threads with the given `deps`.
    pub fn spawn(deps: VoiceInputDeps) -> VoiceInput {
        let VoiceInputDeps {
            mic_factory,
            stt_factory,
            sink,
        } = deps;
        let shared = Arc::new(Shared {
            inner: Mutex::new(Inner {
                generation: 0,
                status: VoiceStatus::off(),
                vad_reset: false,
            }),
            sink,
        });
        let queue = Arc::new(SttQueue::new());
        let (command_tx, command_rx) = mpsc::channel();

        let speech = {
            let shared = Arc::clone(&shared);
            let queue = Arc::clone(&queue);
            let command_tx = command_tx.clone();
            thread::spawn(move || stt_loop(shared, queue, stt_factory, command_tx))
        };

        let control = {
            let control = Control {
                shared: Arc::clone(&shared),
                queue,
                command_rx,
                command_tx: command_tx.clone(),
                mic_factory,
                mic: None,
            };
            thread::spawn(move || control.run())
        };

        VoiceInput {
            shared,
            command_tx,
            control: Some(control),
            speech: Some(speech),
        }
    }

    /// Loads `config` (or shuts the engine down with `None`).
    pub fn start(&self, config: Option<SttConfig>) {
        let _ = self.command_tx.send(Command::Start(config));
    }

    /// Drops the microphone and the engine and returns to Off.
    pub fn stop(&self) {
        let _ = self.command_tx.send(Command::Stop);
    }

    /// Opens or closes the microphone; waits for the result.
    pub fn set_listening(&self, on: bool) -> Result<(), String> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.command_tx
            .send(Command::SetListening(on, reply_tx))
            .map_err(|_| NOT_READY_MESSAGE.to_string())?;
        reply_rx.recv().map_err(|_| NOT_READY_MESSAGE.to_string())?
    }

    /// Discards incoming audio while `paused`.
    pub fn set_paused(&self, paused: bool) {
        let _ = self.command_tx.send(Command::SetPaused(paused));
    }

    /// The current snapshot, the same one the sink sees.
    pub fn status(&self) -> VoiceStatus {
        self.shared.status()
    }
}

impl Drop for VoiceInput {
    fn drop(&mut self) {
        let _ = self.command_tx.send(Command::Stop);
        let _ = self.command_tx.send(Command::Shutdown);
        if let Some(control) = self.control.take() {
            let _ = control.join();
        }
        if let Some(speech) = self.speech.take() {
            let _ = speech.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::engine::{FakeMic, FakeMicHandle, FakeStt, FakeSttHandle};
    use crate::voice::vad::test_signals::{noise, tone};
    use std::path::PathBuf;
    use std::time::{Duration, Instant};
    use tempfile::NamedTempFile;

    const RATE: u32 = 16_000;

    fn seconds(seconds: f32) -> usize {
        (seconds * RATE as f32) as usize
    }

    fn utterance_signal() -> Vec<f32> {
        let mut signal = noise(0.001, seconds(1.0));
        signal.extend(tone(0.3, 440.0, RATE, seconds(1.0)));
        signal.extend(noise(0.001, seconds(1.5)));
        signal
    }

    fn config(path: impl Into<PathBuf>) -> SttConfig {
        SttConfig {
            model_path: path.into(),
            language: "en".to_string(),
        }
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
        input: VoiceInput,
        mic: FakeMicHandle,
        stt: FakeSttHandle,
        events: Arc<Mutex<Vec<VoiceEvent>>>,
    }

    impl Harness {
        fn new(text: &str) -> Harness {
            let (mic_factory, mic) = FakeMic::new();
            let (stt_factory, stt) = FakeStt::factory(text);
            let events = Arc::new(Mutex::new(Vec::new()));
            let sink_events = Arc::clone(&events);
            let sink: VoiceSink = Box::new(move |event| sink_events.lock().unwrap().push(event));
            let input = VoiceInput::spawn(VoiceInputDeps {
                mic_factory,
                stt_factory,
                sink,
            });
            Harness {
                input,
                mic,
                stt,
                events,
            }
        }

        fn status(&self) -> VoiceStatus {
            self.input.status()
        }

        fn events(&self) -> Vec<VoiceEvent> {
            self.events.lock().unwrap().clone()
        }

        fn transcripts(&self) -> Vec<String> {
            self.events()
                .into_iter()
                .filter_map(|event| match event {
                    VoiceEvent::Transcript(text) => Some(text),
                    _ => None,
                })
                .collect()
        }

        fn levels(&self) -> Vec<f32> {
            self.events()
                .into_iter()
                .filter_map(|event| match event {
                    VoiceEvent::Level(level) => Some(level),
                    _ => None,
                })
                .collect()
        }

        fn statuses(&self) -> Vec<VoiceStatus> {
            self.events()
                .into_iter()
                .filter_map(|event| match event {
                    VoiceEvent::Status(status) => Some(status),
                    _ => None,
                })
                .collect()
        }

        fn start_with_model(&self) -> NamedTempFile {
            let model = NamedTempFile::new().unwrap();
            self.input.start(Some(config(model.path())));
            model
        }

        fn wait_ready(&self) -> VoiceStatus {
            wait_until(|| {
                let status = self.status();
                (status.state == VoiceState::Ready && status.listening).then_some(status)
            })
        }

        fn push_signal(&self, signal: &[f32]) {
            for chunk in signal.chunks(320) {
                self.mic.push(chunk);
            }
        }

        fn wait_for_transcribe_calls(&self, count: usize) -> Vec<usize> {
            wait_until(|| {
                let calls = self.stt.calls();
                (calls.len() >= count).then_some(calls)
            })
        }
    }

    #[test]
    fn start_with_none_is_off() {
        let harness = Harness::new("Hello.");
        harness.input.start(None);
        thread::sleep(Duration::from_millis(50));

        let status = harness.status();
        assert_eq!(status.state, VoiceState::Off);
        assert!(!status.listening);
        assert!(!status.paused);
        assert_eq!(status.error, None);
        assert_eq!(harness.mic.opens(), 0);
    }

    #[test]
    fn start_with_a_model_loads_and_listens() {
        let harness = Harness::new("Hello.");
        let _model = harness.start_with_model();
        let status = harness.wait_ready();
        assert_eq!(status.state, VoiceState::Ready);
        assert_eq!(status.error, None);
        assert_eq!(harness.mic.opens(), 1);

        let states: Vec<VoiceState> = harness.statuses().iter().map(|it| it.state).collect();
        let loading = states
            .iter()
            .position(|state| *state == VoiceState::Loading)
            .expect("loading was published");
        let ready = states
            .iter()
            .position(|state| *state == VoiceState::Ready)
            .expect("ready was published");
        assert!(loading < ready, "states: {states:?}");
    }

    #[test]
    fn an_utterance_gets_transcribed_and_cleaned() {
        let harness = Harness::new("  Hello   there.  ");
        let _model = harness.start_with_model();
        harness.wait_ready();

        harness.push_signal(&utterance_signal());
        let text = wait_until(|| harness.transcripts().into_iter().next());
        assert_eq!(text, "Hello there.");

        let calls = harness.wait_for_transcribe_calls(1);
        assert_eq!(calls.len(), 1);
        assert!(
            (25_600..=33_600).contains(&calls[0]),
            "{} samples",
            calls[0]
        );
    }

    #[test]
    fn a_stock_phrase_produces_no_transcript() {
        let harness = Harness::new("Thank you.");
        let _model = harness.start_with_model();
        harness.wait_ready();

        harness.push_signal(&utterance_signal());
        harness.wait_for_transcribe_calls(1);
        thread::sleep(Duration::from_millis(100));
        assert!(harness.transcripts().is_empty());
    }

    #[test]
    fn the_queue_holds_one_in_flight_and_three_waiting() {
        let harness = Harness::new("Hello there.");
        let _model = harness.start_with_model();
        harness.wait_ready();

        harness.stt.hold(true);
        for _ in 0..6 {
            harness.push_signal(&utterance_signal());
        }
        harness.stt.hold(false);

        harness.wait_for_transcribe_calls(4);
        thread::sleep(Duration::from_millis(200));
        assert_eq!(harness.stt.calls().len(), 4);
    }

    #[test]
    fn a_paused_utterance_is_dropped_and_the_next_one_works() {
        let harness = Harness::new("Hello there.");
        let _model = harness.start_with_model();
        harness.wait_ready();

        harness.input.set_paused(true);
        wait_until(|| harness.status().paused.then_some(()));
        harness.push_signal(&utterance_signal());
        thread::sleep(Duration::from_millis(200));
        assert!(harness.stt.calls().is_empty());
        assert!(harness.transcripts().is_empty());

        harness.input.set_paused(false);
        wait_until(|| (!harness.status().paused).then_some(()));
        harness.push_signal(&utterance_signal());
        let text = wait_until(|| harness.transcripts().into_iter().next());
        assert_eq!(text, "Hello there.");
    }

    #[test]
    fn a_microphone_that_cannot_open_reports_an_error() {
        let harness = Harness::new("Hello.");
        harness.mic.set_open_error(Some("no device".to_string()));
        let _model = harness.start_with_model();

        let status = wait_until(|| {
            let status = harness.status();
            (status.state == VoiceState::Ready).then_some(status)
        });
        assert!(!status.listening);
        assert_eq!(status.error.as_deref(), Some("no device"));

        let result = harness.input.set_listening(true);
        assert_eq!(result, Err("no device".to_string()));

        let status = harness.status();
        assert!(!status.listening);
        assert_eq!(status.error.as_deref(), Some("no device"));
        assert!(harness.mic.opens() >= 1);
    }

    #[test]
    fn set_listening_before_ready_is_refused() {
        let harness = Harness::new("Hello.");
        assert_eq!(
            harness.input.set_listening(true),
            Err("Voice input isn't ready yet.".to_string())
        );
        assert_eq!(harness.mic.opens(), 0);
    }

    #[test]
    fn a_microphone_failure_closes_the_mic_once() {
        let harness = Harness::new("Hello.");
        let _model = harness.start_with_model();
        harness.wait_ready();

        harness.mic.fail("device lost");
        wait_until(|| {
            let status = harness.status();
            (!status.listening && status.error.as_deref() == Some("device lost")).then_some(())
        });
        wait_until(|| (!harness.mic.is_open()).then_some(()));

        thread::sleep(Duration::from_millis(100));
        assert_eq!(harness.mic.opens(), 1);
        let errors = harness
            .statuses()
            .into_iter()
            .filter(|status| status.error.as_deref() == Some("device lost"))
            .count();
        assert_eq!(errors, 1);
    }

    #[test]
    fn a_load_error_lands_in_the_status() {
        let harness = Harness::new("Hello.");
        harness.stt.set_load_error(Some("bad model".to_string()));
        let model = NamedTempFile::new().unwrap();
        harness.input.start(Some(config(model.path())));

        let status = wait_until(|| {
            let status = harness.status();
            (status.state == VoiceState::Error).then_some(status)
        });
        assert_eq!(status.error.as_deref(), Some("bad model"));
        assert!(!status.listening);
        assert_eq!(harness.mic.opens(), 0);
    }

    #[test]
    fn stop_drops_the_microphone() {
        let harness = Harness::new("Hello.");
        let _model = harness.start_with_model();
        harness.wait_ready();
        assert!(harness.mic.is_open());

        harness.input.stop();
        wait_until(|| (!harness.mic.is_open()).then_some(()));
        let status = wait_until(|| {
            let status = harness.status();
            (status.state == VoiceState::Off).then_some(status)
        });
        assert!(!status.listening);
        assert!(!status.paused);
        assert_eq!(status.error, None);
    }

    #[test]
    fn levels_are_reported_while_speaking_and_end_at_zero() {
        let harness = Harness::new("Hello there.");
        let _model = harness.start_with_model();
        harness.wait_ready();

        harness.push_signal(&utterance_signal());
        wait_until(|| harness.transcripts().into_iter().next());

        let levels = harness.levels();
        assert!(
            (5..=25).contains(&levels.len()),
            "{} level events: {levels:?}",
            levels.len()
        );
        assert_eq!(levels.last().copied(), Some(0.0));
    }

    #[test]
    fn a_missing_model_file_is_reported() {
        let harness = Harness::new("Hello.");
        harness
            .input
            .start(Some(config("/definitely/not/here/model.bin")));

        let status = wait_until(|| {
            let status = harness.status();
            (status.state == VoiceState::Error).then_some(status)
        });
        assert_eq!(
            status.error.as_deref(),
            Some("The speech model file couldn't be found. Choose it again in Settings.")
        );
        assert!(!status.listening);
        assert_eq!(harness.mic.opens(), 0);
    }
}
