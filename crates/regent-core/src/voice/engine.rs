//! The microphone and speech-to-text engines behind traits, plus fakes.
//!
//! [`VoiceInput`](super::VoiceInput) never touches a real device or model
//! directly: it is given a [`MicFactory`] and an [`SttFactory`], so every test
//! runs the whole pipeline with [`FakeMic`] and [`FakeStt`].

use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};

/// What speech-to-text needs to load a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SttConfig {
    /// The model file on disk.
    pub model_path: PathBuf,
    /// The language to recognise, as the engine names it.
    pub language: String,
}

/// A loaded speech-to-text engine.
///
/// [`transcribe`](SttEngine::transcribe) runs on the voice service's speech
/// thread, never on the audio thread, and sees mono `f32` samples at
/// [`STT_SAMPLE_RATE`](super::STT_SAMPLE_RATE).
pub trait SttEngine: Send {
    /// Transcribes one utterance, or explains why it could not.
    fn transcribe(&mut self, samples: &[f32]) -> Result<String, String>;
}

/// Loads an [`SttEngine`] from an [`SttConfig`]; called on the speech thread.
pub type SttFactory = Box<dyn Fn(&SttConfig) -> Result<Box<dyn SttEngine>, String> + Send + Sync>;

/// Receives mono audio samples; see [`MicCallbacks::on_samples`].
pub type SampleCallback = Box<dyn FnMut(&[f32]) + Send>;

/// Receives a device error message; see [`MicCallbacks::on_error`].
pub type ErrorCallback = Box<dyn FnMut(String) + Send>;

/// What a microphone calls back into the voice service.
pub struct MicCallbacks {
    /// Receives mono samples at [`STT_SAMPLE_RATE`](super::STT_SAMPLE_RATE),
    /// always from a non-realtime thread.
    pub on_samples: SampleCallback,
    /// Reports a device failure with a message fit for the user. Called at
    /// most once per stream and never after the stream is dropped.
    pub on_error: ErrorCallback,
}

/// An open microphone.
///
/// Dropping it closes the device and stops the callbacks.
pub trait MicStream: Send {}

/// Opens a microphone and starts feeding its [`MicCallbacks`].
pub type MicFactory = Box<dyn Fn(MicCallbacks) -> Result<Box<dyn MicStream>, String> + Send + Sync>;

/// A controllable fake microphone.
///
/// [`FakeMic::new`] returns the factory the service opens and a handle the
/// test uses to feed samples, fail the device, and inspect what happened.
pub struct FakeMic;

struct FakeMicShared {
    on_samples: Option<SampleCallback>,
    on_error: Option<ErrorCallback>,
    open_error: Option<String>,
    opens: usize,
    open: bool,
}

impl FakeMicShared {
    fn new() -> FakeMicShared {
        FakeMicShared {
            on_samples: None,
            on_error: None,
            open_error: None,
            opens: 0,
            open: false,
        }
    }
}

/// A cloneable view of a [`FakeMic`].
#[derive(Clone)]
pub struct FakeMicHandle {
    shared: Arc<Mutex<FakeMicShared>>,
}

impl FakeMicHandle {
    /// Calls the open stream's `on_samples` with `samples`; does nothing when
    /// no stream is open.
    pub fn push(&self, samples: &[f32]) {
        let mut shared = self.shared.lock().unwrap();
        if let Some(on_samples) = shared.on_samples.as_mut() {
            on_samples(samples);
        }
    }

    /// Calls the open stream's `on_error` once with `message`.
    pub fn fail(&self, message: &str) {
        let mut shared = self.shared.lock().unwrap();
        if let Some(mut on_error) = shared.on_error.take() {
            on_error(message.to_string());
        }
    }

    /// Makes the factory fail with `error` (or succeed again with `None`).
    pub fn set_open_error(&self, error: Option<String>) {
        self.shared.lock().unwrap().open_error = error;
    }

    /// How many times the factory was called.
    pub fn opens(&self) -> usize {
        self.shared.lock().unwrap().opens
    }

    /// Whether a stream is currently open.
    pub fn is_open(&self) -> bool {
        self.shared.lock().unwrap().open
    }
}

impl FakeMic {
    /// A fake microphone factory and the handle that drives it.
    ///
    /// The factory/handle pair is what the voice service and a test need, so
    /// this constructor cannot return `Self`.
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> (MicFactory, FakeMicHandle) {
        let shared = Arc::new(Mutex::new(FakeMicShared::new()));
        let handle = FakeMicHandle {
            shared: Arc::clone(&shared),
        };
        let factory: MicFactory = Box::new(move |callbacks: MicCallbacks| {
            let mut state = shared.lock().unwrap();
            state.opens += 1;
            if let Some(error) = state.open_error.clone() {
                return Err(error);
            }
            state.on_samples = Some(callbacks.on_samples);
            state.on_error = Some(callbacks.on_error);
            state.open = true;
            Ok(Box::new(FakeMicStream {
                shared: Arc::clone(&shared),
            }) as Box<dyn MicStream>)
        });
        (factory, handle)
    }
}

struct FakeMicStream {
    shared: Arc<Mutex<FakeMicShared>>,
}

impl MicStream for FakeMicStream {}

impl Drop for FakeMicStream {
    fn drop(&mut self) {
        let mut state = self.shared.lock().unwrap();
        state.open = false;
        state.on_samples = None;
        state.on_error = None;
    }
}

/// A fake speech-to-text engine.
///
/// [`FakeStt::factory`] returns the factory the service loads an engine with
/// and a handle that records calls, can refuse to load, and can hold
/// transcription so a test can fill the queue.
pub struct FakeStt;

struct FakeSttState {
    text: String,
    calls: Vec<usize>,
    load_error: Option<String>,
    held: bool,
}

struct FakeSttShared {
    state: Mutex<FakeSttState>,
    released: Condvar,
}

/// A cloneable view of a [`FakeStt`].
#[derive(Clone)]
pub struct FakeSttHandle {
    shared: Arc<FakeSttShared>,
}

impl FakeSttHandle {
    /// The sample count of every completed `transcribe` call, in order.
    pub fn calls(&self) -> Vec<usize> {
        self.shared.state.lock().unwrap().calls.clone()
    }

    /// Makes the factory fail with `error` (or succeed again with `None`).
    pub fn set_load_error(&self, error: Option<String>) {
        self.shared.state.lock().unwrap().load_error = error;
    }

    /// While held, `transcribe` blocks until released.
    pub fn hold(&self, held: bool) {
        let mut state = self.shared.state.lock().unwrap();
        state.held = held;
        if !held {
            self.shared.released.notify_all();
        }
    }
}

impl FakeStt {
    /// A fake engine factory whose transcripts are always `text`.
    pub fn factory(text: &str) -> (SttFactory, FakeSttHandle) {
        let shared = Arc::new(FakeSttShared {
            state: Mutex::new(FakeSttState {
                text: text.to_string(),
                calls: Vec::new(),
                load_error: None,
                held: false,
            }),
            released: Condvar::new(),
        });
        let handle = FakeSttHandle {
            shared: Arc::clone(&shared),
        };
        let factory: SttFactory = Box::new(move |_config: &SttConfig| {
            let state = shared.state.lock().unwrap();
            if let Some(error) = state.load_error.clone() {
                return Err(error);
            }
            drop(state);
            Ok(Box::new(FakeSttEngine {
                shared: Arc::clone(&shared),
            }) as Box<dyn SttEngine>)
        });
        (factory, handle)
    }
}

struct FakeSttEngine {
    shared: Arc<FakeSttShared>,
}

impl SttEngine for FakeSttEngine {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String, String> {
        let mut state = self.shared.state.lock().unwrap();
        while state.held {
            state = self.shared.released.wait(state).unwrap();
        }
        state.calls.push(samples.len());
        Ok(state.text.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn fake_mic_tracks_opens_and_reports_samples() {
        let (factory, handle) = FakeMic::new();
        let (tx, rx) = std::sync::mpsc::channel();
        let callbacks = MicCallbacks {
            on_samples: Box::new(move |samples| {
                let _ = tx.send(samples.to_vec());
            }),
            on_error: Box::new(|message| panic!("unexpected error: {message}")),
        };
        assert_eq!(handle.opens(), 0);
        let stream = factory(callbacks).unwrap();
        assert_eq!(handle.opens(), 1);
        assert!(handle.is_open());
        handle.push(&[0.5, -0.5]);
        assert_eq!(rx.recv().unwrap(), vec![0.5, -0.5]);
        drop(stream);
        assert!(!handle.is_open());
    }

    #[test]
    fn fake_mic_can_refuse_to_open() {
        let (factory, handle) = FakeMic::new();
        handle.set_open_error(Some("no device".to_string()));
        let callbacks = MicCallbacks {
            on_samples: Box::new(|_| {}),
            on_error: Box::new(|_| {}),
        };
        assert_eq!(factory(callbacks).err().as_deref(), Some("no device"));
        assert_eq!(handle.opens(), 1);
        assert!(!handle.is_open());
    }

    #[test]
    fn fake_stt_records_calls_and_can_hold() {
        let (factory, handle) = FakeStt::factory("hello");
        let config = SttConfig {
            model_path: PathBuf::from("model.bin"),
            language: "en".to_string(),
        };
        let mut engine = factory(&config).unwrap();
        assert_eq!(engine.transcribe(&[0.0; 42]).unwrap(), "hello");
        assert_eq!(handle.calls(), vec![42]);

        let mut held = factory(&config).unwrap();
        handle.hold(true);
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let text = held.transcribe(&[0.0; 7]).unwrap();
            let _ = tx.send(text);
        });
        std::thread::sleep(Duration::from_millis(50));
        assert!(rx.try_recv().is_err());
        handle.hold(false);
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), "hello");
        worker.join().unwrap();
        assert_eq!(handle.calls(), vec![42, 7]);
    }

    #[test]
    fn fake_stt_can_refuse_to_load() {
        let (factory, handle) = FakeStt::factory("hello");
        handle.set_load_error(Some("bad model".to_string()));
        let config = SttConfig {
            model_path: PathBuf::from("model.bin"),
            language: "en".to_string(),
        };
        assert_eq!(factory(&config).err().as_deref(), Some("bad model"));
    }
}
