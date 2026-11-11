//! Test fakes for the vision service, like `FakeMic` and `FakeAudioOut` in
//! `voice/`: public, so later tasks can drive the service in their tests.

use std::sync::{Arc, Mutex};

use super::{ScreenCapture, ScreenImage, VisionConfig, VisionEngine, VisionFactory};

/// A controllable fake vision engine.
///
/// [`FakeVision::factory`] returns the factory the service loads an engine with
/// and a handle a test uses to inspect loads and questions, set the answer, and
/// make loading or answering fail.
pub struct FakeVision;

struct FakeVisionState {
    loads: usize,
    questions: Vec<String>,
    answer: String,
    load_error: Option<String>,
    answer_error: Option<String>,
}

/// A cloneable view of a [`FakeVision`].
#[derive(Clone)]
pub struct FakeVisionHandle {
    shared: Arc<Mutex<FakeVisionState>>,
}

impl FakeVisionHandle {
    /// How many times the factory was called.
    pub fn loads(&self) -> usize {
        self.shared.lock().unwrap().loads
    }

    /// The question passed to every completed answer call, in order.
    pub fn questions(&self) -> Vec<String> {
        self.shared.lock().unwrap().questions.clone()
    }

    /// Sets the text every answer call returns.
    pub fn set_answer(&self, answer: String) {
        self.shared.lock().unwrap().answer = answer;
    }

    /// Makes the factory fail with `error` (or succeed again with `None`).
    pub fn fail_load(&self, error: Option<String>) {
        self.shared.lock().unwrap().load_error = error;
    }

    /// Makes the answer call fail with `error` (or succeed again with `None`).
    pub fn fail_answer(&self, error: Option<String>) {
        self.shared.lock().unwrap().answer_error = error;
    }
}

impl FakeVision {
    /// A fake engine factory and the handle that drives it.
    ///
    /// The factory/handle pair is what the vision service and a test need, so
    /// this constructor cannot return `Self`.
    #[allow(clippy::new_ret_no_self)]
    pub fn factory() -> (VisionFactory, FakeVisionHandle) {
        let shared = Arc::new(Mutex::new(FakeVisionState {
            loads: 0,
            questions: Vec::new(),
            answer: "A text editor with code.".to_string(),
            load_error: None,
            answer_error: None,
        }));
        let handle = FakeVisionHandle {
            shared: Arc::clone(&shared),
        };
        let factory: VisionFactory = Box::new(move |_config: &VisionConfig| {
            let mut state = shared.lock().unwrap();
            state.loads += 1;
            if let Some(error) = state.load_error.clone() {
                return Err(error);
            }
            drop(state);
            Ok(Box::new(FakeVisionEngine {
                shared: Arc::clone(&shared),
            }) as Box<dyn VisionEngine>)
        });
        (factory, handle)
    }
}

struct FakeVisionEngine {
    shared: Arc<Mutex<FakeVisionState>>,
}

impl VisionEngine for FakeVisionEngine {
    fn answer(
        &mut self,
        _image: &ScreenImage,
        question: &str,
        _max_tokens: usize,
        _cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<String, String> {
        let mut state = self.shared.lock().unwrap();
        if let Some(error) = state.answer_error.clone() {
            return Err(error);
        }
        state.questions.push(question.to_string());
        Ok(state.answer.clone())
    }
}

struct FakeCaptureState {
    calls: usize,
    last_max_edge: u32,
    error: Option<String>,
}

/// A cloneable view of a fake screen capture.
#[derive(Clone)]
pub struct FakeCaptureHandle {
    shared: Arc<Mutex<FakeCaptureState>>,
}

impl FakeCaptureHandle {
    /// How many times the capture was called.
    pub fn calls(&self) -> usize {
        self.shared.lock().unwrap().calls
    }

    /// The `max_edge` of the most recent capture call, or 0 before one.
    pub fn last_max_edge(&self) -> u32 {
        self.shared.lock().unwrap().last_max_edge
    }

    /// Makes the capture fail with `error` (or succeed again with `None`).
    pub fn fail(&self, error: Option<String>) {
        self.shared.lock().unwrap().error = error;
    }
}

/// A fake screen capture and the handle that drives it.
///
/// The capture always returns a 2x2 image, or the configured error.
pub fn fake_capture() -> (ScreenCapture, FakeCaptureHandle) {
    let shared = Arc::new(Mutex::new(FakeCaptureState {
        calls: 0,
        last_max_edge: 0,
        error: None,
    }));
    let handle = FakeCaptureHandle {
        shared: Arc::clone(&shared),
    };
    let capture: ScreenCapture = Box::new(move |max_edge: u32| {
        let mut state = shared.lock().unwrap();
        state.calls += 1;
        state.last_max_edge = max_edge;
        if let Some(error) = state.error.clone() {
            return Err(error);
        }
        drop(state);
        ScreenImage::new(2, 2, vec![0u8; 2 * 2 * 3])
    });
    (capture, handle)
}
