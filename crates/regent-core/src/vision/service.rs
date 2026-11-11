//! The vision service: one worker thread owns the vision engine and loads it
//! on demand, dropping it again after a keep-alive period.
//!
//! The worker never holds a lock while answering: cancellation lives in a
//! shared atomic flag the caller passes to [`VisionService::look`]. Every state
//! change is reported through the vision sink, and only actual changes reach
//! it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use super::{
    VisionConfig, VisionDeps, VisionEngine, VisionSink, VisionState, VisionStatus,
    VISION_MAX_TOKENS, VISION_NOT_SET_UP, VISION_NO_ANSWER,
};

/// The message used when the worker thread has gone away.
const WORKER_GONE: &str = "Regent's vision service stopped unexpectedly. Restart Regent.";

/// The message for a cancelled look.
const CANCELLED: &str = "Cancelled.";

/// The keep-alive used if an engine is somehow loaded without a config.
const DEFAULT_KEEP_ALIVE: Duration = Duration::from_secs(60);

/// One request for the worker thread.
enum Command {
    /// Replace the configuration (or turn vision off with `None`).
    Configure {
        config: Option<VisionConfig>,
        reply: Sender<()>,
    },
    /// Look at the screen and answer a question.
    Look {
        question: String,
        cancel: Arc<AtomicBool>,
        reply: Sender<Result<String, String>>,
    },
    /// Leave the worker loop.
    Shutdown,
}

/// The handle the app uses to talk to the vision worker.
pub struct VisionService {
    commands: Sender<Command>,
    status: Arc<Mutex<VisionStatus>>,
    configured: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl VisionService {
    /// Starts the worker thread.
    ///
    /// The worker owns the vision engine and captures the screen only through
    /// the closures in `deps`. Every state change is passed to the sink.
    pub fn spawn(deps: VisionDeps) -> VisionService {
        let status = Arc::new(Mutex::new(VisionStatus {
            state: VisionState::Off,
            error: None,
        }));
        let configured = Arc::new(AtomicBool::new(false));
        let (commands, incoming) = mpsc::channel();

        let worker = {
            let status = Arc::clone(&status);
            let configured = Arc::clone(&configured);
            let VisionDeps {
                factory,
                capture,
                sink,
            } = deps;
            std::thread::spawn(move || {
                Worker {
                    factory,
                    capture,
                    sink,
                    status,
                    configured,
                    config: None,
                    engine: None,
                }
                .run(incoming);
            })
        };

        VisionService {
            commands,
            status,
            configured,
            worker: Some(worker),
        }
    }

    /// Applies a configuration, waiting for the worker to store it.
    ///
    /// `None` turns vision off. A config equal to the current one changes
    /// nothing, so a loaded engine stays loaded.
    pub fn configure(&self, config: Option<VisionConfig>) {
        let (reply, done) = mpsc::channel();
        if self
            .commands
            .send(Command::Configure { config, reply })
            .is_ok()
        {
            let _ = done.recv();
        }
    }

    /// Whether a complete configuration has been applied.
    pub fn is_configured(&self) -> bool {
        self.configured.load(Ordering::SeqCst)
    }

    /// Looks at the screen right now and answers `question`.
    ///
    /// Blocks until answered. The model is loaded on first use and kept for the
    /// configured keep-alive after every look.
    pub fn look(&self, question: &str, cancel: Arc<AtomicBool>) -> Result<String, String> {
        let (reply, answer) = mpsc::channel();
        self.commands
            .send(Command::Look {
                question: question.to_string(),
                cancel,
                reply,
            })
            .map_err(|_| WORKER_GONE.to_string())?;
        answer.recv().map_err(|_| WORKER_GONE.to_string())?
    }

    /// The service's current state.
    pub fn status(&self) -> VisionStatus {
        self.status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

impl Drop for VisionService {
    fn drop(&mut self) {
        let _ = self.commands.send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// The worker thread's private state.
struct Worker {
    factory: super::VisionFactory,
    capture: super::ScreenCapture,
    sink: VisionSink,
    status: Arc<Mutex<VisionStatus>>,
    configured: Arc<AtomicBool>,
    config: Option<VisionConfig>,
    engine: Option<Box<dyn VisionEngine>>,
}

impl Worker {
    /// The worker loop: an engine is dropped when no command arrives within the
    /// keep-alive; with no engine the loop waits for a command.
    fn run(mut self, commands: Receiver<Command>) {
        loop {
            let command = if self.engine.is_some() {
                let keep_alive = self
                    .config
                    .as_ref()
                    .map_or(DEFAULT_KEEP_ALIVE, |config| config.keep_alive);
                match commands.recv_timeout(keep_alive) {
                    Ok(command) => command,
                    Err(RecvTimeoutError::Timeout) => {
                        self.engine = None;
                        self.set_status(VisionState::Idle, None);
                        continue;
                    }
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            } else {
                match commands.recv() {
                    Ok(command) => command,
                    Err(_) => return,
                }
            };

            match command {
                Command::Configure { config, reply } => {
                    self.configure(config);
                    let _ = reply.send(());
                }
                Command::Look {
                    question,
                    cancel,
                    reply,
                } => {
                    let result = self.look(&question, &cancel);
                    let _ = reply.send(result);
                }
                Command::Shutdown => {
                    self.engine = None;
                    self.set_status(VisionState::Off, None);
                    return;
                }
            }
        }
    }

    /// Replaces the configuration, dropping any loaded engine only when the
    /// config actually changes.
    fn configure(&mut self, config: Option<VisionConfig>) {
        if config == self.config {
            return;
        }
        self.engine = None;
        self.config = config;
        self.configured
            .store(self.config.is_some(), Ordering::SeqCst);
        if self.config.is_some() {
            self.set_status(VisionState::Idle, None);
        } else {
            self.set_status(VisionState::Off, None);
        }
    }

    /// Loads the model on demand, then captures and answers.
    fn look(&mut self, question: &str, cancel: &AtomicBool) -> Result<String, String> {
        let Some(config) = self.config.clone() else {
            return Err(VISION_NOT_SET_UP.to_string());
        };

        if self.engine.is_none() {
            self.set_status(VisionState::Loading, None);
            let loaded = (self.factory)(&config);
            match loaded {
                Ok(engine) => {
                    self.engine = Some(engine);
                    self.set_status(VisionState::Ready, None);
                }
                Err(message) => {
                    self.set_status(VisionState::Idle, Some(message.clone()));
                    return Err(message);
                }
            }
        }

        if cancel.load(Ordering::SeqCst) {
            return Err(CANCELLED.to_string());
        }

        let captured = (self.capture)(config.max_image_edge);
        let image = match captured {
            Ok(image) => image,
            Err(message) => {
                self.set_status(VisionState::Ready, Some(message.clone()));
                return Err(message);
            }
        };

        let answer = match self.engine.as_mut() {
            Some(engine) => engine.answer(&image, question, VISION_MAX_TOKENS, cancel),
            None => return Err(VISION_NOT_SET_UP.to_string()),
        };
        drop(image);

        match answer {
            Ok(text) => {
                let text = text.trim().to_string();
                if text.is_empty() {
                    self.set_status(VisionState::Ready, Some(VISION_NO_ANSWER.to_string()));
                    Err(VISION_NO_ANSWER.to_string())
                } else {
                    self.set_status(VisionState::Ready, None);
                    Ok(text)
                }
            }
            Err(message) => {
                self.set_status(VisionState::Ready, Some(message.clone()));
                Err(message)
            }
        }
    }

    /// Publishes a state change to the snapshot and the sink, but only when the
    /// status actually changed.
    fn set_status(&self, state: VisionState, error: Option<String>) {
        let next = VisionStatus { state, error };
        {
            let mut current = self
                .status
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if *current == next {
                return;
            }
            *current = next.clone();
        }
        (self.sink)(next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::fake::{fake_capture, FakeCaptureHandle, FakeVision, FakeVisionHandle};
    use crate::vision::ScreenImage;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    const KEEP_ALIVE: Duration = Duration::from_millis(100);
    const WAIT: Duration = Duration::from_secs(5);

    /// An empty two-pixel image.
    fn tiny() -> ScreenImage {
        ScreenImage::new(2, 2, vec![0u8; 12]).unwrap()
    }

    fn config(keep_alive: Duration) -> VisionConfig {
        VisionConfig {
            model_path: PathBuf::from("vision.gguf"),
            projector_path: PathBuf::from("projector.gguf"),
            gpu_layers: 0,
            max_image_edge: 640,
            keep_alive,
        }
    }

    fn recording_sink() -> (VisionSink, Arc<Mutex<Vec<VisionStatus>>>) {
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let sink: VisionSink = {
            let recorded = Arc::clone(&recorded);
            Box::new(move |status| {
                recorded
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(status);
            })
        };
        (sink, recorded)
    }

    struct Harness {
        service: VisionService,
        vision: FakeVisionHandle,
        capture: FakeCaptureHandle,
        recorded: Arc<Mutex<Vec<VisionStatus>>>,
    }

    fn harness() -> Harness {
        let (factory, vision) = FakeVision::factory();
        let (capture, capture_handle) = fake_capture();
        let (sink, recorded) = recording_sink();
        let service = VisionService::spawn(VisionDeps {
            factory,
            capture,
            sink,
        });
        Harness {
            service,
            vision,
            capture: capture_handle,
            recorded,
        }
    }

    fn states(recorded: &Arc<Mutex<Vec<VisionStatus>>>) -> Vec<VisionState> {
        recorded
            .lock()
            .unwrap()
            .iter()
            .map(|status| status.state)
            .collect()
    }

    fn wait_for_state(service: &VisionService, state: VisionState) {
        let start = Instant::now();
        while service.status().state != state {
            assert!(
                start.elapsed() < WAIT,
                "timed out waiting for {state:?}; status: {:?}",
                service.status()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn not_configured_never_loads_or_captures() {
        let h = harness();
        assert!(!h.service.is_configured());
        let error = h
            .service
            .look("what is this?", Arc::new(AtomicBool::new(false)))
            .unwrap_err();
        assert_eq!(error, VISION_NOT_SET_UP);
        assert_eq!(h.vision.loads(), 0);
        assert_eq!(h.capture.calls(), 0);
    }

    #[test]
    fn first_look_loads_captures_and_answers() {
        let h = harness();
        h.service.configure(Some(config(WAIT)));

        let answer = h
            .service
            .look("what is this?", Arc::new(AtomicBool::new(false)))
            .unwrap();

        assert_eq!(answer, "A text editor with code.");
        assert_eq!(h.vision.loads(), 1);
        assert_eq!(h.capture.calls(), 1);
        assert_eq!(h.capture.last_max_edge(), 640);
        assert_eq!(h.vision.questions(), vec!["what is this?".to_string()]);
        assert_eq!(
            states(&h.recorded),
            vec![VisionState::Idle, VisionState::Loading, VisionState::Ready]
        );
    }

    #[test]
    fn a_second_look_inside_the_keep_alive_does_not_reload() {
        let h = harness();
        h.service.configure(Some(config(WAIT)));
        h.service
            .look("first?", Arc::new(AtomicBool::new(false)))
            .unwrap();
        h.service
            .look("second?", Arc::new(AtomicBool::new(false)))
            .unwrap();
        assert_eq!(h.vision.loads(), 1);
        assert_eq!(h.capture.calls(), 2);
        assert_eq!(
            h.vision.questions(),
            vec!["first?".to_string(), "second?".to_string()]
        );
    }

    #[test]
    fn the_engine_unloads_after_the_keep_alive_and_reloads() {
        let h = harness();
        h.service.configure(Some(config(KEEP_ALIVE)));
        h.service
            .look("first?", Arc::new(AtomicBool::new(false)))
            .unwrap();

        wait_for_state(&h.service, VisionState::Idle);
        assert!(h.service.status().error.is_none());

        h.service
            .look("second?", Arc::new(AtomicBool::new(false)))
            .unwrap();
        assert_eq!(h.vision.loads(), 2);
    }

    #[test]
    fn a_load_failure_is_reported_and_retried() {
        let h = harness();
        h.service.configure(Some(config(WAIT)));
        h.vision.fail_load(Some("no model".to_string()));

        let error = h
            .service
            .look("what is this?", Arc::new(AtomicBool::new(false)))
            .unwrap_err();
        assert_eq!(error, "no model");
        let status = h.service.status();
        assert_eq!(status.state, VisionState::Idle);
        assert_eq!(status.error.as_deref(), Some("no model"));
        assert_eq!(h.capture.calls(), 0);

        h.vision.fail_load(None);
        h.service
            .look("what is this?", Arc::new(AtomicBool::new(false)))
            .unwrap();
        assert_eq!(h.vision.loads(), 2);
    }

    #[test]
    fn a_capture_failure_keeps_the_engine_loaded() {
        let h = harness();
        h.service.configure(Some(config(WAIT)));
        h.capture.fail(Some("no screen".to_string()));

        let error = h
            .service
            .look("what is this?", Arc::new(AtomicBool::new(false)))
            .unwrap_err();
        assert_eq!(error, "no screen");
        assert_eq!(h.service.status().error.as_deref(), Some("no screen"));
        assert_eq!(h.vision.loads(), 1);

        h.capture.fail(None);
        h.service
            .look("what is this?", Arc::new(AtomicBool::new(false)))
            .unwrap();
        assert_eq!(h.vision.loads(), 1);
    }

    #[test]
    fn a_blank_answer_becomes_no_answer() {
        let h = harness();
        h.service.configure(Some(config(WAIT)));
        h.vision.set_answer("  ".to_string());

        let error = h
            .service
            .look("what is this?", Arc::new(AtomicBool::new(false)))
            .unwrap_err();
        assert_eq!(error, VISION_NO_ANSWER);
        assert_eq!(h.service.status().error.as_deref(), Some(VISION_NO_ANSWER));
    }

    #[test]
    fn configure_none_turns_vision_off_and_the_same_config_changes_nothing() {
        let h = harness();
        h.service.configure(Some(config(WAIT)));
        assert!(h.service.is_configured());

        h.service.configure(Some(config(WAIT)));
        h.service
            .look("what is this?", Arc::new(AtomicBool::new(false)))
            .unwrap();
        assert_eq!(h.vision.loads(), 1);

        h.service.configure(None);
        assert!(!h.service.is_configured());
        assert_eq!(h.service.status().state, VisionState::Off);
    }

    #[test]
    fn a_set_cancel_flag_skips_the_capture() {
        let h = harness();
        h.service.configure(Some(config(WAIT)));
        let cancel = Arc::new(AtomicBool::new(true));

        let error = h.service.look("what is this?", cancel).unwrap_err();
        assert_eq!(error, "Cancelled.");
        assert_eq!(h.capture.calls(), 0);
    }

    #[test]
    fn screen_image_rejects_a_wrong_buffer_length() {
        assert!(ScreenImage::new(2, 2, vec![0u8; 11]).is_err());
        assert!(ScreenImage::new(0, 2, Vec::new()).is_err());
        let image = tiny();
        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 2);
        assert_eq!(image.rgb().len(), 12);
    }
}
