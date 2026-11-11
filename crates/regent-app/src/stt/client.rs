//! The app-side speech-to-text engine: one helper process per loaded model.
//!
//! [`HelperStt`] sends one request frame per utterance and waits for the
//! matching JSON response. If the helper dies it is restarted once; a second
//! failure marks the engine dead so the user gets a single clear error instead
//! of a restart loop (decision 0005).

use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

use regent_core::voice::{SttConfig, SttEngine, SttFactory, STT_SAMPLE_RATE};

use crate::stt::process::{HelperProcess, SpawnError, HELPER_MISSING};
use crate::stt::wire::{encode_frame, Response};

/// How long the helper may take to load its model before it is killed.
pub const READY_TIMEOUT: Duration = Duration::from_secs(120);

/// How long one utterance may take before the helper is treated as dead.
pub const TRANSCRIBE_TIMEOUT: Duration = Duration::from_secs(120);

/// Shown when the helper dies twice and voice input must be restarted.
pub const HELPER_STOPPED: &str =
    "The speech helper stopped working. Restart Regent to use voice again.";

/// Shown when the helper could not load the selected model.
pub const MODEL_FAILED: &str =
    "The speech model couldn't be loaded. It may be damaged or not a Whisper model.";

/// Shown when loading the model takes longer than [`READY_TIMEOUT`].
pub const MODEL_TIMEOUT: &str = "The speech model took too long to load.";

/// Shown when the helper process cannot be launched at all.
pub const HELPER_START_FAILED: &str = "The speech helper couldn't start.";

/// Shown when the helper rejects one utterance but keeps running.
pub const UTTERANCE_FAILED: &str = "Speech recognition failed for the last thing you said.";

/// Starts a fresh helper process. Returns a user-facing message on failure.
pub type Spawner = Box<dyn FnMut() -> Result<HelperProcess, String> + Send>;

/// A loaded helper process implementing [`SttEngine`].
pub struct HelperStt {
    spawner: Spawner,
    process: Option<HelperProcess>,
    restarted: bool,
    dead: bool,
    next_id: u64,
}

/// What one attempt at talking to the helper produced.
///
/// `Rejected` is the helper's own `error` response: the request was bad, the
/// helper is still fine, so no restart is needed.
enum Attempt {
    Text(String),
    Rejected,
    Failed,
}

impl HelperStt {
    /// Spawn the helper for the first time.
    pub fn new(mut spawner: Spawner) -> Result<Self, String> {
        let process = spawner()?;
        Ok(Self {
            spawner,
            process: Some(process),
            restarted: false,
            dead: false,
            next_id: 1,
        })
    }

    fn take_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        id
    }

    fn attempt(&mut self, id: u64, samples: &[f32]) -> Attempt {
        let Some(process) = self.process.as_mut() else {
            return Attempt::Failed;
        };

        if process
            .write_frame(&encode_frame(id, STT_SAMPLE_RATE, samples))
            .is_err()
        {
            return Attempt::Failed;
        }

        let deadline = Instant::now() + TRANSCRIBE_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Attempt::Failed;
            }

            match process.recv_timeout(remaining) {
                Ok(Response::Transcript {
                    id: response_id,
                    text,
                    ..
                }) if response_id == id => return Attempt::Text(text),
                Ok(Response::Error {
                    id: response_id, ..
                }) if response_id == id => return Attempt::Rejected,
                // A response for another utterance, or a late `ready`, is skipped.
                Ok(_) => continue,
                Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => {
                    return Attempt::Failed;
                }
            }
        }
    }

    fn fail_or_restart(&mut self, id: u64, samples: &[f32]) -> Result<String, String> {
        if self.restarted {
            self.dead = true;
            self.process = None;
            return Err(HELPER_STOPPED.to_string());
        }

        self.restarted = true;
        match (self.spawner)() {
            Ok(process) => self.process = Some(process),
            Err(_) => {
                self.dead = true;
                self.process = None;
                return Err(HELPER_STOPPED.to_string());
            }
        }

        let retry_id = id;
        match self.attempt(retry_id, samples) {
            Attempt::Text(text) => Ok(text),
            Attempt::Rejected => Err(UTTERANCE_FAILED.to_string()),
            Attempt::Failed => {
                self.dead = true;
                self.process = None;
                Err(HELPER_STOPPED.to_string())
            }
        }
    }
}

impl SttEngine for HelperStt {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String, String> {
        if self.dead {
            return Err(HELPER_STOPPED.to_string());
        }

        let id = self.take_id();
        match self.attempt(id, samples) {
            Attempt::Text(text) => Ok(text),
            Attempt::Rejected => Err(UTTERANCE_FAILED.to_string()),
            Attempt::Failed => {
                let retry_id = self.take_id();
                self.fail_or_restart(retry_id, samples)
            }
        }
    }
}

/// Build the [`SttFactory`] that spawns the bundled helper.
///
/// `helper` is the result of [`crate::stt::resolve_helper`], so a missing helper
/// is reported when the model is loaded rather than when the app starts.
pub fn stt_factory(helper: Result<PathBuf, String>, log_file: PathBuf) -> SttFactory {
    Box::new(move |config: &SttConfig| {
        let helper = match &helper {
            Ok(path) => path.clone(),
            Err(message) => return Err(message.clone()),
        };
        let log_file = log_file.clone();
        let model = config.model_path.clone();
        let language = config.language.clone();

        let spawner: Spawner = Box::new(move || {
            let mut command = Command::new(&helper);
            command
                .arg("--model")
                .arg(&model)
                .arg("--language")
                .arg(&language);

            HelperProcess::spawn(command, &log_file, READY_TIMEOUT)
                .map_err(|error| spawn_error_message(&error))
        });

        Ok(Box::new(HelperStt::new(spawner)?) as Box<dyn SttEngine>)
    })
}

fn spawn_error_message(error: &SpawnError) -> String {
    match error {
        SpawnError::Fatal(_) => MODEL_FAILED.to_string(),
        SpawnError::Timeout => MODEL_TIMEOUT.to_string(),
        SpawnError::Io(std::io::ErrorKind::NotFound) => HELPER_MISSING.to_string(),
        _ => HELPER_START_FAILED.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Write};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc, Arc};

    fn channel_process(stdin: Box<dyn Write + Send>) -> (HelperProcess, mpsc::Sender<Response>) {
        let (sender, receiver) = mpsc::channel();
        (HelperProcess::from_parts(stdin, receiver, None), sender)
    }

    fn transcript(id: u64, text: &str) -> Response {
        Response::Transcript {
            id,
            text: text.to_string(),
            duration_ms: 0,
        }
    }

    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "pipe closed"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_matching_transcript_is_returned_and_other_ids_are_skipped() {
        let (process, sender) = channel_process(Box::new(Vec::new()));
        let mut first = Some(process);
        let spawner: Spawner =
            Box::new(move || first.take().ok_or_else(|| "should not respawn".to_string()));
        let mut engine = HelperStt::new(spawner).expect("engine should load");

        sender.send(transcript(99, "wrong")).expect("send");
        sender.send(transcript(1, "hello")).expect("send");

        assert_eq!(engine.transcribe(&[0.1, 0.2]), Ok("hello".to_string()));
    }

    #[test]
    fn an_error_response_is_reported_without_restarting() {
        let (process, sender) = channel_process(Box::new(Vec::new()));
        let opens = Arc::new(AtomicUsize::new(0));
        let opens_for_spawner = Arc::clone(&opens);
        let mut first = Some(process);
        let spawner: Spawner = Box::new(move || {
            opens_for_spawner.fetch_add(1, Ordering::SeqCst);
            first.take().ok_or_else(|| "should not respawn".to_string())
        });
        let mut engine = HelperStt::new(spawner).expect("engine should load");
        assert_eq!(opens.load(Ordering::SeqCst), 1);

        sender
            .send(Response::Error {
                id: 1,
                message: "bad request".to_string(),
            })
            .expect("send");

        assert_eq!(
            engine.transcribe(&[0.0; 8]),
            Err(UTTERANCE_FAILED.to_string())
        );
        assert_eq!(opens.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_failing_writer_causes_one_respawn_then_succeeds() {
        let (failing, _failing_sender) = channel_process(Box::new(FailingWriter));
        let (working, sender) = channel_process(Box::new(Vec::new()));
        let opens = Arc::new(AtomicUsize::new(0));
        let opens_for_spawner = Arc::clone(&opens);
        let mut first = Some(failing);
        let mut second = Some(working);
        let spawner: Spawner = Box::new(move || {
            opens_for_spawner.fetch_add(1, Ordering::SeqCst);
            if let Some(process) = first.take() {
                return Ok(process);
            }
            second
                .take()
                .ok_or_else(|| "should not respawn more than once".to_string())
        });
        let mut engine = HelperStt::new(spawner).expect("engine should load");

        // The first attempt fails writing and restarts; the retry uses id 2.
        sender.send(transcript(2, "second try")).expect("send");

        assert_eq!(engine.transcribe(&[0.0; 4]), Ok("second try".to_string()));
        assert_eq!(opens.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_second_failure_marks_the_engine_dead_for_good() {
        let (first_process, _first_sender) = channel_process(Box::new(FailingWriter));
        let (second_process, _second_sender) = channel_process(Box::new(FailingWriter));
        let opens = Arc::new(AtomicUsize::new(0));
        let opens_for_spawner = Arc::clone(&opens);
        let mut first = Some(first_process);
        let mut second = Some(second_process);
        let spawner: Spawner = Box::new(move || {
            opens_for_spawner.fetch_add(1, Ordering::SeqCst);
            if let Some(process) = first.take() {
                return Ok(process);
            }
            second
                .take()
                .ok_or_else(|| "should not respawn more than once".to_string())
        });
        let mut engine = HelperStt::new(spawner).expect("engine should load");

        assert_eq!(
            engine.transcribe(&[0.0; 4]),
            Err(HELPER_STOPPED.to_string())
        );
        assert_eq!(opens.load(Ordering::SeqCst), 2);

        // Already dead: fails at once without spawning again.
        assert_eq!(
            engine.transcribe(&[0.0; 4]),
            Err(HELPER_STOPPED.to_string())
        );
        assert_eq!(opens.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_missing_helper_makes_the_factory_fail_with_its_message() {
        let factory = stt_factory(Err(HELPER_MISSING.to_string()), PathBuf::from("log"));
        let config = SttConfig {
            model_path: PathBuf::from("model.bin"),
            language: "en".to_string(),
        };

        assert_eq!(factory(&config).err(), Some(HELPER_MISSING.to_string()));
    }
}
