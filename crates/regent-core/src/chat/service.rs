//! The chat service: one worker thread owns the engine and the history store,
//! and the rest of the app talks to it through blocking methods.
//!
//! The worker never holds a lock while generating: cancellation lives in a
//! shared atomic flag, so [`ChatService::cancel`] works while a reply is
//! streaming. Every state change is reported through the status sink.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use serde::Serialize;

use super::engine::{ChatEngine, ChatEngineError, EngineConfig, GenerationStats, Outcome};
use super::message::{ChatMessage, Role};
use super::prompt::build_prompt;
use super::store::{ChatStore, HISTORY_READ_CAP};

/// The message for a start request that names no model.
const NO_MODEL: &str = "No text model chosen. Open Settings and pick one.";

/// The message for a model file that has gone missing.
const MISSING_MODEL: &str = "The text model file couldn't be found. Choose it again in Settings.";

/// The message for a start request while one is already running.
const ALREADY_RUNNING: &str = "Regent is already running.";

/// The message for a send request before the engine has started.
const NOT_STARTED: &str = "Start Regent first.";

/// The message used when the worker thread has gone away.
const WORKER_GONE: &str = "Regent's chat service stopped unexpectedly. Restart Regent.";

/// The fallback message for a store that could not be opened.
const STORE_GONE: &str = "Regent couldn't open its chat history file.";

/// Where the chat service is in its life cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatState {
    /// No engine is loaded.
    Stopped,
    /// The engine is being loaded.
    Loading,
    /// The engine is loaded and ready for a turn.
    Ready,
    /// A reply is being generated.
    Generating,
    /// Loading or a turn failed; the message is in [`ChatStatus::error`].
    Error,
}

/// The chat service's current state, as the frontend sees it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChatStatus {
    /// The current state.
    pub state: ChatState,
    /// The plain-language error, when the state is [`ChatState::Error`].
    pub error: Option<String>,
}

/// One event of a chat turn, streamed to the caller of
/// [`ChatService::send`].
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
    /// A piece of the reply, as it is produced.
    Token {
        /// The piece of text.
        text: String,
    },
    /// The reply finished and was saved.
    Done {
        /// The saved assistant message.
        message_id: i64,
        /// What the generation cost.
        stats: GenerationStats,
    },
    /// The reply was cancelled; any partial text was saved.
    Cancelled {
        /// The saved partial message, or `None` when nothing was produced.
        message_id: Option<i64>,
    },
    /// The turn failed before a reply could be saved.
    Error {
        /// The plain-language error.
        message: String,
    },
}

/// Builds an engine from a configuration; the Tauri layer supplies the real
/// one, tests supply a fake.
pub type EngineFactory =
    Box<dyn Fn(&EngineConfig) -> Result<Box<dyn ChatEngine>, ChatEngineError> + Send + Sync>;

/// Receives every chat state change.
pub type StatusSink = Box<dyn Fn(ChatStatus) + Send + Sync>;

/// One request for the worker thread.
enum Command {
    /// Load the engine with the given configuration.
    Start {
        config: EngineConfig,
        reply: Sender<Result<(), String>>,
    },
    /// Drop the engine and report the service as stopped.
    Stop { reply: Sender<()> },
    /// Run one turn.
    Send {
        text: String,
        system_addition: String,
        events: Sender<TurnMessage>,
        acks: Receiver<()>,
    },
    /// Read stored history.
    History {
        limit: usize,
        reply: Sender<Result<Vec<ChatMessage>, String>>,
    },
    /// Leave the worker loop.
    Shutdown,
}

/// One message from the worker during a turn.
enum TurnMessage {
    /// An event to hand to the caller's `on_event`.
    Event(ChatEvent),
    /// The turn as seen by [`ChatService::send`].
    Finished(Result<(), String>),
}

/// The handle the app uses to talk to the chat worker.
pub struct ChatService {
    commands: Sender<Command>,
    status: Arc<Mutex<ChatStatus>>,
    cancel_flag: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl ChatService {
    /// Starts the worker thread.
    ///
    /// The worker owns the engine and opens the history store at `db_path` the
    /// first time it needs it. Every state change is passed to `status_sink`.
    pub fn spawn(db_path: PathBuf, factory: EngineFactory, status_sink: StatusSink) -> ChatService {
        let status = Arc::new(Mutex::new(ChatStatus {
            state: ChatState::Stopped,
            error: None,
        }));
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let (commands, incoming) = mpsc::channel();

        let worker = {
            let status = Arc::clone(&status);
            let cancel_flag = Arc::clone(&cancel_flag);
            std::thread::spawn(move || {
                Worker {
                    db_path,
                    factory,
                    status,
                    sink: status_sink,
                    cancel: cancel_flag,
                    engine: None,
                    store: None,
                }
                .run(incoming);
            })
        };

        ChatService {
            commands,
            status,
            cancel_flag,
            worker: Some(worker),
        }
    }

    /// Loads the text model.
    ///
    /// `None` means no model has been chosen; the error messages name exactly
    /// what the person should do next.
    pub fn start(&self, config: Option<EngineConfig>) -> Result<(), String> {
        let Some(config) = config else {
            return Err(NO_MODEL.to_string());
        };
        if !config.model_path.is_file() {
            return Err(MISSING_MODEL.to_string());
        }

        let (reply, answer) = mpsc::channel();
        self.commands
            .send(Command::Start { config, reply })
            .map_err(|_| WORKER_GONE.to_string())?;
        answer.recv().map_err(|_| WORKER_GONE.to_string())?
    }

    /// Drops the engine and reports the service as stopped.
    ///
    /// The cancel flag is set first, so a reply that is streaming right now
    /// stops before the engine goes away, and the call returns once the worker
    /// has finished the turn. From inside [`ChatService::send`]'s callback use
    /// [`ChatService::cancel`] instead: stopping from there would wait on the
    /// turn that is calling it.
    pub fn stop(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
        let (reply, done) = mpsc::channel();
        if self.commands.send(Command::Stop { reply }).is_ok() {
            let _ = done.recv();
        }
    }

    /// Runs one chat turn, handing every streamed event to `on_event`.
    ///
    /// The whole prompt is rebuilt and generated on each turn; no cache is
    /// reused. The call blocks until the reply finishes, is cancelled or
    /// fails.
    pub fn send(
        &self,
        text: &str,
        system_addition: &str,
        on_event: &mut dyn FnMut(ChatEvent),
    ) -> Result<(), String> {
        let (events, incoming) = mpsc::channel();
        // The worker waits for this acknowledgement after every token, so the
        // engine stops as soon as `on_event` cancels the turn.
        let (acknowledgements, acks) = mpsc::channel();
        self.commands
            .send(Command::Send {
                text: text.to_string(),
                system_addition: system_addition.to_string(),
                events,
                acks,
            })
            .map_err(|_| WORKER_GONE.to_string())?;

        loop {
            match incoming.recv() {
                Ok(TurnMessage::Event(event)) => {
                    let token = matches!(event, ChatEvent::Token { .. });
                    on_event(event);
                    if token {
                        let _ = acknowledgements.send(());
                    }
                }
                Ok(TurnMessage::Finished(result)) => return result,
                Err(_) => return Err(WORKER_GONE.to_string()),
            }
        }
    }

    /// Stops the reply that is currently streaming, if any.
    ///
    /// This only touches an atomic flag, so it is reachable while
    /// [`ChatService::send`] is running on another thread.
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
    }

    /// The service's current state.
    pub fn status(&self) -> ChatStatus {
        self.status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// The newest `limit` stored messages, oldest first.
    ///
    /// Works while the engine is stopped; a store problem comes back as its
    /// plain-language message.
    pub fn history(&self, limit: usize) -> Result<Vec<ChatMessage>, String> {
        let (reply, answer) = mpsc::channel();
        self.commands
            .send(Command::History { limit, reply })
            .map_err(|_| WORKER_GONE.to_string())?;
        answer.recv().map_err(|_| WORKER_GONE.to_string())?
    }
}

impl Drop for ChatService {
    fn drop(&mut self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
        let _ = self.commands.send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// The worker thread's private state.
struct Worker {
    db_path: PathBuf,
    factory: EngineFactory,
    status: Arc<Mutex<ChatStatus>>,
    sink: StatusSink,
    cancel: Arc<AtomicBool>,
    engine: Option<Box<dyn ChatEngine>>,
    store: Option<ChatStore>,
}

impl Worker {
    /// The worker loop: one command at a time, in arrival order.
    fn run(mut self, commands: Receiver<Command>) {
        for command in commands {
            match command {
                Command::Start { config, reply } => {
                    let _ = reply.send(self.start(&config));
                }
                Command::Stop { reply } => {
                    self.engine = None;
                    self.set_status(ChatState::Stopped, None);
                    let _ = reply.send(());
                }
                Command::Send {
                    text,
                    system_addition,
                    events,
                    acks,
                } => {
                    let result = self.send(&text, &system_addition, &events, &acks);
                    let _ = events.send(TurnMessage::Finished(result));
                }
                Command::History { limit, reply } => {
                    let _ = reply.send(self.history(limit));
                }
                Command::Shutdown => {
                    self.engine = None;
                    self.set_status(ChatState::Stopped, None);
                    return;
                }
            }
        }
    }

    /// Publishes a state change to the status snapshot and the sink.
    fn set_status(&self, state: ChatState, error: Option<String>) {
        let next = ChatStatus { state, error };
        {
            let mut current = self
                .status
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *current = next.clone();
        }
        (self.sink)(next);
    }

    /// The current state, read from the shared snapshot.
    fn state(&self) -> ChatState {
        self.status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .state
    }

    /// The history store, opened on first use.
    fn store_mut(&mut self) -> Result<&mut ChatStore, String> {
        if self.store.is_none() {
            let opened =
                ChatStore::open(self.db_path.as_path()).map_err(|error| error.to_string())?;
            self.store = Some(opened);
        }
        self.store.as_mut().ok_or_else(|| STORE_GONE.to_string())
    }

    /// Loads the engine for `config`.
    fn start(&mut self, config: &EngineConfig) -> Result<(), String> {
        if self.engine.is_some() {
            return Err(ALREADY_RUNNING.to_string());
        }

        self.set_status(ChatState::Loading, None);
        match (self.factory)(config) {
            Ok(engine) => {
                self.engine = Some(engine);
                self.set_status(ChatState::Ready, None);
                Ok(())
            }
            Err(error) => {
                let message = error.0;
                self.set_status(ChatState::Error, Some(message.clone()));
                Err(message)
            }
        }
    }

    /// Runs one turn and streams its events to `events`.
    fn send(
        &mut self,
        text: &str,
        system_addition: &str,
        events: &Sender<TurnMessage>,
        acks: &Receiver<()>,
    ) -> Result<(), String> {
        if self.state() != ChatState::Ready {
            return Err(NOT_STARTED.to_string());
        }
        if text.trim().is_empty() {
            return Ok(());
        }

        // Each turn starts with a fresh cancel flag.
        self.cancel.store(false, Ordering::SeqCst);

        {
            let store = self.store_mut()?;
            store
                .append(Role::User, text, false)
                .map_err(|error| error.to_string())?;
        }
        let history = self
            .store_mut()?
            .recent(HISTORY_READ_CAP)
            .map_err(|error| error.to_string())?;

        let engine = match self.engine.as_ref() {
            Some(engine) => engine,
            None => return Err(NOT_STARTED.to_string()),
        };
        let prompt = build_prompt(
            system_addition,
            None,
            &history,
            engine.context_tokens(),
            &|piece| engine.count_tokens(piece),
        );

        self.set_status(ChatState::Generating, None);

        let cancel = Arc::clone(&self.cancel);
        let mut reply = String::new();
        let outcome = match self.engine.as_mut() {
            Some(engine) => engine.generate(&prompt, &cancel, &mut |piece| {
                reply.push_str(piece);
                let _ = events.send(TurnMessage::Event(ChatEvent::Token {
                    text: piece.to_string(),
                }));
                // Wait for the caller to hand the piece to `on_event`, which
                // may cancel the turn, before producing the next one.
                if acks.recv().is_err() {
                    cancel.store(true, Ordering::SeqCst);
                }
            }),
            None => return Err(NOT_STARTED.to_string()),
        };

        let turn = match outcome {
            Ok(Outcome::Finished(stats)) => {
                let saved = self.store_mut().and_then(|store| {
                    store
                        .append(Role::Assistant, &reply, false)
                        .map_err(|error| error.to_string())
                });
                match saved {
                    Ok(message) => {
                        let _ = events.send(TurnMessage::Event(ChatEvent::Done {
                            message_id: message.id,
                            stats,
                        }));
                        Ok(())
                    }
                    Err(error) => {
                        let _ = events.send(TurnMessage::Event(ChatEvent::Error {
                            message: error.clone(),
                        }));
                        Err(error)
                    }
                }
            }
            Ok(Outcome::Cancelled(_)) => {
                let message_id = if reply.is_empty() {
                    None
                } else {
                    self.store_mut()
                        .and_then(|store| {
                            store
                                .append(Role::Assistant, &reply, false)
                                .map_err(|error| error.to_string())
                        })
                        .map(|message| message.id)
                        .ok()
                };
                let _ = events.send(TurnMessage::Event(ChatEvent::Cancelled { message_id }));
                Ok(())
            }
            Err(error) => {
                let _ = events.send(TurnMessage::Event(ChatEvent::Error {
                    message: error.0.clone(),
                }));
                Ok(())
            }
        };

        // A finished, cancelled or failed turn always leaves the service ready
        // for the next one.
        self.set_status(ChatState::Ready, None);
        turn
    }

    /// Reads stored history, opening the store if needed.
    fn history(&mut self, limit: usize) -> Result<Vec<ChatMessage>, String> {
        self.store_mut()?
            .recent(limit)
            .map_err(|error| error.to_string())
    }
}

/// The history database path for a test directory.
#[cfg(test)]
fn db_path(dir: &tempfile::TempDir) -> PathBuf {
    dir.path().join("regent.db")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::engine::FakeEngine;
    use serde_json::json;

    /// A factory that returns a [`FakeEngine`] emitting `pieces`.
    fn piece_factory(pieces: &[&str]) -> EngineFactory {
        let pieces: Vec<String> = pieces.iter().map(|piece| piece.to_string()).collect();
        Box::new(move |_config| Ok(Box::new(FakeEngine::new(pieces.clone(), 4096))))
    }

    /// A factory whose engine always fails.
    fn failing_factory(message: &str) -> EngineFactory {
        let message = message.to_string();
        Box::new(move |_config| Ok(Box::new(FakeEngine::failing(message.clone()))))
    }

    /// A status sink that records every change.
    fn recording_sink() -> (StatusSink, Arc<Mutex<Vec<ChatStatus>>>) {
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let sink: StatusSink = {
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

    /// A config whose model file exists.
    fn config(dir: &tempfile::TempDir) -> EngineConfig {
        let model_path = dir.path().join("text.gguf");
        std::fs::write(&model_path, b"fake model").expect("write model fixture");
        EngineConfig {
            model_path,
            gpu_layers: 0,
            context_size: 4096,
        }
    }

    fn service(dir: &tempfile::TempDir, factory: EngineFactory) -> ChatService {
        ChatService::spawn(db_path(dir), factory, Box::new(|_| {}))
    }

    #[test]
    fn start_then_send_streams_tokens_and_saves_both_messages() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (sink, statuses) = recording_sink();
        let service = ChatService::spawn(db_path(&dir), piece_factory(&["Hello", " there"]), sink);

        service
            .start(Some(config(&dir)))
            .expect("start should succeed");
        assert_eq!(service.status().state, ChatState::Ready);

        let mut events = Vec::new();
        service
            .send("hi", "", &mut |event| events.push(event))
            .expect("send should succeed");

        let tokens: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                ChatEvent::Token { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(tokens, vec!["Hello", " there"]);

        let done = events
            .iter()
            .find_map(|event| match event {
                ChatEvent::Done { message_id, stats } => Some((*message_id, stats.clone())),
                _ => None,
            })
            .expect("a Done event");
        assert_eq!(done.1.generated_tokens, 2);

        let history = service.history(10).expect("history");
        assert_eq!(
            history
                .iter()
                .map(|message| message.text.as_str())
                .collect::<Vec<_>>(),
            vec!["hi", "Hello there"]
        );
        assert_eq!(history[0].role, Role::User);
        assert_eq!(history[1].role, Role::Assistant);
        assert_eq!(history[1].id, done.0);

        assert_eq!(service.status().state, ChatState::Ready);
        let seen: Vec<ChatState> = statuses
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|status| status.state)
            .collect();
        assert_eq!(
            seen,
            vec![
                ChatState::Loading,
                ChatState::Ready,
                ChatState::Generating,
                ChatState::Ready
            ]
        );
    }

    #[test]
    fn cancelling_mid_reply_saves_the_partial_text() {
        let dir = tempfile::tempdir().expect("temp dir");
        let service = service(&dir, piece_factory(&["Hello", " world"]));
        service.start(Some(config(&dir))).expect("start");

        let mut events = Vec::new();
        service
            .send("hi", "", &mut |event| {
                if matches!(event, ChatEvent::Token { .. }) {
                    service.cancel();
                }
                events.push(event);
            })
            .expect("send should succeed");

        let partial_id = events
            .iter()
            .find_map(|event| match event {
                ChatEvent::Cancelled { message_id } => Some(*message_id),
                _ => None,
            })
            .expect("a Cancelled event");
        assert!(partial_id.is_some());

        let history = service.history(10).expect("history");
        assert_eq!(
            history
                .iter()
                .map(|message| message.text.as_str())
                .collect::<Vec<_>>(),
            vec!["hi", "Hello"]
        );
        assert_eq!(service.status().state, ChatState::Ready);
    }

    #[test]
    fn sending_while_stopped_errors() {
        let dir = tempfile::tempdir().expect("temp dir");
        let service = service(&dir, piece_factory(&["never"]));

        let error = service
            .send("hi", "", &mut |_| {})
            .expect_err("send should fail");
        assert_eq!(error, "Start Regent first.");
        assert!(service.history(10).expect("history").is_empty());
    }

    #[test]
    fn starting_without_a_model_or_with_a_missing_file_gives_exact_messages() {
        let dir = tempfile::tempdir().expect("temp dir");
        let service = service(&dir, piece_factory(&["never"]));

        assert_eq!(
            service.start(None).expect_err("start should fail"),
            "No text model chosen. Open Settings and pick one."
        );

        let missing = EngineConfig {
            model_path: dir.path().join("nothing-here.gguf"),
            gpu_layers: 0,
            context_size: 4096,
        };
        assert_eq!(
            service.start(Some(missing)).expect_err("start should fail"),
            "The text model file couldn't be found. Choose it again in Settings."
        );
        assert_eq!(service.status().state, ChatState::Stopped);
    }

    #[test]
    fn starting_twice_errors() {
        let dir = tempfile::tempdir().expect("temp dir");
        let service = service(&dir, piece_factory(&["hi"]));
        let config = config(&dir);
        service.start(Some(config.clone())).expect("start");

        assert_eq!(
            service
                .start(Some(config))
                .expect_err("second start should fail"),
            "Regent is already running."
        );
    }

    #[test]
    fn history_survives_a_new_service_on_the_same_file() {
        let dir = tempfile::tempdir().expect("temp dir");
        {
            let service = service(&dir, piece_factory(&["remembered"]));
            service.start(Some(config(&dir))).expect("start");
            service.send("hi", "", &mut |_| {}).expect("send");
        }

        let reopened = service(&dir, piece_factory(&["again"]));
        let history = reopened.history(10).expect("history");
        assert_eq!(
            history
                .iter()
                .map(|message| message.text.as_str())
                .collect::<Vec<_>>(),
            vec!["hi", "remembered"]
        );
    }

    #[test]
    fn an_engine_failure_emits_error_and_returns_to_ready() {
        let dir = tempfile::tempdir().expect("temp dir");
        let service = service(&dir, failing_factory("The model stopped unexpectedly."));
        service.start(Some(config(&dir))).expect("start");

        let mut events = Vec::new();
        service
            .send("hi", "", &mut |event| events.push(event))
            .expect("send should succeed");

        let message = events
            .iter()
            .find_map(|event| match event {
                ChatEvent::Error { message } => Some(message.clone()),
                _ => None,
            })
            .expect("an Error event");
        assert_eq!(message, "The model stopped unexpectedly.");
        assert_eq!(service.status().state, ChatState::Ready);

        // Only the user's message was stored: no empty reply.
        let history = service.history(10).expect("history");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].text, "hi");
    }

    #[test]
    fn a_blank_message_is_ignored() {
        let dir = tempfile::tempdir().expect("temp dir");
        let service = service(&dir, piece_factory(&["hi"]));
        service.start(Some(config(&dir))).expect("start");

        let mut count = 0;
        service
            .send("   ", "", &mut |_| count += 1)
            .expect("send should succeed");
        assert_eq!(count, 0);
        assert!(service.history(10).expect("history").is_empty());
    }

    #[test]
    fn stopped_after_stop_and_can_start_again() {
        let dir = tempfile::tempdir().expect("temp dir");
        let service = service(&dir, piece_factory(&["hi"]));
        service.start(Some(config(&dir))).expect("start");
        service.stop();
        assert_eq!(service.status().state, ChatState::Stopped);

        service.start(Some(config(&dir))).expect("restart");
        assert_eq!(service.status().state, ChatState::Ready);
    }

    #[test]
    fn chat_event_json_shapes_are_stable() {
        let token = ChatEvent::Token {
            text: "hi".to_string(),
        };
        assert_eq!(
            serde_json::to_value(&token).expect("serialise"),
            json!({"type": "token", "text": "hi"})
        );

        let done = ChatEvent::Done {
            message_id: 7,
            stats: GenerationStats {
                prompt_tokens: 3,
                generated_tokens: 2,
                tokens_per_second: 1.5,
            },
        };
        assert_eq!(
            serde_json::to_value(&done).expect("serialise"),
            json!({
                "type": "done",
                "message_id": 7,
                "stats": {
                    "prompt_tokens": 3,
                    "generated_tokens": 2,
                    "tokens_per_second": 1.5
                }
            })
        );

        let cancelled = ChatEvent::Cancelled { message_id: None };
        assert_eq!(
            serde_json::to_value(&cancelled).expect("serialise"),
            json!({"type": "cancelled", "message_id": null})
        );

        let error = ChatEvent::Error {
            message: "Nope.".to_string(),
        };
        assert_eq!(
            serde_json::to_value(&error).expect("serialise"),
            json!({"type": "error", "message": "Nope."})
        );
    }

    #[test]
    fn chat_status_json_shapes_are_stable() {
        let ready = ChatStatus {
            state: ChatState::Ready,
            error: None,
        };
        assert_eq!(
            serde_json::to_value(&ready).expect("serialise"),
            json!({"state": "ready", "error": null})
        );

        let error = ChatStatus {
            state: ChatState::Error,
            error: Some("The model didn't load.".to_string()),
        };
        assert_eq!(
            serde_json::to_value(&error).expect("serialise"),
            json!({"state": "error", "error": "The model didn't load."})
        );

        for (state, expected) in [
            (ChatState::Stopped, "stopped"),
            (ChatState::Loading, "loading"),
            (ChatState::Ready, "ready"),
            (ChatState::Generating, "generating"),
            (ChatState::Error, "error"),
        ] {
            assert_eq!(
                serde_json::to_value(state).expect("serialise"),
                json!(expected)
            );
        }
    }

    #[test]
    fn a_failed_load_sets_the_error_state_and_can_be_retried() {
        let dir = tempfile::tempdir().expect("temp dir");
        let attempts = Arc::new(Mutex::new(0));
        let factory: EngineFactory = {
            let attempts = Arc::clone(&attempts);
            Box::new(move |_config| {
                let mut attempts = attempts
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                *attempts += 1;
                if *attempts == 1 {
                    Err(ChatEngineError("The model didn't load.".to_string()))
                } else {
                    Ok(Box::new(FakeEngine::new(vec!["hi".to_string()], 4096)))
                }
            })
        };
        let (sink, statuses) = recording_sink();
        let service = ChatService::spawn(db_path(&dir), factory, sink);

        let config = config(&dir);
        assert_eq!(
            service
                .start(Some(config.clone()))
                .expect_err("first start should fail"),
            "The model didn't load."
        );
        let status = service.status();
        assert_eq!(status.state, ChatState::Error);
        assert_eq!(status.error.as_deref(), Some("The model didn't load."));

        service.start(Some(config)).expect("retry should work");
        assert_eq!(service.status().state, ChatState::Ready);

        let seen: Vec<ChatState> = statuses
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|status| status.state)
            .collect();
        assert_eq!(
            seen,
            vec![
                ChatState::Loading,
                ChatState::Error,
                ChatState::Loading,
                ChatState::Ready
            ]
        );
    }
}
