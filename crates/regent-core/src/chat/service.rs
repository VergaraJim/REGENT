//! The chat service: one worker thread owns the engine and the history store,
//! and the rest of the app talks to it through blocking methods.
//!
//! The worker never holds a lock while generating: cancellation lives in a
//! shared atomic flag, so [`ChatService::cancel`] works while a reply is
//! streaming. Every state change is reported through the status sink.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use serde::Serialize;

use super::compaction::{
    accept_summary, build_summary_prompt, plan_compaction, summarizer_budget,
    working_memory_tokens, CompactionPlan, MAX_COMPACTION_PASSES,
};
use super::engine::{ChatEngine, ChatEngineError, EngineConfig, GenerationStats, Outcome};
use super::message::{ChatMessage, Role};
use super::prefix_cache::{PrefixCache, PrefixKey};
use super::prompt::{build_prompt, system_text, PromptMessage, PromptRole};
use super::store::{ChatStore, HISTORY_READ_CAP};
use crate::context::compaction_thresholds;
use crate::memory::engine::{EmbeddingConfig, EmbeddingEngine, EmbeddingFactory};
use crate::memory::status::{MemorySink, MemoryState, MemoryStatus};
use crate::memory::vector::{normalize, VectorIndex};
use crate::memory::{
    chunk_text, model_id, resolve_markers, Markers, INDEX_BATCH, MAX_INDEX_FAILURES,
};
use crate::settings::CompactionSettings;

/// The message for a start request that names no model.
const NO_MODEL: &str = "No text model chosen. Open Settings and pick one.";

/// The message for a model file that has gone missing.
const MISSING_MODEL: &str = "The text model file couldn't be found. Choose it again in Settings.";

/// The message for a memory model file that has gone missing.
const MISSING_MEMORY_MODEL: &str =
    "The memory model file couldn't be found. Choose it again in Settings.";

/// The message for a memory model file that could not be read.
const UNREADABLE_MEMORY_MODEL: &str = "Regent couldn't read the memory model file.";

/// The message memory indexing pauses with after repeated failures.
const INDEX_PAUSED: &str =
    "Memory indexing paused after repeated errors. Restart Regent to try again.";

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
    /// Older turns are being folded into a summary.
    Compacting,
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
    /// Older turns are being folded into a summary before the reply.
    Compacting,
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

/// The per-turn settings the worker builds a turn with.
#[derive(Debug, Clone, Default)]
pub struct TurnSettings {
    /// Extra text appended to the built-in system prompt.
    pub system_addition: String,
    /// When and how far history is compacted.
    pub compaction: CompactionSettings,
    /// Whether the fixed prompt prefix is saved and reused between runs.
    pub persist_prefix_cache: bool,
}

/// Builds an engine from a configuration; the Tauri layer supplies the real
/// one, tests supply a fake.
pub type EngineFactory =
    Box<dyn Fn(&EngineConfig) -> Result<Box<dyn ChatEngine>, ChatEngineError> + Send + Sync>;

/// Receives every chat state change.
pub type StatusSink = Box<dyn Fn(ChatStatus) + Send + Sync>;

/// Everything [`ChatService::spawn`] needs to start its worker.
pub struct ChatServiceDeps {
    /// Where the chat history lives.
    pub db_path: PathBuf,
    /// Where persisted prefix states live.
    pub prefix_cache_dir: PathBuf,
    /// Builds the text engine.
    pub engine_factory: EngineFactory,
    /// Builds the optional embedding engine.
    pub embedding_factory: EmbeddingFactory,
    /// Receives every chat state change.
    pub status_sink: StatusSink,
    /// Receives every memory status change.
    pub memory_sink: MemorySink,
}

/// One request for the worker thread.
enum Command {
    /// Load the engine with the given configuration.
    Start {
        config: EngineConfig,
        embedding: Option<EmbeddingConfig>,
        reply: Sender<Result<(), String>>,
    },
    /// Drop the engine and report the service as stopped.
    Stop { reply: Sender<()> },
    /// Run one turn.
    Send {
        text: String,
        settings: TurnSettings,
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
    memory: Arc<Mutex<MemoryStatus>>,
    cancel_flag: Arc<AtomicBool>,
    idle_cancel: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl ChatService {
    /// Starts the worker thread.
    ///
    /// The worker owns the engine and opens the history store at `db_path` the
    /// first time it needs it. Persisted prefix states are kept under
    /// `prefix_cache_dir`. Every state change is passed to `status_sink`.
    pub fn spawn(deps: ChatServiceDeps) -> ChatService {
        let status = Arc::new(Mutex::new(ChatStatus {
            state: ChatState::Stopped,
            error: None,
        }));
        let memory = Arc::new(Mutex::new(MemoryStatus {
            state: MemoryState::Off,
            indexed: 0,
            total: 0,
            error: None,
        }));
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let idle_cancel = Arc::new(AtomicBool::new(false));
        let (commands, incoming) = mpsc::channel();

        let worker = {
            let status = Arc::clone(&status);
            let memory = Arc::clone(&memory);
            let cancel_flag = Arc::clone(&cancel_flag);
            let idle_cancel = Arc::clone(&idle_cancel);
            let ChatServiceDeps {
                db_path,
                prefix_cache_dir,
                engine_factory,
                embedding_factory,
                status_sink,
                memory_sink,
            } = deps;
            std::thread::spawn(move || {
                Worker {
                    db_path,
                    factory: engine_factory,
                    embedding_factory,
                    status,
                    memory,
                    sink: status_sink,
                    memory_sink,
                    cancel: cancel_flag,
                    idle_cancel,
                    engine: None,
                    memory_runtime: None,
                    store: None,
                    last_settings: None,
                    pending: VecDeque::new(),
                    prefix_cache: PrefixCache::new(prefix_cache_dir),
                    started_config: None,
                    prefix_load_tried: false,
                    prefix_saved: false,
                }
                .run(incoming);
            })
        };

        ChatService {
            commands,
            status,
            memory,
            cancel_flag,
            idle_cancel,
            worker: Some(worker),
        }
    }

    /// Loads the text model.
    ///
    /// `None` means no model has been chosen; the error messages name exactly
    /// what the person should do next.
    ///
    /// `embedding` selects the optional memory model. Memory setup never fails
    /// the start: a bad or missing memory model only turns memory off.
    pub fn start(
        &self,
        config: Option<EngineConfig>,
        embedding: Option<EmbeddingConfig>,
    ) -> Result<(), String> {
        let Some(config) = config else {
            return Err(NO_MODEL.to_string());
        };
        if !config.model_path.is_file() {
            return Err(MISSING_MODEL.to_string());
        }

        let (reply, answer) = mpsc::channel();
        // Cancel any idle work before the command can be seen, so the worker
        // never treats a start request as an idle-window command.
        self.idle_cancel.store(true, Ordering::SeqCst);
        self.commands
            .send(Command::Start {
                config,
                embedding,
                reply,
            })
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
        // Cancel idle work before the command can be seen, so the stop is not
        // delayed by an indexing batch.
        self.idle_cancel.store(true, Ordering::SeqCst);
        if self.commands.send(Command::Stop { reply }).is_ok() {
            let _ = done.recv();
        }
    }

    /// Runs one chat turn, handing every streamed event to `on_event`.
    ///
    /// The prompt is rebuilt from the stored history on every turn, including
    /// the rolling summary when one exists. When the prompt would grow past the
    /// window's hard limit, a compaction pass runs first and is announced with
    /// [`ChatEvent::Compacting`]. The call blocks until the reply finishes, is
    /// cancelled or fails.
    pub fn send(
        &self,
        text: &str,
        settings: &TurnSettings,
        on_event: &mut dyn FnMut(ChatEvent),
    ) -> Result<(), String> {
        let (events, incoming) = mpsc::channel();
        // The worker waits for this acknowledgement after every token, so the
        // engine stops as soon as `on_event` cancels the turn.
        let (acknowledgements, acks) = mpsc::channel();
        // Cancel idle work before the command can be seen: a new turn outranks
        // any compaction or indexing the worker may be running.
        self.idle_cancel.store(true, Ordering::SeqCst);
        self.commands
            .send(Command::Send {
                text: text.to_string(),
                settings: settings.clone(),
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
        // Cancel idle work before the command can be seen.
        self.idle_cancel.store(true, Ordering::SeqCst);
        self.commands
            .send(Command::History { limit, reply })
            .map_err(|_| WORKER_GONE.to_string())?;
        answer.recv().map_err(|_| WORKER_GONE.to_string())?
    }

    /// The memory indexer's current state.
    pub fn memory_status(&self) -> MemoryStatus {
        self.memory
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

impl Drop for ChatService {
    fn drop(&mut self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
        self.idle_cancel.store(true, Ordering::SeqCst);
        let _ = self.commands.send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// The embedding model and the index the worker keeps in memory.
struct MemoryRuntime {
    engine: Box<dyn EmbeddingEngine>,
    markers: Markers,
    model_id: String,
    index: VectorIndex,
    failures: u32,
}

/// The worker thread's private state.
struct Worker {
    db_path: PathBuf,
    factory: EngineFactory,
    embedding_factory: EmbeddingFactory,
    status: Arc<Mutex<ChatStatus>>,
    memory: Arc<Mutex<MemoryStatus>>,
    sink: StatusSink,
    memory_sink: MemorySink,
    cancel: Arc<AtomicBool>,
    idle_cancel: Arc<AtomicBool>,
    engine: Option<Box<dyn ChatEngine>>,
    /// The memory model and index, when memory is on.
    memory_runtime: Option<MemoryRuntime>,
    store: Option<ChatStore>,
    /// The settings of the most recent turn, reused by the idle pass.
    last_settings: Option<TurnSettings>,
    /// Commands picked up while an idle pass was starting.
    pending: VecDeque<Command>,
    /// Where persisted prefix states live.
    prefix_cache: PrefixCache,
    /// The configuration the engine was started with, kept for the prefix key.
    started_config: Option<EngineConfig>,
    /// Whether this session has tried to load the prefix cache.
    prefix_load_tried: bool,
    /// Whether this session already has the prefix state in the cache.
    prefix_saved: bool,
}

impl Worker {
    /// The worker loop: queued commands first, then the channel, one at a time.
    fn run(mut self, commands: Receiver<Command>) {
        loop {
            let command = match self.pending.pop_front() {
                Some(command) => command,
                None => match commands.recv() {
                    Ok(command) => command,
                    Err(_) => return,
                },
            };

            // Every handle method sets `idle_cancel` before it sends a command,
            // so the worker clears it only once the command is in hand.
            self.idle_cancel.store(false, Ordering::SeqCst);

            match command {
                Command::Start {
                    config,
                    embedding,
                    reply,
                } => {
                    let result = self.start(&config, embedding);
                    let started = result.is_ok();
                    let _ = reply.send(result);
                    if started {
                        self.idle_work(&commands);
                    }
                }
                Command::Stop { reply } => {
                    self.engine = None;
                    self.memory_runtime = None;
                    self.set_status(ChatState::Stopped, None);
                    self.set_memory_status(MemoryState::Off, 0, 0, None);
                    let _ = reply.send(());
                }
                Command::Send {
                    text,
                    settings,
                    events,
                    acks,
                } => {
                    let result = self.send(&text, &settings, &events, &acks);
                    let _ = events.send(TurnMessage::Finished(result));
                    self.idle_work(&commands);
                }
                Command::History { limit, reply } => {
                    let _ = reply.send(self.history(limit));
                }
                Command::Shutdown => {
                    self.engine = None;
                    self.memory_runtime = None;
                    self.set_status(ChatState::Stopped, None);
                    self.set_memory_status(MemoryState::Off, 0, 0, None);
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

    /// Publishes a memory change to the snapshot and the sink.
    fn set_memory_status(
        &self,
        state: MemoryState,
        indexed: usize,
        total: usize,
        error: Option<String>,
    ) {
        let next = MemoryStatus {
            state,
            indexed,
            total,
            error,
        };
        {
            let mut current = self
                .memory
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *current = next.clone();
        }
        (self.memory_sink)(next);
    }

    /// The memory state, read from the shared snapshot.
    fn memory_state(&self) -> MemoryState {
        self.memory
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .state
    }

    /// The current memory counts, read from the store or the snapshot.
    fn memory_counts(&mut self) -> (usize, usize) {
        let Some(model_id) = self
            .memory_runtime
            .as_ref()
            .map(|runtime| runtime.model_id.clone())
        else {
            return (0, 0);
        };
        let counts = self.store_mut().and_then(|store| {
            store
                .index_counts(&model_id)
                .map_err(|error| error.to_string())
        });
        match counts {
            Ok(counts) => counts,
            Err(_) => {
                let status = self
                    .memory
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                (status.indexed, status.total)
            }
        }
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

    /// Loads the engine for `config` and then sets up memory.
    fn start(
        &mut self,
        config: &EngineConfig,
        embedding: Option<EmbeddingConfig>,
    ) -> Result<(), String> {
        if self.engine.is_some() {
            return Err(ALREADY_RUNNING.to_string());
        }

        self.set_status(ChatState::Loading, None);
        match (self.factory)(config) {
            Ok(engine) => {
                self.engine = Some(engine);
                self.started_config = Some(config.clone());
                self.prefix_load_tried = false;
                self.prefix_saved = false;
                self.set_status(ChatState::Ready, None);
                self.setup_memory(embedding);
                Ok(())
            }
            Err(error) => {
                let message = error.0;
                self.set_status(ChatState::Error, Some(message.clone()));
                Err(message)
            }
        }
    }

    /// Loads the embedding model and its index. This never fails the start:
    /// anything wrong only turns memory off, with a readable error.
    fn setup_memory(&mut self, config: Option<EmbeddingConfig>) {
        self.memory_runtime = None;

        let Some(config) = config else {
            self.set_memory_status(MemoryState::Off, 0, 0, None);
            return;
        };
        if !config.model_path.is_file() {
            self.set_memory_status(
                MemoryState::Off,
                0,
                0,
                Some(MISSING_MEMORY_MODEL.to_string()),
            );
            return;
        }
        let engine = match (self.embedding_factory)(&config) {
            Ok(engine) => engine,
            Err(error) => {
                self.set_memory_status(MemoryState::Off, 0, 0, Some(error.0));
                return;
            }
        };

        let file_name = config
            .model_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let markers = resolve_markers(
            &file_name,
            config.document_prefix.as_deref(),
            config.query_prefix.as_deref(),
        );
        let model_id = match model_id(&config.model_path, &markers) {
            Ok(model_id) => model_id,
            Err(_) => {
                self.set_memory_status(
                    MemoryState::Off,
                    0,
                    0,
                    Some(UNREADABLE_MEMORY_MODEL.to_string()),
                );
                return;
            }
        };

        let vectors = self
            .store_mut()
            .and_then(|store| {
                store
                    .load_vectors(&model_id)
                    .map_err(|error| error.to_string())
            })
            .unwrap_or_default();
        let mut index = VectorIndex::new();
        for (message_id, vector) in vectors {
            index.insert(message_id, vector);
        }

        self.memory_runtime = Some(MemoryRuntime {
            engine,
            markers,
            model_id,
            index,
            failures: 0,
        });
        let (indexed, total) = self.memory_counts();
        self.set_memory_status(MemoryState::Ready, indexed, total, None);
    }

    /// The latest summary's text and the messages that summary doesn't cover.
    fn working_memory(&mut self) -> Result<(Option<String>, Vec<ChatMessage>), String> {
        let store = self.store_mut()?;
        let latest = store.latest_summary().map_err(|error| error.to_string())?;
        let after = latest.as_ref().map_or(0, |summary| summary.covers_through);
        let messages = store
            .messages_after(after, HISTORY_READ_CAP)
            .map_err(|error| error.to_string())?;
        Ok((latest.map(|summary| summary.text), messages))
    }

    /// Plans the next compaction pass for `threshold`, or `None` when the
    /// working prompt is small enough or too little history exists.
    fn next_plan(&mut self, settings: &TurnSettings, threshold: usize) -> Option<CompactionPlan> {
        let system = system_text(&settings.system_addition);
        let (summary, messages) = self.working_memory().ok()?;
        let engine = self.engine.as_ref()?;
        let context_tokens = engine.context_tokens();
        let thresholds = compaction_thresholds(
            u32::try_from(context_tokens).unwrap_or(u32::MAX),
            &settings.compaction,
        );
        plan_compaction(
            &system,
            summary.as_deref(),
            &messages,
            threshold,
            thresholds.target as usize,
            summarizer_budget(context_tokens),
            &|piece| engine.count_tokens(piece),
        )
    }

    /// Folds old turns into summaries until the prompt is small enough.
    ///
    /// Returns whether at least one summary was stored. A failure, a rejected
    /// reply or a cancellation stops the loop quietly: the prompt builder's
    /// fallback trimming still keeps the turn under the window.
    fn compact(&mut self, settings: &TurnSettings, threshold: usize, cancel: &AtomicBool) -> bool {
        let mut stored = false;
        for _ in 0..MAX_COMPACTION_PASSES {
            let Some(plan) = self.next_plan(settings, threshold) else {
                break;
            };
            let prompt = build_summary_prompt(plan.previous_summary.as_deref(), &plan.fold);
            let summary = match self.engine.as_mut() {
                Some(engine) => engine.summarize(&prompt, plan.cap_tokens, cancel),
                None => break,
            };
            let Ok(Some(text)) = summary else {
                break;
            };
            let accepted = match self.engine.as_ref() {
                Some(engine) => {
                    accept_summary(&text, plan.cap_tokens, &|piece| engine.count_tokens(piece))
                }
                None => break,
            };
            let Some(text) = accepted else {
                break;
            };
            let appended = match self.store_mut() {
                Ok(store) => store.append_summary(&text, plan.covers_through),
                Err(_) => break,
            };
            match appended {
                Ok(_) => stored = true,
                Err(_) => break,
            }
            if cancel.load(Ordering::SeqCst) {
                break;
            }
        }
        stored
    }

    /// Runs the idle window after a reply or a successful start: queued
    /// commands are picked up first, then compaction and the prefix save, and
    /// finally memory indexing.
    ///
    /// Any command that arrives first is queued and cancels the pass, so a new
    /// turn never waits on idle work.
    fn idle_work(&mut self, commands: &Receiver<Command>) {
        match commands.try_recv() {
            Ok(command) => {
                self.pending.push_back(command);
                return;
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => return,
        }

        if let Some(settings) = self.last_settings.clone() {
            self.idle_compaction(&settings);
        }

        self.index_messages();
    }

    /// Folds old turns into summaries and saves the prefix, when there are
    /// turn settings to work from.
    fn idle_compaction(&mut self, settings: &TurnSettings) {
        let Some(engine) = self.engine.as_ref() else {
            return;
        };
        let context_tokens = engine.context_tokens();

        let trigger = compaction_thresholds(
            u32::try_from(context_tokens).unwrap_or(u32::MAX),
            &settings.compaction,
        )
        .trigger as usize;
        if self.next_plan(settings, trigger).is_some() {
            self.set_status(ChatState::Compacting, None);
            let cancel = Arc::clone(&self.idle_cancel);
            self.compact(settings, trigger, &cancel);
            self.set_status(ChatState::Ready, None);
        }

        self.save_prefix_cache(settings);
    }

    /// Indexes unindexed messages until the queue is empty or idle work is
    /// cancelled. Every status change goes to the memory sink.
    fn index_messages(&mut self) {
        loop {
            if self.idle_cancel.load(Ordering::SeqCst) {
                return;
            }
            let Some(runtime) = self.memory_runtime.as_ref() else {
                return;
            };
            if runtime.failures >= MAX_INDEX_FAILURES {
                let (indexed, total) = self.memory_counts();
                self.set_memory_status(
                    MemoryState::Paused,
                    indexed,
                    total,
                    Some(INDEX_PAUSED.to_string()),
                );
                return;
            }
            let model_id = runtime.model_id.clone();

            let batch = match self.store_mut() {
                Ok(store) => match store.unindexed_messages(&model_id, INDEX_BATCH) {
                    Ok(batch) => batch,
                    Err(_) => return,
                },
                Err(_) => return,
            };
            if batch.is_empty() {
                let (indexed, total) = self.memory_counts();
                if self.memory_state() != MemoryState::Ready {
                    self.set_memory_status(MemoryState::Ready, indexed, total, None);
                }
                return;
            }
            if self.memory_state() != MemoryState::Indexing {
                let (indexed, total) = self.memory_counts();
                self.set_memory_status(MemoryState::Indexing, indexed, total, None);
            }

            for message in batch {
                if self.idle_cancel.load(Ordering::SeqCst) {
                    return;
                }
                self.index_one(&model_id, &message);
            }

            let (indexed, total) = self.memory_counts();
            if self.memory_state() != MemoryState::Paused {
                self.set_memory_status(MemoryState::Indexing, indexed, total, None);
            }
        }
    }

    /// Embeds one message and stores its vectors. A failure counts against the
    /// session's consecutive failure limit and moves on.
    fn index_one(&mut self, model_id: &str, message: &ChatMessage) {
        let chunks = chunk_text(&message.text);
        let Some(runtime) = self.memory_runtime.as_mut() else {
            return;
        };
        let texts: Vec<String> = chunks
            .iter()
            .map(|chunk| format!("{}{}", runtime.markers.document, chunk))
            .collect();
        let embedded = runtime.engine.embed(&texts);

        match embedded {
            Ok(mut vectors) => {
                let count_ok = vectors.len() == texts.len();
                let normal_ok = count_ok && vectors.iter_mut().all(|vector| normalize(vector));
                if !normal_ok {
                    if let Some(runtime) = self.memory_runtime.as_mut() {
                        runtime.failures += 1;
                    }
                    return;
                }
                let stored = self.store_mut().and_then(|store| {
                    store
                        .append_embeddings(message.id, model_id, &vectors)
                        .map_err(|error| error.to_string())
                });
                if stored.is_err() {
                    if let Some(runtime) = self.memory_runtime.as_mut() {
                        runtime.failures += 1;
                    }
                    return;
                }
                if let Some(runtime) = self.memory_runtime.as_mut() {
                    for vector in vectors {
                        runtime.index.insert(message.id, vector);
                    }
                    runtime.failures = 0;
                }
            }
            Err(_) => {
                if let Some(runtime) = self.memory_runtime.as_mut() {
                    runtime.failures += 1;
                }
            }
        }
    }

    /// Runs one turn and streams its events to `events`.
    fn send(
        &mut self,
        text: &str,
        settings: &TurnSettings,
        events: &Sender<TurnMessage>,
        acks: &Receiver<()>,
    ) -> Result<(), String> {
        if self.state() != ChatState::Ready {
            return Err(NOT_STARTED.to_string());
        }
        if text.trim().is_empty() {
            return Ok(());
        }

        self.last_settings = Some(settings.clone());

        // Each turn starts with a fresh cancel flag.
        self.cancel.store(false, Ordering::SeqCst);

        self.store_mut()?
            .append(Role::User, text, false)
            .map_err(|error| error.to_string())?;

        self.set_status(ChatState::Generating, None);

        let system = system_text(&settings.system_addition);
        let context_tokens = match self.engine.as_ref() {
            Some(engine) => engine.context_tokens(),
            None => return Err(NOT_STARTED.to_string()),
        };
        let hard_limit = compaction_thresholds(
            u32::try_from(context_tokens).unwrap_or(u32::MAX),
            &settings.compaction,
        )
        .hard_limit as usize;

        let (mut summary, mut messages) = self.working_memory()?;
        let too_big = match self.engine.as_ref() {
            Some(engine) => {
                working_memory_tokens(&system, summary.as_deref(), &messages, &|piece| {
                    engine.count_tokens(piece)
                }) > hard_limit
            }
            None => return Err(NOT_STARTED.to_string()),
        };
        if too_big {
            let _ = events.send(TurnMessage::Event(ChatEvent::Compacting));
            let cancel = Arc::clone(&self.cancel);
            self.compact(settings, hard_limit, &cancel);
            let (next_summary, next_messages) = self.working_memory()?;
            summary = next_summary;
            messages = next_messages;
        }

        let Some(engine) = self.engine.as_ref() else {
            return Err(NOT_STARTED.to_string());
        };
        let prompt = build_prompt(
            &settings.system_addition,
            "",
            summary.as_deref(),
            &messages,
            context_tokens,
            &|piece| engine.count_tokens(piece),
        );

        if settings.persist_prefix_cache && !self.prefix_load_tried {
            self.prefix_load_tried = true;
            self.maybe_load_prefix(settings);
        }

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

    /// Loads the persisted prefix state on the first turn of a session, when
    /// the engine supports it. Failures are silent: the turn simply decodes
    /// the prefix as usual.
    fn maybe_load_prefix(&mut self, settings: &TurnSettings) {
        let Some(identity) = self
            .engine
            .as_ref()
            .and_then(|engine| engine.cache_identity())
        else {
            return;
        };
        let Some(model_path) = self
            .started_config
            .as_ref()
            .map(|config| config.model_path.clone())
        else {
            return;
        };
        let prefix = prefix_messages(settings);
        let Ok(key) = PrefixKey::new(&model_path, &identity, &prefix[0].text) else {
            return;
        };
        let Some(path) = self.prefix_cache.lookup(&key) else {
            return;
        };
        if let Some(engine) = self.engine.as_mut() {
            if matches!(engine.load_prefix(&prefix, &path), Ok(true)) {
                self.prefix_saved = true;
            }
        }
    }

    /// Saves the prefix state at most once per session, in the idle window
    /// after a reply. Failures are silent and never change the chat state.
    fn save_prefix_cache(&mut self, settings: &TurnSettings) {
        if !settings.persist_prefix_cache
            || self.prefix_saved
            || self.idle_cancel.load(Ordering::SeqCst)
        {
            return;
        }
        let Some(identity) = self
            .engine
            .as_ref()
            .and_then(|engine| engine.cache_identity())
        else {
            return;
        };
        let Some(model_path) = self
            .started_config
            .as_ref()
            .map(|config| config.model_path.clone())
        else {
            return;
        };
        let prefix = prefix_messages(settings);
        let Ok(key) = PrefixKey::new(&model_path, &identity, &prefix[0].text) else {
            return;
        };
        self.prefix_saved = true;
        if self.prefix_cache.lookup(&key).is_some() {
            return;
        }
        let Some(engine) = self.engine.as_mut() else {
            return;
        };
        let _ = self.prefix_cache.store(&key, &mut |path| {
            engine.save_prefix(&prefix, path).map_err(|error| error.0)
        });
    }

    /// Reads stored history, opening the store if needed.
    fn history(&mut self, limit: usize) -> Result<Vec<ChatMessage>, String> {
        self.store_mut()?
            .recent(limit)
            .map_err(|error| error.to_string())
    }
}

/// The fixed prompt prefix for `settings`: the system message alone.
fn prefix_messages(settings: &TurnSettings) -> Vec<PromptMessage> {
    vec![PromptMessage {
        role: PromptRole::System,
        text: system_text(&settings.system_addition),
    }]
}

/// The history database path for a test directory.
#[cfg(test)]
fn db_path(dir: &tempfile::TempDir) -> PathBuf {
    dir.path().join("regent.db")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::compaction::{KEEP_VERBATIM, SUMMARY_HEADING};
    use crate::chat::engine::FakeEngine;
    use crate::chat::prompt::PromptMessage;
    use crate::memory::FakeEmbedder;
    use serde_json::json;
    use std::path::Path;
    use std::time::{Duration, Instant};

    /// The window size the compaction tests use: trigger 288, target 0 (summary
    /// cap 32) and hard limit 488 with the default settings.
    const CONTEXT_TOKENS: usize = 1000;

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

    /// `count` distinct words: the fake engine counts one token per word.
    fn words(count: usize, label: &str) -> String {
        (0..count)
            .map(|index| format!("{label}{index}"))
            .collect::<Vec<_>>()
            .join(" ")
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

    /// A memory sink that records every change.
    fn recording_memory_sink() -> (MemorySink, Arc<Mutex<Vec<MemoryStatus>>>) {
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let sink: MemorySink = {
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

    /// Starts a service whose memory sink records status changes.
    fn spawn_service(
        db_path: PathBuf,
        prefix_cache_dir: PathBuf,
        factory: EngineFactory,
        sink: StatusSink,
    ) -> ChatService {
        ChatService::spawn(ChatServiceDeps {
            db_path,
            prefix_cache_dir,
            engine_factory: factory,
            embedding_factory: Box::new(|_config| Ok(Box::new(FakeEmbedder::new()))),
            status_sink: sink,
            memory_sink: Box::new(|_| {}),
        })
    }

    /// Starts a service with a custom embedding factory and memory sink.
    fn spawn_with_embedding(
        dir: &tempfile::TempDir,
        embedding_factory: EmbeddingFactory,
        memory_sink: MemorySink,
    ) -> ChatService {
        ChatService::spawn(ChatServiceDeps {
            db_path: db_path(dir),
            prefix_cache_dir: prefix_dir(dir),
            engine_factory: piece_factory(&["hi"]),
            embedding_factory,
            status_sink: Box::new(|_| {}),
            memory_sink,
        })
    }

    /// An embedding config whose model file exists.
    fn embedding_config(dir: &tempfile::TempDir) -> EmbeddingConfig {
        let model_path = dir.path().join("nomic-embed.gguf");
        std::fs::write(&model_path, b"fake embedding model").expect("write embedding fixture");
        EmbeddingConfig {
            model_path,
            gpu_layers: 0,
            context_size: 2048,
            document_prefix: None,
            query_prefix: None,
        }
    }

    /// The model id the service derives for `config`.
    fn id_for(config: &EmbeddingConfig) -> String {
        let file_name = config
            .model_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let markers = resolve_markers(
            &file_name,
            config.document_prefix.as_deref(),
            config.query_prefix.as_deref(),
        );
        model_id(&config.model_path, &markers).expect("model id")
    }

    /// Waits up to 5 s for a recorded memory status to satisfy `done`.
    fn wait_for_memory(
        statuses: &Arc<Mutex<Vec<MemoryStatus>>>,
        done: impl Fn(&MemoryStatus) -> bool,
    ) -> Option<MemoryStatus> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let found = statuses
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .iter()
                .find(|status| done(status))
                .cloned();
            if found.is_some() || Instant::now() >= deadline {
                return found;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// The logged texts, if an embedder log is attached.
    fn embedded_texts(log: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
        log.lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
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

    fn prefix_dir(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("kv")
    }

    fn service(dir: &tempfile::TempDir, factory: EngineFactory) -> ChatService {
        spawn_service(db_path(dir), prefix_dir(dir), factory, Box::new(|_| {}))
    }

    fn states(statuses: &Arc<Mutex<Vec<ChatStatus>>>) -> Vec<ChatState> {
        statuses
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|status| status.state)
            .collect()
    }

    /// Whether a `Compacting` state is followed by a `Ready` one.
    fn compaction_finished(statuses: &Arc<Mutex<Vec<ChatStatus>>>) -> bool {
        let mut compacting = false;
        for state in states(statuses) {
            match state {
                ChatState::Compacting => compacting = true,
                ChatState::Ready if compacting => return true,
                _ => {}
            }
        }
        false
    }

    /// Waits up to 5 s for the idle pass, without sending a command (which
    /// would cancel it).
    fn wait_for_compaction(statuses: &Arc<Mutex<Vec<ChatStatus>>>) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if compaction_finished(statuses) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        compaction_finished(statuses)
    }

    /// Seeds `count` messages of `width` words each, directly through the store.
    fn seed(path: &Path, count: usize, width: usize) {
        let mut store = ChatStore::open(path).expect("open store");
        for index in 0..count {
            let role = if index % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            };
            store
                .append(role, &words(width, &format!("seed{index}_")), false)
                .expect("append seed message");
        }
    }

    #[test]
    fn start_then_send_streams_tokens_and_saves_both_messages() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (sink, statuses) = recording_sink();
        let service = spawn_service(
            db_path(&dir),
            prefix_dir(&dir),
            piece_factory(&["Hello", " there"]),
            sink,
        );

        service
            .start(Some(config(&dir)), None)
            .expect("start should succeed");
        assert_eq!(service.status().state, ChatState::Ready);

        let mut events = Vec::new();
        service
            .send("hi", &TurnSettings::default(), &mut |event| {
                events.push(event)
            })
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
        assert_eq!(done.1.reused_tokens, 0);
        assert_eq!(done.1.processed_tokens, done.1.prompt_tokens);

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
        assert_eq!(
            states(&statuses),
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
        service.start(Some(config(&dir)), None).expect("start");

        let mut events = Vec::new();
        service
            .send("hi", &TurnSettings::default(), &mut |event| {
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
            .send("hi", &TurnSettings::default(), &mut |_| {})
            .expect_err("send should fail");
        assert_eq!(error, "Start Regent first.");
        assert!(service.history(10).expect("history").is_empty());
    }

    #[test]
    fn starting_without_a_model_or_with_a_missing_file_gives_exact_messages() {
        let dir = tempfile::tempdir().expect("temp dir");
        let service = service(&dir, piece_factory(&["never"]));

        assert_eq!(
            service.start(None, None).expect_err("start should fail"),
            "No text model chosen. Open Settings and pick one."
        );

        let missing = EngineConfig {
            model_path: dir.path().join("nothing-here.gguf"),
            gpu_layers: 0,
            context_size: 4096,
        };
        assert_eq!(
            service
                .start(Some(missing), None)
                .expect_err("start should fail"),
            "The text model file couldn't be found. Choose it again in Settings."
        );
        assert_eq!(service.status().state, ChatState::Stopped);
    }

    #[test]
    fn starting_twice_errors() {
        let dir = tempfile::tempdir().expect("temp dir");
        let service = service(&dir, piece_factory(&["hi"]));
        let config = config(&dir);
        service.start(Some(config.clone()), None).expect("start");

        assert_eq!(
            service
                .start(Some(config), None)
                .expect_err("second start should fail"),
            "Regent is already running."
        );
    }

    #[test]
    fn history_survives_a_new_service_on_the_same_file() {
        let dir = tempfile::tempdir().expect("temp dir");
        {
            let service = service(&dir, piece_factory(&["remembered"]));
            service.start(Some(config(&dir)), None).expect("start");
            service
                .send("hi", &TurnSettings::default(), &mut |_| {})
                .expect("send");
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
        service.start(Some(config(&dir)), None).expect("start");

        let mut events = Vec::new();
        service
            .send("hi", &TurnSettings::default(), &mut |event| {
                events.push(event)
            })
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
        service.start(Some(config(&dir)), None).expect("start");

        let mut count = 0;
        service
            .send("   ", &TurnSettings::default(), &mut |_| count += 1)
            .expect("send should succeed");
        assert_eq!(count, 0);
        assert!(service.history(10).expect("history").is_empty());
    }

    #[test]
    fn stopped_after_stop_and_can_start_again() {
        let dir = tempfile::tempdir().expect("temp dir");
        let service = service(&dir, piece_factory(&["hi"]));
        service.start(Some(config(&dir)), None).expect("start");
        service.stop();
        assert_eq!(service.status().state, ChatState::Stopped);

        service.start(Some(config(&dir)), None).expect("restart");
        assert_eq!(service.status().state, ChatState::Ready);
    }

    #[test]
    fn idle_compaction_stores_a_summary_after_a_reply() {
        let dir = tempfile::tempdir().expect("temp dir");
        let stored_summary = "they talked about the weather";
        let reply = words(30, "reply");
        let factory: EngineFactory = {
            let reply = reply.clone();
            Box::new(move |_config| {
                Ok(Box::new(
                    FakeEngine::new(vec![reply.clone()], CONTEXT_TOKENS)
                        .with_summaries(vec![stored_summary.to_string()]),
                ))
            })
        };
        let (sink, statuses) = recording_sink();
        let service = spawn_service(db_path(&dir), prefix_dir(&dir), factory, sink);
        service.start(Some(config(&dir)), None).expect("start");

        // The system message costs 34 words + 4 template tokens, so after four
        // 30-word turns (8 messages) the prompt passes the 288-token trigger.
        let mut compacted = false;
        for turn in 0..6 {
            let text = words(30, &format!("turn{turn}_"));
            service
                .send(&text, &TurnSettings::default(), &mut |_| {})
                .expect("send");
            if turn >= 3 && wait_for_compaction(&statuses) {
                compacted = true;
                break;
            }
        }
        assert!(compacted, "idle compaction should run after a reply");

        let seen = states(&statuses);
        let compacting_at = seen
            .iter()
            .position(|state| *state == ChatState::Compacting)
            .expect("a Compacting status");
        assert!(
            seen[compacting_at + 1..].contains(&ChatState::Ready),
            "Compacting should be followed by Ready: {seen:?}"
        );

        let store = ChatStore::open(&db_path(&dir)).expect("open a second store");
        let summary = store
            .latest_summary()
            .expect("read summary")
            .expect("a stored summary");
        assert_eq!(summary.text, stored_summary);

        let history = store.recent(10).expect("history");
        assert!(history.len() > KEEP_VERBATIM);
        assert_eq!(
            summary.covers_through,
            history[history.len() - KEEP_VERBATIM - 1].id,
            "the newest {KEEP_VERBATIM} messages stay out of the summary"
        );
    }

    #[test]
    fn an_oversized_prompt_compacts_before_generating() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = db_path(&dir);
        seed(&path, 14, 40);
        let first_seed = words(40, "seed0_");
        let folded_summary = "the user and regent discussed seeded topics";
        let log = Arc::new(Mutex::new(Vec::new()));
        let factory: EngineFactory = {
            let log = Arc::clone(&log);
            Box::new(move |_config| {
                Ok(Box::new(
                    FakeEngine::new(vec!["ok".to_string()], CONTEXT_TOKENS)
                        .with_summaries(vec![folded_summary.to_string(); 4])
                        .with_prompt_log(Arc::clone(&log)),
                ))
            })
        };
        let service = spawn_service(path.clone(), prefix_dir(&dir), factory, Box::new(|_| {}));
        service.start(Some(config(&dir)), None).expect("start");

        let mut events = Vec::new();
        service
            .send(
                &words(40, "newest_"),
                &TurnSettings::default(),
                &mut |event| events.push(event),
            )
            .expect("send");

        let compacting_at = events
            .iter()
            .position(|event| matches!(event, ChatEvent::Compacting))
            .expect("a Compacting event");
        let first_token = events
            .iter()
            .position(|event| matches!(event, ChatEvent::Token { .. }))
            .expect("a Token event");
        assert!(
            compacting_at < first_token,
            "Compacting must come before the reply: {events:?}"
        );

        let prompts = log
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let generate: &Vec<PromptMessage> = prompts.last().expect("a logged prompt");
        assert!(generate[0].text.contains(SUMMARY_HEADING));
        assert!(generate[0].text.contains(folded_summary));
        assert!(
            generate
                .iter()
                .all(|message| !message.text.contains(&first_seed)),
            "the folded messages are left out of the prompt"
        );

        assert!(events
            .iter()
            .any(|event| matches!(event, ChatEvent::Done { .. })));
    }

    #[test]
    fn a_failed_summary_still_answers() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = db_path(&dir);
        seed(&path, 14, 40);
        let service = service(&dir, {
            let pieces = vec!["ok".to_string()];
            Box::new(move |_config| Ok(Box::new(FakeEngine::new(pieces.clone(), CONTEXT_TOKENS))))
        });
        service.start(Some(config(&dir)), None).expect("start");

        let mut events = Vec::new();
        service
            .send(
                &words(40, "newest_"),
                &TurnSettings::default(),
                &mut |event| events.push(event),
            )
            .expect("send should succeed even without a summary");

        assert!(events
            .iter()
            .any(|event| matches!(event, ChatEvent::Done { .. })));
        assert!(events
            .iter()
            .any(|event| matches!(event, ChatEvent::Compacting)));

        let store = ChatStore::open(&path).expect("open a second store");
        assert_eq!(store.latest_summary().expect("read summary"), None);
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

        let compacting = ChatEvent::Compacting;
        assert_eq!(
            serde_json::to_value(&compacting).expect("serialise"),
            json!({"type": "compacting"})
        );

        let done = ChatEvent::Done {
            message_id: 7,
            stats: GenerationStats {
                prompt_tokens: 3,
                reused_tokens: 1,
                processed_tokens: 2,
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
                    "reused_tokens": 1,
                    "processed_tokens": 2,
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
            (ChatState::Compacting, "compacting"),
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
        let service = spawn_service(db_path(&dir), prefix_dir(&dir), factory, sink);

        let config = config(&dir);
        assert_eq!(
            service
                .start(Some(config.clone()), None)
                .expect_err("first start should fail"),
            "The model didn't load."
        );
        let status = service.status();
        assert_eq!(status.state, ChatState::Error);
        assert_eq!(status.error.as_deref(), Some("The model didn't load."));

        service
            .start(Some(config), None)
            .expect("retry should work");
        assert_eq!(service.status().state, ChatState::Ready);

        assert_eq!(
            states(&statuses),
            vec![
                ChatState::Loading,
                ChatState::Error,
                ChatState::Loading,
                ChatState::Ready
            ]
        );
    }

    /// Settings with prefix persistence on.
    fn prefix_settings() -> TurnSettings {
        TurnSettings {
            persist_prefix_cache: true,
            ..TurnSettings::default()
        }
    }

    /// A factory whose engine has a prefix cache logging into `log`.
    fn prefix_factory(log: &Arc<Mutex<Vec<String>>>) -> EngineFactory {
        let log = Arc::clone(log);
        Box::new(move |_config| {
            Ok(Box::new(
                FakeEngine::new(vec!["ok".to_string()], 4096).with_prefix_cache(Arc::clone(&log)),
            ))
        })
    }

    fn log_entries(log: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
        log.lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Waits up to 5 s for the idle pass to log `needle`.
    fn wait_for_log(log: &Arc<Mutex<Vec<String>>>, needle: &str) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if log_entries(log).iter().any(|entry| entry == needle) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        log_entries(log).iter().any(|entry| entry == needle)
    }

    fn state_files(dir: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|entry| entry.path())
                    .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("state"))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn wait_for_state_files(dir: &Path) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if !state_files(dir).is_empty() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        !state_files(dir).is_empty()
    }

    #[test]
    fn a_prefix_cache_is_saved_then_loaded_by_a_new_session() {
        let dir = tempfile::tempdir().expect("temp dir");
        let kv = prefix_dir(&dir);
        let config = config(&dir);

        let first_log = Arc::new(Mutex::new(Vec::new()));
        {
            let service = spawn_service(
                db_path(&dir),
                kv.clone(),
                prefix_factory(&first_log),
                Box::new(|_| {}),
            );
            service.start(Some(config.clone()), None).expect("start");
            service
                .send("hi", &prefix_settings(), &mut |_| {})
                .expect("send");

            assert!(
                wait_for_log(&first_log, "save"),
                "the first session should save its prefix: {:?}",
                log_entries(&first_log)
            );
            assert!(
                wait_for_state_files(&kv),
                "a persisted state file should exist"
            );
        }

        let second_log = Arc::new(Mutex::new(Vec::new()));
        let service = spawn_service(
            db_path(&dir),
            kv.clone(),
            prefix_factory(&second_log),
            Box::new(|_| {}),
        );
        service.start(Some(config), None).expect("start");

        let mut at_first_token = None;
        service
            .send("again", &prefix_settings(), &mut |event| {
                if at_first_token.is_none() && matches!(event, ChatEvent::Token { .. }) {
                    at_first_token = Some(log_entries(&second_log));
                }
            })
            .expect("send");

        let at_first_token = at_first_token.expect("a Token event");
        assert!(
            at_first_token.iter().any(|entry| entry == "load:true"),
            "the prefix should load before the reply: {at_first_token:?}"
        );
    }

    #[test]
    fn the_prefix_cache_stays_empty_when_persistence_is_off() {
        let dir = tempfile::tempdir().expect("temp dir");
        let kv = prefix_dir(&dir);
        let log = Arc::new(Mutex::new(Vec::new()));
        let service = spawn_service(
            db_path(&dir),
            kv.clone(),
            prefix_factory(&log),
            Box::new(|_| {}),
        );
        service.start(Some(config(&dir)), None).expect("start");

        service
            .send("hi", &TurnSettings::default(), &mut |_| {})
            .expect("send");
        std::thread::sleep(Duration::from_millis(100));

        assert!(
            log_entries(&log).is_empty(),
            "no prefix work should run: {:?}",
            log_entries(&log)
        );
        assert!(state_files(&kv).is_empty());
    }

    /// A logging embedding factory and the log it writes to.
    fn logging_factory() -> (EmbeddingFactory, Arc<Mutex<Vec<String>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let factory_log = Arc::clone(&log);
        let factory: EmbeddingFactory = Box::new(move |_config| {
            Ok(Box::new(
                FakeEmbedder::new().with_log(Arc::clone(&factory_log)),
            ))
        });
        (factory, log)
    }

    /// Seeds the database and indexes it with `embedding`, leaving the vectors
    /// stored and the service dropped.
    fn seed_and_index(dir: &tempfile::TempDir, embedding: &EmbeddingConfig) {
        let path = db_path(dir);
        seed(&path, 3, 2);
        let (factory, _log) = logging_factory();
        let (memory_sink, statuses) = recording_memory_sink();
        let service = spawn_with_embedding(dir, factory, memory_sink);
        service
            .start(Some(config(dir)), Some(embedding.clone()))
            .expect("start");
        wait_for_memory(&statuses, |status| {
            status.state == MemoryState::Ready && status.indexed == 3 && status.total == 3
        })
        .expect("the first session should index 3 of 3");
    }

    #[test]
    fn memory_indexes_seeded_messages_and_reports_ready() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = db_path(&dir);
        seed(&path, 3, 2);
        let embedding = embedding_config(&dir);
        let (factory, log) = logging_factory();
        let (memory_sink, statuses) = recording_memory_sink();
        let service = spawn_with_embedding(&dir, factory, memory_sink);

        service
            .start(Some(config(&dir)), Some(embedding.clone()))
            .expect("start");

        let ready = wait_for_memory(&statuses, |status| {
            status.state == MemoryState::Ready && status.indexed == 3 && status.total == 3
        })
        .expect("memory should reach Ready with 3 of 3 indexed");
        assert_eq!(ready.error, None);

        let seen: Vec<MemoryState> = statuses
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|status| status.state)
            .collect();
        assert!(
            seen.contains(&MemoryState::Indexing),
            "the status should pass through Indexing: {seen:?}"
        );

        assert_eq!(
            embedded_texts(&log).len(),
            3,
            "each seeded message should be embedded once"
        );

        let store = ChatStore::open(&path).expect("open a second store");
        let vectors = store
            .load_vectors(&id_for(&embedding))
            .expect("load vectors");
        assert!(!vectors.is_empty(), "the vectors should be stored");
    }

    #[test]
    fn a_new_service_does_not_reembed_stored_vectors() {
        let dir = tempfile::tempdir().expect("temp dir");
        let embedding = embedding_config(&dir);
        seed_and_index(&dir, &embedding);

        let (factory, log) = logging_factory();
        let (memory_sink, statuses) = recording_memory_sink();
        let service = spawn_with_embedding(&dir, factory, memory_sink);

        service
            .start(Some(config(&dir)), Some(embedding))
            .expect("start");

        let ready = wait_for_memory(&statuses, |status| {
            status.state == MemoryState::Ready && status.indexed == 3 && status.total == 3
        })
        .expect("the second session should report 3 of 3");
        assert_eq!(ready.indexed, 3);

        std::thread::sleep(Duration::from_millis(100));
        assert!(
            embedded_texts(&log).is_empty(),
            "nothing should be embedded again: {:?}",
            embedded_texts(&log)
        );
    }

    #[test]
    fn changing_the_document_prefix_reindexes_everything() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut embedding = embedding_config(&dir);
        seed_and_index(&dir, &embedding);

        embedding.document_prefix = Some("doc: ".to_string());
        let (factory, log) = logging_factory();
        let (memory_sink, statuses) = recording_memory_sink();
        let service = spawn_with_embedding(&dir, factory, memory_sink);

        service
            .start(Some(config(&dir)), Some(embedding))
            .expect("start");

        wait_for_memory(&statuses, |status| {
            status.state == MemoryState::Ready && status.indexed == 3 && status.total == 3
        })
        .expect("the new marker should index 3 of 3");

        assert_eq!(
            embedded_texts(&log).len(),
            3,
            "all three messages should be re-embedded"
        );
    }

    #[test]
    fn repeated_embedding_failures_pause_indexing() {
        let dir = tempfile::tempdir().expect("temp dir");
        seed(&db_path(&dir), 3, 2);
        let embedding = embedding_config(&dir);
        let (memory_sink, statuses) = recording_memory_sink();
        let factory: EmbeddingFactory =
            Box::new(|_config| Ok(Box::new(FakeEmbedder::new().with_failures(10))));
        let service = spawn_with_embedding(&dir, factory, memory_sink);

        service
            .start(Some(config(&dir)), Some(embedding))
            .expect("start");

        let paused = wait_for_memory(&statuses, |status| status.state == MemoryState::Paused)
            .expect("indexing should pause after repeated failures");
        assert_eq!(paused.error.as_deref(), Some(INDEX_PAUSED));
    }

    #[test]
    fn starting_without_an_embedding_config_turns_memory_off() {
        let dir = tempfile::tempdir().expect("temp dir");
        let service = service(&dir, piece_factory(&["hi"]));

        service.start(Some(config(&dir)), None).expect("start");

        assert_eq!(service.status().state, ChatState::Ready);
        let status = service.memory_status();
        assert_eq!(status.state, MemoryState::Off);
        assert_eq!((status.indexed, status.total), (0, 0));
    }

    #[test]
    fn a_missing_embedding_file_turns_memory_off_with_the_exact_error() {
        let dir = tempfile::tempdir().expect("temp dir");
        let service = service(&dir, piece_factory(&["hi"]));
        let missing = EmbeddingConfig {
            model_path: dir.path().join("nothing-here.gguf"),
            gpu_layers: 0,
            context_size: 2048,
            document_prefix: None,
            query_prefix: None,
        };

        service
            .start(Some(config(&dir)), Some(missing))
            .expect("start should still succeed");

        assert_eq!(service.status().state, ChatState::Ready);
        let status = service.memory_status();
        assert_eq!(status.state, MemoryState::Off);
        assert_eq!(
            status.error.as_deref(),
            Some("The memory model file couldn't be found. Choose it again in Settings.")
        );
    }

    #[test]
    fn stopping_turns_memory_off() {
        let dir = tempfile::tempdir().expect("temp dir");
        let embedding = embedding_config(&dir);
        let (factory, _log) = logging_factory();
        let (memory_sink, statuses) = recording_memory_sink();
        let service = spawn_with_embedding(&dir, factory, memory_sink);

        service
            .start(Some(config(&dir)), Some(embedding))
            .expect("start");
        wait_for_memory(&statuses, |status| status.state == MemoryState::Ready)
            .expect("memory should reach Ready");

        service.stop();
        assert_eq!(service.memory_status().state, MemoryState::Off);
    }
}
