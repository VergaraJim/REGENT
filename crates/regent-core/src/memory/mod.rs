//! Long-term memory without a model.
//!
//! This module holds everything the memory feature needs that runs without an
//! embedding model (spec §5.6) and the output format of `search_memory`
//! (spec §5.7):
//!
//! - [`chunk`]: splitting a message into overlapping character windows.
//! - [`vector`]: vector maths and an exact brute-force cosine index.
//! - [`markers`]: the instruction prefixes an embedding model expects, and the
//!   id that keeps vectors from different models apart.
//! - [`recall`]: turning recalled messages into the text the model reads.
//!
//! It has no I/O apart from the one `fs::metadata` call in
//! [`markers::model_id`].

pub mod chunk;
pub mod engine;
pub mod markers;
pub mod recall;
pub mod status;
pub mod vector;

pub use chunk::{chunk_text, CHUNK_CHARS, CHUNK_OVERLAP_CHARS};
pub use engine::{EmbeddingConfig, EmbeddingEngine, EmbeddingFactory, FakeEmbedder};
pub use markers::{detect_markers, model_id, resolve_markers, Markers};
pub use recall::{
    format_date, format_recall, NOTHING_INDEXED, NO_MATCHES, RECALL_MAX_CHARS, SNIPPET_CHARS,
};
pub use status::{MemorySink, MemoryState, MemoryStatus};
pub use vector::{dot, normalize, Hit, VectorIndex};

/// How many messages are embedded per indexing batch before the index writes
/// progress, so a model failure loses at most one batch.
pub const INDEX_BATCH: usize = 4;

/// How many consecutive index failures are tolerated before indexing stops
/// for the session.
pub const MAX_INDEX_FAILURES: u32 = 3;

/// How many recalled messages `search_memory` returns by default.
pub const DEFAULT_RECALL_LIMIT: usize = 4;

/// The most recalled messages `search_memory` will ever return.
pub const MAX_RECALL_LIMIT: usize = 8;
