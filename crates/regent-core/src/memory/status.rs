//! The memory indexer's progress, as the frontend sees it.

use serde::Serialize;

/// Where memory indexing is in its life cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryState {
    /// No embedding model is loaded.
    Off,
    /// The model is loaded and everything indexable is indexed.
    Ready,
    /// Messages are being embedded.
    Indexing,
    /// Repeated failures stopped indexing for this session.
    Paused,
}

/// The memory indexer's current state, as the frontend sees it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MemoryStatus {
    /// The current state.
    pub state: MemoryState,
    /// How many indexable messages are indexed.
    pub indexed: usize,
    /// How many indexable messages exist.
    pub total: usize,
    /// The plain-language error, when something went wrong.
    pub error: Option<String>,
}

/// Receives every memory status change.
pub type MemorySink = Box<dyn Fn(MemoryStatus) + Send + Sync>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_states_serialise_snake_case() {
        for (state, expected) in [
            (MemoryState::Off, "off"),
            (MemoryState::Ready, "ready"),
            (MemoryState::Indexing, "indexing"),
            (MemoryState::Paused, "paused"),
        ] {
            assert_eq!(
                serde_json::to_value(state).expect("serialise"),
                serde_json::json!(expected)
            );
        }
    }

    #[test]
    fn memory_status_json_has_the_documented_keys() {
        let status = MemoryStatus {
            state: MemoryState::Ready,
            indexed: 2,
            total: 3,
            error: None,
        };
        let json = serde_json::to_value(&status).expect("serialise");
        assert_eq!(
            json,
            serde_json::json!({
                "state": "ready",
                "indexed": 2,
                "total": 3,
                "error": null,
            })
        );
    }
}
