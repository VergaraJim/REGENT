//! The instruction prefixes an embedding model expects.
//!
//! Embedding models are trained with different task prefixes for stored
//! documents and for search queries. Applying the wrong prefix, or mixing
//! vectors made with different prefixes, degrades recall badly, so the pair is
//! detected from the model file name and folded into the model id.

use std::fs;
use std::io;
use std::path::Path;

use crate::hash::fnv1a64;

/// The document and query prefixes for one embedding model.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Markers {
    /// Prepended to a message before it is embedded for storage.
    pub document: String,
    /// Prepended to a search query before it is embedded.
    pub query: String,
}

/// The markers the model file name suggests.
///
/// The name is matched case-insensitively; unknown models get no prefixes.
pub fn detect_markers(model_file_name: &str) -> Markers {
    let name = model_file_name.to_lowercase();
    let (document, query) = if name.contains("nomic-embed") {
        ("search_document: ", "search_query: ")
    } else if name.contains("embeddinggemma") {
        ("title: none | text: ", "task: search result | query: ")
    } else if name.contains("qwen3-embedding") {
        (
            "",
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        )
    } else if name.contains("e5-") {
        ("passage: ", "query: ")
    } else if name.contains("bge-m3") {
        ("", "")
    } else if name.contains("bge-")
        || name.contains("mxbai-embed")
        || name.contains("snowflake-arctic-embed")
    {
        (
            "",
            "Represent this sentence for searching relevant passages: ",
        )
    } else {
        ("", "")
    };
    Markers {
        document: document.to_string(),
        query: query.to_string(),
    }
}

/// The detected markers, with each `Some` override replacing that value.
///
/// `Some("")` deliberately clears a detected prefix.
pub fn resolve_markers(
    model_file_name: &str,
    document_override: Option<&str>,
    query_override: Option<&str>,
) -> Markers {
    let mut markers = detect_markers(model_file_name);
    if let Some(document) = document_override {
        markers.document = document.to_string();
    }
    if let Some(query) = query_override {
        markers.query = query.to_string();
    }
    markers
}

/// A stable id for the vectors `model_path` produces with `markers`.
///
/// Vectors made with another file, another file size or another document
/// marker never mix: they get different ids, so an index is never compared
/// with vectors from a different model.
pub fn model_id(model_path: &Path, markers: &Markers) -> io::Result<String> {
    let metadata = fs::metadata(model_path)?;
    let file_name = model_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(format!(
        "{file_name}|{}|{:016x}",
        metadata.len(),
        fnv1a64(markers.document.as_bytes())
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn markers(document: &str, query: &str) -> Markers {
        Markers {
            document: document.to_string(),
            query: query.to_string(),
        }
    }

    #[test]
    fn each_supported_model_has_its_markers() {
        let rows = [
            (
                "nomic-embed-text-v1.5.gguf",
                "search_document: ",
                "search_query: ",
            ),
            (
                "embeddinggemma-300m.gguf",
                "title: none | text: ",
                "task: search result | query: ",
            ),
            (
                "qwen3-embedding-0.6b.gguf",
                "",
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            ),
            ("e5-small-v2.gguf", "passage: ", "query: "),
            ("bge-m3.gguf", "", ""),
            (
                "bge-small-en.gguf",
                "",
                "Represent this sentence for searching relevant passages: ",
            ),
            (
                "mxbai-embed-large.gguf",
                "",
                "Represent this sentence for searching relevant passages: ",
            ),
            (
                "snowflake-arctic-embed-m.gguf",
                "",
                "Represent this sentence for searching relevant passages: ",
            ),
            ("all-minilm.gguf", "", ""),
        ];
        for (name, document, query) in rows {
            assert_eq!(detect_markers(name), markers(document, query), "{name}");
        }
    }

    #[test]
    fn detection_ignores_case() {
        assert_eq!(
            detect_markers("Nomic-Embed-Text.gguf"),
            markers("search_document: ", "search_query: ")
        );
    }

    #[test]
    fn e5_needs_its_dash() {
        assert_eq!(
            detect_markers("e5-small.gguf"),
            markers("passage: ", "query: ")
        );
        assert_eq!(detect_markers("model.gguf"), markers("", ""));
        assert_eq!(
            detect_markers("my-e5-model.gguf"),
            markers("passage: ", "query: ")
        );
    }

    #[test]
    fn overrides_replace_the_detected_values() {
        assert_eq!(
            resolve_markers("nomic-embed.gguf", Some("doc: "), Some("ask: ")),
            markers("doc: ", "ask: ")
        );
        assert_eq!(
            resolve_markers("nomic-embed.gguf", None, Some("ask: ")),
            markers("search_document: ", "ask: ")
        );
        assert_eq!(
            resolve_markers("nomic-embed.gguf", Some(""), Some("")),
            markers("", "")
        );
    }

    #[test]
    fn model_id_covers_the_file_size_and_document_marker() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("model.gguf");
        fs::write(&path, b"fake model").expect("write model fixture");

        let first =
            model_id(&path, &markers("search_document: ", "search_query: ")).expect("model id");
        assert!(first.starts_with("model.gguf|10|"), "{first}");

        let other_document =
            model_id(&path, &markers("passage: ", "search_query: ")).expect("model id");
        assert_ne!(first, other_document);

        let other_query =
            model_id(&path, &markers("search_document: ", "query: ")).expect("model id");
        assert_eq!(
            first, other_query,
            "the query marker does not change the id"
        );

        assert!(model_id(&dir.path().join("missing.gguf"), &Markers::default()).is_err());
    }
}
