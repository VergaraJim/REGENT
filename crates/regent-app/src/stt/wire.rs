//! The app side of the `regent-stt` stdin/stdout wire format.
//!
//! This is a small copy of `crates/regent-stt/src/protocol.rs`. The helper is
//! deliberately *not* a dependency of `regent-app`: cargo feature unification
//! would enable the helper's default `whisper` feature and link a second ggml
//! next to llama.cpp, which is the clash decision 0005 avoids. Keep the two
//! definitions in sync; see `docs/decisions/0005-stt-helper.md`.
//!
//! Requests are binary frames, little-endian:
//!
//! ```text
//! id:           u64
//! sample_rate:  u32
//! sample_count: u32
//! samples:      sample_count × f32 (mono)
//! ```
//!
//! Responses are one UTF-8 JSON object per line.

use serde::{Deserialize, Serialize};

/// Encode one request frame: a 16-byte header followed by little-endian f32
/// samples. Mirrors `protocol::write_request` in `crates/regent-stt`.
pub fn encode_frame(id: u64, sample_rate: u32, samples: &[f32]) -> Vec<u8> {
    let sample_count = u32::try_from(samples.len()).unwrap_or(u32::MAX);

    let mut frame = Vec::with_capacity(16 + std::mem::size_of_val(samples));
    frame.extend_from_slice(&id.to_le_bytes());
    frame.extend_from_slice(&sample_rate.to_le_bytes());
    frame.extend_from_slice(&sample_count.to_le_bytes());
    for sample in samples {
        frame.extend_from_slice(&sample.to_le_bytes());
    }
    frame
}

/// One JSON response line from the helper.
///
/// The variants and their fields mirror `regent_stt::protocol::Response`.
/// Fields the app does not currently read (`version`, `duration_ms`) are part
/// of the protocol and kept so the copy stays faithful.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Response {
    Ready {
        version: String,
    },
    Transcript {
        id: u64,
        text: String,
        duration_ms: u64,
    },
    Error {
        id: u64,
        message: String,
    },
    Fatal {
        message: String,
    },
}

/// Parse one response line, returning `None` for anything unparsable (a stray
/// log line, for example).
pub fn parse_response(line: &str) -> Option<Response> {
    serde_json::from_str(line).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_header_is_little_endian_and_the_length_matches_the_samples() {
        let frame = encode_frame(7, 16_000, &[0.5, -0.25]);

        assert_eq!(frame.len(), 16 + 2 * std::mem::size_of::<f32>());
        assert_eq!(u64::from_le_bytes(frame[0..8].try_into().unwrap()), 7);
        assert_eq!(u32::from_le_bytes(frame[8..12].try_into().unwrap()), 16_000);
        assert_eq!(u32::from_le_bytes(frame[12..16].try_into().unwrap()), 2);
        assert_eq!(f32::from_le_bytes(frame[16..20].try_into().unwrap()), 0.5);
        assert_eq!(f32::from_le_bytes(frame[20..24].try_into().unwrap()), -0.25);
    }

    #[test]
    fn an_empty_frame_still_has_a_header() {
        let frame = encode_frame(1, 16_000, &[]);

        assert_eq!(frame.len(), 16);
        assert_eq!(u32::from_le_bytes(frame[12..16].try_into().unwrap()), 0);
    }

    #[test]
    fn every_response_shape_parses() {
        assert_eq!(
            parse_response(r#"{"type":"ready","version":"1.2.3"}"#),
            Some(Response::Ready {
                version: "1.2.3".to_string()
            })
        );
        assert_eq!(
            parse_response(r#"{"type":"transcript","id":4,"text":"hello","duration_ms":12}"#),
            Some(Response::Transcript {
                id: 4,
                text: "hello".to_string(),
                duration_ms: 12
            })
        );
        assert_eq!(
            parse_response(r#"{"type":"error","id":5,"message":"bad rate"}"#),
            Some(Response::Error {
                id: 5,
                message: "bad rate".to_string()
            })
        );
        assert_eq!(
            parse_response(r#"{"type":"fatal","message":"no model"}"#),
            Some(Response::Fatal {
                message: "no model".to_string()
            })
        );
    }

    #[test]
    fn garbage_lines_are_skipped() {
        assert_eq!(parse_response("not json at all"), None);
        assert_eq!(parse_response(r#"{"type":"unknown"}"#), None);
        assert_eq!(parse_response(""), None);
    }
}
