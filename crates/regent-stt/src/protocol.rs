//! The stdin/stdout protocol between the app and the `regent-stt` helper.
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

use std::io::{self, Read, Write};

use serde::{Deserialize, Serialize};

/// The only sample rate the helper accepts.
pub const SAMPLE_RATE: u32 = 16_000;

/// The longest utterance the helper accepts: 35 seconds at 16 kHz.
pub const MAX_SAMPLES: u32 = 560_000;

const HEADER_BYTES: usize = 16;

/// One transcription request, already decoded from its wire frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub id: u64,
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

/// Everything that can go wrong while reading a request frame.
#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("i/o error: {0}")]
    Io(#[from] io::Error),

    #[error("unexpected end of input in the middle of a frame")]
    UnexpectedEof,

    #[error(
        "request {id} declares {sample_count} samples; the maximum is {max} samples",
        max = MAX_SAMPLES
    )]
    TooManySamples { id: u64, sample_count: u32 },
}

impl ProtocolError {
    /// Turn a protocol error into the `io::Error` the request loop reports.
    pub fn into_io_error(self) -> io::Error {
        match self {
            ProtocolError::Io(error) => error,
            other => io::Error::new(io::ErrorKind::InvalidData, other.to_string()),
        }
    }
}

/// Read one request frame.
///
/// Returns `Ok(None)` when the input ends cleanly between frames. A frame
/// with `sample_count` above [`MAX_SAMPLES`] returns [`ProtocolError::TooManySamples`]
/// *without consuming or allocating its payload*; the caller is expected to
/// answer with an error and then skip the payload in fixed-size chunks (see
/// [`crate::service::run_loop`]). That ordering means a short stream still gets
/// an error response instead of a bare EOF.
pub fn read_request<R: Read>(reader: &mut R) -> Result<Option<Request>, ProtocolError> {
    let Some(header) = read_header(reader)? else {
        return Ok(None);
    };

    let id = u64::from_le_bytes(header[0..8].try_into().expect("8-byte id"));
    let sample_rate = u32::from_le_bytes(header[8..12].try_into().expect("4-byte sample rate"));
    let sample_count = u32::from_le_bytes(header[12..16].try_into().expect("4-byte sample count"));

    if sample_count > MAX_SAMPLES {
        return Err(ProtocolError::TooManySamples { id, sample_count });
    }

    let byte_len = sample_count as usize * std::mem::size_of::<f32>();
    let mut raw = vec![0u8; byte_len];
    read_exact_or_eof(reader, &mut raw)?;

    let samples = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();

    Ok(Some(Request {
        id,
        sample_rate,
        samples,
    }))
}

/// Write one request frame. Used by the app side and by tests.
pub fn write_request<W: Write>(writer: &mut W, request: &Request) -> io::Result<()> {
    let sample_count = u32::try_from(request.samples.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "too many samples"))?;

    let mut header = [0u8; HEADER_BYTES];
    header[0..8].copy_from_slice(&request.id.to_le_bytes());
    header[8..12].copy_from_slice(&request.sample_rate.to_le_bytes());
    header[12..16].copy_from_slice(&sample_count.to_le_bytes());
    writer.write_all(&header)?;

    for sample in &request.samples {
        writer.write_all(&sample.to_le_bytes())?;
    }
    writer.flush()
}

/// One JSON response line.
///
/// `serde`'s internal tagging produces exactly the object shapes the spec asks
/// for: `type` first, then the variant's fields in declaration order.
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

/// Write one response as a single JSON line, then flush it.
pub fn write_response<W: Write>(writer: &mut W, response: &Response) -> io::Result<()> {
    serde_json::to_writer(&mut *writer, response).map_err(json_to_io)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

/// Parse one response line. Used by the app side and by tests.
pub fn parse_response(line: &str) -> Result<Response, serde_json::Error> {
    serde_json::from_str(line)
}

/// Length of the audio carried by a request, rounded down to whole ms.
pub fn duration_ms(request: &Request) -> u64 {
    if request.sample_rate == 0 {
        return 0;
    }
    request.samples.len() as u64 * 1_000 / u64::from(request.sample_rate)
}

/// Read the fixed 16-byte header, distinguishing a clean EOF from a partial one.
fn read_header<R: Read>(reader: &mut R) -> Result<Option<[u8; HEADER_BYTES]>, ProtocolError> {
    let mut header = [0u8; HEADER_BYTES];
    let mut filled = 0;
    while filled < HEADER_BYTES {
        match reader.read(&mut header[filled..]) {
            // Zero bytes before the first header byte is a clean end of input.
            Ok(0) if filled == 0 => return Ok(None),
            Ok(0) => return Err(ProtocolError::UnexpectedEof),
            Ok(read) => filled += read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(ProtocolError::Io(error)),
        }
    }
    Ok(Some(header))
}

/// Fill `buffer` exactly, reporting any early end as a truncated frame.
fn read_exact_or_eof<R: Read>(reader: &mut R, buffer: &mut [u8]) -> Result<(), ProtocolError> {
    let mut filled = 0;
    while filled < buffer.len() {
        match reader.read(&mut buffer[filled..]) {
            Ok(0) => return Err(ProtocolError::UnexpectedEof),
            Ok(read) => filled += read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(ProtocolError::Io(error)),
        }
    }
    Ok(())
}

fn json_to_io(error: serde_json::Error) -> io::Error {
    match error.io_error_kind() {
        Some(kind) => io::Error::new(kind, error),
        None => io::Error::new(io::ErrorKind::InvalidData, error),
    }
}
