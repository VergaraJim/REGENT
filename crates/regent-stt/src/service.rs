//! The request loop: read frames from stdin, answer on stdout, never block on
//! anything but the pipes. This module has no whisper dependency.

use std::io::{self, Read, Write};

use crate::protocol::{
    self, duration_ms, ProtocolError, Request, Response, MAX_SAMPLES, SAMPLE_RATE,
};

/// Anything that can turn mono 16 kHz samples into text.
///
/// The whisper backend implements this behind the `whisper` feature; tests use
/// a fake so the protocol is exercised without a model or a native library.
pub trait Transcriber {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String, String>;
}

/// Run until the input ends, answering every well-formed request in order.
///
/// Rules implemented here:
/// - clean EOF between frames exits `Ok(())`;
/// - EOF in the middle of a frame is logged to stderr, then exits `Ok(())`;
/// - a non-16 kHz frame gets an error response and its payload is consumed;
/// - an oversized frame gets an error response and its payload is skipped in
///   fixed-size chunks without ever being allocated.
pub fn run_loop<R: Read, W: Write, T: Transcriber>(
    mut input: R,
    mut output: W,
    transcriber: &mut T,
) -> io::Result<()> {
    loop {
        match protocol::read_request(&mut input) {
            Ok(None) => return Ok(()),
            Ok(Some(request)) => {
                if request.sample_rate != SAMPLE_RATE {
                    let message = format!(
                        "unsupported sample rate {}; expected {}",
                        request.sample_rate, SAMPLE_RATE
                    );
                    protocol::write_response(
                        &mut output,
                        &Response::Error {
                            id: request.id,
                            message,
                        },
                    )?;
                    continue;
                }
                answer(&mut output, &request, transcriber)?;
            }
            Err(ProtocolError::TooManySamples { id, sample_count }) => {
                let message = format!(
                    "request declares {sample_count} samples; the maximum is {MAX_SAMPLES} (35 s at {SAMPLE_RATE} Hz)"
                );
                protocol::write_response(&mut output, &Response::Error { id, message })?;

                match skip_payload(&mut input, sample_count) {
                    Ok(()) => {}
                    Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                        eprintln!(
                            "regent-stt: input ended in the middle of an oversized frame; exiting"
                        );
                        return Ok(());
                    }
                    Err(error) => return Err(error),
                }
            }
            Err(ProtocolError::UnexpectedEof) => {
                eprintln!("regent-stt: input ended in the middle of a frame; exiting");
                return Ok(());
            }
            Err(error) => return Err(error.into_io_error()),
        }
    }
}

fn answer<W: Write, T: Transcriber>(
    output: &mut W,
    request: &Request,
    transcriber: &mut T,
) -> io::Result<()> {
    let duration_ms = duration_ms(request);
    let response = match transcriber.transcribe(&request.samples) {
        Ok(text) => Response::Transcript {
            id: request.id,
            text,
            duration_ms,
        },
        Err(message) => Response::Error {
            id: request.id,
            message,
        },
    };
    protocol::write_response(output, &response)
}

/// Consume `sample_count` f32 values using one fixed buffer. The coefficient
/// arithmetic is `u64`, so a count near `u32::MAX` cannot overflow.
fn skip_payload<R: Read>(input: &mut R, sample_count: u32) -> io::Result<()> {
    const CHUNK_BYTES: usize = 64 * 1024;

    let mut remaining = u64::from(sample_count) * std::mem::size_of::<f32>() as u64;
    let mut chunk = [0u8; CHUNK_BYTES];
    while remaining > 0 {
        let want = remaining.min(CHUNK_BYTES as u64) as usize;
        match input.read(&mut chunk[..want]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "input ended while skipping an oversized frame",
                ))
            }
            Ok(read) => remaining -= read as u64,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}
