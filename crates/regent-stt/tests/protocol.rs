//! Protocol and request-loop tests. They need no model, native library or
//! audio hardware; run with `--no-default-features`.

use std::io::Cursor;

use regent_stt::protocol::{
    parse_response, read_request, write_request, write_response, ProtocolError, Request, Response,
};
use regent_stt::service::{run_loop, Transcriber};

struct FakeTranscriber {
    calls: usize,
}

impl Transcriber for FakeTranscriber {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String, String> {
        self.calls += 1;
        Ok(format!(
            "call {} with {} samples",
            self.calls,
            samples.len()
        ))
    }
}

fn fake() -> FakeTranscriber {
    FakeTranscriber { calls: 0 }
}

fn frames(requests: &[Request]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for request in requests {
        write_request(&mut bytes, request).unwrap();
    }
    bytes
}

fn responses(output: &[u8]) -> Vec<Response> {
    let text = std::str::from_utf8(output).expect("stdout is UTF-8 JSON lines");
    text.lines()
        .map(|line| parse_response(line).unwrap())
        .collect()
}

fn request(id: u64, sample_rate: u32, sample_count: usize) -> Request {
    Request {
        id,
        sample_rate,
        samples: vec![0.25; sample_count],
    }
}

#[test]
fn frame_round_trip() {
    let original = Request {
        id: 42,
        sample_rate: 16_000,
        samples: vec![0.0, 0.5, -0.5, 1.0e-6],
    };

    let mut encoded = Vec::new();
    write_request(&mut encoded, &original).unwrap();
    assert_eq!(encoded.len(), 16 + original.samples.len() * 4);
    // Little-endian header: id, sample rate, sample count.
    assert_eq!(&encoded[0..8], &42u64.to_le_bytes());
    assert_eq!(&encoded[8..12], &16_000u32.to_le_bytes());
    assert_eq!(&encoded[12..16], &4u32.to_le_bytes());

    let mut input = Cursor::new(encoded);
    let decoded = read_request(&mut input).unwrap();
    assert_eq!(decoded, Some(original));
    assert_eq!(read_request(&mut input).unwrap(), None);
}

#[test]
fn clean_eof_returns_none() {
    let mut input = Cursor::new(Vec::new());
    assert_eq!(read_request(&mut input).unwrap(), None);
}

#[test]
fn eof_in_the_middle_of_a_frame() {
    let mut encoded = frames(&[request(1, 16_000, 8)]);
    encoded.truncate(encoded.len() - 3);

    let mut input = Cursor::new(encoded.clone());
    assert!(matches!(
        read_request(&mut input),
        Err(ProtocolError::UnexpectedEof)
    ));

    // The loop logs to stderr and exits cleanly rather than answering.
    let mut output = Vec::new();
    run_loop(Cursor::new(encoded), &mut output, &mut fake()).unwrap();
    assert!(output.is_empty(), "no response for a truncated frame");
}

#[test]
fn wrong_sample_rate_gets_an_error_and_the_next_request_is_answered() {
    let input = frames(&[request(7, 8_000, 16), request(8, 16_000, 16_000)]);
    let mut output = Vec::new();
    let mut transcriber = fake();

    run_loop(Cursor::new(input), &mut output, &mut transcriber).unwrap();

    let responses = responses(&output);
    assert_eq!(responses.len(), 2);
    match &responses[0] {
        Response::Error { id, message } => {
            assert_eq!(*id, 7);
            assert!(message.contains("8000"), "message was {message:?}");
        }
        other => panic!("expected error, got {other:?}"),
    }
    assert_eq!(
        responses[1],
        Response::Transcript {
            id: 8,
            text: "call 1 with 16000 samples".to_string(),
            duration_ms: 1_000,
        }
    );
    assert_eq!(transcriber.calls, 1);
}

#[test]
fn oversized_frame_gets_an_error_without_a_huge_allocation() {
    // Header claims 4_000_000_000 samples (~16 GB) but only 16 bytes follow.
    let mut input = Vec::new();
    input.extend_from_slice(&9u64.to_le_bytes());
    input.extend_from_slice(&16_000u32.to_le_bytes());
    input.extend_from_slice(&4_000_000_000u32.to_le_bytes());
    input.extend_from_slice(&[0u8; 16]);

    let mut output = Vec::new();
    let mut transcriber = fake();

    // Short stream: the error is written first, then skipping hits EOF, which
    // is a clean exit (exit code 0).
    run_loop(Cursor::new(input), &mut output, &mut transcriber).unwrap();

    let responses = responses(&output);
    assert_eq!(responses.len(), 1);
    match &responses[0] {
        Response::Error { id, message } => {
            assert_eq!(*id, 9);
            assert!(message.contains("4000000000"), "message was {message:?}");
        }
        other => panic!("expected error, got {other:?}"),
    }
    assert_eq!(transcriber.calls, 0);
}

#[test]
fn responses_serialise_exactly() {
    let cases = [
        (
            Response::Ready {
                version: "0.1.0".to_string(),
            },
            "{\"type\":\"ready\",\"version\":\"0.1.0\"}",
        ),
        (
            Response::Transcript {
                id: 5,
                text: "hello".to_string(),
                duration_ms: 1_250,
            },
            "{\"type\":\"transcript\",\"id\":5,\"text\":\"hello\",\"duration_ms\":1250}",
        ),
        (
            Response::Error {
                id: 6,
                message: "bad".to_string(),
            },
            "{\"type\":\"error\",\"id\":6,\"message\":\"bad\"}",
        ),
        (
            Response::Fatal {
                message: "boom".to_string(),
            },
            "{\"type\":\"fatal\",\"message\":\"boom\"}",
        ),
    ];

    for (response, expected) in cases {
        let mut line = Vec::new();
        write_response(&mut line, &response).unwrap();
        assert_eq!(std::str::from_utf8(&line).unwrap(), format!("{expected}\n"));
        assert_eq!(parse_response(expected).unwrap(), response);
    }
}

#[test]
fn responses_come_back_in_request_order() {
    let input = frames(&[
        request(30, 16_000, 1_600),
        request(10, 16_000, 3_200),
        request(20, 16_000, 4_800),
    ]);

    let mut output = Vec::new();
    run_loop(Cursor::new(input), &mut output, &mut fake()).unwrap();

    let ids: Vec<u64> = responses(&output)
        .into_iter()
        .map(|response| match response {
            Response::Transcript { id, .. } => id,
            other => panic!("expected transcript, got {other:?}"),
        })
        .collect();
    assert_eq!(ids, vec![30, 10, 20]);
}
