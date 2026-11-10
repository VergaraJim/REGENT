//! `regent-stt` — speech-to-text helper process.
//!
//! Launch: `regent-stt --model <path> [--language <code|auto>] [--threads <n>]`
//! Reads binary request frames from stdin, writes one JSON object per line to
//! stdout. Logs go to stderr. Exits on stdin EOF.

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use regent_stt::protocol::{write_response, Response};
#[cfg(feature = "whisper")]
use regent_stt::service::run_loop;

#[cfg_attr(not(feature = "whisper"), allow(dead_code))]
struct Options {
    model: PathBuf,
    language: String,
    threads: u32,
}

fn main() -> ExitCode {
    let options = match parse_args(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(message) => {
            let stdout = io::stdout();
            let mut output = stdout.lock();
            fatal(&mut output, &message);
        }
    };

    #[cfg(feature = "whisper")]
    {
        run_with_whisper(options)
    }

    #[cfg(not(feature = "whisper"))]
    {
        let _ = options;
        let stdout = io::stdout();
        let mut output = stdout.lock();
        fatal(&mut output, "built without speech recognition");
    }
}

#[cfg(feature = "whisper")]
fn run_with_whisper(options: Options) -> ExitCode {
    let stdout = io::stdout();
    let mut output = stdout.lock();

    let mut transcriber = match regent_stt::whisper::WhisperTranscriber::load(
        &options.model,
        &options.language,
        options.threads,
    ) {
        Ok(transcriber) => transcriber,
        Err(message) => fatal(&mut output, &message),
    };

    if let Err(error) = write_response(
        &mut output,
        &Response::Ready {
            version: env!("CARGO_PKG_VERSION").to_string(),
        },
    ) {
        eprintln!("regent-stt: failed to write ready response: {error}");
        return ExitCode::FAILURE;
    }

    let stdin = io::stdin();
    let mut input = stdin.lock();
    match run_loop(&mut input, &mut output, &mut transcriber) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("regent-stt: request loop failed: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Send a `fatal` line (best effort) and exit with code 1.
fn fatal(output: &mut impl Write, message: &str) -> ! {
    let _ = write_response(
        output,
        &Response::Fatal {
            message: message.to_string(),
        },
    );
    std::process::exit(1);
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut model = None;
    let mut language = "auto".to_string();
    let mut threads = default_threads();

    let mut args = args.peekable();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--model" => {
                model = Some(PathBuf::from(next_value(&mut args, "--model")?));
            }
            "--language" => {
                language = next_value(&mut args, "--language")?;
            }
            "--threads" => {
                let value = next_value(&mut args, "--threads")?;
                threads = value
                    .parse::<u32>()
                    .ok()
                    .filter(|threads| *threads > 0)
                    .ok_or_else(|| format!("invalid --threads value: {value}"))?;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let model = model.ok_or_else(|| "missing required --model <path>".to_string())?;
    Ok(Options {
        model,
        language,
        threads,
    })
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn default_threads() -> u32 {
    let available = std::thread::available_parallelism()
        .map(|count| count.get() as u32)
        .unwrap_or(2);
    available.saturating_sub(1).max(1)
}
