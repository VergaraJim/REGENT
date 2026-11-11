//! Locating, spawning and supervising the bundled `regent-stt` helper.
//!
//! The helper runs as a child process: requests go to its stdin, JSON response
//! lines come back on its stdout, and its stderr is appended to a log file.
//! See `docs/decisions/0005-stt-helper.md`.

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use crate::stt::wire::{parse_response, Response};

/// Shown when the helper executable is not where the install left it.
pub const HELPER_MISSING: &str =
    "The speech helper is missing from this install — reinstall Regent.";

/// Suffix of the helper executable for this platform.
#[cfg(windows)]
pub const EXE_SUFFIX: &str = ".exe";
/// Suffix of the helper executable for this platform.
#[cfg(not(windows))]
pub const EXE_SUFFIX: &str = "";

/// Windows `CREATE_NO_WINDOW`: never flash a console window for the child process.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Everything that can go wrong while starting the helper.
#[derive(Debug, PartialEq, Eq)]
pub enum SpawnError {
    /// Spawning the process failed; carries the io error kind.
    Io(std::io::ErrorKind),
    /// The helper reported `fatal` (bad arguments or a model that failed to load).
    Fatal(String),
    /// The helper did not print `ready` before the timeout.
    Timeout,
    /// The helper's stdout closed before `ready` arrived.
    Exited,
}

/// Resolve the helper next to the app executable, honouring an explicit override.
pub fn resolve_helper_in(
    exe_dir: &Path,
    override_path: Option<PathBuf>,
) -> Result<PathBuf, String> {
    let path = match override_path {
        Some(path) => path,
        None => exe_dir.join(format!("regent-stt{EXE_SUFFIX}")),
    };

    if path.is_file() {
        Ok(path)
    } else {
        Err(HELPER_MISSING.to_string())
    }
}

/// Find the helper next to the running app.
///
/// Debug builds also honour `REGENT_STT_PATH` so development can point at a
/// freshly built helper without reinstalling; release builds ignore it.
pub fn resolve_helper() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|_| HELPER_MISSING.to_string())?;
    let exe_dir = exe.parent().ok_or_else(|| HELPER_MISSING.to_string())?;

    #[cfg(debug_assertions)]
    let override_path = std::env::var_os("REGENT_STT_PATH").map(PathBuf::from);
    #[cfg(not(debug_assertions))]
    let override_path: Option<PathBuf> = None;

    resolve_helper_in(exe_dir, override_path)
}

/// A running helper: its stdin pipe, the parsed responses and the child itself.
///
/// Dropping it closes stdin (the helper exits on EOF), waits briefly, then
/// kills the child if it is still running.
pub struct HelperProcess {
    stdin: Box<dyn Write + Send>,
    responses: Receiver<Response>,
    child: Option<Child>,
}

impl std::fmt::Debug for HelperProcess {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HelperProcess")
            .field("running", &self.child.is_some())
            .finish_non_exhaustive()
    }
}

impl HelperProcess {
    /// Build one from already-opened parts. Exists so tests can drive the client
    /// without a real process.
    pub fn from_parts(
        stdin: Box<dyn Write + Send>,
        responses: Receiver<Response>,
        child: Option<Child>,
    ) -> Self {
        Self {
            stdin,
            responses,
            child,
        }
    }

    /// Spawn `command`, wire up its pipes, and wait for its `ready` line.
    ///
    /// stderr goes to `log_file` (append mode, parent directories created);
    /// when the log cannot be opened the helper runs with stderr discarded.
    pub fn spawn(
        mut command: Command,
        log_file: &Path,
        ready_timeout: Duration,
    ) -> Result<Self, SpawnError> {
        command.stdin(Stdio::piped()).stdout(Stdio::piped());

        if let Some(parent) = log_file.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let stderr = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_file)
            .map(Stdio::from)
            .unwrap_or_else(|_| Stdio::null());
        command.stderr(stderr);

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = command
            .spawn()
            .map_err(|error| SpawnError::Io(error.kind()))?;
        let stdin = child
            .stdin
            .take()
            .ok_or(SpawnError::Io(std::io::ErrorKind::BrokenPipe))?;
        let stdout = child
            .stdout
            .take()
            .ok_or(SpawnError::Io(std::io::ErrorKind::BrokenPipe))?;

        let (sender, responses) = mpsc::channel();
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else {
                    break;
                };
                if let Some(response) = parse_response(&line) {
                    if sender.send(response).is_err() {
                        break;
                    }
                }
            }
        });

        let mut process = Self {
            stdin: Box::new(stdin),
            responses,
            child: Some(child),
        };

        let deadline = Instant::now() + ready_timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                process.kill();
                return Err(SpawnError::Timeout);
            }

            match process.responses.recv_timeout(remaining) {
                Ok(Response::Ready { .. }) => return Ok(process),
                Ok(Response::Fatal { message }) => return Err(SpawnError::Fatal(message)),
                // Anything else before `ready` is not part of the handshake.
                Ok(_) => continue,
                Err(RecvTimeoutError::Timeout) => {
                    process.kill();
                    return Err(SpawnError::Timeout);
                }
                Err(RecvTimeoutError::Disconnected) => return Err(SpawnError::Exited),
            }
        }
    }

    /// Write one encoded request frame and flush it to the helper.
    pub fn write_frame(&mut self, frame: &[u8]) -> std::io::Result<()> {
        self.stdin.write_all(frame)?;
        self.stdin.flush()
    }

    /// Wait at most `timeout` for the next response line.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Response, RecvTimeoutError> {
        self.responses.recv_timeout(timeout)
    }

    /// Kill and reap the child, if it is still there.
    fn kill(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for HelperProcess {
    fn drop(&mut self) {
        if self.child.is_none() {
            return;
        }

        // Closing stdin gives the helper a clean EOF, which makes it exit.
        self.stdin = Box::new(std::io::sink());

        if let Some(mut child) = self.child.take() {
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => return,
                    Ok(None) => {
                        if Instant::now() >= deadline {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    Err(_) => break,
                }
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn shell(script: &str) -> Command {
        let mut command = Command::new("sh");
        command.arg("-c").arg(script);
        command
    }

    fn ready_line() -> &'static str {
        r#"printf '%s\n' '{"type":"ready","version":"test"}'"#
    }

    #[test]
    fn missing_helper_is_reported() {
        let dir = tempfile::tempdir().expect("temp dir");

        let error = resolve_helper_in(dir.path(), None).expect_err("helper should be missing");

        assert_eq!(error, HELPER_MISSING);
    }

    #[test]
    fn helper_next_to_the_executable_is_found() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join(format!("regent-stt{EXE_SUFFIX}"));
        fs::write(&path, b"").expect("write helper");

        let resolved = resolve_helper_in(dir.path(), None).expect("helper should resolve");

        assert_eq!(resolved, path);
    }

    #[test]
    fn explicit_override_is_used() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("custom-helper");
        fs::write(&path, b"").expect("write helper");

        let resolved =
            resolve_helper_in(dir.path(), Some(path.clone())).expect("override should resolve");

        assert_eq!(resolved, path);
    }

    #[test]
    fn missing_override_still_reports_the_install_message() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("absent-helper");

        let error = resolve_helper_in(dir.path(), Some(path)).expect_err("should be missing");

        assert_eq!(error, HELPER_MISSING);
    }

    #[cfg(unix)]
    #[test]
    fn spawn_waits_for_ready_and_keeps_running() {
        let dir = tempfile::tempdir().expect("temp dir");
        let log = dir.path().join("helper.log");
        let script = format!("{}; cat >/dev/null", ready_line());

        let process = HelperProcess::spawn(shell(&script), &log, Duration::from_secs(5))
            .expect("helper should start");

        drop(process);
    }

    #[cfg(unix)]
    #[test]
    fn fatal_line_is_reported() {
        let dir = tempfile::tempdir().expect("temp dir");
        let log = dir.path().join("helper.log");
        let script = r#"printf '%s\n' '{"type":"fatal","message":"boom"}'"#;

        let error = HelperProcess::spawn(shell(script), &log, Duration::from_secs(5))
            .expect_err("fatal should fail the spawn");

        assert_eq!(error, SpawnError::Fatal("boom".to_string()));
    }

    #[cfg(unix)]
    #[test]
    fn no_ready_within_the_timeout_is_reported_quickly() {
        let dir = tempfile::tempdir().expect("temp dir");
        let log = dir.path().join("helper.log");
        let start = Instant::now();

        let error = HelperProcess::spawn(shell("sleep 5"), &log, Duration::from_millis(200))
            .expect_err("timeout should fail the spawn");

        assert_eq!(error, SpawnError::Timeout);
        assert!(start.elapsed() < Duration::from_secs(2));
    }

    #[cfg(unix)]
    #[test]
    fn stderr_is_written_to_the_log_file() {
        let dir = tempfile::tempdir().expect("temp dir");
        let log = dir.path().join("nested").join("helper.log");
        let script = format!(
            "{}; printf '%s\\n' 'hello-stderr' >&2; cat >/dev/null",
            ready_line()
        );

        let process = HelperProcess::spawn(shell(&script), &log, Duration::from_secs(5))
            .expect("helper should start");
        drop(process);

        let contents = fs::read_to_string(&log).expect("log file should exist");
        assert!(contents.contains("hello-stderr"), "log was: {contents}");
    }

    #[cfg(unix)]
    #[test]
    fn dropping_the_process_ends_the_child() {
        let dir = tempfile::tempdir().expect("temp dir");
        let log = dir.path().join("helper.log");
        let marker = dir.path().join("exited");
        let script = format!(
            "{}; cat >/dev/null; printf 'done' > '{}'",
            ready_line(),
            marker.display()
        );

        let process = HelperProcess::spawn(shell(&script), &log, Duration::from_secs(5))
            .expect("helper should start");
        drop(process);

        assert!(
            marker.is_file(),
            "the child should have seen EOF and exited"
        );
    }
}
