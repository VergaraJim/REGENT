//! Locating and driving the bundled Piper executable.

use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Suffix of the Piper executable for this platform.
#[cfg(windows)]
pub const EXE_SUFFIX: &str = ".exe";
/// Suffix of the Piper executable for this platform.
#[cfg(not(windows))]
pub const EXE_SUFFIX: &str = "";

/// Windows `CREATE_NO_WINDOW`: never flash a console window for the child process.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Errors raised while locating or running the voice engine.
///
/// The first two variants carry user-facing wording; they are shown as-is.
#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    /// The Piper executable or its espeak-ng data folder is absent from the resource dir.
    #[error("The voice engine is missing from this install — reinstall Regent.")]
    EngineMissing,

    /// The picked voice has no `.onnx.json` next to it.
    #[error("This voice needs its .onnx.json file in the same folder.")]
    VoiceConfigMissing,

    /// Talking to the child process failed (spawn, pipe or wait).
    #[error("could not run the voice engine: {0}")]
    Io(#[from] std::io::Error),

    /// Piper ran but reported failure.
    #[error("the voice engine failed ({status}): {stderr}")]
    EngineFailed {
        /// Exit status, formatted.
        status: String,
        /// Whatever Piper printed on stderr.
        stderr: String,
    },
}

/// Where the Piper executable and its espeak-ng data live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiperPaths {
    /// `<resource_dir>/piper/piper{EXE_SUFFIX}`.
    pub exe: PathBuf,
    /// `<resource_dir>/piper/espeak-ng-data`.
    pub espeak_data: PathBuf,
}

/// The folder holding the engine: `<resource_dir>/piper`.
///
/// In debug builds only, `REGENT_PIPER_DIR` overrides it, so development can point at
/// a fetched folder somewhere else without rebuilding.
fn piper_dir(resource_dir: &Path) -> PathBuf {
    #[cfg(debug_assertions)]
    if let Some(dir) = std::env::var_os("REGENT_PIPER_DIR") {
        return PathBuf::from(dir);
    }

    resource_dir.join("piper")
}

/// Finds the bundled Piper executable and espeak-ng data under `resource_dir`.
///
pub fn resolve_piper(resource_dir: &Path) -> Result<PiperPaths, TtsError> {
    let dir = piper_dir(resource_dir);
    let exe = dir.join(format!("piper{EXE_SUFFIX}"));
    let espeak_data = dir.join("espeak-ng-data");

    if !exe.is_file() || !espeak_data.is_dir() {
        return Err(TtsError::EngineMissing);
    }

    Ok(PiperPaths { exe, espeak_data })
}

/// The voice's configuration file: the voice path with `.json` appended.
///
pub fn voice_config_path(voice: &Path) -> Result<PathBuf, TtsError> {
    let mut raw = voice.as_os_str().to_os_string();
    raw.push(".json");
    let path = PathBuf::from(raw);

    if !path.is_file() {
        return Err(TtsError::VoiceConfigMissing);
    }

    Ok(path)
}

/// `dir` followed by the entries already in the loader's path variable.
///
/// Building the value here keeps the platform tricks out of the test-only assertions.
fn library_path(dir: &Path, existing: Option<OsString>) -> OsString {
    let mut entries = vec![dir.to_path_buf()];
    if let Some(existing) = existing {
        if !existing.is_empty() {
            entries.extend(std::env::split_paths(&existing));
        }
    }
    std::env::join_paths(entries).unwrap_or_else(|_| dir.as_os_str().to_os_string())
}

/// Builds the Piper command for one voice.
///
/// The child runs with the Piper folder as its working directory, so its own shared
/// libraries and data are found next to the executable in dev and in installed builds.
/// The loader is also pointed at that folder: Piper is not built with an `$ORIGIN`
/// rpath, so the libraries next to the executable are only found through this path.
pub fn piper_command(paths: &PiperPaths, voice: &Path) -> Command {
    let mut command = Command::new(&paths.exe);
    command
        .arg("--model")
        .arg(voice)
        .arg("--espeak_data")
        .arg(&paths.espeak_data)
        .arg("--output_raw");

    if let Some(dir) = paths.exe.parent() {
        command.current_dir(dir);

        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            #[cfg(target_os = "linux")]
            let variable = "LD_LIBRARY_PATH";
            #[cfg(target_os = "macos")]
            let variable = "DYLD_LIBRARY_PATH";
            command.env(variable, library_path(dir, std::env::var_os(variable)));
        }
    }

    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    command
}

/// Speaks `text` once and returns the raw mono samples Piper produced.
///
pub fn synthesize_once(paths: &PiperPaths, voice: &Path, text: &str) -> Result<Vec<i16>, TtsError> {
    voice_config_path(voice)?;

    let mut child = piper_command(paths, voice).spawn()?;

    {
        let mut stdin = child.stdin.take().ok_or(TtsError::EngineMissing)?;
        stdin.write_all(text.as_bytes())?;
        stdin.write_all(b"\n")?;
        // Dropping stdin closes the pipe, which is what tells Piper the text is done.
    }

    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(TtsError::EngineFailed {
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }

    // Raw output is mono little-endian i16; a trailing odd byte is dropped.
    let (pairs, _rest) = output.stdout.as_chunks::<2>();
    Ok(pairs.iter().map(|pair| i16::from_le_bytes(*pair)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A resource dir with no `piper/` folder at all.
    #[test]
    fn missing_engine_is_reported() {
        let resource_dir = tempfile::tempdir().expect("temp dir");

        let error = resolve_piper(resource_dir.path()).expect_err("engine should be missing");

        assert!(matches!(error, TtsError::EngineMissing));
    }

    /// A voice without its `.onnx.json` sibling.
    #[test]
    fn missing_voice_config_is_reported() {
        let dir = tempfile::tempdir().expect("temp dir");
        let voice = dir.path().join("en_US-lessac-low.onnx");
        fs::write(&voice, b"not a real model").expect("write voice");

        let error = voice_config_path(&voice).expect_err("config should be missing");

        assert!(matches!(error, TtsError::VoiceConfigMissing));
    }

    /// The layout the fetch script installs.
    #[test]
    fn correct_layout_resolves() {
        let resource_dir = tempfile::tempdir().expect("temp dir");
        let piper_dir = resource_dir.path().join("piper");
        fs::create_dir_all(piper_dir.join("espeak-ng-data")).expect("espeak data dir");
        fs::write(piper_dir.join(format!("piper{EXE_SUFFIX}")), b"").expect("write exe");

        let paths = resolve_piper(resource_dir.path()).expect("layout should resolve");

        assert_eq!(paths.exe, piper_dir.join(format!("piper{EXE_SUFFIX}")));
        assert_eq!(paths.espeak_data, piper_dir.join("espeak-ng-data"));
    }

    /// With nothing inherited, the loader path is just the Piper folder.
    #[test]
    fn library_path_without_existing_value_is_the_dir() {
        let dir = Path::new("/opt/regent/piper");

        let value = library_path(dir, None);

        let entries: Vec<PathBuf> = std::env::split_paths(&value).collect();
        assert_eq!(entries, vec![dir.to_path_buf()]);
    }

    /// An inherited value keeps all of its entries, after the Piper folder.
    #[test]
    fn library_path_puts_the_dir_first() {
        let dir = Path::new("/opt/regent/piper");
        let existing =
            std::env::join_paths([Path::new("/usr/lib"), Path::new("/lib")]).expect("join paths");

        let value = library_path(dir, Some(existing));

        let entries: Vec<PathBuf> = std::env::split_paths(&value).collect();
        assert_eq!(
            entries,
            vec![
                dir.to_path_buf(),
                PathBuf::from("/usr/lib"),
                PathBuf::from("/lib")
            ]
        );
    }

    /// The spawned child must be told where its libraries are on Linux.
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_command_sets_library_path_to_the_piper_folder() {
        let temp = tempfile::tempdir().expect("temp dir");
        let folder = temp.path().join("piper");
        let paths = PiperPaths {
            exe: folder.join(format!("piper{EXE_SUFFIX}")),
            espeak_data: folder.join("espeak-ng-data"),
        };

        let command = piper_command(&paths, &temp.path().join("voice.onnx"));
        let value = command
            .get_envs()
            .find(|(name, _)| *name == "LD_LIBRARY_PATH")
            .and_then(|(_, value)| value)
            .expect("LD_LIBRARY_PATH should be set");

        let first = std::env::split_paths(value)
            .next()
            .expect("at least one entry");
        assert_eq!(first, folder);
    }
}
