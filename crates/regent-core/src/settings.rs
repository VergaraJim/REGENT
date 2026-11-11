//! The settings model, its safe TOML persistence, and the store the rest of the
//! app talks to.
//!
//! Two rules keep user data safe:
//! - loading never writes to disk;
//! - a settings file that exists but cannot be read is never overwritten, so a
//!   typo can't silently destroy the user's configuration.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Everything Regent remembers between runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// The model file for each slot.
    pub models: ModelSlots,
    /// When conversation compaction kicks in and how far it compresses.
    pub compaction: CompactionSettings,
    /// Whether the text model's prefix cache is kept between runs.
    pub persist_prefix_cache: bool,
    /// Extra text appended to the built-in system prompt.
    pub system_prompt_addition: String,
    /// Manual overrides for the memory budget detection.
    pub memory: MemoryOverrides,
    /// Voice and microphone behaviour.
    pub voice: VoiceSettings,
    /// Window look.
    pub appearance: Appearance,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            models: ModelSlots::default(),
            compaction: CompactionSettings::default(),
            persist_prefix_cache: true,
            system_prompt_addition: String::new(),
            memory: MemoryOverrides::default(),
            voice: VoiceSettings::default(),
            appearance: Appearance::default(),
        }
    }
}

impl Settings {
    /// Returns settings that are safe to act on.
    ///
    /// The compaction pair is replaced by its validated values, and memory
    /// overrides that are not finite positive numbers are dropped.
    pub fn normalized(mut self) -> Settings {
        let (trigger, target) = self.compaction.validated();
        self.compaction.trigger_percent = trigger;
        self.compaction.target_percent = target;
        self.memory.ram_gb = sane_override(self.memory.ram_gb);
        self.memory.vram_gb = sane_override(self.memory.vram_gb);
        self
    }
}

fn sane_override(value: Option<f64>) -> Option<f64> {
    match value {
        Some(v) if v.is_finite() && v > 0.0 => Some(v),
        _ => None,
    }
}

/// Where each model lives and how it is loaded.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelSlots {
    /// The local text model.
    pub text: TextSlot,
    /// The optional vision model.
    pub vision: VisionSlot,
    /// The embedding model used for memory.
    pub embedding: EmbeddingSlot,
    /// The speech-to-text model.
    pub stt: SttSlot,
    /// The text-to-speech voice.
    pub tts: TtsSlot,
}

/// The text model slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TextSlot {
    /// Path to the GGUF file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    /// Layers offloaded to the GPU.
    pub gpu_layers: u32,
    /// Context window in tokens.
    pub context_size: u32,
}

impl Default for TextSlot {
    fn default() -> Self {
        Self {
            path: None,
            gpu_layers: 0,
            context_size: 4096,
        }
    }
}

/// The vision slot: a model plus its multimodal projector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VisionSlot {
    /// Path to the GGUF file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<PathBuf>,
    /// Path to the `mmproj` projector file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projector_path: Option<PathBuf>,
    /// Layers offloaded to the GPU.
    pub gpu_layers: u32,
    /// Longest image edge, in pixels, before downscaling.
    pub max_image_edge: u32,
    /// How long the vision model stays loaded, in seconds.
    pub keep_alive_secs: u32,
}

impl Default for VisionSlot {
    fn default() -> Self {
        Self {
            model_path: None,
            projector_path: None,
            gpu_layers: 0,
            max_image_edge: 1024,
            keep_alive_secs: 120,
        }
    }
}

/// The embedding model slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingSlot {
    /// Path to the GGUF file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    /// Layers offloaded to the GPU.
    pub gpu_layers: u32,
    /// Context window in tokens.
    pub context_size: u32,
    /// Instruction prefix used when embedding stored documents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_prefix: Option<String>,
    /// Instruction prefix used when embedding a search query.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_prefix: Option<String>,
}

impl Default for EmbeddingSlot {
    fn default() -> Self {
        Self {
            path: None,
            gpu_layers: 0,
            context_size: 2048,
            document_prefix: None,
            query_prefix: None,
        }
    }
}

/// The speech-to-text model slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SttSlot {
    /// Path to the model file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    /// Spoken language, or `"auto"` for detection.
    pub language: String,
}

impl Default for SttSlot {
    fn default() -> Self {
        Self {
            path: None,
            language: "auto".to_string(),
        }
    }
}

/// The text-to-speech voice slot.
///
/// Piper reads its own config from `voice_path + ".json"`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TtsSlot {
    /// Path to the Piper voice model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_path: Option<PathBuf>,
}

/// When and how far history is compacted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CompactionSettings {
    /// Fraction of the context window that triggers compaction.
    pub trigger_percent: u8,
    /// Fraction of the context window kept after compaction.
    pub target_percent: u8,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            trigger_percent: 80,
            target_percent: 45,
        }
    }
}

impl CompactionSettings {
    /// Returns the usable `(trigger, target)` pair.
    ///
    /// Anything outside `1 <= target < trigger < 100` falls back to `(80, 45)`.
    pub fn validated(&self) -> (u8, u8) {
        let (trigger, target) = (self.trigger_percent, self.target_percent);
        if target >= 1 && target < trigger && trigger < 100 {
            (trigger, target)
        } else {
            (80, 45)
        }
    }
}

/// Manual overrides for the memory budget detection.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryOverrides {
    /// Overrides the detected system RAM, in GB.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ram_gb: Option<f64>,
    /// Overrides the detected VRAM, in GB.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_gb: Option<f64>,
}

/// Voice and microphone behaviour.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VoiceSettings {
    /// Read replies to voice messages aloud.
    pub speak_replies: bool,
    /// Mute the microphone while Regent speaks.
    pub pause_mic_while_speaking: bool,
}

impl Default for VoiceSettings {
    fn default() -> Self {
        Self {
            speak_replies: true,
            pause_mic_while_speaking: true,
        }
    }
}

/// Window look.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Appearance {
    /// Follow the OS, or force light or dark.
    pub theme: Theme,
    /// Keep the window above other windows.
    pub always_on_top: bool,
}

impl Default for Appearance {
    fn default() -> Self {
        Self {
            theme: Theme::System,
            always_on_top: true,
        }
    }
}

/// The colour theme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Theme {
    /// Follow the operating system.
    #[default]
    System,
    /// Always light.
    Light,
    /// Always dark.
    Dark,
}

/// The result of reading a settings file.
///
/// The settings are returned by value because the caller always needs them,
/// either to keep or to drop; boxing them would only add indirection.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum LoadOutcome {
    /// The file was read and parsed.
    Loaded(Settings),
    /// There is no file yet.
    Missing,
    /// A file exists but could not be read or parsed.
    Invalid {
        /// A human-readable description, including line information when the
        /// TOML parser provides it.
        error: String,
    },
}

/// Reads a settings file without ever writing to disk.
pub fn load(path: &Path) -> LoadOutcome {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return LoadOutcome::Missing,
        Err(err) => {
            return LoadOutcome::Invalid {
                error: format!("Couldn't read {}: {err}", path.display()),
            }
        }
    };

    match toml::from_str::<Settings>(&text) {
        Ok(settings) => LoadOutcome::Loaded(settings),
        Err(err) => LoadOutcome::Invalid {
            error: format!("{} isn't valid TOML: {err}", path.display()),
        },
    }
}

/// Writes settings atomically: a temp file in the same folder, then a rename.
pub fn save(path: &Path, settings: &Settings) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let text = toml::to_string_pretty(settings).map_err(io::Error::other)?;

    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);

    fs::write(&tmp, text)?;
    fs::rename(&tmp, path)
}

/// What `update` and `reset_to_defaults` can fail with.
#[derive(Debug, thiserror::Error)]
pub enum SaveError {
    /// The file that exists could not be read, so saving is refused.
    #[error("Your settings file couldn't be read, so changes aren't saved until you reset it.")]
    Blocked,
    /// The file could not be written.
    #[error("Couldn't save settings. Check that the settings folder is writable.")]
    Io(#[from] io::Error),
}

/// The app's single handle on settings.
///
/// It remembers a bad file instead of overwriting it: while the error is
/// remembered, `update` keeps the new values in memory but refuses to save.
#[derive(Debug, Clone)]
pub struct SettingsStore {
    path: PathBuf,
    settings: Settings,
    invalid_error: Option<String>,
}

impl SettingsStore {
    /// Opens the store at `path`, falling back to defaults when the file is
    /// missing or unreadable.
    pub fn open(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let (settings, invalid_error) = match load(&path) {
            LoadOutcome::Loaded(settings) => (settings.normalized(), None),
            LoadOutcome::Missing => (Settings::default(), None),
            LoadOutcome::Invalid { error } => (Settings::default(), Some(error)),
        };
        Self {
            path,
            settings,
            invalid_error,
        }
    }

    /// The current settings, always normalised.
    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// The error from a settings file that exists but could not be read.
    pub fn load_error(&self) -> Option<&str> {
        self.invalid_error.as_deref()
    }

    /// Whether saving is refused because of an unreadable file.
    pub fn saving_blocked(&self) -> bool {
        self.invalid_error.is_some()
    }

    /// Where the settings file lives.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Replaces the settings and saves them, unless saving is blocked.
    ///
    /// The new values are kept in memory even when saving is blocked.
    pub fn update(&mut self, new: Settings) -> Result<(), SaveError> {
        self.settings = new.normalized();
        if self.saving_blocked() {
            return Err(SaveError::Blocked);
        }
        save(&self.path, &self.settings)?;
        Ok(())
    }

    /// Restores defaults, backing up an unreadable file first.
    ///
    /// Returns the backup path when one was made.
    pub fn reset_to_defaults(&mut self) -> Result<Option<PathBuf>, SaveError> {
        let backup = if self.saving_blocked() && self.path.exists() {
            let seconds = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let file_name = self
                .path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "settings.toml".to_string());
            let backup = self
                .path
                .with_file_name(format!("{file_name}.bak-{seconds}"));
            fs::rename(&self.path, &backup)?;
            Some(backup)
        } else {
            None
        };

        self.settings = Settings::default();
        save(&self.path, &self.settings)?;
        self.invalid_error = None;
        Ok(backup)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, text: &str) {
        fs::write(path, text).expect("write fixture");
    }

    fn temp_path() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("settings.toml");
        (dir, path)
    }

    #[test]
    fn defaults_are_the_documented_ones() {
        let settings = Settings::default();

        assert!(settings.persist_prefix_cache);
        assert_eq!(settings.system_prompt_addition, "");
        assert_eq!(settings.memory.ram_gb, None);
        assert_eq!(settings.memory.vram_gb, None);

        assert_eq!(settings.models.text.path, None);
        assert_eq!(settings.models.text.gpu_layers, 0);
        assert_eq!(settings.models.text.context_size, 4096);

        assert_eq!(settings.models.vision.model_path, None);
        assert_eq!(settings.models.vision.projector_path, None);
        assert_eq!(settings.models.vision.gpu_layers, 0);
        assert_eq!(settings.models.vision.max_image_edge, 1024);
        assert_eq!(settings.models.vision.keep_alive_secs, 120);

        assert_eq!(settings.models.embedding.path, None);
        assert_eq!(settings.models.embedding.gpu_layers, 0);
        assert_eq!(settings.models.embedding.context_size, 2048);
        assert_eq!(settings.models.embedding.document_prefix, None);
        assert_eq!(settings.models.embedding.query_prefix, None);

        assert_eq!(settings.models.stt.path, None);
        assert_eq!(settings.models.stt.language, "auto");

        assert_eq!(settings.models.tts.voice_path, None);

        assert_eq!(settings.compaction.trigger_percent, 80);
        assert_eq!(settings.compaction.target_percent, 45);

        assert!(settings.voice.speak_replies);
        assert!(settings.voice.pause_mic_while_speaking);

        assert_eq!(settings.appearance.theme, Theme::System);
        assert!(settings.appearance.always_on_top);
    }

    #[test]
    fn a_partial_file_loads_with_defaults_elsewhere() {
        let (_dir, path) = temp_path();
        write(&path, "[models.text]\npath = \"C:/m.gguf\"\n");

        let LoadOutcome::Loaded(settings) = load(&path) else {
            panic!("expected a loaded outcome");
        };

        assert_eq!(settings.models.text.path, Some(PathBuf::from("C:/m.gguf")));
        assert_eq!(settings.models.text.context_size, 4096);
        assert_eq!(settings.models.vision, VisionSlot::default());
        assert_eq!(settings.compaction, CompactionSettings::default());
        assert_eq!(settings.appearance, Appearance::default());
    }

    #[test]
    fn unknown_keys_are_ignored() {
        let (_dir, path) = temp_path();
        write(
            &path,
            "surprise = 1\n[models.text]\npath = \"C:/m.gguf\"\nnope = true\n",
        );

        let LoadOutcome::Loaded(settings) = load(&path) else {
            panic!("expected a loaded outcome");
        };
        assert_eq!(settings.models.text.path, Some(PathBuf::from("C:/m.gguf")));
    }

    #[test]
    fn invalid_toml_blocks_saving_and_leaves_the_file_alone() {
        let (_dir, path) = temp_path();
        let original = b"[models.text\npath = \"C:/m.gguf\"\n".to_vec();
        fs::write(&path, &original).expect("write fixture");

        let mut store = SettingsStore::open(&path);
        let error = store.load_error().expect("load error");
        assert!(
            error.contains("line"),
            "error should mention a line: {error}"
        );
        assert!(store.saving_blocked());
        assert_eq!(store.settings(), &Settings::default());

        let changed = Settings {
            appearance: Appearance {
                always_on_top: false,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(matches!(store.update(changed), Err(SaveError::Blocked)));
        assert!(!store.settings().appearance.always_on_top);
        assert_eq!(fs::read(&path).unwrap(), original);
    }

    #[test]
    fn reset_backs_up_an_invalid_file_and_unblocks_saving() {
        let (_dir, path) = temp_path();
        let original = b"this is not toml at all\n".to_vec();
        fs::write(&path, &original).expect("write fixture");

        let mut store = SettingsStore::open(&path);
        assert!(store.saving_blocked());

        let backup = store
            .reset_to_defaults()
            .expect("reset should succeed")
            .expect("a backup should be made");
        assert_eq!(fs::read(&backup).unwrap(), original);

        let LoadOutcome::Loaded(settings) = load(&path) else {
            panic!("the rewritten file should load");
        };
        assert_eq!(settings, Settings::default());
        assert!(!store.saving_blocked());

        let next = Settings {
            voice: VoiceSettings {
                speak_replies: false,
                ..Default::default()
            },
            ..Default::default()
        };
        store.update(next.clone()).expect("saving works again");
        let LoadOutcome::Loaded(saved) = load(&path) else {
            panic!("the saved file should load");
        };
        assert_eq!(saved, next.normalized());
    }

    #[test]
    fn save_load_round_trip() {
        let (_dir, path) = temp_path();
        let settings = Settings {
            models: ModelSlots {
                text: TextSlot {
                    path: Some(PathBuf::from("C:/models/text.gguf")),
                    ..Default::default()
                },
                embedding: EmbeddingSlot {
                    query_prefix: Some("query: ".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            },
            memory: MemoryOverrides {
                ram_gb: Some(32.0),
                ..Default::default()
            },
            appearance: Appearance {
                theme: Theme::Dark,
                ..Default::default()
            },
            ..Default::default()
        };

        save(&path, &settings).expect("save");
        let LoadOutcome::Loaded(loaded) = load(&path) else {
            panic!("expected a loaded outcome");
        };
        assert_eq!(loaded, settings);
    }

    #[test]
    fn compaction_validation() {
        let case = |t, p| CompactionSettings {
            trigger_percent: t,
            target_percent: p,
        };
        assert_eq!(case(80, 45).validated(), (80, 45));
        assert_eq!(case(80, 80).validated(), (80, 45));
        assert_eq!(case(100, 50).validated(), (80, 45));
        assert_eq!(case(0, 0).validated(), (80, 45));
        assert_eq!(case(70, 30).validated(), (70, 30));
    }

    #[test]
    fn theme_serialises_lowercase() {
        let text = toml::to_string(&Appearance::default()).expect("serialise");
        assert!(text.contains("theme = \"system\""), "{text}");
    }

    #[test]
    fn normalized_drops_bad_memory_overrides() {
        let settings = Settings {
            memory: MemoryOverrides {
                ram_gb: Some(f64::NAN),
                vram_gb: Some(-1.0),
            },
            compaction: CompactionSettings {
                trigger_percent: 10,
                target_percent: 20,
            },
            ..Default::default()
        };

        let settings = settings.normalized();
        assert_eq!(settings.memory.ram_gb, None);
        assert_eq!(settings.memory.vram_gb, None);
        assert_eq!(settings.compaction.trigger_percent, 80);
        assert_eq!(settings.compaction.target_percent, 45);
    }

    #[test]
    fn missing_file_opens_with_defaults() {
        let (_dir, path) = temp_path();
        let store = SettingsStore::open(&path);
        assert_eq!(store.settings(), &Settings::default());
        assert!(store.load_error().is_none());
        assert!(!store.saving_blocked());
        assert!(!path.exists());
    }
}
