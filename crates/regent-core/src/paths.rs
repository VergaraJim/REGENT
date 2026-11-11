//! Where Regent keeps its settings, database, models and caches.
//!
//! The locations follow the platform conventions provided by the `dirs` crate:
//! settings under the config dir, persistent data under the local data dir, and
//! throw-away data under the cache dir.

use std::path::{Path, PathBuf};

/// Every folder and file Regent uses, resolved once at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppPaths {
    /// `<config>/regent/settings.toml`
    pub settings_file: PathBuf,
    /// `<data_local>/regent`
    pub data_dir: PathBuf,
    /// `<data_local>/regent/regent.db`
    pub database_file: PathBuf,
    /// `<data_local>/regent/models`
    pub models_dir: PathBuf,
    /// `<data_local>/regent/logs`
    pub log_dir: PathBuf,
    /// `<cache>/regent/kv`
    pub prefix_cache_dir: PathBuf,
}

impl AppPaths {
    /// Builds the path set from the three platform base directories.
    ///
    /// This is pure: it touches no disk, so tests can pass any base dirs.
    pub fn from_base_dirs(config: &Path, data_local: &Path, cache: &Path) -> Self {
        let data_dir = data_local.join("regent");
        Self {
            settings_file: config.join("regent").join("settings.toml"),
            database_file: data_dir.join("regent.db"),
            models_dir: data_dir.join("models"),
            log_dir: data_dir.join("logs"),
            data_dir,
            prefix_cache_dir: cache.join("regent").join("kv"),
        }
    }

    /// Detects the platform base directories, or `None` when the platform
    /// reports none of them.
    pub fn detect() -> Option<Self> {
        let config = dirs::config_dir()?;
        let data_local = dirs::data_local_dir()?;
        let cache = dirs::cache_dir()?;
        Some(Self::from_base_dirs(&config, &data_local, &cache))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_base_dirs_produces_the_expected_paths() {
        let paths = AppPaths::from_base_dirs(
            Path::new("/cfg"),
            Path::new("/data-local"),
            Path::new("/cache"),
        );
        assert_eq!(
            paths.settings_file,
            PathBuf::from("/cfg").join("regent").join("settings.toml")
        );
        assert_eq!(paths.data_dir, PathBuf::from("/data-local").join("regent"));
        assert_eq!(
            paths.database_file,
            PathBuf::from("/data-local")
                .join("regent")
                .join("regent.db")
        );
        assert_eq!(
            paths.models_dir,
            PathBuf::from("/data-local").join("regent").join("models")
        );
        assert_eq!(
            paths.log_dir,
            PathBuf::from("/data-local").join("regent").join("logs")
        );
        assert_eq!(
            paths.prefix_cache_dir,
            PathBuf::from("/cache").join("regent").join("kv")
        );
    }
}
