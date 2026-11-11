//! The persisted prefix cache.
//!
//! The fixed opening of every prompt (the system text) is the same each time
//! Regent runs. When the engine supports it, the KV state for that prefix is
//! written under the cache directory and loaded on the first turn of a later
//! session, so that turn does not have to decode the system prompt again.
//!
//! A cache entry is a pair of files named after a hash of its [`PrefixKey`]:
//! `<stem>.state` holds the engine's bytes and `<stem>.toml` holds the key and
//! the state length, so a truncated or replaced state file is never trusted.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

use crate::hash::fnv1a64;

/// The engine traits a cache entry is only valid for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheIdentity {
    /// The KV cache element type, for example `"f16"`.
    pub kv_type: String,
    /// The chat template the engine applies to the prompt.
    pub chat_template: String,
    /// The engine's own version marker.
    pub engine_version: String,
    /// The engine's context window, in tokens.
    pub context_size: usize,
}

/// Identifies one prefix cache entry.
///
/// Two entries match only when every field matches, so any change to the
/// model, the engine or the system text makes the old state unusable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixKey {
    /// The model file's path.
    pub model_path: String,
    /// The model file's size, in bytes.
    pub model_size: u64,
    /// The model file's modification time, in unix seconds.
    pub model_mtime_secs: i64,
    /// The engine's context window, in tokens.
    pub context_size: u64,
    /// The KV cache element type.
    pub kv_type: String,
    /// The chat template the engine applies.
    pub chat_template: String,
    /// The fixed prompt prefix itself, byte for byte.
    pub prefix_text: String,
    /// The engine's version marker.
    pub engine_version: String,
}

impl PrefixKey {
    /// Builds the key for `prefix_text`, reading the model file's size and
    /// modification time.
    pub fn new(
        model_path: &Path,
        identity: &CacheIdentity,
        prefix_text: &str,
    ) -> io::Result<PrefixKey> {
        let metadata = fs::metadata(model_path)?;
        let mtime_secs = metadata
            .modified()
            .ok()
            .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |duration| duration.as_secs() as i64);
        Ok(PrefixKey {
            model_path: model_path.to_string_lossy().into_owned(),
            model_size: metadata.len(),
            model_mtime_secs: mtime_secs,
            context_size: identity.context_size as u64,
            kv_type: identity.kv_type.clone(),
            chat_template: identity.chat_template.clone(),
            prefix_text: prefix_text.to_string(),
            engine_version: identity.engine_version.clone(),
        })
    }
}

/// The metadata written beside a state file.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PrefixMeta {
    key: PrefixKey,
    state_len: u64,
}

/// How many state files the cache keeps; older ones are pruned on every store.
pub const MAX_STATES: usize = 3;

/// The on-disk prefix cache.
pub struct PrefixCache {
    dir: PathBuf,
}

impl PrefixCache {
    /// A cache rooted at `dir`; the directory is created on the first store.
    pub fn new(dir: impl Into<PathBuf>) -> PrefixCache {
        PrefixCache { dir: dir.into() }
    }

    /// The state file for `key`, when a matching, complete entry exists.
    pub fn lookup(&self, key: &PrefixKey) -> Option<PathBuf> {
        let stem = self.stem(key);
        let meta_text = fs::read_to_string(self.dir.join(format!("{stem}.toml"))).ok()?;
        let meta: PrefixMeta = toml::from_str(&meta_text).ok()?;
        if meta.key != *key {
            return None;
        }
        let state = self.dir.join(format!("{stem}.state"));
        let state_len = fs::metadata(&state).ok()?.len();
        (state_len == meta.state_len).then_some(state)
    }

    /// Writes the state for `key` through `write` and records it.
    ///
    /// `write` receives the temporary path to fill. An empty file, a write
    /// failure or any file-system error leaves no entry behind and returns a
    /// plain-language error.
    pub fn store(
        &self,
        key: &PrefixKey,
        write: &mut dyn FnMut(&Path) -> Result<(), String>,
    ) -> Result<PathBuf, String> {
        let stem = self.stem(key);
        let state = self.dir.join(format!("{stem}.state"));
        let state_tmp = self.dir.join(format!("{stem}.state.tmp"));
        let meta = self.dir.join(format!("{stem}.toml"));
        let meta_tmp = self.dir.join(format!("{stem}.toml.tmp"));

        let result = self.store_files(key, write, &state, &state_tmp, &meta, &meta_tmp);
        if result.is_err() {
            let _ = fs::remove_file(&state_tmp);
            let _ = fs::remove_file(&meta_tmp);
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn store_files(
        &self,
        key: &PrefixKey,
        write: &mut dyn FnMut(&Path) -> Result<(), String>,
        state: &Path,
        state_tmp: &Path,
        meta: &Path,
        meta_tmp: &Path,
    ) -> Result<PathBuf, String> {
        fs::create_dir_all(&self.dir).map_err(string_error)?;

        write(state_tmp)?;
        let state_len = fs::metadata(state_tmp).map_err(string_error)?.len();
        if state_len == 0 {
            let _ = fs::remove_file(state_tmp);
            return Err("The engine wrote an empty prompt cache.".to_string());
        }

        fs::rename(state_tmp, state).map_err(string_error)?;

        let contents = PrefixMeta {
            key: key.clone(),
            state_len,
        };
        let written = toml::to_string(&contents).map_err(string_error)?;
        fs::write(meta_tmp, written).map_err(string_error)?;
        fs::rename(meta_tmp, meta).map_err(string_error)?;

        self.prune();
        Ok(state.to_path_buf())
    }

    /// Keeps the newest [`MAX_STATES`] metadata files and removes the rest with
    /// their state files.
    fn prune(&self) {
        let Ok(entries) = fs::read_dir(&self.dir) else {
            return;
        };
        let mut metas: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("toml") {
                continue;
            }
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            let Ok(modified) = metadata.modified() else {
                continue;
            };
            metas.push((modified, path));
        }
        metas.sort_by_key(|entry| std::cmp::Reverse(entry.0));
        for (_, meta) in metas.into_iter().skip(MAX_STATES) {
            if let Some(stem) = meta.file_stem().and_then(|stem| stem.to_str()) {
                let _ = fs::remove_file(self.dir.join(format!("{stem}.state")));
            }
            let _ = fs::remove_file(&meta);
        }
    }

    /// The file stem for `key`: an FNV-1a hash of its TOML form.
    fn stem(&self, key: &PrefixKey) -> String {
        let encoded = toml::to_string(key).unwrap_or_default();
        format!("{:016x}", fnv1a64(encoded.as_bytes()))
    }
}

fn string_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn identity() -> CacheIdentity {
        CacheIdentity {
            kv_type: "f16".to_string(),
            chat_template: "test-template".to_string(),
            engine_version: "test-1".to_string(),
            context_size: 4096,
        }
    }

    fn model(dir: &Path) -> PathBuf {
        let path = dir.join("text.gguf");
        fs::write(&path, b"fake model").expect("write model fixture");
        path
    }

    fn key(model: &Path, prefix_text: &str) -> PrefixKey {
        PrefixKey::new(model, &identity(), prefix_text).expect("build key")
    }

    fn store(cache: &PrefixCache, key: &PrefixKey, text: &str) -> PathBuf {
        cache
            .store(key, &mut |path| {
                fs::write(path, text).map_err(|error| error.to_string())
            })
            .expect("store should succeed")
    }

    fn state_files(dir: &Path) -> Vec<PathBuf> {
        fs::read_dir(dir)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|entry| entry.path())
                    .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("state"))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn tmp_files(dir: &Path) -> Vec<PathBuf> {
        fs::read_dir(dir)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|entry| entry.path())
                    .filter(|path| path.to_string_lossy().ends_with(".tmp"))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn an_empty_dir_finds_nothing() {
        let dir = tempfile::tempdir().expect("temp dir");
        let cache = PrefixCache::new(dir.path());
        let model = model(dir.path());
        assert_eq!(cache.lookup(&key(&model, "system")), None);
    }

    #[test]
    fn store_then_lookup_finds_the_state() {
        let dir = tempfile::tempdir().expect("temp dir");
        let cache = PrefixCache::new(dir.path());
        let model = model(dir.path());
        let key = key(&model, "system");

        let path = store(&cache, &key, "system");

        assert_eq!(cache.lookup(&key), Some(path.clone()));
        assert_eq!(fs::read_to_string(path).expect("read state"), "system");
    }

    #[test]
    fn a_different_prefix_text_finds_nothing() {
        let dir = tempfile::tempdir().expect("temp dir");
        let cache = PrefixCache::new(dir.path());
        let model = model(dir.path());
        store(&cache, &key(&model, "first"), "first");

        assert_eq!(cache.lookup(&key(&model, "second")), None);
    }

    #[test]
    fn a_truncated_state_file_finds_nothing() {
        let dir = tempfile::tempdir().expect("temp dir");
        let cache = PrefixCache::new(dir.path());
        let model = model(dir.path());
        let key = key(&model, "system");
        let path = store(&cache, &key, "system");

        let len = fs::metadata(&path).expect("metadata").len();
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open state")
            .set_len(len - 1)
            .expect("truncate state");

        assert_eq!(cache.lookup(&key), None);
    }

    #[test]
    fn storing_four_keys_keeps_only_the_newest_three() {
        let dir = tempfile::tempdir().expect("temp dir");
        let cache = PrefixCache::new(dir.path());
        let model = model(dir.path());

        let oldest = key(&model, "oldest");
        store(&cache, &oldest, "oldest");
        let mut newest = key(&model, "prefix1");
        for index in 1..4 {
            std::thread::sleep(Duration::from_millis(20));
            newest = key(&model, &format!("prefix{index}"));
            store(&cache, &newest, &format!("prefix{index}"));
        }

        assert_eq!(state_files(dir.path()).len(), MAX_STATES);
        assert_eq!(cache.lookup(&oldest), None, "the oldest entry is pruned");
        assert!(cache.lookup(&newest).is_some());
    }

    #[test]
    fn no_tmp_files_remain_after_a_successful_store() {
        let dir = tempfile::tempdir().expect("temp dir");
        let cache = PrefixCache::new(dir.path());
        let model = model(dir.path());
        store(&cache, &key(&model, "system"), "system");

        assert!(tmp_files(dir.path()).is_empty());
    }

    #[test]
    fn no_tmp_files_remain_after_a_failing_store() {
        let dir = tempfile::tempdir().expect("temp dir");
        let cache = PrefixCache::new(dir.path());
        let model = model(dir.path());
        let key = key(&model, "system");

        let result = cache.store(&key, &mut |path| {
            let _ = fs::write(path, b"partial");
            Err("the engine gave up".to_string())
        });

        assert!(result.is_err());
        assert!(tmp_files(dir.path()).is_empty());
        assert_eq!(cache.lookup(&key), None);
    }
}
