//! The SQLite chat history store.
//!
//! History is append-only. A file that exists but cannot be read is never
//! overwritten or deleted: the caller gets an error and the bytes stay put.

use std::fs;
use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::Connection;

use super::message::{ChatMessage, Role};

/// The schema version this build writes and understands.
const SCHEMA_VERSION: i64 = 1;

/// The most stored messages a single turn reads back.
///
/// Compaction (a later stage) bounds the actual prompt.
pub const HISTORY_READ_CAP: usize = 512;

/// What opening or using the chat history can fail with.
#[derive(Debug, thiserror::Error)]
pub enum ChatStoreError {
    /// The file or its folder could not be created or opened.
    #[error("Regent couldn't open its chat history file. Check that the Regent data folder is writable.")]
    Io(#[from] io::Error),
    /// The file exists but is not a readable database; it was left alone.
    #[error("Regent couldn't read its chat history file. It was left untouched.")]
    Unreadable(#[source] rusqlite::Error),
    /// The file was written by a newer version of Regent.
    #[error(
        "Regent's chat history was made by a newer version of Regent. Update Regent to use it."
    )]
    NewerSchema,
    /// A query against an otherwise readable database failed.
    #[error("Regent couldn't update its chat history. Your existing messages are safe.")]
    Database(#[source] rusqlite::Error),
}

/// A handle on the append-only chat history.
#[derive(Debug)]
pub struct ChatStore {
    conn: Connection,
}

impl ChatStore {
    /// Opens (or creates) the history database at `path`.
    ///
    /// A file that exists but is not a readable database is reported as an
    /// error and left byte-for-byte untouched.
    pub fn open(path: &Path) -> Result<ChatStore, ChatStoreError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        let existed = path.exists();
        let conn = Connection::open(path).map_err(|err| {
            if existed {
                ChatStoreError::Unreadable(err)
            } else {
                ChatStoreError::Io(io::Error::other(err))
            }
        })?;

        // Reading the version forces SQLite to inspect the file header, so a
        // garbage file fails here before any write can happen.
        let version: i64 = conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .map_err(ChatStoreError::Unreadable)?;

        if version > SCHEMA_VERSION {
            return Err(ChatStoreError::NewerSchema);
        }

        if version == 0 {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS messages (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     role TEXT NOT NULL,
                     text TEXT NOT NULL,
                     from_voice INTEGER NOT NULL DEFAULT 0,
                     created_at INTEGER NOT NULL
                 );",
            )
            .map_err(ChatStoreError::Database)?;
            conn.pragma_update(None, "user_version", SCHEMA_VERSION)
                .map_err(ChatStoreError::Database)?;
        }

        Ok(ChatStore { conn })
    }

    /// Appends one message and returns it with its assigned id.
    pub fn append(
        &mut self,
        role: Role,
        text: &str,
        from_voice: bool,
    ) -> Result<ChatMessage, ChatStoreError> {
        let created_at = unix_seconds();
        self.conn
            .execute(
                "INSERT INTO messages (role, text, from_voice, created_at)
                 VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![role.as_str(), text, i64::from(from_voice), created_at],
            )
            .map_err(ChatStoreError::Database)?;
        let id = self.conn.last_insert_rowid();
        Ok(ChatMessage {
            id,
            role,
            text: text.to_string(),
            from_voice,
            created_at,
        })
    }

    /// Returns the newest `limit` messages, ordered oldest first.
    pub fn recent(&self, limit: usize) -> Result<Vec<ChatMessage>, ChatStoreError> {
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, role, text, from_voice, created_at
                 FROM messages ORDER BY id DESC LIMIT ?1",
            )
            .map_err(ChatStoreError::Database)?;

        let rows = stmt
            .query_map([limit], |row| {
                let role_text: String = row.get(1)?;
                Ok((
                    row.get::<_, i64>(0)?,
                    role_text,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            })
            .map_err(ChatStoreError::Database)?;

        let mut messages = Vec::new();
        for row in rows {
            let (id, role_text, text, from_voice, created_at) =
                row.map_err(ChatStoreError::Database)?;
            let role = Role::from_storage(&role_text).ok_or_else(|| {
                ChatStoreError::Database(rusqlite::Error::InvalidColumnType(
                    1,
                    "role".to_string(),
                    rusqlite::types::Type::Text,
                ))
            })?;
            messages.push(ChatMessage {
                id,
                role,
                text,
                from_voice: from_voice != 0,
                created_at,
            });
        }

        // The query returns newest first; callers want oldest first.
        messages.reverse();
        Ok(messages)
    }

    /// The newest message, if any.
    pub fn latest(&self) -> Result<Option<ChatMessage>, ChatStoreError> {
        let one = self.recent(1)?;
        Ok(one.into_iter().next())
    }
}

fn unix_seconds() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs() as i64,
        Err(_) => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_in(dir: &tempfile::TempDir) -> ChatStore {
        ChatStore::open(&dir.path().join("regent.db")).expect("open store")
    }

    #[test]
    fn append_and_recent_keep_order_and_limit() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut store = store_in(&dir);

        store.append(Role::User, "one", false).expect("append");
        store.append(Role::Assistant, "two", false).expect("append");
        store.append(Role::User, "three", true).expect("append");

        let all = store.recent(10).expect("recent");
        assert_eq!(
            all.iter().map(|m| m.text.as_str()).collect::<Vec<_>>(),
            vec!["one", "two", "three"]
        );
        assert!(all[2].from_voice);

        let last_two = store.recent(2).expect("recent");
        assert_eq!(
            last_two.iter().map(|m| m.text.as_str()).collect::<Vec<_>>(),
            vec!["two", "three"]
        );
    }

    #[test]
    fn a_reopened_database_keeps_its_data() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("regent.db");
        {
            let mut store = ChatStore::open(&path).expect("open store");
            store
                .append(Role::User, "remember me", false)
                .expect("append");
        }
        let store = ChatStore::open(&path).expect("reopen store");
        let messages = store.recent(HISTORY_READ_CAP).expect("recent");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text, "remember me");
        assert_eq!(messages[0].role, Role::User);
    }

    #[test]
    fn a_garbage_file_fails_to_open_and_is_untouched() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("regent.db");
        let original = b"this is definitely not a sqlite database\n".to_vec();
        fs::write(&path, &original).expect("write fixture");

        let error = ChatStore::open(&path).expect_err("open should fail");
        assert!(matches!(error, ChatStoreError::Unreadable(_)));
        assert!(error.to_string().contains("left untouched"));
        assert_eq!(fs::read(&path).expect("read back"), original);
    }

    #[test]
    fn a_newer_schema_is_refused() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("regent.db");
        {
            let conn = Connection::open(&path).expect("sqlite open");
            conn.pragma_update(None, "user_version", SCHEMA_VERSION + 1)
                .expect("bump version");
        }
        let error = ChatStore::open(&path).expect_err("open should fail");
        assert!(matches!(error, ChatStoreError::NewerSchema));
    }

    #[test]
    fn open_creates_missing_parent_directories() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("nested").join("regent.db");
        let store = ChatStore::open(&path).expect("open store");
        assert!(store.recent(1).expect("recent").is_empty());
        assert!(path.exists());
    }
}
