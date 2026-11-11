//! The SQLite chat history store.
//!
//! History is append-only. A file that exists but cannot be read is never
//! overwritten or deleted: the caller gets an error and the bytes stay put.
//!
//! The database also holds conversation summaries: each one folds in every
//! message up to and including [`Summary::covers_through`], so compaction
//! (a later stage) can replace old turns with a short text. Both tables are
//! guarded by `BEFORE UPDATE`/`BEFORE DELETE` triggers that raise
//! `chat history is append-only`, so a mistake in a later stage cannot rewrite
//! or drop stored history.

use std::fs;
use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension};

use super::message::{ChatMessage, Role};

/// The schema version this build writes and understands.
const SCHEMA_VERSION: i64 = 2;

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

/// One stored conversation summary.
///
/// A summary stands in for every message up to [`Summary::covers_through`], so
/// later prompt building can send the summary instead of those turns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    /// The database row id; higher means newer.
    pub id: i64,
    /// The summary text that replaces the folded-in messages.
    pub text: String,
    /// The id of the newest message this summary folds in.
    pub covers_through: i64,
    /// When it was stored, in unix seconds.
    pub created_at: i64,
}

/// A handle on the append-only chat history.
#[derive(Debug)]
pub struct ChatStore {
    conn: Connection,
}

impl ChatStore {
    /// Opens (or creates) the history database at `path`, migrating it to the
    /// current schema.
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
        let mut conn = Connection::open(path).map_err(|err| {
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

        // Migrations are stepwise and each step commits on its own, so an
        // interrupted upgrade resumes where it stopped.
        if version < 1 {
            let tx = conn.transaction().map_err(ChatStoreError::Database)?;
            tx.execute_batch(
                "CREATE TABLE IF NOT EXISTS messages (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     role TEXT NOT NULL,
                     text TEXT NOT NULL,
                     from_voice INTEGER NOT NULL DEFAULT 0,
                     created_at INTEGER NOT NULL
                 );",
            )
            .map_err(ChatStoreError::Database)?;
            tx.pragma_update(None, "user_version", 1)
                .map_err(ChatStoreError::Database)?;
            tx.commit().map_err(ChatStoreError::Database)?;
        }

        if version < 2 {
            let tx = conn.transaction().map_err(ChatStoreError::Database)?;
            tx.execute_batch(
                "CREATE TABLE IF NOT EXISTS summaries (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     text TEXT NOT NULL,
                     covers_through INTEGER NOT NULL,
                     created_at INTEGER NOT NULL
                 );
                 CREATE TRIGGER IF NOT EXISTS messages_no_update BEFORE UPDATE ON messages
                     BEGIN SELECT RAISE(ABORT, 'chat history is append-only'); END;
                 CREATE TRIGGER IF NOT EXISTS messages_no_delete BEFORE DELETE ON messages
                     BEGIN SELECT RAISE(ABORT, 'chat history is append-only'); END;
                 CREATE TRIGGER IF NOT EXISTS summaries_no_update BEFORE UPDATE ON summaries
                     BEGIN SELECT RAISE(ABORT, 'chat history is append-only'); END;
                 CREATE TRIGGER IF NOT EXISTS summaries_no_delete BEFORE DELETE ON summaries
                     BEGIN SELECT RAISE(ABORT, 'chat history is append-only'); END;",
            )
            .map_err(ChatStoreError::Database)?;
            tx.pragma_update(None, "user_version", SCHEMA_VERSION)
                .map_err(ChatStoreError::Database)?;
            tx.commit().map_err(ChatStoreError::Database)?;
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
            .query_map([limit], message_from_row)
            .map_err(ChatStoreError::Database)?;

        collect_oldest_first(rows)
    }

    /// Returns the newest `limit` messages with an id above `after_id`,
    /// ordered oldest first.
    pub fn messages_after(
        &self,
        after_id: i64,
        limit: usize,
    ) -> Result<Vec<ChatMessage>, ChatStoreError> {
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, role, text, from_voice, created_at
                 FROM messages WHERE id > ?1 ORDER BY id DESC LIMIT ?2",
            )
            .map_err(ChatStoreError::Database)?;

        let rows = stmt
            .query_map(rusqlite::params![after_id, limit], message_from_row)
            .map_err(ChatStoreError::Database)?;

        collect_oldest_first(rows)
    }

    /// The newest message, if any.
    pub fn latest(&self) -> Result<Option<ChatMessage>, ChatStoreError> {
        let one = self.recent(1)?;
        Ok(one.into_iter().next())
    }

    /// Appends one summary and returns it with its assigned id.
    ///
    /// `covers_through` is the id of the newest message the summary folds in.
    pub fn append_summary(
        &mut self,
        text: &str,
        covers_through: i64,
    ) -> Result<Summary, ChatStoreError> {
        let created_at = unix_seconds();
        self.conn
            .execute(
                "INSERT INTO summaries (text, covers_through, created_at)
                 VALUES (?1, ?2, ?3)",
                rusqlite::params![text, covers_through, created_at],
            )
            .map_err(ChatStoreError::Database)?;
        Ok(Summary {
            id: self.conn.last_insert_rowid(),
            text: text.to_string(),
            covers_through,
            created_at,
        })
    }

    /// The newest summary (the row with the highest id), if any.
    pub fn latest_summary(&self) -> Result<Option<Summary>, ChatStoreError> {
        self.conn
            .query_row(
                "SELECT id, text, covers_through, created_at
                 FROM summaries ORDER BY id DESC LIMIT 1",
                [],
                |row| {
                    Ok(Summary {
                        id: row.get(0)?,
                        text: row.get(1)?,
                        covers_through: row.get(2)?,
                        created_at: row.get(3)?,
                    })
                },
            )
            .optional()
            .map_err(ChatStoreError::Database)
    }

    /// The raw connection, for tests only.
    #[cfg(test)]
    fn conn(&self) -> &Connection {
        &self.conn
    }
}

/// Decodes one `SELECT id, role, text, from_voice, created_at` row.
fn message_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChatMessage> {
    let role_text: String = row.get(1)?;
    let role = Role::from_storage(&role_text).ok_or_else(|| {
        rusqlite::Error::InvalidColumnType(1, "role".to_string(), rusqlite::types::Type::Text)
    })?;
    Ok(ChatMessage {
        id: row.get(0)?,
        role,
        text: row.get(2)?,
        from_voice: row.get::<_, i64>(3)? != 0,
        created_at: row.get(4)?,
    })
}

/// Collects a newest-first result set into messages ordered oldest first.
fn collect_oldest_first<I>(rows: I) -> Result<Vec<ChatMessage>, ChatStoreError>
where
    I: Iterator<Item = rusqlite::Result<ChatMessage>>,
{
    let mut messages = rows
        .collect::<Result<Vec<_>, _>>()
        .map_err(ChatStoreError::Database)?;
    messages.reverse();
    Ok(messages)
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

    /// Writes a v1 database by hand: the old `messages` table, the given user
    /// messages and `user_version = 1`.
    fn write_v1_database(path: &Path, texts: &[&str]) {
        let conn = Connection::open(path).expect("sqlite open");
        conn.execute_batch(
            "CREATE TABLE messages (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 role TEXT NOT NULL,
                 text TEXT NOT NULL,
                 from_voice INTEGER NOT NULL DEFAULT 0,
                 created_at INTEGER NOT NULL
             );",
        )
        .expect("create v1 table");
        for text in texts {
            conn.execute(
                "INSERT INTO messages (role, text, from_voice, created_at)
                 VALUES ('user', ?1, 0, 0)",
                [text],
            )
            .expect("insert v1 message");
        }
        conn.pragma_update(None, "user_version", 1)
            .expect("set v1 version");
    }

    #[test]
    fn a_v1_database_is_upgraded_and_keeps_its_messages() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("regent.db");
        write_v1_database(&path, &["old one", "old two"]);

        let mut store = ChatStore::open(&path).expect("open v1 store");
        let messages = store.recent(10).expect("recent");
        assert_eq!(
            messages.iter().map(|m| m.text.as_str()).collect::<Vec<_>>(),
            vec!["old one", "old two"]
        );

        let version: i64 = store
            .conn()
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .expect("read version");
        assert_eq!(version, 2);

        let summary = store
            .append_summary("they talked", messages[1].id)
            .expect("append summary");
        assert_eq!(summary.text, "they talked");
        assert_eq!(store.latest_summary().expect("latest"), Some(summary));
    }

    #[test]
    fn messages_cannot_be_updated_or_deleted() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut store = store_in(&dir);
        let message = store.append(Role::User, "keep me", false).expect("append");

        assert!(store
            .conn()
            .execute("UPDATE messages SET text = 'x' WHERE id = ?1", [message.id])
            .is_err());
        assert!(store.conn().execute("DELETE FROM messages", []).is_err());

        let stored = store.recent(10).expect("recent");
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].text, "keep me");

        let summary = store
            .append_summary("folds in", message.id)
            .expect("append summary");
        assert!(store
            .conn()
            .execute("UPDATE summaries SET text = 'x'", [])
            .is_err());
        assert!(store.conn().execute("DELETE FROM summaries", []).is_err());
        assert_eq!(store.latest_summary().expect("latest"), Some(summary));
    }

    #[test]
    fn latest_summary_is_the_newest() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut store = store_in(&dir);

        assert_eq!(store.latest_summary().expect("latest"), None);

        let first = store.append_summary("first", 1).expect("append summary");
        let second = store.append_summary("second", 3).expect("append summary");
        assert!(second.id > first.id);

        let latest = store.latest_summary().expect("latest").expect("some");
        assert_eq!(latest, second);
        assert_eq!(latest.covers_through, 3);
    }

    #[test]
    fn messages_after_respects_the_boundary_and_limit() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut store = store_in(&dir);
        let mut ids = Vec::new();
        for text in ["one", "two", "three", "four", "five"] {
            ids.push(store.append(Role::User, text, false).expect("append").id);
        }

        let all_after = store.messages_after(ids[1], 10).expect("messages after");
        assert_eq!(
            all_after
                .iter()
                .map(|m| m.text.as_str())
                .collect::<Vec<_>>(),
            vec!["three", "four", "five"]
        );

        let two_after = store.messages_after(ids[1], 2).expect("messages after");
        assert_eq!(
            two_after
                .iter()
                .map(|m| m.text.as_str())
                .collect::<Vec<_>>(),
            vec!["four", "five"]
        );
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
