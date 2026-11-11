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
//!
//! Messages can also be `tool` rows: the result of a tool call Regent made,
//! carrying the tool's name and the raw call JSON. The `embeddings` table holds
//! one vector per message chunk, tagged with the embedding model id, so old
//! turns can be found by meaning long after they left the prompt.

use std::fs;
use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension};

use super::message::{ChatMessage, Role};

/// The schema version this build writes and understands.
const SCHEMA_VERSION: i64 = 3;

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
            tx.pragma_update(None, "user_version", 2)
                .map_err(ChatStoreError::Database)?;
            tx.commit().map_err(ChatStoreError::Database)?;
        }

        if version < 3 {
            let tx = conn.transaction().map_err(ChatStoreError::Database)?;
            tx.execute_batch(
                "ALTER TABLE messages ADD COLUMN tool_name TEXT;
                 ALTER TABLE messages ADD COLUMN tool_call TEXT;
                 CREATE TABLE IF NOT EXISTS embeddings (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     message_id INTEGER NOT NULL,
                     chunk_index INTEGER NOT NULL,
                     model_id TEXT NOT NULL,
                     dimension INTEGER NOT NULL,
                     vector BLOB NOT NULL,
                     UNIQUE (message_id, chunk_index, model_id)
                 );
                 CREATE INDEX IF NOT EXISTS embeddings_by_model ON embeddings (model_id, message_id);",
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
            tool_name: None,
            tool_call: None,
        })
    }

    /// Appends one `tool` row and returns it with its assigned id.
    ///
    /// `text` is the tool's result, `tool_name` is the tool that produced it
    /// and `tool_call` is the raw JSON of the call as the model wrote it.
    pub fn append_tool(
        &mut self,
        tool_name: &str,
        tool_call: &str,
        result: &str,
    ) -> Result<ChatMessage, ChatStoreError> {
        let created_at = unix_seconds();
        self.conn
            .execute(
                "INSERT INTO messages (role, text, from_voice, created_at, tool_name, tool_call)
                 VALUES (?1, ?2, 0, ?3, ?4, ?5)",
                rusqlite::params![
                    Role::Tool.as_str(),
                    result,
                    created_at,
                    tool_name,
                    tool_call
                ],
            )
            .map_err(ChatStoreError::Database)?;
        let id = self.conn.last_insert_rowid();
        Ok(ChatMessage {
            id,
            role: Role::Tool,
            text: result.to_string(),
            from_voice: false,
            created_at,
            tool_name: Some(tool_name.to_string()),
            tool_call: Some(tool_call.to_string()),
        })
    }

    /// Returns the newest `limit` messages, ordered oldest first.
    pub fn recent(&self, limit: usize) -> Result<Vec<ChatMessage>, ChatStoreError> {
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, role, text, from_voice, created_at, tool_name, tool_call
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
                "SELECT id, role, text, from_voice, created_at, tool_name, tool_call
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

    /// One message by row id, if it exists.
    pub fn message(&self, id: i64) -> Result<Option<ChatMessage>, ChatStoreError> {
        self.conn
            .query_row(
                "SELECT id, role, text, from_voice, created_at, tool_name, tool_call
                 FROM messages WHERE id = ?1",
                [id],
                message_from_row,
            )
            .optional()
            .map_err(ChatStoreError::Database)
    }

    /// The nearest message before `id` whose role is `user` or `assistant`.
    ///
    /// Tool rows are skipped: they are never a conversational turn on their
    /// own.
    pub fn previous_message(&self, id: i64) -> Result<Option<ChatMessage>, ChatStoreError> {
        self.conn
            .query_row(
                "SELECT id, role, text, from_voice, created_at, tool_name, tool_call
                 FROM messages
                 WHERE id < ?1 AND role IN ('user', 'assistant')
                 ORDER BY id DESC LIMIT 1",
                [id],
                message_from_row,
            )
            .optional()
            .map_err(ChatStoreError::Database)
    }

    /// Returns the oldest `limit` `user`/`assistant` messages with non-blank
    /// text that have no embedding yet for `model_id`, ordered by id ascending.
    ///
    /// Tool rows and blank messages are never indexed.
    pub fn unindexed_messages(
        &self,
        model_id: &str,
        limit: usize,
    ) -> Result<Vec<ChatMessage>, ChatStoreError> {
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, role, text, from_voice, created_at, tool_name, tool_call
                 FROM messages
                 WHERE role IN ('user', 'assistant')
                   AND trim(text) <> ''
                   AND NOT EXISTS (
                       SELECT 1 FROM embeddings
                       WHERE embeddings.message_id = messages.id
                         AND embeddings.model_id = ?1
                   )
                 ORDER BY id ASC LIMIT ?2",
            )
            .map_err(ChatStoreError::Database)?;

        let rows = stmt
            .query_map(rusqlite::params![model_id, limit], message_from_row)
            .map_err(ChatStoreError::Database)?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(ChatStoreError::Database)
    }

    /// Stores one vector per chunk of `message_id` for `model_id`.
    ///
    /// Chunk indexes are the position in `vectors`. All rows are written in one
    /// transaction, so a failure stores nothing. An empty `vectors` is an
    /// error.
    pub fn append_embeddings(
        &mut self,
        message_id: i64,
        model_id: &str,
        vectors: &[Vec<f32>],
    ) -> Result<(), ChatStoreError> {
        if vectors.is_empty() {
            return Err(ChatStoreError::Database(
                rusqlite::Error::InvalidParameterName("no embedding vectors given".to_string()),
            ));
        }

        let tx = self.conn.transaction().map_err(ChatStoreError::Database)?;
        {
            let mut stmt = tx
                .prepare(
                    "INSERT INTO embeddings
                         (message_id, chunk_index, model_id, dimension, vector)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                )
                .map_err(ChatStoreError::Database)?;
            for (chunk_index, vector) in vectors.iter().enumerate() {
                let chunk_index = i64::try_from(chunk_index).map_err(|err| {
                    ChatStoreError::Database(rusqlite::Error::InvalidParameterName(err.to_string()))
                })?;
                let dimension = i64::try_from(vector.len()).map_err(|err| {
                    ChatStoreError::Database(rusqlite::Error::InvalidParameterName(err.to_string()))
                })?;
                stmt.execute(rusqlite::params![
                    message_id,
                    chunk_index,
                    model_id,
                    dimension,
                    f32s_to_bytes(vector)
                ])
                .map_err(ChatStoreError::Database)?;
            }
        }
        tx.commit().map_err(ChatStoreError::Database)
    }

    /// Every stored vector for `model_id`, ordered by message id and then
    /// chunk index.
    pub fn load_vectors(&self, model_id: &str) -> Result<Vec<(i64, Vec<f32>)>, ChatStoreError> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT message_id, vector FROM embeddings
                 WHERE model_id = ?1 ORDER BY message_id, chunk_index",
            )
            .map_err(ChatStoreError::Database)?;

        let rows = stmt
            .query_map([model_id], |row| {
                let message_id: i64 = row.get(0)?;
                let bytes: Vec<u8> = row.get(1)?;
                Ok((message_id, bytes_to_f32s(&bytes)))
            })
            .map_err(ChatStoreError::Database)?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(ChatStoreError::Database)
    }

    /// How many indexable messages exist and how many of them are indexed for
    /// `model_id`, as `(indexed, total)`.
    pub fn index_counts(&self, model_id: &str) -> Result<(usize, usize), ChatStoreError> {
        let (total, indexed): (i64, i64) = self
            .conn
            .query_row(
                "SELECT
                     (SELECT COUNT(*) FROM messages
                      WHERE role IN ('user', 'assistant') AND trim(text) <> ''),
                     (SELECT COUNT(*) FROM messages m
                      WHERE m.role IN ('user', 'assistant') AND trim(m.text) <> ''
                        AND EXISTS (
                            SELECT 1 FROM embeddings e
                            WHERE e.message_id = m.id AND e.model_id = ?1
                        ))",
                [model_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(ChatStoreError::Database)?;

        Ok((count_to_usize(indexed), count_to_usize(total)))
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

/// Decodes one `SELECT id, role, text, from_voice, created_at, tool_name,
/// tool_call` row.
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
        tool_name: row.get(5)?,
        tool_call: row.get(6)?,
    })
}

/// Writes `values` as little-endian `f32`s, four bytes each.
fn f32s_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Reads little-endian `f32`s written by [`f32s_to_bytes`].
///
/// A trailing partial value is ignored.
fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// Converts a SQL `COUNT(*)` to `usize`, saturating on absurd values.
fn count_to_usize(count: i64) -> usize {
    usize::try_from(count).unwrap_or(usize::MAX)
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

    /// The v1 `messages` table SQL.
    const V1_MESSAGES_SQL: &str = "CREATE TABLE messages (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             role TEXT NOT NULL,
             text TEXT NOT NULL,
             from_voice INTEGER NOT NULL DEFAULT 0,
             created_at INTEGER NOT NULL
         );";

    /// The v2 `summaries` table and append-only triggers.
    const V2_SUMMARIES_SQL: &str = "CREATE TABLE summaries (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             text TEXT NOT NULL,
             covers_through INTEGER NOT NULL,
             created_at INTEGER NOT NULL
         );
         CREATE TRIGGER messages_no_update BEFORE UPDATE ON messages
             BEGIN SELECT RAISE(ABORT, 'chat history is append-only'); END;
         CREATE TRIGGER messages_no_delete BEFORE DELETE ON messages
             BEGIN SELECT RAISE(ABORT, 'chat history is append-only'); END;
         CREATE TRIGGER summaries_no_update BEFORE UPDATE ON summaries
             BEGIN SELECT RAISE(ABORT, 'chat history is append-only'); END;
         CREATE TRIGGER summaries_no_delete BEFORE DELETE ON summaries
             BEGIN SELECT RAISE(ABORT, 'chat history is append-only'); END;";

    /// Writes a v1 database by hand: the old `messages` table, the given user
    /// messages and `user_version = 1`.
    fn write_v1_database(path: &Path, texts: &[&str]) {
        let conn = Connection::open(path).expect("sqlite open");
        conn.execute_batch(V1_MESSAGES_SQL)
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

    /// Writes a v2 database by hand: the v1 `messages` table, the v2
    /// `summaries` table and triggers, one message and `user_version = 2`.
    fn write_v2_database(path: &Path) {
        let conn = Connection::open(path).expect("sqlite open");
        conn.execute_batch(V1_MESSAGES_SQL)
            .expect("create v1 table");
        conn.execute_batch(V2_SUMMARIES_SQL)
            .expect("create v2 objects");
        conn.execute(
            "INSERT INTO messages (role, text, from_voice, created_at)
             VALUES ('user', 'kept from v2', 0, 0)",
            [],
        )
        .expect("insert v2 message");
        conn.pragma_update(None, "user_version", 2)
            .expect("set v2 version");
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
        assert_eq!(version, SCHEMA_VERSION);

        let summary = store
            .append_summary("they talked", messages[1].id)
            .expect("append summary");
        assert_eq!(summary.text, "they talked");
        assert_eq!(store.latest_summary().expect("latest"), Some(summary));
    }

    #[test]
    fn a_v2_database_is_upgraded_and_keeps_its_messages() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("regent.db");
        write_v2_database(&path);

        let mut store = ChatStore::open(&path).expect("open v2 store");
        let messages = store.recent(10).expect("recent");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text, "kept from v2");
        assert_eq!(messages[0].tool_name, None);
        assert_eq!(messages[0].tool_call, None);

        let version: i64 = store
            .conn()
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .expect("read version");
        assert_eq!(version, 3);

        let tool = store
            .append_tool("clock", "{\"at\":\"now\"}", "12:00")
            .expect("append tool");
        assert_eq!(tool.role, Role::Tool);

        store
            .append_embeddings(messages[0].id, "model-a", &[vec![0.5, -2.0]])
            .expect("append embeddings");
        assert_eq!(
            store.load_vectors("model-a").expect("load vectors"),
            vec![(messages[0].id, vec![0.5, -2.0])]
        );
    }

    #[test]
    fn append_tool_stores_the_name_call_and_result() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut store = store_in(&dir);

        let row = store
            .append_tool("weather", "{\"city\":\"Oslo\"}", "snow, -3 C")
            .expect("append tool");
        assert_eq!(row.role, Role::Tool);
        assert_eq!(row.tool_name.as_deref(), Some("weather"));
        assert_eq!(row.tool_call.as_deref(), Some("{\"city\":\"Oslo\"}"));
        assert_eq!(row.text, "snow, -3 C");
        assert!(!row.from_voice);

        let recent = store.recent(10).expect("recent");
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0], row);
        assert_eq!(store.message(row.id).expect("message"), Some(row));
    }

    #[test]
    fn unindexed_messages_skip_tools_blanks_and_other_models() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut store = store_in(&dir);

        let first = store.append(Role::User, "first", false).expect("append");
        store
            .append_tool("clock", "{}", "12:00")
            .expect("append tool");
        store.append(Role::Assistant, "   ", false).expect("append");
        let second = store
            .append(Role::Assistant, "second", false)
            .expect("append");
        let third = store.append(Role::User, "third", false).expect("append");

        // `first` gets an embedding for the model being asked about.
        store
            .append_embeddings(first.id, "model-a", &[vec![1.0]])
            .expect("append embeddings");
        // `third` is only indexed for a different model.
        store
            .append_embeddings(third.id, "model-b", &[vec![1.0]])
            .expect("append embeddings");

        let unindexed = store.unindexed_messages("model-a", 10).expect("unindexed");
        assert_eq!(
            unindexed.iter().map(|m| m.id).collect::<Vec<_>>(),
            vec![second.id, third.id]
        );

        let limited = store.unindexed_messages("model-a", 1).expect("unindexed");
        assert_eq!(
            limited.iter().map(|m| m.id).collect::<Vec<_>>(),
            vec![second.id]
        );
    }

    #[test]
    fn embeddings_round_trip_exact_f32s() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut store = store_in(&dir);
        let message = store.append(Role::User, "index me", false).expect("append");

        let vectors = vec![vec![0.5_f32, -1.25, 0.0, 3.5], vec![-0.5, 42.0]];
        store
            .append_embeddings(message.id, "model-a", &vectors)
            .expect("append embeddings");

        let loaded = store.load_vectors("model-a").expect("load vectors");
        assert_eq!(
            loaded,
            vec![
                (message.id, vectors[0].clone()),
                (message.id, vectors[1].clone())
            ]
        );
        assert!(store.load_vectors("model-b").expect("load").is_empty());

        // An empty batch is an error and stores nothing.
        let error = store
            .append_embeddings(message.id, "model-a", &[])
            .expect_err("empty batch should fail");
        assert!(matches!(error, ChatStoreError::Database(_)));
        assert_eq!(store.load_vectors("model-a").expect("load").len(), 2);
    }

    #[test]
    fn index_counts_track_indexed_and_total_messages() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut store = store_in(&dir);

        assert_eq!(store.index_counts("model-a").expect("counts"), (0, 0));

        let first = store.append(Role::User, "first", false).expect("append");
        store
            .append(Role::Assistant, "second", false)
            .expect("append");
        store.append(Role::Assistant, "  ", false).expect("append");
        store.append_tool("clock", "{}", "12:00").expect("tool");

        assert_eq!(store.index_counts("model-a").expect("counts"), (0, 2));

        store
            .append_embeddings(first.id, "model-a", &[vec![1.0]])
            .expect("append embeddings");
        assert_eq!(store.index_counts("model-a").expect("counts"), (1, 2));
        assert_eq!(store.index_counts("model-b").expect("counts"), (0, 2));
    }

    #[test]
    fn previous_message_skips_tool_rows() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut store = store_in(&dir);

        let user = store.append(Role::User, "ask", false).expect("append");
        store
            .append(Role::Assistant, "calling a tool", false)
            .expect("append");
        store
            .append_tool("clock", "{}", "12:00")
            .expect("append tool");
        let assistant = store
            .append(Role::Assistant, "the time is noon", false)
            .expect("append");

        let previous = store
            .previous_message(assistant.id)
            .expect("previous")
            .expect("some");
        assert_eq!(previous.role, Role::Assistant);
        assert_eq!(previous.text, "calling a tool");

        assert_eq!(store.previous_message(user.id).expect("previous"), None);
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
        assert!(all.iter().all(|m| m.tool_name.is_none()));

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
