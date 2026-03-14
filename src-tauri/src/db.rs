use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: Option<i64>,
    pub username: String,
    pub message: String,
    pub pack: String,
    pub scanned_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSummary {
    pub username: String,
    pub message_count: i64,
    pub last_seen: String,
}

pub struct Database {
    pub conn: Mutex<Connection>,
}

impl Database {
    pub fn new(path: &Path) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                message TEXT NOT NULL,
                pack TEXT NOT NULL,
                scanned_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_username ON messages(username);
            CREATE INDEX IF NOT EXISTS idx_pack ON messages(pack);

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            "
        )?;
        Ok(Database { conn: Mutex::new(conn) })
    }

    pub fn insert_message(&self, msg: &ChatMessage) -> Result<bool, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();

        // Dedup: ayni mesaj, ayni kullanici, yakin pack'te varsa ekleme
        let pack_num: i64 = msg.pack.parse().unwrap_or(0);
        let exists: bool = conn.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM messages
                WHERE username = ?1 AND message = ?2
                AND ABS(CAST(pack AS INTEGER) - ?3) <= 5
            )",
            params![msg.username, msg.message, pack_num],
            |row| row.get(0),
        )?;

        if exists {
            return Ok(false);
        }

        conn.execute(
            "INSERT INTO messages (username, message, pack, scanned_at) VALUES (?1, ?2, ?3, ?4)",
            params![msg.username, msg.message, msg.pack, msg.scanned_at],
        )?;
        Ok(true)
    }

    pub fn get_users(&self) -> Result<Vec<UserSummary>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT username, COUNT(*) as cnt, MAX(scanned_at) as last
             FROM messages GROUP BY username ORDER BY cnt DESC"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(UserSummary {
                username: row.get(0)?,
                message_count: row.get(1)?,
                last_seen: row.get(2)?,
            })
        })?;
        rows.collect()
    }

    pub fn get_user_messages(&self, username: &str) -> Result<Vec<ChatMessage>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, username, message, pack, scanned_at
             FROM messages WHERE username = ?1
             ORDER BY CAST(pack AS INTEGER), scanned_at"
        )?;
        let rows = stmt.query_map(params![username], |row| {
            Ok(ChatMessage {
                id: row.get(0)?,
                username: row.get(1)?,
                message: row.get(2)?,
                pack: row.get(3)?,
                scanned_at: row.get(4)?,
            })
        })?;
        rows.collect()
    }

    pub fn search_messages(&self, query: &str) -> Result<Vec<ChatMessage>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let pattern = format!("%{}%", query);
        let mut stmt = conn.prepare(
            "SELECT id, username, message, pack, scanned_at
             FROM messages WHERE message LIKE ?1 OR username LIKE ?1
             ORDER BY scanned_at DESC LIMIT 200"
        )?;
        let rows = stmt.query_map(params![pattern], |row| {
            Ok(ChatMessage {
                id: row.get(0)?,
                username: row.get(1)?,
                message: row.get(2)?,
                pack: row.get(3)?,
                scanned_at: row.get(4)?,
            })
        })?;
        rows.collect()
    }

    pub fn get_all_usernames(&self) -> Result<Vec<(String, i64)>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT username, COUNT(*) FROM messages GROUP BY username"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        rows.collect()
    }

    pub fn rename_user(&self, old: &str, new: &str) -> Result<usize, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE messages SET username = ?1 WHERE username = ?2",
            params![new, old],
        )
    }

    pub fn get_setting(&self, key: &str) -> Result<Option<String>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT value FROM settings WHERE key = ?1")?;
        let mut rows = stmt.query(params![key])?;
        match rows.next()? {
            Some(row) => Ok(Some(row.get(0)?)),
            None => Ok(None),
        }
    }

    pub fn set_setting(&self, key: &str, value: &str) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn get_stats(&self) -> Result<(i64, i64), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let total_msgs: i64 = conn.query_row(
            "SELECT COUNT(*) FROM messages", [], |r| r.get(0)
        )?;
        let total_users: i64 = conn.query_row(
            "SELECT COUNT(DISTINCT username) FROM messages", [], |r| r.get(0)
        )?;
        Ok((total_msgs, total_users))
    }
}
