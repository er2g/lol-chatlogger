use crate::db::{ChatMessage, Database};
use chrono::Local;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::NamedTempFile;
use tokio::process::Command;

#[derive(Debug, Deserialize)]
struct RawMessage {
    username: String,
    message: String,
    #[serde(default)]
    pack: Option<String>,
}

const PROMPT: &str = concat!(
    "This image is a League of Legends chat screenshot. ",
    "Extract ONLY player-typed chat messages. ",
    "IGNORE system notifications (item purchases, kills, disconnects, buffs, etc). ",
    "Chat format is \"PlayerName (ChampionName): message\" - use ONLY PlayerName as username, ",
    "do NOT include the (ChampionName) part. ",
    "Do NOT include channel tags like [All] or [Team] in the username. ",
    "Return ONLY a JSON array, nothing else: ",
    "[{\"username\":\"PlayerName\",\"message\":\"text\"}] ",
    "If no messages found return: []"
);

const SYSTEM_PATTERNS: &[&str] = &[
    "kazand", "nektar", "koptu", "hissetti", "dondu",
    "eldiven", "nitelik", "satt",
];

fn is_system_message(text: &str) -> bool {
    let lower = text.to_lowercase();
    SYSTEM_PATTERNS.iter().any(|p| lower.contains(p))
        || text.starts_with("http://")
        || text.starts_with("https://")
}

fn clean_username(name: &str) -> String {
    // "PlayerName (ChampionName)" -> "PlayerName"
    if let Some(idx) = name.rfind('(') {
        name[..idx].trim().to_string()
    } else {
        name.trim().to_string()
    }
}

fn find_codex_bin() -> Option<PathBuf> {
    // Windows: codex.cmd, others: codex
    if cfg!(windows) {
        which::which("codex.cmd")
            .or_else(|_| which::which("codex"))
            .ok()
    } else {
        which::which("codex").ok()
    }
}

fn parse_json_response(raw: &str) -> Vec<RawMessage> {
    let trimmed = raw.trim();

    // Try direct parse
    if let Ok(msgs) = serde_json::from_str::<Vec<RawMessage>>(trimmed) {
        return msgs;
    }

    // Try extracting from code block
    if trimmed.contains("```") {
        for part in trimmed.split("```") {
            let p = part.trim().strip_prefix("json").unwrap_or(part.trim());
            if p.trim_start().starts_with('[') {
                if let Ok(msgs) = serde_json::from_str::<Vec<RawMessage>>(p.trim()) {
                    return msgs;
                }
            }
        }
    }

    // Last resort: find [...] block
    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            if let Ok(msgs) = serde_json::from_str::<Vec<RawMessage>>(&trimmed[start..=end]) {
                return msgs;
            }
        }
    }

    Vec::new()
}

pub async fn scan_screenshot(
    image_path: &Path,
    pack_label: &str,
    model: Option<&str>,
) -> Result<Vec<ChatMessage>, String> {
    let codex = find_codex_bin().ok_or("Codex CLI bulunamadi. npm install -g @openai/codex")?;

    let out_file = NamedTempFile::new().map_err(|e| format!("Temp dosya: {e}"))?;
    let out_path = out_file.path().to_path_buf();

    let mut cmd = Command::new(&codex);
    cmd.arg("exec")
        .arg("--image").arg(image_path)
        .arg("--output-last-message").arg(&out_path)
        .arg("--ephemeral")
        .arg("--skip-git-repo-check")
        .arg("--full-auto");

    if let Some(m) = model {
        cmd.arg("--model").arg(m);
    }

    cmd.arg(PROMPT);

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("Codex calistirilamadi: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Codex hata ({}): {}", output.status, &stderr[..stderr.len().min(200)]));
    }

    let response = tokio::fs::read_to_string(&out_path)
        .await
        .unwrap_or_default();
    let response = if response.trim().is_empty() {
        String::from_utf8_lossy(&output.stdout).to_string()
    } else {
        response
    };

    let raw_msgs = parse_json_response(&response);
    let now = Local::now().to_rfc3339();

    let messages: Vec<ChatMessage> = raw_msgs
        .into_iter()
        .filter(|m| !m.username.trim().is_empty() && !m.message.trim().is_empty())
        .filter(|m| !is_system_message(&m.message))
        .map(|m| ChatMessage {
            id: None,
            username: clean_username(&m.username),
            message: m.message.trim().to_string(),
            pack: m.pack.unwrap_or_else(|| pack_label.to_string()),
            scanned_at: now.clone(),
        })
        .collect();

    Ok(messages)
}

pub fn flush_to_db(db: &Arc<Database>, messages: Vec<ChatMessage>) -> Result<usize, String> {
    let mut inserted = 0;
    for msg in messages {
        match db.insert_message(&msg) {
            Ok(true) => inserted += 1,
            Ok(false) => {} // dedup
            Err(e) => eprintln!("DB insert hatasi: {e}"),
        }
    }
    Ok(inserted)
}

pub fn merge_similar_usernames(db: &Arc<Database>, threshold: f64) -> Result<usize, String> {
    let users = db.get_all_usernames().map_err(|e| e.to_string())?;
    if users.len() < 2 {
        return Ok(0);
    }

    let mut merged = 0;
    let mut skip: std::collections::HashSet<String> = std::collections::HashSet::new();

    for i in 0..users.len() {
        if skip.contains(&users[i].0) { continue; }
        for j in (i + 1)..users.len() {
            if skip.contains(&users[j].0) { continue; }
            let sim = strsim::jaro_winkler(&users[i].0.to_lowercase(), &users[j].0.to_lowercase());
            if sim >= threshold {
                // Daha cok mesaji olan canonical
                let (canonical, alias) = if users[i].1 >= users[j].1 {
                    (&users[i].0, &users[j].0)
                } else {
                    (&users[j].0, &users[i].0)
                };
                if let Ok(n) = db.rename_user(alias, canonical) {
                    if n > 0 {
                        merged += 1;
                        skip.insert(alias.clone());
                    }
                }
            }
        }
    }

    Ok(merged)
}
