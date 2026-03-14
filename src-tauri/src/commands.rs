use crate::db::{ChatMessage, Database, UserSummary};
use crate::scanner;
use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;

pub struct AppState {
    pub db: Arc<Database>,
    pub data_dir: PathBuf,
}

#[tauri::command]
pub fn get_users(state: State<AppState>) -> Result<Vec<UserSummary>, String> {
    state.db.get_users().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_user_messages(state: State<AppState>, username: String) -> Result<Vec<ChatMessage>, String> {
    state.db.get_user_messages(&username).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn search_messages(state: State<AppState>, query: String) -> Result<Vec<ChatMessage>, String> {
    state.db.search_messages(&query).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_stats(state: State<AppState>) -> Result<(i64, i64), String> {
    state.db.get_stats().map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn scan_screenshot(
    state: State<'_, AppState>,
    image_path: String,
    pack_label: String,
) -> Result<usize, String> {
    let path = PathBuf::from(&image_path);
    if !path.exists() {
        return Err(format!("Dosya bulunamadi: {image_path}"));
    }

    let model = state.db.get_setting("codex_model")
        .ok()
        .flatten();

    let messages = scanner::scan_screenshot(
        &path,
        &pack_label,
        model.as_deref(),
    ).await?;

    let db = state.db.clone();
    scanner::flush_to_db(&db, messages)
}

#[tauri::command]
pub async fn scan_directory(
    state: State<'_, AppState>,
) -> Result<ScanResult, String> {
    let ss_dir = state.data_dir.join("screenshots");
    if !ss_dir.exists() {
        return Ok(ScanResult { scanned: 0, messages: 0 });
    }

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&ss_dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|ext| ext == "png")
                && !p.file_stem().unwrap_or_default().to_string_lossy().ends_with("_scanned")
        })
        .collect();
    entries.sort();

    if entries.is_empty() {
        return Ok(ScanResult { scanned: 0, messages: 0 });
    }

    let model = state.db.get_setting("codex_model").ok().flatten();
    let mut total_scanned = 0;
    let mut total_messages = 0;

    for (i, path) in entries.iter().enumerate() {
        let label = format!("{}", i + 1);
        match scanner::scan_screenshot(path, &label, model.as_deref()).await {
            Ok(messages) => {
                let db = state.db.clone();
                let inserted = scanner::flush_to_db(&db, messages).unwrap_or(0);
                total_messages += inserted;
                total_scanned += 1;
                // Rename to _scanned
                let new_name = path.with_file_name(format!(
                    "{}_scanned.png",
                    path.file_stem().unwrap_or_default().to_string_lossy()
                ));
                let _ = std::fs::rename(path, new_name);
            }
            Err(e) => {
                eprintln!("Scan hatasi {}: {e}", path.display());
            }
        }
    }

    // Merge similar usernames
    let _ = scanner::merge_similar_usernames(&state.db, 0.82);

    Ok(ScanResult { scanned: total_scanned, messages: total_messages })
}

#[derive(serde::Serialize)]
pub struct ScanResult {
    pub scanned: usize,
    pub messages: usize,
}

#[tauri::command]
pub fn merge_usernames(state: State<AppState>) -> Result<usize, String> {
    scanner::merge_similar_usernames(&state.db, 0.82)
}

#[tauri::command]
pub fn get_setting(state: State<AppState>, key: String) -> Result<Option<String>, String> {
    state.db.get_setting(&key).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn set_setting(state: State<AppState>, key: String, value: String) -> Result<(), String> {
    state.db.set_setting(&key, &value).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_data_dir(state: State<AppState>) -> String {
    state.data_dir.to_string_lossy().to_string()
}
