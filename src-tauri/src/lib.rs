mod capture;
mod commands;
mod db;
mod scanner;

use commands::AppState;
use db::Database;
use std::sync::Arc;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let data_dir = app.path().app_data_dir().expect("app data dir");
            std::fs::create_dir_all(&data_dir).expect("create data dir");

            let db_path = data_dir.join("chat.db");
            let db = Database::new(&db_path).expect("database init");

            // Screenshots klasoru
            let ss_dir = data_dir.join("screenshots");
            std::fs::create_dir_all(&ss_dir).ok();

            app.manage(AppState {
                db: Arc::new(db),
                data_dir,
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_users,
            commands::get_user_messages,
            commands::search_messages,
            commands::get_stats,
            commands::scan_screenshot,
            commands::scan_directory,
            commands::merge_usernames,
            commands::get_setting,
            commands::set_setting,
            commands::get_data_dir,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
