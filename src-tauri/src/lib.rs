mod capture;
mod commands;
mod db;
mod scanner;
mod tracker;

use commands::AppState;
use db::Database;
use std::sync::Arc;
use tauri::Manager;
use tracker::Tracker;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let data_dir = app.path().app_data_dir().expect("app data dir");
            std::fs::create_dir_all(&data_dir).expect("create data dir");

            let db_path = data_dir.join("chat.db");
            let db = Arc::new(Database::new(&db_path).expect("database init"));

            let ss_dir = data_dir.join("screenshots");
            std::fs::create_dir_all(&ss_dir).ok();

            let tracker = Tracker::new(ss_dir, db.clone());

            app.manage(AppState {
                db,
                data_dir,
                tracker: Arc::new(tokio::sync::Mutex::new(tracker)),
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
            commands::start_tracker,
            commands::stop_tracker,
            commands::get_tracker_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
