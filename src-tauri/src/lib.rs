mod capture;
mod commands;
mod db;
mod scanner;
mod tracker;

use commands::AppState;
use db::Database;
use std::sync::Arc;
use tauri::{
    menu::{MenuBuilder, MenuItemBuilder},
    tray::TrayIconBuilder,
    Manager, WindowEvent,
};
use tauri_plugin_autostart::MacosLauncher;
use tracker::Tracker;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_autostart::init(MacosLauncher::LaunchAgent, None))
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

            // System tray
            let show = MenuItemBuilder::with_id("show", "Goster").build(app)?;
            let quit = MenuItemBuilder::with_id("quit", "Kapat").build(app)?;
            let menu = MenuBuilder::new(app).items(&[&show, &quit]).build()?;

            let icon = app.default_window_icon().cloned()
                .expect("window icon");

            let _tray = TrayIconBuilder::new()
                .icon(icon)
                .tooltip("LoL Chat Logger")
                .menu(&menu)
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "show" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                    "quit" => {
                        app.exit(0);
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let tauri::tray::TrayIconEvent::DoubleClick { .. } = event {
                        if let Some(w) = tray.app_handle().get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                })
                .build(app)?;

            Ok(())
        })
        .on_window_event(|window, event| {
            // Kapat butonuna basınca tray'e küçült (çıkma)
            if let WindowEvent::CloseRequested { api, .. } = event {
                let app = window.app_handle();
                let db = &app.state::<AppState>().db;
                let minimize_to_tray = db
                    .get_setting("minimize_to_tray")
                    .ok()
                    .flatten()
                    .unwrap_or_else(|| "true".into());

                if minimize_to_tray == "true" {
                    let _ = window.hide();
                    api.prevent_close();
                }
            }
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
