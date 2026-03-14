use crate::capture;
use crate::db::Database;
use crate::scanner;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Clone, serde::Serialize)]
pub struct TrackerStatus {
    pub running: bool,
    pub game_active: bool,
    pub captured: u32,
    pub skipped: u32,
    pub last_event: String,
}

pub struct Tracker {
    running: Arc<AtomicBool>,
    status: Arc<Mutex<TrackerStatus>>,
    screenshots_dir: PathBuf,
    db: Arc<Database>,
}

enum CaptureResult {
    Ok { path: PathBuf, tag: String },
    Fail(String),
    GameEnded,
    NoGame,
}

impl Tracker {
    pub fn new(screenshots_dir: PathBuf, db: Arc<Database>) -> Self {
        Tracker {
            running: Arc::new(AtomicBool::new(false)),
            status: Arc::new(Mutex::new(TrackerStatus {
                running: false,
                game_active: false,
                captured: 0,
                skipped: 0,
                last_event: "Bekliyor".into(),
            })),
            screenshots_dir,
            db,
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    pub async fn get_status(&self) -> TrackerStatus {
        self.status.lock().await.clone()
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    pub fn start(&self) -> tokio::task::JoinHandle<()> {
        let running = self.running.clone();
        let status = self.status.clone();
        let ss_dir = self.screenshots_dir.clone();
        let db = self.db.clone();

        running.store(true, Ordering::Relaxed);

        tokio::spawn(async move {
            let mut was_game_active = false;

            {
                let mut s = status.lock().await;
                s.running = true;
                s.captured = 0;
                s.skipped = 0;
                s.last_event = "Tracker baslatildi".into();
            }

            while running.load(Ordering::Relaxed) {
                // Ayarlari DB'den oku
                let interval_secs: u64 = db.get_setting("capture_interval")
                    .ok().flatten()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(5);
                let capture_mode = db.get_setting("capture_mode")
                    .ok().flatten()
                    .unwrap_or_else(|| "background".into());

                let result = do_capture(was_game_active, &ss_dir, &capture_mode);

                match result {
                    CaptureResult::Ok { path, tag } => {
                        was_game_active = true;
                        let fname = path.file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();
                        let mut s = status.lock().await;
                        s.game_active = true;
                        s.captured += 1;
                        s.last_event = format!("[{}] #{} {}", tag, s.captured, fname);
                    }
                    CaptureResult::Fail(e) => {
                        was_game_active = true;
                        let mut s = status.lock().await;
                        s.game_active = true;
                        s.skipped += 1;
                        s.last_event = format!("Hata: {}", e);
                    }
                    CaptureResult::GameEnded => {
                        was_game_active = false;
                        {
                            let mut s = status.lock().await;
                            s.game_active = false;
                            s.last_event = "Oyun kapandi, tarama basliyor...".into();
                        }
                        run_scan_pipeline(&ss_dir, &db).await;
                        {
                            let mut s = status.lock().await;
                            s.last_event = "Tarama tamamlandi, oyun bekleniyor...".into();
                        }
                    }
                    CaptureResult::NoGame => {
                        let mut s = status.lock().await;
                        s.game_active = false;
                        s.last_event = "LoL bekleniyor...".into();
                    }
                }

                tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
            }

            {
                let mut s = status.lock().await;
                s.last_event = "Durduruluyor, son tarama...".into();
            }
            run_scan_pipeline(&ss_dir, &db).await;
            {
                let mut s = status.lock().await;
                s.running = false;
                s.last_event = "Tracker durduruldu".into();
            }
        })
    }
}

fn do_capture(was_game_active: bool, ss_dir: &PathBuf, capture_mode: &str) -> CaptureResult {
    let hwnd = match capture::find_lol_window() {
        Some(h) => h,
        None => {
            return if was_game_active {
                CaptureResult::GameEnded
            } else {
                CaptureResult::NoGame
            };
        }
    };

    match capture_mode {
        "foreground" => {
            // Eski yontem: pencereyi one getir, yakala, geri koy
            let is_fg = capture::is_lol_foreground(hwnd);
            let prev = if !is_fg {
                capture::bring_to_front(hwnd)
            } else {
                None
            };

            let result = capture::capture_chat_foreground(hwnd, ss_dir);

            if !is_fg {
                if let Some(p) = prev {
                    capture::restore_window(p);
                }
            }

            match result {
                Ok(path) => CaptureResult::Ok {
                    path,
                    tag: if is_fg { "EKRAN" } else { "ALTAB" }.into(),
                },
                Err(e) => CaptureResult::Fail(e),
            }
        }
        _ => {
            // Background: PrintWindow ile pencereyi one getirmeden yakala
            match capture::capture_chat_background(hwnd, ss_dir) {
                Ok(path) => CaptureResult::Ok {
                    path,
                    tag: "ARKA".into(),
                },
                Err(e) => CaptureResult::Fail(e),
            }
        }
    }
}

async fn run_scan_pipeline(ss_dir: &PathBuf, db: &Arc<Database>) {
    let mut entries: Vec<PathBuf> = match std::fs::read_dir(ss_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().is_some_and(|ext| ext == "png")
                    && !p.file_stem().unwrap_or_default().to_string_lossy().ends_with("_scanned")
            })
            .collect(),
        Err(_) => return,
    };
    entries.sort();

    if entries.is_empty() {
        return;
    }

    let model = db.get_setting("codex_model").ok().flatten();

    for (i, path) in entries.iter().enumerate() {
        let label = format!("{}", i + 1);
        match scanner::scan_screenshot(path, &label, model.as_deref()).await {
            Ok(messages) => {
                let _ = scanner::flush_to_db(db, messages);
                let new_name = path.with_file_name(format!(
                    "{}_scanned.png",
                    path.file_stem().unwrap_or_default().to_string_lossy()
                ));
                let _ = std::fs::rename(path, new_name);
            }
            Err(e) => eprintln!("Scan hatasi {}: {e}", path.display()),
        }
    }

    let _ = scanner::merge_similar_usernames(db, 0.82);
}
