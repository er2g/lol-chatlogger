use screenshots::Screen;
use std::path::{Path, PathBuf};

/// Birincil ekranin goruntusu alir, PNG olarak kaydeder
pub fn capture_and_save(dir: &Path, prefix: &str) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("Klasor olusturulamadi: {e}"))?;

    let screens = Screen::all().map_err(|e| format!("Ekranlar alinamadi: {e}"))?;
    let primary = screens.first().ok_or("Ekran bulunamadi")?;
    let img = primary.capture().map_err(|e| format!("Ekran goruntusu alinamadi: {e}"))?;

    let ts = chrono::Local::now().format("%Y%m%d_%H%M%S_%3f");
    let filename = format!("{}_{}.png", prefix, ts);
    let path = dir.join(&filename);

    // screenshots crate RgbaImage (image 0.24) dondurur, save() destekliyor
    img.save(&path).map_err(|e| format!("Kayit hatasi: {e}"))?;
    Ok(path)
}
