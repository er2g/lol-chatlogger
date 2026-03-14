use screenshots::Screen;
use std::path::{Path, PathBuf};
use windows::Win32::Foundation::{HWND, RECT, LPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetWindowTextW, GetClientRect, GetForegroundWindow,
    SetForegroundWindow, ShowWindow, IsWindow, IsIconic,
    SW_RESTORE,
};
use windows::Win32::Graphics::Gdi::ClientToScreen;
use windows::Win32::Foundation::POINT;

const LOL_TITLE_LOWER: &str = "league of legends (tm) client";

const BASE_W: f64 = 1456.0;
const BASE_H: f64 = 816.0;
const CHAT_LEFT: f64 = 55.0;
const CHAT_TOP: f64 = 255.0;
const CHAT_RIGHT: f64 = 460.0;
const CHAT_BOTTOM: f64 = 425.0;

pub fn find_lol_window() -> Option<HWND> {
    let mut found: Vec<HWND> = Vec::new();

    unsafe {
        let _ = EnumWindows(
            Some(enum_callback),
            LPARAM(&mut found as *mut Vec<HWND> as isize),
        );
    }

    found.into_iter().next()
}

unsafe extern "system" fn enum_callback(hwnd: HWND, lparam: LPARAM) -> windows::core::BOOL {
    let found = &mut *(lparam.0 as *mut Vec<HWND>);
    let mut buf = [0u16; 256];
    let len = GetWindowTextW(hwnd, &mut buf);
    if len > 0 {
        let title = String::from_utf16_lossy(&buf[..len as usize]);
        if title.to_lowercase().contains(LOL_TITLE_LOWER) {
            found.push(hwnd);
        }
    }
    windows::core::BOOL(1)
}

fn get_window_screen_rect(hwnd: HWND) -> Option<(i32, i32, i32, i32)> {
    unsafe {
        let mut rect = RECT::default();
        GetClientRect(hwnd, &mut rect).ok()?;
        let mut pt = POINT { x: 0, y: 0 };
        let _ = ClientToScreen(hwnd, &mut pt);
        let w = rect.right - rect.left;
        let h = rect.bottom - rect.top;
        if w <= 0 || h <= 0 {
            return None;
        }
        Some((pt.x, pt.y, w, h))
    }
}

fn scale_chat_region(win_w: i32, win_h: i32) -> (i32, i32, i32, i32) {
    let sx = win_w as f64 / BASE_W;
    let sy = win_h as f64 / BASE_H;
    (
        (CHAT_LEFT * sx) as i32,
        (CHAT_TOP * sy) as i32,
        (CHAT_RIGHT * sx) as i32,
        (CHAT_BOTTOM * sy) as i32,
    )
}

pub fn bring_to_front(hwnd: HWND) -> Option<HWND> {
    unsafe {
        let prev = GetForegroundWindow();
        let _ = SetForegroundWindow(hwnd);
        if IsIconic(hwnd).as_bool() {
            let _ = ShowWindow(hwnd, SW_RESTORE);
        }
        std::thread::sleep(std::time::Duration::from_millis(150));
        Some(prev)
    }
}

pub fn restore_window(prev: HWND) {
    unsafe {
        if IsWindow(Some(prev)).as_bool() {
            let _ = SetForegroundWindow(prev);
        }
    }
}

pub fn is_lol_foreground(hwnd: HWND) -> bool {
    unsafe { GetForegroundWindow() == hwnd }
}

pub fn capture_chat(hwnd: HWND, save_dir: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(save_dir).map_err(|e| format!("Klasor olusturulamadi: {e}"))?;

    let (wx, wy, ww, wh) = get_window_screen_rect(hwnd)
        .ok_or("Pencere boyutu alinamadi")?;

    let (cl, ct, cr, cb) = scale_chat_region(ww, wh);
    let chat_x = wx + cl;
    let chat_y = wy + ct;
    let chat_w = cr - cl;
    let chat_h = cb - ct;

    if chat_w <= 0 || chat_h <= 0 {
        return Err("Chat bolgesi boyutu gecersiz".into());
    }

    let screens = Screen::all().map_err(|e| format!("Ekran alinamadi: {e}"))?;
    let screen = screens.first().ok_or("Ekran bulunamadi")?;
    let full = screen.capture().map_err(|e| format!("SS alinamadi: {e}"))?;

    let cropped = screenshots::image::imageops::crop_imm(
        &full,
        chat_x.max(0) as u32,
        chat_y.max(0) as u32,
        chat_w as u32,
        chat_h as u32,
    )
    .to_image();

    let ts = chrono::Local::now().format("%Y%m%d_%H%M%S_%3f");
    let path = save_dir.join(format!("chat_{}.png", ts));
    cropped.save(&path).map_err(|e| format!("Kayit hatasi: {e}"))?;

    Ok(path)
}
