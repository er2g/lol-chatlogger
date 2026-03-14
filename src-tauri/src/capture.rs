use screenshots::Screen;
use std::path::{Path, PathBuf};
use windows::Win32::Foundation::{HWND, RECT, LPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetWindowTextW, GetClientRect, GetForegroundWindow,
    SetForegroundWindow, ShowWindow, IsWindow, IsIconic,
    SW_RESTORE,
};
use windows::Win32::Storage::Xps::{PrintWindow, PRINT_WINDOW_FLAGS};
use windows::Win32::Graphics::Gdi::{
    ClientToScreen, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject,
    GetDC, GetDIBits, ReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER, BI_RGB,
    DIB_RGB_COLORS,
};
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

fn get_client_size(hwnd: HWND) -> Option<(i32, i32)> {
    unsafe {
        let mut rect = RECT::default();
        GetClientRect(hwnd, &mut rect).ok()?;
        let w = rect.right - rect.left;
        let h = rect.bottom - rect.top;
        if w <= 0 || h <= 0 { return None; }
        Some((w, h))
    }
}

fn get_window_screen_rect(hwnd: HWND) -> Option<(i32, i32, i32, i32)> {
    unsafe {
        let mut rect = RECT::default();
        GetClientRect(hwnd, &mut rect).ok()?;
        let mut pt = POINT { x: 0, y: 0 };
        let _ = ClientToScreen(hwnd, &mut pt);
        let w = rect.right - rect.left;
        let h = rect.bottom - rect.top;
        if w <= 0 || h <= 0 { return None; }
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

/// Arka planda yakalama: PrintWindow API ile pencereyi öne getirmeden
/// Pencere client area'sini bitmap olarak alır, chat bölgesini kırpar ve kaydeder
pub fn capture_chat_background(hwnd: HWND, save_dir: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(save_dir).map_err(|e| format!("Klasor: {e}"))?;

    let (cw, ch) = get_client_size(hwnd).ok_or("Pencere boyutu alinamadi")?;

    unsafe {
        let hdc_window = GetDC(Some(hwnd));
        if hdc_window.is_invalid() {
            return Err("GetDC basarisiz".into());
        }

        let hdc_mem = CreateCompatibleDC(Some(hdc_window));
        let hbm = CreateCompatibleBitmap(hdc_window, cw, ch);
        let old = SelectObject(hdc_mem, hbm.into());

        // PrintWindow ile arka plandan yakala
        let ok = PrintWindow(hwnd, hdc_mem, PRINT_WINDOW_FLAGS(1)); // PW_CLIENTONLY = 1
        if !ok.as_bool() {
            SelectObject(hdc_mem, old);
            let _ = DeleteObject(hbm.into());
            let _ = DeleteDC(hdc_mem);
            ReleaseDC(Some(hwnd), hdc_window);
            return Err("PrintWindow basarisiz".into());
        }

        // Bitmap verisini oku
        let mut bmi = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: cw,
                biHeight: -ch, // top-down
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut pixels = vec![0u8; (cw * ch * 4) as usize];
        GetDIBits(
            hdc_mem,
            hbm,
            0,
            ch as u32,
            Some(pixels.as_mut_ptr() as *mut _),
            &mut bmi,
            DIB_RGB_COLORS,
        );

        SelectObject(hdc_mem, old);
        let _ = DeleteObject(hbm.into());
        let _ = DeleteDC(hdc_mem);
        ReleaseDC(Some(hwnd), hdc_window);

        // BGRA -> RGBA
        for chunk in pixels.chunks_exact_mut(4) {
            chunk.swap(0, 2);
        }

        // Chat bölgesini kırp
        let (cl, ct, cr, cb) = scale_chat_region(cw, ch);
        let chat_w = (cr - cl).max(1) as u32;
        let chat_h = (cb - ct).max(1) as u32;
        let cl = cl.max(0) as u32;
        let ct = ct.max(0) as u32;

        let mut chat_pixels = Vec::with_capacity((chat_w * chat_h * 4) as usize);
        for y in ct..(ct + chat_h).min(ch as u32) {
            let row_start = (y * cw as u32 * 4 + cl * 4) as usize;
            let row_end = row_start + (chat_w * 4) as usize;
            if row_end <= pixels.len() {
                chat_pixels.extend_from_slice(&pixels[row_start..row_end]);
            }
        }

        let ts = chrono::Local::now().format("%Y%m%d_%H%M%S_%3f");
        let path = save_dir.join(format!("chat_{}.png", ts));

        // screenshots crate'in image'ını kullan
        let img: screenshots::image::RgbaImage =
            screenshots::image::ImageBuffer::from_raw(chat_w, chat_h, chat_pixels)
                .ok_or("Image buffer olusturulamadi")?;
        img.save(&path).map_err(|e| format!("Kayit: {e}"))?;

        Ok(path)
    }
}

/// Ön plana getirerek yakalama (eski yöntem, screen capture)
pub fn capture_chat_foreground(hwnd: HWND, save_dir: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(save_dir).map_err(|e| format!("Klasor: {e}"))?;

    let (wx, wy, ww, wh) = get_window_screen_rect(hwnd).ok_or("Pencere boyutu alinamadi")?;
    let (cl, ct, cr, cb) = scale_chat_region(ww, wh);
    let chat_x = wx + cl;
    let chat_y = wy + ct;
    let chat_w = cr - cl;
    let chat_h = cb - ct;

    if chat_w <= 0 || chat_h <= 0 {
        return Err("Chat bolgesi gecersiz".into());
    }

    let screens = Screen::all().map_err(|e| format!("Ekran: {e}"))?;
    let screen = screens.first().ok_or("Ekran bulunamadi")?;
    let full = screen.capture().map_err(|e| format!("SS: {e}"))?;

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
    cropped.save(&path).map_err(|e| format!("Kayit: {e}"))?;
    Ok(path)
}
