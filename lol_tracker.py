"""
LoL Chat Tracker - v3.0
========================
1. Oyun sirasinda her 5sn chat ss alir
2. Oyun bitince ss'leri 4x4 paketler
3. Paketleri 4 paralel Gemini istegi ile tara
4. Her kullanici icin chat_db/<isim>.txt olusturur
5. Taranan paket: 1.png -> 1_scanned.png (processed.json yok)

Kurulum:
    pip install pillow pywin32 mss google-generativeai

Kullanim:
    python lol_tracker.py                       # normal
    python lol_tracker.py --user "OyuncuAdi"    # profil
    python lol_tracker.py --list                # tum kullanicilar
    python lol_tracker.py --scan-only           # sadece paketle+tara
"""

import os, sys, json, time, ctypes, argparse, re, threading
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed  # pipeline thread icin kalabilir
from datetime import datetime
from pathlib import Path

try:
    import win32gui, win32con
    from PIL import Image
    import mss
    import google.generativeai as genai
except ImportError as e:
    print(f"Eksik kutuphane: {e}")
    print("  pip install pillow pywin32 mss google-generativeai")
    sys.exit(1)

# ╔══════════════════════════════════════════════════╗
# ║                    AYARLAR                        ║
# ╚══════════════════════════════════════════════════╝

def load_dotenv(path: str = ".env"):
    """Minimal .env loader (ek kutuphane gerektirmez)."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].strip()
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


load_dotenv()

API_KEY         = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_PRIMARY  = "gemini-3-flash-preview"
GEMINI_FALLBACK = "gemini-2.5-flash"
BATCH_SIZE      = 20   # tek istekte kac pack gonderilsin (context window limiti)

LOL_TITLE    = "League of Legends (TM) Client"
CHAT_REF     = (55, 255, 460, 425)         # 1456x816 referans
BASE_W       = 1456
BASE_H       = 816
INTERVAL     = 5                           # ss arasi bekleme (sn)
SWITCH_WAIT  = 0.15                        # alt-tab flash suresi (sn)

GRID_COLS    = 4
GRID_ROWS    = 4
PER_PACK     = GRID_COLS * GRID_ROWS       # 16
THUMB_W      = 400
THUMB_H      = 170

SHOTS_DIR    = Path("lol_chat_screenshots")
DONE_DIR     = SHOTS_DIR / "islendi"
PACKS_DIR    = Path("lol_chat_packs")
DB_DIR       = Path("chat_db")

# ╔══════════════════════════════════════════════════╗
# ║                   CAPTURE                         ║
# ╚══════════════════════════════════════════════════╝

def find_lol():
    found = []
    def cb(hwnd, _):
        if LOL_TITLE.lower() in win32gui.GetWindowText(hwnd).lower():
            found.append(hwnd)
    win32gui.EnumWindows(cb, None)
    return found[0] if found else None


def get_window_rect(hwnd):
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    r = win32gui.GetClientRect(hwnd)
    return left, top, r[2], r[3]


def scale_chat(win_w, win_h):
    sx, sy = win_w / BASE_W, win_h / BASE_H
    return (int(CHAT_REF[0]*sx), int(CHAT_REF[1]*sy),
            int(CHAT_REF[2]*sx), int(CHAT_REF[3]*sy))


def bring_to_front(hwnd):
    prev = win32gui.GetForegroundWindow()
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    if win32gui.GetWindowPlacement(hwnd)[1] == win32con.SW_SHOWMINIMIZED:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.BringWindowToTop(hwnd)
    time.sleep(SWITCH_WAIT)
    return prev


def restore_window(prev):
    if prev and win32gui.IsWindow(prev):
        ctypes.windll.user32.SetForegroundWindow(prev)


def capture_chat(hwnd):
    wx, wy, ww, wh = get_window_rect(hwnd)
    if ww == 0 or wh == 0:
        return None
    cl, ct, cr, cb = scale_chat(ww, wh)
    mon = {"left": wx+cl, "top": wy+ct, "width": cr-cl, "height": cb-ct}
    with mss.mss() as sct:
        raw = sct.grab(mon)
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")


# ╔══════════════════════════════════════════════════╗
# ║                    PACKER                         ║
# ╚══════════════════════════════════════════════════╝

def next_pack_number():
    """Hem 1.png hem 1_scanned.png'e bakarak en yuksek numarayi bulur."""
    if not PACKS_DIR.exists():
        return 1
    nums = []
    for f in PACKS_DIR.glob("*.png"):
        stem = f.stem.replace("_scanned", "")
        if stem.isdigit():
            nums.append(int(stem))
    return max(nums, default=0) + 1


def build_grid(images):
    canvas = Image.new("RGB", (GRID_COLS*THUMB_W, GRID_ROWS*THUMB_H), (20, 20, 20))
    for i, img in enumerate(images):
        col, row = i % GRID_COLS, i // GRID_COLS
        canvas.paste(img.resize((THUMB_W, THUMB_H), Image.LANCZOS), (col*THUMB_W, row*THUMB_H))
    return canvas


def pack_screenshots():
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    PACKS_DIR.mkdir(parents=True, exist_ok=True)

    shots = sorted(SHOTS_DIR.glob("*.png"))
    if not shots:
        print("  [PAKET] Islenecek screenshot yok.")
        return

    print(f"  [PAKET] {len(shots)} screenshot paketleniyor...")
    pack_num = next_pack_number()
    chunks   = [shots[i:i+PER_PACK] for i in range(0, len(shots), PER_PACK)]
    total    = 0

    for chunk in chunks:
        images = []
        for p in chunk:
            try:
                images.append(Image.open(p))
            except Exception:
                pass
        if not images:
            continue

        out = PACKS_DIR / f"{pack_num}.png"
        build_grid(images).save(out)
        print(f"    -> {out.name}  ({len(images)} screenshot)")
        pack_num += 1
        total += len(chunk)

        for p in chunk:
            try:
                p.rename(DONE_DIR / p.name)
            except Exception:
                pass

    print(f"  [PAKET] Tamamlandi. {total} screenshot islendi.")


# ╔══════════════════════════════════════════════════╗
# ║                   SCANNER                         ║
# ╚══════════════════════════════════════════════════╝

# Kac pack'i tek bir Gemini isteğine sigdiralim
# Gemini Flash ~1M token context'e sahip, 20 pack rahatça giriyor
BATCH_SIZE = 20

PROMPT_BATCH = """
Sana League of Legends oyunundan alınmış {n} adet ekran görüntüsü paketi gönderiyorum.
Her paket 4x4 grid şeklinde 16 chat ekranı içeriyor ve PACK_1, PACK_2, ... şeklinde etiketlenmiştir.

GÖREVİN:
1. Her paketteki chat mesajlarını soldan sağa, yukarıdan aşağı sırayla oku
2. Tüm paketler boyunca aynı mesajın tekrarlarını tespit et — chat geçmişi kayar,
   yani bir mesaj birden fazla pakette görünebilir. Her benzersiz mesajı SADECE
   ilk göründüğü pakette kaydet, tekrarlarını ATLAT
3. Kullanıcı adlarını tutarlı tut — aynı oyuncu farklı paketlerde farklı görünse bile
   en net okunan halini kullan, tutarlı bir isim belirle
4. SADECE şu JSON formatında yanıt ver, başka hiçbir şey yazma:

[
  {{"pack": "1", "username": "KullaniciAdi", "message": "mesaj icerigi"}},
  {{"pack": "3", "username": "DigerKullanici", "message": "mesaj icerigi"}}
]

KURALLAR:
- Sistem bildirimleri (eşya kazandın, nektar içiyor, bağlantısı koptu, kristal kazandın,
  Sıcağı Hissetti, Geri Döndü, eşya sattı, vb.) KESİNLİKLE EKLEME
- Sadece oyuncuların bizzat yazdığı chat mesajları
- [Genel], [Takım] gibi kanal etiketlerini kullanıcı adına EKLEME
- LoL chat formatı: "OyuncuAdi (SampiyonAdi): mesaj"
  → Username olarak SADECE OyuncuAdi al, (SampiyonAdi) kısmını ALMA
  → Örnek: "KOCAMAN PIPIMVAR (Vladimir): merhaba" → username: "KOCAMAN PIPIMVAR"
- Hiç mesaj yoksa boş array dön: []
- pack alanı string olmalı: "1", "2" gibi
"""


def clean_username(name: str) -> str:
    """'OyuncuAdi (SampiyonAdi)' -> 'OyuncuAdi'"""
    return re.sub(r'\s*\(.*?\)\s*$', '', name).strip()


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in " ._-" else "_" for c in name).strip()


def load_user_data(username: str) -> list:
    path = DB_DIR / f"{sanitize(username)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def save_user_data(username: str, messages: list):
    DB_DIR.mkdir(exist_ok=True)
    path = DB_DIR / f"{sanitize(username)}.json"
    path.write_text(json.dumps(messages, indent=2, ensure_ascii=False), encoding="utf-8")


def write_user_txt(username: str, messages: list):
    path = DB_DIR / f"{sanitize(username)}.txt"
    lines = [
        f"=== {username} ===\n",
        f"Toplam mesaj: {len(messages)}\n",
        f"Son guncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "=" * 50 + "\n\n"
    ]
    for i, msg in enumerate(messages, 1):
        lines.append(f"[{i:04d}] [Paket:{msg.get('pack','?'):>4}]  {msg['message']}\n")
    path.write_text("".join(lines), encoding="utf-8")


def get_unscanned_packs() -> list[Path]:
    if not PACKS_DIR.exists():
        return []
    packs = [f for f in PACKS_DIR.glob("*.png") if not f.stem.endswith("_scanned")]
    return sorted(packs, key=lambda p: int(p.stem) if p.stem.isdigit() else 999999)


def mark_scanned(pack_paths: list[Path]):
    for p in pack_paths:
        new_name = p.parent / f"{p.stem}_scanned.png"
        try:
            p.rename(new_name)
        except Exception as e:
            print(f"  [UYARI] Rename basarisiz {p.name}: {e}")


def parse_gemini_response(raw: str) -> list:
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                raw = part
                break
    return json.loads(raw)


def scan_batch(packs: list[Path], model_primary, model_fallback) -> list[dict]:
    """
    Birden fazla pack'i tek bir Gemini isteğine gonder.
    Her image'dan once metin etiket koy: "=== PACK_N ==="
    Donulen liste: [{"pack": "N", "username": "...", "message": "..."}]
    """
    content = [PROMPT_BATCH.format(n=len(packs))]

    for pack in packs:
        pack_num = pack.stem  # "7", "12" gibi
        content.append(f"=== PACK_{pack_num} ===")
        content.append(Image.open(pack))

    def _call(model):
        resp = model.generate_content(content)
        return parse_gemini_response(resp.text)

    try:
        messages = _call(model_primary)
        print(f"  [OK] {len(packs)} pack -> {len(messages)} benzersiz mesaj")
        return messages
    except Exception as e1:
        print(f"  [FALLBACK] Primary hata: {e1}")
        try:
            messages = _call(model_fallback)
            print(f"  [OK-FB] {len(packs)} pack -> {len(messages)} benzersiz mesaj")
            return messages
        except Exception as e2:
            print(f"  [HATA] Her iki model basarisiz: {e2}")
            return []


# DB yazma thread-safe
_db_lock = threading.Lock()

# Sistem mesajı güvenlik filtresi (Gemini yine de sızdırırsa)
SYSTEM_PATTERNS = [
    r"mücevherli eldiven", r"kristali kazandın", r"nektar içiyor",
    r"bağlantısı koptu", r"kazandın!", r"hissetti", r"geri döndü",
    r"^https?://", r"nitelik$",
]
_system_re = re.compile("|".join(SYSTEM_PATTERNS), re.IGNORECASE)

def is_system_message(text: str) -> bool:
    return bool(_system_re.search(text))


def flush_to_db(messages: list[dict]):
    """Gemini'den gelen temizlenmiş mesaj listesini DB'ye yazar."""
    with _db_lock:
        user_data = {}

        # Pack numarasına göre sıralı ilerle
        sorted_msgs = sorted(messages, key=lambda m: (
            int(m.get("pack", 0)) if str(m.get("pack","0")).isdigit() else 999999
        ))

        for msg in sorted_msgs:
            uname = clean_username(msg.get("username", "").strip())
            text  = msg.get("message", "").strip()
            pack  = str(msg.get("pack", "?"))
            if not uname or not text:
                continue
            if is_system_message(text):
                continue

            if uname not in user_data:
                user_data[uname] = load_user_data(uname)

            user_data[uname].append({
                "username"  : uname,
                "pack"      : pack,
                "message"   : text,
                "scanned_at": datetime.now().isoformat()
            })

        for uname, msgs in user_data.items():
            save_user_data(uname, msgs)
            write_user_txt(uname, msgs)

        return sum(len(v) for v in user_data.values())


def scan_all_packs():
    if not API_KEY:
        print("\n[HATA] API key girilmemis!")
        print("  Ortam degiskeni tanimla: GEMINI_API_KEY")
        print("  Key icin: https://aistudio.google.com/app/apikey\n")
        return

    genai.configure(api_key=API_KEY)
    model_primary  = genai.GenerativeModel(GEMINI_PRIMARY)
    model_fallback = genai.GenerativeModel(GEMINI_FALLBACK)

    DB_DIR.mkdir(exist_ok=True)
    new_packs = get_unscanned_packs()

    if not new_packs:
        print("  [TARAMA] Taranacak yeni paket yok.")
        return

    # BATCH_SIZE'dan fazlaysa birden fazla istek at (context window güvenliği)
    batches = [new_packs[i:i+BATCH_SIZE] for i in range(0, len(new_packs), BATCH_SIZE)]
    print(f"\n  [TARAMA] {len(new_packs)} paket, {len(batches)} batch halinde taranacak...\n")

    total_msgs = 0

    for i, batch in enumerate(batches, 1):
        pack_names = ", ".join(p.stem for p in batch)
        print(f"  [BATCH {i}/{len(batches)}] Pack'ler: {pack_names}")

        messages = scan_batch(batch, model_primary, model_fallback)

        if messages:
            added = flush_to_db(messages)
            total_msgs += added
            mark_scanned(batch)
        else:
            print(f"  [UYARI] Batch {i} bos veya basarisiz, pack'ler _scanned yapilmadi.")

    print(f"\n  [TARAMA] Tamamlandi.")
    print(f"  Toplam yeni mesaj : {total_msgs}")
    print(f"  Taranan pack      : {len(new_packs)}\n")

    # Tarama bittikten sonra benzer kullanici adlarini birlestir
    merge_similar_usernames()



# ╔══════════════════════════════════════════════════╗
# ║            USERNAME FUZZY MERGE                   ║
# ╚══════════════════════════════════════════════════╝

MERGE_THRESHOLD = 0.82   # %82 benzerlik esigi


def username_similarity(a: str, b: str) -> float:
    """Iki kullanici adinin benzerlik oranini dondurur (0.0 - 1.0)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def merge_similar_usernames():
    """
    DB'deki tum kullanici adlarini karsilastirir.
    %82+ benzer olanlari en cok mesaji olan (canonical) isim altinda birlestir,
    diger dosyalari siler.
    """
    if not DB_DIR.exists():
        return

    db_files = list(DB_DIR.glob("*.json"))
    if len(db_files) < 2:
        return

    # username -> (dosya_yolu, mesaj_listesi)
    users = {}
    for f in db_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            real = data[0].get("username", f.stem) if data else f.stem
            users[real] = (f, data)
        except Exception:
            pass

    usernames  = list(users.keys())
    merged_map = {}   # alias -> canonical

    # Union-Find ile grupla
    parent = {u: u for u in usernames}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Daha cok mesaji olan canonical olsun
        if len(users[ra][1]) >= len(users[rb][1]):
            parent[rb] = ra
        else:
            parent[ra] = rb

    # Tum ciftleri karsilastir
    pairs_merged = []
    for i in range(len(usernames)):
        for j in range(i + 1, len(usernames)):
            a, b = usernames[i], usernames[j]
            sim  = username_similarity(a, b)
            if sim >= MERGE_THRESHOLD:
                union(a, b)
                pairs_merged.append((a, b, sim))

    if not pairs_merged:
        return

    print(f"\n  [MERGE] {len(pairs_merged)} benzer kullanici adi bulundu:")
    for a, b, sim in pairs_merged:
        print(f"    '{a}'  <->  '{b}'  (benzerlik: {sim:.0%})")

    # Gruplari olustur: canonical -> [aliases]
    groups = {}
    for u in usernames:
        canon = find(u)
        groups.setdefault(canon, []).append(u)

    merged_count = 0
    for canon, members in groups.items():
        if len(members) < 2:
            continue

        # Canonical'in mesajlarini topla, pack sirasi koru
        all_msgs = []
        for member in members:
            fpath, data = users[member]
            for msg in data:
                msg["username"] = canon   # canonical isimle uzerine yaz
            all_msgs.extend(data)

        # Pack numarasina, sonra scanned_at'e gore sirala
        all_msgs.sort(key=lambda m: (
            int(m.get("pack", 0)) if str(m.get("pack","0")).isdigit() else 999999,
            m.get("scanned_at", "")
        ))

        # Kaydet
        save_user_data(canon, all_msgs)
        write_user_txt(canon, all_msgs)

        # Alias dosyalarini sil (canonical haric)
        for member in members:
            if member == canon:
                continue
            fpath, _ = users[member]
            try:
                fpath.unlink()
                txt = fpath.with_suffix(".txt")
                if txt.exists():
                    txt.unlink()
                print(f"    [MERGE] '{member}' -> '{canon}' birlesti, eski dosya silindi.")
            except Exception as e:
                print(f"    [UYARI] {fpath.name} silinemedi: {e}")

        merged_count += len(members) - 1

    if merged_count:
        print(f"  [MERGE] {merged_count} kullanici birlestirildi.\n")
    else:
        print(f"  [MERGE] Birlestirilecek kullanici bulunamadi.\n")


# ╔══════════════════════════════════════════════════╗
# ║                    QUERY                          ║
# ╚══════════════════════════════════════════════════╝

def show_user(username: str):
    if not DB_DIR.exists():
        print("Veritabani bos. Once tarama yapin.")
        return

    db_files = list(DB_DIR.glob("*.json"))
    match_msgs = None

    # Tam eslesme
    for f in db_files:
        msgs = json.loads(f.read_text(encoding="utf-8"))
        if msgs and msgs[0].get("username", "").lower() == username.lower():
            match_msgs = msgs
            break

    # Yakin eslesme (dosya adina gore)
    if not match_msgs:
        target = sanitize(username).lower()
        for f in db_files:
            if target in f.stem.lower():
                match_msgs = json.loads(f.read_text(encoding="utf-8"))
                break

    if not match_msgs:
        print(f"\n'{username}' bulunamadi.\n")
        list_users()
        return

    real = match_msgs[0].get("username", username)
    print(f"\n{'='*55}")
    print(f"  {real}  ({len(match_msgs)} mesaj)")
    print(f"{'='*55}\n")
    for i, msg in enumerate(match_msgs, 1):
        print(f"  [{i:04d}] [Paket {msg.get('pack','?'):>4}]  {msg['message']}")
    print()


def list_users():
    if not DB_DIR.exists():
        print("Veritabani bos.")
        return
    files = sorted(DB_DIR.glob("*.json"))
    if not files:
        print("Kayitli kullanici yok.")
        return
    print(f"\n{'Kullanici':<45} {'Mesaj':>6}")
    print("-" * 53)
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        real = data[0].get("username", f.stem) if data else f.stem
        print(f"  {real:<43} {len(data):>6}")
    print()


# ╔══════════════════════════════════════════════════╗
# ║               POST-GAME PIPELINE                  ║
# ╚══════════════════════════════════════════════════╝

_pipeline_lock    = threading.Lock()
_pipeline_running = threading.Event()


def run_pipeline(label=""):
    """Arka plan thread'inde paketle + tara + kaydet."""
    def _work():
        if not _pipeline_lock.acquire(blocking=False):
            print(f"  [PIPELINE] Zaten calisıyor, atlanıyor ({label})")
            return
        _pipeline_running.set()
        try:
            tag = f"[{label}]" if label else "[PIPELINE]"
            print(f"\n{tag} {'='*45}")
            print(f"{tag} Pipeline basliyor... (arka planda)")
            print(f"{tag} {'='*45}")
            pack_screenshots()
            scan_all_packs()
            print(f"{tag} Pipeline tamamlandi.\n")
        finally:
            _pipeline_running.clear()
            _pipeline_lock.release()

    t = threading.Thread(target=_work, daemon=True, name=f"pipeline-{label}")
    t.start()
    return t


# ╔══════════════════════════════════════════════════╗
# ║                   MAIN LOOP                       ║
# ╚══════════════════════════════════════════════════╝

def run_tracker():
    SHOTS_DIR.mkdir(exist_ok=True)

    print("╔══════════════════════════════════════════════╗")
    print("║         LoL Chat Tracker  v3.0               ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  Model    : {GEMINI_PRIMARY:<33}║")
    print(f"║  Fallback : {GEMINI_FALLBACK:<33}║")
    print(f"║  Batch    : {BATCH_SIZE} pack/istek{' '*22}║")
    print(f"║  Interval : {INTERVAL} sn{' '*29}║")
    print("╚══════════════════════════════════════════════╝\n")

    # Baslangicta bekleyen is varsa hallet
    shots_left   = list(SHOTS_DIR.glob("*.png"))
    packs_left   = get_unscanned_packs()
    if shots_left or packs_left:
        print(f"[BASLANGIC] {len(shots_left)} islenmemis ss, {len(packs_left)} taranmamis paket bulundu.")
        run_pipeline("BASLANGIC")

    captured    = 0
    skipped     = 0
    was_running = False

    while True:
        try:
            hwnd = find_lol()
            now  = datetime.now().strftime("%H:%M:%S")

            if hwnd is None:
                if was_running:
                    print(f"\n[{now}] Oyun kapandi!")
                    run_pipeline("OYUN SONU")
                    was_running = False
                else:
                    skipped += 1
                    print(f"[{now}] [BEKLE] LoL bekleniyor... ({skipped})")
            else:
                was_running   = True
                is_foreground = (win32gui.GetForegroundWindow() == hwnd)
                prev          = None

                if not is_foreground:
                    prev = bring_to_front(hwnd)

                img = capture_chat(hwnd)

                if not is_foreground and prev:
                    restore_window(prev)

                if img is None:
                    skipped += 1
                    print(f"[{now}] [ATLA] Yakalama basarisiz.")
                else:
                    captured += 1
                    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    path = SHOTS_DIR / f"chat_{ts}.png"
                    img.save(path)
                    tag  = "EKRAN" if is_foreground else "ALTAB"
                    print(f"[{now}] [{captured:04d}] [{tag}] {path.name}")

            time.sleep(INTERVAL)

        except KeyboardInterrupt:
            print(f"\n[DURDURULDU] Kaydedilen: {captured} | Atlanan: {skipped}")
            shots_left = list(SHOTS_DIR.glob("*.png"))
            if shots_left or was_running:
                print("Son pipeline calistiriliyor...")
                t = run_pipeline("KAPAT")
            else:
                t = None
            if _pipeline_running.is_set() or (t and t.is_alive()):
                print("Pipeline bitmesi bekleniyor (tekrar Ctrl+C ile iptal)...")
                try:
                    while _pipeline_running.is_set():
                        time.sleep(0.5)
                except KeyboardInterrupt:
                    print("Pipeline iptal edildi.")
            break
        except Exception as e:
            print(f"[HATA] {e}")
            time.sleep(INTERVAL)


# ╔══════════════════════════════════════════════════╗
# ║                     ENTRY                         ║
# ╚══════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="LoL Chat Tracker v3.0")
    parser.add_argument("--user",      "-u", help="Kullanici profilini goster")
    parser.add_argument("--list",      "-l", action="store_true", help="Tum kullanicilari listele")
    parser.add_argument("--scan-only", "-s", action="store_true", help="Sadece paketle+tara")
    args = parser.parse_args()

    if args.list:
        list_users()
    elif args.user:
        show_user(args.user)
    elif args.scan_only:
        pack_screenshots()
        scan_all_packs()
    else:
        run_tracker()


if __name__ == "__main__":
    main()
