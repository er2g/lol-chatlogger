"""
chat_db/ icindeki bozuk dosyalari duzeltir.
"KOCAMAN_PIPIMVAR__Vladimir_.json" -> "KOCAMAN_PIPIMVAR.json" seklinde birlestirir.
Calistirmadan once chat_db/ klasorunu yanina koy.
"""

import json, re, shutil
from pathlib import Path

DB_DIR = Path("chat_db")
DEDUP_WINDOW = 5

SYSTEM_PATTERNS = [
    r"mücevherli eldiven", r"kristali kazandın", r"nektar içiyor",
    r"bağlantısı koptu", r"kazandın!", r"hissetti", r"geri döndü",
    r"^https?://", r"nitelik$",
]
_system_re = re.compile("|".join(SYSTEM_PATTERNS), re.IGNORECASE)

def is_system(text): return bool(_system_re.search(text))
def clean_username(name): return re.sub(r'\s*\(.*?\)\s*$', '', name).strip()
def sanitize(name): return "".join(c if c.isalnum() or c in " ._-" else "_" for c in name).strip()

def write_txt(username, messages, path):
    lines = [f"=== {username} ===\n", f"Toplam mesaj: {len(messages)}\n", "="*50+"\n\n"]
    for i, msg in enumerate(messages, 1):
        lines.append(f"[{i:04d}] [Paket:{msg.get('pack','?'):>4}]  {msg['message']}\n")
    path.write_text("".join(lines), encoding="utf-8")

# Tum json dosyalarini oku, username'leri temizle, birlestir
merged = {}  # clean_username -> [messages]

for f in DB_DIR.glob("*.json"):
    data = json.loads(f.read_text(encoding="utf-8"))
    if not data:
        continue
    raw_name  = data[0].get("username", f.stem)
    clean     = clean_username(raw_name)
    merged.setdefault(clean, []).extend(data)

# Siralama: pack numarasina gore
for uname, msgs in merged.items():
    msgs.sort(key=lambda m: (int(m.get("pack", 0)) if str(m.get("pack","0")).isdigit() else 0))
    for m in msgs:
        m["username"] = uname

    # Sistem mesajlarini filtrele
    msgs = [m for m in msgs if not is_system(m.get("message",""))]

    # Sliding window dedup
    deduped = []
    for m in msgs:
        pack_num = int(m.get("pack", 0)) if str(m.get("pack","0")).isdigit() else 0
        recent = [x for x in deduped if abs(pack_num - (int(x.get("pack",0)) if str(x.get("pack","0")).isdigit() else 0)) <= DEDUP_WINDOW]
        if not any(x["message"] == m["message"] for x in recent):
            deduped.append(m)
    merged[uname] = deduped

# Eski dosyalari yedekle
backup = DB_DIR / "_backup"
backup.mkdir(exist_ok=True)
for f in DB_DIR.glob("*.json"):
    shutil.copy(f, backup / f.name)
for f in DB_DIR.glob("*.txt"):
    shutil.copy(f, backup / f.name)
print(f"Yedek: {backup}/")

# Temiz dosyalari yaz
for f in DB_DIR.glob("*.json"): f.unlink()
for f in DB_DIR.glob("*.txt"):  f.unlink()

for uname, msgs in merged.items():
    safe = sanitize(uname)
    jp = DB_DIR / f"{safe}.json"
    tp = DB_DIR / f"{safe}.txt"
    jp.write_text(json.dumps(msgs, indent=2, ensure_ascii=False), encoding="utf-8")
    write_txt(uname, msgs, tp)
    print(f"  {uname:40s} -> {len(msgs):4d} mesaj")

print(f"\nTamamlandi. {len(merged)} kullanici.")
