# lol-chatlogger

League of Legends oyun chat'ini ekran goruntulerinden toplayip oyuncu bazli dosyalara yazan Python araci.

## Ozellikler
- Oyun acikken belirli araliklarla chat bolgesinin ekran goruntusunu alir
- Goruntuleri 4x4 grid paketlere cevirir
- Paketleri Gemini ile OCR/ayiklama icin tarar
- Mesajlari oyuncu bazinda `chat_db` altinda `.json` ve `.txt` olarak kaydeder
- Benzer kullanici adlarini birlestirir

## Gereksinimler
- Windows
- Python 3.10+
- League of Legends istemcisi
- Gemini API key

## Kurulum
```bash
pip install -r requirements.txt
```

PowerShell:
```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

## Kullanim
Normal takip:
```bash
python lol_tracker.py
```

Sadece paketle + tara:
```bash
python lol_tracker.py --scan-only
```

Kullanicilari listele:
```bash
python lol_tracker.py --list
```

Belirli kullaniciyi goster:
```bash
python lol_tracker.py --user "OyuncuAdi"
```

## Notlar
- `lol_chat_screenshots/`, `lol_chat_packs/` ve `chat_db/` klasorleri uretilen veridir; `.gitignore` ile repoya alinmaz.
- Varsayilan olarak API key `GEMINI_API_KEY` (alternatif: `GOOGLE_API_KEY`) ortam degiskeninden okunur.

