<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { enable, disable, isEnabled } from "@tauri-apps/plugin-autostart";

  let captureMode = $state("background");
  let captureInterval = $state("5");
  let codexModel = $state("");
  let mergeThreshold = $state("0.82");
  let minimizeToTray = $state(true);
  let autoStart = $state(false);
  let saved = $state(false);
  let dataDir = $state("");

  async function load() {
    const settings: Record<string, [string, (v: string) => void]> = {
      capture_mode: [captureMode, (v) => (captureMode = v)],
      capture_interval: [captureInterval, (v) => (captureInterval = v)],
      codex_model: [codexModel, (v) => (codexModel = v)],
      merge_threshold: [mergeThreshold, (v) => (mergeThreshold = v)],
      minimize_to_tray: ["true", (v) => (minimizeToTray = v === "true")],
    };

    for (const [key, [def, setter]] of Object.entries(settings)) {
      try {
        const val = await invoke<string | null>("get_setting", { key });
        setter(val ?? def);
      } catch {}
    }

    try {
      autoStart = await isEnabled();
    } catch {}

    try {
      dataDir = await invoke<string>("get_data_dir");
    } catch {}
  }

  async function save() {
    const pairs: [string, string][] = [
      ["capture_mode", captureMode],
      ["capture_interval", captureInterval],
      ["codex_model", codexModel],
      ["merge_threshold", mergeThreshold],
      ["minimize_to_tray", minimizeToTray ? "true" : "false"],
    ];

    for (const [key, value] of pairs) {
      await invoke("set_setting", { key, value });
    }

    try {
      if (autoStart) {
        await enable();
      } else {
        await disable();
      }
    } catch (e) {
      console.error("Autostart ayari:", e);
    }

    saved = true;
    setTimeout(() => (saved = false), 2000);
  }

  async function doMerge() {
    const count = await invoke<number>("merge_usernames");
    alert(`${count} kullanici birlestirildi.`);
  }

  load();
</script>

<div class="settings">
  <h2>Ayarlar</h2>

  <!-- Capture Mode -->
  <div class="group">
    <label class="group-title">Yakalama Modu</label>
    <p class="desc">Arka plan: pencereyi one getirmeden yakalar. On plan: pencereyi one getirir, SS alir, geri koyar.</p>
    <div class="radio-group">
      <label class="radio">
        <input type="radio" bind:group={captureMode} value="background" />
        <span>Arka Plan</span>
        <span class="badge rec">Onerilen</span>
      </label>
      <label class="radio">
        <input type="radio" bind:group={captureMode} value="foreground" />
        <span>On Plan (Alt-Tab)</span>
      </label>
    </div>
  </div>

  <!-- Capture Interval -->
  <div class="group">
    <label class="group-title">SS Araligi (saniye)</label>
    <p class="desc">Her kac saniyede bir ekran goruntusu alinacak.</p>
    <div class="input-row">
      <input type="number" bind:value={captureInterval} min="1" max="60" />
      <span class="unit">sn</span>
    </div>
  </div>

  <!-- Codex Model -->
  <div class="group">
    <label class="group-title">Codex Model</label>
    <p class="desc">Bos birakirsan Codex varsayilani kullanilir.</p>
    <input type="text" bind:value={codexModel} placeholder="ornek: o3, gpt-4.1" />
  </div>

  <!-- Merge Threshold -->
  <div class="group">
    <label class="group-title">Isim Birlestirme Esigi</label>
    <p class="desc">Benzer kullanici adlarini birlestirir (0.0 - 1.0). Dusuk = daha agresif birlestirme.</p>
    <div class="input-row">
      <input type="number" bind:value={mergeThreshold} min="0.5" max="1" step="0.01" />
      <button class="btn small" onclick={doMerge}>Simdi Birlestir</button>
    </div>
  </div>

  <div class="divider"></div>

  <!-- System -->
  <div class="group">
    <label class="group-title">Sistem</label>

    <label class="toggle">
      <input type="checkbox" bind:checked={minimizeToTray} />
      <span>Kapatinca tray'e kucult</span>
      <span class="hint">Pencereyi kapatinca uygulama arka planda calismaya devam eder</span>
    </label>

    <label class="toggle">
      <input type="checkbox" bind:checked={autoStart} />
      <span>Bilgisayar aciliginda basla</span>
      <span class="hint">Windows baslangicinda otomatik acilir</span>
    </label>
  </div>

  <!-- Data Dir -->
  <div class="group">
    <label class="group-title">Veri Klasoru</label>
    <p class="path">{dataDir}</p>
  </div>

  <div class="actions">
    <button class="btn save" onclick={save}>Kaydet</button>
    {#if saved}
      <span class="saved-msg">Kaydedildi!</span>
    {/if}
  </div>
</div>

<style>
  .settings { max-width: 560px; }

  h2 { color: #c89b3c; font-size: 20px; margin: 0 0 20px; }

  .group { margin-bottom: 20px; }

  .group-title {
    display: block;
    color: #e0e0f0;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 4px;
  }

  .desc { color: #5a5a6a; font-size: 12px; margin: 0 0 8px; }

  .radio-group { display: flex; flex-direction: column; gap: 6px; }

  .radio {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: #14141e;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    color: #c4c4d4;
    transition: border-color 0.15s;
  }

  .radio:has(input:checked) { border-color: #c89b3c44; background: #c89b3c08; }
  .radio input { accent-color: #c89b3c; }

  .badge {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 600;
  }
  .badge.rec { background: #c89b3c22; color: #c89b3c; }

  .input-row { display: flex; align-items: center; gap: 8px; }

  input[type="number"], input[type="text"] {
    background: #14141e;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    padding: 8px 12px;
    color: #d4d4e4;
    font-size: 13px;
    outline: none;
    width: 200px;
    transition: border-color 0.15s;
  }

  input:focus { border-color: #c89b3c44; }

  .unit { color: #5a5a6a; font-size: 13px; }

  .toggle {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: #14141e;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    cursor: pointer;
    margin-bottom: 6px;
    font-size: 13px;
    color: #c4c4d4;
  }

  .toggle input { accent-color: #c89b3c; }

  .toggle .hint {
    width: 100%;
    font-size: 11px;
    color: #4a4a5a;
    margin-top: -2px;
  }

  .divider { height: 1px; background: #1e1e2e; margin: 24px 0; }

  .path {
    font-family: monospace;
    font-size: 12px;
    color: #4a4a5a;
    word-break: break-all;
    margin: 4px 0 0;
  }

  .actions { display: flex; align-items: center; gap: 12px; margin-top: 24px; }

  .btn {
    padding: 8px 20px;
    border-radius: 6px;
    border: none;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.15s;
  }

  .btn.save { background: #c89b3c; color: #0a0a0f; font-weight: 600; }
  .btn.save:hover { background: #daa844; }

  .btn.small {
    background: #1a1a28;
    border: 1px solid #2a2a3a;
    color: #8888a0;
    padding: 6px 14px;
    font-size: 12px;
  }
  .btn.small:hover { background: #222236; color: #c4c4d4; }

  .saved-msg { color: #4ade80; font-size: 13px; }
</style>
