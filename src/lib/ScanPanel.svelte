<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { open } from "@tauri-apps/plugin-dialog";

  let { onDone }: { onDone: () => void } = $props();
  let scanning = $state(false);
  let status = $state("");
  let dataDir = $state("");
  let selectedFiles = $state<string[]>([]);

  async function init() {
    dataDir = await invoke<string>("get_data_dir");
  }

  async function pickFiles() {
    const result = await open({
      multiple: true,
      filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "bmp"] }],
    });
    if (result) {
      selectedFiles = Array.isArray(result) ? result.map(String) : [String(result)];
    }
  }

  async function scanSelected() {
    if (selectedFiles.length === 0) return;
    scanning = true;
    let totalMsgs = 0;

    for (let i = 0; i < selectedFiles.length; i++) {
      status = `Taraniyor: ${i + 1}/${selectedFiles.length}`;
      try {
        const msgs = await invoke<number>("scan_screenshot", {
          imagePath: selectedFiles[i],
          packLabel: String(i + 1),
        });
        totalMsgs += msgs;
      } catch (e) {
        console.error(`Scan error for ${selectedFiles[i]}:`, e);
      }
    }

    status = `Tamamlandi! ${totalMsgs} yeni mesaj, ${selectedFiles.length} gorsel tarandi.`;
    scanning = false;
    selectedFiles = [];
    setTimeout(onDone, 1500);
  }

  async function scanDirectory() {
    scanning = true;
    status = "Screenshots klasoru taraniyor...";

    try {
      const result = await invoke<{ scanned: number; messages: number }>("scan_directory");
      status = `Tamamlandi! ${result.messages} yeni mesaj, ${result.scanned} gorsel tarandi.`;
    } catch (e) {
      status = `Hata: ${e}`;
    }

    scanning = false;
    setTimeout(onDone, 1500);
  }

  init();
</script>

<div class="panel">
  <h2>Gorsel Tara</h2>

  <div class="section">
    <h3>Dosya Sec</h3>
    <p class="desc">LoL chat screenshot'larini secin, Codex CLI ile taranacak.</p>

    <div class="actions">
      <button class="btn primary" onclick={pickFiles} disabled={scanning}>
        Dosya Sec
      </button>

      {#if selectedFiles.length > 0}
        <span class="file-count">{selectedFiles.length} dosya secildi</span>
        <button class="btn gold" onclick={scanSelected} disabled={scanning}>
          Tara
        </button>
      {/if}
    </div>
  </div>

  <div class="divider"></div>

  <div class="section">
    <h3>Otomatik Tara</h3>
    <p class="desc">Screenshots klasorundeki tum taranmamis gorselleri isler.</p>
    <p class="path">{dataDir}/screenshots/</p>

    <button class="btn primary" onclick={scanDirectory} disabled={scanning}>
      Klasoru Tara
    </button>
  </div>

  {#if status}
    <div class="status" class:scanning>
      {#if scanning}
        <span class="spinner"></span>
      {/if}
      {status}
    </div>
  {/if}
</div>

<style>
  .panel { max-width: 600px; }

  h2 { color: #c89b3c; font-size: 20px; margin: 0 0 20px; }
  h3 { color: #e0e0f0; font-size: 15px; margin: 0 0 6px; }

  .desc { color: #6a6a7a; font-size: 13px; margin: 0 0 12px; }

  .path {
    font-family: monospace;
    font-size: 12px;
    color: #4a4a5a;
    margin: 0 0 12px;
    word-break: break-all;
  }

  .actions { display: flex; align-items: center; gap: 10px; }
  .file-count { font-size: 13px; color: #8888a0; }
  .divider { height: 1px; background: #1e1e2e; margin: 24px 0; }

  .btn {
    padding: 8px 20px;
    border-radius: 6px;
    border: 1px solid #2a2a3a;
    background: #1a1a28;
    color: #c4c4d4;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.15s;
  }

  .btn:hover:not(:disabled) { background: #222236; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn.primary { background: #1a2838; border-color: #2a4858; color: #6ab0f3; }
  .btn.primary:hover:not(:disabled) { background: #1e3048; }
  .btn.gold { background: #c89b3c22; border-color: #c89b3c44; color: #c89b3c; }
  .btn.gold:hover:not(:disabled) { background: #c89b3c33; }

  .status {
    margin-top: 20px;
    padding: 12px 16px;
    border-radius: 6px;
    background: #14141e;
    border: 1px solid #1e1e2e;
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .status.scanning { border-color: #c89b3c44; }

  .spinner {
    width: 14px; height: 14px;
    border: 2px solid #c89b3c44;
    border-top-color: #c89b3c;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin { to { transform: rotate(360deg); } }
</style>
