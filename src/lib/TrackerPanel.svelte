<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";

  interface TrackerStatus {
    running: boolean;
    game_active: boolean;
    captured: number;
    skipped: number;
    last_event: string;
  }

  let status = $state<TrackerStatus>({
    running: false,
    game_active: false,
    captured: 0,
    skipped: 0,
    last_event: "Bekliyor",
  });

  let pollTimer: ReturnType<typeof setInterval>;

  async function refresh() {
    try {
      status = await invoke<TrackerStatus>("get_tracker_status");
    } catch {}
  }

  async function start() {
    await invoke("start_tracker");
    await refresh();
    pollTimer = setInterval(refresh, 2000);
  }

  async function stop() {
    await invoke("stop_tracker");
    clearInterval(pollTimer);
    // Bir süre sonra son durumu al
    setTimeout(refresh, 3000);
  }

  // Sayfa açılınca durumu kontrol et
  refresh().then(() => {
    if (status.running) {
      pollTimer = setInterval(refresh, 2000);
    }
  });
</script>

<div class="tracker">
  <div class="header-row">
    <h2>Oyun Takip</h2>
    {#if status.running}
      <button class="btn stop" onclick={stop}>Durdur</button>
    {:else}
      <button class="btn start" onclick={start}>Basla</button>
    {/if}
  </div>

  <p class="desc">
    LoL acikken her 5 saniyede chat bolgesinin ekran goruntusunu alir.
    Oyun kapaninca otomatik tarar.
  </p>

  <div class="status-card" class:active={status.running}>
    <div class="row">
      <span class="label">Durum</span>
      <span class="value" class:running={status.running} class:stopped={!status.running}>
        {status.running ? "Calisiyor" : "Durmus"}
      </span>
    </div>
    <div class="row">
      <span class="label">Oyun</span>
      <span class="value" class:game-on={status.game_active}>
        {status.game_active ? "AKTIF" : "Bekleniyor"}
      </span>
    </div>
    <div class="row">
      <span class="label">SS Alindi</span>
      <span class="value num">{status.captured}</span>
    </div>
    <div class="row">
      <span class="label">Atlanan</span>
      <span class="value num">{status.skipped}</span>
    </div>
    <div class="event">
      {status.last_event}
    </div>
  </div>

  {#if status.running}
    <div class="pulse-bar">
      <div class="pulse"></div>
    </div>
  {/if}
</div>

<style>
  .tracker { max-width: 500px; }

  .header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  h2 { color: #c89b3c; font-size: 20px; margin: 0; }

  .desc { color: #5a5a6a; font-size: 13px; margin: 0 0 16px; }

  .btn {
    padding: 8px 24px;
    border-radius: 6px;
    border: none;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    transition: all 0.15s;
  }

  .btn.start {
    background: #1a4a2a;
    color: #4ade80;
  }
  .btn.start:hover { background: #1e5a32; }

  .btn.stop {
    background: #4a1a1a;
    color: #f87171;
  }
  .btn.stop:hover { background: #5a2020; }

  .status-card {
    background: #14141e;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 16px;
    transition: border-color 0.3s;
  }

  .status-card.active {
    border-color: #c89b3c44;
  }

  .row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #1a1a28;
  }

  .row:last-of-type { border-bottom: none; }

  .label { color: #6a6a7a; font-size: 13px; }

  .value { font-size: 13px; font-weight: 600; }
  .value.running { color: #4ade80; }
  .value.stopped { color: #6a6a7a; }
  .value.game-on { color: #c89b3c; }
  .value.num { color: #e0e0f0; }

  .event {
    margin-top: 12px;
    padding: 10px 12px;
    background: #0c0c14;
    border-radius: 6px;
    font-family: "Cascadia Code", "Fira Code", monospace;
    font-size: 12px;
    color: #8888a0;
    word-break: break-all;
  }

  .pulse-bar {
    margin-top: 12px;
    height: 3px;
    background: #1e1e2e;
    border-radius: 2px;
    overflow: hidden;
  }

  .pulse {
    width: 30%;
    height: 100%;
    background: #c89b3c;
    border-radius: 2px;
    animation: slide 1.5s ease-in-out infinite;
  }

  @keyframes slide {
    0% { transform: translateX(-100%); }
    50% { transform: translateX(233%); }
    100% { transform: translateX(-100%); }
  }
</style>
