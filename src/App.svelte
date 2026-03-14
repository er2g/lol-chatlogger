<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import UserList from "./lib/UserList.svelte";
  import ChatView from "./lib/ChatView.svelte";
  import ScanPanel from "./lib/ScanPanel.svelte";
  import SearchBar from "./lib/SearchBar.svelte";

  let currentView = $state<"users" | "chat" | "scan">("users");
  let selectedUser = $state<string>("");
  let stats = $state<{ msgs: number; users: number }>({ msgs: 0, users: 0 });

  async function loadStats() {
    try {
      const [msgs, users] = await invoke<[number, number]>("get_stats");
      stats = { msgs, users };
    } catch {}
  }

  function selectUser(username: string) {
    selectedUser = username;
    currentView = "chat";
  }

  function goHome() {
    currentView = "users";
    selectedUser = "";
    loadStats();
  }

  loadStats();
</script>

<main>
  <header>
    <div class="header-left">
      <button class="logo" onclick={goHome}>LoL Chat Logger</button>
      <span class="stats">{stats.users} oyuncu / {stats.msgs} mesaj</span>
    </div>
    <nav>
      <button class:active={currentView === "users"} onclick={goHome}>Oyuncular</button>
      <button class:active={currentView === "scan"} onclick={() => currentView = "scan"}>Tara</button>
    </nav>
  </header>

  <div class="content">
    <SearchBar onSelect={selectUser} />

    {#if currentView === "users"}
      <UserList onSelect={selectUser} />
    {:else if currentView === "chat"}
      <ChatView username={selectedUser} />
    {:else if currentView === "scan"}
      <ScanPanel onDone={() => { loadStats(); currentView = "users"; }} />
    {/if}
  </div>
</main>

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0a0a0f;
    color: #c4c4d4;
    overflow: hidden;
    height: 100vh;
  }

  :global(*) {
    box-sizing: border-box;
  }

  :global(::-webkit-scrollbar) {
    width: 6px;
  }

  :global(::-webkit-scrollbar-track) {
    background: #12121a;
  }

  :global(::-webkit-scrollbar-thumb) {
    background: #2a2a3a;
    border-radius: 3px;
  }

  main {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    background: #12121a;
    border-bottom: 1px solid #1e1e2e;
    -webkit-app-region: drag;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .logo {
    background: none;
    border: none;
    color: #c89b3c;
    font-size: 18px;
    font-weight: 700;
    cursor: pointer;
    padding: 0;
    -webkit-app-region: no-drag;
  }

  .logo:hover { color: #f0c75e; }

  .stats {
    font-size: 12px;
    color: #5a5a6a;
  }

  nav {
    display: flex;
    gap: 4px;
    -webkit-app-region: no-drag;
  }

  nav button {
    background: #1a1a28;
    border: 1px solid #2a2a3a;
    color: #8888a0;
    padding: 6px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.15s;
  }

  nav button:hover { background: #222236; color: #c4c4d4; }
  nav button.active { background: #c89b3c22; color: #c89b3c; border-color: #c89b3c44; }

  .content {
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px;
  }
</style>
