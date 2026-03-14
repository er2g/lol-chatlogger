<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";

  interface ChatMessage {
    id: number;
    username: string;
    message: string;
    pack: string;
    scanned_at: string;
  }

  let { onSelect }: { onSelect: (u: string) => void } = $props();
  let query = $state("");
  let results = $state<ChatMessage[]>([]);
  let showResults = $state(false);
  let timer: ReturnType<typeof setTimeout>;

  function handleInput() {
    clearTimeout(timer);
    if (query.length < 2) {
      results = [];
      showResults = false;
      return;
    }
    timer = setTimeout(async () => {
      try {
        results = await invoke<ChatMessage[]>("search_messages", { query });
        showResults = results.length > 0;
      } catch {
        results = [];
      }
    }, 300);
  }

  function select(username: string) {
    query = "";
    results = [];
    showResults = false;
    onSelect(username);
  }
</script>

<div class="search-wrapper">
  <input
    type="text"
    placeholder="Oyuncu veya mesaj ara..."
    bind:value={query}
    oninput={handleInput}
    onfocus={() => { if (results.length) showResults = true; }}
    onblur={() => setTimeout(() => showResults = false, 200)}
  />

  {#if showResults}
    <div class="dropdown">
      {#each results.slice(0, 15) as r}
        <button class="result" onclick={() => select(r.username)}>
          <span class="r-user">{r.username}</span>
          <span class="r-msg">{r.message}</span>
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  .search-wrapper {
    position: relative;
    margin-bottom: 16px;
  }

  input {
    width: 100%;
    padding: 10px 14px;
    background: #14141e;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    color: #d4d4e4;
    font-size: 14px;
    outline: none;
    transition: border-color 0.15s;
  }

  input::placeholder { color: #3a3a4a; }
  input:focus { border-color: #c89b3c44; }

  .dropdown {
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background: #14141e;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    margin-top: 4px;
    max-height: 300px;
    overflow-y: auto;
    z-index: 100;
    box-shadow: 0 8px 24px #00000080;
  }

  .result {
    display: flex;
    gap: 10px;
    padding: 8px 12px;
    width: 100%;
    background: none;
    border: none;
    border-bottom: 1px solid #1a1a28;
    cursor: pointer;
    text-align: left;
    color: inherit;
  }

  .result:hover { background: #1a1a28; }
  .result:last-child { border-bottom: none; }

  .r-user {
    color: #c89b3c;
    font-size: 13px;
    font-weight: 600;
    flex-shrink: 0;
    min-width: 120px;
  }

  .r-msg {
    color: #8888a0;
    font-size: 13px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
