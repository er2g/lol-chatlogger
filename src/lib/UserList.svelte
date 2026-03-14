<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";

  interface UserSummary {
    username: string;
    message_count: number;
    last_seen: string;
  }

  let { onSelect }: { onSelect: (u: string) => void } = $props();
  let users = $state<UserSummary[]>([]);
  let loading = $state(true);

  async function load() {
    loading = true;
    try {
      users = await invoke<UserSummary[]>("get_users");
    } catch (e) {
      console.error(e);
    }
    loading = false;
  }

  load();
</script>

{#if loading}
  <div class="loading">Yukleniyor...</div>
{:else if users.length === 0}
  <div class="empty">
    <p>Henuz kayitli oyuncu yok.</p>
    <p class="hint">Tara sekmesinden screenshot'lari tarayin.</p>
  </div>
{:else}
  <div class="grid">
    {#each users as user}
      <button class="user-card" onclick={() => onSelect(user.username)}>
        <div class="name">{user.username}</div>
        <div class="meta">
          <span class="count">{user.message_count} mesaj</span>
        </div>
      </button>
    {/each}
  </div>
{/if}

<style>
  .loading, .empty {
    text-align: center;
    padding: 60px 20px;
    color: #5a5a6a;
  }

  .empty p { margin: 4px 0; }
  .hint { font-size: 13px; }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 10px;
  }

  .user-card {
    background: #14141e;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 14px 16px;
    cursor: pointer;
    text-align: left;
    transition: all 0.15s;
    width: 100%;
  }

  .user-card:hover {
    background: #1a1a28;
    border-color: #c89b3c44;
    transform: translateY(-1px);
  }

  .name {
    color: #e0e0f0;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 6px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .count {
    font-size: 12px;
    color: #c89b3c;
  }
</style>
