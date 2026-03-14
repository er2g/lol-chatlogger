<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";

  interface ChatMessage {
    id: number;
    username: string;
    message: string;
    pack: string;
    scanned_at: string;
  }

  let { username }: { username: string } = $props();
  let messages = $state<ChatMessage[]>([]);
  let loading = $state(true);

  async function load() {
    loading = true;
    try {
      messages = await invoke<ChatMessage[]>("get_user_messages", { username });
    } catch (e) {
      console.error(e);
    }
    loading = false;
  }

  $effect(() => {
    if (username) load();
  });
</script>

<div class="chat-header">
  <h2>{username}</h2>
  <span class="msg-count">{messages.length} mesaj</span>
</div>

{#if loading}
  <div class="loading">Yukleniyor...</div>
{:else}
  <div class="messages">
    {#each messages as msg, i}
      <div class="msg">
        <span class="idx">{String(i + 1).padStart(4, "0")}</span>
        <span class="pack">P{msg.pack}</span>
        <span class="text">{msg.message}</span>
      </div>
    {/each}
  </div>
{/if}

<style>
  .chat-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #1e1e2e;
  }

  h2 {
    margin: 0;
    color: #c89b3c;
    font-size: 20px;
  }

  .msg-count {
    font-size: 13px;
    color: #5a5a6a;
  }

  .loading {
    text-align: center;
    padding: 40px;
    color: #5a5a6a;
  }

  .messages {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .msg {
    display: flex;
    align-items: baseline;
    gap: 10px;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 13px;
    font-family: "Cascadia Code", "Fira Code", monospace;
  }

  .msg:hover {
    background: #14141e;
  }

  .idx {
    color: #3a3a4a;
    font-size: 11px;
    flex-shrink: 0;
  }

  .pack {
    color: #c89b3c88;
    font-size: 11px;
    min-width: 32px;
    flex-shrink: 0;
  }

  .text {
    color: #d4d4e4;
  }
</style>
