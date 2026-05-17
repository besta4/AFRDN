/**
 * api.js — REST + WebSocket client for the Jatayu batch pipeline.
 *
 * All batch endpoints are prefixed with /batch-api/ to keep them
 * completely independent from the real-time fraud detection app.
 */

const BASE = "/batch-api";   // batch pipeline lives under its own prefix

// ── REST helpers ──────────────────────────────────────────────────────────────

export async function uploadCSV(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();  // { task_id, filename }
}

export async function fetchSummary(taskId) {
  const res = await fetch(`${BASE}/summary/${taskId}`);
  if (!res.ok) throw new Error("Failed to fetch summary");
  return res.json();
}

export async function fetchResults(taskId, limit = 1000, offset = 0) {
  const res = await fetch(`${BASE}/results/${taskId}?limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error("Failed to fetch results");
  return res.json();
}

export async function fetchAudit(taskId = null) {
  const url = taskId ? `${BASE}/audit?task_id=${taskId}` : `${BASE}/audit`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch audit");
  return res.json();
}

export async function fetchTasks() {
  const res = await fetch(`${BASE}/tasks`);
  if (!res.ok) throw new Error("Failed to fetch tasks");
  return res.json();
}

export async function fetchIntelligence(taskId) {
  const res = await fetch(`${BASE}/intelligence/${taskId}`);
  if (!res.ok) throw new Error("Failed to fetch intelligence report");
  return res.json();
}

/**
 * streamIntelligence(taskId, callbacks)
 *
 * Connects to /batch-api/intelligence-stream/{task_id} via Server-Sent Events.
 * callbacks:
 *   onToken(text)            — called for each streaming token
 *   onComplete(data)         — called when full result is ready
 *   onCached(data)           — called if result was already cached server-side
 *   onError(message, data?)  — called on error (with optional fallback data)
 *
 * Returns a close() function to abort the stream.
 */
export function streamIntelligence(taskId, { onToken, onComplete, onCached, onError } = {}) {
  const es = new EventSource(`${BASE}/intelligence-stream/${taskId}`);

  es.addEventListener("message", (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }

    switch (msg.type) {
      case "token":
        onToken?.(msg.text);
        break;
      case "complete":
        onComplete?.(msg.payload);
        es.close();
        break;
      case "cached":
        onCached?.(msg.payload);
        es.close();
        break;
      case "error_fallback":
        onError?.("LLM unavailable \u2013 showing pattern metrics.", msg.payload);
        es.close();
        break;
      case "error":
        onError?.(msg.message || "Unknown error");
        es.close();
        break;
    }
  });

  es.addEventListener("error", () => {
    onError?.("SSE connection error");
    es.close();
  });

  return () => es.close();
}

export async function fetchAgentOutputs(taskId) {
  const res = await fetch(`${BASE}/agent-outputs/${taskId}`);
  if (!res.ok) throw new Error("Failed to fetch agent outputs");
  return res.json();
}

// ── WebSocket helper ──────────────────────────────────────────────────────────

/**
 * openProgressSocket(taskId, callbacks)
 *
 * callbacks:
 *   onProgress({ progress, message, processed, total })
 *   onComplete({ fraud_count, processed, total })
 *   onError(message)
 */
export function openProgressSocket(taskId, { onProgress, onComplete, onError }) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/batch-api/ws/${taskId}`);

  ws.addEventListener("message", (e) => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; }

    if (data.status === "ping") return;
    if (data.status === "progress" || data.status === "started") {
      onProgress?.(data);
    } else if (data.status === "complete") {
      onComplete?.(data);
      ws.close();
    } else if (data.status === "error") {
      onError?.(data.message || "Unknown error");
      ws.close();
    }
  });

  ws.addEventListener("error", () => onError?.("WebSocket connection error"));
  ws.addEventListener("close", (e) => {
    if (e.code !== 1000 && e.code !== 1005) {
      onError?.("WebSocket disconnected unexpectedly");
    }
  });

  return ws;
}
