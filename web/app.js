const form = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const askBtn = document.getElementById("ask-btn");
const statusEl = document.getElementById("status");
const answerSection = document.getElementById("answer");
const claimsEl = document.getElementById("claims");
const evidencePanel = document.getElementById("evidence-panel");
const evidenceContent = document.getElementById("evidence-content");
const telemetryPanel = document.getElementById("telemetry-panel");
const telemetrySummary = document.getElementById("telemetry-summary");
const telemetryTableBody = document.querySelector("#telemetry-table tbody");
const aggregateContent = document.getElementById("aggregate-content");
const sidebar = document.getElementById("sidebar");
const historyList = document.getElementById("history-list");
const newChatBtn = document.getElementById("new-chat-btn");
const sidebarToggle = document.getElementById("sidebar-toggle");

const STORAGE_KEY = "rc_chat_history";
let evidenceById = {};
let activeId = null;

// --- LocalStorage helpers ---

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistory(history) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
}

function addToHistory(entry) {
  const history = loadHistory();
  history.unshift(entry);
  if (history.length > 50) history.length = 50;
  saveHistory(history);
}

function removeFromHistory(id) {
  const history = loadHistory().filter((h) => h.id !== id);
  saveHistory(history);
  if (activeId === id) resetView();
  renderHistory();
}

// --- Sidebar rendering ---

function timeAgo(ts) {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

function renderHistory() {
  const history = loadHistory();
  historyList.innerHTML = "";
  if (!history.length) {
    historyList.innerHTML = '<div style="padding:1rem;color:var(--muted);font-size:0.8rem;text-align:center;">No history yet.<br>Ask a question to get started.</div>';
    return;
  }
  for (const entry of history) {
    const item = document.createElement("div");
    item.className = "history-item" + (entry.id === activeId ? " active" : "");

    const content = document.createElement("div");
    content.className = "history-item-content";

    const q = document.createElement("div");
    q.className = "history-item-question";
    q.textContent = entry.question;
    content.appendChild(q);

    const meta = document.createElement("div");
    meta.className = "history-item-meta";
    const cost = entry.telemetry ? ` · $${entry.telemetry.total_cost_usd.toFixed(5)}` : "";
    meta.textContent = timeAgo(entry.ts) + cost;
    content.appendChild(meta);

    item.appendChild(content);

    const del = document.createElement("button");
    del.className = "history-item-delete";
    del.innerHTML = "&#x2715;";
    del.title = "Delete";
    del.addEventListener("click", (ev) => {
      ev.stopPropagation();
      removeFromHistory(entry.id);
    });
    item.appendChild(del);

    item.addEventListener("click", () => loadEntry(entry));
    historyList.appendChild(item);
  }
}

function loadEntry(entry) {
  activeId = entry.id;
  evidenceById = {};
  if (entry.evidence) {
    evidenceById = Object.fromEntries(entry.evidence.map((e) => [e.chunk_id, e]));
  }
  questionInput.value = entry.question;
  setStatus("");
  answerSection.hidden = false;
  renderClaims(entry.claims);
  renderTelemetry(entry.telemetry);
  evidencePanel.hidden = true;
  renderHistory();
}

// --- View helpers ---

function setStatus(text, isError = false) {
  statusEl.hidden = !text;
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function resetView() {
  activeId = null;
  questionInput.value = "";
  answerSection.hidden = true;
  evidencePanel.hidden = true;
  telemetryPanel.hidden = true;
  setStatus("");
  evidenceById = {};
  renderHistory();
  questionInput.focus();
}

function renderClaims(claims) {
  claimsEl.innerHTML = "";
  for (const claim of claims) {
    const div = document.createElement("div");
    div.className = "claim";

    const textSpan = document.createElement("span");
    textSpan.textContent = claim.text;
    div.appendChild(textSpan);

    if (claim.unable_to_answer) {
      div.classList.add("unable-to-answer");
      if (claim.reason) {
        const reason = document.createElement("div");
        reason.className = "abstention-reason";
        reason.textContent = claim.reason;
        div.appendChild(reason);
      }
    } else if (claim.supported) {
      for (const cid of claim.source_chunk_ids) {
        const badge = document.createElement("span");
        badge.className = "citation supported";
        badge.textContent = cid;
        badge.title = "Click to view source";
        badge.addEventListener("click", () => showEvidence(cid));
        div.appendChild(badge);
      }
    } else {
      const badge = document.createElement("span");
      badge.className = "citation unsupported";
      badge.textContent = "UNSUPPORTED — no citation";
      div.appendChild(badge);
    }

    claimsEl.appendChild(div);
  }
}

function showEvidence(chunkId) {
  const e = evidenceById[chunkId];
  if (!e) return;
  evidencePanel.hidden = false;
  evidenceContent.innerHTML =
    `<span class="chunk-id">${escapeHtml(e.chunk_id)} (${escapeHtml(e.source)})</span>` +
    escapeHtml(e.text);
  evidencePanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderTelemetry(telemetry) {
  if (!telemetry) {
    telemetryPanel.hidden = true;
    return;
  }
  telemetryPanel.hidden = false;

  const stats = [
    { label: "Total Latency", value: (telemetry.total_latency_ms / 1000).toFixed(1) + "s" },
    { label: "LLM Calls", value: telemetry.total_calls },
    { label: "Tokens In", value: telemetry.total_tokens_in.toLocaleString() },
    { label: "Tokens Out", value: telemetry.total_tokens_out.toLocaleString() },
    { label: "Est. Cost", value: "$" + telemetry.total_cost_usd.toFixed(6) },
  ];

  telemetrySummary.innerHTML = stats
    .map((s) => `<div class="telemetry-stat"><span class="label">${s.label}</span><span class="value">${s.value}</span></div>`)
    .join("");

  telemetryTableBody.innerHTML = telemetry.calls
    .map(
      (c) =>
        `<tr><td>${escapeHtml(c.call_type)}</td><td>${escapeHtml(c.provider)}</td>` +
        `<td>${c.tokens_in.toLocaleString()}</td><td>${c.tokens_out.toLocaleString()}</td>` +
        `<td>${c.latency_ms.toFixed(0)}ms</td><td>$${c.cost_usd.toFixed(6)}</td></tr>`
    )
    .join("");
}

async function loadAggregate() {
  try {
    const resp = await fetch("/api/telemetry/summary");
    if (!resp.ok) return;
    const data = await resp.json();
    if (!data.total_queries) {
      aggregateContent.innerHTML = "<em>No telemetry data yet — ask a question first.</em>";
      return;
    }
    const c1k = data.cost_per_1000_users_monthly;
    aggregateContent.innerHTML =
      `<div class="section-title">Per-query averages (${data.total_queries} queries)</div>` +
      `Tokens: ${data.avg_tokens_per_query.toLocaleString()} · ` +
      `Latency: ${(data.avg_latency_per_query_ms / 1000).toFixed(1)}s · ` +
      `Cost: $${data.avg_cost_per_query_usd.toFixed(6)}` +
      `<div class="section-title">Cost per 1,000 users/month</div>` +
      `5 queries/day: <strong>$${c1k.at_5_queries_per_day.toFixed(2)}</strong> · ` +
      `10/day: <strong>$${c1k.at_10_queries_per_day.toFixed(2)}</strong> · ` +
      `20/day: <strong>$${c1k.at_20_queries_per_day.toFixed(2)}</strong>`;
  } catch {
    aggregateContent.innerHTML = "<em>Could not load aggregate stats.</em>";
  }
}

// --- Events ---

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  askBtn.disabled = true;
  answerSection.hidden = true;
  evidencePanel.hidden = true;
  telemetryPanel.hidden = true;
  setStatus("Searching and synthesizing an answer — this can take 10-30s...");

  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${resp.status}`);
    }

    const data = await resp.json();
    evidenceById = Object.fromEntries(data.evidence.map((e) => [e.chunk_id, e]));

    const entry = {
      id: crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2),
      ts: Date.now(),
      question,
      claims: data.claims,
      evidence: data.evidence,
      telemetry: data.telemetry,
    };
    addToHistory(entry);
    activeId = entry.id;

    setStatus("");
    answerSection.hidden = false;
    renderClaims(data.claims);
    renderTelemetry(data.telemetry);
    loadAggregate();
    renderHistory();
  } catch (err) {
    setStatus(`Error: ${err.message}`, true);
  } finally {
    askBtn.disabled = false;
  }
});

newChatBtn.addEventListener("click", resetView);

sidebarToggle.addEventListener("click", () => {
  sidebar.classList.toggle("collapsed");
});

// --- Init ---

renderHistory();
loadAggregate();
