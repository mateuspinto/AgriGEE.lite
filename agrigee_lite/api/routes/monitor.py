"""Live monitoring dashboard: ``GET /monitor``.

A single self-contained HTML page (inline CSS/JS, no build step, no external
assets) that shows what the API is downloading, what finished successfully,
and what failed, updating in place via Server-Sent Events — no page reload.

- ``GET /monitor``        -> the dashboard page.
- ``GET /monitor/state``  -> JSON snapshot (recent log entries + counters),
  used for the initial render.
- ``GET /monitor/stream`` -> ``text/event-stream`` of new log entries as they
  happen. ``?after=<seq>`` replays any entries the client missed between its
  ``/monitor/state`` fetch and opening the stream, so there is no gap.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from agrigee_lite.api._monitor import LogEntry, monitor_state

router = APIRouter(prefix="/monitor", tags=["monitor"])

_KEEPALIVE_SECONDS = 15


@router.get("/state", operation_id="get_monitor_state")
async def get_monitor_state() -> JSONResponse:
    """Recent log entries and per-category counters, for the initial page load."""
    return JSONResponse(monitor_state.snapshot())


def _sse(entry: LogEntry) -> str:
    return f"id: {entry.seq}\ndata: {json.dumps(entry.to_dict())}\n\n"


@router.get("/stream", operation_id="stream_monitor_events")
async def stream_monitor_events(request: Request, after: int = Query(0, ge=0)) -> StreamingResponse:
    """SSE stream of log entries, replaying anything newer than ``after`` first."""

    async def event_generator():
        for entry in monitor_state.entries_after(after):
            yield _sse(entry)

        queue = monitor_state.subscribe()
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_SECONDS)
                    yield _sse(entry)
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            monitor_state.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("", response_class=HTMLResponse, operation_id="monitor_dashboard")
async def monitor_dashboard() -> HTMLResponse:
    return HTMLResponse(_PAGE)


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgriGEE.lite — Monitor</title>
<style>
  :root {
    --bg: #0b1210;
    --panel: #101a17;
    --panel-border: #1e2b26;
    --text: #dfe8e4;
    --muted: #7d9089;
    --success: #35d07f;
    --success-bg: rgba(53, 208, 127, .10);
    --error: #ef5c5c;
    --error-bg: rgba(239, 92, 92, .10);
    --warning: #e8b339;
    --warning-bg: rgba(232, 179, 57, .10);
    --debug: #4fa3f7;
    --debug-bg: rgba(79, 163, 247, .10);
    --info: #9aa8a2;
    --info-bg: rgba(154, 168, 162, .08);
    --accent: #35d07f;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  header {
    display: flex;
    align-items: center;
    gap: .75rem;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--panel-border);
    position: sticky;
    top: 0;
    background: var(--bg);
    z-index: 5;
  }
  header h1 {
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0;
    letter-spacing: .01em;
  }
  header h1 span { color: var(--accent); }
  .conn {
    display: flex;
    align-items: center;
    gap: .4rem;
    font-size: .8rem;
    color: var(--muted);
    margin-left: auto;
  }
  .dot {
    width: .5rem;
    height: .5rem;
    border-radius: 999px;
    background: var(--muted);
  }
  .dot.live { background: var(--success); box-shadow: 0 0 0 3px var(--success-bg); animation: pulse 2s infinite; }
  .dot.down { background: var(--error); box-shadow: 0 0 0 3px var(--error-bg); }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: .45; }
  }
  main {
    max-width: 1080px;
    margin: 0 auto;
    padding: 1.25rem 1.5rem 3rem;
  }
  .stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: .75rem;
    margin-bottom: 1.25rem;
  }
  .stat {
    border: 1px solid var(--panel-border);
    background: var(--panel);
    border-radius: 10px;
    padding: .75rem .9rem;
  }
  .stat .n { font-size: 1.5rem; font-weight: 700; line-height: 1; }
  .stat .l { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin-top: .3rem; }
  .stat.success .n { color: var(--success); }
  .stat.error .n { color: var(--error); }
  .stat.warning .n { color: var(--warning); }
  .stat.debug .n { color: var(--debug); }
  .stat.info .n { color: var(--info); }

  section.card {
    border: 1px solid var(--panel-border);
    background: var(--panel);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1.25rem;
  }
  .card > .card-head {
    display: flex;
    align-items: center;
    gap: .6rem;
    padding: .7rem 1rem;
    border-bottom: 1px solid var(--panel-border);
    font-size: .85rem;
    font-weight: 600;
    color: var(--muted);
  }
  .card-head .actions { margin-left: auto; display: flex; gap: .4rem; }
  button.btn {
    background: transparent;
    border: 1px solid var(--panel-border);
    color: var(--text);
    border-radius: 7px;
    padding: .3rem .65rem;
    font-size: .75rem;
    cursor: pointer;
  }
  button.btn:hover { border-color: var(--accent); color: var(--accent); }
  button.btn.active { background: var(--success-bg); border-color: var(--success); color: var(--success); }

  table.jobs { width: 100%; border-collapse: collapse; font-size: .82rem; }
  table.jobs th, table.jobs td { text-align: left; padding: .5rem 1rem; border-bottom: 1px solid var(--panel-border); }
  table.jobs th { color: var(--muted); font-weight: 500; font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; }
  table.jobs tr:last-child td { border-bottom: none; }
  table.jobs td.id { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted); }
  .badge { display: inline-block; padding: .12rem .5rem; border-radius: 999px; font-size: .72rem; font-weight: 600; }
  .badge.completed { background: var(--success-bg); color: var(--success); }
  .badge.failed { background: var(--error-bg); color: var(--error); }
  .badge.running { background: var(--debug-bg); color: var(--debug); }
  .badge.pending { background: var(--warning-bg); color: var(--warning); }
  .empty { padding: 1.1rem; color: var(--muted); font-size: .85rem; }
  a.dl-btn {
    display: inline-block;
    border: 1px solid var(--success);
    color: var(--success);
    background: var(--success-bg);
    border-radius: 7px;
    padding: .15rem .55rem;
    font-size: .72rem;
    font-weight: 600;
    text-decoration: none;
  }
  a.dl-btn:hover { background: var(--success); color: #06130c; }
  span.dl-dash { color: var(--muted); }

  #log {
    max-height: 60vh;
    overflow-y: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: .8rem;
  }
  .row {
    display: grid;
    grid-template-columns: 6.5rem 3.6rem 10rem 1fr;
    gap: .6rem;
    padding: .38rem 1rem;
    border-left: 3px solid transparent;
    border-bottom: 1px solid rgba(255,255,255,.03);
    white-space: pre-wrap;
    word-break: break-word;
  }
  .row.success { border-left-color: var(--success); background: var(--success-bg); }
  .row.error { border-left-color: var(--error); background: var(--error-bg); }
  .row.warning { border-left-color: var(--warning); background: var(--warning-bg); }
  .row.debug { border-left-color: var(--debug); background: var(--debug-bg); }
  .row.info { border-left-color: var(--info); }
  .row .ts { color: var(--muted); }
  .row .lvl.success { color: var(--success); }
  .row .lvl.error { color: var(--error); }
  .row .lvl.warning { color: var(--warning); }
  .row .lvl.debug { color: var(--debug); }
  .row .lvl.info { color: var(--info); }
  .row .lg { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .row .msg { color: var(--text); }
  footer { text-align: center; color: var(--muted); font-size: .75rem; padding: 1rem; }
</style>
</head>
<body>
<header>
  <h1>AgriGEE<span>.lite</span> — Monitor</h1>
  <div class="conn"><span class="dot" id="dot"></span><span id="connLabel">connecting…</span></div>
</header>
<main>
  <div class="stats">
    <div class="stat success"><div class="n" id="c-success">0</div><div class="l">Success</div></div>
    <div class="stat error"><div class="n" id="c-error">0</div><div class="l">Errors</div></div>
    <div class="stat warning"><div class="n" id="c-warning">0</div><div class="l">Warnings</div></div>
    <div class="stat debug"><div class="n" id="c-debug">0</div><div class="l">Debug</div></div>
    <div class="stat info"><div class="n" id="c-info">0</div><div class="l">Info</div></div>
  </div>

  <section class="card">
    <div class="card-head">
      Download jobs
      <div class="actions"><button class="btn" id="refreshJobs">refresh</button></div>
    </div>
    <div id="jobsWrap"><div class="empty">No jobs yet.</div></div>
  </section>

  <section class="card">
    <div class="card-head">
      Live log
      <div class="actions">
        <button class="btn active" id="autoscroll">auto-scroll</button>
        <button class="btn" id="clearLog">clear</button>
      </div>
    </div>
    <div id="log"></div>
  </section>
</main>
<footer>Updates automatically via Server-Sent Events — no reload needed.</footer>

<script>
(function () {
  "use strict";
  var logEl = document.getElementById("log");
  var dot = document.getElementById("dot");
  var connLabel = document.getElementById("connLabel");
  var autoscrollBtn = document.getElementById("autoscroll");
  var autoscroll = true;
  var counters = { success: 0, error: 0, warning: 0, debug: 0, info: 0 };
  var MAX_ROWS = 1000;
  var lastSeq = 0;

  autoscrollBtn.addEventListener("click", function () {
    autoscroll = !autoscroll;
    autoscrollBtn.classList.toggle("active", autoscroll);
  });
  document.getElementById("clearLog").addEventListener("click", function () {
    logEl.innerHTML = "";
  });

  function esc(s) {
    var d = document.createElement("div");
    d.innerText = s;
    return d.innerHTML;
  }

  function renderCounters() {
    Object.keys(counters).forEach(function (k) {
      var el = document.getElementById("c-" + k);
      if (el) el.textContent = counters[k];
    });
  }

  function appendEntry(entry, prepend) {
    var row = document.createElement("div");
    row.className = "row " + entry.category;
    row.innerHTML =
      '<span class="ts">' + esc(entry.ts.replace("T", " ").replace("+00:00", "Z")) + '</span>' +
      '<span class="lvl ' + entry.category + '">' + esc(entry.level) + '</span>' +
      '<span class="lg" title="' + esc(entry.logger) + '">' + esc(entry.logger) + '</span>' +
      '<span class="msg">' + esc(entry.message) + '</span>';
    if (prepend) {
      logEl.insertBefore(row, logEl.firstChild);
    } else {
      logEl.appendChild(row);
    }
    while (logEl.children.length > MAX_ROWS) {
      logEl.removeChild(logEl.lastChild);
    }
    if (autoscroll) logEl.scrollTop = 0;
  }

  fetch("/monitor/state")
    .then(function (r) { return r.json(); })
    .then(function (data) {
      counters = data.counters;
      renderCounters();
      var entries = data.entries.slice().reverse();
      entries.forEach(function (e) { appendEntry(e, false); });
      if (entries.length) lastSeq = data.last_seq;
      connect();
    })
    .catch(function () { connect(); });

  function connect() {
    var es = new EventSource("/monitor/stream?after=" + lastSeq);
    es.onopen = function () {
      dot.className = "dot live";
      connLabel.textContent = "live";
    };
    es.onerror = function () {
      dot.className = "dot down";
      connLabel.textContent = "reconnecting…";
    };
    es.onmessage = function (ev) {
      var entry = JSON.parse(ev.data);
      lastSeq = Math.max(lastSeq, entry.seq);
      if (entry.category in counters) counters[entry.category] += 1;
      renderCounters();
      appendEntry(entry, false);
    };
  }
  renderCounters();

  // ---- jobs panel -----------------------------------------------------
  var jobsWrap = document.getElementById("jobsWrap");

  function statusBadge(s) {
    return '<span class="badge ' + s + '">' + s + '</span>';
  }

  function downloadCell(j) {
    if (j.status !== "completed") return '<span class="dl-dash">—</span>';
    return '<a class="dl-btn" href="/jobs/' + encodeURIComponent(j.id) + '/download" download>Download</a>';
  }

  function renderJobs(jobs) {
    if (!jobs.length) {
      jobsWrap.innerHTML = '<div class="empty">No jobs yet.</div>';
      return;
    }
    var rows = jobs.slice().reverse().slice(0, 20).map(function (j) {
      return "<tr><td class=\"id\">" + esc(j.id) + "</td><td>" + esc(j.type || "-") + "</td><td>" +
        statusBadge(j.status) + "</td><td>" + downloadCell(j) + "</td><td>" + esc(j.error || "") + "</td></tr>";
    }).join("");
    jobsWrap.innerHTML =
      '<table class="jobs"><thead><tr><th>ID</th><th>Type</th><th>Status</th><th>File</th><th>Error</th></tr></thead><tbody>' +
      rows + "</tbody></table>";
  }

  function refreshJobs() {
    fetch("/jobs").then(function (r) { return r.json(); }).then(renderJobs).catch(function () {});
  }
  document.getElementById("refreshJobs").addEventListener("click", refreshJobs);
  refreshJobs();
  setInterval(refreshJobs, 4000);
})();
</script>
</body>
</html>
"""
