# app3 Live Console Logging — Design (2026-05-31)

**Status:** look approved by user; implementing on branch `merge-all-to-main-2026-05-27`. **No deploy** (server change ships when the branch is merged + the VM redeployed — user's action).

## Problem

Requests now carry a participant id (`pid`, percent-encoded in the multipart filename) and we resolve the client IP. The existing per-batch console readout (`pretty_print_batch` in `app3.py`) predates all of that: it prints the raw percent-encoded filename truncated to junk, shows no IP and no per-user color, has no per-participant grouping, and emits one tall block per batch. When two remote users stream at once the blocks interleave into an unreadable wall.

Goal: a live console where it's obvious **who is connected** (distinct users), **the per-person verdict**, **errors as they happen**, and a **rolling summary** of activity.

## Decisions (approved)

- **Layout:** rich per-frame blocks, *always* (not compact, not auto-switching).
- **Capabilities (all four):** tell users apart · per-person verdict · catch problems fast · rolling tallies.

## Output

```
═ 14:32:07 ●A 93.19.226.214  BATCH·6  thr=.50  18ms ═
 Guest          mean .928 FAKE (4f)
  ██████████░░ .931   ██████████░░ .902
  ····gated···· no_face   ██████████░░ .949
 PC Protector   mean .728 REAL (2f)
  ███████░░░░░ .701   ███████░░░░░ .755
 ⚠ Guest 1/4 NO-FACE — check crop
═ 14:32:08 ●B 84.12.7.9      BATCH·30 thr=.50  41ms ═
 dor            mean .180 REAL (30f)
  ██░░░░░░░░░░ .180   ██░░░░░░░░░░ .172   ██░░░░░░░░░░ .166
  +24 more · mean .19
──── live · 2 users · 14:32:10 ────
 ●A 93.19.226.214  Guest, PC Protector   6f  33%fake  17%gated  seen 3s
 ●B 84.12.7.9      dor                  30f   0%fake   3%gated  seen 2s
```

### A. Per-batch block

- **Header:** `═ {clock} ●{letter} {ip}  BATCH·{N} thr={t} {lat}ms ═`. The `●{letter}` badge is colored by a **stable hash of the IP** from a palette that **excludes red & green** (those stay reserved for the verdict). `{letter}` is `A,B,C…` assigned in **first-seen order** within the process. `{clock}` is a short `HH:MM:SS` so the block is self-contained even though the logger also stamps the record.
- **Grouped by participant** (pid via `observability.parse_pid_from_filename`; `None` → `unknown`), sorted **most-suspicious-first** (mean desc, then name).
- **Per participant:** ` {name}  mean {m:.3f} {VERDICT} ({n}f)` (VERDICT red/green), then up to `frame_cap` per-frame cells `{bar} .{prob}` (bar red/green by frame verdict). Gated/sentinel frames render dim with a short reason tag; decode/processing failures render dim. Overflow beyond the cap collapses to `+{k} more · mean {m}`.

### B. Anomaly lines (bright, directly under the block)

- ≥ `no_face_frac` of a participant's frames `no_face` → `⚠ {name} {k}/{n} NO-FACE — check crop`.
- Any decode/processing failure for a participant → `⚠ {name} {k} frame(s) FAILED`.
- Whole batch scored nothing → `⚠ ●{letter} {ip} BATCH all-gated/failed — nothing scored`.
- Latency > `slow_ms` → `⚠ slow {lat}ms`.

Warnings are bold yellow; hard failures bold red.

### C. Rolling dashboard (daemon thread)

- Reprints every `dashboard_seconds`, **only when there was traffic** since the last print (silent when idle).
- One row per IP (same color as its badge), counts over a rolling `window_seconds`: `●{letter} {ip}  {frames}f  {fake%}  {gated%}  seen {Δ}s  {participants}`.
- Header: `──── live · {N} users · {clock} ────`.

## Knobs (env vars — tunable live, no redeploy)

`OBS_LOG_FRAME_CAP=8` · `OBS_LOG_FRAMES_PER_LINE=3` · `OBS_LOG_BAR_WIDTH=12` · `OBS_LOG_NOFACE_FRAC=0.5` · `OBS_LOG_SLOW_MS=1000` · `OBS_LOG_DASHBOARD_SECONDS=15` · `OBS_LOG_WINDOW_SECONDS=300`.

## Architecture

New module **`live_log.py`** — import-light (**stdlib only**: `re, os, time, hashlib, threading, dataclasses`). No torch/cv2/GCS/observability imports, so it unit-tests in milliseconds.

- ANSI constants + `PALETTE` (deliberately omits codes 91/red and 92/green).
- `LogConfig` dataclass (+ `from_env()`).
- `BadgeRegistry.badge(ip) -> Badge(letter, color)` — first-seen letter, **md5-hash** color (deterministic across processes, unlike salted `hash()`), lock-guarded.
- Pure formatting: `group_by_participant(filenames, per_frame_status, pid_parser, threshold)`, `format_frame_cell`, `format_batch_block`, `detect_anomalies`, `render_batch`.
- `ActivityRegistry.record / should_print / render` — rolling per-IP counters, lock-guarded, injectable `now` for tests.
- `LiveLog(log_fn, cfg, …)` — orchestration + the daemon dashboard thread: `log_batch(...)`, `start()`, `stop()`, and `_tick()` (one dashboard step, called directly in tests so no sleeping).

**Wiring in `app3.py`:**

- `startup_event`: `app.state.live = live_log.LiveLog(log_fn=logger.info, cfg=live_log.LogConfig.from_env()); app.state.live.start()`. Created **unconditionally** — it's pure logging, independent of `OBS_ENABLED`.
- `shutdown_event`: best-effort `app.state.live.stop()`.
- `/check_frame_batch` (replaces the `pretty_print_batch(...)` call at ~app3.py:1251): resolve `ip = observability.get_client_ip(request)[0]`, `latency_ms = (time.perf_counter() - _t0) * 1000`, then `app.state.live.log_batch([f.filename for f in files], per_frame_status, threshold, ip, latency_ms, observability.parse_pid_from_filename)`. The old `pretty_print_batch` function is removed.
- Pid parsing for the **log** is independent of the observability capture path (works even with `OBS_ENABLED=false`), reusing `observability.parse_pid_from_filename`.

## Testing (TDD, no server)

`tests/test_live_log.py` (pytest, stdlib only). Assertions strip ANSI for visible text; color is checked by code membership.

- **Badges:** first-seen letters `A,B,…`; same IP stable; color ∈ palette and ∉ {red, green}; two IPs → distinct letters.
- **Grouping:** multi-pid split; `unknown` bucket; mean from scored frames only (sentinels/gated excluded); verdict by threshold; suspicious-first order; gated / no_face / failed counts.
- **Cells / block:** scored cell colored + prob shown; gated cell dim + short tag; frame cap → `+k more`; header carries ip / letter / thr / N.
- **Anomalies:** no_face ≥ frac; failures; nothing-scored; slow latency; none when clean.
- **Activity:** `record` → `should_print` True, then False after `render`; counts / % correct; window prune drops a stale IP; multi-IP user count; participant names listed.
- **Config:** `from_env` honors overrides and defaults.
- **LiveLog:** `log_batch` calls `log_fn` with the expected text and records activity; `_tick()` prints the dashboard after a record and is silent when idle.

## Out of scope (revisit later if wanted)

- `/check_frame` (single-frame endpoint) feeding the dashboard.
- Compact / auto layout (`OBS_LOG_STYLE`).
- Per-participant-name colors (only per-IP color now).
