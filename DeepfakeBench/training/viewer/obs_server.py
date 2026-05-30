"""
Local web viewer for per-request inference observability data in GCS.

Browses what `app3.py` + `observability.py` upload to `gs://remote-live-data`:
sessions (per client IP, gap-split) → requests → the actual frames + metadata.
Latest-first and lazy by date partition (the cheap NDJSON index drives listing;
heavy frame bytes load only on drill-in). Lets you assign human names to IPs and
persists those names/config back to the bucket so they're durable + shared.

Runs locally; reads (and writes the small config object) via your gcloud ADC:
    python -m viewer.obs_server --port 8502

Read path reuses the pure offline API in
`analysis/observability_sessions/sessionize.py`. Deliberately standalone — it
does NOT import the training-data viewer (`viewer/server.py`); the few GCS/frame
helpers are copied so this module has no heavy import chain.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import shutil
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from flask import Flask, jsonify, request, send_file, send_from_directory
from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage
from PIL import Image

from analysis.observability_sessions.sessionize import (
    sessionize,
    summarize_session,
)
# Reuse the server's pid parser so the viewer and the capture path share ONE
# source of truth for the WMA filename wire-format (incl. the URL-decode).
from observability import parse_pid_from_filename

logger = logging.getLogger(__name__)

# ── Config (overridable by CLI in main()) ─────────────────────────────────────
BUCKET = os.environ.get("OBS_BUCKET", "remote-live-data")
PREFIX = os.environ.get("OBS_PREFIX", "v1")
CONFIG_PATH = "_viewer/config.json"               # bucket-root, version-independent
OBS_FRAME_CACHE_DIR = Path(".viewer_cache/obs_frames").resolve()
# Bound the on-disk thumbnail cache so a long QC session over a big meeting
# (tens of thousands of frames) can't fill the disk. Oldest thumbnails evicted
# first once MAX is exceeded, down to TARGET. Tunable via env.
OBS_FRAME_CACHE_MAX_BYTES = int(os.environ.get("OBS_FRAME_CACHE_MAX_BYTES", 300 * 1024 * 1024))
OBS_FRAME_CACHE_TARGET_BYTES = int(os.environ.get("OBS_FRAME_CACHE_TARGET_BYTES",
                                                  int(OBS_FRAME_CACHE_MAX_BYTES * 0.8)))
OBS_FRAME_CACHE_CHECK_EVERY = int(os.environ.get("OBS_FRAME_CACHE_CHECK_EVERY", 40))
DEFAULT_SETTINGS = {"gap_minutes": 30, "lookup_days": 14}
_INDEX_CACHE_MAX_DATES = 16   # >= the sidebar/IP lookup window so /api/ips then /api/ip don't re-download
# Max request meta.json files the participant board reads per fetch — bounds work
# for huge meetings (4000+ requests); the UI raises it via "load more".
DEFAULT_BOARD_REQUEST_LIMIT = int(os.environ.get("OBS_BOARD_REQUEST_LIMIT", 80))

app = Flask(__name__, template_folder="templates", static_folder="templates")

_DATE_RE = re.compile(r"date=(\d{4}-\d{2}-\d{2})")


# ── Copied helpers (from viewer/server.py — kept local to avoid its import chain)
_gcs_client = None


def _get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        proj = os.environ.get("GOOGLE_CLOUD_PROJECT", "train-cvit2")
        _gcs_client = storage.Client(project=proj)
    return _gcs_client


def _human_size(n: int) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _guess_mime(name: str) -> str:
    n = (name or "").lower()
    if n.endswith(".png"):
        return "image/png"
    if n.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    return "application/octet-stream"


# ── Pure transforms (unit-tested) ─────────────────────────────────────────────
def parse_dates(prefixes: Iterable[str]) -> List[str]:
    """Extract `YYYY-MM-DD` from `.../date=YYYY-MM-DD/` prefixes; dedup, desc."""
    dates = set()
    for p in prefixes or []:
        m = _DATE_RE.search(p or "")
        if m:
            dates.add(m.group(1))
    return sorted(dates, reverse=True)


def sessions_payload(rows: List[Dict[str, Any]], gap_minutes: int = 30) -> Dict[str, Any]:
    """Sessionize index rows → newest-first sessions, each carrying its requests."""
    out = []
    for sess in sessionize(rows, gap_minutes=gap_minutes):
        summ = summarize_session(sess)
        summ["requests"] = sorted(sess, key=lambda r: _as_int(r.get("ts_epoch_ms")), reverse=True)
        out.append(summ)
    out.sort(key=lambda s: _as_int(s.get("start_ts_epoch_ms")), reverse=True)
    return {"n_requests": len(rows), "n_sessions": len(out), "sessions": out}


def ip_directory(rows: List[Dict[str, Any]], labels: Dict[str, Any],
                 gap_minutes: int = 30) -> List[Dict[str, Any]]:
    """Per-IP rollup for the user picker: counts, last-seen, name; last-seen desc."""
    labels = labels or {}
    by_ip: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_ip[r.get("client_ip") or "unknown"].append(r)

    out = []
    for ip, ip_rows in by_ip.items():
        last = max(ip_rows, key=lambda r: _as_int(r.get("ts_epoch_ms")))
        label = labels.get(ip) or {}
        dates = sorted({m.group(1) for r in ip_rows
                        for m in [_DATE_RE.search(r.get("gcs_prefix") or "")] if m}, reverse=True)
        out.append({
            "ip": ip,
            "name": label.get("name"),
            "notes": label.get("notes"),
            "n_requests": len(ip_rows),
            "n_sessions": len(sessionize(ip_rows, gap_minutes=gap_minutes)),
            "n_frames": sum(_as_int(r.get("n_frames")) for r in ip_rows),
            "fake_requests": sum(1 for r in ip_rows if r.get("pred_label") == "FAKE"),
            "last_seen_ts": _as_int(last.get("ts_epoch_ms")),
            "last_seen_utc": last.get("utc"),
            "dates": dates,
        })
    out.sort(key=lambda e: e["last_seen_ts"], reverse=True)
    return out


def attach_frame_urls(meta: Dict[str, Any], bucket: str) -> Dict[str, Any]:
    """Add a proxy `frame_url` (or None) + human byte size to each frame entry."""
    for f in meta.get("frames", []) or []:
        path = f.get("gcs_object_path")
        f["frame_url"] = f"/api/frame/{bucket}/{path}" if (path and not f.get("capture_skipped")) else None
        f["bytes_human"] = _human_size(_as_int(f.get("bytes_size")))
    return meta


def group_frames_by_participant(frames, threshold: float = 0.5):
    """Group a session's frames by participant identity for the session board.

    `frames` is the flat list of frame dicts from one or more request meta.json
    records (optionally already passed through `attach_frame_urls`). Identity is
    parsed from each frame's WMA-encoded filename via the shared
    `parse_pid_from_filename`, so the percent-encoded wire form (`pid%3D…`)
    groups correctly on today's already-captured data. Frames with no encoded
    pid land under participant_id ``"unknown"``. Each frame is annotated in
    place with ``participant_id`` + ``participant_seq``.

    Returns a list of participant dicts::

        {participant_id, n_frames, n_scored, n_gated, mean_score, verdict, frames}

    Sorted most-fake-suspicious first (mean_score desc, unscored/None last), then
    by frame count desc. Frames within a participant are ordered by
    ``participant_seq`` (the WMA per-participant counter), falling back to the
    request-local frame ``seq``. ``mean_score`` averages only model-scored frames
    (``scored`` true and a real, non-sentinel prob); ``verdict`` is by
    ``mean_score`` vs ``threshold`` (``None`` when nothing was scored).
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for f in frames or []:
        pid, pseq = parse_pid_from_filename(f.get("filename"))
        f["participant_id"] = pid or "unknown"
        f["participant_seq"] = pseq
        groups[f["participant_id"]].append(f)

    out: List[Dict[str, Any]] = []
    for pid, fs in groups.items():
        fs.sort(key=lambda f: f["participant_seq"] if f.get("participant_seq") is not None
                else _as_int(f.get("seq")))
        scored = [f for f in fs
                  if f.get("scored") and f.get("prob") is not None and f["prob"] >= 0.0]
        n_gated = sum(1 for f in fs if not f.get("scored")
                      and (f.get("gate_pass") is False or f.get("face_found") is False))
        mean_score = (sum(f["prob"] for f in scored) / len(scored)) if scored else None
        verdict = None if mean_score is None else ("FAKE" if mean_score >= threshold else "REAL")
        out.append({
            "participant_id": pid,
            "n_frames": len(fs),
            "n_scored": len(scored),
            "n_gated": n_gated,
            "mean_score": mean_score,
            "verdict": verdict,
            "frames": fs,
        })

    # Most fake-suspicious first; participants with nothing scored sort last.
    out.sort(key=lambda p: (p["mean_score"] if p["mean_score"] is not None else -1.0,
                            p["n_frames"]), reverse=True)
    return out


def normalize_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = raw or {}
    settings = dict(DEFAULT_SETTINGS)
    settings.update(raw.get("settings") or {})
    return {
        "version": raw.get("version", 1),
        "ip_labels": dict(raw.get("ip_labels") or {}),
        "settings": settings,
    }


def apply_label(config: Dict[str, Any], ip: str, name: str,
                notes: Optional[str] = None) -> Dict[str, Any]:
    config = normalize_config(config)
    entry: Dict[str, Any] = {"name": name}
    if notes is not None:
        entry["notes"] = notes
    config["ip_labels"][ip] = entry
    return config


# ── Bucket-persisted config (read-modify-write with generation-match) ──────────
def read_config(client) -> Tuple[Dict[str, Any], int]:
    blob = client.bucket(BUCKET).get_blob(CONFIG_PATH)
    if blob is None:
        return normalize_config(None), 0
    try:
        raw = json.loads(blob.download_as_text())
    except (ValueError, NotFound):
        raw = None
    return normalize_config(raw), (getattr(blob, "generation", 0) or 0)


def write_config(client, config: Dict[str, Any], generation: int) -> Dict[str, Any]:
    blob = client.bucket(BUCKET).blob(CONFIG_PATH)
    blob.upload_from_string(
        json.dumps(config, indent=2),
        content_type="application/json",
        if_generation_match=(generation or 0),   # 0 == create-only
    )
    return config


def set_label(client, ip: str, name: str, notes: Optional[str] = None,
              max_retries: int = 2) -> Dict[str, Any]:
    """Read current config, set ip→name, write back; retry on generation conflict."""
    last_exc: Optional[Exception] = None
    for _ in range(max_retries + 1):
        cfg, gen = read_config(client)
        cfg = apply_label(cfg, ip, name, notes)
        try:
            write_config(client, cfg, gen)
            return cfg
        except PreconditionFailed as e:
            last_exc = e
    raise last_exc  # exhausted retries


# ── Data access (GCS; monkeypatched in route tests) ───────────────────────────
_date_cache: "OrderedDict[str, List[dict]]" = OrderedDict()
_cache_lock = threading.Lock()


def list_dates() -> List[str]:
    """Cheap delimiter listing of `PREFIX/date=` common prefixes, newest first."""
    client = _get_gcs_client()
    it = client.list_blobs(client.bucket(BUCKET), prefix=f"{PREFIX}/date=", delimiter="/")
    list(it)  # consume so .prefixes is populated
    return parse_dates(getattr(it, "prefixes", []) or [])


def _safe_index_text(blob) -> str:
    try:
        return blob.download_as_text()
    except Exception:
        return ""


def _download_index_rows(date: str) -> List[Dict[str, Any]]:
    """Download + parse all `_index/*.jsonl` part-files for one date, downloading
    the parts CONCURRENTLY. A busy day has hundreds of tiny part-files; on a slow
    link the time is almost all round-trip latency, so fan-out is a big win."""
    client = _get_gcs_client()
    bucket = client.bucket(BUCKET)
    blobs = [b for b in client.list_blobs(bucket, prefix=f"{PREFIX}/date={date}/_index/")
             if b.name.endswith(".jsonl")]
    rows: List[Dict[str, Any]] = []
    if not blobs:
        return rows
    with ThreadPoolExecutor(max_workers=16) as ex:
        for text in ex.map(_safe_index_text, blobs):
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def read_index_rows(date: str) -> List[Dict[str, Any]]:
    """All index lines for one date, with a small LRU in-memory cache."""
    with _cache_lock:
        if date in _date_cache:
            _date_cache.move_to_end(date)
            return _date_cache[date]
    rows = _download_index_rows(date)
    with _cache_lock:
        _date_cache[date] = rows
        _date_cache.move_to_end(date)
        while len(_date_cache) > _INDEX_CACHE_MAX_DATES:
            _date_cache.popitem(last=False)
    return rows


def download_meta(gcs_prefix: str) -> Dict[str, Any]:
    client = _get_gcs_client()
    blob = client.bucket(BUCKET).blob(gcs_prefix.rstrip("/") + "/meta.json")
    return json.loads(blob.download_as_text())


def _load_meta_for_prefix(prefix: str) -> Optional[Dict[str, Any]]:
    """Read one request's meta.json and attach frame URLs. Fail-soft: returns
    None for a missing/corrupt meta so one bad request can't break the board."""
    try:
        meta = download_meta(prefix)
    except Exception:
        return None
    attach_frame_urls(meta, BUCKET)
    return meta


_cache_write_lock = threading.Lock()
_cache_write_count = 0


def _evict_frame_cache(max_bytes, target_bytes, cache_dir=None):
    """LRU-ish cap on the thumbnail cache: if it exceeds ``max_bytes``, delete
    files oldest-first (by mtime) until the total is <= ``target_bytes``. Returns
    bytes freed. Safe if the dir is missing or a file vanishes mid-scan."""
    d = Path(cache_dir) if cache_dir is not None else OBS_FRAME_CACHE_DIR
    if not d.exists():
        return 0
    files, total = [], 0
    for f in d.rglob("*"):
        if not f.is_file():
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        files.append((st.st_mtime, st.st_size, f))
        total += st.st_size
    if total <= max_bytes:
        return 0
    files.sort(key=lambda t: t[0])         # oldest mtime first
    freed = 0
    for _mtime, size, f in files:
        if total - freed <= target_bytes:
            break
        try:
            f.unlink()
            freed += size
        except OSError:
            continue
    return freed


def _maybe_evict_cache():
    """Throttle: run a full eviction sweep only every OBS_FRAME_CACHE_CHECK_EVERY
    cached writes, so the cap is enforced cheaply (no dir scan per request)."""
    global _cache_write_count
    with _cache_write_lock:
        _cache_write_count += 1
        due = (_cache_write_count % OBS_FRAME_CACHE_CHECK_EVERY) == 0
    if due:
        try:
            freed = _evict_frame_cache(OBS_FRAME_CACHE_MAX_BYTES, OBS_FRAME_CACHE_TARGET_BYTES)
            if freed:
                logger.info("frame cache evicted %s", _human_size(freed))
        except Exception as e:
            logger.warning("frame cache eviction failed: %s", e)


def _cache_size_human() -> str:
    if not OBS_FRAME_CACHE_DIR.exists():
        return "0 B"
    total = sum(f.stat().st_size for f in OBS_FRAME_CACHE_DIR.rglob("*") if f.is_file())
    return _human_size(total)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    resp = send_from_directory(app.template_folder, "obs.html")
    resp.headers["Cache-Control"] = "no-store"   # always serve fresh during dev
    return resp


@app.route("/api/dates")
def api_dates():
    try:
        return jsonify({"dates": list_dates(), "bucket": BUCKET, "prefix": PREFIX})
    except Exception as e:  # surface, don't hang
        return jsonify({"error": str(e)}), 502


@app.route("/api/sessions")
def api_sessions():
    date = request.args.get("date")
    if not date:
        return jsonify({"error": "date required"}), 400
    gap = _as_int(request.args.get("gap_minutes"), 30)
    try:
        rows = read_index_rows(date)
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    body = sessions_payload(rows, gap_minutes=gap)
    body.update(date=date, gap_minutes=gap)
    return jsonify(body)


@app.route("/api/ips")
def api_ips():
    days = _as_int(request.args.get("days"), DEFAULT_SETTINGS["lookup_days"])
    gap = _as_int(request.args.get("gap_minutes"), 30)
    try:
        dates = list_dates()[:max(1, days)]
        rows: List[dict] = []
        for d in dates:
            rows.extend(read_index_rows(d))
        labels = read_config(_get_gcs_client())[0]["ip_labels"]
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"days": days, "dates": dates, "ips": ip_directory(rows, labels, gap_minutes=gap)})


@app.route("/api/ip")
def api_ip():
    ip = request.args.get("ip")
    if not ip:
        return jsonify({"error": "ip required"}), 400
    gap = _as_int(request.args.get("gap_minutes"), 30)
    date = request.args.get("date")
    try:
        if date:
            # Single-day load: cheap, lets the UI page a user's history in parts
            # (newest day first) instead of sweeping every recent day at once.
            rows = [r for r in read_index_rows(date) if r.get("client_ip") == ip]
        else:
            days = _as_int(request.args.get("days"), DEFAULT_SETTINGS["lookup_days"])
            dates = list_dates()[:max(1, days)]
            rows = [r for d in dates for r in read_index_rows(d) if r.get("client_ip") == ip]
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    body = sessions_payload(rows, gap_minutes=gap)
    body.update(ip=ip, date=date)
    return jsonify(body)


@app.route("/api/request")
def api_request():
    prefix = request.args.get("gcs_prefix")
    if not prefix:
        return jsonify({"error": "gcs_prefix required"}), 400
    try:
        meta = download_meta(prefix)
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    return jsonify(attach_frame_urls(meta, BUCKET))


@app.route("/api/participants", methods=["POST"])
def api_participants():
    """Per-participant board for ONE session.

    Body: ``{"prefixes": [<request gcs_prefix>, ...], "threshold"?: float}``.
    Reads each request's meta.json, groups every frame by participant identity
    (parsed from the encoded filename) and returns frames + per-participant
    scores so the browser can show each image next to the score it got.
    Threshold defaults to the session's own request params (else 0.5).
    """
    body = request.get_json(silent=True) or {}
    prefixes = body.get("prefixes") or []
    if not prefixes:
        return jsonify({"error": "prefixes required"}), 400

    try:
        limit = int(body.get("limit") or DEFAULT_BOARD_REQUEST_LIMIT)
    except (TypeError, ValueError):
        limit = DEFAULT_BOARD_REQUEST_LIMIT
    limit = max(1, limit)
    # `prefixes` arrive chronological ascending; read only the most-recent
    # `limit` so a 4000-request meeting doesn't fetch every meta.json. The UI
    # raises `limit` ("load more") to reach older history.
    used = prefixes[-limit:]

    # meta.json reads are IO-bound; fan them out so even `limit` reads are quick.
    metas: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for m in ex.map(_load_meta_for_prefix, used):
            if m is not None:
                metas.append(m)

    # Threshold: trust the session's own request params; allow a body override.
    thr = body.get("threshold")
    if thr is None:
        thr = next((m.get("params", {}).get("threshold")
                    for m in metas if (m.get("params") or {}).get("threshold") is not None), 0.5)
    try:
        threshold = float(thr)
    except (TypeError, ValueError):
        threshold = 0.5

    frames = [f for m in metas for f in (m.get("frames") or [])]
    participants = group_frames_by_participant(frames, threshold=threshold)
    return jsonify({
        "participants": participants,
        "n_participants": len(participants),
        "n_frames": len(frames),
        "n_requests_total": len(prefixes),
        "n_requests_read": len(used),
        "truncated": len(prefixes) > len(used),
        "threshold": threshold,
    })


@app.route("/api/config")
def api_config():
    try:
        cfg, gen = read_config(_get_gcs_client())
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"config": cfg, "generation": gen})


@app.route("/api/label", methods=["POST"])
def api_label():
    body = request.get_json(silent=True) or {}
    ip = (body.get("ip") or "").strip()
    name = (body.get("name") or "").strip()
    if not ip or not name:
        return jsonify({"error": "ip and name required"}), 400
    try:
        cfg = set_label(_get_gcs_client(), ip, name, body.get("notes"))
    except Exception as e:
        return jsonify({"error": f"could not save label: {e}"}), 502
    return jsonify(cfg)


@app.route("/api/stats")
def api_stats():
    date = request.args.get("date")
    if not date:
        return jsonify({"error": "date required"}), 400
    gap = _as_int(request.args.get("gap_minutes"), 30)
    try:
        rows = read_index_rows(date)
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({
        "date": date,
        "n_requests": len(rows),
        "n_sessions": len(sessionize(rows, gap_minutes=gap)),
        "n_fake": sum(1 for r in rows if r.get("pred_label") == "FAKE"),
        "n_real": sum(1 for r in rows if r.get("pred_label") == "REAL"),
        "n_gated": sum(1 for r in rows if _as_int(r.get("n_gated")) > 0),
        "n_errored": sum(1 for r in rows if r.get("status") != "ok"),
        "unique_ips": sorted({r.get("client_ip") for r in rows if r.get("client_ip")}),
        "cache_size_human": _cache_size_human(),
    })


@app.route("/api/frame/<bucket>/<path:blob_path>")
def proxy_frame(bucket, blob_path):
    """Proxy a GCS frame, thumbnail by default, cached on disk. ?full=1, ?size=."""
    thumb = request.args.get("full") != "1"
    thumb_size = _as_int(request.args.get("size"), 224)
    cache_key = f"{bucket}/{blob_path}" + (f"__thumb{thumb_size}" if thumb else "")
    local_path = OBS_FRAME_CACHE_DIR / cache_key
    if local_path.exists():
        return send_file(str(local_path.resolve()),
                         mimetype="image/jpeg" if thumb else _guess_mime(blob_path))
    try:
        blob = _get_gcs_client().bucket(bucket).blob(blob_path)
        data = blob.download_as_bytes()
        content_type = blob.content_type
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502

    mimetype = "image/jpeg"
    if thumb:
        try:
            img = Image.open(io.BytesIO(data))
            img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
        except Exception:
            mimetype = content_type or _guess_mime(blob_path)  # serve original if resize fails
    else:
        mimetype = content_type or _guess_mime(blob_path)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)
    _maybe_evict_cache()                       # keep the on-disk cache under its size cap
    return send_file(io.BytesIO(data), mimetype=mimetype)


@app.route("/api/cache", methods=["DELETE"])
def clear_cache():
    if OBS_FRAME_CACHE_DIR.exists():
        shutil.rmtree(OBS_FRAME_CACHE_DIR)
    return jsonify({"status": "cleared"})


@app.route("/api/cache-size")
def cache_size():
    return jsonify({"human": _cache_size_human()})


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    global BUCKET, PREFIX
    parser = argparse.ArgumentParser(description="Local observability data viewer")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    BUCKET = args.bucket
    PREFIX = args.prefix
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Observability viewer → gs://%s/%s  at http://%s:%d",
                BUCKET, PREFIX, args.host, args.port)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
