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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from flask import Flask, jsonify, request, send_file, send_from_directory
from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage
from PIL import Image

from analysis.observability_sessions.sessionize import (
    read_index_lines,
    sessionize,
    summarize_session,
)

logger = logging.getLogger(__name__)

# ── Config (overridable by CLI in main()) ─────────────────────────────────────
BUCKET = os.environ.get("OBS_BUCKET", "remote-live-data")
PREFIX = os.environ.get("OBS_PREFIX", "v1")
CONFIG_PATH = "_viewer/config.json"               # bucket-root, version-independent
OBS_FRAME_CACHE_DIR = Path(".viewer_cache/obs_frames").resolve()
DEFAULT_SETTINGS = {"gap_minutes": 30, "lookup_days": 14}
_INDEX_CACHE_MAX_DATES = 8

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


def read_index_rows(date: str) -> List[Dict[str, Any]]:
    """All index lines for one date, with a small LRU in-memory cache."""
    with _cache_lock:
        if date in _date_cache:
            _date_cache.move_to_end(date)
            return _date_cache[date]
    rows = read_index_lines(BUCKET, PREFIX, [date])
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
    days = _as_int(request.args.get("days"), DEFAULT_SETTINGS["lookup_days"])
    gap = _as_int(request.args.get("gap_minutes"), 30)
    try:
        dates = list_dates()[:max(1, days)]
        rows = [r for d in dates for r in read_index_rows(d) if r.get("client_ip") == ip]
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    body = sessions_payload(rows, gap_minutes=gap)
    body.update(ip=ip, days=days)
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
