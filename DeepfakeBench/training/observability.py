"""Per-request observability for the live inference server (app3.py).

Captures, for every /check_frame and /check_frame_batch request, the client IP,
the original uploaded image bytes, the request parameters (threshold, gate
profile, ...), the model identity, and the per-frame + aggregate results, then
uploads them asynchronously to a GCS bucket via background worker threads.

Design contract (see plans/ + docs):
  * The request path NEVER blocks on upload — producers do a non-blocking
    enqueue and return immediately. Blocking GCS I/O runs on daemon threads.
  * Observability can NEVER crash or 500 inference — every public entry point is
    wrapped so it fails open (logs + drops, never raises into the caller).
  * Nothing is written to local disk — image bytes live in an in-memory,
    byte-bounded queue and are uploaded directly, then freed.

This module is import-safe with no side effects beyond reading env config.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("effort-aigi-api-v3.observability")

SCHEMA_VERSION = "obs-1"


# ──────────────────────────────────────────
# Env config (read once at import; no side effects)
# ──────────────────────────────────────────
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default


OBS_ENABLED = _env_bool("OBS_ENABLED", True)
OBS_BUCKET = os.getenv("OBS_BUCKET", "remote-live-data").strip()
OBS_PREFIX = os.getenv("OBS_PREFIX", "v1").strip().strip("/")
OBS_QUEUE_MAX_BYTES = _env_int("OBS_QUEUE_MAX_BYTES", 512 * 1024 * 1024)
OBS_MAX_RECORD_BYTES = _env_int("OBS_MAX_RECORD_BYTES", 64 * 1024 * 1024)
OBS_NUM_WORKERS = _env_int("OBS_NUM_WORKERS", 2)
OBS_DRAIN_TIMEOUT_S = _env_int("OBS_DRAIN_TIMEOUT_S", 15)
OBS_INDEX_FLUSH_EVERY = _env_int("OBS_INDEX_FLUSH_EVERY", 25)
OBS_INDEX_FLUSH_SECONDS = _env_int("OBS_INDEX_FLUSH_SECONDS", 10)
OBS_MAX_RETRIES = _env_int("OBS_MAX_RETRIES", 3)
GIT_SHA = os.getenv("GIT_SHA")

# ──────────────────────────────────────────
# Stateless helpers (no GCS, no threads)
# ──────────────────────────────────────────
_NON_IP_SAFE = re.compile(r"[^a-z0-9-]+")
_NON_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._ -]+")
_MAX_FILENAME_LEN = 128


def get_client_ip(request):
    """Resolve (client_ip, remote_addr, xff_chain) from a Starlette request.

    * remote_addr   — the direct peer (request.client.host), or "unknown".
    * xff_chain     — the parsed X-Forwarded-For list (trimmed, empties dropped).
    * client_ip     — the first XFF hop if present (the real client behind a
                      trusted proxy), else the direct peer.

    X-Forwarded-For is client-spoofable, so the raw chain and the direct peer
    are both preserved for offline trust re-derivation. Never raises.
    """
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client is not None else None
    remote_addr = host if host else "unknown"

    xff_raw = ""
    try:
        xff_raw = request.headers.get("x-forwarded-for") or ""
    except Exception:
        xff_raw = ""
    chain = [part.strip() for part in xff_raw.split(",") if part.strip()]

    client_ip = chain[0] if chain else remote_addr
    return client_ip, remote_addr, chain


def sanitize_ip(ip: str) -> str:
    """Make an IP safe for a GCS object-key segment: lowercase, [a-z0-9-] only.

    `.`/`:`/`/` (IPv4 dots, IPv6 colons, path separators) collapse to `-` so no
    spurious key "directories" are introduced. Empty -> "unknown".
    """
    if not ip:
        return "unknown"
    s = ip.strip().lower().replace(":", "-").replace(".", "-").replace("/", "-")
    s = _NON_IP_SAFE.sub("-", s)
    return s or "unknown"


def sanitize_filename(name, seq: int) -> str:
    """Make a user-supplied filename safe + non-traversing for metadata/keys.

    Drops any directory part, removes `..`, replaces unsafe chars with `_`, caps
    length. Empty/None -> "frame_<seq>".
    """
    if not name or not name.strip():
        return f"frame_{seq}"
    base = os.path.basename(name.strip()).replace("..", "")
    base = _NON_FILENAME_SAFE.sub("_", base).strip()
    if not base:
        return f"frame_{seq}"
    return base[:_MAX_FILENAME_LEN]


# ──────────────────────────────────────────
# Capture records (serializable; no GCS)
# ──────────────────────────────────────────
@dataclass
class FrameCapture:
    """One frame within a request: original bytes + its per-frame result.

    `raw_bytes` is held in memory only until uploaded, then freed; it is never
    serialized into meta.json (see `to_meta`).
    """

    seq: int
    raw_bytes: bytes
    filename: Optional[str] = None
    content_type: Optional[str] = None
    dims_hw: Optional[Tuple[int, int]] = None  # (height, width); None if undecodable
    gate_pass: Optional[bool] = None           # None = not evaluated (decode/processing failed)
    gate_reason: Optional[str] = None
    face_found: Optional[bool] = None          # tri-state: None = recrop not requested
    prob: Optional[float] = None
    scored: bool = False                       # did the model actually score this frame?
    verdict: Optional[str] = None
    gcs_object_path: Optional[str] = None       # set by the uploader ONLY after a successful upload
    capture_skipped: bool = False               # image bytes intentionally not retained (over byte budget)

    @property
    def bytes_size(self) -> int:
        return len(self.raw_bytes or b"")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw_bytes or b"").hexdigest()

    def to_meta(self) -> Dict[str, Any]:
        h, w = self.dims_hw if self.dims_hw else (None, None)
        return {
            "seq": self.seq,
            "filename": self.filename,
            "content_type": self.content_type,
            "dims": ({"h": h, "w": w} if self.dims_hw else None),
            "bytes_size": self.bytes_size,
            "sha256": self.sha256,
            "gate_pass": self.gate_pass,
            "gate_reason": self.gate_reason,
            "face_found": self.face_found,
            "prob": self.prob,
            "scored": self.scored,
            "verdict": self.verdict,
            "gcs_object_path": self.gcs_object_path,
            "capture_skipped": self.capture_skipped,
        }


@dataclass
class CaptureRecord:
    """Everything captured for one inference request."""

    request_id: str
    received_utc: str
    received_epoch_ms: int
    endpoint: str
    # client
    client_ip: str
    remote_addr: str
    xff_chain: List[str]
    user_agent: Optional[str]
    x_client_id: Optional[str]
    # request params
    model_type: Optional[str]
    threshold: float
    gate_profile: Optional[str]
    gate_spec: Optional[Dict[str, Any]]
    yolo_conf_threshold: Optional[float]
    recrop: Optional[bool]
    debug: Optional[bool]
    # results
    frames: List[FrameCapture] = field(default_factory=list)
    status: str = "handler_exited_without_status"
    pred_label: Optional[str] = None
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    # set by the uploader at upload time
    gcs_prefix: Optional[str] = None

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes_size for f in self.frames)

    def counts(self) -> Dict[str, int]:
        total = len(self.frames)
        scored = sum(1 for f in self.frames if f.scored)
        gated = sum(
            1 for f in self.frames
            if not f.scored and (f.gate_pass is False or f.face_found is False)
        )
        failed = total - scored - gated
        return {"total": total, "scored": scored, "gated": gated, "failed": failed}

    def to_meta_dict(self, static: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        static = static or {}
        checkpoint_paths = static.get("checkpoint_paths", {}) or {}
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": self.request_id,
            "received_utc": self.received_utc,
            "received_epoch_ms": self.received_epoch_ms,
            "endpoint": self.endpoint,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "gcs_prefix": self.gcs_prefix,
            "client": {
                "client_ip": self.client_ip,
                "remote_addr": self.remote_addr,
                "xff_chain": self.xff_chain,
                "user_agent": self.user_agent,
                "x_client_id": self.x_client_id,
            },
            "model": {
                "model_type": self.model_type,
                "checkpoint_path": checkpoint_paths.get(self.model_type),
                "use_arcface": static.get("use_arcface"),
                "device": static.get("device"),
                "app_version": static.get("app_version"),
                "git_sha": static.get("git_sha"),
                "hostname": static.get("hostname"),
                "pid": static.get("pid"),
            },
            "params": {
                "threshold": self.threshold,
                "gate_profile": self.gate_profile,
                "gate_spec": self.gate_spec,
                "yolo_conf_threshold": self.yolo_conf_threshold,
                "recrop": self.recrop,
                "debug": self.debug,
            },
            "aggregate": {
                "pred_label": self.pred_label,
                "confidence": self.confidence,
            },
            "counts": self.counts(),
            "frames": [f.to_meta() for f in self.frames],
        }

    def build_index_line(self) -> Dict[str, Any]:
        """Compact one-line summary for the date-partitioned _index JSONL.

        Self-sufficient for offline sessionization (group by client_ip + time
        gap) without dereferencing meta.json or the image objects.
        """
        c = self.counts()
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": self.request_id,
            "ts_epoch_ms": self.received_epoch_ms,
            "utc": self.received_utc,
            "client_ip": self.client_ip,
            "endpoint": self.endpoint,
            "model_type": self.model_type,
            "pred_label": self.pred_label,
            "confidence": self.confidence,
            "status": self.status,
            "n_frames": c["total"],
            "n_scored": c["scored"],
            "n_gated": c["gated"],
            "n_failed": c["failed"],
            "gcs_prefix": self.gcs_prefix,
        }


# ──────────────────────────────────────────
# Async GCS uploader (queue + daemon worker threads)
# ──────────────────────────────────────────
def _ext_for(content_type: Optional[str]) -> str:
    ct = (content_type or "").lower()
    if ct == "image/png":
        return ".png"
    if ct in ("image/jpeg", "image/jpg"):
        return ".jpg"
    return ".bin"


def _date_from_epoch_ms(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")


def new_record(
    request,
    endpoint: str,
    *,
    model_type: Optional[str],
    threshold: float,
    gate_profile: Optional[str],
    gate_spec: Optional[Dict[str, Any]],
    yolo_conf_threshold: Optional[float] = None,
    recrop: Optional[bool] = None,
    debug: Optional[bool] = None,
) -> "CaptureRecord":
    """Build a CaptureRecord envelope for one request (client + params + ids).

    `received_utc` is a compact, lexicographically-sortable UTC stamp so that
    listing `req=` prefixes within an `ip=` directory is chronological.
    """
    client_ip, remote_addr, xff_chain = get_client_ip(request)
    try:
        user_agent = request.headers.get("user-agent")
        x_client_id = request.headers.get("x-client-id")
    except Exception:
        user_agent = x_client_id = None

    epoch_ms = int(time.time() * 1000)
    dt = datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
    received_utc = dt.strftime("%Y%m%dT%H%M%S") + f"{dt.microsecond // 1000:03d}Z"

    return CaptureRecord(
        request_id=uuid.uuid4().hex,
        received_utc=received_utc,
        received_epoch_ms=epoch_ms,
        endpoint=endpoint,
        client_ip=client_ip,
        remote_addr=remote_addr,
        xff_chain=xff_chain,
        user_agent=user_agent,
        x_client_id=x_client_id,
        model_type=model_type,
        threshold=threshold,
        gate_profile=gate_profile,
        gate_spec=gate_spec,
        yolo_conf_threshold=yolo_conf_threshold,
        recrop=recrop,
        debug=debug,
    )


def _safe_enqueue(uploader: Optional["ObservabilityUploader"], record) -> None:
    """Enqueue for upload, swallowing everything. Used at request-handler exits."""
    try:
        if uploader is not None and record is not None:
            uploader.enqueue(record)
    except Exception:
        pass


def _default_bucket_factory(bucket_name: str):
    # Imported lazily so the module stays import-safe even if the GCS SDK or
    # credentials are unavailable (the uploader then fails open).
    from google.cloud import storage
    return storage.Client().bucket(bucket_name)


_DONE = object()  # queue sentinel


class ObservabilityUploader:
    """Buffers CaptureRecords and uploads them to GCS on background threads.

    The producer side (`enqueue`) is non-blocking and fully exception-guarded so
    it can never slow or crash the request path. The consumer side does the
    blocking GCS I/O on daemon threads. The queue is bounded by *bytes in flight*
    (not record count) so a burst of large batches cannot OOM the process.
    """

    def __init__(
        self,
        bucket_name: str,
        *,
        bucket: Any = None,
        bucket_factory: Optional[Callable[[], Any]] = None,
        num_workers: int = OBS_NUM_WORKERS,
        max_queue_bytes: int = OBS_QUEUE_MAX_BYTES,
        max_record_bytes: int = OBS_MAX_RECORD_BYTES,
        max_retries: int = OBS_MAX_RETRIES,
        index_flush_every: int = OBS_INDEX_FLUSH_EVERY,
        index_flush_seconds: int = OBS_INDEX_FLUSH_SECONDS,
        prefix: str = OBS_PREFIX,
        static: Optional[Dict[str, Any]] = None,
        retry_sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.bucket_name = bucket_name
        self.num_workers = max(1, num_workers)
        self.max_queue_bytes = max_queue_bytes
        self.max_record_bytes = max_record_bytes
        self.max_retries = max_retries
        self.index_flush_every = max(1, index_flush_every)
        self.index_flush_seconds = index_flush_seconds
        self._prefix = prefix
        self._static = static or {}
        self._retry_sleep = retry_sleep
        self._clock = clock

        self._q: "queue.Queue[Any]" = queue.Queue()
        self._lock = threading.Lock()
        self._index_lock = threading.Lock()
        self._bytes_in_flight = 0
        self._accepting = True
        self._stopped = False
        self._threads: List[threading.Thread] = []
        self._flusher: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._index_buffer: List[Tuple[str, str]] = []   # (date, json_line)
        self._last_drop_log = 0.0
        self._stats = {
            "enqueued": 0, "uploaded_records": 0, "uploaded_objects": 0,
            "failed": 0, "failed_objects": 0, "dropped_records": 0, "dropped_bytes": 0,
        }
        self._pid = os.getpid()

        # Resolve the bucket. Any failure => fail open (disabled, never raise).
        self.enabled = True
        self._bucket = None
        if bucket is not None:
            self._bucket = bucket
        else:
            factory = bucket_factory or (lambda: _default_bucket_factory(bucket_name))
            try:
                self._bucket = factory()
            except Exception as e:
                self.enabled = False
                logger.warning("Observability disabled — bucket init failed: %s", e)

    # ---- lifecycle ----
    def start(self) -> None:
        if not self.enabled:
            return
        self._accepting = True
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, name=f"obs-uploader-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        # Background flusher so a request followed by idle still gets its index
        # line flushed (the index is the offline-sessionization read path).
        self._flusher = threading.Thread(target=self._flusher_loop, name="obs-index-flusher", daemon=True)
        self._flusher.start()
        logger.info("Observability uploader started: bucket=%s workers=%d", self.bucket_name, self.num_workers)

    def stop(self, timeout: Optional[float] = None) -> None:
        if not self.enabled or self._stopped:
            return
        self._stopped = True
        self._accepting = False
        deadline = (self._clock() + timeout) if timeout else None

        # Drain phase: ALL workers run concurrently; wait until the queue is
        # empty AND nothing is in flight, bounded by the deadline. This spends
        # the whole budget on actual draining rather than serially joining one
        # (possibly slow) worker first.
        while deadline is None or self._clock() < deadline:
            with self._lock:
                drained = self._bytes_in_flight == 0 and self._q.qsize() == 0
            if drained:
                break
            time.sleep(0.02)

        # Signal workers + flusher to exit, then join within the remaining budget.
        self._stop_evt.set()
        for _ in self._threads:
            self._q.put(_DONE)
        for t in self._threads:
            t.join(None if deadline is None else max(0.0, deadline - self._clock()))
        if self._flusher is not None:
            self._flusher.join(None if deadline is None else max(0.0, deadline - self._clock()))
        self.flush_index()  # final best-effort flush of any trailing index lines

    # ---- producer (request path; must never raise) ----
    def enqueue(self, record: "CaptureRecord") -> None:
        try:
            if not self.enabled or not self._accepting:
                return
            nbytes = record.total_bytes
            if nbytes > self.max_record_bytes:
                self._count_dropped(nbytes, "record_too_big")
                return
            with self._lock:
                if self._bytes_in_flight + nbytes > self.max_queue_bytes:
                    self._count_dropped_locked(nbytes)
                    over = True
                else:
                    # put first (cannot raise on an unbounded queue), THEN reserve
                    # bytes — so the reservation and the enqueue are atomic and a
                    # put failure can never leak the byte budget.
                    self._q.put_nowait(record)
                    self._bytes_in_flight += nbytes
                    self._stats["enqueued"] += 1
                    over = False
            if over:
                self._maybe_log_drop("queue_full")
                return
        except Exception as e:  # observability must never throw into inference
            try:
                self._count_dropped(getattr(record, "total_bytes", 0), "enqueue_error")
            except Exception:
                pass
            logger.warning("Observability enqueue error (record dropped): %s", e)

    # ---- consumer ----
    def _worker_loop(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is _DONE:
                    break
                nbytes = getattr(item, "total_bytes", 0)
                try:
                    self._upload_record(item)
                except Exception as e:
                    with self._lock:
                        self._stats["failed"] += 1
                    logger.warning("Observability upload_record failed: %s", e)
                finally:
                    with self._lock:
                        self._bytes_in_flight = max(0, self._bytes_in_flight - nbytes)
                    for f in getattr(item, "frames", []):
                        f.raw_bytes = b""  # free memory ASAP
            finally:
                self._q.task_done()

    def _upload_record(self, rec: "CaptureRecord") -> None:
        date = _date_from_epoch_ms(rec.received_epoch_ms)
        # Full request_id (a uuid4 hex) in the key so two same-IP, same-ms
        # requests cannot collide and overwrite each other's objects.
        prefix = (
            f"{self._prefix}/date={date}/ip={sanitize_ip(rec.client_ip)}"
            f"/req={rec.received_utc}_{rec.request_id or 'noid'}"
        )
        rec.gcs_prefix = prefix

        objs_ok = 0
        for f in rec.frames:
            # Frames whose bytes were not retained (over byte budget) or that have
            # no bytes are metadata-only — never write an empty/bogus object.
            if f.capture_skipped or not f.raw_bytes:
                f.gcs_object_path = None
                continue
            name = f"{prefix}/frame_{f.seq:03d}{_ext_for(f.content_type)}"
            if self._upload_blob(name, f.raw_bytes, f.content_type or "application/octet-stream"):
                f.gcs_object_path = name      # set ONLY after a confirmed upload
                objs_ok += 1
            else:
                f.gcs_object_path = None       # don't advertise an object that 404s
                with self._lock:
                    self._stats["failed_objects"] += 1

        # meta.json LAST — its presence is the per-record commit marker.
        meta_json = json.dumps(rec.to_meta_dict(self._static), separators=(",", ":"))
        meta_ok = self._upload_blob(f"{prefix}/meta.json", meta_json.encode("utf-8"), "application/json")

        with self._lock:
            self._stats["uploaded_objects"] += objs_ok + (1 if meta_ok else 0)
            if meta_ok:
                self._stats["uploaded_records"] += 1
            else:
                self._stats["failed"] += 1

        # Only advertise the record in the index once meta.json (the commit
        # marker) is actually committed — otherwise the offline reader would
        # surface a request whose prefix has no meta.json.
        if meta_ok:
            self._buffer_index(date, rec.build_index_line())

    def _upload_blob(self, name: str, data: bytes, content_type: str) -> bool:
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                self._bucket.blob(name).upload_from_string(data, content_type=content_type)
                return True
            except Exception as e:
                last = e
                if attempt < self.max_retries:
                    self._retry_sleep(min(2.0 ** attempt, 8.0) * 0.5)
        logger.warning("Observability blob upload failed (%s) after %d attempts: %s",
                       name, self.max_retries + 1, last)
        return False

    def _flusher_loop(self) -> None:
        # Time-based flush: wake every index_flush_seconds (or when stopping) and
        # flush whatever index lines have accumulated, so trailing lines aren't
        # stranded in memory waiting for the next request.
        while not self._stop_evt.wait(self.index_flush_seconds):
            try:
                self.flush_index()
            except Exception as e:  # never let the flusher thread die
                logger.warning("Observability index flush error: %s", e)

    # ---- index buffering ----
    def _buffer_index(self, date: str, line: Dict[str, Any]) -> None:
        with self._index_lock:
            self._index_buffer.append((date, json.dumps(line, separators=(",", ":"))))
            due = len(self._index_buffer) >= self.index_flush_every
        if due:  # count trigger; the background flusher handles the time trigger
            self.flush_index()

    def flush_index(self, force: bool = False) -> None:
        with self._index_lock:
            if not self._index_buffer:
                return
            buffered = self._index_buffer
            self._index_buffer = []
        # Group by date so a midnight rollover lands lines under the right day.
        by_date: Dict[str, List[str]] = {}
        for date, line in buffered:
            by_date.setdefault(date, []).append(line)
        for date, lines in by_date.items():
            # Each flush writes a NEW immutable object (GCS objects can't be appended).
            name = f"{self._prefix}/date={date}/_index/part-{self._pid}-{uuid.uuid4().hex[:8]}.jsonl"
            payload = ("\n".join(lines) + "\n").encode("utf-8")
            if not self._upload_blob(name, payload, "application/x-ndjson"):
                # Re-buffer on failure so the lines aren't lost; the next flush retries.
                with self._index_lock:
                    self._index_buffer.extend((date, l) for l in lines)

    # ---- drop accounting ----
    def _count_dropped(self, nbytes: int, reason: str) -> None:
        with self._lock:
            self._count_dropped_locked(nbytes)
        self._maybe_log_drop(reason)

    def _count_dropped_locked(self, nbytes: int) -> None:
        self._stats["dropped_records"] += 1
        self._stats["dropped_bytes"] += nbytes

    def _maybe_log_drop(self, reason: str) -> None:
        now = self._clock()
        do_log = False
        with self._lock:  # rate-limit state must be read+written atomically
            if now - self._last_drop_log >= 1.0:  # at most ~1/s
                self._last_drop_log = now
                do_log = True
        if do_log:
            logger.warning("Observability dropped a record (%s); see /obs_stats", reason)

    # ---- introspection ----
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            s = dict(self._stats)
            s["bytes_in_flight"] = self._bytes_in_flight
        s["enabled"] = self.enabled
        s["accepting"] = self._accepting
        s["queue_depth"] = self._q.qsize()
        s["workers_alive"] = sum(1 for t in self._threads if t.is_alive())
        s["bucket"] = self.bucket_name
        return s


def build_uploader(static: Optional[Dict[str, Any]] = None) -> ObservabilityUploader:
    """Construct the uploader from module env config (used by app3 startup)."""
    return ObservabilityUploader(OBS_BUCKET, num_workers=OBS_NUM_WORKERS, static=static)
