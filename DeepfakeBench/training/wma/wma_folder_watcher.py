from __future__ import annotations

# -*- coding: utf-8 -*-
"""
WMA folder watcher (Windows)
- Monitors:  C:\Program Files\WMA\data\video\
- Detects new chunk folders: video_chunk_*
- Sends VIDEO-ONLY uplinks (frames per participant) via your existing gRPC proto
- Receives downlinks and logs them (JSON + TXT), like in cloud_client_tester.py

Run (PowerShell):
  $env:WMA_SERVER="34.116.214.60:50051"
  $env:WMA_ROOT="C:\Program Files\WMA"
  python .\wma_folder_watcher.py --meeting-id meet_local --client-id win-edge-01

Notes:
- No audio is sent.
- If you don’t run as Administrator, avoid writing to Program Files for state; this script
  defaults state/logs to %ProgramData%\WMA (override via env vars below).
"""

import os
import re
import cv2
import json
import time
import uuid
import grpc
import queue
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from threading import Lock
import urllib.request
import os
import threading
import urllib.request

# === Your generated protos (must be installed/available like in cloud_client_tester.py) ===
import wma_streaming_pb2 as pb2
import wma_streaming_pb2_grpc as pb2_grpc

# Optional for JSON logging of protobufs
try:
    from google.protobuf.json_format import MessageToJson
except Exception:
    MessageToJson = None

# ---------------------
# Configuration
# ---------------------
SERVER_ADDR = os.environ.get("WMA_SERVER", "34.116.214.60:50051")
ROOT_DIR = Path(os.environ.get("WMA_ROOT", r"C:\Program Files\WMA"))
DATA_DIR = Path(os.environ.get("WMA_DATA_DIR", str(ROOT_DIR / "data" / "video")))
STATE_FILE = Path(os.environ.get("WMA_STATE_FILE", r"C:\ProgramData\WMA\sent_index.json"))
LOG_DIR = Path(os.environ.get("WMA_LOG_DIR", r"C:\Program Files\WMA\my_logs"))
POLL_SEC = float(os.environ.get("WMA_POLL_SEC", "2.0"))  # how often to scan the folder

LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


# ---------------------
# Logging helpers
# ---------------------
def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_info(msg: str) -> None:
    print(f"[INFO {now()}] {msg}")


def log_recv(msg: str) -> None:
    print(f"[<-- RECV {now()}] {msg}")


def log_send(msg: str) -> None:
    print(f"[--> SENT {now()}] {msg}")


def log_err(msg: str) -> None:
    print(f"[ERROR {now()}] {msg}")


# ---------------------
# Persistent "already-sent" index
# ---------------------
def load_state() -> Dict[str, dict]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            log_err(f"Failed reading state file {STATE_FILE}: {e}")
    return {}


def save_state(state: Dict[str, dict]) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as e:
        log_err(f"Failed writing state file {STATE_FILE}: {e}")


# ---------------------
# Downlink logging (JSON + TXT, per-message)
# ---------------------
class DownlinkLogger:
    def __init__(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = LOG_DIR / f"downlinks_{ts}_msgs"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        print(f"[LOG] Downlinks will be saved to: {self.run_dir}")

    def write(self, msg) -> None:
        idx = self.counter
        self.counter += 1
        seq = getattr(msg, "sequence_number", None)
        base = f"downlink_{seq if seq is not None else 'nseq'}_{idx}"
        msg_dir = self.run_dir / base
        try:
            msg_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log_err(f"Failed to create log dir {msg_dir}: {e}")
            return

        # TXT (best-effort)
        try:
            txt_path = msg_dir / f"{base}.txt"
            txt_path.write_text(str(msg), encoding="utf-8")
        except Exception as e:
            log_err(f"Failed to write TXT {txt_path}: {e}")

        # JSON (best-effort)
        json_path = msg_dir / f"{base}.json"
        try:
            if MessageToJson:
                try:
                    js = MessageToJson(msg, including_default_value_fields=True, preserving_proto_field_name=True)
                except TypeError:
                    js = MessageToJson(msg)
            # Forward to local Electron UI (non-blocking best effort)
            try:
                _ui_post_async(js)  # js is your JSON string for this downlink
                # _post_local_ui(js, ports=(4588,)) # (optional) if you spin a 2nd popup-only app
            except Exception:
                pass
            else:
                js = json.dumps({"fallback_str": str(msg)})
            json_path.write_text(js, encoding="utf-8")
        except Exception as e:
            log_err(f"Failed to write JSON {json_path}: {e}")

        # Always tell you where it went
        print(f"[LOG] Saved downlink -> {json_path}")


# ---------------------
# POST downlink JSON to local Electron UI (if running)
# ---------------------
def _post_local_ui(json_payload: str, ports=(4587,)):
    """Best-effort fire-and-forget POST to local Electron bridge."""
    for p in ports:
        try:
            req = urllib.request.Request(
                url=f"http://127.0.0.1:{p}/downlink",
                data=json_payload.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            # very small timeout; we don't want to block the stream
            urllib.request.urlopen(req, timeout=0.25).read()
        except Exception:
            # swallow all errors; UI may not be up yet
            pass


# ---------------------
# Utilities for chunk detection
# ---------------------
CHUNK_DIR_RE = re.compile(r"^video_chunk_\d{8}_\d{6}_[A-Za-z0-9]+$")  # flexible; your names look like this


def is_chunk_dir(p: Path) -> bool:
    return p.is_dir() and CHUNK_DIR_RE.match(p.name) is not None


def has_manifest(chunk: Path) -> bool:
    # Accept either "chunk_manifest" or "chunk_manifest.json"
    return (chunk / "chunk_manifest").exists() or (chunk / "chunk_manifest.json").exists()


def participant_id_from_name(name: str) -> Optional[int]:
    m = re.search(r"participant_(\d+)", name)
    return int(m.group(1)) if m else None


FRAME_RE = re.compile(r"frame_(\d+)_crop_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def sorted_frame_paths(participant_dir: Path) -> List[Path]:
    frames = []
    for p in participant_dir.iterdir():
        if p.is_file():
            m = FRAME_RE.match(p.name)
            if m:
                idx = int(m.group(1))
                frames.append((idx, p))
    frames.sort(key=lambda t: t[0])
    return [p for _, p in frames]


# ===== UI Forwarding Config =====
BRIDGE_HOST = os.getenv("WMA_PANEL_HOST", "127.0.0.1")
BRIDGE_PORT = int(os.getenv("WMA_PANEL_PORT", "4587"))
UI_DEBUG = os.getenv("WMA_UI_DEBUG", "0") == "1"


def _ui_post_sync(json_payload: str):
    url = f"http://{BRIDGE_HOST}:{BRIDGE_PORT}/downlink"
    req = urllib.request.Request(
        url=url,
        data=json_payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # keep tiny timeout so we never stall the hot path
    with urllib.request.urlopen(req, timeout=0.25) as r:
        r.read()
    if UI_DEBUG:
        print(f"[UI] POST -> {url} ({len(json_payload)} bytes)")


def _ui_post_async(json_payload: str):
    # fire-and-forget so uplink/downlink loops never block
    threading.Thread(target=_try_ui_post, args=(json_payload,), daemon=True).start()


def _try_ui_post(js: str):
    try:
        _ui_post_sync(js)
    except Exception as e:
        # Only print if explicitly opted-in; otherwise keep silent
        if UI_DEBUG:
            print(f"[UI] POST failed: {e}")


# ---------------------
# Build VIDEO-ONLY Uplink messages for one chunk
# (Adapt field names if your proto differs; this follows a typical schema)
# ---------------------
def build_video_only_messages_for_chunk(chunk_dir: Path,
                                        meeting_id: str,
                                        session_id: str,
                                        client_id: str,
                                        seq_start: int = 1) -> list:
    participants_frames = []
    total_crops = 0

    for sub in sorted(chunk_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = re.search(r"participant_(\d+)", sub.name)
        if not m:
            continue
        pid_str = m.group(1)  # proto expects string participant_id

        # collect frames in order
        frames = sorted_frame_paths(sub)
        if not frames:
            continue

        crops = []
        for i, fpath in enumerate(frames):
            img_bytes = fpath.read_bytes()
            crops.append(pb2.ParticipantCrop(
                participant_id=pid_str,
                image_data=img_bytes,
                # bbox_* / confidence / timestamps optional; defaults ok
                sequence_number=i
            ))
        total_crops += len(crops)

        # one ParticipantFrame per participant
        pf = pb2.ParticipantFrame(
            participant_id=pid_str,
            crops=crops,
            meeting_id=meeting_id,
            session_id=session_id,
            chunk_id=chunk_dir.name,
            frame_count=len(crops),
            # original_frame_width/height optional
        )
        participants_frames.append(pf)

    if not participants_frames:
        return []

    uplink = pb2.Uplink(
        participants=participants_frames,  # (video-only)
        # audio left empty (we're not sending audio now)
        timestamp_ms=int(time.time() * 1000),
        client_id=client_id,
        sequence_number=seq_start,
    )

    log_send(f"Uplink seq={seq_start}, type=Video Only, crops={total_crops}")
    return [uplink]


# ---------------------
# gRPC Stream (uplink generator + downlink listener)
# ---------------------
class StreamClient:
    def __init__(self, server_addr: str):
        self.server_addr = server_addr
        self.channel = grpc.insecure_channel(server_addr)
        self.stub = pb2_grpc.StreamingServiceStub(self.channel)
        self.downlink_logger = DownlinkLogger()
        self._stop = threading.Event()

    def ping(self) -> None:
        try:
            resp = self.stub.Ping(pb2.PingRequest())
            log_info(f"Ping: {getattr(resp, 'status', 'ok')}, version: {getattr(resp, 'version', 'n/a')}")
        except Exception as e:
            log_err(f"Ping failed: {e}")

    def wait_until_available(self, ping_interval_sec: float = 2.0):
        while True:
            try:
                resp = self.stub.Ping(pb2.PingRequest())
                log_info(f"Server ready: {getattr(resp, 'status', 'ok')}, version: {getattr(resp, 'version', 'n/a')}")
                return
            except Exception as e:
                log_err(f"Server not reachable yet: {e}")
                time.sleep(ping_interval_sec)

    def stream(self, uplink_iter, on_ack=None, on_fail=None):
        """
        Live mode:
          - Any non-error downlink => ACK (oldest pending).
          - Error downlink => FAIL (oldest pending if no explicit mapping).
        """
        response_iterator = self.stub.StreamData(uplink_iter)

        def _downlink_worker():
            try:
                for msg in response_iterator:
                    err = getattr(msg, "error_message", "")

                    # Optional side messages
                    if getattr(msg, "screen_banner", None):
                        b = msg.screen_banner
                        log_recv(f"ScreenBanner: level={getattr(b, 'level', '?')}, ttl={getattr(b, 'ttl_ms', '?')}ms")

                    if err:
                        log_err(f"Downlink error: {err}")
                        if on_fail:
                            try:
                                on_fail(None, err)  # <- ignore server seq; fail oldest
                            except Exception as e:
                                log_err(f"on_fail error: {e}")
                    else:
                        if on_ack:
                            try:
                                on_ack(None)  # <- ignore server seq; ack oldest
                            except Exception as e:
                                log_err(f"on_ack error: {e}")

                    # Always persist the raw downlink
                    self.downlink_logger.write(msg)

            except grpc.RpcError as e:
                log_err(f"Downlink stream error: {e.code()} - {e.details()}")

        t = threading.Thread(target=_downlink_worker, daemon=True)
        t.start()
        return t


# ---------------------
# Folder watch loop
# ---------------------
def watch_and_stream(video_dir: Path,
                     client: "StreamClient",
                     meeting_id: str,
                     session_id: str,
                     client_id: str,
                     state: Dict[str, dict]) -> None:
    """
    Polls for new chunk folders and streams them once (live mode).
    Rules:
      - As soon as a chunk is queued, mark it as .queued and persist in `state` (no re-queue spam).
      - ANY non-error downlink is treated as an ACK -> mark .sent and finalize.
      - Errors are logged and finalized as .failed (no retry).
      - If server omits sequence_number, we finalize the oldest pending item.
    """
    log_info(f"Watching: {video_dir}")
    if not video_dir.exists():
        log_err(f"Path does not exist: {video_dir}")
        return

    q: "queue.Queue[pb2.Uplink]" = queue.Queue(maxsize=1000)
    stop_flag = {"stop": False}

    # track queued but not yet acked
    pending_by_seq: Dict[int, Dict[str, object]] = {}  # seq -> {"chunk_path": Path, "chunk_name": str}
    pending_lock = Lock()

    # prevent duplicate queueing within this run even if ACKs don’t come
    queued_once: set[str] = set()  # chunk_name strings

    def uplink_gen():
        while not stop_flag["stop"]:
            try:
                msg = q.get(timeout=1.0)
                if msg is None:
                    break
                yield msg
            except queue.Empty:
                continue

    def _pop_oldest_pending():
        with pending_lock:
            if not pending_by_seq:
                return None
            oldest_seq = min(pending_by_seq.keys())
            return oldest_seq, pending_by_seq.pop(oldest_seq)

    def on_ack(_ignored_seqno: Optional[int]):
        # finalize oldest pending item (if any)
        with pending_lock:
            if not pending_by_seq:
                log_info("ACK received but no pending items; ignoring.")
                return
            seqno, info = min(pending_by_seq.items(), key=lambda kv: kv[0])
            pending_by_seq.pop(seqno)

        chunk: Path = info["chunk_path"]
        chunk_name: str = info["chunk_name"]
        try:
            (chunk / ".sent").write_text(now(), encoding="utf-8")
        except Exception as e:
            log_err(f"Failed to write .sent for {chunk}: {e}")

        state[chunk_name] = {
            "sent_at": now(),
            "path": str(chunk),
            "status": "sent"
        }
        save_state(state)
        log_info(f"Chunk {chunk_name} marked as SENT (acked oldest pending).")

    def on_fail(seqno: Optional[int], err_msg: str):
        # If server didn’t echo seq, fail the oldest pending
        if seqno is None:
            popped = _pop_oldest_pending()
            if not popped:
                log_err("ERROR without sequence_number and no pending items; ignoring.")
                return
            seqno_, info = popped
            log_err(f"ERROR (no seq in downlink) → using oldest pending seq={seqno_}")
        else:
            with pending_lock:
                info = pending_by_seq.pop(seqno, None)
            if not info:
                log_err(f"ERROR seq={seqno} had no matching pending chunk.")
                return

        chunk: Path = info["chunk_path"]  # type: ignore[index]
        chunk_name: str = info["chunk_name"]  # type: ignore[index]
        try:
            (chunk / ".failed").write_text(f"{now()}\n{err_msg}", encoding="utf-8")
        except Exception as e:
            log_err(f"Failed to write .failed for {chunk}: {e}")

        state[chunk_name] = {
            "failed_at": now(),
            "path": str(chunk),
            "status": "error",
            "error": err_msg
        }
        save_state(state)
        log_err(f"Chunk {chunk_name} finalized as FAILED. No retry (live mode).")

    # Start stream with live-mode callbacks
    downlink_thread = client.stream(uplink_gen(), on_ack=on_ack, on_fail=on_fail)

    last_empty_log = 0.0
    seq = 1

    try:
        while True:
            # discover candidate chunks
            try:
                chunk_dirs = [d for d in video_dir.iterdir() if is_chunk_dir(d)]
            except FileNotFoundError:
                chunk_dirs = []

            new_chunks: List[Path] = []
            for c in sorted(chunk_dirs, key=lambda p: p.name):
                if c.name in state:  # finalized (sent/failed/ignored)
                    continue
                if (c / ".sent").exists() or (c / ".failed").exists():
                    # backfill state if present on disk
                    state[c.name] = {
                        "path": str(c),
                        "status": "sent" if (c / ".sent").exists() else "error"
                    }
                    continue
                if c.name in queued_once or (c / ".queued").exists():  # already queued; don't re-queue
                    queued_once.add(c.name)
                    continue
                if has_manifest(c):  # ready to send
                    new_chunks.append(c)

            if not new_chunks:
                now_ts = time.time()
                if now_ts - last_empty_log > 10:
                    log_info("No ready chunks found. Waiting…")
                    last_empty_log = now_ts
                time.sleep(POLL_SEC)
                continue

            for chunk in new_chunks:
                try:
                    msgs = build_video_only_messages_for_chunk(
                        chunk_dir=chunk,
                        meeting_id=meeting_id,
                        session_id=session_id,
                        client_id=client_id,
                        seq_start=seq
                    )
                    if not msgs:
                        log_info(f"Chunk {chunk.name}: no frames; marking ignored.")
                        state[chunk.name] = {"ignored": True, "path": str(chunk), "status": "ignored"}
                        save_state(state)
                        continue

                    # queue exactly once per chunk
                    for m in msgs:
                        q.put(m)
                        with pending_lock:
                            pending_by_seq[m.sequence_number] = {"chunk_path": chunk, "chunk_name": chunk.name}
                        seq += 1

                    # persist queued marker and state so we never re-queue
                    try:
                        (chunk / ".queued").write_text(now(), encoding="utf-8")
                    except Exception as e:
                        log_err(f"Failed to write .queued for {chunk}: {e}")
                    queued_once.add(chunk.name)
                    state[chunk.name] = {"queued_at": now(), "path": str(chunk), "status": "queued"}
                    save_state(state)
                    log_info(f"Chunk {chunk.name} queued (live mode; awaiting any response).")

                except Exception as e:
                    log_err(f"Failed processing {chunk.name}: {e}")

            time.sleep(POLL_SEC)

    except KeyboardInterrupt:
        log_info("Ctrl+C received, shutting down…")
    finally:
        stop_flag["stop"] = True
        q.put(None)
        downlink_thread.join(timeout=5.0)


# ---------------------
# Main
# ---------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="WMA Folder Watcher (video-only)")
    parser.add_argument("--server", default=SERVER_ADDR, help="gRPC server host:port")
    parser.add_argument("--watch-dir", default=str(DATA_DIR), help="Folder with video_chunk_* subfolders")
    parser.add_argument("--meeting-id", default=f"meet_{uuid.uuid4()}", help="Meeting ID to use")
    parser.add_argument("--session-id", default=f"sess_{uuid.uuid4()}", help="Session ID to use")
    parser.add_argument("--client-id", default=f"client_{uuid.uuid4()}", help="Client ID to use")
    args = parser.parse_args()

    log_info(f"Server: {args.server}")
    log_info(f"Watch dir: {args.watch_dir}")

    client = StreamClient(args.server)
    client.ping()

    state = load_state()
    try:
        client.wait_until_available()
        watch_and_stream(
            video_dir=Path(args.watch_dir),
            client=client,
            meeting_id=args.meeting_id,
            session_id=args.session_id,
            client_id=args.client_id,
            state=state
        )
    finally:
        client.stop()


if __name__ == "__main__":
    main()
