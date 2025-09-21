from __future__ import annotations

# -*- coding: utf-8 -*-
"""
WMA folder watcher (Windows)
- Monitors:
  - C:\Program Files\WMA\data\video\ for video_chunk_* folders
  - C:\Program Files\WMA\data\audio\ for audio_*.ogg files
- Sends VIDEO-ONLY and AUDIO-ONLY uplinks via your existing gRPC proto
- Receives downlinks and logs them (JSON + TXT), like in cloud_client_tester.py

Run (PowerShell):
  $env:WMA_SERVER="34.116.214.60:50051"
  $env:WMA_ROOT="C:\Program Files\WMA"
  python .\wma_folder_watcher.py --meeting-id meet_local --client-id win-edge-01

Notes:
- Video and Audio are sent in separate, independent uplink messages.
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
from typing import Dict, List, Tuple, Optional, Any
from threading import Lock
import urllib.request
import threading

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
VIDEO_DIR = Path(os.environ.get("WMA_VIDEO_DIR", str(ROOT_DIR / "data" / "video")))
AUDIO_DIR = Path(os.environ.get("WMA_AUDIO_DIR", str(ROOT_DIR / "data" / "audio")))
STATE_FILE = Path(os.environ.get("WMA_STATE_FILE", r"C:\ProgramData\WMA\sent_index.json"))
LOG_DIR = Path(os.environ.get("WMA_LOG_DIR", r"C:\Program Files\WMA\my_logs"))
POLL_SEC = float(os.environ.get("WMA_POLL_SEC", "2.0"))  # how often to scan the folders

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
# Utilities for data detection
# ---------------------
CHUNK_DIR_RE = re.compile(r"^video_chunk_\d{8}_\d{6}_[A-Za-z0-9]+$")  # flexible; your names look like this
FRAME_RE = re.compile(r"frame_(\d+)_crop_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def is_chunk_dir(p: Path) -> bool:
    return p.is_dir() and CHUNK_DIR_RE.match(p.name) is not None


def has_manifest(chunk: Path) -> bool:
    return (chunk / "chunk_manifest").exists() or (chunk / "chunk_manifest.json").exists()


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
    with urllib.request.urlopen(req, timeout=0.25) as r:
        r.read()
    if UI_DEBUG:
        print(f"[UI] POST -> {url} ({len(json_payload)} bytes)")


def _ui_post_async(json_payload: str):
    threading.Thread(target=_try_ui_post, args=(json_payload,), daemon=True).start()


def _try_ui_post(js: str):
    try:
        _ui_post_sync(js)
    except Exception as e:
        if UI_DEBUG:
            print(f"[UI] POST failed: {e}")


# ---------------------
# Build Uplink messages
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
        pid_str = m.group(1)
        frames = sorted_frame_paths(sub)
        if not frames:
            continue
        crops = []
        for i, fpath in enumerate(frames):
            img_bytes = fpath.read_bytes()
            crops.append(pb2.ParticipantCrop(
                participant_id=pid_str, image_data=img_bytes, sequence_number=i))
        total_crops += len(crops)
        pf = pb2.ParticipantFrame(
            participant_id=pid_str, crops=crops, meeting_id=meeting_id,
            session_id=session_id, chunk_id=chunk_dir.name, frame_count=len(crops))
        participants_frames.append(pf)
    if not participants_frames:
        return []
    uplink = pb2.Uplink(
        participants=participants_frames,
        timestamp_ms=int(time.time() * 1000), client_id=client_id, sequence_number=seq_start)
    log_send(f"Uplink seq={seq_start}, type=Video Only, crops={total_crops}, chunk={chunk_dir.name}")
    return [uplink]


def build_audio_only_message(audio_path: Path, meta_path: Path, client_id: str, seq: int) -> Optional[pb2.Uplink]:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        audio_props = meta.get("audio_properties", {})
        audio_batch = pb2.AudioBatch(
            ogg_data=audio_path.read_bytes(),
            start_ts_ms=meta.get("start_ts_ms"),
            duration_ms=meta.get("duration_ms"),
            meeting_id=meta.get("meeting_id"),
            session_id=meta.get("session_id"),
            chunk_id=meta.get("chunk_id"),
            sample_rate=audio_props.get("sample_rate"),
            channels=audio_props.get("channels"),
            bit_rate=audio_props.get("bit_rate"),
            codec=audio_props.get("codec"),
            container=audio_props.get("container"),
            frame_count=meta.get("frame_count"),
        )
        uplink = pb2.Uplink(
            audio=audio_batch,
            timestamp_ms=int(time.time() * 1000),
            client_id=client_id,
            sequence_number=seq,
        )
        log_send(f"Uplink seq={seq}, type=Audio Only, chunk_id={meta.get('chunk_id')}, file={audio_path.name}")
        return uplink
    except Exception as e:
        log_err(f"Failed to build audio message for {audio_path.name}: {e}")
        return None


# ---------------------
# gRPC Stream (uplink generator + downlink listener)
# ---------------------
class StreamClient:
    def __init__(self, server_addr: str):
        self.server_addr = server_addr
        self.channel = grpc.insecure_channel(server_addr)
        self.stub = pb2_grpc.StreamingServiceStub(self.channel)
        self.downlink_logger = DownlinkLogger()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()
        if self.channel:
            self.channel.close()
            log_info("gRPC channel closed.")

    def ping(self) -> None:
        try:
            resp = self.stub.Ping(pb2.PingRequest())
            log_info(f"Ping: {getattr(resp, 'status', 'ok')}, version: {getattr(resp, 'version', 'n/a')}")
        except Exception as e:
            log_err(f"Ping failed: {e}")

    def wait_until_available(self, ping_interval_sec: float = 2.0):
        while not self._stop_event.is_set():
            try:
                resp = self.stub.Ping(pb2.PingRequest())
                log_info(f"Server ready: {getattr(resp, 'status', 'ok')}, version: {getattr(resp, 'version', 'n/a')}")
                return
            except Exception as e:
                log_err(f"Server not reachable yet: {e}")
                time.sleep(ping_interval_sec)

    def stream(self, uplink_iter, on_ack=None, on_fail=None):
        response_iterator = self.stub.StreamData(uplink_iter)

        def _downlink_worker():
            try:
                for msg in response_iterator:
                    if self._stop_event.is_set(): break
                    err = getattr(msg, "error_message", "")
                    if getattr(msg, "screen_banner", None):
                        b = msg.screen_banner
                        log_recv(f"ScreenBanner: level={getattr(b, 'level', '?')}, ttl={getattr(b, 'ttl_ms', '?')}ms")
                    if err:
                        log_err(f"Downlink error: {err}")
                        if on_fail:
                            try:
                                on_fail(None, err)
                            except Exception as e:
                                log_err(f"on_fail error: {e}")
                    else:
                        if on_ack:
                            try:
                                on_ack(None)
                            except Exception as e:
                                log_err(f"on_ack error: {e}")
                    self.downlink_logger.write(msg)
            except grpc.RpcError as e:
                if e.code() != grpc.StatusCode.CANCELLED:
                    log_err(f"Downlink stream error: {e.code()} - {e.details()}")

        t = threading.Thread(target=_downlink_worker, daemon=True)
        t.start()
        return t


# ---------------------
# Folder watch loop
# ---------------------
def watch_and_stream(video_dir: Path,
                     audio_dir: Path,
                     client: "StreamClient",
                     meeting_id: str,
                     session_id: str,
                     client_id: str,
                     state: Dict[str, dict]) -> None:
    log_info(f"Watching Video: {video_dir}")
    log_info(f"Watching Audio: {audio_dir}")

    q: "queue.Queue[pb2.Uplink]" = queue.Queue(maxsize=1000)
    stop_flag = {"stop": False}

    pending_by_seq: Dict[int, Dict[str, Any]] = {}
    pending_lock = Lock()
    queued_once: set[str] = set()

    def uplink_gen():
        while not stop_flag["stop"]:
            try:
                msg = q.get(timeout=1.0)
                if msg is None: break
                yield msg
            except queue.Empty:
                continue

    def on_ack(_ignored_seqno: Optional[int]):
        with pending_lock:
            if not pending_by_seq:
                log_info("ACK received but no pending items; ignoring.")
                return
            seqno, info = min(pending_by_seq.items(), key=lambda kv: kv[0])
            pending_by_seq.pop(seqno)

        item_path: Path = info["path"]
        item_name: str = info["name"]
        item_type: str = info["type"]
        marker_path = item_path if item_type == "video" else item_path.with_suffix(item_path.suffix + '.sent')

        try:
            (marker_path / ".sent" if item_type == "video" else marker_path).write_text(now(), encoding="utf-8")
        except Exception as e:
            log_err(f"Failed to write .sent for {item_name}: {e}")

        state[item_name] = {"sent_at": now(), "path": str(item_path), "status": "sent"}
        save_state(state)
        log_info(f"{item_type.capitalize()} {item_name} marked as SENT (acked oldest pending).")

    def on_fail(seqno: Optional[int], err_msg: str):
        with pending_lock:
            if seqno is None:  # If server didn’t echo seq, fail the oldest pending
                if not pending_by_seq:
                    log_err("ERROR without sequence_number and no pending items; ignoring.")
                    return
                seqno, info = min(pending_by_seq.items(), key=lambda kv: kv[0])
                pending_by_seq.pop(seqno)
                log_err(f"ERROR (no seq in downlink) → using oldest pending seq={seqno}")
            else:
                info = pending_by_seq.pop(seqno, None)
            if not info:
                log_err(f"ERROR seq={seqno} had no matching pending item.")
                return

        item_path: Path = info["path"]
        item_name: str = info["name"]
        item_type: str = info["type"]
        marker_path = item_path if item_type == "video" else item_path.with_suffix(item_path.suffix + '.failed')

        try:
            (marker_path / ".failed" if item_type == "video" else marker_path).write_text(f"{now()}\n{err_msg}",
                                                                                          encoding="utf-8")
        except Exception as e:
            log_err(f"Failed to write .failed for {item_name}: {e}")

        state[item_name] = {"failed_at": now(), "path": str(item_path), "status": "error", "error": err_msg}
        save_state(state)
        log_err(f"{item_type.capitalize()} {item_name} finalized as FAILED. No retry.")

    downlink_thread = client.stream(uplink_gen(), on_ack=on_ack, on_fail=on_fail)
    last_empty_log = 0.0
    seq = 1

    try:
        while True:
            found_new_item = False

            # --- 1. VIDEO DISCOVERY ---
            try:
                for c in sorted(video_dir.iterdir(), key=lambda p: p.name):
                    if not is_chunk_dir(c): continue
                    if c.name in state or c.name in queued_once: continue
                    if (c / ".sent").exists() or (c / ".failed").exists() or (c / ".queued").exists():
                        queued_once.add(c.name)
                        continue
                    if has_manifest(c):
                        msgs = build_video_only_messages_for_chunk(c, meeting_id, session_id, client_id, seq)
                        if not msgs:
                            log_info(f"Chunk {c.name}: no frames; marking ignored.")
                            state[c.name] = {"ignored": True, "path": str(c), "status": "ignored"}
                        else:
                            for m in msgs:
                                q.put(m)
                                with pending_lock:
                                    pending_by_seq[m.sequence_number] = {"type": "video", "path": c, "name": c.name}
                                seq += 1
                            (c / ".queued").write_text(now(), encoding="utf-8")
                            state[c.name] = {"queued_at": now(), "path": str(c), "status": "queued"}
                            log_info(f"Video chunk {c.name} queued.")
                            found_new_item = True
                        queued_once.add(c.name)
                        save_state(state)
            except FileNotFoundError:
                log_err(f"Video directory not found: {video_dir}")
            except Exception as e:
                log_err(f"Error scanning video directory: {e}")

            # --- 2. AUDIO DISCOVERY ---
            try:
                for f in sorted(audio_dir.iterdir(), key=lambda p: p.name):
                    if not (f.is_file() and f.suffix == '.ogg'): continue
                    if f.name in state or f.name in queued_once: continue
                    meta_path = f.with_suffix(".ogg.meta.json")
                    if (f.with_suffix(".ogg.sent").exists() or
                            f.with_suffix(".ogg.failed").exists() or
                            f.with_suffix(".ogg.queued").exists()):
                        queued_once.add(f.name)
                        continue
                    if meta_path.exists():
                        msg = build_audio_only_message(f, meta_path, client_id, seq)
                        if msg:
                            q.put(msg)
                            with pending_lock:
                                pending_by_seq[msg.sequence_number] = {"type": "audio", "path": f, "name": f.name}
                            seq += 1
                            f.with_suffix(".ogg.queued").write_text(now(), encoding="utf-8")
                            state[f.name] = {"queued_at": now(), "path": str(f), "status": "queued"}
                            log_info(f"Audio file {f.name} queued.")
                            found_new_item = True
                        else:
                            state[f.name] = {"status": "error", "error": "build_failed"}
                        queued_once.add(f.name)
                        save_state(state)
            except FileNotFoundError:
                log_err(f"Audio directory not found: {audio_dir}")
            except Exception as e:
                log_err(f"Error scanning audio directory: {e}")

            if not found_new_item:
                now_ts = time.time()
                if now_ts - last_empty_log > 10:
                    log_info("No new video or audio found. Waiting…")
                    last_empty_log = now_ts
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
    parser = argparse.ArgumentParser(description="WMA Folder Watcher (Video and Audio)")
    parser.add_argument("--server", default=SERVER_ADDR, help="gRPC server host:port")
    parser.add_argument("--video-dir", default=str(VIDEO_DIR), help="Folder with video_chunk_* subfolders")
    parser.add_argument("--audio-dir", default=str(AUDIO_DIR), help="Folder with audio_*.ogg files")
    parser.add_argument("--meeting-id", required=True, help="Meeting ID to use")
    parser.add_argument("--session-id", default=f"sess_{uuid.uuid4()}", help="Session ID to use")
    parser.add_argument("--client-id", required=True, help="Client ID to use")
    args = parser.parse_args()

    log_info(f"Server: {args.server}")
    log_info(f"Video dir: {args.video_dir}")
    log_info(f"Audio dir: {args.audio_dir}")

    client = StreamClient(args.server)
    state = load_state()
    try:
        client.wait_until_available()
        watch_and_stream(
            video_dir=Path(args.video_dir),
            audio_dir=Path(args.audio_dir),
            client=client,
            meeting_id=args.meeting_id,
            session_id=args.session_id,
            client_id=args.client_id,
            state=state
        )
    except Exception as e:
        log_err(f"Unhandled exception in main loop: {e}")
    finally:
        client.stop()


if __name__ == "__main__":
    main()
