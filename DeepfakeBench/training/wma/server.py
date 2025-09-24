"""
Backend gRPC Server for WMA Streaming Service.

Implements bidirectional streaming communication with Service 5,
receives video participant frames and audio batches,
responds with banner actions for testing.
"""

import asyncio
import signal
import time
import uuid
import sys
import os
from concurrent import futures
from typing import AsyncIterator, Dict, Any, List

import grpc
from grpc import aio

# Import generated protobuf classes
import wma_streaming_pb2 as pb2
import wma_streaming_pb2_grpc as pb2_grpc

# Import backend components
from wma.storage.data_writer import BackendDataWriter
from wma.utils.banner_simulator import BannerSimulator

# For Effort detector
import threading, queue
import cv2, numpy as np, torch
from app3 import app as model_app, startup_event, calculate_analysis

# Participant state manager
from participant_manager import ParticipantManager

# for audio processing
import requests
import soundfile as sf
import io

# ──────────────────────────
# Inline image preprocessing (same as video_preprocessor)
# ──────────────────────────
_MODEL_IMG_SIZE = 224
_MODEL_NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)  # CLIP mean
_MODEL_NORM_STD = (0.26862954, 0.26130258, 0.27577711)  # CLIP std


def _prep_rgb_to_tensor(rgb_uint8):
    """
    Resize to 224x224 (INTER_AREA), then normalize with CLIP mean/std.
    Input: HxWx3 uint8 RGB np.ndarray
    Output: torch.FloatTensor [3, 224, 224]
    """
    # Ensure 224x224
    h, w = rgb_uint8.shape[:2]
    if (h, w) != (_MODEL_IMG_SIZE, _MODEL_IMG_SIZE):
        rgb_uint8 = cv2.resize(rgb_uint8, (_MODEL_IMG_SIZE, _MODEL_IMG_SIZE), interpolation=cv2.INTER_AREA)

    # To [C,H,W] float32 in [0,1]
    t = torch.from_numpy(rgb_uint8).permute(2, 0, 1).contiguous().float().div(255.0)

    # Normalize (no torchvision import needed)
    mean = torch.tensor(_MODEL_NORM_MEAN, dtype=t.dtype).view(3, 1, 1)
    std = torch.tensor(_MODEL_NORM_STD, dtype=t.dtype).view(3, 1, 1)
    t = (t - mean) / std
    return t


def get_base_path():
    """ Get the base path for the application, handling frozen executables. """
    if getattr(sys, 'frozen', False):
        # If run as a bundled executable, the base path is the directory of the exe.
        # sys.executable for the backend will point to python_backend.exe,
        # so we go up one level to the `collected_components` directory.
        return os.path.dirname(os.path.dirname(sys.executable))
    else:
        # If run as a script, go up one level from the script's location.
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Define the base path for the entire application
BASE_PATH = get_base_path()


# ---------- Background I/O worker (non-blocking) ----------
class FrameIOWorker:
    def __init__(self, maxsize: int = 10000):
        self.q = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._count = 0
        self._lock = threading.Lock()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def submit(self, frame_bytes: bytes):
        try:
            self.q.put_nowait(frame_bytes)
        except queue.Full:
            # drop or log; choose your policy
            pass

    def _loop(self):
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            # TODO: replace with your real I/O (upload, write, etc.)
            with self._lock:
                self._count += 1
                if self._count % 100 == 0:
                    print(f"[I/O] processed frames so far: {self._count}")
            self.q.task_done()

    def stop(self):
        self._stop.set()
        self.t.join(timeout=2)

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


# ---------- Model manager (load once, reuse) ----------
class ModelManager:
    def __init__(self):
        # Ensure app3 loads its models according to ENV (set in start_backend.py)
        if not hasattr(model_app, "state") or not getattr(model_app.state, "models", None):
            startup_event()  # app3's init hook

        # prefer "custom" if available, else "base"
        self.model = model_app.state.models.get("custom") or model_app.state.models["base"]
        self.model.eval()
        self.device = next(self.model.parameters()).device

        # Inline transform parameters (no torchvision import)
        self.model_img_size = _MODEL_IMG_SIZE
        self.threshold = float(os.getenv("WMA_INFER_THRESHOLD", "0.46"))
        self.batch_size = int(os.getenv("WMA_INFER_BATCH", "16"))

    @torch.inference_mode()
    def infer_probs(self, frame_rgbs: List[np.ndarray]) -> List[float]:
        """frame_rgbs: list of HxWx3 RGB uint8 arrays"""
        probs: List[float] = []
        tensors: List[torch.Tensor] = []
        for rgb in frame_rgbs:
            tensors.append(_prep_rgb_to_tensor(rgb))
            if len(tensors) == self.batch_size:
                probs.extend(self._forward_stack(tensors))
                tensors = []
        if tensors:
            probs.extend(self._forward_stack(tensors))
        return probs

    @torch.inference_mode()
    def infer_mean_decision(self, frames_rgb, bs=16, thr=None):
        """frames_rgb: list of HxWx3 RGB uint8 arrays"""
        device = self.device
        thr = float(thr if thr is not None else os.getenv("WMA_INFER_THRESHOLD", "0.46"))
        bs = int(os.getenv("WMA_INFER_BATCH", bs))
        probs, buf = [], []

        for rgb in frames_rgb:
            buf.append(_prep_rgb_to_tensor(rgb))  # -> [C,H,W]
            if len(buf) == bs:
                out = self.model({"image": torch.stack(buf).to(device)}, inference=True)["prob"]
                probs.extend(map(float, out.detach().cpu()))
                buf = []

        if buf:
            out = self.model({"image": torch.stack(buf).to(device)}, inference=True)["prob"]
            probs.extend(map(float, out.detach().cpu()))

        mean_prob = float(np.mean(probs)) if probs else 0.0
        decision = "FAKE" if mean_prob >= thr else "REAL"
        return {"decision": decision, "mean_prob": mean_prob, "probs": probs}

    def _forward_stack(self, tensors: List[torch.Tensor]) -> List[float]:
        batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)
        out = self.model({"image": batch}, inference=True)
        fake_probs = out["prob"].detach().float().cpu().tolist()
        return [float(p) for p in fake_probs]


class StreamingServiceImpl(pb2_grpc.StreamingServiceServicer):
    """Implementation of the WMA StreamingService."""

    def __init__(self):
        """Initialize the streaming service."""
        # data_directory = os.path.join(BASE_PATH, "data")
        data_directory = "/home/roee/repos/Effort-AIGI-Detection-Fork/DeepfakeBench/training/wma/data"
        self.data_writer = BackendDataWriter(data_directory)
        # self.banner_simulator = BannerSimulator(banner_probability=0.15,
        #                                         per_person_probability=1.0)  # 15% global, 100% per-person FOR IMMEDIATE TESTING

        # Disable random banners by default; we now use real inference
        self.banner_simulator = BannerSimulator(banner_probability=0.0, per_person_probability=0.0)

        # --- Real Effort detector (batched) ---
        self.mm = ModelManager()
        self.io = FrameIOWorker()
        self.margin = float(os.getenv("WMA_BAND_MARGIN", "0.05"))

        # --- Participant state manager ---
        self.participant_manager = ParticipantManager(
            threshold=self.mm.threshold,
            margin=self.margin
        )

        # --- ASV API config ---
        self.asv_api_url = os.getenv("ASV_API_URL", "http://34.125.106.206:8000/asv/predict-live")
        self.asv_api_timeout = int(os.getenv("ASV_API_TIMEOUT", "20"))

        # TTLs per level (ms)
        self.ttl_map = {
            pb2.GREEN: 2000,
            pb2.YELLOW: 3000,
            pb2.RED: 4000,
        }

        # Server state
        self.server_id = f"backend-{uuid.uuid4().hex[:8]}"
        self.active_streams = {}
        self.message_sequence = 0

        # Statistics
        self.stats = {
            "active_connections": 0,
            "total_connections": 0,
            "uplink_messages": 0,
            "downlink_messages": 0,
            "participant_batches": 0,
            "audio_batches": 0,
            "banners_sent": 0,
            "start_time": time.time()
        }

        print(f"[Backend] StreamingService initialized with server_id: {self.server_id}")

    def _decode_jpeg_to_rgb(self, image_bytes: bytes):
        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _band_level(self, mean_prob: float) -> int:
        """
        DEPRECATED in favor of ParticipantManager._calculate_band_level.
        Kept for any part of the code that might still use it directly, but
        the main banner logic now uses the stateful manager.
        """
        thr = self.mm.threshold
        m = self.margin
        if mean_prob >= thr + m:
            return pb2.RED
        elif mean_prob >= thr - m:
            return pb2.YELLOW
        else:
            return pb2.GREEN

    def _sanitize_participant_id(self, participant_id_raw: str) -> str:
        """
        Sanitize participant_id to handle JSON corruption in gRPC messages.
        
        The participant_id field sometimes contains raw JSON fragments instead of clean IDs.
        This method extracts the actual participant ID and sanitizes it for banner generation.
        
        Args:
            participant_id_raw: Raw participant_id from gRPC message
            
        Returns:
            Clean participant ID safe for use
        """
        try:
            # If it looks like a clean participant ID already, use it
            if len(participant_id_raw) < 50 and not any(c in participant_id_raw for c in ['"', '\n', ',', '{', '}']):
                # Sanitize for safety
                import re
                invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
                sanitized = re.sub(invalid_chars, '_', participant_id_raw).strip(' .')
                return sanitized[:50] if len(sanitized) > 50 else (sanitized or "participant_unknown")

            # Handle JSON corruption - extract participant ID from JSON fragment
            import re

            # Try to extract participant ID from corrupted data
            # Look for pattern: participant_[digits] before any JSON punctuation
            match = re.match(r'(participant_\d+)', participant_id_raw)
            if match:
                clean_id = match.group(1)
                invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
                sanitized = re.sub(invalid_chars, '_', clean_id).strip(' .')
                return sanitized[:50] if len(sanitized) > 50 else (sanitized or "participant_unknown")

            # Fallback: Take first part before any JSON punctuation and sanitize
            clean_id = participant_id_raw.split('"')[0].split(',')[0].strip()
            invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
            sanitized = re.sub(invalid_chars, '_', clean_id).strip(' .')
            return sanitized[:50] if len(sanitized) > 50 else (sanitized or "participant_unknown")

        except Exception:
            # Last resort: generate a safe ID
            import hashlib
            return f"participant_{hashlib.md5(participant_id_raw.encode()).hexdigest()[:8]}"

    async def StreamData(self, request_iterator: AsyncIterator[pb2.Uplink],
                         context: grpc.aio.ServicerContext) -> AsyncIterator[pb2.Downlink]:
        """
        Bidirectional streaming endpoint for video/audio data exchange.
        
        Args:
            request_iterator: Stream of Uplink messages from Service 5
            context: gRPC service context
            
        Yields:
            Downlink messages with banner actions and confirmations
        """
        client_id = None
        stream_id = f"stream-{uuid.uuid4().hex[:8]}"

        try:
            # Update connection stats
            self.stats["active_connections"] += 1
            self.stats["total_connections"] += 1

            print(f"[Backend] New streaming connection established: {stream_id}")

            # Process incoming messages
            async for uplink_msg in request_iterator:
                try:
                    # Extract client information
                    if not client_id and uplink_msg.client_id:
                        client_id = uplink_msg.client_id
                        self.active_streams[stream_id] = {
                            "client_id": client_id,
                            "start_time": time.time(),
                            "messages_received": 0
                        }
                        print(f"[Backend] Client identified: {client_id} on stream {stream_id}")

                    # Update message counter
                    self.stats["uplink_messages"] += 1
                    if stream_id in self.active_streams:
                        self.active_streams[stream_id]["messages_received"] += 1

                    # Process the message and send a standard ACK
                    # response = await self._process_uplink_message(uplink_msg, stream_id)
                    # if response:
                    #     self.stats["downlink_messages"] += 1
                    #     yield response

                    await self._process_uplink_message(uplink_msg, stream_id)

                    # --- INFERENCE SECTION ---

                    # 1. Send inference-driven per-participant banners for VIDEO
                    if uplink_msg.participants:
                        loop = asyncio.get_running_loop()
                        inference_banners = await loop.run_in_executor(
                            None, self._generate_inference_banners, uplink_msg
                        )
                        for banner_msg in inference_banners:
                            self.stats["banners_sent"] += 1
                            yield banner_msg

                    # 2. Send inference-driven global banner for AUDIO
                    if uplink_msg.HasField('audio'):
                        audio_banner_msg = await self._generate_audio_inference_banner(uplink_msg)
                        if audio_banner_msg:
                            self.stats["banners_sent"] += 1
                            yield audio_banner_msg

                except Exception as e:
                    print(f"[Backend] Error processing uplink message: {e}")
                    # Send error response
                    error_response = self._create_error_response(str(e))
                    yield error_response

        except grpc.aio.AbortedError:
            print(f"[Backend] Stream {stream_id} aborted by client")
        except Exception as e:
            print(f"[Backend] Stream {stream_id} error: {e}")
        finally:
            # Clean up connection
            if stream_id in self.active_streams:
                duration = time.time() - self.active_streams[stream_id]["start_time"]
                messages = self.active_streams[stream_id]["messages_received"]
                print(f"[Backend] Stream {stream_id} closed: {duration:.1f}s, {messages} messages")
                del self.active_streams[stream_id]

            self.stats["active_connections"] -= 1

    async def _process_uplink_message(self, uplink_msg: pb2.Uplink,
                                      stream_id: str) -> pb2.Downlink:
        """
        Process incoming uplink message.
        
        Args:
            uplink_msg: Uplink message from Service 5
            stream_id: Stream identifier
            
        Returns:
            Downlink response message
        """
        # Extract metadata
        uplink_metadata = {
            "timestamp_ms": uplink_msg.timestamp_ms,
            "client_id": uplink_msg.client_id,
            "sequence_number": uplink_msg.sequence_number,
            "stream_id": stream_id
        }

        # Process participant video frames
        if uplink_msg.participants:
            await self._process_participant_frames(uplink_msg.participants, uplink_metadata)
            self.stats["participant_batches"] += 1

        # Process audio batch
        if uplink_msg.HasField('audio'):
            await self._process_audio_batch(uplink_msg.audio, uplink_metadata)
            self.stats["audio_batches"] += 1

        # Generate response with possible banner
        return await self._create_response(uplink_msg)

    async def _process_participant_frames(self, participant_frames: list,
                                          metadata: Dict[str, Any]) -> None:
        """Process participant video frames."""
        try:
            # Write to storage (traditional format)
            chunk_path = self.data_writer.write_video_chunk(participant_frames, metadata)

            # Also create API-compliant multipart format for demonstration
            api_data = self.data_writer.create_multipart_api_data(participant_frames, metadata)

            # Log processing
            participant_count = len(participant_frames)
            total_crops = sum(len(pf.crops) for pf in participant_frames)
            total_files = len(api_data.get('files', {}))

            print(f"[Backend] Processed video batch: {participant_count} participants, "
                  f"{total_crops} crops -> {chunk_path}")
            print(f"[Backend] API format ready: manifest + {total_files} files for multipart/form-data")

        except Exception as e:
            print(f"[Backend] Error processing participant frames: {e}")

    async def _process_audio_batch(self, audio_batch: pb2.AudioBatch,
                                   metadata: Dict[str, Any]) -> None:
        """Process audio batch."""
        try:
            # Write to storage  
            audio_path = self.data_writer.write_audio_chunk(audio_batch, metadata)

            # Log processing
            frame_count = audio_batch.frame_count if hasattr(audio_batch, 'frame_count') else 0
            ogg_size = len(audio_batch.ogg_data) if hasattr(audio_batch, 'ogg_data') else 0
            print(f"[Backend] Processed audio batch: {audio_batch.chunk_id}, "
                  f"{frame_count} frames, {ogg_size} bytes, {audio_batch.duration_ms}ms -> {audio_path}")

        except Exception as e:
            print(f"[Backend] Error processing audio batch: {e}")

    # async def _create_response(self, uplink_msg: pb2.Uplink) -> pb2.Downlink:
    #     """Create response message with possible banner."""
    #     response = pb2.Downlink()
    #     response.timestamp_ms = int(time.time() * 1000)
    #     response.server_id = self.server_id
    #     response.sequence_number = self._next_sequence()
    #     response.received = True
    #
    #     # Check if we should generate a banner
    #     banner = self.banner_simulator.generate_banner()
    #     if banner:
    #         response.screen_banner.CopyFrom(banner)
    #         print(f"[Backend] Generated banner: {pb2.BannerLevel.Name(banner.level)} "
    #               f"TTL={banner.ttl_ms}ms ID={banner.action_id}")
    #
    #     return response

    async def _create_response(self, uplink_msg: pb2.Uplink) -> pb2.Downlink:
        """ACK-only response (no random banners)."""
        response = pb2.Downlink()
        response.timestamp_ms = int(time.time() * 1000)
        response.server_id = self.server_id
        response.sequence_number = self._next_sequence()
        response.received = True
        return response

    def _call_asv_api(self, audio_batch: pb2.AudioBatch) -> Dict[str, Any] | None:
        """
        Synchronous helper to decode audio and call the ASV API.
        This is designed to be run in a thread pool executor to avoid blocking asyncio.
        """
        try:
            # 1. Decode the in-memory OGG file into raw audio samples (NumPy array)
            # We use io.BytesIO to treat the `bytes` object like a file
            audio_data, sample_rate = sf.read(io.BytesIO(audio_batch.ogg_data))

            # 2. Prepare the payload for the API, as seen in live_test.py
            # The API expects a list of floats, not a NumPy array.
            payload = {
                "audio": audio_data.tolist(),
                "sample_rate": sample_rate,
                # These can be hardcoded or passed from the client if needed
                "window_step": 4000,
                "aggregation_window": 10.0,
                "aggregation_type": 'mean',
                "threshold": 0.85,
                "use_vad": True,
                "vol_norm": False,
            }

            # 3. Make the HTTP POST request
            print(f"[ASV API] Sending {len(audio_data) / sample_rate:.2f}s audio chunk for analysis...")
            response = requests.post(self.asv_api_url, json=payload, timeout=self.asv_api_timeout)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            result = response.json()
            print(
                f"[ASV API] Received response: prediction={result.get('prediction')}, confidence={result.get('confidence'):.3f}")
            return result

        except requests.exceptions.RequestException as e:
            print(f"[ASV API] Error calling API: {e}")
            return None
        except Exception as e:
            # This can catch errors from sf.read if the OGG data is corrupt
            print(f"[ASV API] Error processing audio for API call: {e}")
            return None

    async def _generate_audio_inference_banner(self, uplink_msg: pb2.Uplink) -> pb2.Downlink | None:
        """
        Processes an audio batch, calls the ASV API, and generates a global banner.
        """
        loop = asyncio.get_running_loop()

        audio_batch = uplink_msg.audio
        request_seq = uplink_msg.sequence_number

        # Run the blocking I/O (HTTP request) in a separate thread
        api_result = await loop.run_in_executor(None, self._call_asv_api, audio_batch)

        if not api_result or 'prediction' not in api_result:
            return None

        # Map the API prediction to a banner level
        # Assuming `prediction` is boolean or 0/1 where True/1 is REAL
        is_real = api_result['prediction']
        level = pb2.GREEN if is_real else pb2.RED

        now_ms = int(time.time() * 1000)
        ttl_ms = self.ttl_map.get(level, 3000)

        # Build a GLOBAL ScreenBanner
        banner = pb2.ScreenBanner()
        banner.level = level
        banner.ttl_ms = ttl_ms
        banner.placement = "TopCenter"
        banner.action_id = f"act-audio-{uuid.uuid4().hex[:8]}"
        banner.scope = "global"
        banner.scope_enum = pb2.SCOPE_GLOBAL
        banner.banner_type = "audio_ok" if is_real else "audio_alert"
        banner.expiry_timestamp_ms = now_ms + ttl_ms

        # Wrap into Downlink message
        down = pb2.Downlink()
        down.timestamp_ms = now_ms
        down.server_id = self.server_id
        down.sequence_number = request_seq  # Echo the request sequence number
        down.received = True
        down.screen_banner.CopyFrom(banner)

        print(f"[Backend] Generated GLOBAL audio banner -> {pb2.BannerLevel.Name(level)}")

        return down

    def _generate_inference_banners(self, uplink_msg: pb2.Uplink) -> list:
        """
        Generate per-participant banners using the stateful ParticipantManager
        to ensure stable and non-flickering verdicts.
        """
        responses = []
        now_ms = int(time.time() * 1000)
        request_seq = uplink_msg.sequence_number

        for pf in uplink_msg.participants:
            # Sanitize participant id (your existing utility)
            pid_raw = getattr(pf, "participant_id", "") or ""
            participant_id = self._sanitize_participant_id(pid_raw)

            #  Handle the special [RESTART] signal
            if participant_id == "[RESTART]":
                print("[Backend] Received [RESTART] signal from client. Resetting all participant states.")
                self.participant_manager.reset_all()
                continue

            # Decode all crops to RGB and push bytes to non-blocking I/O counter
            rgbs = []
            for crop in pf.crops:
                img_bytes = getattr(crop, "image_data", b"")
                if not img_bytes:
                    continue
                self.io.submit(img_bytes)
                rgb = self._decode_jpeg_to_rgb(img_bytes)
                if rgb is not None:
                    rgbs.append(rgb)

            if not rgbs:
                continue

            # Run inference to get probabilities (16 micro-batch by default)
            res = self.mm.infer_mean_decision(rgbs)
            individual_probs = res["probs"]

            # Handle the new return type from the manager
            manager_result = self.participant_manager.process_and_decide(
                participant_id, individual_probs
            )

            # Only create a banner if the manager returns a result
            if manager_result is not None:
                new_verdict_level, confidence_score = manager_result
                ttl_ms = self.ttl_map.get(new_verdict_level, 2500)

                # --- CONFIDENCE ENCODING LOGIC ---
                # 1. Calculate the true expiry timestamp.
                true_expiry_ms = now_ms + ttl_ms

                # 2. Convert confidence (0.0 to 1.0) to a two-digit integer (0 to 99).
                confidence_code = int(confidence_score * 100) % 100

                # 3. Create the encoded timestamp: zero out the last 3 digits of the
                #    true expiry and add the confidence code.
                #    This "hides" the confidence in a way that minimally affects the
                #    absolute expiry time for old clients.
                encoded_expiry_ms = (true_expiry_ms // 1000) * 1000 + confidence_code

                # --- BANNER CONSTRUCTION ---
                # Revert banner_type to its simple, original form for the old client.
                banner_type = "alert" if new_verdict_level == pb2.RED else \
                    ("attention" if new_verdict_level == pb2.YELLOW else "info")

                # Build ScreenBanner (per-participant)
                banner = pb2.ScreenBanner()
                banner.level = new_verdict_level
                banner.ttl_ms = ttl_ms
                banner.placement = "TopRight"
                banner.action_id = f"act-{uuid.uuid4().hex[:8]}"
                banner.scope = "participant"
                banner.scope_enum = pb2.SCOPE_PARTICIPANT
                banner.participant_id = participant_id
                banner.banner_type = banner_type  # Use the simple type
                banner.expiry_timestamp_ms = encoded_expiry_ms  # Use the encoded timestamp

                # Wrap into Downlink
                down = pb2.Downlink()
                down.timestamp_ms = now_ms
                down.server_id = self.server_id
                down.sequence_number = request_seq
                down.received = True
                down.screen_banner.CopyFrom(banner)

                print(f"[Backend] Sending banner for {participant_id}. "
                      f"Level={pb2.BannerLevel.Name(new_verdict_level)}, "
                      f"Conf={confidence_score:.3f}, "
                      f"EncodedExpiry={encoded_expiry_ms} (code={confidence_code})")

                responses.append(down)

        return responses

    async def _check_for_banner_trigger(self) -> pb2.Downlink:
        """Check for additional banner triggers (simulating async events)."""
        # Small chance of additional banners (simulating external events)
        if self.banner_simulator.should_generate_banner():
            banner = self.banner_simulator.generate_banner()
            if banner:
                response = pb2.Downlink()
                response.timestamp_ms = int(time.time() * 1000)
                response.server_id = self.server_id
                response.sequence_number = self._next_sequence()
                response.received = True
                response.screen_banner.CopyFrom(banner)

                print(f"[Backend] Async banner trigger: {pb2.BannerLevel.Name(banner.level)} "
                      f"TTL={banner.ttl_ms}ms ID={banner.action_id}")
                return response

        return None

    def _create_error_response(self, error_message: str) -> pb2.Downlink:
        """Create error response."""
        response = pb2.Downlink()
        response.timestamp_ms = int(time.time() * 1000)
        response.server_id = self.server_id
        response.sequence_number = self._next_sequence()
        response.received = False
        response.error_message = error_message
        return response

    def _next_sequence(self) -> int:
        """Get next message sequence number."""
        self.message_sequence += 1
        return self.message_sequence

    async def Ping(self, request: pb2.PingRequest,
                   context: grpc.aio.ServicerContext) -> pb2.PingResponse:
        """Handle ping requests."""
        response = pb2.PingResponse()
        response.status = "ok"
        response.server_time_ms = int(time.time() * 1000)
        response.version = "1.0.0-stub"
        response.request_timestamp_ms = request.timestamp_ms

        print(f"[Backend] Ping from client {request.client_id}")
        return response

    async def _generate_per_person_banners(self, participant_frames: list) -> list:
        """
        Generate per-person banners for participants.
        
        Args:
            participant_frames: List of ParticipantFrame messages
            
        Returns:
            List of Downlink messages with per-person banners
        """
        responses = []

        # Extract AND SANITIZE participant IDs  
        print(f"[Backend] Processing {len(participant_frames)} participant frames for per-person banners")
        raw_participant_ids = []
        sanitized_participant_ids = []

        for i, pf in enumerate(participant_frames):
            print(f"[Backend] Frame {i + 1}: participant_id='{pf.participant_id}' has_id={bool(pf.participant_id)}")
            if pf.participant_id:
                raw_id = pf.participant_id
                sanitized_id = self._sanitize_participant_id(raw_id)
                raw_participant_ids.append(raw_id)
                sanitized_participant_ids.append(sanitized_id)
                print(f"[Backend] Frame {i + 1}: raw_id='{raw_id}' -> sanitized_id='{sanitized_id}'")

        print(
            f"[Backend] Extracted {len(sanitized_participant_ids)} participant IDs for banner generation: {sanitized_participant_ids}")

        # 🔧 TEMPORARY FIX FOR TESTING: If no participant IDs received, generate test ones
        if not sanitized_participant_ids and len(participant_frames) > 0:
            print(f"[Backend] ⚠️ No participant IDs received, generating test participant IDs for debugging")
            test_participant_ids = [f"participant_{i + 1000}" for i in range(min(3, len(participant_frames)))]
            sanitized_participant_ids = test_participant_ids
            print(f"[Backend] Using test participant IDs: {sanitized_participant_ids}")

        # Generate banners for participants with clean IDs
        banners = self.banner_simulator.generate_participant_banners(sanitized_participant_ids)
        print(f"[Backend] Generated {len(banners)} per-person banners")

        # Create response messages for each banner
        for banner in banners:
            response = pb2.Downlink()
            response.timestamp_ms = int(time.time() * 1000)
            response.server_id = self.server_id
            response.sequence_number = self._next_sequence()
            response.received = True
            response.screen_banner.CopyFrom(banner)

            print(f"[Backend] Generated per-person banner for {banner.participant_id}: "
                  f"{pb2.BannerLevel.Name(banner.level)} "
                  f"TTL={banner.ttl_ms}ms Type={banner.banner_type} ID={banner.action_id} "
                  f"Scope={banner.scope} ScopeEnum={banner.scope_enum}")

            # DEBUG: Check what fields are actually set in the banner
            print(
                f"[Backend] Banner protobuf fields - participant_id='{banner.participant_id}' banner_type='{banner.banner_type}' scope='{banner.scope}' level={banner.level}")

            # DEBUG: Check what fields are set in the response
            print(f"[Backend] Downlink protobuf fields - has_screen_banner={response.HasField('screen_banner')} " +
                  f"screen_banner.participant_id='{response.screen_banner.participant_id}' " +
                  f"screen_banner.banner_type='{response.screen_banner.banner_type}' " +
                  f"screen_banner.scope='{response.screen_banner.scope}'")

            responses.append(response)

        return responses

    def get_statistics(self) -> Dict[str, Any]:
        """Get server statistics."""
        uptime = time.time() - self.stats["start_time"]
        stats = self.stats.copy()
        stats.update({
            "uptime_seconds": uptime,
            "data_writer_stats": self.data_writer.get_statistics(),
            "banner_simulator_stats": self.banner_simulator.get_banner_stats(),
            "active_streams": len(self.active_streams)
        })

        if uptime > 0:
            stats["messages_per_second"] = stats["uplink_messages"] / uptime

        return stats


async def serve():
    """Start the gRPC server."""
    # Raise gRPC message size limits (default is 4 MiB). 50 MiB is usually plenty.
    MAX_MSG_MB = 20
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_receive_message_length', MAX_MSG_MB * 1024 * 1024),
            ('grpc.max_send_message_length', MAX_MSG_MB * 1024 * 1024),
        ],
    )
    shutdown_event = asyncio.Event()

    # Add service
    service_impl = StreamingServiceImpl()
    pb2_grpc.add_StreamingServiceServicer_to_server(service_impl, server)

    # Configure server address
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)

    # Setup graceful shutdown handlers
    def signal_handler(signum, frame):
        print(f"[Backend] Received signal {signum}, initiating graceful shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    print(f"[Backend] Starting gRPC server on {listen_addr}")
    await server.start()

    # Print startup information
    print(f"[Backend] Server started successfully")
    print(f"[Backend] Server ID: {service_impl.server_id}")
    print(f"[Backend] Data directory: data/")
    print(f"[Backend] Banner probability: {service_impl.banner_simulator.banner_probability}")
    print(f"[Backend] Ready to receive streaming data from Service 5")

    try:
        # Keep server running and print stats periodically
        while not shutdown_event.is_set():
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=30.0)
                break  # Shutdown event was set
            except asyncio.TimeoutError:
                # Print stats every 30 seconds
                stats = service_impl.get_statistics()
                print(f"[Backend] Stats: {stats['active_connections']} active, "
                      f"{stats['uplink_messages']} uplink msgs, "
                      f"{stats['participant_batches']} video batches, "
                      f"{stats['audio_batches']} audio batches, "
                      f"{stats['banners_sent']} banners sent")

    except KeyboardInterrupt:
        print("[Backend] Shutting down server...")
    finally:
        print("[Backend] Closing active connections...")
        await server.stop(5)


if __name__ == '__main__':
    print("=== WMA Backend gRPC Server ===")
    print("Stub implementation for testing Service 5 communication")
    print("Press Ctrl+C to stop\n")

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        print("\n[Backend] Server stopped")
