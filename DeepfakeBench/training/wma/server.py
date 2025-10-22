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
import argparse
from concurrent import futures
from typing import AsyncIterator, Dict, Any, List
import threading
from collections import deque
import cv2
import numpy as np

import grpc
import grpc.aio
from pydub import AudioSegment
import io
import queue
import requests
import soundfile as sf

# For HTTP API calls
import aiohttp

# Import generated protobuf classes
import wma_streaming_pb2 as pb2
import wma_streaming_pb2_grpc as pb2_grpc

# Import backend components
from wma.storage.data_writer import BackendDataWriter
from wma.utils.banner_simulator import BannerSimulator

# Participant state manager
from participant_manager import ParticipantManager, AudioWindowManager

# Participant name matcher
from participant_name_matcher import ParticipantNameMatcher

# for audio processing (keep for audio endpoint)

# Global variables for GCP buckets
GCP_VIDEO_BUCKET = None
GCP_AUDIO_BUCKET = None
ENABLE_BUCKET_SAVE = False
IO_WORKER_COUNT = 2
DEBUG_MODE = True

# ──────────────────────────
# Image utilities (keeping decode function for debugging)
# ──────────────────────────


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


# ---------- Background I/O worker (generic for video/audio) ----------
class MediaIOWorker:
    """Generic I/O worker for processing media data (video frames or audio chunks)."""

    def __init__(self, name: str, bucket_name: str = None, maxsize: int = 10000):
        self.name = name
        self.bucket_name = bucket_name
        self.q = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._count = 0
        self._lock = threading.Lock()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

        # Initialize GCP client if bucket is specified and saving is enabled
        self.gcp_client = None
        self.gcp_bucket = None
        if ENABLE_BUCKET_SAVE and bucket_name:
            try:
                from google.cloud import storage
                self.gcp_client = storage.Client()
                self.gcp_bucket = self.gcp_client.bucket(bucket_name)
                print(f"[{self.name}] Initialized GCP bucket: {bucket_name}")
            except Exception as e:
                print(f"[{self.name}] Failed to initialize GCP bucket {bucket_name}: {e}")

    def submit(self, data: bytes, metadata: Dict[str, Any] = None):
        """Submit data for processing."""
        try:
            item = {"data": data, "metadata": metadata or {}}
            self.q.put_nowait(item)
        except queue.Full:
            # drop or log; choose your policy
            print(f"[{self.name}] Queue full, dropping item")

    def _loop(self):
        """Main processing loop."""
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            # Process the item
            self._process_item(item)

            with self._lock:
                self._count += 1
                if self._count % 100 == 0:
                    print(f"[{self.name}] processed items so far: {self._count}")

            self.q.task_done()

    def _process_item(self, item: Dict[str, Any]):
        """Process a single item - override for specific behavior."""
        data = item["data"]
        metadata = item["metadata"]

        if ENABLE_BUCKET_SAVE and self.gcp_bucket:
            self._save_to_bucket(data, metadata)
        else:
            # Just count for now - could add local file saving here
            pass

    def _save_to_bucket(self, data: bytes, metadata: Dict[str, Any]):
        """Save data to GCP bucket with MP3 support."""
        try:
            # Generate unique filename
            timestamp = int(time.time() * 1000)
            file_id = uuid.uuid4().hex[:8]

            if self.name.lower().startswith('video'):
                filename = f"video/{timestamp}_{file_id}.jpg"
                content_type = "image/jpeg"
            else:  # audio
                # Use MP3 format if converted, otherwise fallback to original
                audio_format = metadata.get('format', 'ogg')
                if audio_format == 'mp3':
                    filename = f"audio/{timestamp}_{file_id}.mp3"
                    content_type = "audio/mpeg"
                else:
                    filename = f"audio/{timestamp}_{file_id}.{audio_format}"
                    content_type = f"audio/{audio_format}"

            # Upload to bucket
            blob = self.gcp_bucket.blob(filename)
            blob.upload_from_string(data, content_type=content_type)

            # Set metadata if provided
            if metadata:
                blob.metadata = {k: str(v) for k, v in metadata.items()}
                blob.patch()

            print(f"[{self.name}] Saved to bucket: {filename} ({len(data)} bytes)")

        except Exception as e:
            print(f"[{self.name}] Error saving to bucket: {e}")

    def stop(self):
        """Stop the worker."""
        self._stop.set()
        self.t.join(timeout=2)

    @property
    def count(self) -> int:
        """Get processed item count."""
        with self._lock:
            return self._count


# ---------- API manager for video inference ----------
class VideoAPIManager:
    def __init__(self):
        self.api_host = os.getenv("VIDEO_API_HOST", "34.16.217.28")
        self.api_port = os.getenv("VIDEO_API_PORT", "8999")
        self.api_timeout = int(os.getenv("VIDEO_API_TIMEOUT", "30"))
        self.yolo_conf_threshold = float(os.getenv("VIDEO_YOLO_CONF_THRESHOLD", "0.80"))
        self.threshold = float(os.getenv("WMA_INFER_THRESHOLD", "0.75"))
        
        self.api_url = f"http://{self.api_host}:{self.api_port}/check_frame_batch"
        
        print(f"[VideoAPIManager] Initialized with API URL: {self.api_url}")
        print(f"[VideoAPIManager] YOLO confidence threshold: {self.yolo_conf_threshold}")
        print(f"[VideoAPIManager] Inference threshold: {self.threshold}")

    async def infer_probs_from_bytes(self, image_bytes_list: List[bytes]) -> List[float]:
        """Send image bytes to API and return probabilities."""
        if not image_bytes_list:
            return []
            
        try:
            if DEBUG_MODE:
                print(f"[VideoAPIManager] Sending {len(image_bytes_list)} frames to API: {self.api_url}")
            
            # Prepare multipart form data
            data = aiohttp.FormData()
            for i, img_bytes in enumerate(image_bytes_list):
                data.add_field('files', img_bytes, filename=f'frame_{i}.jpg', content_type='image/jpeg')
            
            # API parameters
            params = {
                'model_type': 'custom',
                'threshold': str(self.threshold),
                'debug': 'false',
                'yolo_conf_threshold': str(self.yolo_conf_threshold)
            }
            
            timeout = aiohttp.ClientTimeout(total=self.api_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.api_url, data=data, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        probs = result.get('probs', [])
                        if DEBUG_MODE:
                            print(f"[VideoAPIManager] API response: {len(probs)} frame probabilities")
                        return probs
                    else:
                        error_text = await response.text()
                        print(f"[VideoAPIManager] API error {response.status}: {error_text}")
                        return []
                        
        except asyncio.TimeoutError:
            print(f"[VideoAPIManager] API timeout after {self.api_timeout}s")
            return []
        except Exception as e:
            print(f"[VideoAPIManager] API call failed: {e}")
            return []

    async def infer_mean_decision_from_bytes(self, image_bytes_list: List[bytes], thr=None) -> Dict[str, Any]:
        """Send image bytes to API and return decision with probabilities."""
        probs = await self.infer_probs_from_bytes(image_bytes_list)
        
        thr = float(thr if thr is not None else self.threshold)
        mean_prob = float(np.mean(probs)) if probs else 0.0
        decision = "FAKE" if mean_prob >= thr else "REAL"
        
        return {"decision": decision, "mean_prob": mean_prob, "probs": probs}


class StreamingServiceImpl(pb2_grpc.StreamingServiceServicer):
    """Implementation of the WMA StreamingService."""

    def __init__(self):
        """Initialize the streaming service."""
        # data_directory = os.path.join(BASE_PATH, "data")
        data_directory = "/home/roee/repos/Effort-AIGI-Detection-Fork/DeepfakeBench/training/wma/data"
        self.data_writer = BackendDataWriter(data_directory)

        # Disable random banners by default; we now use real inference
        self.banner_simulator = BannerSimulator(banner_probability=0.0, per_person_probability=0.0)

        # --- Video API manager ---
        self.video_api = VideoAPIManager()

        # --- I/O Workers for video and audio (only if bucket saving is enabled) ---
        self.video_io_workers = []
        self.audio_io_workers = []

        if ENABLE_BUCKET_SAVE:
            # Create video workers
            for i in range(IO_WORKER_COUNT):
                worker = MediaIOWorker(f"Video-IO-{i + 1}", GCP_VIDEO_BUCKET)
                self.video_io_workers.append(worker)

            # Create audio workers
            for i in range(IO_WORKER_COUNT):
                worker = MediaIOWorker(f"Audio-IO-{i + 1}", GCP_AUDIO_BUCKET)
                self.audio_io_workers.append(worker)

            print(
                f"[Backend] Initialized {len(self.video_io_workers)} video workers and {len(self.audio_io_workers)} audio workers")
            print(f"[Backend] GCP bucket saving enabled - Video: {GCP_VIDEO_BUCKET}, Audio: {GCP_AUDIO_BUCKET}")
        else:
            print(f"[Backend] GCP bucket saving disabled - no I/O workers created")

        self.margin = float(os.getenv("WMA_BAND_MARGIN", "0.05"))

        # --- Participant state manager ---
        self.participant_manager = ParticipantManager(
            threshold=self.video_api.threshold,
            margin=self.margin
        )

        # --- Participant name matcher ---
        # Initialize with configurable sensitivity (can be adjusted via env var)
        name_match_threshold = float(os.getenv("WMA_NAME_MATCH_THRESHOLD", "0.3"))
        self.name_matcher = ParticipantNameMatcher(similarity_threshold=name_match_threshold)
        print(f"[Backend] Participant name matching threshold: {name_match_threshold}")

        # --- Audio sliding window manager ---
        self.audio_window_manager = AudioWindowManager()

        # --- ASV API config ---
        self.asv_api_url = os.getenv("ASV_API_URL", "http://34.125.106.206:8000/asv/predict")
        self.asv_api_timeout = int(os.getenv("ASV_API_TIMEOUT", "20"))

        fixed_ttl = 1000000

        self.ttl_map = {
            pb2.GREEN: fixed_ttl,
            pb2.YELLOW: fixed_ttl,
            pb2.RED: fixed_ttl,
        }

        # # TTLs per level (ms)
        # self.ttl_map = {
        #     pb2.GREEN: 2000,
        #     pb2.YELLOW: 3000,
        #     pb2.RED: 4000,
        # }

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

    def _get_next_video_worker(self) -> MediaIOWorker:
        """Round-robin selection of video workers."""
        if not self.video_io_workers:
            return None
        return self.video_io_workers[self.stats["participant_batches"] % len(self.video_io_workers)]

    def _get_next_audio_worker(self) -> MediaIOWorker:
        """Round-robin selection of audio workers."""
        if not self.audio_io_workers:
            return None
        return self.audio_io_workers[self.stats["audio_batches"] % len(self.audio_io_workers)]

    def _decode_image_to_rgb(self, image_bytes: bytes):
        """
        Decode image bytes to RGB format.
        Supports both JPEG and PNG formats.
        """
        try:
            # Use cv2.imdecode which automatically detects format
            arr = np.frombuffer(image_bytes, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[Backend] Failed to decode image data ({len(image_bytes)} bytes)")
                return None
            # Convert BGR to RGB
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[Backend] Error decoding image: {e}")
            return None

    def _band_level(self, mean_prob: float) -> int:
        """
        DEPRECATED in favor of ParticipantManager._calculate_band_level.
        Kept for any part of the code that might still use it directly, but
        the main banner logic now uses the stateful manager.
        """
        thr = self.video_api.threshold
        m = self.margin
        if mean_prob >= thr + m:
            return pb2.RED
        elif mean_prob >= thr - m:
            return pb2.YELLOW
        else:
            return pb2.GREEN

    def _calculate_confidence_label(self, verdict_level: int, confidence_score: float,
                                   green_split: tuple = (0.5, 0.25, 0.25),
                                   red_split: tuple = (1/3, 1/3, 1/3)) -> str:
        """
        Calculate a confidence label (High/Medium/Low) based on verdict and score.
        
        For GREEN verdicts: Lower scores indicate higher confidence (it's real)
        For RED verdicts: Higher scores indicate higher confidence (it's fake)
        For YELLOW verdicts: Return "Uncertain" (no confidence calculation)
        
        Args:
            verdict_level: The banner level (GREEN, YELLOW, or RED)
            confidence_score: The mean probability score (0.0 to 1.0)
            green_split: Tuple of (high_fraction, medium_fraction, low_fraction) for GREEN range
                        Default: (0.5, 0.25, 0.25) - half high confidence, quarter medium, quarter low
            red_split: Tuple of (low_fraction, medium_fraction, high_fraction) for RED range
                      Default: (1/3, 1/3, 1/3) - equal thirds
            
        Returns:
            Confidence label: "High", "Medium", "Low", or "Uncertain"
        """
        thr = self.video_api.threshold
        m = self.margin
        
        if verdict_level == pb2.YELLOW:
            return "Uncertain"
        
        elif verdict_level == pb2.GREEN:
            # GREEN range: [0, threshold - margin]
            green_max = thr - m
            
            # Validate split proportions sum to 1.0
            assert abs(sum(green_split) - 1.0) < 0.001, "green_split must sum to 1.0"
            
            high_fraction, medium_fraction, low_fraction = green_split
            
            # Calculate boundaries
            high_boundary = green_max * high_fraction
            medium_boundary = green_max * (high_fraction + medium_fraction)
            
            # Map: A->High, B->Medium, C->Low (lower score = more confident it's real)
            if confidence_score < high_boundary:
                return "High"
            elif confidence_score < medium_boundary:
                return "Medium"
            else:
                return "Low"
        
        elif verdict_level == pb2.RED:
            # RED range: [threshold + margin, 1.0]
            red_min = thr + m
            red_range = 1.0 - red_min
            
            # Validate split proportions sum to 1.0
            assert abs(sum(red_split) - 1.0) < 0.001, "red_split must sum to 1.0"
            
            low_fraction, medium_fraction, high_fraction = red_split
            
            # Calculate boundaries
            low_boundary = red_range * low_fraction
            medium_boundary = red_range * (low_fraction + medium_fraction)
            
            # Map: A->Low, B->Medium, C->High (higher score = more confident it's fake)
            relative_score = confidence_score - red_min
            if relative_score < low_boundary:
                return "Low"
            elif relative_score < medium_boundary:
                return "Medium"
            else:
                return "High"
        
        return "Unknown"

    def _convert_audio_to_mp3(self, audio_data: bytes, source_format: str) -> bytes:
        """
        Convert audio data to MP3 format.

        Args:
            audio_data: Raw audio bytes (WAV or OGG)
            source_format: Source format ('wav' or 'ogg')

        Returns:
            MP3 encoded audio bytes
        """
        try:
            # Load audio from bytes
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_data),
                format=source_format.lower()
            )

            # Convert to MP3
            mp3_buffer = io.BytesIO()
            audio_segment.export(mp3_buffer, format="mp3", bitrate="128k")
            mp3_data = mp3_buffer.getvalue()

            print(f"[Audio Conversion] {source_format.upper()} -> MP3: {len(audio_data)} -> {len(mp3_data)} bytes")
            return mp3_data

        except Exception as e:
            print(f"[Audio Conversion] Failed to convert {source_format} to MP3: {e}")
            return audio_data  # Return original data as fallback

    def _sanitize_participant_id(self, participant_id_raw: str) -> str:
        """
        Sanitize participant_id to handle JSON corruption in gRPC messages.
        Now includes similarity matching to handle OCR errors in participant names.

        The participant_id field sometimes contains raw JSON fragments instead of clean IDs.
        This method extracts the actual participant ID and sanitizes it for banner generation.
        
        Nonsensical IDs (with more than one digit/special character except parentheses) are
        aggregated under the name 'UNKNOWN' to share the same memory window.

        Args:
            participant_id_raw: Raw participant_id from gRPC message

        Returns:
            Clean participant ID safe for use, or 'UNKNOWN' if the ID is nonsensical
        """

        if DEBUG_MODE:
            print(f"[DEBUG] Sanitizing participant ID: '{participant_id_raw}'")

        # Check for nonsensical IDs and aggregate them under 'UNKNOWN'
        if self._is_nonsensical_id(participant_id_raw):
            print(f"⚠️ [AGGREGATING] NONSENSICAL PARTICIPANT ID: '{participant_id_raw}' - "
                  f"Aggregating under 'UNKNOWN' participant")
            return "UNKNOWN"

        try:
            # If it looks like a clean participant ID already, use it
            if len(participant_id_raw) < 50 and not any(c in participant_id_raw for c in ['"', '\n', ',', '{', '}']):
                # Sanitize for safety
                import re
                invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
                sanitized = re.sub(invalid_chars, '_', participant_id_raw).strip(' .')
                
                # Try to find a matching participant using similarity matching
                matched_name = self.name_matcher.find_matching_participant(sanitized)
                if matched_name:
                    # Use the matched canonical name
                    if DEBUG_MODE:
                        print(f"[DEBUG] ID matched to existing participant: '{sanitized}' → '{matched_name}'")
                    return matched_name
                else:
                    # Register as new participant
                    final_name = sanitized[:50] if len(sanitized) > 50 else (sanitized or "participant_unknown")
                    self.name_matcher.register_participant(final_name)
                    if DEBUG_MODE:
                        print(f"[DEBUG] ID sanitized and registered: '{participant_id_raw}' → '{final_name}'")
                    return final_name

            # Handle JSON corruption - extract participant ID from corrupted data
            import re

            # Try to extract participant ID from corrupted data
            # Look for pattern: participant_[digits] before any JSON punctuation
            match = re.match(r'(participant_\d+)', participant_id_raw)
            if match:
                clean_id = match.group(1)
                invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
                sanitized = re.sub(invalid_chars, '_', clean_id).strip(' .')
                
                # Try matching
                matched_name = self.name_matcher.find_matching_participant(sanitized)
                if matched_name:
                    if DEBUG_MODE:
                        print(f"[DEBUG] ID matched to existing participant: '{sanitized}' → '{matched_name}'")
                    return matched_name
                else:
                    final_name = sanitized[:50] if len(sanitized) > 50 else (sanitized or "participant_unknown")
                    self.name_matcher.register_participant(final_name)
                    if DEBUG_MODE:
                        print(f"[DEBUG] ID sanitized and registered: '{participant_id_raw}' → '{final_name}'")
                    return final_name

            # Fallback: Take first part before any JSON punctuation and sanitize
            clean_id = participant_id_raw.split('"')[0].split(',')[0].strip()
            invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
            sanitized = re.sub(invalid_chars, '_', clean_id).strip(' .')
            
            # Try matching
            matched_name = self.name_matcher.find_matching_participant(sanitized)
            if matched_name:
                if DEBUG_MODE:
                    print(f"[DEBUG] ID matched to existing participant: '{sanitized}' → '{matched_name}'")
                return matched_name
            else:
                final_name = sanitized[:50] if len(sanitized) > 50 else (sanitized or "participant_unknown")
                self.name_matcher.register_participant(final_name)
                if DEBUG_MODE:
                    print(f"[DEBUG] ID sanitized and registered: '{participant_id_raw}' → '{final_name}'")
                return final_name

        except Exception:
            # Last resort: generate a safe ID
            import hashlib
            return f"participant_{hashlib.md5(participant_id_raw.encode()).hexdigest()[:8]}"

    def _is_nonsensical_id(self, participant_id_raw: str) -> bool:
        """
        Check if a participant ID is nonsensical based on the number of digits and special characters.
        
        A participant ID is considered nonsensical if it contains more than one digit or
        special character (excluding parentheses), which usually indicates corrupted or malformed data.
        
        Args:
            participant_id_raw: Raw participant ID to check
            
        Returns:
            True if the ID appears nonsensical, False otherwise
        """
        if not participant_id_raw or len(participant_id_raw.strip()) == 0:
            return True
            
        import re
        
        # Count digits
        digit_count = len(re.findall(r'\d', participant_id_raw))
        
        # Count special characters (excluding alphanumeric, underscore, hyphen, dot, space, and parentheses)
        special_chars = re.findall(r'[^a-zA-Z0-9_\-\.\s\(\)]', participant_id_raw)
        special_count = len(special_chars)
        
        # Nonsensical if more than one digit OR more than one special character
        is_nonsensical = digit_count > 1 or special_count > 1
        
        if DEBUG_MODE and is_nonsensical:
            print(f"[DEBUG] Nonsensical ID detected: '{participant_id_raw}' - "
                  f"Digits: {digit_count}, SpecialChars: {special_count} (excluding parentheses)")
        
        return is_nonsensical

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

                    await self._process_uplink_message(uplink_msg, stream_id)

                    # --- INFERENCE SECTION ---

                    # 1. Send inference-driven per-participant banners for VIDEO
                    if uplink_msg.participants:
                        inference_banners = await self._generate_inference_banners(uplink_msg)
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

        if DEBUG_MODE:
            print(f"[DEBUG] Received message from {uplink_msg.client_id} (seq: {uplink_msg.sequence_number})")
            print(
                f"[DEBUG] Message has: participants={bool(uplink_msg.participants)}, audio={uplink_msg.HasField('audio')}")

            # Detailed participant inspection
            if uplink_msg.participants:
                for i, p in enumerate(uplink_msg.participants):
                    print(f"[DEBUG] Participant #{i + 1}:")
                    print(f"[DEBUG]   - ID: '{p.participant_id}'")
                    print(f"[DEBUG]   - Has {len(p.crops)} crops")

                    # Check for problematic IDs
                    if ' ' in p.participant_id:
                        print(f"[DEBUG] ⚠️ WARNING: Participant ID contains spaces: '{p.participant_id}'")
                    if not p.participant_id:
                        print(f"[DEBUG] ⚠️ WARNING: Empty participant ID")

                    # Inspect crops
                    for j, crop in enumerate(p.crops):
                        img_bytes = getattr(crop, "image_data", b"")
                        print(f"[DEBUG]   - Crop #{j + 1}: {len(img_bytes)} bytes")
                        if not img_bytes:
                            print(f"[DEBUG] ⚠️ WARNING: Empty crop data")

            # Audio inspection
            if uplink_msg.HasField('audio'):
                audio = uplink_msg.audio
                ogg_size = len(audio.ogg_data) if hasattr(audio, 'ogg_data') and audio.ogg_data else 0
                wav_size = len(audio.wav_data) if hasattr(audio, 'wav_data') and audio.wav_data else 0
                print(f"[DEBUG] Audio chunk: '{audio.chunk_id}'")
                print(f"[DEBUG]   - OGG data: {ogg_size} bytes")
                print(f"[DEBUG]   - WAV data: {wav_size} bytes")
                print(f"[DEBUG]   - Duration: {audio.duration_ms}ms")

                if ogg_size == 0 and wav_size == 0:
                    print(f"[DEBUG] ⚠️ WARNING: No audio data found (both OGG and WAV empty)")

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

            # Submit frames to background workers for bucket upload
            if ENABLE_BUCKET_SAVE and self.video_io_workers:
                worker = self._get_next_video_worker()
                if worker:
                    for pf in participant_frames:
                        for crop in pf.crops:
                            img_bytes = getattr(crop, "image_data", b"")
                            if img_bytes:
                                frame_metadata = {
                                    "participant_id": getattr(pf, "participant_id", ""),
                                    "timestamp_ms": metadata.get("timestamp_ms", 0),
                                    "stream_id": metadata.get("stream_id", "")
                                }
                                worker.submit(img_bytes, frame_metadata)

            # Log processing
            participant_count = len(participant_frames)
            total_crops = sum(len(pf.crops) for pf in participant_frames)
            total_files = len(api_data.get('files', {}))

            print(f"[Backend] Processed video batch: {participant_count} participants, "
                  f"{total_crops} crops -> {chunk_path}")
            if ENABLE_BUCKET_SAVE and self.video_io_workers:
                print(f"[Backend] Submitted {total_crops} frames to video workers")

        except Exception as e:
            print(f"[Backend] Error processing participant frames: {e}")

    async def _process_audio_batch(self, audio_batch: pb2.AudioBatch,
                                   metadata: Dict[str, Any]) -> None:
        """Process audio batch with MP3 conversion."""
        try:
            # Convert audio to MP3 first
            mp3_data = None
            original_format = None

            # Try OGG first
            if hasattr(audio_batch, 'ogg_data') and audio_batch.ogg_data:
                mp3_data = self._convert_audio_to_mp3(audio_batch.ogg_data, 'ogg')
                original_format = 'ogg'
            # Try WAV if OGG not available
            elif hasattr(audio_batch, 'wav_data') and audio_batch.wav_data:
                mp3_data = self._convert_audio_to_mp3(audio_batch.wav_data, 'wav')
                original_format = 'wav'

            if not mp3_data:
                print("[Backend] No valid audio data found for conversion")
                return

            # Update metadata to reflect MP3 conversion
            metadata['converted_format'] = 'mp3'
            metadata['original_format'] = original_format
            metadata['mp3_size'] = len(mp3_data)

            # Write MP3 to storage
            audio_path = self.data_writer.write_audio_chunk_mp3(mp3_data, audio_batch, metadata)

            # Submit MP3 to background worker for bucket upload
            if ENABLE_BUCKET_SAVE and self.audio_io_workers:
                worker = self._get_next_audio_worker()
                if worker:
                    audio_metadata = {
                        "chunk_id": audio_batch.chunk_id,
                        "duration_ms": audio_batch.duration_ms,
                        "timestamp_ms": metadata.get("timestamp_ms", 0),
                        "stream_id": metadata.get("stream_id", ""),
                        "format": "mp3",
                        "original_format": original_format
                    }
                    worker.submit(mp3_data, audio_metadata)

            # Log processing
            frame_count = audio_batch.frame_count if hasattr(audio_batch, 'frame_count') else 0
            print(f"[Backend] Processed audio batch: {audio_batch.chunk_id}, "
                  f"{frame_count} frames, {len(mp3_data)} bytes MP3 (from {original_format}), "
                  f"{audio_batch.duration_ms}ms -> {audio_path}")
            if ENABLE_BUCKET_SAVE and self.audio_io_workers:
                print(f"[Backend] Submitted {len(mp3_data)} byte MP3 chunk to worker")

        except Exception as e:
            print(f"[Backend] Error processing audio batch: {e}")

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
        Synchronous helper to convert audio to MP3 and call the ASV API using multipart/form-data.
        """
        try:
            mp3_data = None
            original_format = None

            # Convert to MP3 first
            if hasattr(audio_batch, 'ogg_data') and audio_batch.ogg_data:
                mp3_data = self._convert_audio_to_mp3(audio_batch.ogg_data, 'ogg')
                original_format = 'ogg'
            elif hasattr(audio_batch, 'wav_data') and audio_batch.wav_data:
                mp3_data = self._convert_audio_to_mp3(audio_batch.wav_data, 'wav')
                original_format = 'wav'

            if not mp3_data:
                print(f"[ASV API] No valid audio data found for conversion")
                return None

            # Prepare the multipart/form-data payload
            files = {
                'audio': ('audio.mp3', mp3_data, 'audio/mpeg')
            }
            form_data = {
                'window_step': '500',
                'use_vad': 'true',
                'vol_norm': 'false',
                'threshold': '0.55'
            }

            # Make the HTTP POST request with multipart/form-data
            print(
                f"[ASV API] Sending {len(mp3_data)} bytes of MP3 audio (from {original_format}) for analysis to {self.asv_api_url}...")
            response = requests.post(self.asv_api_url, data=form_data, files=files, timeout=self.asv_api_timeout)
            response.raise_for_status()

            result = response.json()
            print(f"[ASV API] Received response: {result}")
            return result

        except requests.exceptions.RequestException as e:
            print(f"[ASV API] Error calling API: {e}")
            return None
        except Exception as e:
            print(f"[ASV API] Error processing audio for API call: {e}")
            return None

    async def _generate_audio_inference_banner(self, uplink_msg: pb2.Uplink) -> pb2.Downlink | None:
        """
        Processes an audio batch, calls the ASV API, and generates a global banner using sliding window.
        """
        loop = asyncio.get_running_loop()

        audio_batch = uplink_msg.audio
        request_seq = uplink_msg.sequence_number
        
        # Check for restart signal in session_id
        if hasattr(audio_batch, 'session_id') and audio_batch.session_id == "[RESTART]":
            print(f"[Backend] *** RESTART SIGNAL DETECTED IN AUDIO SESSION_ID ***")
            self.audio_window_manager.reset()
            return None  # Don't process this as a normal audio batch

        # Run the blocking I/O (HTTP request) in a separate thread
        api_result = await loop.run_in_executor(None, self._call_asv_api, audio_batch)

        if not api_result or 'prediction' not in api_result:
            return None

        # Process the result through the audio sliding window manager
        verdict_level = self.audio_window_manager.process_audio_result(api_result)
        
        # Only generate a banner if the verdict changed
        if verdict_level is None:
            return None

        now_ms = int(time.time() * 1000)
        ttl_ms = self.ttl_map.get(verdict_level, 3000)

        # Build a GLOBAL ScreenBanner
        banner = pb2.ScreenBanner()
        banner.level = verdict_level
        banner.ttl_ms = ttl_ms
        banner.placement = "TopCenter"
        banner.action_id = f"act-audio-{uuid.uuid4().hex[:8]}"
        banner.scope = "global"
        banner.scope_enum = pb2.SCOPE_GLOBAL
        banner.banner_type = "audio_ok" if verdict_level == pb2.GREEN else "audio_alert"
        banner.expiry_timestamp_ms = now_ms + ttl_ms

        # Wrap into Downlink message
        down = pb2.Downlink()
        down.timestamp_ms = now_ms
        down.server_id = self.server_id
        down.sequence_number = request_seq  # Echo the request sequence number
        down.received = True
        down.screen_banner.CopyFrom(banner)

        print(f"!******** SENDING AUDIO BANNER RESPONSE ********!")
        print(f"[Backend] Generated GLOBAL audio banner (sliding window) -> {pb2.BannerLevel.Name(verdict_level)}")
        print(f"!***********************************************!")

        return down

    async def _generate_inference_banners(self, uplink_msg: pb2.Uplink) -> list:
        """
        Generate per-participant banners using the stateful ParticipantManager
        to ensure stable and non-flickering verdicts.
        """
        responses = []
        now_ms = int(time.time() * 1000)
        request_seq = uplink_msg.sequence_number

        for pf in uplink_msg.participants:
            # Sanitize participant id (now includes similarity matching)
            pid_raw = getattr(pf, "participant_id", "") or ""
            participant_id = self._sanitize_participant_id(pid_raw)

            # Note: nonsensical participant IDs are now aggregated under 'UNKNOWN'

            #  Handle the special [RESTART] signal
            if participant_id == "[RESTART]":
                print("[Backend] Received [RESTART] signal from client. Resetting all participant states.")
                self.participant_manager.reset_all()
                self.name_matcher.reset()  # Also reset name matcher
                continue
            
            # Skip UNKNOWN participants - don't send banners for them
            if participant_id == "UNKNOWN":
                if DEBUG_MODE:
                    print(f"[Backend] Skipping UNKNOWN participant (raw: {pid_raw})")
                continue

            # Collect image bytes for API call
            image_bytes_list = []
            for crop in pf.crops:
                img_bytes = getattr(crop, "image_data", b"")
                if not img_bytes:
                    continue

                # Submit to video worker if bucket saving is enabled
                if ENABLE_BUCKET_SAVE and self.video_io_workers:
                    worker = self._get_next_video_worker()
                    if worker:
                        frame_metadata = {
                            "participant_id": participant_id,
                            "timestamp_ms": now_ms,
                        }
                        worker.submit(img_bytes, frame_metadata)

                image_bytes_list.append(img_bytes)

            if not image_bytes_list:
                continue

            # Run inference via API to get probabilities
            individual_probs = await self.video_api.infer_probs_from_bytes(image_bytes_list)
            
            # Skip if no faces were detected (empty probs)
            if not individual_probs:
                if DEBUG_MODE:
                    print(f"[Backend] No faces detected for participant {participant_id}, skipping")
                continue

            # Handle the new return type from the manager
            manager_result = self.participant_manager.process_and_decide(
                participant_id, individual_probs
            )

            # Only create a banner if the manager returns a result
            if manager_result is not None:
                new_verdict_level, confidence_score = manager_result
                ttl_ms = self.ttl_map.get(new_verdict_level, 2500)

                # --- CONFIDENCE LABEL CALCULATION ---
                confidence_label = self._calculate_confidence_label(new_verdict_level, confidence_score)
                print(f"[Backend] Confidence calculation for {participant_id}: "
                      f"verdict={pb2.BannerLevel.Name(new_verdict_level)}, "
                      f"score={confidence_score:.3f}, "
                      f"label={confidence_label}")

                # --- CONFIDENCE ENCODING LOGIC (BINARY) ---
                # 1. Calculate the true expiry timestamp.
                true_expiry_ms = now_ms + ttl_ms

                # 2. Map confidence label to binary code in last 3 digits
                #    Low -> 100, Medium -> 101, High -> 110, Uncertain -> 111
                confidence_code_map = {
                    "Low": 100,
                    "Medium": 101,
                    "High": 110,
                    "Uncertain": 111
                }
                confidence_code = confidence_code_map.get(confidence_label, 111)

                # 3. Create the encoded timestamp: zero out the last 3 digits of the
                #    true expiry and add the binary confidence code.
                #    This "hides" the confidence in a way that minimally affects the
                #    absolute expiry time for old clients (max 7ms difference).
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
                      f"Score={confidence_score:.3f}, "
                      f"ConfLabel={confidence_label}, "
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
                
                # Note: nonsensical participant IDs are now aggregated under 'UNKNOWN'
                raw_participant_ids.append(raw_id)
                sanitized_participant_ids.append(sanitized_id)
                print(f"[Backend] Frame {i + 1}: raw_id='{raw_id}' -> sanitized_id='{sanitized_id}'")
                if sanitized_id == "UNKNOWN":
                    print(f"[Backend] Frame {i + 1}: Nonsensical ID '{raw_id}' aggregated as 'UNKNOWN'")

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

        # Add I/O worker stats
        video_worker_count = sum(worker.count for worker in self.video_io_workers)
        audio_worker_count = sum(worker.count for worker in self.audio_io_workers)

        stats.update({
            "uptime_seconds": uptime,
            "data_writer_stats": self.data_writer.get_statistics(),
            "banner_simulator_stats": self.banner_simulator.get_banner_stats(),
            "name_matcher_stats": self.name_matcher.get_statistics(),
            "active_streams": len(self.active_streams),
            "video_worker_processed": video_worker_count,
            "audio_worker_processed": audio_worker_count,
            "bucket_save_enabled": ENABLE_BUCKET_SAVE,
            "video_bucket": GCP_VIDEO_BUCKET,
            "audio_bucket": GCP_AUDIO_BUCKET
        })

        if uptime > 0:
            stats["messages_per_second"] = stats["uplink_messages"] / uptime

        return stats

    def cleanup(self):
        """Clean up resources."""
        print("[Backend] Cleaning up I/O workers...")
        for worker in self.video_io_workers + self.audio_io_workers:
            worker.stop()
        
        # Reset audio window manager
        self.audio_window_manager.reset()
        print("[Backend] Audio window manager reset")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='WMA Backend gRPC Server')

    parser.add_argument('--video-bucket', type=str, default=None,
                        help='GCP bucket name for video frames')
    parser.add_argument('--audio-bucket', type=str, default=None,
                        help='GCP bucket name for audio chunks')
    parser.add_argument('--enable-bucket-save', action='store_true',
                        help='Enable saving to GCP buckets (default: disabled)')
    parser.add_argument('--io-workers', type=int, default=2,
                        help='Number of I/O workers per media type (default: 2)')
    parser.add_argument('--port', type=int, default=50051,
                        help='gRPC server port (default: 50051)')

    parser.add_argument('--debug', action='store_true',
                        help='Enable verbose debug logging')

    return parser.parse_args()


async def serve():
    """Start the gRPC server."""
    # Parse command line arguments
    args = parse_args()

    # Set global variables
    global GCP_VIDEO_BUCKET, GCP_AUDIO_BUCKET, ENABLE_BUCKET_SAVE, IO_WORKER_COUNT, DEBUG_MODE
    GCP_VIDEO_BUCKET = args.video_bucket
    GCP_AUDIO_BUCKET = args.audio_bucket
    ENABLE_BUCKET_SAVE = args.enable_bucket_save
    IO_WORKER_COUNT = args.io_workers
    DEBUG_MODE = args.debug

    print(f"[Backend] Configuration:")
    print(f"  - Debug mode: {'Enabled' if DEBUG_MODE else 'Disabled'}")
    print(f"  - Video bucket: {GCP_VIDEO_BUCKET or 'None'}")
    print(f"  - Audio bucket: {GCP_AUDIO_BUCKET or 'None'}")
    print(f"  - Bucket saving: {'Enabled' if ENABLE_BUCKET_SAVE else 'Disabled'}")
    print(f"  - I/O workers per media type: {IO_WORKER_COUNT}")
    print(f"  - Port: {args.port}")

    # Validate bucket configuration
    if ENABLE_BUCKET_SAVE and (not GCP_VIDEO_BUCKET or not GCP_AUDIO_BUCKET):
        print("[Backend] Error: --enable-bucket-save requires both --video-bucket and --audio-bucket")
        return

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
    listen_addr = f'[::]:{args.port}'
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
                      f"{stats['banners_sent']} banners sent, "
                      f"I/O processed: {stats['video_worker_processed']} video, {stats['audio_worker_processed']} audio")

    except KeyboardInterrupt:
        print("[Backend] Shutting down server...")
    finally:
        print("[Backend] Closing active connections...")
        service_impl.cleanup()
        await server.stop(5)


if __name__ == '__main__':
    print("=== WMA Backend gRPC Server ===")
    print("Stub implementation for testing Service 5 communication")
    print("Press Ctrl+C to stop\n")

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        print("\n[Backend] Server stopped")
