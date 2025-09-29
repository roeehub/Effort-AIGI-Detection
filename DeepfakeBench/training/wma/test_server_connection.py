"""
Mock gRPC Client for WMA Streaming Service

Extracts video frames and audio chunks from a video file and streams them
to the WMA backend server using the gRPC StreamingService.
"""

import asyncio
import grpc
import cv2
import numpy as np
import soundfile as sf
import librosa
import io
import time
import uuid
import argparse
from pathlib import Path
from typing import List, Iterator, Tuple

# Import generated protobuf classes
import wma_streaming_pb2 as pb2
import wma_streaming_pb2_grpc as pb2_grpc

# Global configuration
VIDEO_PATH = "low_res_video_test.mp4"  # Change this to your video file path
SERVER_IP = "34.116.214.60"
SERVER_PORT = 50051
CLIENT_ID = f"mock-client-{uuid.uuid4().hex[:8]}"

# Processing parameters
FRAME_RATE = 10  # Extract every Nth frame (10 = 1 frame per 10 original frames)
AUDIO_CHUNK_DURATION = 2.0  # seconds per audio chunk
CROP_SIZE = (224, 224)  # Size for face crops
MAX_PARTICIPANTS = 3  # Maximum number of simulated participants per frame


class VideoAudioExtractor:
    """Extract frames and audio chunks from video file."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self.audio_data = None
        self.sample_rate = None
        self._load_video()
        self._load_audio()

    def _load_video(self):
        """Load video for frame extraction."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps

        print(f"[Extractor] Video loaded: {self.total_frames} frames, {self.fps} FPS, {self.duration:.1f}s")

    def _load_audio(self):
        """Load audio data using librosa."""
        try:
            self.audio_data, self.sample_rate = librosa.load(self.video_path, sr=None, mono=False)
            if len(self.audio_data.shape) == 1:
                # Convert mono to stereo for consistency
                self.audio_data = np.stack([self.audio_data, self.audio_data])
            print(f"[Extractor] Audio loaded: {self.audio_data.shape}, SR={self.sample_rate}")
        except Exception as e:
            print(f"[Extractor] Failed to load audio: {e}")
            self.audio_data = None

    def extract_frames(self, frame_rate: int = FRAME_RATE) -> Iterator[Tuple[int, np.ndarray]]:
        """Extract frames at specified rate."""
        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_count % frame_rate == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield extracted_count, rgb_frame
                extracted_count += 1

            frame_count += 1

        print(f"[Extractor] Extracted {extracted_count} frames from {frame_count} total frames")

    def extract_audio_chunks(self, chunk_duration: float = AUDIO_CHUNK_DURATION) -> Iterator[Tuple[int, bytes]]:
        """Extract audio chunks and convert to OGG format."""
        if self.audio_data is None:
            print("[Extractor] No audio data available")
            return

        chunk_samples = int(chunk_duration * self.sample_rate)
        total_samples = self.audio_data.shape[1] if len(self.audio_data.shape) > 1 else len(self.audio_data)

        chunk_id = 0
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)

            if len(self.audio_data.shape) > 1:
                chunk = self.audio_data[:, start:end].T  # Convert to (samples, channels)
            else:
                chunk = self.audio_data[start:end]

            # Convert to OGG bytes
            ogg_bytes = self._audio_to_ogg(chunk)
            if ogg_bytes:
                yield chunk_id, ogg_bytes
                chunk_id += 1

        print(f"[Extractor] Extracted {chunk_id} audio chunks")

    def _audio_to_ogg(self, audio_chunk: np.ndarray) -> bytes:
        """Convert audio chunk to OGG format."""
        try:
            buffer = io.BytesIO()
            sf.write(buffer, audio_chunk, self.sample_rate, format='OGG', subtype='VORBIS')
            return buffer.getvalue()
        except Exception as e:
            print(f"[Extractor] Error converting audio to OGG: {e}")
            return b""

    def cleanup(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()


class MockParticipantGenerator:
    """Generate mock participant crops from video frames."""

    @staticmethod
    def create_participant_crops(frame: np.ndarray, max_participants: int = MAX_PARTICIPANTS) -> List[
        pb2.ParticipantFrame]:
        """Create mock participant frames with crops from the input frame."""
        participants = []
        height, width = frame.shape[:2]

        # Create different crop regions to simulate multiple participants
        crop_regions = [
            (0, 0, width // 2, height // 2),  # Top-left
            (width // 2, 0, width, height // 2),  # Top-right
            (width // 4, height // 4, 3 * width // 4, 3 * height // 4),  # Center
        ]

        for i in range(min(max_participants, len(crop_regions))):
            participant_id = f"participant_{i + 100}"
            x1, y1, x2, y2 = crop_regions[i]

            # Extract crop region
            crop_region = frame[y1:y2, x1:x2]

            # Resize to standard crop size
            crop_resized = cv2.resize(crop_region, CROP_SIZE, interpolation=cv2.INTER_AREA)

            # Convert to JPEG bytes
            success, jpeg_bytes = cv2.imencode('.jpg', cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR),
                                               [cv2.IMWRITE_JPEG_QUALITY, 85])

            if success:
                # Create crop message
                crop = pb2.Crop()
                crop.image_data = jpeg_bytes.tobytes()
                crop.x = x1
                crop.y = y1
                crop.width = x2 - x1
                crop.height = y2 - y1
                crop.confidence_score = 0.8 + (i * 0.05)  # Mock confidence

                # Create participant frame
                participant_frame = pb2.ParticipantFrame()
                participant_frame.participant_id = participant_id
                participant_frame.crops.append(crop)

                participants.append(participant_frame)

        return participants


class MockWMAClient:
    """Mock client for WMA streaming service."""

    def __init__(self, server_address: str, client_id: str):
        self.server_address = server_address
        self.client_id = client_id
        self.channel = None
        self.stub = None
        self.sequence_number = 0

    async def connect(self):
        """Connect to the gRPC server."""
        self.channel = grpc.aio.insecure_channel(
            self.server_address,
            options=[
                ('grpc.max_receive_message_length', 20 * 1024 * 1024),
                ('grpc.max_send_message_length', 20 * 1024 * 1024),
            ]
        )
        self.stub = pb2_grpc.StreamingServiceStub(self.channel)

        # Test connection with ping
        await self.ping()
        print(f"[Client] Connected to {self.server_address}")

    async def ping(self):
        """Send ping request to server."""
        ping_request = pb2.PingRequest()
        ping_request.timestamp_ms = int(time.time() * 1000)
        ping_request.client_id = self.client_id

        try:
            response = await self.stub.Ping(ping_request)
            print(f"[Client] Ping response: {response.status}, server_time: {response.server_time_ms}")
        except grpc.RpcError as e:
            print(f"[Client] Ping failed: {e}")
            raise

    async def stream_video_audio(self, video_path: str):
        """Stream video frames and audio chunks to server."""
        extractor = VideoAudioExtractor(video_path)

        try:
            # Start bidirectional streaming with timeout
            request_generator = self._create_request_generator(extractor)
            stream = self.stub.StreamData(request_generator, timeout=300)  # 5 minute timeout
            print("[Client] Started streaming data to server")

            # Process responses from server
            response_count = 0
            try:
                async for response in stream:
                    await self._handle_server_response(response)
                    response_count += 1
                    if response_count % 10 == 0:  # Log every 10 responses
                        print(f"[Client] Processed {response_count} responses")

                print(f"[Client] Stream completed. Total responses: {response_count}")

            except asyncio.TimeoutError:
                print("[Client] Stream timed out")
            except asyncio.CancelledError:
                print("[Client] Stream was cancelled")

        except grpc.RpcError as e:
            print(f"[Client] gRPC error: {e.code()}: {e.details()}")
        except Exception as e:
            print(f"[Client] Streaming error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            extractor.cleanup()

    async def _create_request_generator(self, extractor: VideoAudioExtractor):
        """Generate uplink requests with video frames and audio chunks (async)."""
        # Send initial connection message
        initial_request = pb2.Uplink()
        initial_request.timestamp_ms = int(time.time() * 1000)
        initial_request.client_id = self.client_id
        initial_request.sequence_number = self._next_sequence()
        yield initial_request

        # Small delay after initial message
        await asyncio.sleep(0.1)

        # Create iterators (don't convert to lists)
        frame_iter = extractor.extract_frames()
        audio_iter = extractor.extract_audio_chunks()

        frame_exhausted = False
        audio_exhausted = False
        current_frame = None
        current_audio = None

        # Get first items
        try:
            current_frame = next(frame_iter)
        except StopIteration:
            frame_exhausted = True

        try:
            current_audio = next(audio_iter)
        except StopIteration:
            audio_exhausted = True

        # Send data alternating between video and audio
        while not frame_exhausted or not audio_exhausted:
            # Send video frame if available
            if not frame_exhausted and current_frame is not None:
                frame_data_idx, frame_data = current_frame
                request = pb2.Uplink()
                request.timestamp_ms = int(time.time() * 1000)
                request.client_id = self.client_id
                request.sequence_number = self._next_sequence()

                participants = MockParticipantGenerator.create_participant_crops(frame_data)
                request.participants.extend(participants)

                print(f"[Client] Sending frame {frame_data_idx} with {len(participants)} participants")
                yield request

                # Get next frame
                try:
                    current_frame = next(frame_iter)
                except StopIteration:
                    frame_exhausted = True
                    current_frame = None

                await asyncio.sleep(0.01)  # Very small delay

            # Send audio chunk if available
            if not audio_exhausted and current_audio is not None:
                chunk_id, ogg_data = current_audio
                audio_request = pb2.Uplink()
                audio_request.timestamp_ms = int(time.time() * 1000)
                audio_request.client_id = self.client_id
                audio_request.sequence_number = self._next_sequence()

                audio_batch = pb2.AudioBatch()
                audio_batch.chunk_id = f"chunk_{chunk_id}"
                audio_batch.ogg_data = ogg_data
                audio_batch.duration_ms = int(AUDIO_CHUNK_DURATION * 1000)
                audio_batch.frame_count = len(ogg_data)

                audio_request.audio.CopyFrom(audio_batch)

                print(f"[Client] Sending audio chunk {chunk_id} ({len(ogg_data)} bytes)")
                yield audio_request

                # Get next audio chunk
                try:
                    current_audio = next(audio_iter)
                except StopIteration:
                    audio_exhausted = True
                    current_audio = None

                await asyncio.sleep(0.01)  # Very small delay

        print("[Client] Finished sending all data")

    async def _handle_server_response(self, response: pb2.Downlink):
        """Handle response from server."""
        if response.HasField('screen_banner'):
            banner = response.screen_banner
            print(f"[Client] Received banner: level={pb2.BannerLevel.Name(banner.level)}, "
                  f"participant={banner.participant_id}, type={banner.banner_type}, "
                  f"ttl={banner.ttl_ms}ms")
        elif response.error_message:
            print(f"[Client] Server error: {response.error_message}")
        else:
            print(f"[Client] Server ACK: seq={response.sequence_number}")

    def _next_sequence(self) -> int:
        """Get next sequence number."""
        self.sequence_number += 1
        return self.sequence_number

    async def disconnect(self):
        """Disconnect from server."""
        if self.channel:
            await self.channel.close()


async def main():
    """Main function to run the mock client."""
    parser = argparse.ArgumentParser(description='Mock WMA gRPC Client')
    parser.add_argument('--video', type=str, default=VIDEO_PATH,
                        help=f'Path to video file (default: {VIDEO_PATH})')
    parser.add_argument('--server', type=str, default=f"{SERVER_IP}:{SERVER_PORT}",
                        help=f'Server address (default: {SERVER_IP}:{SERVER_PORT})')
    parser.add_argument('--client-id', type=str, default=CLIENT_ID,
                        help=f'Client ID (default: {CLIENT_ID})')

    args = parser.parse_args()

    # Validate video file exists
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return

    print(f"=== Mock WMA gRPC Client ===")
    print(f"Video file: {args.video}")
    print(f"Server: {args.server}")
    print(f"Client ID: {args.client_id}")
    print()

    # Create and run client
    client = MockWMAClient(args.server, args.client_id)

    try:
        await client.connect()
        await client.stream_video_audio(args.video)

    except KeyboardInterrupt:
        print("\n[Client] Interrupted by user")
    except Exception as e:
        print(f"[Client] Error: {e}")
    finally:
        await client.disconnect()
        print("[Client] Disconnected")


if __name__ == '__main__':
    asyncio.run(main())
