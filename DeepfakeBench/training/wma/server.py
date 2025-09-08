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
from typing import AsyncIterator, Dict, Any
import grpc
from grpc import aio

# ==================== START: EASY FIX ====================
# Add the script's own directory to the Python path.
# This makes the 'except' block's imports (like 'from storage...') work reliably.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ===================== END: EASY FIX =====================


print("Hello 1")


try:
    # --- Preferred relative imports for running as a package (e.g., with systemd) ---
    from . import wma_streaming_pb2 as pb2
    from . import wma_streaming_pb2_grpc as pb2_grpc
    from .storage.data_writer import BackendDataWriter
    from .utils.banner_simulator import BannerSimulator
except ImportError:
    # --- Fallback absolute imports for running as a standalone script (e.g., for local testing) ---
    import wma_streaming_pb2 as pb2
    import wma_streaming_pb2_grpc as pb2_grpc
    from storage.data_writer import BackendDataWriter
    from utils.banner_simulator import BannerSimulator


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


class StreamingServiceImpl(pb2_grpc.StreamingServiceServicer):
    """Implementation of the WMA StreamingService."""

    def __init__(self):
        """Initialize the streaming service."""
        # Get data directory from environment variable, with a fallback for local dev
        default_path = os.path.join(BASE_PATH, "data")
        data_directory = os.getenv("DATA_DIRECTORY", default_path)

        print(f"[Backend] Using data storage directory: {data_directory}")

        # Ensure the directory exists before using it
        if not os.path.exists(data_directory):
            print(f"[Backend] Creating data directory: {data_directory}")
            os.makedirs(data_directory)
        self.data_writer = BackendDataWriter(data_directory)
        self.banner_simulator = BannerSimulator(banner_probability=0.15)  # 15% chance

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

                    # Process the message
                    response = await self._process_uplink_message(uplink_msg, stream_id)

                    # Send response if generated
                    if response:
                        self.stats["downlink_messages"] += 1
                        yield response

                    # Check for additional banner responses (simulate async banner triggers)
                    banner_response = await self._check_for_banner_trigger()
                    if banner_response:
                        self.stats["banners_sent"] += 1
                        yield banner_response

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

    async def _create_response(self, uplink_msg: pb2.Uplink) -> pb2.Downlink:
        """Create response message with possible banner."""
        response = pb2.Downlink()
        response.timestamp_ms = int(time.time() * 1000)
        response.server_id = self.server_id
        response.sequence_number = self._next_sequence()
        response.received = True

        # Check if we should generate a banner
        banner = self.banner_simulator.generate_banner()
        if banner:
            response.screen_banner.CopyFrom(banner)
            print(f"[Backend] Generated banner: {pb2.BannerLevel.Name(banner.level)} "
                  f"TTL={banner.ttl_ms}ms ID={banner.action_id}")

        return response

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
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
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
