# cloud_client_tester.py

import grpc
import time
import uuid
import threading
import os
from datetime import datetime

# Import your generated protobuf files
import wma_streaming_pb2 as pb2
import wma_streaming_pb2_grpc as pb2_grpc

# --- Video/Audio Processing Libraries ---
import cv2  # OpenCV for video
from pydub import AudioSegment  # pydub for audio
from google.protobuf.json_format import MessageToJson
# Also dump any raw bytes fields we can find, to separate .bin files
from google.protobuf.descriptor import FieldDescriptor

# --- Configuration ---
SERVER_IP = "34.116.214.60"
SERVER_PORT = 50051
VIDEO_FILE_PATH = "low_res_video_test.mp4"  # Path to your test video file


# --- Helper for Colored Output ---
class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_sent(msg):
    print(f"{bcolors.OKGREEN}[--> SENT] {msg}{bcolors.ENDC}")


def print_recv(msg):
    print(f"{bcolors.WARNING}[<-- RECV] {msg}{bcolors.ENDC}")


def print_error(msg):
    print(f"{bcolors.FAIL}[ERROR] {msg}{bcolors.ENDC}")


def get_current_ms():
    """Returns the current epoch timestamp in milliseconds."""
    return int(time.time() * 1000)


def listen_for_downlinks(response_iterator):
    """
    A dedicated function to run in a thread, listening for server responses.
    """
    try:
        for downlink_msg in response_iterator:
            # === BEGIN: super-simple "I really wrote a file" logging ===
            log_dir = os.environ.get("WMA_LOG_DIR", os.getcwd())
            os.makedirs(log_dir, exist_ok=True)

            # one-time init per run
            if not hasattr(listen_for_downlinks, "_run_init"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                listen_for_downlinks._run_dir = os.path.join(log_dir, f"downlinks_{ts}_msgs")
                os.makedirs(listen_for_downlinks._run_dir, exist_ok=True)
                listen_for_downlinks._counter = 0
                print(f"[LOG] Downlinks will be saved to: {listen_for_downlinks._run_dir}")
                listen_for_downlinks._run_init = True

            # pick a filename and write *something* no matter what
            idx = listen_for_downlinks._counter
            listen_for_downlinks._counter += 1
            seq = getattr(downlink_msg, "sequence_number", None)
            base = f"downlink_{seq if seq is not None else 'nseq'}_{idx}"
            msg_dir = os.path.join(listen_for_downlinks._run_dir, base)
            os.makedirs(msg_dir, exist_ok=True)

            # 1) Debug TXT (always)
            dbg_path = os.path.join(msg_dir, base + ".txt")
            with open(dbg_path, "a", encoding="utf-8") as df:
                df.write(f"{datetime.now().isoformat()}  {downlink_msg}\n")

            # 2) JSON (prefer protobuf->json; fallback to str)
            json_path = os.path.join(msg_dir, base + ".json")
            try:
                if MessageToJson is not None:
                    try:
                        json_str = MessageToJson(
                            downlink_msg,
                            including_default_value_fields=True,
                            preserving_proto_field_name=True,
                        )
                    except TypeError:
                        json_str = MessageToJson(downlink_msg)
                else:
                    json_str = str(downlink_msg)
            except Exception as e:
                json_str = f'{{"fallback_str": "{str(downlink_msg).replace(chr(10), " ")}", "error": "{e}"}}'

            with open(json_path, "w", encoding="utf-8") as jf:
                jf.write(json_str)

            print(f"[LOG] Saved downlink -> {json_path}")
            # === END: super-simple logging ===

            if downlink_msg.HasField("screen_banner"):
                banner = downlink_msg.screen_banner
                print_recv(f"Received ScreenBanner: level={banner.level}, ttl={banner.ttl_ms}ms")
            elif downlink_msg.received:
                print_recv(f"Received ACK for sequence={downlink_msg.sequence_number}")
            elif downlink_msg.error_message:
                print_error(f"Received error: {downlink_msg.error_message}")
            else:
                print_recv(f"Received Downlink: {downlink_msg}")

    except grpc.RpcError as e:
        print_error(f"Connection lost or error in downlink stream: {e.code()} - {e.details()}")


def generate_uplink_messages(video_path, meeting_id, session_id, client_id):
    """
    A generator function that reads video/audio, batches it, and yields Uplink messages.
    """
    if not os.path.exists(video_path):
        print_error(f"Video file not found at '{video_path}'")
        return

    # 1. Pre-process Audio into 4-second OGG/Opus chunks
    print("[INFO] Pre-processing audio into 4-second OGG/Opus chunks...")
    full_audio = AudioSegment.from_file(video_path)
    audio_chunks = []
    for i in range(0, len(full_audio), 4000):
        chunk = full_audio[i:i + 4000]
        # Pad with silence if the last chunk is too short
        if len(chunk) < 4000:
            silence = AudioSegment.silent(duration=(4000 - len(chunk)))
            chunk += silence

        # pydub needs bytes to work with for export
        from io import BytesIO
        ogg_buffer = BytesIO()
        chunk.export(ogg_buffer, format="ogg", codec="libopus", parameters=["-b:a", "64k"])
        audio_chunks.append(ogg_buffer.getvalue())
    print(f"[INFO] Generated {len(audio_chunks)} audio chunks.")

    # 2. Open Video for Frame-by-Frame Processing
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video opened: {width}x{height} @ {fps:.2f} FPS. Looping indefinitely.")

    # 3. Main Streaming Loop
    sequence_number = 0
    audio_chunk_index = 0
    crops_batch = []

    last_video_send_time = time.time()
    last_audio_send_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            # If video ends, loop back to the beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Encode frame to JPEG
        _, image_data = cv2.imencode('.jpg', frame)

        # Create a dummy participant crop
        crop = pb2.ParticipantCrop(
            participant_id="participant_001",
            image_data=image_data.tobytes(),
            bbox_x=10, bbox_y=10, bbox_width=100, bbox_height=100,
            confidence=0.95,
            timestamp_ms=get_current_ms(),
            sequence_number=sequence_number
        )
        crops_batch.append(crop)

        # Check if it's time to send a video batch (every 2 seconds)
        if time.time() - last_video_send_time >= 2.0:
            uplink_msg = pb2.Uplink(
                timestamp_ms=get_current_ms(),
                client_id=client_id,
                sequence_number=sequence_number
            )

            # Add video data
            participant_frame = pb2.ParticipantFrame(
                participant_id="participant_001",
                crops=crops_batch,
                start_ts_ms=get_current_ms() - 2000,
                end_ts_ms=get_current_ms(),
                meeting_id=meeting_id,
                session_id=session_id,
                chunk_id=str(uuid.uuid4()),
                frame_count=len(crops_batch)
            )
            uplink_msg.participants.append(participant_frame)

            # Check if it's also time to send an audio batch (every 4 seconds)
            if time.time() - last_audio_send_time >= 4.0:
                audio_data = audio_chunks[audio_chunk_index % len(audio_chunks)]
                audio_batch = pb2.AudioBatch(
                    ogg_data=audio_data,
                    start_ts_ms=get_current_ms() - 4000,
                    duration_ms=4000,
                    meeting_id=meeting_id,
                    session_id=session_id,
                    chunk_id=str(uuid.uuid4())
                )
                uplink_msg.audio.CopyFrom(audio_batch)
                last_audio_send_time = time.time()
                audio_chunk_index += 1

            # Yield the message to the gRPC stream
            message_type = "Video + Audio" if uplink_msg.HasField("audio") else "Video Only"
            print_sent(f"Uplink seq={sequence_number}, type={message_type}, crops={len(crops_batch)}")
            yield uplink_msg

            # Reset for next batch
            sequence_number += 1
            crops_batch = []
            last_video_send_time = time.time()

        # Sleep to simulate real-time frame capture
        time.sleep(1 / fps)


def run():
    """Main function to connect to the server and start streaming."""
    server_address = f"{SERVER_IP}:{SERVER_PORT}"
    print(f"[INFO] Attempting to connect to gRPC server at {server_address}...")

    # Define options to increase the max message size (e.g., to 20MB)
    options = [
        ('grpc.max_receive_message_length', 20 * 1024 * 1024),
        ('grpc.max_send_message_length', 20 * 1024 * 1024)
    ]

    # Use an insecure channel for testing (no TLS)
    with grpc.insecure_channel(server_address, options=options) as channel:
        # First, test with a Ping to ensure the server is responsive
        try:
            stub = pb2_grpc.StreamingServiceStub(channel)
            ping_req = pb2.PingRequest(timestamp_ms=get_current_ms(), client_id="ping-tester")
            ping_res = stub.Ping(ping_req)
            print(f"[INFO] Ping successful! Server status: {ping_res.status}, version: {ping_res.version}")
        except grpc.RpcError as e:
            print_error(f"Could not connect to server. Ping failed: {e.code()} - {e.details()}")
            print_error("Please ensure the server is running and the IP/port are correct.")
            return

        # Prepare unique IDs for this session
        meeting_id = f"meet_{uuid.uuid4()}"
        session_id = f"sess_{uuid.uuid4()}"
        client_id = "python-cloud-tester-v1"
        print(f"[INFO] Starting stream with MeetingID: {meeting_id}")

        # Create the generator that produces uplink messages
        uplink_generator = generate_uplink_messages(VIDEO_FILE_PATH, meeting_id, session_id, client_id)

        # Start the bidirectional stream
        response_iterator = stub.StreamData(uplink_generator)

        # Start a separate thread to listen for downlink messages
        downlink_thread = threading.Thread(
            target=listen_for_downlinks,
            args=(response_iterator,),
            daemon=True
        )
        downlink_thread.start()

        # Keep the main thread alive while the downlink thread is running
        downlink_thread.join()


if __name__ == "__main__":
    run()
