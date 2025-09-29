import asyncio
import grpc
import uuid
import time
import wma_streaming_pb2 as pb2
import wma_streaming_pb2_grpc as pb2_grpc
import cv2
import numpy as np

SERVER_ADDRESS = "34.116.214.60:50051"
CLIENT_ID = f"test-client-{uuid.uuid4().hex[:8]}"

async def test_server_connection():
    print(f"[Test] Creating connection to {SERVER_ADDRESS}")
    channel = grpc.aio.insecure_channel(
        SERVER_ADDRESS,
        options=[
            ('grpc.max_receive_message_length', 20 * 1024 * 1024),
            ('grpc.max_send_message_length', 20 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.keepalive_permit_without_calls', 1)
        ]
    )

    stub = pb2_grpc.StreamingServiceStub(channel)

    # Test ping first
    try:
        print(f"[Test] Sending ping with client ID: {CLIENT_ID}")
        ping_request = pb2.PingRequest(
            timestamp_ms=int(time.time() * 1000),
            client_id=CLIENT_ID
        )
        ping_response = await stub.Ping(ping_request)
        print(f"[Test] Ping successful: {ping_response.status}, time: {ping_response.server_time_ms}")
    except Exception as e:
        print(f"[Test] Ping failed: {e}")
        await channel.close()
        return False

    # Create a simple test image
    def create_test_image():
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        # Add a red rectangle - red is a color often detected as "fake"
        cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)
        # Add text
        cv2.putText(img, "TEST", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Encode to JPEG
        _, jpeg_data = cv2.imencode('.jpg', img)
        return jpeg_data.tobytes()

    # Test stream connection with a video frame
    try:
        print(f"[Test] Testing stream connection with video frame")

        async def video_generator():
            # Initial connection message
            initial_msg = pb2.Uplink(
                timestamp_ms=int(time.time() * 1000),
                client_id=CLIENT_ID,
                sequence_number=1
            )
            print(f"[Test] Sending initial connection message")
            yield initial_msg
            await asyncio.sleep(1)

            # Create a participant with a test image
            participant_msg = pb2.Uplink(
                timestamp_ms=int(time.time() * 1000),
                client_id=CLIENT_ID,
                sequence_number=2
            )

            # Create participant frame
            participant = pb2.ParticipantFrame()
            participant.participant_id = "test-participant-1"

            # Add crop with test image
            crop = pb2.ParticipantCrop()
            crop.image_data = create_test_image()
            participant.crops.append(crop)

            # Add participant to message
            participant_msg.participants.append(participant)

            print(f"[Test] Sending participant frame")
            yield participant_msg
            await asyncio.sleep(2)

        stream = stub.StreamData(video_generator())
        print("[Test] Stream connected")

        msg_count = 0
        async for response in stream:
            msg_count += 1
            print(f"[Test] Received response #{msg_count}: {response}")

        print(f"[Test] Stream completed normally with {msg_count} responses")

    except asyncio.CancelledError:
        print("[Test] Stream was cancelled by the server")
    except grpc.RpcError as e:
        print(f"[Test] gRPC error: {e.code()}: {e.details()}")
    except Exception as e:
        print(f"[Test] Stream error: {type(e).__name__}: {e}")

    await channel.close()
    print("[Test] Connection closed")
    return True

if __name__ == "__main__":
    asyncio.run(test_server_connection())