#!/usr/bin/env python3

"""Simple test to verify basic gRPC functionality."""

import asyncio
import grpc
from grpc import aio
import wma_streaming_pb2 as pb2
import wma_streaming_pb2_grpc as pb2_grpc


async def test_server():
    """Test basic server functionality."""
    server = aio.server()
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    
    print(f"Starting test server on {listen_addr}")
    await server.start()
    print("Server started successfully!")
    
    # Run for a few seconds then stop
    await asyncio.sleep(3)
    await server.stop(1)
    print("Server stopped")


if __name__ == '__main__':
    print("=== gRPC Test Server ===")
    try:
        asyncio.run(test_server())
        print("Test completed successfully")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()