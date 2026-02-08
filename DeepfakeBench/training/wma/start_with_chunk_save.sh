#!/bin/bash
# Quick start script with chunk saving enabled

echo "Starting WMA Backend with chunk saving enabled..."
echo ""
echo "Chunks will be saved to:"
echo "  - Video: data/video/"
echo "  - Audio: data/audio/"
echo ""
echo "Look for log messages like:"
echo "  [ChunkSave] ✓ Saved video chunk to: data/video/video_chunk_..."
echo "  [ChunkSave] ✓ Saved audio chunk to: data/audio/audio_..."
echo ""

# Start the server with chunk saving enabled
python server.py --enable-chunk-save --debug "$@"
