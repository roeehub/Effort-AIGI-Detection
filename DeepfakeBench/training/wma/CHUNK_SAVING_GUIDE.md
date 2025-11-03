# Chunk Saving Configuration Guide

## Overview
The WMA backend now supports optional chunk saving to disk for debugging purposes. This feature is controlled by a command-line flag and uses non-blocking I/O to avoid performance degradation.

## How to Enable

### Command Line Flag
Add `--enable-chunk-save` when starting the server:

```bash
python server.py --enable-chunk-save
```

Or when using start_backend.py, you can pass arguments through:

```bash
python start_backend.py --enable-chunk-save
```

### Full Example with Other Flags
```bash
python server.py \
  --enable-chunk-save \
  --debug \
  --port 50051
```

## How It Works

### Architecture
When `--enable-chunk-save` is enabled:

1. **Video chunks**: Saved to `data/video/` directory as they arrive
2. **Audio chunks**: Converted to MP3 and saved to `data/audio/` directory

### Non-Blocking Design
All disk I/O operations are offloaded to background thread pools using `asyncio.to_thread()`:

- Video saving: Immediate, queued to thread pool
- Audio saving: MP3 conversion + writing, both in thread pool
- Fast consumer loop: Never blocks on disk I/O

### Performance Impact
- **When disabled** (default): Zero overhead, no disk writes
- **When enabled**: Minimal impact (~1-2ms per chunk, non-blocking)
- **All chunks saved**: Every incoming video and audio chunk (not just samples)

## Storage Structure

### Video Chunks
```
data/video/
├── video_chunk_20250103_143521_a4b3c2d1/
│   ├── chunk_manifest.json
│   ├── participant_roee/
│   │   ├── crop_0.jpg
│   │   ├── crop_1.jpg
│   │   └── participant_manifest.json
│   └── participant_alice/
│       ├── crop_0.jpg
│       └── participant_manifest.json
└── video_chunk_20250103_143522_b5c3d2e1/
    └── ...
```

### Audio Chunks
```
data/audio/
├── audio_20250103_143521_chunk_id_001.mp3
├── audio_20250103_143521_chunk_id_001.mp3.meta.json
├── audio_20250103_143522_chunk_id_002.mp3
└── audio_20250103_143522_chunk_id_002.mp3.meta.json
```

## Configuration Summary

| Flag | Default | Purpose |
|------|---------|---------|
| `--enable-chunk-save` | `False` | Enable saving all chunks to disk |
| `--debug` | `False` | Enable verbose debug logging |
| `--port` | `50051` | gRPC server port |

## Use Cases

### Development & Debugging
Enable chunk saving to inspect:
- Raw frame data from Service 5
- Audio chunk quality and format
- Participant ID handling
- Data flow and timing

### Production
Keep disabled for:
- Optimal performance
- Reduced disk usage
- Lower I/O overhead

## Checking Status

The server logs will show the configuration on startup:

```
[Backend] Configuration:
  - Debug mode: Enabled
  - Chunk saving: Enabled    <-- Look for this line
  - Video bucket: None
  - Audio bucket: None
  - Bucket saving: Disabled
  - I/O workers per media type: 2
  - Port: 50051
```

## Implementation Details

### Code Changes
The changes restore chunk saving that was lost during the queue-based architecture refactor:

**Old flow (removed):**
```
StreamData() → _process_uplink_message() → _process_participant_frames() → write_video_chunk()
```

**New flow (current):**
```
StreamData() → _fast_consumer_loop() → [if ENABLE_CHUNK_SAVE] → asyncio.to_thread(write_video_chunk())
                                      → _route_frame_to_queue() → workers (inference only)
```

### Key Features
1. **Configurable**: Off by default, enable when needed
2. **Non-blocking**: All I/O in background thread pool
3. **Complete**: Saves ALL chunks, not samples
4. **Safe**: Error handling prevents one bad chunk from stopping the stream
5. **Flexible**: Easy to toggle on/off without code changes

## Tips

### Disk Space Management
When enabled, chunks can accumulate quickly. Monitor disk usage:

```bash
# Check disk usage
du -sh data/video data/audio

# Clean old chunks (be careful!)
find data/video -type d -mtime +1 -exec rm -rf {} +
find data/audio -name "*.mp3" -mtime +1 -delete
```

### Performance Monitoring
Watch for I/O bottlenecks in logs:
- No "Error saving chunk" messages = healthy
- Check message processing rate in periodic log messages
- Monitor for dropped frames in queue statistics

## Troubleshooting

**Problem**: Chunks not appearing in `data/` directory  
**Solution**: Check that `--enable-chunk-save` flag is passed

**Problem**: Permission denied errors  
**Solution**: Ensure write permissions on `data/` directory

**Problem**: Disk full errors  
**Solution**: Clean old chunks or increase disk space

**Problem**: Slow performance with saving enabled  
**Solution**: Check disk I/O speed; consider faster storage or disabling
