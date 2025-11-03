# How to Enable Chunk Saving

## Your Current Command

You run the server with:
```bash
PYTHONPATH="$(pwd)" python -m wma.start_backend
```

## To Enable Chunk Saving

Simply add `--enable-chunk-save` at the end:

```bash
PYTHONPATH="$(pwd)" python -m wma.start_backend --enable-chunk-save
```

## To Enable Debug Mode + Chunk Saving

```bash
PYTHONPATH="$(pwd)" python -m wma.start_backend --enable-chunk-save --debug
```

## What to Expect

When chunk saving is enabled, you'll see:

### 1. Startup confirmation:
```
[BackendStartup] ✓ Chunk saving is ENABLED
[BackendStartup] Chunks will be saved to data/video/ and data/audio/
[Backend] Configuration:
  - Chunk saving: Enabled
  - Video chunks → /home/roee/repos/.../wma/data/video/
  - Audio chunks → /home/roee/repos/.../wma/data/audio/
```

### 2. Per-chunk save logs:
```
[ChunkSave] ✓ Saved video chunk to: data/video/video_chunk_20251103_120345_a1b2c3d4/
[ChunkSave] ✓ Saved audio chunk to: data/audio/audio_1730635425_chunk_id_001.mp3
```

### 3. Check saved files:
```bash
ls -lh data/video/
ls -lh data/audio/
```

## Important Notes

- **ALL chunks are saved** when enabled (not samples)
- **Performance**: Minimal impact, I/O runs in background threads
- **Disk space**: Monitor usage, chunks accumulate quickly
- **Production**: Keep disabled (default) for optimal performance
- **Debugging**: Enable to inspect raw incoming data

## Troubleshooting

**No chunks appearing?**
- Check that you see "Chunk saving: Enabled" in the logs
- Look for `[ChunkSave] ✓` messages
- Verify directory exists: `ls -la data/`

**Want to disable?**
- Just remove the `--enable-chunk-save` flag
- Or stop and restart without the flag
