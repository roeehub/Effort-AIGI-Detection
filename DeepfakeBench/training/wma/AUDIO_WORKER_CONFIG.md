# Audio Worker Configuration Guide

## Quick Reference

### Default Configuration
```python
'audio': {
    'worker_count': 1,        # Number of concurrent audio workers
    'queue_size': 2,          # Max audio chunks in queue (2 × 4s = 8s max staleness)
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AUDIO_WORKER_COUNT` | `1` | Number of audio inference workers |
| `AUDIO_API_URLS` | `http://34.125.106.206:8000/asv/predict` | ASV API endpoint(s) |
| `ASV_API_TIMEOUT` | `20` | Audio API timeout in seconds |

### Starting the Server

**Standard (1 audio worker):**
```bash
python server.py
```

**With 2 audio workers:**
```bash
AUDIO_WORKER_COUNT=2 python server.py
```

**With debug logging:**
```bash
python server.py --debug
```

## Configuration Scenarios

### Scenario 1: Low Audio Load (Default)
**Use Case:** 1-3 participants, occasional audio
```bash
AUDIO_WORKER_COUNT=1
```
- Single worker handles all audio
- Queue rarely fills
- Minimal resource usage

### Scenario 2: High Audio Load
**Use Case:** 5+ participants, continuous audio
```bash
AUDIO_WORKER_COUNT=2
```
- Two workers share the load
- Better throughput
- Redundancy if one worker has issues

### Scenario 3: Multiple API Endpoints (Future)
**Use Case:** Load balancing across multiple ASV servers
```bash
AUDIO_API_URLS="http://server1:8000/asv/predict,http://server2:8000/asv/predict"
AUDIO_WORKER_COUNT=2
```
- Each worker can use different endpoint
- Round-robin distribution
- Failover capability

## Monitoring

### Log Messages to Watch

**Worker Starting:**
```
[Backend] Starting 1 audio inference worker(s)...
[AudioWorker-1] Audio inference worker started
```

**Processing Audio:**
```
[AudioWorker-1] Processing audio chunk (age: 0.12s, size: 153600 bytes)
[ASV API Worker] Sending 122880 bytes of MP3 audio (from wav) to http://...
[ASV API Worker] Received response: {'prediction': 'bonafide', ...}
```

**Banner Queued:**
```
[AudioWorker-1] ✓ Queued GLOBAL audio banner (GREEN)
!******** AUDIO BANNER QUEUED FOR SENDING ********!
```

**Banner Sent:**
```
[ResponseSender] Sending banner: GLOBAL, level=GREEN, seq=1234
```

**Final Statistics:**
```
[Backend] Stream xyz final stats: ... audio queue: 0 chunks
  - Audio: 0/2 chunks, 45 received, 2 dropped
```

### Health Indicators

**✅ Healthy:**
- `total_dropped` < 5% of `total_received`
- `current_size` usually 0-1 (queue not backing up)
- Regular `Processing audio chunk` messages
- Banners sent to client

**⚠️ Warning:**
- `total_dropped` > 10% (processing too slow)
- `current_size` always at max (2) (queue constantly full)
- Long gaps between chunk processing

**🚨 Problem:**
- No `Processing audio chunk` messages (audio not reaching workers)
- API errors in logs
- `total_dropped` > 50%

## Troubleshooting

### Issue: No Audio Chunks Processed

**Check:**
1. Audio being sent by client? Look for `Route audio to global queue`
2. Worker started? Look for `Audio inference worker started`
3. Queue empty? Check final stats for `total_received`

**Fix:**
- Verify client is sending audio with WAV or OGG data
- Check server logs for errors during worker startup

### Issue: High Drop Rate

**Symptoms:**
- `total_dropped` > 10% of `total_received`
- Queue always full (size = 2)

**Causes:**
1. ASV API too slow (taking >8 seconds per chunk)
2. Network latency to API
3. Single worker overwhelmed

**Fix:**
```bash
# Increase workers to handle load
AUDIO_WORKER_COUNT=2 python server.py
```

### Issue: No Banners Sent

**Symptoms:**
- Audio processed but no banners in client
- Logs show `No verdict change, skipping banner`

**Explanation:**
- This is NORMAL behavior!
- AudioWindowManager only sends banners when verdict CHANGES
- If audio consistently "bonafide", no banner after first one

**Verify it's working:**
- First audio chunk should generate a banner
- Look for `AudioWindowManager` decision changes

### Issue: API Timeout

**Symptoms:**
```
[ASV API Worker] API timeout after 20s
```

**Causes:**
- ASV API server down or slow
- Network issues
- API overloaded

**Fix:**
```bash
# Increase timeout
ASV_API_TIMEOUT=30 python server.py

# Or check API server health
curl http://34.125.106.206:8000/health
```

## Performance Tuning

### Reduce Latency
- ✅ Already optimized (workers process immediately)
- Only latency is ASV API response time (~1-2s)

### Increase Throughput
```bash
# Add more workers (if multiple API endpoints available)
AUDIO_WORKER_COUNT=2
```

### Reduce Drops
```bash
# Increase queue size (trades freshness for fewer drops)
# Note: Requires code change - default is 2, max recommended is 3
```

## Queue Size Rationale

**Why queue_size = 2?**
- Audio chunks are 4 seconds each
- 2 chunks = 8 seconds max staleness
- ASV API responds in ~1-2 seconds
- Under normal load, queue stays empty or has 1 chunk
- Only fills (and drops) when API is slow or multiple chunks arrive rapidly

**When would you need more?**
- Multiple concurrent audio streams (not current use case)
- Very slow ASV API (>4s response time)
- Burst audio sending pattern

**Trade-off:**
- Larger queue = fewer drops but staler data
- Smaller queue = fresher data but more drops
- Size 2 is sweet spot for 4-second chunks

## Advanced Configuration

### Custom ASV API Parameters

Currently hardcoded in `_call_asv_api_async()`:
```python
params = {
    'window_step': '500',
    'use_vad': 'true',
    'vol_norm': 'false',
    'threshold': '0.55'
}
```

To customize, modify `server.py` or add environment variables (future enhancement).

### Multiple API Endpoints (Future)

When implemented:
```bash
AUDIO_API_URLS="http://server1:8000/asv/predict,http://server2:8000/asv/predict,http://server3:8000/asv/predict"
AUDIO_WORKER_COUNT=3
```

Each worker will round-robin through endpoints with health tracking.

---

**Last Updated:** October 28, 2025  
**Version:** 1.0 (Phase 3 Complete)
