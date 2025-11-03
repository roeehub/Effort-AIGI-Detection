# Audio Worker Testing Checklist

## Pre-Testing Setup

- [ ] Code compiles without errors (`python -m py_compile server.py queue_manager.py`)
- [ ] Server starts successfully (`python server.py`)
- [ ] Log file created (`wma_server.log`)
- [ ] ASV API endpoint reachable (`curl http://34.125.106.206:8000/`)

## Startup Tests

### Worker Initialization
- [ ] Log shows: `[Backend] NEW ARCHITECTURE: 4 video workers, 1 audio workers`
- [ ] Log shows: `[Backend] Starting 1 audio inference worker(s)...`
- [ ] Log shows: `[AudioWorker-1] Audio inference worker started`
- [ ] No errors in startup logs

### Queue Creation
- [ ] Log shows: `audio queue` in configuration
- [ ] Audio queue initialized with max_size=2

## Runtime Tests

### Audio Reception
- [ ] Client sends audio chunks
- [ ] Log shows: `Route audio to global queue` (in fast consumer)
- [ ] Audio queue stats show `total_received > 0`

### Audio Processing
- [ ] Log shows: `[AudioWorker-1] Processing audio chunk`
- [ ] Log shows chunk age (should be <1 second normally)
- [ ] Log shows chunk size (should be >0 bytes)

### API Communication
- [ ] Log shows: `[ASV API Worker] Sending X bytes of MP3 audio`
- [ ] Log shows conversion format: `(from wav)` or `(from ogg)`
- [ ] Log shows: `[ASV API Worker] Received response`
- [ ] Response includes `prediction` field

### Banner Generation
- [ ] First audio chunk generates a banner
- [ ] Log shows: `[AudioWorker-1] ✓ Queued GLOBAL audio banner`
- [ ] Log shows banner level (GREEN, YELLOW, or RED)
- [ ] Log shows: `!******** AUDIO BANNER QUEUED FOR SENDING ********!`

### Banner Delivery
- [ ] Log shows: `[ResponseSender] Sending banner: GLOBAL`
- [ ] Client receives audio banner
- [ ] Banner has correct properties:
  - [ ] `scope = "global"`
  - [ ] `scope_enum = SCOPE_GLOBAL`
  - [ ] `placement = "TopCenter"`
  - [ ] `banner_type = "audio_ok"` or `"audio_alert"`

### Subsequent Audio Chunks
- [ ] Additional chunks processed (log shows multiple `Processing audio chunk`)
- [ ] AudioWindowManager processes results
- [ ] Banners only sent when verdict CHANGES (not every chunk)
- [ ] Log shows: `No verdict change, skipping banner` (this is NORMAL)

## Performance Tests

### Latency
- [ ] Measure time from client send to banner receive
- [ ] Should be <4 seconds (API call ~1-2s + minimal queueing)
- [ ] Compare to old architecture (should be ~20-25s faster)

### Throughput
- [ ] Send 10 audio chunks rapidly
- [ ] All chunks processed (check `total_received` in stats)
- [ ] Minimal drops (check `total_dropped` in stats)
- [ ] Drop rate <5%

### Queue Behavior
- [ ] Under normal load, queue size = 0 or 1
- [ ] Queue only fills to 2 when processing slow
- [ ] Old chunks dropped when queue full (log shows drops)

## Stress Tests

### Rapid Audio Sending
- [ ] Send 20 chunks in 10 seconds (2 chunks/sec)
- [ ] Worker keeps up (queue doesn't stay full)
- [ ] API calls complete successfully
- [ ] Some drops acceptable (queue size = 2)

### API Slowdown Simulation
- [ ] (If possible) Slow down API to 5-10 seconds
- [ ] Queue fills to max (size = 2)
- [ ] Chunks start dropping (expected behavior)
- [ ] System remains stable (no crashes)
- [ ] Drop rate reported in final stats

### Long-Running Stream
- [ ] Run for 5+ minutes with continuous audio
- [ ] Worker remains stable (no memory leaks)
- [ ] Consistent processing times
- [ ] Banners continue to send when verdicts change

## Edge Cases

### No Audio Data
- [ ] Send uplink messages without audio
- [ ] Worker remains idle (no errors)
- [ ] No audio chunks queued

### Invalid Audio Format
- [ ] (If possible) Send corrupted audio
- [ ] Worker handles gracefully (logs error, continues)
- [ ] No worker crash

### RESTART Signal
- [ ] Send audio with `session_id = "[RESTART]"`
- [ ] AudioWindowManager reset
- [ ] Log shows: `*** RESTART SIGNAL DETECTED ***`
- [ ] Next audio chunk processed normally

### API Failure
- [ ] Stop ASV API server (or block network)
- [ ] Worker logs API error
- [ ] Worker continues running (doesn't crash)
- [ ] When API restored, processing resumes

## Shutdown Tests

### Graceful Shutdown
- [ ] Stop server (Ctrl+C)
- [ ] Log shows: `[Backend] Stopping audio inference workers...`
- [ ] Log shows: `[AudioWorker-1] Audio inference worker stopped`
- [ ] Final stats include audio queue metrics
- [ ] Stats show: `total_received`, `total_dropped`, drop rate

### Queue Statistics
- [ ] Final log shows audio queue stats
- [ ] Format: `- Audio: X/2 chunks, Y received, Z dropped`
- [ ] Drop rate calculated correctly

## Comparison with Old Architecture

### Before (Sequential Processing)
- Audio processed in `_generate_audio_inference_banner()`
- Blocked consumer loop for 1-2 seconds per chunk
- 20s timeout for API call
- Accumulated delay over time

### After (Worker Architecture)
- Audio routed to queue (instant)
- Worker processes asynchronously
- Consumer never blocked
- Minimal delay

### Metrics to Compare
- [ ] Processing latency (should be ~20s faster)
- [ ] Audio chunks received (should be same or more)
- [ ] Banner frequency (should be same - only on verdict change)
- [ ] System stability (should be same or better)

## Known Good Behavior

✅ **These are EXPECTED and NORMAL:**
1. `No verdict change, skipping banner` - AudioWindowManager smoothing
2. Some dropped chunks when queue full - freshness guarantee working
3. First chunk always generates a banner
4. Subsequent chunks may not generate banners (verdict stable)
5. Queue usually empty (size = 0) - good sign, processing keeping up

## Known Issues / Limitations

⚠️ **Current Limitations:**
1. Only 1 ASV API endpoint (no load balancing yet)
2. Single chunk per API call (no batching)
3. No health tracking for audio API
4. Queue size hardcoded to 2 (requires code change to adjust)

## Success Criteria

**Phase 3 is successful if:**
- ✅ Audio chunks received and queued
- ✅ Audio worker processes chunks asynchronously
- ✅ GLOBAL banners generated correctly
- ✅ Banners delivered to client
- ✅ No blocking I/O in consumer loop
- ✅ Drop rate <10% under normal load
- ✅ Latency reduced compared to old architecture
- ✅ System stable for extended runs

## Failure Scenarios

**Phase 3 has issues if:**
- ❌ No audio chunks reach worker (routing problem)
- ❌ Worker crashes on error (error handling broken)
- ❌ Banners not sent (response queue integration broken)
- ❌ Drop rate >50% (queue too small or worker too slow)
- ❌ Memory leak (worker not cleaning up resources)
- ❌ Consumer blocked (worker still using blocking I/O)

---

**Testing Date:** _____________  
**Tester:** _____________  
**Result:** ⬜ PASS ⬜ FAIL ⬜ NEEDS INVESTIGATION  

**Notes:**
```


```
