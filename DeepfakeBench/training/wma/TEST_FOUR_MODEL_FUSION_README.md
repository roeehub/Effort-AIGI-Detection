# Four-Model Ensemble Fusion Testing Guide

## Overview

This document explains the standalone test system for the four-model ensemble that you'll integrate into your WMA server.

## What This Test Does

The `test_four_model_fusion.py` script mimics **exactly** how your production system will work:

```
Incoming Frames (1-32)
    ↓
Call 4 Model Servers in Parallel
    ↓
Get Per-Frame Probabilities (4 lists)
    ↓
Aggregate Per Model (topk4 or softmax_b5)
    ↓
Apply Isotonic Calibration (per model)
    ↓
Fuse with Noisy-OR
    ↓
Classify (REAL / UNCERTAIN / FAKE)
    ↓
Return Decision + Confidence
```

## Why Test Standalone First?

✅ **Validate the math** - Ensure aggregation and fusion work correctly  
✅ **Test without risk** - No changes to production code  
✅ **Understand the data flow** - See exactly what happens at each step  
✅ **Debug easily** - Isolated test environment with clear logging  
✅ **Prepare for integration** - Once working, copy logic to `participant_manager.py`

## File Structure

```
wma/
├── test_four_model_fusion.py          # ← The standalone test (NEW)
├── TEST_FOUR_MODEL_FUSION_README.md   # ← This file (NEW)
├── server.py                          # Your current production server
├── participant_manager.py             # Will be modified later
└── ...
```

## Configuration

### Step 1: Update Model Server IPs

In `test_four_model_fusion.py`, update the `MODEL_SERVERS` dictionary (lines 40-63):

```python
MODEL_SERVERS = {
    '9rfa62j1': {  # Model 1 - EFS+Platform specialist
        'host': 'YOUR_ACTUAL_IP_HERE',  # ← UPDATE THIS
        'port': '8999',
        'aggregator': 'topk4',
        'endpoint': '/check_frame_batch'
    },
    '1mjgo9w1': {  # Model 2 - FaceSwap specialist
        'host': 'YOUR_ACTUAL_IP_HERE',  # ← UPDATE THIS
        'port': '8999',
        'aggregator': 'softmax_b5',  # ← Note: Different aggregator!
        'endpoint': '/check_frame_batch'
    },
    # ... models 3 and 4
}
```

**Important:** Model 2 uses `softmax_b5`, all others use `topk4`!

### Step 2: Verify Thresholds

The thresholds from your training results are already configured:

```python
THRESHOLDS = {
    'T_low': 0.996700,   # Below this = REAL
    'T_high': 0.998248,  # Above this = FAKE
}
```

These are the validated thresholds from `detection strategy results out full V3.txt`.

## Running Tests

### Test 1: Mock Data (No API Calls)

Start here! This tests the fusion logic without needing running servers:

```bash
cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection/DeepfakeBench/training/wma
python test_four_model_fusion.py --mock
```

**What it does:**
- Generates fake predictions (high scores for fake, low for real)
- Tests aggregation (topk4, softmax_b5)
- Tests Noisy-OR fusion
- Tests three-way classification
- Verifies the math is correct

**Expected output:**
```
================================================================================
TEST 1: Real video (low scores)
================================================================================
[9rfa62j1] topk4: 27 frames → aggregated=0.1234 → calibrated=0.1234
[1mjgo9w1] softmax_b5: 27 frames → aggregated=0.1567 → calibrated=0.1567
...
Noisy-OR fusion:
  9rfa62j1: 0.123400
  1mjgo9w1: 0.156700
  ...
  → Fused: 0.450000
Result: REAL (confidence: High, score: 0.450000)

================================================================================
TEST 2: Fake video (high scores)
================================================================================
...
Result: FAKE (confidence: High, score: 0.999500)
```

**What to verify:**
- ✅ Real videos get scores < 0.9967 → classified as REAL
- ✅ Fake videos get scores > 0.9982 → classified as FAKE
- ✅ High input scores (0.90+) lead to fusion scores near 1.0 (this is correct!)

### Test 2: Real API Calls (When Servers Ready)

Once your 4 model servers are deployed:

```bash
# First, save some test frames
mkdir test_frames
# ... copy some JPEG frames here ...

# Run test with real API calls
python test_four_model_fusion.py \
    --participant-name "Test User" \
    --num-frames 27 \
    --image-dir ./test_frames
```

**What it does:**
- Loads actual frame images from disk
- Calls all 4 model APIs in parallel
- Runs the full fusion pipeline
- Shows real results

### Test 3: Debug Mode

For detailed logging:

```bash
python test_four_model_fusion.py --mock --debug
```

Shows step-by-step details:
- Individual frame probabilities
- Aggregation intermediate values
- Calibration transformations
- Noisy-OR calculation breakdown

## Understanding the Output

### Mock Test Output

```
================================================================================
PROCESSING 27 frames for 'test_participant'
================================================================================
[9rfa62j1] Mock probs: mean=0.9200, min=0.8700, max=0.9500
[1mjgo9w1] Mock probs: mean=0.9100, min=0.8600, max=0.9400
[dfsesrgu] Mock probs: mean=0.9300, min=0.8800, max=0.9600
[4vtny88m] Mock probs: mean=0.9150, min=0.8650, max=0.9450

Aggregating predictions per model...
[9rfa62j1] topk4: 27 frames → aggregated=0.9450 → calibrated=0.9450
[1mjgo9w1] softmax_b5: 27 frames → aggregated=0.9380 → calibrated=0.9380
[dfsesrgu] topk4: 27 frames → aggregated=0.9550 → calibrated=0.9550
[4vtny88m] topk4: 27 frames → aggregated=0.9430 → calibrated=0.9430

Noisy-OR fusion:
  9rfa62j1: 0.945000
  1mjgo9w1: 0.938000
  dfsesrgu: 0.955000
  4vtny88m: 0.943000
  → Fused: 0.999997  ← High confidence!

================================================================================
FINAL RESULT
================================================================================
  Participant: test_participant
  Frames: 27
  Fusion Score: 0.999997
  Verdict: FAKE
  Confidence: High
================================================================================
```

**Key observations:**
1. Each model returns ~27 per-frame probabilities
2. `topk4` selects top 4 → average ~0.945 (top 15% of frames)
3. `softmax_b5` gives similar result with exponential weighting
4. Noisy-OR with 4 high scores → 0.9999+ (mathematically correct!)
5. Score > 0.9982 → FAKE verdict

### Why Fusion Scores Are So High

This is **correct behavior**, not a bug!

**Math example:**
```python
# If all 4 models say 0.90:
fusion = 1 - (1 - 0.90)^4 = 1 - 0.0001 = 0.9999

# If all 4 models say 0.95:
fusion = 1 - (1 - 0.95)^4 = 1 - 0.00000625 = 0.999994
```

This is why your thresholds are so high:
- T_low = 0.9967 (below = REAL)
- T_high = 0.9982 (above = FAKE)

The system requires **strong agreement** from all 4 models to call something FAKE.

## Integration Path to WMA Server

Once you've validated the standalone test works, here's how to integrate:

### Step 1: Export Calibrators from Training

In `combined_model/`, run this to save calibrators:

```python
import pickle
from run_video_level_fusion_v2 import *

# Load your training results
df = pd.read_parquet('out_full_v3/per_video_features.parquet')

# Fit calibrators (or load existing ones)
# ... your calibration code ...

# Save
with open('out_full_v3/calibrators.pkl', 'wb') as f:
    pickle.dump(calibrators, f)
```

### Step 2: Copy Code to `participant_manager.py`

Add a new class to `participant_manager.py`:

```python
class FourModelParticipantManager(ParticipantManager):
    """
    Enhanced participant manager using 4-model ensemble.
    """
    
    def __init__(self, calibrators_path: Path, ...):
        super().__init__(...)
        # Copy FourModelAPIClient logic here
        self.api_client = FourModelAPIClient(...)
    
    async def process_and_decide(self, participant_id: str, 
                                 image_bytes_list: List[bytes]):
        """
        Replace single-model logic with 4-model fusion.
        """
        # Use self.api_client.process_frames() from test file
        result = await self.api_client.process_frames(image_bytes_list, participant_id)
        
        # Convert to banner level
        verdict_map = {
            'FAKE': pb2.RED,
            'UNCERTAIN': pb2.YELLOW,
            'REAL': pb2.GREEN
        }
        
        return (verdict_map[result['verdict']], result['fusion_score'])
```

### Step 3: Update `server.py`

Replace `VideoAPIManager` usage:

```python
# OLD (single model):
self.video_api = VideoAPIManager()
individual_probs = await self.video_api.infer_probs_from_bytes(image_bytes_list)

# NEW (four models):
self.four_model_client = FourModelAPIClient(calibrator_manager)
result = await self.four_model_client.process_frames(image_bytes_list, participant_id)
```

### Step 4: Test in Production Shadow Mode

Run both systems in parallel and log results:

```python
# Get results from both systems
old_result = await single_model_inference(frames)
new_result = await four_model_inference(frames)

# Log comparison
logger.info(f"Single model: {old_result['verdict']}")
logger.info(f"Four model: {new_result['verdict']}")

# For now, use old system for actual decisions
# Later, switch to new system when confident
```

## Troubleshooting

### Issue: All verdicts are FAKE

**Cause:** Calibrators not loaded, using identity function (no calibration)

**Fix:** 
1. Export calibrators from training: `calibrators.pkl`
2. Pass path when initializing: `CalibratorManager(Path('calibrators.pkl'))`

### Issue: API timeouts

**Cause:** Model servers taking too long (>30s)

**Fix:**
1. Increase `API_TIMEOUT` in config
2. Optimize model servers (batch size, GPU utilization)
3. Consider using fewer frames (e.g., 15 instead of 27)

### Issue: Fusion scores not matching training

**Cause:** Different aggregation or frame selection

**Fix:**
1. Verify topk4 vs softmax_b5 mapping is correct
2. Check frame count (should be ~27 for full window)
3. Ensure calibration is being applied

### Issue: No faces detected

**Cause:** YOLO not finding faces in frames

**Fix:**
1. Lower `YOLO_CONF_THRESHOLD` (currently 0.80)
2. Check frame quality (resolution, lighting)
3. Verify frame cropping is including faces

## Next Steps

### Immediate (Testing Phase)
1. ✅ Run mock tests to validate logic
2. ⏸️ Set up 4 model servers on different IPs
3. ⏸️ Update `MODEL_SERVERS` config with real IPs
4. ⏸️ Export calibrators from training
5. ⏸️ Run real API tests with actual frames

### Short-term (Integration Phase)
6. ⏸️ Copy validated code to `participant_manager.py`
7. ⏸️ Add `FourModelParticipantManager` class
8. ⏸️ Update `server.py` to use new manager
9. ⏸️ Test in shadow mode (parallel with existing system)

### Medium-term (Production Phase)
10. ⏸️ Gradually roll out (1% → 10% → 100% traffic)
11. ⏸️ Monitor performance vs single model
12. ⏸️ Tune thresholds if needed (based on production data)

## Questions?

Common questions answered:

**Q: Why separate servers for each model?**  
A: Each model is a specialist (EFS+Platform, FaceSwap, Reenact, etc.). Separating them allows independent scaling and easier debugging.

**Q: Can I use the same server for multiple models?**  
A: Yes, but you'll need to modify the API to accept a `model_id` parameter to route to the correct model.

**Q: What if one model server is down?**  
A: Current code will log error and use 0.0 for that model. You may want to add fallback logic (e.g., skip that model in fusion).

**Q: How do I tune thresholds?**  
A: Use `define_detection_strategy.py` in `combined_model/` to find optimal thresholds on your validation data.

**Q: Why is Model 2 different (softmax_b5)?**  
A: Training found `softmax_b5` worked best for the FaceSwap specialist model. Always use model-specific aggregators!

## Summary

This standalone test gives you:
- ✅ **Safe testing environment** (no production impact)
- ✅ **Clear understanding** of the fusion pipeline
- ✅ **Validated logic** before integration
- ✅ **Easy debugging** with detailed logging
- ✅ **Production-ready code** to copy into WMA server

Run the mock test now to see it in action! 🚀

```bash
python test_four_model_fusion.py --mock
```
