# Integration Checklist: Four-Model Ensemble → WMA Server

## Quick Reference

### Current System (Single Model)
```python
# In server.py
class VideoAPIManager:
    def __init__(self):
        self.api_url = "http://34.16.217.28:8999/check_frame_batch"
    
    async def infer_probs_from_bytes(self, frames) -> List[float]:
        # Returns per-frame probabilities from ONE model
        return probs

# In participant_manager.py
class ParticipantManager:
    def process_and_decide(self, participant_id, probs):
        mean_prob = np.mean(probs)  # Simple average
        verdict = self._calculate_band_level(mean_prob)
        return verdict
```

### Target System (Four Models)
```python
# In test_four_model_fusion.py (to be integrated)
class FourModelAPIClient:
    def __init__(self):
        self.model_servers = {
            '9rfa62j1': 'http://IP1:8999',  # topk4
            '1mjgo9w1': 'http://IP2:8999',  # softmax_b5 ← different!
            'dfsesrgu': 'http://IP3:8999',  # topk4
            '4vtny88m': 'http://IP4:8999',  # topk4
        }
    
    async def process_frames(self, frames, participant_id):
        # 1. Call 4 APIs in parallel
        predictions = await self.get_all_model_predictions(frames)
        
        # 2. Aggregate per model (topk4 or softmax_b5)
        aggregated = self.aggregate_and_calibrate(predictions)
        
        # 3. Fuse with Noisy-OR
        fusion_score = noisy_or_fusion(aggregated)
        
        # 4. Classify with three-way thresholds
        verdict, confidence = classify_score(fusion_score)
        
        return {verdict, confidence, fusion_score, ...}
```

## Pre-Integration Checklist

### Phase 1: Infrastructure Setup
- [ ] Deploy 4 model servers (one per cluster)
  - [ ] Model 1 (9rfa62j1): EFS+Platform specialist → IP: ___________
  - [ ] Model 2 (1mjgo9w1): FaceSwap specialist → IP: ___________
  - [ ] Model 3 (dfsesrgu): Reenact specialist → IP: ___________
  - [ ] Model 4 (4vtny88m): Noisy-OR specialist → IP: ___________
- [ ] Verify each server responds to `/check_frame_batch`
- [ ] Test latency (should be < 5s per batch for 27 frames)

### Phase 2: Calibrator Export
- [ ] Go to `combined_model/` directory
- [ ] Load your training results: `out_full_v3/per_video_features.parquet`
- [ ] Extract fitted calibrators (IsotonicRegression objects)
- [ ] Save to pickle: `out_full_v3/calibrators.pkl`
- [ ] Verify file size (should be ~10-50 KB)

**Script to export calibrators:**
```python
# In combined_model/
import pickle
import pandas as pd
from sklearn.calibration import IsotonicRegression

# Load training metadata
with open('out_full_v3/fusion_meta.json', 'r') as f:
    meta = json.load(f)

# Calibrators should be saved during training
# If not, you need to re-fit them from oof_calibrated_probs.parquet
# ... (use code from run_video_level_fusion_v2.py)

# Save
with open('out_full_v3/calibrators.pkl', 'wb') as f:
    pickle.dump(calibrators_dict, f)

print(f"Saved calibrators: {list(calibrators_dict.keys())}")
```

### Phase 3: Standalone Testing
- [ ] Update `test_four_model_fusion.py`:
  - [ ] Set MODEL_SERVERS IPs
  - [ ] Verify THRESHOLDS (T_low=0.996700, T_high=0.998248)
- [ ] Run mock test: `python test_four_model_fusion.py --mock`
  - [ ] Verify REAL videos get scores < 0.9967
  - [ ] Verify FAKE videos get scores > 0.9982
  - [ ] Verify fusion scores near 1.0 for high inputs (this is correct!)
- [ ] Prepare test frames (27 JPEGs from a known video)
- [ ] Run real API test:
  ```bash
  python test_four_model_fusion.py \
      --participant-name "Test" \
      --num-frames 27 \
      --image-dir ./test_frames \
      --calibrators ../combined_model/out_full_v3/calibrators.pkl
  ```
- [ ] Verify:
  - [ ] All 4 APIs respond successfully
  - [ ] Predictions look reasonable (0-1 range)
  - [ ] Aggregation produces different values per model
  - [ ] Fusion score matches expectations
  - [ ] Final verdict is sensible

## Integration Steps

### Step 1: Copy Core Functions to `participant_manager.py`

Add these functions to the file (you can group them in a section):

```python
# ═══════════════════════════════════════════════════════════════
# Four-Model Ensemble Functions
# ═══════════════════════════════════════════════════════════════

def topk_mean(x: np.ndarray, k: int = 4) -> float:
    """Average of top k highest values."""
    if x.size == 0:
        return 0.0
    k = max(1, min(k, x.size))
    idx = np.argpartition(x, -k)[-k:]
    return float(np.mean(x[idx]))


def softmax_pool(x: np.ndarray, beta: float = 5.0) -> float:
    """Softmax pooling with temperature."""
    if x.size == 0:
        return 0.0
    m = np.max(beta * x)
    return float((np.log(np.mean(np.exp(beta * x - m))) + m) / beta)


def noisy_or_fusion(probs: np.ndarray) -> float:
    """Noisy-OR fusion: P(fake) = 1 - ∏(1 - p_i)"""
    return float(1.0 - np.prod(1.0 - probs))
```

### Step 2: Add CalibratorManager Class

```python
class CalibratorManager:
    """Manages isotonic calibrators for each model."""
    
    def __init__(self, calibrators_path: str):
        import pickle
        with open(calibrators_path, 'rb') as f:
            self.calibrators = pickle.load(f)
        logging.info(f"Loaded {len(self.calibrators)} calibrators")
    
    def calibrate(self, model_id: str, score: float) -> float:
        """Apply calibration to aggregated score."""
        return float(self.calibrators[model_id].transform([score])[0])
```

### Step 3: Add FourModelAPIClient Class

Copy the entire `FourModelAPIClient` class from `test_four_model_fusion.py` to `participant_manager.py`.

**Modifications needed:**
1. Import statements at top of file:
   ```python
   import aiohttp
   from typing import Dict, List, Tuple
   ```

2. Update MODEL_SERVERS config to pull from environment variables:
   ```python
   MODEL_SERVERS = {
       '9rfa62j1': {
           'host': os.getenv('MODEL1_HOST', '34.16.217.28'),
           'port': os.getenv('MODEL1_PORT', '8999'),
           'aggregator': 'topk4',
           'endpoint': '/check_frame_batch'
       },
       # ... similar for other models
   }
   ```

### Step 4: Update StreamingServiceImpl in `server.py`

Replace single-model initialization:

```python
# OLD:
class StreamingServiceImpl:
    def __init__(self):
        self.video_api = VideoAPIManager()
        self.participant_manager = ParticipantManager(
            threshold=self.video_api.threshold,
            margin=self.margin
        )

# NEW:
class StreamingServiceImpl:
    def __init__(self):
        # Load calibrators
        calibrators_path = os.getenv(
            'CALIBRATORS_PATH',
            '/path/to/calibrators.pkl'
        )
        calibrator_mgr = CalibratorManager(calibrators_path)
        
        # Initialize four-model client
        self.four_model_api = FourModelAPIClient(calibrator_mgr)
        
        # Use existing participant manager for now (we'll enhance it)
        self.participant_manager = ParticipantManager(
            threshold=0.75,  # Still used for some logic
            margin=self.margin
        )
```

### Step 5: Update `_generate_inference_banners` in `server.py`

Replace the single-model inference call:

```python
# OLD (line ~1080):
individual_probs = await self.video_api.infer_probs_from_bytes(image_bytes_list)

# Skip if no faces detected
if not individual_probs:
    continue

# Process with participant manager
manager_result = self.participant_manager.process_and_decide(
    participant_id, individual_probs
)

# NEW:
# Use four-model pipeline
result = await self.four_model_api.process_frames(
    image_bytes_list, 
    participant_id
)

# Skip if no faces detected
if result['verdict'] == 'REAL' and result['fusion_score'] == 0.0:
    continue

# Convert verdict to banner level
verdict_map = {
    'FAKE': pb2.RED,
    'UNCERTAIN': pb2.YELLOW,
    'REAL': pb2.GREEN
}
new_verdict_level = verdict_map[result['verdict']]
confidence_score = result['fusion_score']

# Continue with existing banner creation logic...
manager_result = (new_verdict_level, confidence_score)
```

### Step 6: Environment Configuration

Add to your server startup script or `.env` file:

```bash
# Four-Model Ensemble Configuration
export MODEL1_HOST="34.16.217.28"  # Model 9rfa62j1 (EFS+Platform)
export MODEL1_PORT="8999"

export MODEL2_HOST="34.16.217.29"  # Model 1mjgo9w1 (FaceSwap)
export MODEL2_PORT="8999"

export MODEL3_HOST="34.16.217.30"  # Model dfsesrgu (Reenact)
export MODEL3_PORT="8999"

export MODEL4_HOST="34.16.217.31"  # Model 4vtny88m (Noisy-OR)
export MODEL4_PORT="8999"

export CALIBRATORS_PATH="/home/roee/repos/Effort-AIGI-Detection/DeepfakeBench/training/combined_model/out_full_v3/calibrators.pkl"
```

## Testing After Integration

### Test 1: Verify Server Starts
```bash
python server.py --port 50051
# Should see: "Loaded 4 calibrators"
# Should see: "FourModelAPIClient initialized"
```

### Test 2: Shadow Mode Testing

Add dual-path logging temporarily:

```python
# In _generate_inference_banners
old_result = await self.video_api.infer_probs_from_bytes(image_bytes_list)
new_result = await self.four_model_api.process_frames(image_bytes_list, participant_id)

logging.info(f"[SHADOW] Single model: {np.mean(old_result):.4f}")
logging.info(f"[SHADOW] Four model: {new_result['fusion_score']:.6f}")
logging.info(f"[SHADOW] Verdicts: {old_verdict} vs {new_result['verdict']}")

# For now, still use old system for actual banners
# Switch to new system once confident
```

### Test 3: Performance Monitoring

Add timing:

```python
import time

start = time.time()
result = await self.four_model_api.process_frames(frames, pid)
elapsed = time.time() - start

logging.info(f"[PERF] Four-model pipeline: {elapsed:.2f}s for {len(frames)} frames")

# Target: < 5s for 27 frames
```

## Rollout Strategy

### Week 1: Shadow Mode
- [ ] Run both systems in parallel
- [ ] Log all results to file
- [ ] Compare verdicts (expect 90%+ agreement)
- [ ] Monitor latency (should be < 5s per batch)
- [ ] Check for errors/timeouts

### Week 2: Partial Rollout
- [ ] Route 10% of traffic to new system
- [ ] Monitor metrics:
  - [ ] False positive rate (should decrease)
  - [ ] False negative rate (should decrease)
  - [ ] Uncertain rate (~2%)
  - [ ] Latency (acceptable?)
- [ ] Adjust if needed

### Week 3+: Full Rollout
- [ ] Route 100% of traffic
- [ ] Remove old VideoAPIManager code
- [ ] Remove shadow logging
- [ ] Celebrate! 🎉

## Rollback Plan

If issues arise:

1. **Quick rollback**: Comment out new code, uncomment old code
2. **Restart server**: No data loss (stateless)
3. **Debug offline**: Use standalone test to reproduce
4. **Fix and retry**: Iterate until stable

## Common Issues & Solutions

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| "No calibrators found" | Path wrong | Check CALIBRATORS_PATH env var |
| All verdicts FAKE | No calibration applied | Verify calibrators loaded |
| API timeouts | Servers slow | Increase API_TIMEOUT or optimize servers |
| Different verdicts than training | Wrong aggregation | Verify topk4 vs softmax_b5 mapping |
| Fusion score = 0 | No faces detected | Expected behavior, returns REAL |

## Success Criteria

✅ **You're ready for production when:**
- [ ] Mock test passes (verdicts make sense)
- [ ] Real API test passes (all 4 servers respond)
- [ ] Shadow mode shows < 5s latency
- [ ] Shadow mode shows improved accuracy over single model
- [ ] No errors/exceptions in 1000+ test calls

## Resources

- **Standalone test**: `test_four_model_fusion.py`
- **Test guide**: `TEST_FOUR_MODEL_FUSION_README.md`
- **Training results**: `../combined_model/detection strategy results out full V3.txt`
- **Deployment guide**: `../combined_model/NEXT_STEPS_REALTIME_DEPLOYMENT.md`

## Questions?

If you run into issues during integration:
1. Check logs (look for [FourModelAPIClient] tags)
2. Re-run standalone test to isolate issue
3. Compare with training pipeline in `run_video_level_fusion_v2.py`
4. Verify model server responses (check API directly)

---

**Ready to integrate?** Start with the standalone test! 🚀

```bash
cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection/DeepfakeBench/training/wma
python test_four_model_fusion.py --mock
```
