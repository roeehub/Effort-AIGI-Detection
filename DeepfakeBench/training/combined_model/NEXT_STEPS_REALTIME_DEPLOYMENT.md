# Real-Time Deepfake Detection: Implementation Strategy

## Executive Summary

You have successfully trained and validated a **multi-model ensemble system** that achieves:
- **90.52% TPR** (True Positive Rate) on all fakes
- **4.35% FPR** (False Positive Rate) on reals  
- **2.18%** marked as uncertain

Your challenge now is **adapting this batch-trained video-level system to a real-time frame-by-frame streaming architecture**.

---

## Part 1: What Your Training Pipeline Does

### 1.1 Training Data Structure
From your command output:
```
Frame count columns: ['9rfa62j1::n_frames', '1mjgo9w1::n_frames', ...]
First row sample:
   label method  9rfa62j1::n_frames  ...
0      1    dit                  27  ...
1      1    dit                  25  ...
2      1    dit                  29  ...
```

**Key findings:**
- **Frame counts per video:** Typically 25-30 frames (likely mean ~27)
- **All models see the same frames** (n_frames is identical across models for each video)
- **Frame selection:** The training used a subset of frames from full videos

### 1.2 The Four Expert Models

| Model ID    | Specialization | Best Aggregator | Frame-to-Video Strategy |
|-------------|----------------|-----------------|-------------------------|
| 9rfa62j1    | Model 1        | `topk4`         | Average of top 4 frames |
| 1mjgo9w1    | Model 2        | `softmax_b5`    | Softmax pooling (β=5)   |
| dfsesrgu    | Model 3        | `topk4`         | Average of top 4 frames |
| 4vtny88m    | Model 4        | `topk4`         | Average of top 4 frames |

**From `run_video_level_fusion_v2.py` (lines 187-204):**
```python
def aggregate_per_video(df: pd.DataFrame) -> pd.DataFrame:
    # For each video clip:
    probs = g["frame_prob"].to_numpy()
    
    aggs = {
        "topk4": topk_mean(probs, 4),      # Mean of top 4 highest scores
        "softmax_b5": softmax_pool(probs, 5.0),  # Exp-weighted average
        # ... other aggregators
    }
```

**What `topk4` does:**
```python
def topk_mean(x: np.ndarray, k: int) -> float:
    k = max(1, min(k, x.size))
    idx = np.argpartition(x, -k)[-k:]  # Get indices of top k values
    return float(np.mean(x[idx]))      # Return their average
```

**What `softmax_b5` does:**
```python
def softmax_pool(x: np.ndarray, beta: float) -> float:
    # Stabilized log-sum-exp softmax pooling
    m = np.max(beta * x)
    return float((np.log(np.mean(np.exp(beta * x - m))) + m) / beta)
    # Higher beta = more weight to max values
```

### 1.3 Fusion & Calibration Pipeline

```
Per-Frame Predictions (from 4 models)
    ↓
Video-Level Aggregation (topk4/softmax_b5)
    ↓
Isotonic Calibration (5-fold OOF)
    ↓
Noisy-OR Fusion: P(fake) = 1 - ∏(1 - p_i)
    ↓
Final Decision:
  - score < 0.996700 → REAL
  - score > 0.998248 → FAKE
  - between → UNCERTAIN
```

### 1.4 Why Noisy-OR Produces Such High Scores

**Mathematical reality:**
- If 4 models each say 0.90: `1 - (1-0.90)^4 = 1 - 0.0001 = 0.9999`
- If 4 models each say 0.95: `1 - (1-0.95)^4 = 1 - 0.0000062 = 0.999994`

**This is correct behavior:**
- Noisy-OR assumes models make **independent errors**
- Each model is a "detector" that can catch the fake
- If all 4 experts agree it's fake, confidence should be very high
- Hence your threshold jumped from 0.8 → 0.998 (not a bug!)

---

## Part 2: Your Real-Time System Architecture

### 2.1 Current Production Setup
```
Video Call Stream (~2 fps)
    ↓
Each frame → All 4 models → [p1, p2, p3, p4]
    ↓
Noisy-OR fusion per frame → single score
    ↓
Sliding window buffer (N most recent frames)
    ↓
Average window scores → final decision
```

### 2.2 The Critical Mismatch

| Training Pipeline | Your Real-Time System |
|-------------------|----------------------|
| 27 frames per video (typical) | Continuous stream |
| topk4: Select best 4 frames | No frame selection (yet) |
| Video-level aggregation | Sliding window average |
| OOF calibration on aggregated scores | Per-frame fusion, then averaging |

**The problem:**
- Training optimized for "which 4 frames best represent this video?"
- Production gives equal weight to all frames in window
- Calibration was done on aggregated video-level scores
- You're applying it to individual frame-level scores

---

## Part 3: Your Next Steps (Prioritized)

### Step 1: Understand Training Frame Distribution ⚠️ **CRITICAL**

**Action:** Analyze the actual frame count distribution in your training data.

```python
# Run this in combined_model/ directory:
import pandas as pd
import numpy as np

df = pd.read_parquet('out_full_v3/per_video_features.parquet')
frame_counts = df['9rfa62j1::n_frames']

print(f"Mean frames: {frame_counts.mean():.1f}")
print(f"Median: {frame_counts.median():.1f}")
print(f"10/25/75/90/95th percentiles: {np.percentile(frame_counts, [10,25,75,90,95])}")
```

**Why this matters:**
- If typical video has ~27 frames and you use topk4, you're selecting the **top 15%** of frames
- This tells you how selective your training aggregation was
- Informs your sliding window size and frame selection strategy

### Step 2: Choose Your Real-Time Adaptation Strategy

You have **three main options:**

#### Option A: Mimic Training with Frame Selection (Recommended) ✅

**Concept:** Maintain a sliding window and apply the same aggregation logic as training.

```python
# Pseudocode for real-time system
class RealtimeDetector:
    def __init__(self):
        self.window_size = 27  # Match typical training video length
        self.frame_buffer = deque(maxlen=self.window_size)
        
    def process_frame(self, frame):
        # Get predictions from all 4 models
        preds = {
            'model1': model1.predict(frame),
            'model2': model2.predict(frame),
            'model3': model3.predict(frame),
            'model4': model4.predict(frame),
        }
        
        # Add to buffer
        self.frame_buffer.append(preds)
        
        # Aggregate per model (matching training)
        agg_scores = {
            'model1': topk_mean([f['model1'] for f in self.frame_buffer], k=4),
            'model2': softmax_pool([f['model2'] for f in self.frame_buffer], beta=5),
            'model3': topk_mean([f['model3'] for f in self.frame_buffer], k=4),
            'model4': topk_mean([f['model4'] for f in self.frame_buffer], k=4),
        }
        
        # Apply isotonic calibration (pre-fitted on training data)
        cal_scores = {k: calibrators[k].transform([v])[0] for k, v in agg_scores.items()}
        
        # Noisy-OR fusion
        fused = 1 - np.prod([1 - p for p in cal_scores.values()])
        
        # Decision
        if fused < 0.996700:
            return "REAL"
        elif fused > 0.998248:
            return "FAKE"
        else:
            return "UNCERTAIN"
```

**Pros:**
- Closest match to training distribution
- Uses validated thresholds as-is
- Leverages model-specific aggregation strategies

**Cons:**
- Needs 27-frame buffer to stabilize (13.5 seconds at 2fps)
- Initial frames won't have full context

#### Option B: Frame-Level Fusion with Rolling Average (Simpler)

**Concept:** Fuse per-frame, then average over window.

```python
def process_frame_v2(self, frame):
    # Get predictions and fuse immediately
    preds = [m.predict(frame) for m in self.models]
    
    # Apply per-model calibration at frame level (may need re-calibration!)
    cal_preds = [cal.transform([p])[0] for cal, p in zip(self.calibrators, preds)]
    
    # Fuse per frame
    fused_frame = 1 - np.prod([1 - p for p in cal_preds])
    
    # Add to window
    self.fusion_buffer.append(fused_frame)
    
    # Average fused scores
    final_score = np.mean(self.fusion_buffer)
    
    return classify(final_score)
```

**Pros:**
- Simpler logic
- Faster response (can use smaller window)

**Cons:**
- **Different distribution than training**
- May need to **re-calibrate** thresholds on validation set
- Loses model-specific aggregation strategies

#### Option C: Hybrid with Adaptive Thresholding

**Concept:** Option B + dynamic threshold adjustment based on window statistics.

```python
def process_frame_v3(self, frame):
    # ... same as Option B up to final_score ...
    
    # Consider uncertainty of the window
    window_std = np.std(self.fusion_buffer)
    window_range = np.max(self.fusion_buffer) - np.min(self.fusion_buffer)
    
    # Adjust thresholds based on stability
    if window_std < 0.01:  # Very stable
        t_low, t_high = 0.996700, 0.998248
    else:  # High variance = less confident
        t_low = max(0.996700 - window_std, 0.95)
        t_high = min(0.998248 + window_std, 0.999)
    
    # Classify with adaptive thresholds
    if final_score < t_low:
        return "REAL"
    elif final_score > t_high:
        return "FAKE"
    else:
        return "UNCERTAIN"
```

**Pros:**
- More robust to temporal instability
- Can flag high-variance windows as uncertain

**Cons:**
- More complex
- Needs validation on real-world data

### Step 3: Validate on Held-Out Data

**Critical:** Whatever approach you choose, you **must validate** on data the models haven't seen.

```python
# Create a validation script
def validate_realtime_strategy(video_frames, ground_truth):
    """
    Simulates real-time processing on complete videos.
    
    video_frames: List of frame paths for a video
    ground_truth: 0 (real) or 1 (fake)
    """
    detector = RealtimeDetector()
    decisions = []
    
    for frame in video_frames:
        decision = detector.process_frame(frame)
        decisions.append(decision)
    
    # Analyze decision stability
    fake_ratio = decisions.count("FAKE") / len(decisions)
    uncertain_ratio = decisions.count("UNCERTAIN") / len(decisions)
    
    # Final video-level decision (e.g., majority vote)
    final = max(set(decisions), key=decisions.count)
    
    return {
        'ground_truth': ground_truth,
        'final_decision': final,
        'fake_ratio': fake_ratio,
        'uncertain_ratio': uncertain_ratio,
        'stability': 1 - np.std([d == final for d in decisions])
    }
```

### Step 4: Export Calibrators and Thresholds

**Action:** Save your trained components for production.

```python
# In your training directory:
import pickle

# Load calibrators from training
with open('out_full_v3/fusion_meta.json', 'r') as f:
    meta = json.load(f)
    
# You need to export the actual IsotonicRegression objects
# These were fitted in run_video_level_fusion_v2.py but not saved
# You'll need to re-run with:

# Add to run_video_level_fusion_v2.py after line 258:
import pickle
calibrators_export = {
    name: calibrators 
    for name, (_, calibrators) in zip(args.names, [
        calibrate_isotonic_oof(raw_scores[n], y) 
        for n in args.names
    ])
}
with open(os.path.join(args.outdir, "isotonic_calibrators.pkl"), "wb") as f:
    pickle.dump(calibrators_export, f)
```

**Then in production:**
```python
import pickle

# Load calibrators
with open("isotonic_calibrators.pkl", "rb") as f:
    calibrators = pickle.load(f)

# Load thresholds
thresholds = {
    'T_low': 0.996700,
    'T_high': 0.998248
}
```

---

## Part 4: Practical Implementation Checklist

### Phase 1: Understanding (1-2 days)
- [ ] Run frame count analysis (Step 1)
- [ ] Document actual frame distribution (mean, median, percentiles)
- [ ] Calculate what % of frames topk4 represents
- [ ] Review production video call frame rates and typical durations

### Phase 2: Strategy Selection (2-3 days)
- [ ] Choose between Options A, B, or C based on:
  - Latency requirements (how fast must you respond?)
  - Acceptable buffer size (can you wait 10-15 seconds?)
  - Development complexity
- [ ] Write design doc with chosen approach
- [ ] Get stakeholder buy-in

### Phase 3: Implementation (1 week)
- [ ] Implement chosen sliding window strategy
- [ ] Export and load calibrators properly
- [ ] Add logging for debugging (frame scores, window stats, decisions)
- [ ] Handle edge cases (first N frames, model errors, etc.)

### Phase 4: Validation (1-2 weeks)
- [ ] Create validation dataset (held-out videos)
- [ ] Simulate real-time processing on validation set
- [ ] Measure metrics:
  - Frame-level accuracy
  - Video-level accuracy (after window stabilizes)
  - Decision latency (frames until stable)
  - Temporal stability (decision flipping rate)
- [ ] If Option B: Re-calibrate thresholds if needed
- [ ] A/B test against current simple averaging

### Phase 5: Deployment (1 week)
- [ ] Gradual rollout (shadow mode first)
- [ ] Monitor production metrics vs validation
- [ ] Set up alerts for:
  - High uncertainty rate (> 5%)
  - Unstable windows (high variance)
  - Model prediction failures
- [ ] Document operational procedures

---

## Part 5: Key Technical Decisions You Need to Make

### Decision 1: Window Size
**Question:** How many frames should your sliding window hold?

**Options:**
- **Match training (~27 frames):** Most faithful to training, but ~13.5s latency at 2fps
- **Smaller (10-15 frames):** Faster response (~5-7.5s), may need threshold adjustment
- **Adaptive:** Start small, grow to 27, then maintain

**Recommendation:** Start with 27 to match training, optimize later if latency is critical.

### Decision 2: Aggregation Strategy per Model
**Question:** Apply model-specific aggregation (topk4/softmax) in production?

**Trade-offs:**
- **Yes (Option A):** Matches training exactly, uses validated thresholds
- **No (Option B):** Simpler code, but different distribution

**Recommendation:** Use Option A initially. The performance gain from model-specific aggregation (91.58% TPR) justifies the complexity.

### Decision 3: Calibration Timing
**Question:** When to apply calibration?

**Options:**
1. **After aggregation (like training):** 
   - Aggregate 27 frames → single score per model → calibrate → fuse
2. **Before aggregation:**
   - Calibrate each frame → aggregate calibrated scores → fuse
3. **After fusion:**
   - Aggregate → fuse → calibrate final score (not recommended)

**Recommendation:** After aggregation (option 1) to match training.

### Decision 4: Cold Start Strategy
**Question:** What to do when buffer has < 27 frames?

**Options:**
- **Wait:** Don't make decision until buffer is full (conservative)
- **Use available:** Apply topk4 to whatever you have (may be less reliable)
- **Lower threshold initially:** Start with more permissive thresholds, tighten as buffer fills

**Recommendation:** Use available frames with "UNCERTAIN" classification until buffer fills.

---

## Part 6: Code Template for Option A (Recommended)

```python
import numpy as np
from collections import deque
from typing import Dict, List
import pickle

class ProductionDeepfakeDetector:
    """
    Real-time deepfake detector matching training pipeline.
    
    Uses:
    - 27-frame sliding window (typical training video length)
    - Model-specific aggregation (topk4/softmax_b5)
    - Isotonic calibration (fitted on training OOF)
    - Noisy-OR fusion
    - Three-way classification (REAL/FAKE/UNCERTAIN)
    """
    
    def __init__(self, 
                 models: Dict[str, object],
                 calibrators: Dict[str, object],
                 thresholds: Dict[str, float],
                 window_size: int = 27):
        """
        Args:
            models: {'9rfa62j1': model1_obj, '1mjgo9w1': model2_obj, ...}
            calibrators: {'9rfa62j1': IsotonicRegression, ...}
            thresholds: {'T_low': 0.996700, 'T_high': 0.998248}
            window_size: Number of frames in sliding window
        """
        self.models = models
        self.calibrators = calibrators
        self.T_low = thresholds['T_low']
        self.T_high = thresholds['T_high']
        self.window_size = window_size
        
        # Sliding window buffer
        self.frame_buffer = deque(maxlen=window_size)
        
        # Aggregation functions (from training)
        self.aggregators = {
            '9rfa62j1': lambda x: self._topk_mean(x, 4),
            '1mjgo9w1': lambda x: self._softmax_pool(x, 5.0),
            'dfsesrgu': lambda x: self._topk_mean(x, 4),
            '4vtny88m': lambda x: self._topk_mean(x, 4),
        }
    
    def _topk_mean(self, x: np.ndarray, k: int) -> float:
        """Average of top k values."""
        if x.size == 0:
            return 0.0
        k = max(1, min(k, x.size))
        idx = np.argpartition(x, -k)[-k:]
        return float(np.mean(x[idx]))
    
    def _softmax_pool(self, x: np.ndarray, beta: float) -> float:
        """Softmax pooling with temperature beta."""
        if x.size == 0:
            return 0.0
        m = np.max(beta * x)
        return float((np.log(np.mean(np.exp(beta * x - m))) + m) / beta)
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Image frame (numpy array)
            
        Returns:
            Dict with 'decision', 'score', 'confidence', 'buffer_fill'
        """
        # 1. Get per-model predictions for this frame
        frame_preds = {}
        for model_id, model in self.models.items():
            frame_preds[model_id] = model.predict(frame)
        
        # 2. Add to buffer
        self.frame_buffer.append(frame_preds)
        
        # 3. Check if buffer is sufficiently filled
        buffer_fill = len(self.frame_buffer) / self.window_size
        
        # 4. Aggregate per model across buffer
        aggregated = {}
        for model_id in self.models.keys():
            # Extract this model's scores from all buffered frames
            scores = np.array([f[model_id] for f in self.frame_buffer])
            # Apply model-specific aggregation
            aggregated[model_id] = self.aggregators[model_id](scores)
        
        # 5. Apply isotonic calibration
        calibrated = {}
        for model_id, score in aggregated.items():
            calibrated[model_id] = self.calibrators[model_id].transform([score])[0]
        
        # 6. Noisy-OR fusion
        fusion_score = 1.0 - np.prod([1.0 - p for p in calibrated.values()])
        
        # 7. Three-way classification
        if fusion_score < self.T_low:
            decision = "REAL"
            confidence = 1.0 - fusion_score  # Higher confidence = lower score
        elif fusion_score > self.T_high:
            decision = "FAKE"
            confidence = fusion_score  # Higher confidence = higher score
        else:
            decision = "UNCERTAIN"
            confidence = 0.5  # Neutral confidence
        
        # Adjust confidence based on buffer fill (lower if not full)
        confidence *= buffer_fill
        
        return {
            'decision': decision,
            'score': fusion_score,
            'confidence': confidence,
            'buffer_fill': buffer_fill,
            'aggregated': aggregated,  # For debugging
            'calibrated': calibrated,  # For debugging
        }
    
    def reset(self):
        """Clear the buffer (e.g., between different video calls)."""
        self.frame_buffer.clear()


# Usage example
if __name__ == "__main__":
    # Load components
    models = {
        '9rfa62j1': load_model('model1.pth'),
        '1mjgo9w1': load_model('model2.pth'),
        'dfsesrgu': load_model('model3.pth'),
        '4vtny88m': load_model('model4.pth'),
    }
    
    with open('isotonic_calibrators.pkl', 'rb') as f:
        calibrators = pickle.load(f)
    
    thresholds = {'T_low': 0.996700, 'T_high': 0.998248}
    
    # Initialize detector
    detector = ProductionDeepfakeDetector(models, calibrators, thresholds)
    
    # Process video stream
    for frame in video_stream:
        result = detector.process_frame(frame)
        print(f"Decision: {result['decision']}, "
              f"Score: {result['score']:.6f}, "
              f"Confidence: {result['confidence']:.2f}, "
              f"Buffer: {result['buffer_fill']:.0%}")
        
        # Act on decision
        if result['decision'] == "FAKE" and result['confidence'] > 0.8:
            alert_moderator()
```

---

## Part 7: Expected Challenges & Solutions

### Challenge 1: High Latency
**Problem:** 27-frame buffer at 2fps = 13.5 seconds to fill.

**Solutions:**
- Use smaller window (10 frames = 5s) and re-calibrate thresholds on validation
- Start decision-making with partial buffer but mark as lower confidence
- Increase frame rate to 4fps if network/compute allows

### Challenge 2: Temporal Instability
**Problem:** Decision flips between consecutive frames.

**Solutions:**
- Add temporal smoothing: require N consecutive FAKE frames before alerting
- Use exponential moving average instead of simple mean
- Increase uncertainty band width to absorb flips

### Challenge 3: Different Real-World Distribution
**Problem:** Your production fakes may differ from training (lighting, compression, etc.).

**Solutions:**
- Monitor production score distributions
- Set up A/B testing with ground truth labels
- Periodic re-calibration with production data
- Feature drift detection (alert if score distribution shifts)

### Challenge 4: Model Inference Speed
**Problem:** 4 models × 2 fps = 8 inferences/second.

**Solutions:**
- Batch inference (process multiple frames together)
- Model quantization (INT8) for faster inference
- GPU optimization (TensorRT, ONNX Runtime)
- Consider dropping to 3 models if one adds little value

---

## Part 8: Monitoring & Maintenance

### Key Metrics to Track

1. **Decision Distribution:**
   - % REAL / % FAKE / % UNCERTAIN per day
   - Alert if uncertain rate > 5% (indicates drift or edge cases)

2. **Temporal Stability:**
   - Average consecutive frames before decision flip
   - Ideally > 10 frames (5 seconds)

3. **Score Distribution:**
   - Mean, std, percentiles of fusion scores
   - Alert if distribution shifts (indicates model drift)

4. **Per-Model Contribution:**
   - How often each model is the "decisive" one
   - Helps identify if a model becomes redundant

5. **Latency:**
   - Time from frame receipt to decision
   - Target: < 100ms per frame

### Maintenance Schedule

- **Weekly:** Review decision distribution, check for anomalies
- **Monthly:** Validate on new ground-truth labeled data
- **Quarterly:** Re-calibrate if production distribution drifted
- **Annually:** Retrain models on updated dataset

---

## Summary: Your Action Plan

### Immediate (This Week)
1. ✅ Run frame count analysis on training data
2. ✅ Choose strategy (recommend Option A)
3. ✅ Export calibrators and thresholds from training

### Short-term (Next 2 Weeks)
4. ⏳ Implement ProductionDeepfakeDetector class
5. ⏳ Validate on held-out videos (simulate real-time)
6. ⏳ Measure temporal stability and latency

### Medium-term (Next Month)
7. ⏳ Deploy in shadow mode (log decisions but don't act)
8. ⏳ Compare shadow results to ground truth
9. ⏳ Tune thresholds if needed based on production distribution

### Long-term (Next Quarter)
10. ⏳ Full production deployment
11. ⏳ Set up monitoring dashboard
12. ⏳ Establish re-training pipeline for model updates

---

## Questions to Answer Before Proceeding

1. **What is your actual latency requirement?**
   - Can you wait 13.5 seconds for stable decision?
   - Or do you need sub-second response (would require Option B)?

2. **What is your production frame rate?**
   - Exactly 2fps or variable?
   - Can it be increased?

3. **What is your error cost trade-off?**
   - Is false positive (blocking real user) worse than false negative (missing fake)?
   - Informs threshold adjustment

4. **Do you have a validation set?**
   - Videos with ground truth labels not seen during training?
   - Essential for validating real-time strategy

5. **What is your deployment environment?**
   - Edge device, cloud server, mobile?
   - Affects model optimization strategy

---

## Final Recommendation

**Start with Option A** (frame selection matching training):
- Implement 27-frame sliding window
- Use model-specific aggregation (topk4/softmax_b5)
- Apply validated thresholds (T_low=0.996700, T_high=0.998248)
- Mark early frames as uncertain until buffer fills

**Then iterate:**
- Validate on held-out data
- Measure temporal stability
- Optimize window size if latency is critical
- Monitor production distribution and re-calibrate if needed

**You're not starting from scratch** - you have a well-validated system. The challenge is *adaptation*, not *creation*. Take it step-by-step, validate at each stage, and you'll have a production-ready real-time detector in 3-4 weeks.

---

**Questions? Start here:**
1. Run the frame count analysis
2. Review your production requirements (latency, error costs)
3. Pick Option A, B, or C
4. Then we can dive into implementation details
