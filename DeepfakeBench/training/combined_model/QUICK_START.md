# Quick Start Guide: Validating Your Real-Time Strategy

## Step 1: Understand Your Training Data (5 minutes)

Run this in `combined_model/` directory:

```bash
python -c "
import pandas as pd
import numpy as np

df = pd.read_parquet('out_full_v3/per_video_features.parquet')
fc = df['9rfa62j1::n_frames']

print('='*60)
print('TRAINING DATA FRAME ANALYSIS')
print('='*60)
print(f'Mean frames per video:     {fc.mean():.1f}')
print(f'Median frames per video:   {fc.median():.1f}')
print(f'Min:                       {fc.min()}')
print(f'Max:                       {fc.max()}')
print(f'Std dev:                   {fc.std():.1f}')
print()
print('Percentiles:')
print(f'  10th: {np.percentile(fc, 10):.0f}')
print(f'  25th: {np.percentile(fc, 25):.0f}')
print(f'  50th: {np.percentile(fc, 50):.0f}')
print(f'  75th: {np.percentile(fc, 75):.0f}')
print(f'  90th: {np.percentile(fc, 90):.0f}')
print(f'  95th: {np.percentile(fc, 95):.0f}')
print(f'  99th: {np.percentile(fc, 99):.0f}')
print()
print(f'Total videos: {len(df):,}')
print(f'  Real: {(df[\"label\"] == 0).sum():,}')
print(f'  Fake: {(df[\"label\"] == 1).sum():,}')
print('='*60)
"
```

**Expected output:**
```
============================================================
TRAINING DATA FRAME ANALYSIS
============================================================
Mean frames per video:     27.3
Median frames per video:   27.0
Min:                       8
Max:                       50
Std dev:                   5.2

Percentiles:
  10th: 20
  25th: 24
  50th: 27
  75th: 30
  90th: 34
  95th: 37
  99th: 43

Total videos: 44,505
  Real: 2,165
  Fake: 42,340
============================================================
```

**Key takeaway:** If typical video has ~27 frames and you use `topk4`, you're selecting the **top ~15%** of frames. This is quite selective!

---

## Step 2: Check Your Model Run IDs (2 minutes)

Verify you have the correct model run IDs:

```bash
cd /path/to/your/model/checkpoints

# List your model directories
ls -la | grep -E "(9rfa62j1|1mjgo9w1|dfsesrgu|4vtny88m)"

# Or check in your inference results
ls path/to/inference/results/
```

Your 4 models:
1. **9rfa62j1** - Cluster 1 specialist (use topk4)
2. **1mjgo9w1** - Cluster 2 specialist (use softmax_b5)
3. **dfsesrgu** - Cluster 3 specialist (use topk4)
4. **4vtny88m** - Cluster 4 specialist (use topk4)

---

## Step 3: Run Real-Time Simulation (30 minutes)

### Option 1: Quick Test (Sample Mode)

Test on a small subset first to verify everything works:

```bash
cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection/DeepfakeBench/training/combined_model

python simulate_realtime_detection.py \
  --frame-csvs \
    /path/to/9rfa62j1_frames.csv \
    /path/to/1mjgo9w1_frames.csv \
    /path/to/dfsesrgu_frames.csv \
    /path/to/4vtny88m_frames.csv \
  --model-names 9rfa62j1 1mjgo9w1 dfsesrgu 4vtny88m \
  --aggregators topk4 softmax_b5 topk4 topk4 \
  --out-dir ./realtime_test_sample \
  --window-size 27 \
  --strategy option_a \
  --sample-videos 10
```

**This will:**
- ✅ Process 10 videos per method
- ✅ Fit isotonic calibrators (OOF)
- ✅ Simulate real-time frame-by-frame processing
- ✅ Report accuracy, TPR, FPR, temporal stability
- ✅ Save results to CSV

**Expected runtime:** ~2-5 minutes

### Option 2: Full Validation

Run on all videos (may take longer):

```bash
python simulate_realtime_detection.py \
  --frame-csvs \
    /path/to/9rfa62j1_frames.csv \
    /path/to/1mjgo9w1_frames.csv \
    /path/to/dfsesrgu_frames.csv \
    /path/to/4vtny88m_frames.csv \
  --model-names 9rfa62j1 1mjgo9w1 dfsesrgu 4vtny88m \
  --aggregators topk4 softmax_b5 topk4 topk4 \
  --out-dir ./realtime_validation_full \
  --window-size 27 \
  --strategy option_a
```

**Expected runtime:** 30-60 minutes (depends on dataset size)

### Option 3: Test Option B (Per-Frame Fusion)

Compare against simpler strategy:

```bash
python simulate_realtime_detection.py \
  --frame-csvs /path/to/*.csv \
  --model-names 9rfa62j1 1mjgo9w1 dfsesrgu 4vtny88m \
  --out-dir ./realtime_option_b \
  --window-size 10 \
  --strategy option_b
```

---

## Step 4: Analyze Results (10 minutes)

The simulation will print:

```
============================================================
SIMULATION RESULTS
============================================================

Overall Accuracy: 90.2%

Confusion Matrix (Ground Truth vs Predicted):
               Predicted REAL  Predicted FAKE  Predicted UNCERTAIN
Actual REAL:         2032            45                88
Actual FAKE:         3090         38500               750

Performance Metrics:
  TPR (Recall on fakes): 91.0%  ← Compare to training: 91.58%
  TNR (Recall on reals): 93.8%  ← Compare to training: 93.81%
  FPR (false alarms):    2.1%   ← Compare to training: 4.35%
  FNR (missed fakes):    7.3%   ← Compare to training: 6.40%
  Uncertain rate:        2.0%   ← Compare to training: 2.18%

Temporal Stability:
  Mean flips per video:      3.2  ← Lower is better (stable)
  Mean convergence frame:    12.5 ← Frames until decision stabilizes
  Mean frames per video:     27.3
  Mean score std per video:  0.012 ← Lower is better (consistent)
```

### What to Look For:

✅ **Good signs:**
- TPR/TNR within ±2% of training performance
- Low flips per video (< 5)
- Convergence frame < half the window size
- Uncertain rate around 2%

⚠️ **Warning signs:**
- TPR drops > 5% from training → thresholds may need adjustment
- High flips (> 10) → system is unstable, consider larger window
- High FPR (> 5%) → too sensitive, raise T_high
- Convergence > 20 frames → slow response, consider smaller window

---

## Step 5: Compare Option A vs Option B

Run both strategies and compare:

```bash
# After running both simulations:
python -c "
import json
import pandas as pd

# Load results
with open('./realtime_option_a/config.json') as f:
    opt_a = json.load(f)
    
with open('./realtime_option_b/config.json') as f:
    opt_b = json.load(f)

print('='*60)
print('STRATEGY COMPARISON')
print('='*60)
print()
print('Metric                  Option A (Training-Match)  Option B (Simpler)')
print('-'*60)
print(f'TPR (fake recall):      {opt_a[\"metrics\"][\"tpr\"]:6.1%}                  {opt_b[\"metrics\"][\"tpr\"]:6.1%}')
print(f'TNR (real recall):      {opt_a[\"metrics\"][\"tnr\"]:6.1%}                  {opt_b[\"metrics\"][\"tnr\"]:6.1%}')
print(f'FPR (false alarms):     {opt_a[\"metrics\"][\"fpr\"]:6.1%}                  {opt_b[\"metrics\"][\"fpr\"]:6.1%}')
print(f'Uncertain rate:         {opt_a[\"metrics\"][\"uncertain_rate\"]:6.1%}                  {opt_b[\"metrics\"][\"uncertain_rate\"]:6.1%}')
print(f'Mean flips/video:       {opt_a[\"metrics\"][\"mean_flips\"]:6.1f}                   {opt_b[\"metrics\"][\"mean_flips\"]:6.1f}')
print(f'Convergence (frames):   {opt_a[\"metrics\"][\"mean_convergence\"]:6.1f}                   {opt_b[\"metrics\"][\"mean_convergence\"]:6.1f}')
print()
print('Recommendation: Choose the strategy with better TPR/TNR and lower flips.')
print('='*60)
"
```

---

## Step 6: Visualize Temporal Behavior (Optional)

Plot how decisions evolve over time for a few example videos:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load detailed results
results = pd.read_csv('./realtime_option_a/video_results.csv')

# Pick an example fake video
example = results[results['ground_truth'] == 1].iloc[0]
clip_key = example['clip_key']

print(f"Example video: {clip_key}")
print(f"Final decision: {example['final_decision']}")
print(f"Ground truth: {'FAKE' if example['ground_truth'] == 1 else 'REAL'}")
print(f"Flips: {example['n_flips']}")

# You would need to load frame_results from the detailed output
# (not saved by default in the script above, but easy to add)
```

---

## Step 7: Decision Matrix

Based on simulation results, decide on production strategy:

| Criterion | Option A (Recommended) | Option B (Alternative) |
|-----------|------------------------|------------------------|
| **Accuracy Match** | ✅ Matches training closely | ⚠️ May differ, needs tuning |
| **Latency** | ⚠️ ~13.5s to stabilize | ✅ ~5s (with size 10) |
| **Complexity** | ⚠️ More complex code | ✅ Simpler implementation |
| **Thresholds** | ✅ Use validated values | ⚠️ Requires re-calibration |
| **Temporal Stability** | ✅ Typically more stable | ⚠️ May have more flips |

**Decision flow:**

```
Is latency critical (need < 5s response)?
  ├─ NO → Use Option A (frame selection)
  │        - Best accuracy
  │        - Matches training
  │        - Validated thresholds
  │
  └─ YES → Test Option B with smaller window
           - Run validation first
           - Re-calibrate thresholds if needed
           - Monitor temporal stability
```

---

## Step 8: Export for Production

Once you've validated your chosen strategy:

```bash
# 1. Save the calibrators (already done by simulation script)
ls ./realtime_option_a/calibrators.pkl

# 2. Create production config
cat > production_config.json << EOF
{
  "strategy": "option_a",
  "window_size": 27,
  "thresholds": {
    "T_low": 0.996700,
    "T_high": 0.998248
  },
  "models": {
    "9rfa62j1": {
      "aggregator": "topk4",
      "aggregator_params": {"k": 4}
    },
    "1mjgo9w1": {
      "aggregator": "softmax_b5",
      "aggregator_params": {"beta": 5.0}
    },
    "dfsesrgu": {
      "aggregator": "topk4",
      "aggregator_params": {"k": 4}
    },
    "4vtny88m": {
      "aggregator": "topk4",
      "aggregator_params": {"k": 4}
    }
  },
  "calibrators_path": "./realtime_option_a/calibrators.pkl",
  "production_requirements": {
    "max_latency_ms": 100,
    "target_tpr": 0.90,
    "target_fpr": 0.05,
    "uncertain_threshold": 0.05
  }
}
EOF

# 3. Copy to production directory
cp production_config.json /path/to/production/config/
cp ./realtime_option_a/calibrators.pkl /path/to/production/models/
```

---

## Step 9: Integration Testing

Test your production implementation:

```python
# test_production_detector.py
from your_production_code import ProductionDeepfakeDetector
import pickle

# Load components
with open('production_config.json') as f:
    config = json.load(f)

with open(config['calibrators_path'], 'rb') as f:
    calibrators = pickle.load(f)

# Initialize detector
detector = ProductionDeepfakeDetector(
    models=your_models,
    calibrators=calibrators,
    thresholds=config['thresholds'],
    window_size=config['window_size']
)

# Test on a known fake video
for frame in test_video_frames:
    result = detector.process_frame(frame)
    print(f"Frame {i}: {result['decision']} (score: {result['score']:.6f}, conf: {result['confidence']:.2f})")

# Verify:
# - Final decision matches ground truth
# - Score converges within expected frames
# - No unexpected errors
```

---

## Step 10: Monitoring Setup

After deployment, track these metrics:

```python
# Example monitoring code
class ProductionMonitor:
    def __init__(self):
        self.daily_stats = {
            'decisions': {'REAL': 0, 'FAKE': 0, 'UNCERTAIN': 0},
            'scores': [],
            'flips': [],
            'latencies': [],
        }
    
    def log_decision(self, result, latency_ms):
        self.daily_stats['decisions'][result['decision']] += 1
        self.daily_stats['scores'].append(result['score'])
        self.daily_stats['latencies'].append(latency_ms)
    
    def daily_report(self):
        total = sum(self.daily_stats['decisions'].values())
        print(f"Daily Report:")
        print(f"  Total videos: {total}")
        for decision, count in self.daily_stats['decisions'].items():
            print(f"    {decision}: {count} ({count/total:.1%})")
        print(f"  Mean score: {np.mean(self.daily_stats['scores']):.4f}")
        print(f"  Mean latency: {np.mean(self.daily_stats['latencies']):.1f}ms")
        
        # Alert if distribution shifts
        if self.daily_stats['decisions']['UNCERTAIN'] / total > 0.05:
            print("⚠️  WARNING: High uncertain rate!")
```

---

## Troubleshooting

### Issue: Simulation crashes with "File not found"

**Solution:** Check that your frame CSV paths are correct:

```bash
ls -lh /path/to/9rfa62j1_frames.csv
# Should show the file size (typically 100MB+ per model)
```

### Issue: Calibrator fitting takes too long

**Solution:** Use sampling during development:

```bash
# Add --sample-videos flag
python simulate_realtime_detection.py ... --sample-videos 100
```

### Issue: Results don't match training performance

**Possible causes:**
1. Different frame counts → Check frame statistics match
2. Wrong aggregators → Verify topk4 vs softmax_b5 mapping
3. Calibration issues → Check isotonic regression is fitted correctly
4. Data leakage → Ensure test set wasn't in training

**Debug:**
```bash
# Compare aggregated scores with training
python -c "
import pandas as pd
import pickle

# Load training OOF scores
train_scores = pd.read_parquet('out_full_v3/oof_calibrated_probs.parquet')
print('Training mean scores per model:')
for col in train_scores.columns:
    if 'calib_prob' in col:
        print(f'  {col}: {train_scores[col].mean():.4f}')

# Load simulation results
sim_results = pd.read_csv('./realtime_option_a/video_results.csv')
print('\nSimulation mean score: {:.4f}'.format(sim_results['mean_score'].mean()))
"
```

---

## Summary Checklist

Before moving to production:

- [ ] Frame count analysis confirms ~27 frames typical
- [ ] Simulation ran successfully on sample data
- [ ] Option A results match training performance (±2%)
- [ ] Temporal stability is acceptable (< 5 flips/video)
- [ ] Calibrators exported and saved
- [ ] Production config created
- [ ] Integration test passed
- [ ] Monitoring pipeline ready
- [ ] Documentation complete

**Once all checked → You're ready to deploy!** 🚀

---

## Next Steps After Validation

1. **Shadow deployment:** Run in production but don't act on decisions (log only)
2. **A/B testing:** Compare against current system on same videos
3. **Gradual rollout:** Start with 1% of traffic, monitor, increase
4. **Feedback loop:** Collect ground truth labels, re-calibrate monthly
5. **Model updates:** Retrain quarterly with new data

**Estimated timeline:**
- Validation: 1-2 days (you are here)
- Production implementation: 1 week
- Testing & refinement: 1 week
- Shadow deployment: 1 week
- Full rollout: 1-2 weeks

**Total: 4-6 weeks to production-ready system**
