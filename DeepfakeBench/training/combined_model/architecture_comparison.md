# Architecture Comparison: Training vs Production

## Training Pipeline (Batch Processing)

```
Video with N frames (typically ~27)
           ↓
  [Frame 1] [Frame 2] ... [Frame 27]
           ↓
    ┌─────────┴─────────┬─────────┬─────────┐
    ↓                   ↓         ↓         ↓
[Model 1]           [Model 2] [Model 3] [Model 4]
9rfa62j1            1mjgo9w1  dfsesrgu  4vtny88m
    ↓                   ↓         ↓         ↓
[0.89, 0.92,       [0.87,    [0.91,    [0.90,
 0.78, 0.95,        0.89,     0.88,     0.93,
 ..., 0.88]         0.91,     0.94,     0.89,
 (27 scores)        ...]      ...]      ...]
    ↓                   ↓         ↓         ↓
┌───────────────┐  ┌───────────┐ ┌────────┐ ┌────────┐
│ topk4         │  │softmax_b5 │ │ topk4  │ │ topk4  │
│ (avg of top 4)│  │(exp-wtd   │ │        │ │        │
│ = 0.935       │  │ avg)      │ │ = 0.930│ │= 0.928 │
└───────────────┘  │ = 0.905   │ └────────┘ └────────┘
                   └───────────┘
         ↓              ↓            ↓          ↓
    ┌──────────────────────────────────────────┐
    │    Isotonic Calibration (OOF fitted)     │
    │    Transforms scores to calibrated probs │
    └──────────────────────────────────────────┘
         ↓              ↓            ↓          ↓
     [0.945]        [0.920]      [0.935]    [0.930]
         ↓              ↓            ↓          ↓
    ┌──────────────────────────────────────────┐
    │           Noisy-OR Fusion                │
    │  P(fake) = 1 - ∏(1-p_i)                  │
    │  = 1 - (1-0.945)(1-0.920)(1-0.935)(1-0.930)│
    │  = 1 - (0.055 × 0.080 × 0.065 × 0.070)  │
    │  = 1 - 0.0000158 = 0.999984              │
    └──────────────────────────────────────────┘
                        ↓
              [Score: 0.999984]
                        ↓
         ┌──────────────┴──────────────┐
         │  Three-Way Classification   │
         │  T_low  = 0.996700          │
         │  T_high = 0.998248          │
         └─────────────────────────────┘
                        ↓
         < 0.996700 → REAL (93.81% reals)
   0.996700-0.998248 → UNCERTAIN (2.18% all)
         > 0.998248 → FAKE (90.52% fakes)
```

---

## Production Pipeline (Real-Time Streaming)

### Option A: Frame Selection (Recommended - Matches Training)

```
Real-Time Video Stream @ 2fps
           ↓
    [Frame t] [Frame t+1] [Frame t+2] ...
           ↓
  ┌────────────────────────────────────┐
  │  Sliding Window Buffer (27 frames) │
  │  = 13.5 seconds of video           │
  │                                    │
  │  [t-26] [t-25] ... [t-1] [t]      │
  └────────────────────────────────────┘
           ↓
    For Each Model Separately:
           ↓
  ┌─────────────────────────────────────────┐
  │ Model 1: Extract 27 frame predictions   │
  │ [0.89, 0.92, 0.78, ..., 0.88]          │
  │         ↓                               │
  │  Apply topk4 aggregation                │
  │  = 0.935                                │
  │         ↓                               │
  │  Isotonic calibrate: 0.935 → 0.945     │
  └─────────────────────────────────────────┘
           ↓
  Similar for Models 2, 3, 4
  (using softmax_b5, topk4, topk4)
           ↓
  [0.945, 0.920, 0.935, 0.930]
           ↓
  ┌─────────────────────────────────────────┐
  │  Noisy-OR Fusion                        │
  │  → 0.999984                             │
  └─────────────────────────────────────────┘
           ↓
  Apply Thresholds → Decision
           ↓
    Update buffer with new frame
    (drop frame t-27, add frame t+1)
           ↓
    Repeat for next frame
```

**Key Properties:**
- ✅ Matches training distribution exactly
- ✅ Uses validated thresholds (0.996700, 0.998248)
- ✅ Model-specific aggregation strategies
- ⚠️  Requires 13.5s buffer to stabilize
- ⚠️  Initial frames marked as uncertain

---

### Option B: Per-Frame Fusion (Simpler but Different)

```
Real-Time Video Stream @ 2fps
           ↓
    Single Frame [t]
           ↓
    ┌─────┴─────┬─────┬─────┐
    ↓           ↓     ↓     ↓
[Model 1]   [Model 2] [3]  [4]
  0.89       0.87    0.91  0.90
    ↓           ↓     ↓     ↓
Calibrate each (frame-level!)
  0.900      0.880  0.920 0.905
    ↓           ↓     ↓     ↓
    └─────┬─────┴─────┴─────┘
          ↓
  Noisy-OR Fusion
  → 0.999638
          ↓
  ┌────────────────────────────┐
  │ Sliding Window of          │
  │ Fused Scores (e.g., 10)    │
  │ [0.9996, 0.9998, ...]      │
  └────────────────────────────┘
          ↓
  Average window: 0.999720
          ↓
  Apply Thresholds (may need adjustment!)
          ↓
  Decision
```

**Key Differences from Training:**
- ⚠️  Calibration applied at frame level, not aggregated
- ⚠️  Different score distribution
- ⚠️  Loses model-specific aggregation strategies
- ✅  Faster response (can use smaller window)
- ✅  Simpler implementation

**Likely requires:**
- Re-calibration of thresholds on validation set
- Empirical testing to find optimal window size

---

## Visual Comparison of Aggregation Strategies

### topk4 (Models 1, 3, 4)

```
Frame Scores: [0.89, 0.92, 0.78, 0.95, 0.85, 0.88, 0.91, 0.82, ...]
                     ↓
Sort & Select Top 4: [0.95, 0.92, 0.91, 0.89]
                     ↓
Average: (0.95 + 0.92 + 0.91 + 0.89) / 4 = 0.9175

Rationale: Focus on frames with strongest fake signals
           Robust to noise from ambiguous frames
```

### softmax_b5 (Model 2)

```
Frame Scores: [0.89, 0.92, 0.78, 0.95, 0.85]
                     ↓
Softmax with β=5:
  exp(5×0.89) = 66.69
  exp(5×0.92) = 90.02
  exp(5×0.78) = 19.84
  exp(5×0.95) = 114.59
  exp(5×0.85) = 40.05
                     ↓
Weighted average (higher scores get exponentially more weight):
  (0.89×66.69 + 0.92×90.02 + 0.78×19.84 + 0.95×114.59 + 0.85×40.05) / 331.19
  ≈ 0.907

Rationale: Smooth version of max, less sensitive to outliers than topk
           β=5 is moderately selective
```

---

## Score Distribution Comparison

### Training (After Aggregation)

```
Real Videos:
  █████████████                    (mean ~0.20, most < 0.5)
  |-----------|-----------|---------|
  0.0        0.5        0.996      1.0

Fake Videos (Supported):
                        █████████████ (mean ~0.999, concentrated near 1.0)
  |-----------|-----------|---------|
  0.0        0.5        0.996      1.0
                              ↑     ↑
                           T_low T_high
```

### Production (Frame-Level Before Aggregation) - Expected

```
Real Frames:
  ████████████████                  (mean ~0.30, higher variance)
  |-----------|-----------|---------|
  0.0        0.5        0.996      1.0

Fake Frames (even from "good" videos):
                  ██████████████████ (mean ~0.95, wider spread)
  |-----------|-----------|---------|
  0.0        0.5        0.996      1.0

After Aggregation (Option A):
  Should match training distribution ✅
  
After Averaging Fused Frames (Option B):
  May be shifted/compressed ⚠️
  Requires validation to confirm thresholds still work
```

---

## Decision Timeline Comparison

### Training (Batch)

```
Time:   [Video Start] ──────────────── [All frames processed] → [Decision]
Frames: Frame 1, 2, ..., 27 (all at once)
Latency: N/A (offline processing)
Result: Single video-level decision
```

### Production Option A (27-frame buffer)

```
Time:   0s ────── 5s ────── 10s ────── 13.5s ──────→
Frames: [1-5]     [1-10]    [1-20]    [1-27] [2-28] [3-29]...
Status: UNCERTAIN UNCERTAIN UNCERTAIN  STABLE STABLE STABLE
                                       ↑
                               First confident decision
                               (after buffer fills)
```

### Production Option B (10-frame buffer)

```
Time:   0s ─── 2.5s ─── 5s ───────────→
Frames: [1-5]  [1-10]  [11-20] [21-30]...
Status: UNCER  STABLE  STABLE  STABLE
                ↑
        Faster first decision
        (but may need threshold tuning)
```

---

## Buffer Fill Strategy Visualization

```
Cold Start (first 27 frames):

Frame:  1    5     10    15    20    25   27
Buffer: [▓]  [▓▓▓▓] [▓▓▓▓▓][▓▓▓▓▓][▓▓▓▓▓][▓▓]
Fill:   4%   19%   37%   56%   74%   93%  100%
topk4:  topk1 topk2 topk4 topk4 topk4 topk4 topk4
        ↓    ↓     ↓     ↓     ↓     ↓    ↓
Score:  0.89 0.91  0.93  0.94  0.95  0.96 0.97
Conf:   LOW  LOW   MED   MED   HIGH  HIGH HIGH
                                           ↑
                                    First fully-confident
                                    decision

Steady State (buffer full):

New frame arrives → drop oldest → aggregate → decide
                    ↓
Continuous updates every 0.5s (@ 2fps)
```

---

## Summary: Key Architectural Decisions

| Aspect | Training | Option A (Recommended) | Option B (Simpler) |
|--------|----------|------------------------|-------------------|
| **Frame input** | 27 frames/video | 27-frame window | Per-frame |
| **Aggregation** | topk4/softmax_b5 | Same ✅ | None ⚠️ |
| **Calibration timing** | After aggregation | After aggregation ✅ | Per-frame ⚠️ |
| **Fusion** | Noisy-OR | Noisy-OR ✅ | Noisy-OR ✅ |
| **Thresholds** | 0.996700, 0.998248 | Use as-is ✅ | Need re-validation ⚠️ |
| **Latency to stable** | N/A | ~13.5s | ~5s |
| **Distribution match** | N/A | Exact match ✅ | Different ⚠️ |
| **Implementation** | Batch | Complex | Simple |

**Recommendation:** Start with Option A, measure latency in production, optimize to Option B only if necessary.
