# Deployment gate + thresholds — 2026-05-13

## A) Production gate spec

Apply this gate **upstream of the model** (i.e., before scoring a frame). If a
frame fails any check, do not score it — treat it as "unknown" / pass through to
your higher-level decision policy (or simply abstain).

### Hard gate (must-pass)

```python
def production_gate(face_crop_image, face_detector_output) -> dict:
    """Returns {'pass': bool, 'reason': str}."""
    # G1: face detector confidence
    if not face_detector_output.has_face:
        return {'pass': False, 'reason': 'no_face'}

    # G2: minimum resolution — drops ~21% of OOD chronic-tiny-face cohort
    W, H = face_crop_image.size
    if min(W, H) < 200:
        return {'pass': False, 'reason': 'lowres_lt_200'}

    return {'pass': True, 'reason': 'ok'}
```

**Roy_D handling**: G1+G2 will NOT catch Roy_D-style substrate (Roy_D has
median min_wh=270 — passes G2; faces are clearly detected — passes G1). Roy_D's
characteristic is **soft-focus + warm colors**, which neither gate catches. If
your production stream contains warm-color + soft-focus subjects, expect FPR to
run hot on T5C (and to a lesser degree T3_SLOT1_step1500 — they share the
warm-color fragility).

### Soft gate (optional, recommended for T5C)

If you can afford to compute sharpness:

```python
def soft_gate_sharpness(face_crop_image) -> dict:
    """Additional gate that filters soft-focus frames. Adds ~10% drop."""
    import cv2, numpy as np
    gray = cv2.cvtColor(np.asarray(face_crop_image), cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 80:
        return {'pass': False, 'reason': 'soft_focus'}
    return {'pass': True, 'reason': 'ok'}
```

**Justification**: Roy_D's median Laplacian variance is 69 (soft); the same
soft-focus characterizes PC_Generator__s8/s34 and most of the non-chronic FPs
in the existing audit. Setting threshold at 80 filters out ~10% of frames
including most Roy_D-like cases, without aggressively eating into normal Teams
traffic (Cam_Test__s32=91, Test_Cam__s76=101 still pass; typical sharp Teams
content sits at >150 Laplacian). The 2026-05-04 IQ gate viability finding
(`project_iq_gating_viability_2026-05-04`) explicitly notes IQ gating is a P8A
lever — and T5C inherits P8A's responsiveness to sharpness gating.

### Why these specific gates

| Gate | What it drops | What it doesn't catch |
|---|---|---|
| G1 (no-face)   | ~5% of eval-pool with detection failures | nothing model-relevant |
| G2 (min_wh<200)| pc_generator__s22, pc_generator__s45, q__s6 (the 3 tiny-face chronics); bla_bla_chow__s2 (146px) — ~22% of eval | bla_bla_chow (399px wide), Roy_D (270px) |
| G3 sharpness (optional, lap_var<80) | Roy_D, bla_bla_chow__s2 (already lowres), PC_Gen_s4/8/34 | bla_bla_chow (525 sharpness, screen) |

Production-realistic substrate fraction = **2527 frames / 4564 dev frames =
55.4%** with G1+G2 alone. Adding G3 cuts it further by ~10%.

## B) Threshold tables

Three deployment-candidate ckpts. For each, two tables: calibrated on **P2**
(production-realistic = G1+G2 applied) or on **F4** (G1+G2 + chronic-6
identity drop = ideal eval substrate). Pick the tau that matches your
risk/recall trade-off.

**How to read the realized_FPR columns:** `realized_FPR_P2_pct` is what your
production FPR will look like IF the gate is applied. `realized_FPR_F0_pct` is
what happens if NO gate is applied — i.e., the worst case if Roy_D-style or
chronic-style traffic reaches the model.

### T5C_step3500 — calibrated on P2 (production-realistic)

| target FPR | tau | F4 FPR | F0 FPR | viso recall | deeplive recall | teams_fake recall | macro recall |
|---|---|---|---|---|---|---|---|
| 0.5%  | 0.9320 |  0.00% |  0.31% |  0.0% |  0.7% | 23.6% |  8.1% |
| 1.0%  | 0.9280 |  0.00% |  0.61% |  0.0% |  3.9% | 31.0% | 11.6% |
| 2.0%  | 0.9196 |  0.05% |  1.18% |  0.0% |  7.3% | 39.8% | 15.7% |
| 5.0%  | 0.8706 |  0.19% |  3.79% |  4.2% | 37.6% | 56.9% | 32.9% |
| 7.0%  | 0.7931 |  0.72% |  6.64% | 26.9% | 73.6% | 72.7% | 57.7% |
| **10.0%** | **0.6158** | 2.06% | 11.96% | **61.1%** | **96.9%** | **88.8%** | **82.2%** |
| 15.0% | 0.4066 |  5.36% | 18.19% | 79.3% | 98.9% | 95.2% | 91.1% |

### T5C_step3500 — calibrated on F4 (chronic-6 dropped)

| target FPR | tau | P2 FPR | F0 FPR | viso | deeplive | teams_fake | macro |
|---|---|---|---|---|---|---|---|
| 2.0%  | 0.6167 |  4.74% | 11.88% | 61.1% | 96.9% | 88.8% | 82.2% |
| 3.0%  | 0.5450 |  5.66% | 14.15% | 68.7% | 98.2% | 91.8% | 86.2% |
| **5.0%** | **0.4218** | 6.96% | 17.62% | **78.0%** | **98.9%** | **94.8%** | **90.6%** |
| 7.0%  | 0.3578 |  8.55% | 20.20% | 84.0% | 98.9% | 96.4% | 93.1% |
| 10.0% | 0.2931 | 10.84% | 23.79% | 87.6% | 100.0% | 97.4% | 95.0% |

### P8A_step5000 (safer, most substrate-invariant) — calibrated on P2

| target FPR | tau | F4 FPR | F0 FPR | viso | deeplive | teams_fake | macro |
|---|---|---|---|---|---|---|---|
| 2.0%  | 0.9271 |  0.29% |  6.88% | 12.5% | 22.9% | 59.7% | 31.7% |
| 3.0%  | 0.8073 |  0.57% |  8.90% | 22.4% | 34.5% | 66.4% | 41.1% |
| **5.0%** | **0.5183** | 1.82% | 12.03% | **34.9%** | **52.1%** | **75.3%** | **54.1%** |
| 7.0%  | 0.2906 |  2.87% | 14.59% | 41.8% | 63.1% | 80.2% | 61.7% |
| 10.0% | 0.1202 |  4.93% | 18.51% | 55.5% | 75.8% | 86.2% | 72.5% |
| 15.0% | 0.0438 |  8.46% | 24.26% | 63.1% | 85.1% | 90.1% | 79.4% |

### E2B_step3200 (currently deployed) — calibrated on P2

| target FPR | tau | F4 FPR | F0 FPR | viso | deeplive | teams_fake | macro |
|---|---|---|---|---|---|---|---|
| 2.0%  | 0.8973 |  0.48% |  1.97% |  4.5% | 49.2% | 62.6% | 38.8% |
| 3.0%  | 0.8387 |  0.77% |  2.94% |  4.7% | 61.5% | 67.5% | 44.6% |
| **5.0%** | **0.7400** | 1.48% | 5.02% | **4.9%** | **76.0%** | **72.4%** | **51.1%** |
| 7.0%  | 0.6490 |  2.06% |  6.97% |  6.0% | 84.4% | 75.5% | 55.3% |
| 10.0% | 0.5054 |  3.16% | 10.01% |  8.4% | 93.9% | 79.4% | 60.6% |

### P2D_fourier_step3000 (4th candidate — better Roy_D handling) — calibrated on P2

| target FPR | tau | F4 FPR | F0 FPR | viso | deeplive | teams_fake | macro |
|---|---|---|---|---|---|---|---|
| 2.0%  | 0.8185 |  0.05% |  1.36% |  2.2% |  3.1% | 30.5% | 11.9% |
| 5.0%  | 0.4979 |  0.57% |  4.95% | 12.9% | 68.8% | 65.4% | 49.0% |
| 7.0%  | 0.4155 |  1.15% |  7.01% | 23.8% | 83.7% | 74.8% | 60.8% |
| **10.0%** | **0.3187** | 1.82% | 10.41% | **41.1%** | **91.7%** | **83.4%** | **72.1%** |
| 15.0% | 0.1802 |  4.45% | 16.63% | 72.0% | 98.2% | 93.2% | 87.8% |

### Bottom-line recommendations

| Your operating point | Pick | tau | Expected production FPR (P2) | Expected macro recall |
|---|---|---|---|---|
| **5% FPR budget, max-recall** | T5C @ F4-calibrated | 0.4218 | ~7% (P2) | **90.6%** (assuming Roy_D-style absent OR G3 sharpness gate applied) |
| **5% FPR budget, safe** | P8A @ P2 | 0.5183 | 5% (P2) | 54.1% |
| **10% FPR budget, max-recall** | T5C @ P2 | 0.6158 | 10% (P2) | **82.2%** |
| **10% FPR budget, Roy_D-robust** | P2D_fourier @ P2 | 0.3187 | 10% (P2) | 72.1% |
| **No regression risk vs deployment** | E2B (no change) | 0.7400 | 5% (P2) | 51.1% |

**My recommendation for next ship:** **T5C @ F4-calibrated tau=0.4218 with G1+G2+G3
gate** (sharpness filter at lap_var<80). The G3 gate is the price you pay for
the 36pp recall lift over P8A. If G3 is too costly to implement, fall back to
**T5C @ P2-calibrated tau=0.6158 at 10% FPR budget** (82.2% macro recall) — or
to **P8A @ tau=0.5183 at 5% FPR budget** (54.1% macro recall) if FPR is the
hard constraint.

## C) Roy_D substrate characterization — was the wildcard, now characterized

Probe ran on 12 representative Roy_D frames (spanning the full T5C score range
0.56–0.94). Computed: min(W,H), sharpness (Laplacian variance), brightness V,
saturation, face crop area.

| Identity | min_wh | sharpness | brightness | saturation | face_area |
|---|---|---|---|---|---|
| **Roy_D (this probe)** | **270** | **69** | **149** | **125** | **73,170** |
| Cam_Test__s32   | 238 |  91 | 159 |  79 |  43,167 |
| Cam_Test__s38   | 193 | 386 | 144 |  81 |  28,250 |
| PC_Generator__s22 |  88 | 484 | 116 |  55 |   2,136 |
| PC_Generator__s8 | 386 |  68 | 192 | 109 | 144,251 |
| bla_bla_chow    | 512 | 525 | 153 | 127 |  35,255 |
| bla_bla_chow__s2| 146 |  56 | 128 |  73 |  12,452 |
| dor_shkedi__s16 | 314 |  77 | 199 | 145 |  85,533 |

Roy_D is NOT extreme on resolution (270 > 200 production threshold) — it
**passes G2**. It is NOT extreme on face_area (73K is normal). It IS soft (lap=69
— softer than every chronic except PC_Gen_s8 and bla_bla_chow__s2). It IS
warm-colored (sat=125 — close to bla_bla_chow 127 and dor_shkedi 145).

This matches the existing `T3_SLOT1_step1500` viewer note: "Roy_D is the
color-axis fragility; under medium IQ gate, ~94% of Roy_D frames are
filtered. Production exposure depends on whether warm-color users appear at
sharp+large resolution." The frame-browser will let us inspect this visually.

**Production implication**: A warm-color user at 270×270+ resolution with a
slightly soft camera (e.g., a phone front-camera at any age, a slightly
out-of-focus webcam) **will look like Roy_D to T5C**. Without G3 sharpness
gating, T5C will misfire on those users at the deployment tau.

Artifacts:
- `_roy_d_probe/roy_d_iq_features.csv` — per-frame IQ for the 12 sampled
- `_roy_d_probe/roy_d_iq_summary.json` — distribution stats
- `_thresholds/DEPLOYMENT_THRESHOLDS.csv` — full table (8 FPR × 4 ckpts × 2 substrates)
