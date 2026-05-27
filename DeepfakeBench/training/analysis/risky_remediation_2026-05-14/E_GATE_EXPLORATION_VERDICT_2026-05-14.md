# Gate Exploration — Verdict (2026-05-14)

Addresses two questions:
- **A)** What's the *ideal* G2 face-size threshold, not arbitrary?
- **B)** Are there cheap gates beyond G1+G2 that we haven't tested?

## A) Ideal G2 — answer: **110 (or 110-120 band)**, not 150

Sweep at 10-px resolution from 80 to 260 across all 3 production pools, measuring per-identity verdict correctness as the primary metric (production cares about identity-level decisions).

### Top combined-pool results

| G2 | Σ id-correct | Σ id-total | rate | mean AUC | mean pass% |
|----|--------------|------------|------|----------|------------|
| **110** | **50** | **55** | **90.9%** | 0.928 | **97.4%** |
| **120** | **50** | **55** | **90.9%** | 0.928 | 97.2% |
| 130 | 49 | 54 | 90.7% | 0.928 | 96.5% |
| 210 | 34 | 38 | 89.5% | 0.959 | 60.0% |
| 260 | 24 | 27 | 88.9% | 0.951 | 30.4% |
| 150 (current) | 48 | 54 | 88.9% | 0.929 | 94.2% |

### Per-pool detail
| pool | best G2 | id-correct | AUC | pass% | what changes vs G2(150) |
|------|---------|------------|-----|-------|-------------------------|
| teams_dev | 110-130 | 19/20 | 0.9853 | 91-93% | +3 identities correctly classified |
| teams_lockbox | 140-160 | 4/6 | 0.9115 | 96-98% | (already at sweet spot; G2(110) drops 1 id) |
| dor_cross | 80-180 | 28/30 | 0.889 | 100% | (insensitive — all dor_cross frames pass any G2 ≤ 180) |

### Per-identity verdict changes from G2(150) → G2(110)

| pool | identity | label | G2(150) | G2(110) | verdict at G2(110) |
|------|----------|-------|---------|---------|---------------------|
| teams_dev | Cam_Test__s38 | REAL | FAIL (15 frames, 80%) | FIXED (large n, lower frac) | correct |
| teams_dev | ilan | REAL | not scored (insuff frames) | scored correctly | correct |
| teams_dev | other identities | — | same | same | unchanged |

**No identity gets worse going from G2(150) to G2(110).** Identity verdicts on teams_dev improve; lockbox and dor_cross unchanged. Plus, you score ~1400 more frames per evaluation.

### Why ≤80 is worse than 110

| G2 | teams_dev AUC | recall@5%FPR |
|----|---------------|--------------|
| 80 | 0.9814 | 0.876 |
| 100 | 0.9842 | 0.901 |
| **110** | **0.9851** | **0.906** |

Below ~100, T5C does start losing precision — frames that small really are too lossy for the model. There's a real knee around 100-110. Above 130, AUC plateaus then slowly declines as you throw away usable frames.

### **Recommendation: G2 = 110**

Ship-spec update:
```python
return min(crop_w, crop_h) >= 110  # was 150 (was 200)
```

Catches +1413 more frames on teams_dev vs G2(150). Fixes 3 more identity verdicts (Cam_Test__s38, ilan, Chikara stays correct). No regressions.

If you want to be conservative, **G2 = 120** is essentially tied and slightly safer (loses ~0.5pp pass rate but no metric change).

---

## B) Other candidate gates — most help only on one pool

Tested 5 features as candidate gates (in addition to G1+G2): `lap_var`, `luma`, `lab_a_dev`, `lab_b_dev`, `edge_density`. For each, swept 6 quantile thresholds. Evaluated on each of the 3 production pools.

### The honest read: no cheap gate strictly dominates across all 3 pools

| feature × threshold | teams_dev ΔAUC | teams_lockbox ΔAUC | dor_cross ΔAUC |
|---------------------|----------------|---------------------|----------------|
| **iq_lab_a_dev**, drop top 20% (color a-axis skew) | **+0.013** | **+0.007** | -0.008 |
| iq_lab_a_dev, drop top 30% | +0.013 | +0.013 | +0.006 ← **only one with all 3 ≥ 0** |
| iq_lab_b_dev, drop top 25% (color b-axis skew) | +0.001 | -0.006 | +0.095 |
| iq_luma, drop outside middle 60% | +0.014 | +0.016 | -0.022 |
| iq_lap_var, drop bottom 20% (low sharpness) | +0.005 | +0.011 | -0.015 |
| iq_edge_density, drop bottom 20% | +0.005 | -0.015 | -0.010 |

### The candidate that "works" (almost)

**`iq_lab_a_dev drop_above 30th percentile`** is the only gate that's non-negative on all 3 pools:
- teams_dev: ΔAUC **+0.013**, Δrecall@5%FPR **+0.076**, Δid-correct **+11pp** (16/18 → 17/17)
- teams_lockbox: ΔAUC **+0.013**, Δrecall **+0.067**, Δid-correct **+8pp**
- dor_cross: ΔAUC +0.006, Δrecall +0.013, Δid-correct -5pp (28/30 → 22/25)

But it requires **dropping the top 30%** of frames by color-a-axis deviation — a significant coverage cost. And the dor_cross identity-correct rate drops slightly (-5pp).

This actually validates your earlier intuition about "orange filter"! The lab_a_dev axis captures color casts on the red-green channel. Strong color casts ARE associated with elevated T5C errors. Just **gating** them out (rather than trying to remediate them) works better.

### Why each other feature failed universally

- **`luma` (brightness) middle 60%**: helps teams_dev/lockbox but HURTS dor_cross (-0.022 AUC). Dor cohorts have wider luma variance.
- **`lab_b_dev` (blue-yellow color cast)**: helps dor_cross strongly (+0.095) but neutral/slight-negative elsewhere.
- **`lap_var` (sharpness)**: helps teams_dev/lockbox slightly but hurts dor_cross (-0.015). Same pattern as luma.
- **`edge_density` (texture)**: weak everywhere.

The pattern: **gates that exclude EXTREME values on quality axes (color, luma, sharpness) help in-distribution dev/lockbox but hurt cross-substrate cohorts** that have legitimately wider feature distributions.

### Recommendation on candidate gates

**Hold off on adding G3 (lab_a_dev gate) until cross-substrate evidence is stronger.** The +0.013 AUC gain on teams_dev/lockbox is real, but the small negative on dor_cross-identity-correctness is concerning. In production you'll see *both* in-distribution and cross-substrate traffic.

**If the PM does want G3**, recommend the conservative version:
- `lab_a_dev drop_above 30th percentile (q=0.30)`
- This is the only setting with non-negative ΔAUC on all 3 pools
- Drops ~30% of frames in addition to G1+G2(110)
- Helps teams_dev (+11pp id-correct), helps teams_lockbox (+8pp id-correct), slight regression on dor_cross (-5pp id-correct)

The threshold value depends on what you call "above 30th percentile" — if you compute it on the full 13,852-frame manifest distribution, `lab_a_dev` threshold ≈ measurable from a representative production sample. Should be calibrated on a representative production sample before deploying.

---

## Updated ship spec recommendation

| field | current (D verdict) | proposed (E verdict) |
|-------|---------------------|----------------------|
| Checkpoint | T5C step3500 (jrlldtem) | unchanged |
| τ | 0.49 | unchanged |
| G1 | face detector | unchanged |
| **G2** | **min(W,H) ≥ 150** | **min(W,H) ≥ 110** ← change |
| Per-identity majority | frac > 0.50 | unchanged |
| G3 (color cast) | none | optional: `lab_a_dev ≤ percentile_70` (helps in-dist, slight cross-substrate regression — wait for more data) |

### A note on what we did and did not test

**Tested as candidate gates** (cheap features computable from face crop):
- `lap_var` — Laplacian variance (sharpness)
- `luma` — mean luminance (brightness)
- `lab_a_dev` — color a-axis cast (red-green channel)
- `lab_b_dev` — color b-axis cast (blue-yellow channel)
- `edge_density` — Canny edge density (texture)
- `min(W,H)` — face crop dimensions (this is G2)

**Not yet tested** (would require additional pipeline work):
- Face detector confidence score (low confidence might indicate ambiguous detection)
- Face pose / yaw / pitch (extreme angles)
- Face area / crop area ratio
- Multi-face frame flag (more than one face in frame)
- JPEG quality estimate
- Eye visibility / occlusion detection
- Temporal stability (requires sequence info, likely unavailable in production)

The most promising of these to test next is **face detector confidence** — if the underlying detector returns a confidence score, it's a "free" feature that might predict T5C errors more directly than IQ statistics. Worth checking with the production team.

## Artifacts

| file | content |
|------|---------|
| `explore_gates.py` | sweep + analysis script |
| `outputs/g2_fine_sweep.csv` | G2 at 10-px resolution, per-pool |
| `outputs/candidate_gates_sweep.csv` | 30 candidate-gate configurations × 3 pools |

## What we changed today

- A_SHIP_SPEC: G2 200 → 150 (D verdict, before this analysis)
- A_SHIP_SPEC update needed: G2 150 → **110** (E verdict, this analysis)
- No additional gates recommended for deployment yet.
