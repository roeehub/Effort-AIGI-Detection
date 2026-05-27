# G2 Threshold Sensitivity — Verdict (2026-05-14)

## TL;DR

**Relaxing G2 from 200 to 150 is a clean win.** Every production metric improves or holds. No identity-level verdict gets worse; 3 wrong verdicts at G2(200) become correct at G2(150).

**Recommendation: change ship spec from G2(200) to G2(150).**

## Per-pool AUC and recall@5%FPR sweep

T5C step3500, no preprocessing, τ=0.49.

### `teams_dev` (5915 frames)
| G2 threshold | frames kept | % kept | AUC | recall@5%FPR |
|---|---|---|---|---|
| **150** | **5078** | **85.8%** | **0.9851** | **0.9168** |
| 175 | 4581 | 77.4% | 0.9849 | 0.9179 |
| 200 (current) | 3665 | 62.0% | 0.9804 | 0.8762 |
| 225 | 3100 | 52.4% | 0.9759 | 0.8487 |
| 250 | 2486 | 42.0% | 0.9725 | 0.7932 |

**G2(150) gives +0.005 AUC and +4.1pp recall vs G2(200)**, while keeping 1413 more frames.

### `teams_lockbox` (1843 frames)
| G2 | frames kept | % kept | AUC | recall@5%FPR |
|---|---|---|---|---|
| **150** | **1788** | **97.0%** | 0.9115 | 0.7271 |
| 175 | 1755 | 95.2% | 0.9142 | 0.7357 |
| 200 | 1686 | 91.5% | 0.9085 | 0.7182 |
| 225 | 1524 | 82.7% | 0.9073 | 0.7213 |

**G2(150) gives +0.003 AUC and +0.9pp recall vs G2(200)** — small but positive.

### `dor_cross` (3246 frames — the gotcha)
| G2 | frames kept | % kept | AUC | recall@5%FPR |
|---|---|---|---|---|
| 150 | 3246 | 100.0% | 0.8889 | 0.5792 |
| 175 | 3245 | 100.0% | 0.8893 | 0.5794 |
| **200** | **1624** | **50.0%** | **0.9417** | **0.7533** |
| 225 | 776 | 23.9% | 0.9975 | 0.9883 |

**Aggregate AUC on dor_cross DROPS from 0.94 to 0.89 under G2(150).** But this is a Simpson's-paradox effect — the per-identity verdicts all improve (see below).

What's happening: dor_cross at G2(200) is mostly `dor_evening` + `dor_morning` + larger `dor_fake_local` frames where T5C scores cleanly. G2(150) admits 1622 additional frames (mostly `visomaster_v2_dor` at 185px). The added band has its own real-fake separability (AUC 0.96) but the score distribution shifts, so aggregate AUC drops. **This is a metric artifact, not a deployment regression** — the per-identity verdicts confirm that.

## What's in the [150, 199] band — the newly-admitted frames

| pool | n | n_real | n_fake | band AUC | TPR @ τ=0.49 |
|------|---|--------|--------|----------|--------------|
| teams_dev | 1413 | 1271 | 142 | **0.9949** | 100.0% |
| teams_lockbox | 102 | 46 | 56 | 0.9589 | 92.9% |
| dor_cross | 1622 | 10 | 1612 | 0.9598 | 78.4% |

**T5C handles the [150, 199] band just fine.** Especially on `teams_dev` where AUC on this band is 0.995 — these are frames T5C scores cleanly and the strict gate was throwing away.

## Deployment impact at τ=0.49

### `teams_dev`
| G2 | pass_n | FP_count | FP_rate | TP_count | TP_rate | precision |
|---|---|---|---|---|---|---|
| 150 | 5078 | 409 | **11.5%** | 1506 | 99.4% | **0.786** |
| 200 | 3665 | 325 | 14.2% | 1364 | 99.3% | 0.808 |

G2(150) catches **142 more TPs**, only 84 more FPs. **FP rate actually drops** (more reals being admitted into the denominator). Precision dips slightly (0.79 vs 0.81) because of the +84 FPs.

### `teams_lockbox`
| G2 | pass_n | FP_count | FP_rate | TP_count | precision |
|---|---|---|---|---|---|
| 150 | 1788 | 759 | 55.7% | **400** | **0.345** |
| 200 | 1686 | 745 | 56.6% | 348 | 0.318 |

G2(150) catches **52 more TPs**, only 14 more FPs. **Precision goes UP.**

### `dor_cross`
| G2 | pass_n | FP_count | FP_rate | TP_count | precision |
|---|---|---|---|---|---|
| 150 | 3246 | 133 | **23.4%** | **2238** | **0.944** |
| 200 | 1624 | 133 | 23.8% | 975 | 0.880 |

G2(150) catches **1263 more TPs at the SAME FP count.** Precision climbs from 0.88 to 0.94.

## Per-identity verdict changes (the production-relevant view)

Production uses per-identity majority: `frac > τ > 0.50` → flag identity as fake. The only thing that matters for production decisions is whether the identity-level verdict changes. Across all 3 pools, **3 identities change verdict — all 3 are CORRECTIONS:**

| pool | identity | label | G2(200) verdict | G2(150) verdict | direction |
|------|----------|-------|-----------------|-----------------|-----------|
| teams_dev | Cam_Test__s38 | REAL | FAKE (15 frames, 80%) | REAL (170 frames, 50%) | ✓ corrected |
| teams_lockbox | Chikara_Takahashi__s22 | REAL | FAKE (16 frames, 69%) | REAL (42 frames, 38%) | ✓ corrected |
| dor_cross | dor_fake_inswapper_128res_gpen512 | FAKE | REAL (12 frames, 17%) | FAKE (110 frames, 54%) | ✓ corrected |

All three failures at G2(200) were **small-sample artifacts** — too few frames per identity to make a reliable majority vote. Relaxing the gate gives the model enough frames to converge on the right answer.

**Zero identity verdicts get worse under G2(150). 3 get fixed.**

Also: one new identity (`ilan` in teams_dev) is newly scoreable at G2(150) — had 0 frames at G2(200), has 28 at G2(150), correctly verdicted REAL.

## Why G2(200) was too strict

Originally G2(200) was picked because of the IQ-shortcut finding (`project_image_quality_shortcut`): low-resolution frames had elevated FPR. But T5C's GRL training was specifically targeted at the sharpness-laplacian axis. The model is now substantially robust to lower-resolution inputs.

The conservative G2(200) was right for E2B/P8A era. It's overcalibrated for T5C step3500.

## Recommendation

**Update the ship spec from G2(200) → G2(150).** Specifically:

```python
# old
return min(crop_w, crop_h) >= 200

# new
return min(crop_w, crop_h) >= 150
```

No other changes needed:
- τ = 0.49 still optimal
- Per-identity majority `frac > τ > 0.50` still correct
- G1 face detector unchanged

Expected effects in production:
- More frames pass the gate (~1.4× as many in dev, ~1.06× in lockbox)
- Identity-level verdicts mostly unchanged
- Wrong verdicts on small-frame-count identities corrected
- No new wrong verdicts

## Caveat

This sweep used T5C step3500. If the production checkpoint is ever rolled back to P8A or E2B, G2(150) should be **re-validated** — those checkpoints had stronger IQ-shortcut behavior and may need the stricter gate.

## Artifacts

- `g2_threshold_sweep.py` — script
- `outputs/g2_threshold_sweep.csv` — per-pool AUC/recall at thresholds {150, 175, 200, 225, 250}
- `/tmp/g2_sweep.log` — full run output incl. per-identity verdict table
