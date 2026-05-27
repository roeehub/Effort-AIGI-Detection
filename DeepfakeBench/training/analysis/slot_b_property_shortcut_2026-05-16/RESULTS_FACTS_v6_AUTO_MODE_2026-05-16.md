# RESULTS_FACTS_v6 — auto-mode launch + embedding-space sanity check

> **FACTS only.** Two GPU packets launched 2026-05-16 evening under auto mode.
> CPU diagnostic embeds the launch-relevant cohorts to predict which packet
> has mechanistic support for fixing Roy_D-style chronic FPs.

## §1. Launches

| slot | yaml | region | Vertex job | seed | W&B project |
|---|---|---|---:|---:|---|
| A v1 (FAILED) | R13_T5C_ANCHOR_AWARE_2026-05-16.yaml | us-east1 | 1400679201337507840 | 9913 | phase2-round13 |
| **A v2 (relaunched)** | same yaml | us-central1 | **8471062799328477184** | 9913 | phase2-round13 |
| B | R13_T5C_REAL_REBALANCE_2026-05-16.yaml | us-west4 | 3559896493831749632 | 9914 | phase2-round13 |

Image v1: `effort-detector:1.3.291` (built 16:29 UTC).
Image v2: `effort-detector:1.3.292` (built 17:05 UTC).

**Slot A v1 failed at trainer init** with `ModuleNotFoundError: No module named 'analysis.teams_pool_rescore'`. Root cause: `.gcloudignore` excludes `analysis/*` from the Cloud Build upload, but `loss/anchor_aware_penalty.py:_load_cache()` imports from that module when `anchor_aware.enabled=True`. The lazy import was never tested in CI because past packets with `anchor_aware.enabled=False` skip the import entirely.

**Fix**: added whitelist entries in `.gcloudignore`:
```
!analysis/__init__.py
!analysis/teams_pool_rescore.py
```
Rebuilt as 1.3.292 (1m43s). Relaunched in us-central1 since us-east1 had a recent failure and we already had Slot B in us-west4.

Slot B (image v1) was unaffected (anchor_aware disabled → never hits the missing module).

## §2. Lever deltas (single-lever, FT from T5C step3500)

### Slot A — anchor_aware enabled

```yaml
anchor_aware:
  enabled: true
  weight: 5.0
  target_mean_prob: 0.10
  samples_per_step: 16
  pool_names:
    - "dor-real-webcam-false-flag-no-virtual-bg"   # n=30 frames
```

All other config identical to T5C base (multi_axis_grl on 4 axes, no face_scale_jitter, no fourier, no resolution_chain).

### Slot B — family_weight rebalance

```yaml
family_weights:
  df40_real: 0.25       # was 0.50
  realpool_real: 5.0    # was 1.5  (3.3× boost)
  external_real: 6.0    # was 2.0  (3.0× boost)
  # all fake weights unchanged
```

## §3. Embedding-space sanity check

Pre-launch encoder embedding sanity for cohorts relevant to each lever
(vanilla OpenCLIP ViT-B-16, 100 frames sampled per cohort, PCA fit on 194
dev PNG frames):

| cohort | n | PC1 mean | PC2 mean | Roy_D-bias | % closer to Roy_D than clean |
|---|---:|---:|---:|---:|---:|
| Roy_D (target) | 130 | **−3.535** | −0.07 | +0.221 | 100% |
| ilan/orel (clean) | 64 | +7.18 | mixed | −0.207 | 0% |
| **anchor pool falseflag** (Slot A supervision) | **30** | **+2.67** | −0.21 | −0.015 | **16.7%** |
| correct pool whiteish | 30 | +3.60 | −0.56 | −0.058 | 0% |
| **VCD external_real** (Slot B boost target) | **100** | **+1.60** | +0.67 | +0.027 | **67%** |
| training_real_teams (default lane) | 100 | +3.51 | +0.55 | −0.043 | 24% |

### Predictions per slot

**Slot A**: The anchor pool (Slot A's training-time supervision target) sits
at PC1=+2.67, near training reals (+3.51) and far from Roy_D (-3.54). The
penalty will reduce dor's chronic FP rate (direct supervision) but **is
unlikely to generalize to Roy_D's embedding region**. 16.7% of anchor pool
frames are closer to Roy_D than to clean — limited transfer capacity.

**Slot B**: VCD reals (Slot B's boosted family) sit at PC1=+1.60, MORE Roy_D-
biased than the training_real_teams baseline (+3.51). 67% of VCD frames are
closer to Roy_D than to clean. **Boosting VCD weight 3× should put more
real-label gradient signal in the Roy_D-adjacent region.**

This makes Slot B the higher-EV intervention if encoder-axis generalization
is what matters for Roy_D-style chronic FPs (per RESULTS_FACTS_v5).

## §4. Pre-defined falsifier / confirmer criteria (set BEFORE scorecard)

### Falsifiers (any one fails the packet)

- `dev_fake_macro_recall < 0.30` at all sampled steps → fake recall regresses below floor
- `lockbox_real_fpr > 0.05` (raw, no rule) at all sampled steps → over-firing got worse
- viso_enhanced_macro_dev recall drops below 0.10 at all sampled steps → viso destroyed

### Confirmation (deployment-grade)

- `dev_fake_macro_recall ≥ 0.40` at some sampled step (T5C step3500: 0.459)
- `lockbox_real_fpr ≤ 0.03` (T5C step3500: 0.028; P8A 0.018)
- viso_enhanced_macro_dev recall ≥ 0.15 (T5C step3500: 0.138; P8A 0.136)
- PLUS: re-embed Roy_D/ilan/orel cohort with the trained encoder and show
  the Roy_D ↔ clean cosine separation REDUCES vs T5C baseline (this is the
  encoder-axis test)

## §5. Open files

- `analysis/slot_b_property_shortcut_2026-05-16/outputs/all_cohorts_embedded.csv`
- `experiments/phase2_round13/R13_T5C_ANCHOR_AWARE_2026-05-16.yaml`
- `experiments/phase2_round13/R13_T5C_REAL_REBALANCE_2026-05-16.yaml`
- `arena/checkpoint_maps/teams_target_domain.auto_mode_2026-05-16.yaml` (placeholders to fill)
