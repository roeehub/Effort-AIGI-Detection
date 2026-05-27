# AUTO_MODE_ANCHOR_REBALANCE — preview retro

> **PREVIEW** — written 2026-05-16 evening while GPU jobs are training. The
> final retro will replace this once the scorecard verdict lands.

## Purpose

Capture the full diagnostic chain (v3→v7), the two auto-mode launches, the
incident that took down Slot A v1, and the pre-scorecard predictions.

## Diagnostic chain

The session opened with the RESCHAIN_GRL6 scorecard verdict already
documented. User pushed back on the "no single property axis" framing in
RESULTS_FACTS_v2 § 3 noting "shortcuts can live in bands".

### v3 — band shortcuts found

Switched from Cohen's d (monotonic effects) to per-decile binning of Slot β
over-fire rate. 9 single-property bands each have 4-10× over-fire rate
inside the band vs outside.

Compositional dose-response: 0 bands hit → 6.9% Slot β over-fire; 9 bands
hit → 100% Slot β over-fire; 50% P8A over-fire. **All three ckpts share the
mechanism**, this is not a Slot β regression.

Decision tree on 11 properties → over-fire AUC: Slot β 0.89, T5C 0.98, P8A
0.99. P8A's tighter band (`lab_a_dev ∈ [15.77, 18.05]`) is what gives it
the lower aggregate FPR.

### v4 — OOD-band hypothesis

Sampled 470 training real frames + 422 training fake frames. Distribution:

| n_bands | train_real | train_fake | P(real \| n_bands) | test Slot β rate |
|---:|---:|---:|---:|---:|
| 0 | 72 (15%) | 37 (9%) | 0.66 | 7% |
| 5 | 21 (4.5%) | 20 (4.7%) | 0.51 | 30% |
| 6 | **0** | 3 | **0.00** | **66%** |
| 7+ | 1 | 0 | 1.00 (n=1) | 79-100% |

**Training reals have 0 anchors at n_bands ≥ 6 in property space.** Test
cohorts like Roy_D average 7 bands hit. Encoder defaults "fake" in
unanchored region.

### v5 — encoder probe (vanilla openclip)

Vanilla OpenCLIP ViT-B-16 separates Roy_D from ilan/orel at AUC 1.0,
PC1=-3.54 vs +7.18. The discrimination axis is **inherited from
pretraining**, not introduced by FT.

Tested compositional augmentation (push frames into 6+ band region via
blur + chroma shift + luma reduce + upscale). Augmented samples reach the
6+ band region but only 30% closer to Roy_D in encoder space (vs 24%
baseline). **Hand-crafted augmentation lever refuted** at the encoder
level.

### v6 — auto-mode pre-launch sanity

Pre-launch embedding sanity check (vanilla openclip) on cohorts relevant
to each proposed lever:

| cohort | Slot A target? | Slot B boost target? | % closer to Roy_D (vanilla) |
|---|---|---|---:|
| Anchor pool (dor false-flag) | ✓ | — | 16.7% |
| VCD external_real | — | ✓ | **67%** |
| Training reals teams | — | (default) | 24% |

Vanilla openclip predicted Slot B (VCD boost) as the higher-EV intervention
because VCD reals were 67% Roy_D-adjacent.

### v7 — SVD-aware probe of trained encoders

Built SVD-aware loader to load the trained T5C and Slot β backbones (596
state_dict keys, 0 missing). Re-ran the same cohort embeddings.

| encoder | Roy_D vs clean AUC | PC1 sep | across cosine |
|---|---:|---:|---:|
| vanilla openclip | 1.0000 | +10.72 | +0.74 |
| T5C step3500 | 0.9977 | +3.88 | **−0.15** |
| Slot β step3500 | 0.9846 | +3.11 | **−0.93** |

**FT amplifies the axis**. Vanilla had Roy_D and clean in the same general
direction (cosine +0.74). Slot β has them nearly antipodal (cosine -0.93).

Cohort positions in T5C trained encoder (the FT base for both A and B):

| cohort | % closer to Roy_D in T5C |
|---|---:|
| Anchor pool (Slot A) | **53.3%** |
| VCD (Slot B) | 27.0% |
| dor_shkedi lockbox PNG | 59.7% |
| real_dor lockbox PNG | 4.6% |

In Slot β trained encoder:
| cohort | % closer to Roy_D in Slot β |
|---|---:|
| Anchor pool | 13.3% |
| VCD | 18.0% |

The vanilla openclip prediction over-stated Slot B's anchoring potential.
After FT, the encoder has actively repelled VCD from Roy_D. **Slot A
mechanism is stronger in T5C's encoder than vanilla suggested** (53% Roy_D-
adjacent vs vanilla 16.7%); Slot B mechanism is **weaker** (27% vs vanilla
67%).

## Two GPU packets launched

| slot | yaml | mechanism | region | Vertex job | W&B run | seed |
|---|---|---|---|---:|---|---:|
| A v1 (FAILED) | R13_T5C_ANCHOR_AWARE_2026-05-16.yaml | training-time penalty on dor false-flag pool (30 frames, weight=5.0, target=0.10) | us-east1 | 1400679201337507840 | n73ic6ez | 9913 |
| **A v2 (relaunched)** | same yaml | same lever | **us-central1** | **8471062799328477184** | TBD | 9913 |
| B | R13_T5C_REAL_REBALANCE_2026-05-16.yaml | family weight rebalance (realpool 1.5→5.0, external 2.0→6.0, df40 0.5→0.25) | us-west4 | 3559896493831749632 | iw2kk1h0 | 9914 |

### Incident: Slot A v1 ModuleNotFoundError

Slot A v1 crashed at trainer init: `ModuleNotFoundError: No module named 'analysis.teams_pool_rescore'`.

Root cause: `.gcloudignore` excludes `analysis/*` from the Cloud Build upload. `loss/anchor_aware_penalty.py:_load_cache()` lazy-imports from that module when `anchor_aware.enabled=True`. The lazy import was never tested in CI because past packets had `anchor_aware.enabled=False`, which skips the import.

Fix: added whitelist entries in `.gcloudignore`:
```
!analysis/__init__.py
!analysis/teams_pool_rescore.py
```
Rebuilt image 1.3.292 (1m43s). Relaunched in us-central1.

### Pre-scorecard predictions

Based on v7 §9 analysis:

- **Slot A**: moderately likely to reduce dor chronic FP (53% Roy_D-adjacency of anchor pool in T5C encoder). Roy_D generalization is conditional.
- **Slot B**: weak prediction. VCD's 27% Roy_D-adjacency in T5C is modest. Boost may help marginally.
- **Compound**: not tested but mechanistically distinct, could compose.
- The encoder's axis amplification is the dominant structural force that any intervention fights.

## Open loops to be closed by scorecard

- `slot-a-anchor-aware-bounded-by-pool-content` (severity medium)
- `slot-b-real-rebalance-via-vcd-reaches-roy-d-region` (severity high)
- (revised after v7) `encoder-axis-amplification-via-ft` (severity high) — new open loop to add after scorecard

## Files

### Analysis
- `analysis/slot_b_property_shortcut_2026-05-16/RESULTS_FACTS_v3_BANDS_2026-05-16.md`
- `analysis/slot_b_property_shortcut_2026-05-16/RESULTS_FACTS_v4_OOD_BAND_2026-05-16.md`
- `analysis/slot_b_property_shortcut_2026-05-16/RESULTS_FACTS_v5_ENCODER_PROBE_2026-05-16.md`
- `analysis/slot_b_property_shortcut_2026-05-16/RESULTS_FACTS_v6_AUTO_MODE_2026-05-16.md`
- `analysis/slot_b_property_shortcut_2026-05-16/RESULTS_FACTS_v7_TRAINED_ENCODER_2026-05-16.md`
- `analysis/auto_mode_2026-05-16_eval/PENDING_SCORECARD_PLAN.md`

### Yamls
- `experiments/phase2_round13/R13_T5C_ANCHOR_AWARE_2026-05-16.yaml`
- `experiments/phase2_round13/R13_T5C_REAL_REBALANCE_2026-05-16.yaml`
- `arena/checkpoint_maps/teams_target_domain.auto_mode_2026-05-16.yaml`

### Scripts (reusable)
- `scripts/trained_encoder_probe.py` — SVD-aware encoder loader (new)
- `scripts/fill_ckpt_map.py` — post-completion ckpt map automation
- `scripts/test_compositional_aug.py` — refuted aug lever
- `scripts/band_analysis.py` — per-decile band-shortcut diagnostic
- `scripts/embed_anchor_pools.py` — anchor pool embedding diagnostic
- `scripts/embed_aug_samples.py` — augmented sample embedding diagnostic

### Infrastructure
- `.gcloudignore` whitelist fix for analysis/teams_pool_rescore.py
