# Move 1 — Frozen-feature linear probe verdict (2026-04-29)

**Checkpoint:** `P8A_REFERENCE_STEP5000`
**Features:** `gs://training-job-outputs/probe_battery_2026-04-26/p8a_reference_step5000/features.npz` (3996 frames, 512-d)
**Method:** 5-fold stratified CV LogisticRegression on frozen ViT features.
**Runtime:** ~1 second total (CPU).

## Headline

**The bucket gap is NOT dispositive at the feature level.** Master-plan-expected pattern (train-bucket viso AUC > 0.80, eval-bucket viso AUC ~0.65) is **refuted**. Across 8 fake-pool × real-pool comparisons, **7/8 are inconclusive (gap < 0.05)**, 1/8 is moderate (gap 0.06). No comparison meets the dispositive threshold.

## Per-pool linear-probe AUCs

| Pool | n | P8A classifier AUC | Linear probe AUC (5-fold) |
|---|---:|---:|---:|
| **TRAIN viso** (`dl_bucket_visomaster`, fake+real same-bucket) | 256 | 0.9581 | **0.9624 ± 0.024** |
| EVAL viso clean+teams (`proper_visomaster`, no enhancers) | 1200 | 0.9968 | **0.9969 ± 0.003** |
| EVAL viso enhanced (`proper_visomaster_enhanced` + paired reals) | 1200 | 0.9985 | **0.9984 ± 0.001** |
| EVAL viso enhanced v2 (`visomaster_enhanced_v2` + tv2 reals) | 440 | 0.9679 | **0.9710 ± 0.008** |
| EVAL viso tv2 (Teams-v2 transport, same-bucket reals) | 213 | 0.9228 | **0.9240 ± 0.031** |
| DEPLOY train viso vs VCD webcam | 384 | 0.9724 | 0.9833 ± 0.010 |
| DEPLOY eval clean+teams vs VCD webcam | 856 | 0.9911 | 0.9951 ± 0.003 |
| DEPLOY eval enhanced vs VCD webcam | 856 | 0.9946 | 0.9949 ± 0.004 |
| DEPLOY eval enhanced v2 vs VCD webcam | 556 | 0.9568 | 0.9764 ± 0.010 |
| DEPLOY eval tv2 vs VCD webcam | 329 | 0.8806 | **0.9204 ± 0.047** |

The deployment-real-pool variant (each fake pool paired with `external_vcd_real`, the production-realistic webcam pool) shows the same pattern: features separate fake-vs-real well on every viso bucket variant.

## Reconciling with master plan's "frame-level AUC 0.7527 on `visomaster_enhanced_macro_dev`"

The master plan's 0.7527 is computed on the *actual* dev eval suite, which uses a real pool not present in this features.npz cache. The cached real sources (`external_vcd_real`, `proper_real_*__paired`, `tv2_*_real`, etc.) all have **mean prob_fake ≤ 0.08** — P8A does not flag them. The 0.7527 number therefore likely reflects:

1. A real pool that triggers the camera-signature shortcut (lockbox-style frames where P8A produces high prob_fake on REAL frames, lowering AUC), OR
2. Specific harder slices in the eval substrate not captured by the probe-battery sampling.

**Either way, the failure mode is on the REAL side (FPR-driven), not on bucket discriminability of the fake side.** P8A's features rank-order viso fake-vs-real correctly across every bucket variant tested.

## Per-source mean prob_fake (illustrative)

| Source | n | mean prob_fake | p>0.5 rate |
|---|---:|---:|---:|
| dl_bucket_visomaster_fake (TRAIN) | 128 | 0.870 | 87.5% |
| proper_visomaster_enhanced_clean_fake | 300 | 0.980 | 99.0% |
| proper_visomaster_enhanced_teams_fake | 300 | 0.921 | 93.7% |
| visomaster_enhanced_v2_fake | 300 | 0.806 | 82.3% |
| **tv2_visomaster_fake** (Teams-v2 transport) | **73** | **0.437** | **46.6%** |
| All real sources | 1810 | 0.020–0.080 | 0.7%–14.1% |

**Notable exception:** `tv2_visomaster_fake` mean prob_fake = 0.44, only 47% above 0.5 — there is a genuine fake-side miss on Teams-v2 transport viso. But its sample size in the cache (73 frames) is small, AND the AUC against same-substrate reals is still 0.92 (rank-order is fine, threshold crossing is the issue).

## Implication for tonight's slate

| Slot | Original recommendation | Move 1 implication |
|---|---|---|
| 1 | **P14_DATA_FIX** (Hybrid Path-B + Path-A, ~$80) | **DOWNGRADE.** Bucket gap as framed is not the dominant feature-level signal. The predicted ~24% recall lift from substrate matching is unlikely to materialize at that magnitude. The genuine `tv2_visomaster` slice miss is too small to motivate the full $80 retrain. |
| 2 | **P15 GRL** on quality_domain (~$70) | **HOLD priority.** Layer-3 attack on the shortcut is now even more clearly the dominant lever — the FPR/recall problem is shortcut-on-reals, not bucket-gap-on-fakes. |
| 3 | **P8A + face-scale-jitter** (~$70) | **HOLD priority.** Face-size leak r=−0.565/−0.529 is direct shortcut evidence; isolating jitter is consistent with Move 1's Layer-3 reframing. |

## Recommendation

**Recommended slate (~$140, two slots):**
1. P15 GRL on quality_domain (existing YAML, ~$70, ~8h)
2. P8A + face-scale-jitter isolated (needs YAML draft + train_sweep allowlist update, ~$70, ~8h)

Skip P14_DATA_FIX tonight. If P15/face-scale-jitter still leave the eval-suite AUC at 0.75 territory, the right next move is to **investigate the actual real pool used by `visomaster_enhanced_macro_dev`** to characterize what's driving the 0.7527 — not retrain on more fakes.

If the user wants to keep some bucket-gap-related insurance: a much narrower P14_DATA_FIX scoped only to `tv2_visomaster` family (where the genuine fake-side miss exists) is defensible at ~$70 — but the ~24% lift expectation from the master plan should be revised down materially.

## Caveats

- Probe-battery sampling (`n_per_source≈128–300`) is smaller than the full eval suite. Pool counts for `tv2_visomaster_fake` (n=73) are particularly thin.
- Real pools available in this cache do not include `lockbox` or whatever the actual `visomaster_enhanced_macro_dev` real pool is. To fully reconcile the 0.7527 number, repeat this probe with the eval-suite-equivalent real pool extracted explicitly (~hours of additional work, optional).
- This is a frozen-feature read of P8A. P14 (FT-from-P8A) features may differ slightly; for the Layer-2 question this is the right checkpoint to probe.

## Files

- `run_move1_probe.py` — the script.
- `move1_results.json` — full per-pool stats, fold AUCs, verdicts.
- `move1_results.csv` — flat summary table.
- `p8a_features.npz` — local cache of the GCS features file.
- `p8a_sampling_manifest.json` — manifest of which URIs were sampled.
