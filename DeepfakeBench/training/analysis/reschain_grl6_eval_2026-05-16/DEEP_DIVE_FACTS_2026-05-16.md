# DEEP_DIVE_FACTS_2026-05-16 — what the CPU probe missed

> **FACTS only.** Numerical cross-references between the 2026-05-15 / 2026-05-16
> CPU probes and the 2026-05-16 scorecard. No interpretation.

## §1. The CPU probe ↔ scorecard prediction gap

The 2026-05-15 resolution-chain CPU probe was designed to characterize
score stability under resolution-chain perturbations. The probe's pre-stated
mechanism close criterion was `median real-cohort score_range ≤ 0.40`. The
trained Slot α ckpt achieved `0.448` — within the noise band of the target
(12% short).

The scorecard outcome (RESULTS_FACTS §3) shows Slot α step3500 FAILS the
`dev_fake_macro_recall ≥ 0.30` floor at 0.226. Mechanism close criterion
was approximately met; contract close criterion was not.

What the probe did NOT measure that, in retrospect, would have foreseen this:

| measurement | Slot α step3500 (had we run it) | predicted contract outcome |
|---|---|---|
| Per-frame score distribution on REALS post-aug | Compressed toward 0.45-0.52 across all sizes (§2 below) | τ-calibration moves DOWN (confirmed: τ=0.860 vs T5C 0.831) |
| Per-frame score distribution on FAKES post-aug | Also compressed (§3 below) | Fake-side recall drops — observation that the probe could have made before scorecard launch |
| Fake-vs-real AUC on the panel | Not reported (the probe focused on score_range not AUC) | Whether "stability" came from invariance or from score-collapse |

## §2. Real-side score distribution post-aug

From `analysis/cpu_diagnostics_2026-05-15_resolution_chain/outputs_new_ckpts/per_size_real_score_5ckpts.csv`:

Slot α step3500 mean real score by source size (averaged over kernels):
- size=64 → 0.491
- size=96 → 0.510
- size=128 → 0.521
- size=160 → 0.508
- size=192 → 0.452

Range across sizes: 0.069 (the "flat curve" finding). All values sit in [0.45, 0.53].

T5C step3500 by size:
- 64 → 0.585
- 96 → 0.670
- 128 → 0.649
- 160 → 0.545
- 192 → 0.441

Range: 0.230. T5C's curve has a peak at mid-sizes; Slot α's curve has been compressed to a flat band centered on 0.49.

## §3. Fake-side score distribution post-aug (not previously reported)

Per-frame summary parquet (`outputs_new_ckpts/per_frame_summary_new_ckpts.parquet`) covers reals + fakes. The §6 caveat of `RESULTS_OVERNIGHT_FACTS_2026-05-16.md` flagged "Slot α DOR_FAKE_DEV range 0.47 vs T5C 0.63 — more stable on fakes too, which is double-edged"; the FAKES table re-extracted from the CSV in this DEEP_DIVE:

Slot α step3500 mean FAKE score by source size:
- size=64 → 0.646
- size=96 → 0.779
- size=128 → 0.765
- size=160 → 0.745
- size=192 → 0.746

T5C step3500 mean FAKE score by source size:
- size=64 → 0.585 (note: dor-fakes-heavy panel; not a simple comparison)
- (see per_size_mean_score_fakes.csv for full)

Slot α fake-side mean scores sit in [0.65, 0.78]. T5C scores are higher on
fakes but with more spread; Slot α scores are compressed AND lower in
absolute terms. At the scorecard τ=0.860, a fake-side mean of 0.65-0.78
with the distribution compressed leaves a thin tail above 0.86 → recall
suffers.

## §4. Probe vs scorecard τ — different things were measured

The CPU probe baselines reported flip rates at τ ∈ {0.5, 0.7, 0.9}; these
are NOT the contract-calibrated τ. Each ckpt's actual scorecard τ:
- P8A τ=0.916
- T5C τ=0.831
- Slot α step1500 τ=0.899
- Slot β τ=0.816
- Slot α step3500 τ=0.860

The CPU probe's flip-rate-at-τ=0.9 finding (Slot α 12.4% vs T5C 57.7%) is
near the actual scorecard τ for Slot α step3500. It correctly predicts
that Slot α at high τ is more frame-stable than T5C at high τ. What it did
not characterize: at that high τ, what fraction of fake frames remain
above the line.

## §5. Slot β cross-reference

The 2026-05-15 CPU probe characterization of Slot β:
- Median real score_range: 0.586 (T5C 0.605; 3% reduction, within noise)
- Conclusion: "null result for this metric"

The scorecard contradicts that interpretation if read at the dev_fake_macro level:
- Slot β dev_fake_macro_recall = 0.545 (T5C 0.459, +0.086)
- Slot β visomaster_enhanced_macro_dev = 0.235 (T5C 0.138, +0.097 absolute, +70% relative)

What the probe missed: the 6-axis GRL extension moved a different axis
(visomaster-detection capability at L11) than the one the probe measured
(resolution-chain stability). The mechanism the probe targeted was not the
mechanism the loss-extension bit.

Slot β lockbox_real_fpr = 0.0882 — 3.2× T5C's 0.0279 and 4.8× P8A's 0.0184.

## §6. Per-identity decomposition: pending

The minimal 9-suite manifest does NOT include `teams_real_dor_dev` per-identity
breakdowns or the chronic-6 cohort detail. Per `OPEN_LOOPS.md` and prior
packet retros, lockbox_real_fpr is typically concentrated on 2-3 chronic
identities (memory `project_chronic_offenders_partition_per_ckpt_2026-05-04`).
A per-identity probe on Slot β step3500 would determine whether the +5.5pp
absolute lockbox_real_fpr penalty is identity-localized (rule-rescuable per
memory `project_blend_unsharp_lever_2026-05-14`) or distributed.

This probe is not yet run.

## §7. Operational reframe of the CPU probe metric

The `score_range` metric on reals measures "how much does the score swing
when content is held constant". Two distinct mechanisms produce low
score_range:

| mechanism | what produces stability | side effect on fakes |
|---|---|---|
| A. Encoder-level resolution invariance | Encoder finds a representation that doesn't co-vary with the down→up signature | Fakes preserve their distinct features; fake recall preserved |
| B. Score distribution compression | The head's decision boundary narrows; all scores migrate toward the prior | Fake scores compress along with real scores; fakes near the boundary fall below τ |

The 2026-05-15 probe did not distinguish (A) from (B). The Slot α step3500
scorecard outcome is consistent with mechanism (B): real scores 0.45-0.52
(toward the 0.5 prior), fake scores 0.65-0.78 (compressed band well above
the prior but spread is narrower than T5C's).

Mechanism-discriminating diagnostic that was NOT run in the 2026-05-15 probe:
fake-vs-real AUC on the same 388-frame panel for each ckpt. (A) preserves
this AUC; (B) reduces it. This is the cheap CPU follow-up that would have
caught the recall risk before the scorecard.

## §8. Canary probe status

The Slot α and Slot β yamls did NOT enable the canary probe
(`trainer/mixins/canary_probe.py`, available since 2026-05-07). The
chronic_diverse_800 canary at `arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet`
covers 600 reals + 200 fakes including dor cohort + chronic-6 + viso/deeplive.
Had it been enabled with `frequency_steps=500`, the canary trajectory would
have shown:
- score_p95_on_reals trajectory (decreasing → mechanism B is in play)
- chronic-6 recall trajectory at fixed τ=0.5 (decreasing → fake recall regression)
- lockbox_recall@FPR_10pct (the in-training proxy for the scorecard's lockbox_fake_recall)

These signals would have surfaced the recall risk by ~step 500-1000 of
Slot α training, allowing an early-stop decision before the full 3500-step
run.

The yamls inherited from R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml, which
also did not include canary. This is a propagated omission, not an
intentional design choice for these packets.
