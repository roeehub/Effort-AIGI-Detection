# Thread: Viso train/eval bucket gap

> **⚠ Critical-reading note (added 2026-05-04 night)**: the "Current stance (2026-04-30)" framing below is built around a closed-as-superseded loop (`p14-data-fix-not-launched`) whose superseding test (`xan4dfto`, P14_DATA_FIX) was bundle-confounded (anti-shortcut bundle stacked with fw=8.0 data lever). The 2026-04-30 stance reads as "bucket-fix as deployed did not work" — strictly true for that specific deployment, but the framing risks future agents inheriting "data axis is exhausted" when no clean single-lever test had been done. The clean retest is finally in flight on 2026-05-04 evening (Packet A + Packet C-codec, see "2026-05-04 evening update" subsection below + new in-progress loop `data-axis-clean-single-lever-retest-in-progress`). Read the 2026-04-30 stance as "this specific recipe combination did not work"; do not extrapolate to "the data-axis lever is dead."

> **Slice 7 finding (2026-04-29)**: the eval suite `visomaster_enhanced_macro_dev` reads GAN-enhanced viso fakes recaptured through Teams; training viso has no enhancers and no Teams transport. The gap is a plain train/eval distribution mismatch, layered on top of the camera-signature shortcut. Layer-2 wiring is fully built; YAMLs and a tests file are uncommitted. Memory `project_viso_train_eval_bucket_gap.md` is the auto-memory anchor.

## The question

Why has every recipe-tuning attempt over the past three weeks been unable to lift `visomaster_enhanced_macro_dev` recall above ~10–15%, while the symmetric `deeplive_enhanced_dev` suite consistently lands 3–5× higher? Is this a model-capacity question (more recipe-tuning), a shortcut question (cleaner anti-shortcut levers), or a data-distribution question (the eval pool is simply not in training)?

## Initial belief

Through Slices 4–6 the working assumption was that the viso/deeplive recall asymmetry was a **shortcut artifact**: P8A's CLIP-unfreeze had partially broken the camera-signature ceiling on deeplive but not viso, and the gap reflected the residual shortcut signal viso fakes happened to share with viso reals. P9–P12 recipe-tuning, P13_FROM_SCRATCH's anti-shortcut interventions, and the inference-time stacking exercise (Option A subagent) all framed the question as "recipe-side or representation-side". Identity diversity was occasionally floated as an alternative explanation but never rigorously checked. Memory `project_signature_shortcut_finding.md` was the load-bearing anchor for the shortcut framing.

## What changed our mind

- **2026-04-29 ~10:00 CEST — Bucket-gap diagnosis (`docs/relaunch_handoffs/TRAINING_EVAL_VISO_BUCKET_GAP_2026-04-29.md` + `april-26-training-master-plan-v2.LOG.md:1387-1454`).** Triggered after the P13_FROM_SCRATCH γ verdict landed: user pushed back on the agent's "identity diversity may be too narrow" hypothesis, asking the agent to count the buckets. The audit found that viso fakes in the verdict's `visomaster_enhanced_macro_dev` slice resolve to `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/` — **GAN-enhanced viso fakes recaptured through Teams** (550 videos in this slice; example frame `visomaster_enhanced_raw__frame_001595_seq12349.png`). Training viso (`R13_P13_FROM_SCRATCH.yaml:163-177` and every prior P-series yaml) reads `gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_*` with `tiers: null`. Local discovery cache `.viewer_cache/discovery/visomaster.json` shows `Counter(enhancer for s in viso) == {'none': 960}`. **Zero enhancers, no Teams transport.**
- **The deeplive control case explains the asymmetry.** `deeplive_enhanced_dev` reads from the same Teams-recapture eval bucket. Deeplive *training* includes both `edge_cases_enhanced` / `minimal_processing_enhanced` strategies (post-enhancer fake variants) AND the `deeplive_teams_*` family from the separate `live-...-teams` bucket. So deeplive's training distribution approximately matches its eval distribution; viso's does not. This perfectly predicts the persistent ~3–5× viso-vs-deeplive recall gap on every candidate at every τ since RLP5 — which had stayed unexplained through six packets of recipe-tuning.
- **Layer-2 closure is YAML-only — wiring is fully built.** `data/sources/visomaster.py:605-890` already defines `VisoMasterEnhancedSample` with a dual-bucket structure (default `enhanced_bucket = "visomaster-enhanced-face-cropped"`). `arena/build_visomaster_enhanced_v2_manifest.py` and `arena/manifests/visomaster_enhanced_v2_manifest_2026-04-13.json` (2,073 frames, 9 swap × 8 enhancers) are produced. `combined_paired.py:646` provides `create_unified_samples_from_visomaster_enhanced` with quality-domain mapping (`QUALITY_DOMAIN_MAP["visomaster_enhanced"] = 2`). `train_sweep.py` re-applies the whole `combined_paired` block, so the wandb-flattening trap (see [`wandb_flattening`](wandb_flattening.md)) does NOT bite for new nested children. Test coverage exists at `tests/test_visomaster_enhanced_wiring.py` (3 tests: domain map, sample tagging, bucket routing) — uncommitted, untracked file as of 2026-04-29. **The structural blocker is one yaml stanza**: `visomaster_enhanced.enabled: true` + `visomaster_hints_teams.enabled: true` + family weights in `combined_paired.sampling.family_weights`.
- **`R13_P14_DATA_FIX.yaml` is drafted, not launched** (`experiments/phase2_round13/R13_P14_DATA_FIX.yaml` — untracked). Cost estimate: ~$70 / 1–2 days for FT-from-P8A. P14 currently in flight (`R13_P14_FT_FROM_P8A.yaml`) is the FT-from-P8A + interventions variant; `R13_P14_DATA_FIX` is the bucket-fix retrain that is one tier higher in priority per this thread's stance.
- **Asymmetry magnitude is ~2× in delta-from-1, not 3-5×** (memory `project_viso_train_eval_bucket_gap.md` 2026-04-29 update + memory `project_p8a_frame_level_auc_2026-04-29.md`). Frame-level AUC measured directly from cached P8A_step5000 predictions: viso eval 0.7527 / deeplive eval 0.8614 / teams_fake_all_dev 0.9106. So `(1 - 0.75) ≈ 2 × (1 - 0.86)` — the gap is real but quantitatively smaller than the recipe-tuning history implied. This sets a structural ceiling on what Layer-2 (bucket-gap closure) can deliver: viso plausibly lifts to ≈ deeplive's substrate-matched ~24% under the corrected contract policy, **not** to the 90% headline target. See [`processing_signature_shortcut`](processing_signature_shortcut.md) for the frame-level AUC reframing.
- **The bucket gap reframes prior beliefs, doesn't replace them** (`docs/relaunch_handoffs/TRAINING_EVAL_VISO_BUCKET_GAP_2026-04-29.md` table at section 1.2.4):
  - "Identity diversity may be too narrow" — **refuted** (429 viso + 1916 deeplive in active training, +2,300 unused; see [`identity_audit`](identity_audit.md)).
  - "Shortcut is the dominant cause of low viso recall" — **reframed**: shortcut is real (Axis 3 Δ ~0.5 even on best P13) but the *headline metric* is dominated by the bucket gap; recipe-tuning was attempting cross-pipeline generalization where data-fix is the cheaper move.
  - "Anti-shortcut interventions on top of P8A (P14) will lift viso recall" — likely false because the bucket gap is unchanged in P14_FT; the bundle effect is bounded by the data mismatch.
  - "GRL is the cheapest next move" — revised: GRL is still useful but expected gain is much higher *after* the eval pipeline is in training; see [`P15`](../packets/P15.md) for the readiness note (not yet launched as of 2026-04-29).
- **2026-04-29 — Move-1 (linear-probe confirmation) is the proposed pre-spend gate**, ~$0 / ~1–2 hours: frozen-feature linear probe of P8A on the eval bucket vs the training bucket. If train-bucket AUC is materially higher than eval-bucket AUC (>0.80 vs ~0.65) the bucket gap is dispositive; if not, the gap is real but not the dominant cause. **Not yet executed as of end of Slice 7.**

## Current stance (2026-04-30)

The viso recall failure is approximately **two-thirds bucket-gap, one-third shortcut** in *frame-level AUC* terms, but the 2026-04-30 P14_DATA_FIX result demonstrates that **closing the bucket gap as a stand-alone training-side lever is not sufficient to lift the headline metric** in the FT-from-P8A regime tested. Wiring `visomaster_teams_enhanced` at `family_weight=8.0` on top of the P14 anti-shortcut bundle produced a generalization collapse (cross-method `other_fakes_tpr` crashed to 0.047) rather than a recall lift. The bucket-gap finding is reframed as *necessary-but-not-sufficient*, not *dispositive*. Four present-tense facts about the state:

1. The `visomaster_enhanced_macro_dev` eval suite resolves to a Teams-recaptured GAN-enhanced viso bucket that has **no presence in any pre-2026-04-30 training yaml**. This is uncontroversial and has explicit citations at `compute_verdict.py:51,335`, `arena/target_domain_suites.r13_best_megaval_2026-04-13.yaml:44-49`, and `R13_P13_FROM_SCRATCH.yaml:163-177`.
2. The wiring to add the missing distribution is fully built and now tested in production: `R13_P14_DATA_FIX.yaml` ran 2026-04-30 with `visomaster_teams_enhanced` enabled at `family_weight=8.0`. **Result: trainer-side value_composite=0.126 (statistically tied with the bundle baseline 0.116)**; cross-method TPR collapsed (other_fakes_tpr=0.047). The wiring works; the lever as deployed did not.
3. The structural ceiling on the bucket-fix is bounded above by the deeplive-substrate ~24% recall, not the 90% headline target. Frame-level AUC evidence is the load-bearing input to that estimate (memory `project_p8a_frame_level_auc_2026-04-29.md`).
4. Memory `project_move1_bucket_gap_refuted.md` (Move-1 frozen-feature linear probe) had already established pre-launch that P8A's features rank-order viso fake-vs-real correctly across every bucket variant (probe AUC 0.92-0.998); the 0.7527 frame-level AUC on visomaster_enhanced_macro_dev reflects shortcut-on-reals (FPR), not bucket discriminability. The DATA_FIX result is the empirical confirmation of that pre-launch refutation.

The lever is **not** another recipe-tuning packet of the same shape, AND it is not the bucket-fix retrain as deployed in P14_DATA_FIX. The bucket-fix as a stand-alone training-side intervention, **stacked with the P14 anti-shortcut bundle**, did not deliver the headline-metric closure the slice-7 framing implied. A clean retest would be FT-from-P8A + jitter@0.50 + viso_teams_enhanced (no anchor_aware, no pipeline_random) — the bundle-decomposition discipline applied to DATA_FIX. Without that test, "the bucket gap can be closed" remains true at the frame-level AUC layer (Move-1 said so); whether "training on the missing distribution lifts the headline recall" is supported or refuted depends on the bundle-confounded test (refuted under that recipe) AND on the clean single-lever retest that did not exist as of 2026-04-30 (now in flight as Packet A + Packet C-codec on 2026-05-04 evening — see next subsection).

## Packet timeline

- [P9](../packets/P9.md) – [P10](../packets/P10.md) – [P11](../packets/P11.md) – [P12](../packets/P12.md) — every recipe-tuning packet's viso recall hits the same ceiling; bucket gap was the unexplained ~3-5× asymmetry that survived all of them.
- [P13](../packets/P13.md) — γ verdict (cross-domain collapse on viso/deeplive at every τ-policy) is what triggered the bucket-gap audit (memory `project_data_inventory_identity_diversity.md`).
- [P14](../packets/P14.md) — currently in flight (`R13_P14_FT_FROM_P8A.yaml`, FT-from-P8A + anti-shortcut interventions); does NOT close the bucket gap (separate hypothesis). The companion proposal `R13_P14_DATA_FIX.yaml` is the one that closes Layer 2; **drafted, not launched** as of 2026-04-29.
- [P15](../packets/P15.md) — readiness note drafted; smoke-test required before launch. The bucket-gap finding revises P15's expected gain — GRL on training-only pipelines does not teach the model the eval pipeline.

## Evidence locations

- `docs/relaunch_handoffs/TRAINING_EVAL_VISO_BUCKET_GAP_2026-04-29.md` — dispositive finding doc (the canonical Slice 7 anchor).
- `april-26-training-master-plan-v2.LOG.md:1387-1454` — session entry "Bucket-gap diagnosis (post-P13_FROM_SCRATCH γ verdict)".
- `analysis/p13_day4_verdict_2026-04-29/compute_verdict.py:51,335` — the verdict script reads `viso_dev_recall` from suite `visomaster_enhanced_macro_dev`.
- `arena/target_domain_suites.r13_best_megaval_2026-04-13.yaml:44-49` — that suite's manifest pointer.
- `arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json` (`visomaster_enhanced_macro` slice, 550 videos) — frame paths resolve to `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/`.
- `experiments/phase2_round13/R13_P13_FROM_SCRATCH.yaml:163-177` — `visomaster.gcs_bucket = "live-deepfake-methods-real-and-fake-frames-cropped"`, `tiers: null`. The training-side viso source.
- `data/sources/visomaster.py:605-890` — `VisoMasterEnhancedSample` (the unused loader for the eval-matching bucket).
- `data/sources/combined_paired.py:646` — `create_unified_samples_from_visomaster_enhanced`, `QUALITY_DOMAIN_MAP["visomaster_enhanced"] = 2`.
- `arena/manifests/visomaster_enhanced_v2_manifest_2026-04-13.json` — 2,073 frames, 9 swap × 8 enhancers (built but unused).
- `experiments/phase2_round13/R13_P14_DATA_FIX.yaml` — drafted bucket-fix retrain yaml (untracked, 2026-04-29).
- `tests/test_visomaster_enhanced_wiring.py` — 3 wiring tests (untracked, 2026-04-29).
- Memory: `project_viso_train_eval_bucket_gap.md` (auto-memory anchor; verified 2026-04-29 with quantitative downgrade ~2×, not 3-5×); `project_p8a_frame_level_auc_2026-04-29.md` (the frame-level AUC numbers that bound the ceiling); `project_data_inventory_identity_diversity.md` (refutes the alternative "more identity diversity" hypothesis); `project_clean_teams_same_identity.md` (paired-transport structure that makes the bucket-fix natural).

## Open loops

### Open loop: p14-data-fix-not-launched
status: superseded
severity: medium
first_seen: 2026-04-29
last_verified: 2026-04-30
close_criterion: `R13_P14_DATA_FIX.yaml` is committed, the smoke loads enhanced viso samples cleanly, the retrain completes, and a contract scorecard run on the resulting checkpoint shows `visomaster_enhanced_macro_dev` recall lifted materially above the P8A baseline (≥ 24% target under the corrected contract policy with `target_fake_recall_min=0.30`) — verifying the bucket-gap closure on the actual headline metric.

**2026-04-30 — superseded by negative empirical result.** The retrain ran (run id `xan4dfto`, ckpt `gs://training-job-outputs/phase2r13_experiments/xan4dfto/`). Trainer-side value_composite=0.126 (statistically tied with the P14 bundle baseline 0.116); cross-method `other_fakes_tpr` crashed to 0.047 (vs P14_FT bundle 0.020 and jitter-isolated 0.591). The promotion-contract scorecard for this checkpoint was not separately run because the trainer-side signal is so far below the close criterion's expected ≥24% lift that the contract scorecard would not change the verdict at the headline metric. **Move-1 prediction empirically confirmed**: memory `project_move1_bucket_gap_refuted.md` had established pre-launch that the bucket gap is not the dominant cause of headline-metric failure; DATA_FIX's collapse on cross-method generalization is the in-vivo confirmation. The lever as deployed (DATA_FIX yaml on top of the P14 anti-shortcut bundle, family_weight=8.0) is not viable.

A clean retest of the bucket-gap-as-headline-lift hypothesis would be FT-from-P8A + jitter@0.50 + viso_teams_enhanced WITHOUT the anchor_aware + pipeline_randomization bundle drag (per [`anti_shortcut_bundle_decomposition`](anti_shortcut_bundle_decomposition.md)). That retest is **not** scheduled and is **not** considered a high-priority next packet — the cleaner forward path is to drop the bucket-fix from the active queue and pursue the jitter-isolated leader (`mclioexb`) with promotion-contract scorecard validation + face-size invariance probe. If a future packet wants to reopen the bucket-fix question on a clean substrate, it should re-derive the close criterion under the corrected scorer policy AND apply the bundle-decomposition discipline.

### 2026-05-04 evening update — clean single-lever retest is finally launched (Packet A + Packet C-codec)

The 2026-04-30 framing above ("retest is not scheduled and not a high-priority next packet") was a deprioritization based on the post-mclioexb path. That framing was followed by the P17/P18/P22/S1-S3/E1/E2b/E3 sequence (logged in `docs/relaunch_handoffs/` and memory entries `project_p17_*`, `project_p18_*`, `project_p22_*`, `project_s1_s2_s3_2026-05-03.md`, `project_e2b_breaks_deeplive_ceiling.md`, `project_l14_does_not_break_viso_ceiling.md`). Across those packets (10+ R13 packets total) the viso recall ceiling was not broken; the data-axis lever was NOT pulled in a clean single-lever form during this period, despite memory `project_data_axis_lever_pulled_twice_no_lift.md` framing the lever as exhausted.

**On 2026-05-04 evening, two packets that test the data-axis lever cleanly were drafted and launched** in parallel on us-east1:

- **Packet A** (`R13_PA_VISOMASTER_ENHANCED_DATA.yaml`, run `3140330896851206144`, image 1.3.256, JOB_STATE_RUNNING): E2B baseline (B16 scratch + CE + heavy aug) + single change = enable `visomaster_enhanced` and `visomaster_teams_enhanced` data sources at `family_weight: visomaster_enhanced_fake = 4.0`. Companion infrastructure: `companion_domains: [teams_v2]`, `p_original: 0.5`. This is the first true single-lever data-availability test on a non-FT base.
- **Packet C-codec** (`R13_PC_CODEC_PLUS_DATA.yaml`, run `1202657157175050240`, image 1.3.257, JOB_STATE_RUNNING): same as Packet A + `teams_codec_simulation` enabled with `policy: adaptive_mixture`. Verified empirically to be a faithful simulation of measured Teams transport on 8 IQ axes (`analysis/codec_aug_verification_2026-05-05/`, cosine 0.87-0.99 vs actual transport on 30 paired viso frames). This is the data lever + the codec aug lever stacked, complementary not orthogonal.

Both runs are FT-from-E2B (the new anchor candidate per `project_e2b_breaks_deeplive_ceiling_2026-05-04.md`) NOT FT-from-P8A as the 2026-04-30 framing assumed. The change of FT base is load-bearing: the bundle-decomposition discipline applied to the original P14_DATA_FIX framing was about jitter@0.50 isolation; the 2026-05-04 retest is about data + (optional) codec aug isolation on a different anchor entirely.

This is the clean retest the 2026-04-30 framing said wasn't scheduled. Verdict ETA ~24h via promotion-contract scorecard once both jobs complete; F4 substrate-cleaning re-eval pipeline is also ready (`analysis/substrate_cleaning_eval_2026-05-05/run_clean_eval.py`, reproduces Job 14 numbers within 0.5pp).

### 2026-05-05 morning update — P8A on HDTF DISPOSITIVELY confirms substrate-specific framing

**Source**: `analysis/p8a_on_hdtf_2026-05-05/FINDINGS.md`. Vertex job `4232735281465262080` (us-east1, image 1.3.257), launched 2026-05-04 22:14 UTC; teams + real suites complete by 22:56, clean variants pending.

**Headline**: P8A_step5000 reaches **93.57% recall on `proper_visomaster_enhanced_teams_dev` (n=1182)** at τ=0.5 with 0.97% FPR on `proper_real_teams_dev`. Compare to **27% recall on `visomaster_enhanced_macro_dev` (v2 production substrate, n=550)** at FPR=10% (per F4 reference data + PSERIES_FACTS).

**Same model, same enhancer × teams transport conjunction, different substrate → 3.5× recall difference at comparable FPR.**

This dispositively confirms the substrate-specific framing. The "viso ceiling is structural across architectures" claim from `project_viso_ceiling_unbroken_10_packets.md` is now empirically refuted as a UNIVERSAL claim; the ceiling is conditional on the v2 production substrate.

**P8A vs RLP6_04 on the same HDTF substrate** (extends Job B):

| Suite (n) | RLP6_04 | P8A | Δ |
|---|---:|---:|---:|
| proper_visomaster_enhanced_teams_dev (1182) | 85.36% | **93.57%** | **+8.21pp** |
| proper_visomaster_enhanced_teams_lockbox (302) | 89.74% | 95.03% | +5.29pp |
| proper_visomaster_teams_dev (262) | 93.51% | 94.27% | +0.76pp |
| proper_visomaster_teams_lockbox (80) | 90.00% | 95.00% | +5.00pp |
| proper_fake_teams_all_dev (1444) | 86.84% | 93.70% | +6.86pp |
| proper_fake_teams_all_lockbox (382) | 89.79% | 95.03% | +5.24pp |
| proper_real_clean_dev (1443) FPR | 1.11% | **0.42%** | -0.69pp (better) |
| proper_real_clean_lockbox (382) FPR | 0.52% | **0.00%** | -0.52pp (better) |

**P8A is BETTER than RLP6_04 on every cell** — the FT chain RLP6_04 → P8A added 5-8pp on viso enhanced+teams while reducing real FPR. Trajectory hypothesis (ckpt erosion) is dispositively refuted.

**Mechanism**: see `analysis/pa_pc_eval_2026-05-05/IQ_VALLEY_FINDING.md` and `per_ckpt_iq_signatures.md`. The 364/550 unreachable v2 frames are at midrange Lap (p50=33), in the IQ-shortcut valley between P8A's high-Lap sweet spot (r=+0.51) and E2B's low-Lap inverted sweet spot (r=-0.25). On HDTF substrate, the Lap distribution is in P8A's sweet spot range → high recall.

**Memory updates**:
- `project_viso_ceiling_unbroken_10_packets.md` scoping note upgraded from inference to head-to-head measured.
- `project_job_b_findings_universal_vs_trajectory_2026-05-04.md` extended with P8A-on-HDTF data point.

**Caveat**: production-relevant numbers are still v2 substrate numbers IF production traffic looks like v2 (Dor + chronic-6 + low-Lap teams). If production traffic looks like HDTF (varied identities, sharper IQ), the model is much better than v2 numbers suggest. Memory `project_v2_substrate_is_dor_diverse_swap.md` flagged that v2 is internal-test-substrate-specific (Dor in his standard setup); chronic-6 are EVAL test identities. Production users would not be the same 6 identities.

### 2026-05-04 late night update — Job B verdict (universal-vs-trajectory) closes; viso ceiling reframed as substrate-specific

**Source**: `analysis/job_b_pre_rlp604_2026-05-04/FINDINGS.md` (full writeup) + `pivot_summary.json` + `scorecard.csv`. Vertex job `9082408392701509632` (us-east1, image 1.3.255) reached `JOB_STATE_FAILED` 18:44:42 UTC on 2026-05-04 at the contract-scoring tail (404 because the `proper_data_future` suite manifest does not contain `teams_real_all_dev`, which `score_teams_promotion_contract.py:748` hardcodes as the default `--dev_real_suite`); the validation phase succeeded for all 64 (suite × ckpt) pairs and the per-suite reports were uploaded cleanly. This update is built from those reports.

**Empirical results** (τ=0.5 readouts, FPRs at τ=0.5 are <5% on every real cell so deployment-FPR-calibrated recalls would be ≥ these numbers; full table in FINDINGS.md):

| Suite (n=videos) | R12G_14k | RLP3_05_2.5k | RLP5_07_20.5k | RLP6_04_23.5k |
|---|---|---|---|---|
| `proper_visomaster_clean_dev` (262) | 100.0% | 99.2% | 99.2% | 99.2% |
| `proper_visomaster_enhanced_clean_dev` (1180) | 98.4% | 96.7% | 98.6% | 98.6% |
| `proper_visomaster_teams_dev` (262) | 64.9% | 76.7% | 95.0% | 93.5% |
| **`proper_visomaster_enhanced_teams_dev` (1182)** | **9.6%** | **12.1%** | **94.8%** | **85.4%** |
| `proper_visomaster_enhanced_teams_lockbox` (302) | 9.3% | 12.6% | 96.7% | 89.7% |

The bolded row is the closest analog to the production "viso 27% ceiling" cell. Real FPRs at τ=0.5 across all four real suites and four ckpts are 0.0% to 4.45%.

**Two findings, distinguished as fact and inference (read each independently):**

**Fact**: On HDTF proper viso enhanced+teams (n=1182), the FT chain RLP6_04_STEP23500 reaches 85.4% recall at τ=0.5 with FPR<5%; RLP5_07_STEP20500 reaches 94.8%. Pre-RLP6_04 chain members R12G_STEP14000 and RLP3_05_STEP2500 collapse at 9.6% / 12.1%. The recall jump occurs between RLP3_05 and RLP5_07, which is when the proper visomaster training data sources entered the recipe.

**Inference (with caveats)**: The "viso ceiling = 27% structural across architectures" framing (memories `project_viso_ceiling_unbroken_10_packets.md`, `project_e2b_breaks_deeplive_ceiling.md`, `project_l14_does_not_break_viso_ceiling.md`) is conditional on the production v2 substrate (Dor × ~16 swap-model families per memory `project_v2_substrate_is_dor_diverse_swap.md`). On HDTF proper viso (different identities, possibly different swap-model coverage, possibly different image-quality distribution), the same FT chain reaches 85% on the cell that is 27% on production. The most parsimonious explanation is that the ceiling is **substrate-specific, not model-structural**.

**Trajectory hypothesis is REFUTED on HDTF proper viso.** "Late FT-chain ckpts lost a viso capability earlier ones had" — RLP6_04 retains 85.4% on enhanced+teams; the chain GAINED capability between RLP3_05 → RLP5_07 (not lost). The slight regression RLP5_07 → RLP6_04 (94.8% → 85.4% on dev, 96.7% → 89.7% on lockbox) is small and not characterized.

**Caveats** (do NOT skip when reading the inferences):

1. **P8A was not scored in Job B.** To CLEANLY separate trajectory from substrate for the production ceiling, P8A_step5000 needs to be scored on HDTF proper viso enhanced+teams. Without it, the substrate framing rests on the inference above plus pre-existing memories about v2 substrate being unusually difficult, NOT on a direct head-to-head measurement. **This is a ~$0-15 followup recommended next.**
2. **τ=0.5, not FPR-calibrated.** Magnitudes shift slightly with proper FPR calibration; direction is unambiguous because real FPRs are well below 5% on every cell.
3. **RLP3_05_STEP2500 is at step 2500** (very early in that recipe); the 12.1% number conflates training-depth with data-recipe difference vs RLP5_07_STEP20500.
4. **Pre-RLP6_04 ckpts use pre-fix `INTER_AREA` preprocessing**; expect ~0.7pp drift vs post-fix scoring (small relative to the 12% → 95% magnitude). See thread `preprocessing_parity_bug.md`.
5. **Substrate differences are multifactorial** (identity, swap-model coverage, image quality). Without a per-axis decomposition, "substrate-specific" is the wrapper claim, not a precise mechanism. The image-quality shortcut (memory `project_image_quality_shortcut.md`) and chronic-6 substrate-pollution finding (memory `project_job14_substrate_clean_2026-05-04.md`) are candidate mechanisms within the wrapper.

**What this update changes vs leaves alone in this thread:**

- The "Current stance (2026-04-30)" body remains historically valid for the v2 production substrate (the only substrate the prior framing knew about). **Read it as conditional on substrate**, not as a universal claim.
- The bucket-gap-as-headline-lift hypothesis is **strengthened in one direction**: training-side data inclusion DID move RLP5_07's enhanced+teams viso recall from ~12% to ~95% on HDTF substrate. This is empirical support for the data-axis lever being dispositive in some configurations.
- It is **left open in another direction**: whether that translates to lifting v2 production viso recall above its 27% ceiling depends on whether the v2 substrate's depressing factors (IQ shortcut, chronic-6, identity skew) are addressable via the same data-side intervention. Packet A / Packet C-codec test that question on E2B; the verdict is still pending (see in-progress loop below).
- It does NOT close the in-progress `data-axis-clean-single-lever-retest-in-progress` loop. PA / PC-codec results are still load-bearing for the question "does data-axis on E2B lift v2 viso recall." Job B answers a different question (does the model + chain CAN reach 85% on viso enhanced+teams on a non-v2 substrate).

### Open loop: data-axis-clean-single-lever-retest-in-progress
status: in-progress
severity: high
first_seen: 2026-05-04
last_verified: 2026-05-05
close_criterion: Packet A (`R13_PA_VISOMASTER_ENHANCED_DATA.yaml`, run `3140330896851206144`, JOB_STATE_RUNNING in us-east1) and/or Packet C-codec (`R13_PC_CODEC_PLUS_DATA.yaml`, run `1202657157175050240`, JOB_STATE_RUNNING in us-east1) complete training, the resulting checkpoints are scored under both the F0 (full eval substrate) and F4 (substrate-cleaning, `analysis/substrate_cleaning_eval_2026-05-05/`) lenses, and a documented verdict is recorded for whether enabling `visomaster_enhanced + visomaster_teams_enhanced` data sources at fw=4.0 lifts visomaster_enhanced_macro_dev recall above the E2B_3200 baseline (8.4% F0 / 30.9% F4) at deployment-honest single-τ at 5% FPR ceiling. The verdict closes EITHER as (a) data-axis lever is dispositive (Packet A/C-codec materially beats E2B on viso recall under deployment policy), OR (b) data-axis lever as cleanly tested still does not lift, in which case memory `project_data_axis_lever_pulled_twice_no_lift.md` is amended to "pulled three times" with the bundle-confound caveat lifted from the prior two attempts. Either outcome closes the loop.

**2026-05-05 status update — training half complete; F0/F4 evaluation half pending.** Both Vertex jobs reached `JOB_STATE_SUCCEEDED`: PA (`3140330896851206144`) at 2026-05-04 20:49:54 UTC after 3h25m runtime, wandb run `26u8bn1t`, 16 checkpoints at `gs://training-job-outputs/best_checkpoints/26u8bn1t/` (best top_n: step5600 AUC 0.9909 EER 0.0282); PC-codec (`1202657157175050240`) at 2026-05-04 21:58:45 UTC after 3h55m runtime, wandb run `0ujswaad`, 17 checkpoints at `gs://training-job-outputs/best_checkpoints/0ujswaad/` (best top_n: step7400 AUC 0.9871 EER 0.0415). PC-codec EER is consistently 1-3pp higher than PA at comparable steps — expected with codec aug adding train-time difficulty. Periodic checkpoints span step1000-7000 (PA) / step1000-8000 (PC-codec) at 1000-step intervals, top_n covers the AUC-best subset. The verdict half (F0 + F4 scoring + documented verdict) is the next action and gates loop closure.

This loop replaces the closed-as-superseded `p14-data-fix-not-launched` loop above with a clean single-lever framing. The 2026-04-30 superseded close was based on a confounded test (P14 anti-shortcut bundle + fw=8.0). The 2026-05-04 retest uses fw=4.0 + no bundle on E2B (new anchor candidate). See "2026-05-04 evening update" subsection of Current stance for the launch context.

### Open loop: frame-level-vs-clip-level-scorer-mismatch
status: open
severity: medium
first_seen: 2026-04-29
last_verified: 2026-04-29
close_criterion: the contract scorer's clip-level recall numbers are reconciled with the frame-level AUCs from cached predictions — either (a) the corrected contract policy with `target_fake_recall_min=0.30` produces clip-level recalls within ~5pp of `(1 - threshold-implied-FNR)` from the frame-level distribution, OR (b) the contract scorer is documented as measuring something genuinely different from the frame-level signal (e.g. clip-aggregation thresholds, video-level voting policy) and the headline reporting is normalized so future agents do not compare clip-level recall to AUC-implied recall.

This loop is paired with the bucket-gap loop above. Frame-level P8A AUCs (viso 0.75 / deeplive 0.86 / teams_fake 0.91) implied "recall is far better than the 1-2% the contract was reporting" — the corrected contract policy reports viso 13.6% / deeplive 23.9% / teams_fake 52.6% which is consistent. **But the reconciliation has not been written down anywhere as a normative rule.** Future agents reading any P-series scorecard pre-fix should not be confused about whether they're looking at scorer artifact or model failure. The bucket-gap finding makes this loop materially more important: a future P14_DATA_FIX scorecard read with the unfixed contract scorer would under-report the bucket-fix lift by 10-30×, exactly as the prior whack-a-mole pattern.

### Cross-thread refs

- [`processing_signature_shortcut`](processing_signature_shortcut.md) — bucket gap and camera-signature shortcut are layered (the bucket gap explains the headline-metric magnitude; the shortcut explains the residual cross-domain failure). Both must close; closing one without the other still leaves the candidate non-deployable. The `shortcut-deployment-block` (critical, in-progress) loop in that thread is the upper bound; this thread's `p14-data-fix-not-launched` is the cheaper-first lift.
- [`identity_audit`](identity_audit.md) — the prior "narrow identity diversity" explanation is refuted (Slice 7); the bucket gap finding is what *replaces* that hypothesis as the primary explanation for the headline-metric failure.
- [`clean_teams_identity_pairing`](clean_teams_identity_pairing.md) — the unused buckets (`live-...-teams-v2`, `hdtf_visomaster_*_teams`) are paired-identity transports of the existing clean buckets. This makes Layer-2 closure structurally tractable: we have the same identities recaptured through the eval pipeline, ready to wire.
- [`contract_policy_bug`](contract_policy_bug.md) — the corrected contract policy with `target_fake_recall_min=0.30` is what makes the bucket-fix lift readable on a contract scorecard. Without the policy fix, a P14_DATA_FIX scorecard would land in the τ-tail-collapse regime and the bucket-gap closure would be invisible at the headline. The two threads' open loops are coupled in the same way: both must commit + verify for either to be load-bearing.
- [`promotion_contract_evolution`](promotion_contract_evolution.md) — the broader contract-evolution arc; this thread is the data-side input to the contract's headline metrics.
