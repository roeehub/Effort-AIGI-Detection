# Job B (pre-RLP6_04 ckpts on held-out HDTF proper viso) — FINDINGS

**Date**: 2026-05-04 (analysis after Vertex job 9082408392701509632 reached JOB_STATE_FAILED 18:44:42 UTC at the contract-scoring tail; validation phase succeeded for all 64 (suite × checkpoint) pairs)
**Source data**: `gs://training-job-outputs/test_results/job_b_pre_rlp604/job-b-pre-rlp604-2026-05-05/` (reports + diagnostic_scorecard)
**Local artifacts**: `analysis/job_b_pre_rlp604_2026-05-04/scorecard.{csv,json,wide.csv}`, `pivot_at_tau0p5.csv`, `pivot_summary.json`
**Vertex job state at writeup**: FAILED at the contract-scoring tail (404 on `teams_real_all_dev_*_videos_report.csv` because Job B's suite_manifest used `proper_*` suite names). Validation reports are intact and complete in GCS — see "How the data was obtained" below.

---

## Bottom line up front

**Trajectory hypothesis ("late FT-chain ckpts lost the viso capability") is REFUTED on HDTF proper viso.** On `proper_visomaster_enhanced_teams_dev` (1182 videos), recall climbed across the chain — R12G_STEP14000 9.6% → RLP3_05_STEP2500 12.1% → RLP5_07_STEP20500 **94.75%** → RLP6_04_STEP23500 85.4%. The chain GAINED capability between RLP3_05 and RLP5_07 (when the proper visomaster data sources were added), not lost it.

**The "production viso ceiling = 27%" appears to be substrate-specific, NOT model-structural.** The ceiling has been measured on the v2 production substrate (Dor × ~16 swap-model families per memory `project_v2_substrate_is_dor_diverse_swap.md`). On HDTF proper viso (different identities, possibly different swap-model overlap), the same FT chain reaches 85-95% recall on viso enhanced + teams transport.

**The data-axis lever moved viso enhanced+teams recall from ~12% (RLP3_05) to ~95% (RLP5_07).** This is the largest single-step lift in the chain on this substrate. It is not directly comparable to Packet A / Packet C-codec (which test data-axis on E2B base via different mechanism) but provides empirical support that the data-axis can be dispositive in some configurations.

**Important: P8A was NOT scored in Job B.** All readings here are pre-RLP6_04 chain through RLP6_04. To fully close the substrate-vs-trajectory question for the production ceiling, P8A needs to be scored on HDTF proper viso (followup, not done yet).

---

## How the data was obtained (so the next agent can verify)

Job B was a Vertex CustomJob (`9082408392701509632`, us-east1, image 1.3.255) running `arena/run_target_domain_validation_sequential.py` over 16 suites × 4 checkpoints. The validation phase completed cleanly across all 64 (suite × ckpt) pairs (logged as `OK suite=... checkpoint=...`). Per-suite reports were uploaded to GCS:

```
gs://training-job-outputs/test_results/job_b_pre_rlp604/job-b-pre-rlp604-2026-05-05/reports/
  <suite>_<ckpt>_frames_report.csv      (frame-level scores)
  <suite>_<ckpt>_videos_report.csv      (video-aggregated scores)
  <suite>_<ckpt>_group_metrics.csv      (per-method groupings)
  <suite>_<ckpt>_summary_report.txt     (text summary)
```

Plus diagnostic scorecards at `gs://training-job-outputs/test_results/job_b_pre_rlp604/job-b-pre-rlp604-2026-05-05/diagnostic_scorecard/scorecard.{csv,wide.csv,json}`.

The job then attempted to invoke the promotion-contract scorer (`arena/score_teams_promotion_contract.py`) which hardcodes `teams_real_all_dev` as the default `--dev_real_suite` (line 748). Job B's suite_manifest is `proper_data_future.provisional_2026-04-19.yaml` which uses `proper_real_teams_dev` (and 15 sister `proper_*` suites), not `teams_real_all_dev`. The scorer 404'd reading `teams_real_all_dev_r12g_step14000_videos_report.csv` (a file that was never produced because the suite was never validated) and the worker exited status 1. Vertex marked the job FAILED.

**The validation outputs are intact.** This analysis pulls them directly from GCS to `analysis/job_b_pre_rlp604_2026-05-04/scorecard.{csv,wide.csv,json}` and pivots to `pivot_at_tau0p5.csv` for inspection.

Re-running Job B is not required for the science — the failure was at the tail-end contract-scoring stage, after all useful artifacts were already on GCS.

---

## Checkpoints scored

| Key | Path | Notes |
|---|---|---|
| `R12G_STEP14000` | `gs://training-job-outputs/phase2r12_experiments/0xxqwhxg/top_n_effort_20260310_step14000_auc0.9930_eer0.0253.pth` | Pre-FT base used by RLP1+ chain |
| `RLP3_05_STEP2500` | RLP3_05 leader (low-arcface spatial recipe) | Pre-RLP6_04, in active FT chain |
| `RLP5_07_STEP20500` | RLP5_07 leader (E3 seedB) | Pre-RLP6_04, last chain member before RLP6 |
| `RLP6_04_STEP23500` | RLP6_04 leader (add enh clean) | Last chain member; FT base for many P-* descendants |

Caveat: pre-RLP6_04 ckpts predate the `INTER_LINEAR` preprocessing parity fix; expect ~0.7pp drift vs post-fix scoring. See thread `preprocessing_parity_bug.md`. The drift is small relative to the effects reported below.

---

## Results — recall and FPR at τ=0.5 (no FPR calibration)

Reading note: τ=0.5 is the fixed-threshold readout. FPRs at τ=0.5 are well below 5% on every cell, so FPR-calibrated recall at the deployment-style 5% FPR ceiling would be EQUAL OR HIGHER than these numbers (lowering τ to hit a higher FPR target lifts recall). The τ=0.5 readings are therefore **lower bounds** on FPR-calibrated recall at 5% FPR.

### Real suites — FPR @ τ=0.5 (lower is better)

| Suite (n=videos) | R12G_14k | RLP3_05_2.5k | RLP5_07_20.5k | RLP6_04_23.5k |
|---|---|---|---|---|
| `proper_real_clean_dev` (1443) | 2.43% | 1.32% | 1.87% | 1.11% |
| `proper_real_clean_lockbox` (382) | 3.66% | 1.05% | 4.45% | 0.52% |
| `proper_real_teams_dev` (1444) | 0.21% | 0.21% | 1.11% | 1.18% |
| `proper_real_teams_lockbox` (382) | 0.00% | 0.00% | 2.09% | 1.05% |

All FPRs at τ=0.5 are under 5%. The real-pool budget is comfortable across the chain on this substrate.

### Visomaster fake suites — recall @ τ=0.5 (higher is better)

| Suite (n=videos) | R12G_14k | RLP3_05_2.5k | RLP5_07_20.5k | RLP6_04_23.5k |
|---|---|---|---|---|
| `proper_visomaster_clean_dev` (262) | 100.00% | 99.24% | 99.24% | 99.24% |
| `proper_visomaster_clean_lockbox` (80) | 100.00% | 100.00% | 100.00% | 100.00% |
| `proper_visomaster_enhanced_clean_dev` (1180) | 98.39% | 96.69% | 98.56% | 98.56% |
| `proper_visomaster_enhanced_clean_lockbox` (302) | 99.67% | 97.02% | 99.67% | 98.68% |
| `proper_visomaster_teams_dev` (262) | 64.89% | 76.72% | 95.04% | 93.51% |
| `proper_visomaster_teams_lockbox` (80) | 60.00% | 75.00% | 93.75% | 90.00% |
| **`proper_visomaster_enhanced_teams_dev` (1182)** | **9.64%** | **12.10%** | **94.75%** | **85.36%** |
| **`proper_visomaster_enhanced_teams_lockbox` (302)** | **9.27%** | **12.58%** | **96.69%** | **89.74%** |

The bolded row is the closest analog to the production "viso ceiling" cell (enhanced + teams transport, the failure mode that defined the production 27% ceiling).

### Umbrella all-fakes suites — recall @ τ=0.5

| Suite (n=videos) | R12G_14k | RLP3_05_2.5k | RLP5_07_20.5k | RLP6_04_23.5k |
|---|---|---|---|---|
| `proper_fake_clean_all_dev` (1442) | 98.68% | 97.16% | 98.68% | 98.68% |
| `proper_fake_clean_all_lockbox` (382) | 99.74% | 97.64% | 99.74% | 98.95% |
| `proper_fake_teams_all_dev` (1444) | 19.67% | 23.82% | 94.81% | 86.84% |
| `proper_fake_teams_all_lockbox` (382) | 19.90% | 25.65% | 96.07% | 89.79% |

Same trajectory pattern: pre-RLP6_04 (R12G + RLP3_05) is structurally bad on teams-transported fakes (~20-26%); RLP5_07 jumps to 94-96%; RLP6_04 holds at 87-90%.

---

## Facts (computed directly from `scorecard.csv` — verifiable from `pivot_summary.json`)

1. **Viso clean (no enhancement, no teams transport) is solved across the chain.** Recall 96-100% on all 4 ckpts on `proper_visomaster_clean_dev` (n=262) and `proper_visomaster_enhanced_clean_dev` (n=1180).
2. **Viso enhanced + teams transport collapses on pre-RLP6_04 chain members predating proper data inclusion.** R12G_STEP14000 9.64%, RLP3_05_STEP2500 12.10% on `proper_visomaster_enhanced_teams_dev` (n=1182).
3. **The recall jump occurs between RLP3_05 and RLP5_07.** RLP5_07_STEP20500 94.75%, RLP6_04_STEP23500 85.36% on the same suite. This step in the chain is when the proper visomaster training data sources entered the recipe.
4. **RLP6_04 has a slight regression vs RLP5_07 on enhanced+teams viso** (94.75% → 85.36%, n=1182). On the lockbox sister suite (n=302) the regression is smaller (96.69% → 89.74%). Could be noise, drift, or real regression; not characterized.
5. **Real FPR is well-controlled across the chain on HDTF proper substrate.** Maximum FPR at τ=0.5 across all four real suites and four ckpts is 4.45% (RLP5_07 on `proper_real_clean_lockbox`).
6. **Teams transport without enhancement is significantly easier than with enhancement.** On `proper_visomaster_teams_dev` (n=262), R12G hits 64.89% and RLP3_05 hits 76.72% — both far above their enhanced+teams numbers (9.64%, 12.10%). The combination of enhancement + teams transport is what was missing from pre-RLP6_04 chain.

## Inferences (interpretation — read with skepticism, distinguish from facts above)

1. **The "viso ceiling = 27% structural across architectures" framing (memory `project_viso_ceiling_unbroken_10_packets.md`, `project_e2b_breaks_deeplive_ceiling.md`, `project_l14_does_not_break_viso_ceiling.md`) is conditional on the production v2 substrate.** On HDTF proper viso enhanced+teams, RLP6_04 (chain member, pre-P8A) reaches 85.4% at τ=0.5 with FPR<5%. P8A on the production v2 substrate gives 27% on `visomaster_enhanced_macro_dev`. The ~58pp gap is most parsimoniously attributed to substrate differences, not model-structural inability.

2. **The trajectory hypothesis** ("late FT-chain ckpts lost a viso capability that earlier ones had") **is refuted on HDTF proper viso**. The chain GAINED capability through RLP5_07; even with RLP6_04's small regression, late chain members are far ahead of early chain members.

3. **The data-axis lever can be dispositive on viso enhanced+teams in some configurations.** The RLP3_05 → RLP5_07 transition (~12% → ~95% on the enhanced+teams cell) is the largest single-step recall lift visible in any R13 chain step. This is consistent with — but does not directly transfer to — the Packet A / Packet C-codec premise (those FT from E2B and add `visomaster_enhanced` + `visomaster_teams_enhanced` at fw=4.0).

4. **The ~58pp gap between HDTF proper viso (85% at RLP6_04) and v2 production viso (27% at P8A_step5000) is the empirical signature of the substrate problem.** The substrate differs in identity (HDTF vs Dor), swap-model coverage (overlap unknown without manifest audit), and possibly image-quality / sharpness distribution (per memory `project_v2_substrate_is_dor_diverse_swap.md`, v2 is "visually less sharp than v1 enhanced"). The image-quality shortcut (memory `project_image_quality_shortcut.md`) and chronic-6 substrate-pollution finding (memory `project_job14_substrate_clean_2026-05-04.md`) are candidate explanations for why the v2 substrate uniquely depresses viso recall.

## Caveats

- **τ=0.5 readings, not FPR-calibrated.** FPRs are well below 5% on every real suite × ckpt cell, so deployment-style recall at 5% FPR ceiling would be ≥ these numbers. Magnitudes would shift slightly but direction is unambiguous. Frame-level recalc is a follow-up.
- **No P8A in Job B.** To CLEANLY separate trajectory from substrate for the production ceiling, P8A_step5000 needs to be scored on HDTF proper viso enhanced+teams. Without this, the substrate framing rests on inference (1) above and is one cheap experiment short of fully cemented.
- **Pre-RLP6_04 ckpts use pre-fix `INTER_AREA` preprocessing**; expect ~0.7pp drift from the post-fix `INTER_LINEAR` scoring path. See thread `preprocessing_parity_bug.md`. Drift is small relative to the 12% → 95% magnitude reported.
- **RLP3_05_STEP2500 is at step 2500** (very early in that recipe's training); the 12.10% number could partially reflect under-training, not just data absence. RLP5_07_STEP20500 is at step 20500 (much further trained). Single-step comparison conflates training depth with data-recipe diff.
- **Substrate differences are multifactorial.** HDTF identities, swap-model coverage, image-quality distribution all differ from v2. Without an explicit per-axis decomposition, "substrate-specific" is the wrapper claim, not a precise mechanism.
- **The Vertex job FAILED.** The `JOB_STATE_FAILED` is real but caused by a tail-end suite-name mismatch in the contract scorer, not by validation failure. All science artifacts are intact in GCS. Don't confuse the failure status with invalid data.

## Followup options (cheapest first)

1. **(Free, ~30min) Score P8A_step5000 on HDTF proper viso suites by re-using the existing reports infrastructure.** Either launch a Vertex job with checkpoint_map = {P8A_step5000} and the same suite_manifest, or run inference locally if the substrate is small enough. Closes the substrate-vs-trajectory question for the production ceiling. Recommended first.
2. **(Free, ~10min) FPR-calibrate the τ=0.5 numbers using the frame_report CSVs.** Compute τ at FPR=5% on `proper_real_clean_dev` (or pooled real) and re-eval recall on each fake suite at that τ. Refines magnitudes; doesn't change directions.
3. **(Free, ~30min) Audit swap-model overlap between HDTF proper viso and v2 substrate.** Compare method directories under `arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json` against the v2 manifest. If overlap is high, the gap isolates to identity / IQ; if low, swap-model coverage is also a confound.
4. **(Add `--skip_promotion_contract` flag to `arena/run_target_domain_validation_sequential.py`)** so future diagnostic runs over non-production suite manifests don't 404 at the tail. ~5-10 LOC. Optional cleanup; the underlying issue is the contract scorer's hardcoded suite names.

## Cross-references

- Memory `project_viso_ceiling_unbroken_10_packets.md` — needs amendment: the ceiling is unbroken on **production v2 substrate**, not universally.
- Memory `project_v2_substrate_is_dor_diverse_swap.md` — v2 substrate is Dor + diverse-swap-model; relevant to the substrate-difference framing.
- Memory `project_eval_substrate_reframe_2026-05-04.md` — establishes substrate as the binding axis on production scorecard; this finding extends that to "off-production substrates the ceiling does not hold."
- Thread `viso_bucket_gap.md` — primary thread; this finding is the universal-vs-trajectory closure entry.
- Thread `processing_signature_shortcut.md` — the image-quality / camera-signature shortcut is the candidate mechanism for why v2 substrate behaves so differently.
- Thread `preprocessing_parity_bug.md` — caveat applies to pre-RLP6_04 ckpts.
- Memory `project_p8a_frame_level_auc_2026-04-29.md` — P8A frame-level viso AUC is 0.75 on production substrate; not directly comparable but worth contrasting once P8A on HDTF proper viso is measured.
- HANDOFF reference: `docs/relaunch_handoffs/HANDOFF_NEXT_AGENT_2026-05-05.md` flagged Job B as "PAUSED at Phase 1" / failure-undiagnosed; this writeup closes that loop.
