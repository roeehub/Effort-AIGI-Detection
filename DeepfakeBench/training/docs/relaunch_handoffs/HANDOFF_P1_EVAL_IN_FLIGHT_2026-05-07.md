# Handoff: P1 (PE_PAIR_RANK_DRO) evaluation in flight — 2026-05-07

**Generated**: 2026-05-07 12:30 UTC (CEST)
**Branch**: `teams-relaunch-root-2026-04-17`
**Image**: `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.270` (commit `5dccfa4`, Cloud Build `40cf4d7c-4b8b-4af7-81ca-0991f2083450`, 16m39s, digest `sha256:3c65039dfcbd68d7f0fc9643f070b04937647fa22b9275f5711c62bf34bf51e5`)
**Status**: P1 training complete (overnight), evaluation in flight (2 Vertex scorecards RUNNING + 3 CPU phases done + 2 CPU phases pending). No verdict drawn yet.

---

## Read this first — author bias warning

This handoff was written under explicit user instruction to **avoid embedding the previous agent's interpretation** of the raw data. The reason: the previous handoff to this session caused course-corrections from the user that pointed at biased framings the agent had inherited from earlier docs (see `OPEN_LOOPS contract-policy-bug-fix-not-committed` 2026-05-07 update, and the entire `eval_substrate_layering` thread).

**The numbers in this doc are factual** (cited to CSVs you can re-read).
**The "what to do next" sections are decision points, not directives** — the user makes those calls, not you.

If you find yourself wanting to write "P1 improves on P8A" or "GroupDRO is the load-bearing lever" or any verdict-shaped statement, **stop and ask the user**. The data does not yet license those statements; promotion-grade verdicts come from the contract scorecard, not from this agent's prior runs.

---

## 30-second state

Two Vertex scorecards are running in `us-east1` (image 1.3.270). They are the primary eval; everything else is supporting infra.

| | Phase A (F1) | Phase C (F4) |
|---|---|---|
| Vertex job | `7995519158412378112` | `524047376604725248` |
| Display name | `p1-pe-pair-rank-scorecard-2026-05-07` | `p1-pe-hdtf-scorecard-2026-05-07` |
| Suite manifest | `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml` (29 suites, working-tree state baked into 1.3.270) | `arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml` (16 suites, manifest expanded to 7,304 videos in 1.3.270) |
| Checkpoint map | `arena/checkpoint_maps/teams_target_domain.p1_pe_pair_rank_2026-05-07.yaml` (8 ckpts: P8A + E2B + 3 BUNDLE + 3 PAIRRANK) | same |
| Policy flags passed | `--promotion_target_real_fpr 0.07 --promotion_target_stress_fpr 0.10 --promotion_target_fake_recall_min 0.30` | same |
| Started | 2026-05-07 08:24:59 UTC | 2026-05-07 08:24:39 UTC |
| State at writeup | `JOB_STATE_RUNNING` (~4h elapsed) | `JOB_STATE_RUNNING` (~4h elapsed) |
| Output GCS root | `gs://training-job-outputs/test_results/teams_promotion_contract/p1-pe-pair-rank-scorecard-2026-05-07/` | `gs://training-job-outputs/test_results/teams_promotion_contract/p1-pe-hdtf-scorecard-2026-05-07/` |
| Verdict-bearing artifact | `<root>/promotion_contract/promotion_winner.json` + `selected_threshold_scorecard.csv` + `threshold_grid.csv` | same |

**To check current state:**
```bash
gcloud ai custom-jobs describe 7995519158412378112 --region=us-east1 --project=train-cvit2 --format='value(state,startTime)'
gcloud ai custom-jobs describe 524047376604725248  --region=us-east1 --project=train-cvit2 --format='value(state,startTime)'
```

---

## What was trained — context for the eval

P1 (PE_PAIR_RANK_DRO) is the external-advisor's primary recommendation per `NEXT_STEPS_PLAN_2026-05-06.md` §6, §8.2 P1. Two slots ran overnight 2026-05-06 → 2026-05-07 as a single-lever ablation:

| | Slot 1 BUNDLE | Slot 2 PAIRRANK_ONLY |
|---|---|---|
| Vertex job (training) | `6083600379105247232` SUCCEEDED | `6401104152834867200` SUCCEEDED |
| W&B project | `dtect-vision/phase2-experiments` | `dtect-vision/phase2-experiments` |
| W&B run ID | `tznuar61` | `s2mp5fxm` |
| W&B display | `R13_P1_BUNDLE_FT_FROM_P8A_0506-2213` | `R13_P1_PAIRRANK_ONLY_FT_FROM_P8A_0506-2213` |
| Levers | pair_rank_loss(λ=0.2, m=0.5) + multi-axis GroupDRO (R-D real-side + F-B fake-side, β=3.0, chronic_flag) | pair_rank_loss(λ=0.2, m=0.5) ONLY |
| FT base | `P8A_REFERENCE_STEP5000` (run `9lmvb5b4`) on post-`2feea58` codepath | same |
| Training YAML | `experiments/phase2_round13/R13_P1_BUNDLE_FT_FROM_P8A.yaml` | `experiments/phase2_round13/R13_P1_PAIRRANK_ONLY_FT_FROM_P8A.yaml` |
| best val_holdout AUC | 0.99514 at epoch 2, train_step 2250 | 0.99485 at epoch 3, train_step 2750 |
| best EER | 0.02101 | 0.01190 |
| Early-stopped at | epoch 3, train_step 2750 (patience=12) | epoch 4, train_step 2750 (patience=12) |
| GCS ckpt root | `gs://training-job-outputs/best_checkpoints/tznuar61/` | `gs://training-job-outputs/best_checkpoints/s2mp5fxm/` |

**Slot-1-vs-Slot-2 is structured as a single-lever ablation** per `anti_shortcut_bundle_decomposition` discipline (the P14 lesson — every stacked-lever packet must include a single-lever variant). The question Phase F has to answer: did GroupDRO add anything over pair-rank alone, or did pair-rank carry whatever lift exists?

**Trainer-side AUC (≈0.995 on val_holdout) is NOT a deployment-grade metric.** Memory `project_promotion_contract.md`, threads `value_composite_semantics` and `promotion_contract_evolution`. P8A had `value_composite=0.99+` while failing the 2-camera test; pattern repeats. Promotion goes through the contract scorecard, not val_holdout.

---

## Eval framework — six phases

Every P1 close-criterion gate (F1-F5 in `R13_P1_BUNDLE_FT_FROM_P8A.yaml` header + `NEXT_STEPS_PLAN_2026-05-06.md` §8.2 P1) maps to a phase:

| Phase | Maps to | Status | Substrate / source | Where it lives |
|---|---|---|---|---|
| **A** | F1 lockbox fake recall ≥90% at FPR ≤10% | RUNNING (Vertex) | 29-suite contract | `gs://training-job-outputs/test_results/teams_promotion_contract/p1-pe-pair-rank-scorecard-2026-05-07/` |
| **A.5** | diagnostic substrates not in contract (xinhe_may6, live_*, dor_*, etc.) | DONE PARTIAL (4 of 9 suites valid, see §"Phase A.5 partial") | `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv` non-contract slices | `analysis/p1_pe_eval_2026-05-07/diagnostic_substrates/` |
| **B1** | F2 pair-rank metric on missed fakes, vs P8A | PENDING (needs Phase A reports) | Phase A per-frame `reports/` filtered to pre-P1 missed fakes | `analysis/p1_pe_eval_2026-05-07/` (script TBD) |
| **B2** | F3 5+ axis correlation audit, vs P8A; min_dim + color_b_dev computed per-frame | PENDING (needs Phase A reports + script extension) | Phase A `reports/` + new per-frame feature computation | `analysis/p1_pe_eval_2026-05-07/run_audit.py` (template, ckpts commented out pending Phase A) |
| **C** | F4 HDTF cross-substrate FPR ≤5% | RUNNING (Vertex) | proper_data_future suite | `gs://training-job-outputs/test_results/teams_promotion_contract/p1-pe-hdtf-scorecard-2026-05-07/` |
| **D** | F5 chronic-FP cluster, 3-tier close criterion | PENDING (needs Phase A reports) | Phase A per-frame predictions on `teams_real_all_dev` filtered by `chronic_flag_definition.json` | `analysis/group_id_design_audit_2026-05-06/outputs/chronic_flag_definition.json` (canonical chronic-6 list with provenance) |
| **E** | weight-delta diagnostic (instrumented, NOT a promotion gate) | DONE | P8A vs 6 P1 ckpts | `analysis/p1_pe_eval_2026-05-07/weight_delta/compute_weight_delta.py` + `analysis/p1_pe_eval_2026-05-07/outputs/weight_delta_*.csv` |
| **F** | synthesis + decision (4-of-4 gate read; slot-1-vs-slot-2 ablation; P1-vs-PD comparison; §12 entries; plan revision) | PENDING (after A/A.5/B1/B2/C/D/E all in) | all of the above | not yet written |

Phase B1/B2/D scripts will get drafted off Phase A's actual per-frame report shape — not pre-staged on speculation. Phase A.5 + E + the dor probe are the work products this agent leaves on disk.

---

## Phase E — DONE — weight-delta against P8A

**Question (per session decision 2026-05-07)**: did `apply_svd_to_in_proj` actually fire under PE_PAIR_RANK_DRO loss class? Memory `project_in_proj_svd_gradient_bug.md` notes the lever was zero-gradient pre-`2feea58`; P8A trained pre-fix (memory `project_p8a_breakthrough.md` correction block); `packets/P8A.md:108` records the post-fix C-ablation tied within noise on P10_SYM-on-P8A. P1 is the second post-`2feea58` test of this lever class.

**Method**: load each P1 ckpt + P8A baseline; for each SVD'd module (identified by triple `(U_residual, S_residual, V_residual)`), reconstruct ΔW = U @ diag(S) @ V; compute `‖ΔW_p1 − ΔW_p8a‖_F` per layer; group by category (in_proj_qkv / out_proj / mlp). Script: `analysis/p1_pe_eval_2026-05-07/weight_delta/compute_weight_delta.py`.

**Raw output** (`analysis/p1_pe_eval_2026-05-07/outputs/weight_delta_verdict.csv`):

```
ckpt                  qkv_mean   out_proj_mean   mlp_mean   qkv/out_proj   qkv/mlp
p1_bundle_step500     0.00804    0.10099         0.24492    0.07958        0.03281
p1_bundle_step1000    0.01245    0.17415         0.39837    0.07149        0.03125
p1_bundle_step4000    0.01918    0.25279         0.58472    0.07589        0.03281
p1_pairrank_step500   0.00784    0.10213         0.24878    0.07675        0.03151
p1_pairrank_step1000  0.01225    0.17109         0.40821    0.07162        0.03002
p1_pairrank_step6750  0.02032    0.26811         0.61628    0.07580        0.03297
```

Per-layer details: `analysis/p1_pe_eval_2026-05-07/outputs/weight_delta_per_layer.csv` (432 rows = 6 ckpts × 72 SVD'd layers). By-category sums: `analysis/p1_pe_eval_2026-05-07/outputs/weight_delta_by_category.csv`.

**Layer counts**: 36 in_proj_qkv (q/k/v residuals × 12 transformer blocks), 12 out_proj (1 per block), 24 mlp (c_fc + c_proj per block).

**Reproducibility**:
```bash
cd analysis/p1_pe_eval_2026-05-07/weight_delta
python3 compute_weight_delta.py
# requires: gcloud auth, ~4.2GB ckpt download (cached after first run), torch + pandas
```

**Read framework** (per script docstring): qkv/out_proj ratio
- ≈ 1 → lever fires similarly to other unfrozen layers
- ≈ 0 → lever inert (consistent with `packets/P8A.md:108` C-ablation null)
- >> 1 → lever active and dominant

The numbers above and the read framework are factual; how to interpret them in the verdict is for Phase F + user.

---

## Phase A.5 — DONE PARTIAL — diagnostic substrates inference

**Question** (per session decision 2026-05-07 + new thread `eval_substrate_layering`): the 29-suite contract was frozen 2026-04-23 and does not include the substrates that drove the current week's investigation (xinhe_may6_falseflag, live_*_teams_prod variants, dor_evening/morning, team_sanity_may5, dor_fake_local, extra, visomaster_v2_dor). Phase A.5 scores those substrates separately so P1's verdict on the failure modes that motivated the packet is measured.

**Method**: read `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv` (14,626 frames, 14 suites; existing columns `score_P8A`, `score_E2B`, `score_PA_3800`); filter to non-contract suites (9 suites, 6,818 frames); score with P1_BUNDLE_step4000 + P1_PAIRRANK_step6750 via `arena.GCSFrameDataset` on Mac CPU. Script: `analysis/p1_pe_eval_2026-05-07/diagnostic_substrates/run_inference.py`.

**Critical caveat — partial validity** (this is documented as open loop `grouped-manifest-v2-stale-paths` in `threads/eval_substrate_layering.md`): 5,536 of 6,818 inferences (40% of all attempted) returned zero-tensor responses due to stale frame paths. Two distinct path-staleness classes:

| class | suites affected | frames | mechanism |
|---|---|---:|---|
| `gs://local/...` placeholders (files live on a different machine) | `dor_evening`, `dor_morning`, `dor_fake_local`, `extra` | 2,091 | path scheme is a placeholder, not a real GCS URI |
| Bucket layout migration | `live_reals_teams_prod` | 677 | refers to `gs://live-fakes-teams-prod/real/roee_tester_real_2026-03-24/` which has been replaced by `session_<timestamp>/...` folders |

**Verified independently** via `gcloud storage ls`:
```bash
$ gcloud storage ls "gs://live-fakes-teams-prod/real/"
gs://live-fakes-teams-prod/real/session_20260306_105048/
gs://live-fakes-teams-prod/real/session_20260324_174822/
gs://live-fakes-teams-prod/real/session_20260506_125113/
# the roee_tester_real_2026-03-24/ folder is GONE
```

**Symptom of the bug**: zero-tensor input → constant scores. P1_BUNDLE returns ~0.5006 for every frame; P1_PAIRRANK returns ~0.4618. These constants are head-bias artifacts — the existing `score_P8A` / `score_E2B` / `score_PA_3800` columns in the manifest were populated against the same paths at some earlier point and may also be stale.

### Phase A.5 — VALID for verdict purposes (4 suites, 4,050 frames)

Raw output (`analysis/p1_pe_eval_2026-05-07/diagnostic_substrates/per_suite_comparison_tau_0.5.csv`, restricted to suites where decode succeeded):

| suite | n | label | score_P8A | score_E2B | score_PA_3800 | score_P1_BUNDLE_step4000 | score_P1_PAIRRANK_step6750 |
|---|---:|---|---:|---:|---:|---:|---:|
| `xinhe_may6_falseflag` | 92 | real (FPR) | 0.0000 | 0.5761 | 0.1630 | 0.0978 | 0.1304 |
| `live_fakes_teams_prod` | 1,675 | fake (recall) | 0.7534 | 0.8113 | 0.5039 | 0.7409 | 0.7887 |
| `visomaster_v2_dor` | 2,073 | fake (recall) | 0.7381 | 0.5596 | 0.2523 | 0.6522 | 0.7771 |
| `team_sanity_may5` | 210 | real (FPR) | 0.0095 | 0.0095 | 0.0048 | 0.0095 | 0.0143 |

### Phase A.5 — INVALID due to stale paths (5 suites, 2,768 frames — DO NOT USE)

| suite | n | reason |
|---|---:|---|
| `live_reals_teams_prod` | 677 | bucket layout migration — paths gone |
| `dor_evening` | 324 | `gs://local/...` placeholders |
| `dor_morning` | 244 | `gs://local/...` placeholders |
| `dor_fake_local` | 605 | `gs://local/...` placeholders |
| `extra` | 918 | `gs://local/...` placeholders |

The `score_P8A_FPR=1.0`, `score_P1_PAIRRANK_FPR=0.0` etc. for these suites in the per-suite CSV are **constant-score artifacts**, NOT model judgments. **Ignore them in any verdict synthesis.**

**To close this gap**: fix `grouped_manifest_v2.csv` per open loop `grouped-manifest-v2-stale-paths` (regenerate against current GCS state OR add an `is_local: True` column with a local-mirror download fallback).

---

## Dor invariance probe — DONE — fresh substrate (180 frames)

**Why this exists**: Phase A.5's `dor_evening` / `dor_morning` / `dor_fake_local` are stale. The user asked the agent to investigate dor invariance further before starting Tier 2 docs work. The agent found 6 currently-extant dor real substrates in `gs://real-teams-dor-roee/` and built a small targeted probe.

**Method**: enumerate 180 dor real frames across 6 capture conditions; score with P8A + E2B + P1_BUNDLE_step4000 + P1_PAIRRANK_step6750. Script: `analysis/p1_pe_eval_2026-05-07/dor_invariance_2026-05-07/run_inference.py`. Manifest: `manifest.csv` (180 rows). Per-frame scores: `scores_full.csv`.

**Substrate provenance**:
- `dor_laptop_whiteish` (n=30): `gs://real-teams-dor-roee/session_20260424_combined_tags_121458_121007/dor-real-laptop-correct-no-virtual-bg-whiteish/`
- `dor_laptop_yellowish` (n=30): same session, `dor-real-laptop-correct-no-virtual-bg-yellowish/`
- `dor_webcam_no_vbg` (n=30): same session, `dor-real-webcam-false-flag-no-virtual-bg/` — **the canonical 2-camera-test substrate** (memory `project_signature_shortcut_finding.md`)
- `dor_webcam_with_vbg` (n=30): same session, `dor-real-webcam-false-flag/`
- `dor_session_0411` (n=30): `gs://real-teams-dor-roee/session_20260411_193750/uniform30/dor_shkedi/`
- `dor_session_0424` (n=30): `gs://real-teams-dor-roee/session_20260424_110139/uniform30/dor_shkedi/`

All 180 frames are face-cropped (~150-200px square, ~50KB), confirmed via spot-check with PIL. All have valid GCS paths, confirmed via `gcloud storage ls`.

**Raw output** (`analysis/p1_pe_eval_2026-05-07/dor_invariance_2026-05-07/per_variant_fpr.csv`, FPR @ τ=0.5; all frames are reals so lower is better):

```
variant                score_P8A   score_E2B   score_P1_BUNDLE   score_P1_PAIRRANK   n
dor_laptop_whiteish    0.0000      0.0000      0.0000            0.0000              30
dor_laptop_yellowish   0.0000      0.0000      0.3000            0.2000              30
dor_session_0411       0.0333      0.0333      0.0333            0.0333              30
dor_session_0424       0.7000      0.3667      0.6000            0.8667              30
dor_webcam_no_vbg      0.8000      0.3667      0.3333            0.7667              30
dor_webcam_with_vbg    0.7667      0.0000      0.2333            0.6667              30
ALL DOR (combined)     0.3833      0.1278      0.2500            0.4222              180
```

**Score distribution** (selected; p25/p50/p75 from `scores_full.csv`):

```
dor_webcam_no_vbg:
  P8A:       p25=0.617  p50=0.849  p75=0.948  max=0.989
  E2B:       p25=0.162  p50=0.409  p75=0.698  max=0.988
  P1_BUNDLE: p25=0.021  p50=0.223  p75=0.570  max=0.998
  P1_PAIRRANK: p25=0.557  p50=0.889  p75=0.962  max=0.999

dor_webcam_with_vbg:
  P8A:       p25=0.519  p50=0.823  p75=0.969  max=0.990
  E2B:       p25=0.015  p50=0.022  p75=0.032  max=0.067
  P1_BUNDLE: p25=0.026  p50=0.109  p75=0.378  max=0.974
  P1_PAIRRANK: p25=0.376  p50=0.760  p75=0.936  max=0.996
```

**Cross-checks against memory**:
- Memory `project_p8a_breakthrough.md` says: *"13/30 anchor frames still flip (frac_gt_0_9 = 0.43)"* on `dor-real-webcam-false-flag-no-virtual-bg` for P8A. My distribution: max=0.989, p75=0.948 — at τ=0.95, ~25% of frames flip; at τ=0.85, ~50% flip; at τ=0.5, 80% flip (24/30). Memory's "13/30 = 43% at score>0.9" is consistent with my distribution.
- Memory `project_signature_shortcut_finding.md` cites `0.94 mean prob_fake` on dor_webcam-false-flag-no-virtual-bg for *RLP6_04 / slot-07*, NOT P8A. P8A's anchor mean is lower (memory `project_p8a_breakthrough.md`: anchor Δ = −0.188 vs RLP7_08 from a starting point near saturation). My P8A p50 = 0.849 is consistent with "P8A reduced the shortcut from saturation but did not eliminate it."

So the P8A anchor numbers in this probe are NOT contradicted by memory; they reframe the "P8A breakthrough" as anchor-Δ-relative reduction, not invariance.

**Important: `dor_evening` / `dor_morning` per `NEXT_STEPS_PLAN_2026-05-06.md` §3.1 are different substrates than these** (they are the gs://local-stale-paths suites in Phase A.5; their cited numbers — P8A 0%/10.7% — are on frames currently inaccessible to this Mac). The dor probe substrate is the `session_20260424_combined_tags_121458_121007/...` set + uniform30 sessions, all in the canonical "2-camera test" cluster.

**Reproducibility**:
```bash
cd analysis/p1_pe_eval_2026-05-07/dor_invariance_2026-05-07
python3 run_inference.py
# ~5 min on Mac CPU; reuses Phase E ckpt cache; downloads E2B if not cached.
```

---

## Pre-launch eval-infra commits (this session, 2026-05-07)

These commits closed several hygiene gaps the user surfaced during the morning retro. They are **already on `teams-relaunch-root-2026-04-17` and already in image 1.3.270**.

| commit | what it does |
|---|---|
| `ad070d3` | New ckpt map `arena/checkpoint_maps/teams_target_domain.p1_pe_pair_rank_2026-05-07.yaml` (8-entry: P8A + E2B + 3 BUNDLE + 3 PAIRRANK). Added `--promotion-target-real-fpr` / `--promotion-target-stress-fpr` / `--promotion-target-fake-recall-min` flags to `arena/launch_teams_promotion_contract.sh` with v3-fix defaults `0.07/0.10/0.30` (closes the launcher gap of `contract-policy-bug-fix-not-committed` open loop). |
| `7f81e7a` | Canonicalized `infra/cloudbuild/vertex_job_template_scorecard.yaml` (500GB boot disk; was a `/tmp` orphan since PD's run). Added `--yaml-template` flag to the launcher. |
| `5dccfa4` | VERSION 1.3.269 → 1.3.270 |
| `30007e0` | Committed accumulated working-tree eval-infra state: contract suite expanded 9→29 suites (read-only diagnostic-only sub-suites added 2026-05-04, never committed); HDTF manifest expanded from 4,968 to 7,304 videos (47% larger); `arena/build_visomaster_proper_data_artifacts.py` (NEW, 880 LOC); inventory + reports regenerated. |
| `683b7d3` | `analysis/p1_pe_eval_2026-05-07/weight_delta/compute_weight_delta.py` (Phase E script). |
| `78b7fe1` | `analysis/p1_pe_eval_2026-05-07/run_audit.py` (Phase B2 template; CKPTS dict has Phase A entries commented out — uncomment after Phase A `reports/` are pulled locally). |
| `6eba42b` | `analysis/p1_pe_eval_2026-05-07/diagnostic_substrates/run_inference.py` (Phase A.5 script). |
| `79a2a7c` | `threads/contract_policy_bug.md` open loop updated `open` → `in-progress` (audit found scorer was committed `974e033` 2026-04-29; entry text was stale by a week). New thread `threads/eval_substrate_layering.md` documenting the contract / HDTF / diagnostic three-layer model with three new open loops: `diagnostic-substrates-not-in-contract`, `eval-manifests-version-pinning`, `open-loops-stale-state-claims`. |
| `080beee` | NEXT_STEPS_PLAN §12 entry for the morning + new open loop `grouped-manifest-v2-stale-paths`. |

**Open loops added this session** (in `OPEN_LOOPS.md`, regenerated by `tools/regenerate_open_loops.py`):

| loop id | severity | source |
|---|---|---|
| `diagnostic-substrates-not-in-contract` | high | `threads/eval_substrate_layering.md` |
| `eval-manifests-version-pinning` | high | `threads/eval_substrate_layering.md` |
| `grouped-manifest-v2-stale-paths` | high | `threads/eval_substrate_layering.md` |
| `open-loops-stale-state-claims` | medium | `threads/eval_substrate_layering.md` |

**Open loops moved this session**:
| loop id | from → to | source |
|---|---|---|
| `contract-policy-bug-fix-not-committed` | `open` → `in-progress` (components 1+2 met; component 3 in flight via Phase A) | `threads/contract_policy_bug.md` |

---

## What's pending — your work

**You should NOT presume the verdict shape.** When Phase A and Phase C results land, run B1, B2, D, then F — let the data decide.

### When Phase A finishes (ETA: should be soon if not already; submitted 08:24 UTC, expected ~2-3h based on PD's similar-scale run)

1. **Pull per-frame reports locally**:
```bash
gcloud storage cp -r gs://training-job-outputs/test_results/teams_promotion_contract/p1-pe-pair-rank-scorecard-2026-05-07/reports/ \
                    analysis/p1_pe_eval_2026-05-07/raw_reports/
```

2. **Read F1 verdict — DO THIS FIRST**:
```bash
gcloud storage cp gs://training-job-outputs/test_results/teams_promotion_contract/p1-pe-pair-rank-scorecard-2026-05-07/promotion_contract/{promotion_winner.json,selected_threshold_scorecard.csv,threshold_grid.csv,promotion_contract.json} \
                  analysis/p1_pe_eval_2026-05-07/scorecard/
```
**Inspect `selected_threshold` first** (memory `project_contract_policy_bug.md`). If τ ≈ 0.99x the policy bug fired again — fall back to τ=0.5 diagnostic. The `--promotion_target_fake_recall_min 0.30` flag should prevent this; verify `promotion_contract.json` shows `target_fake_recall_min: 0.30`.

3. **Phase B1 — F2 pair-rank metric**: write a script that reads per-frame `reports/<suite>_<ckpt>_frames_report.csv`, joins fakes to their paired reals via `(sample_id, frame_idx)` (paired training lanes only — see `analysis/pair_coverage_audit_2026-05-06/`), computes per-pair `fake_score − real_score`, filters to "pre-P1 missed fakes" (P8A score < 0.5 on those fakes), reports the fraction with fake_score > real_score by lane. Close criterion: ≥30% lift on ≥2 of 6 paired lanes vs P8A.

4. **Phase B2 — F3 5+ axis correlation audit**: extend `analysis/p1_pe_eval_2026-05-07/run_audit.py`:
   - Uncomment the 6 Phase-A ckpt entries in `CKPTS` dict
   - Anchor comparison vs **P8A** (FT base of P1), NOT E2B (PD's FT base)
   - Add `min_dim` and `color_b_dev` axes (un-PD-targeted dor-drift dominant axes per Phase 0e in `NEXT_STEPS_PLAN` §8.1). The `lockbox_tagging` parquet has `width`/`height` so `min_dim = min(width, height)` is trivial; `color_b_dev` requires reading source frames and computing B-channel std per-frame. Reuse `arena.GCSFrameDataset` for the read.
   - Close criterion (F3): no untargeted axis amplifies +50% vs P8A.

5. **Phase D — F5 chronic-FP cohort**: read `analysis/group_id_design_audit_2026-05-06/outputs/chronic_flag_definition.json` (canonical chronic-6 list with provenance: `bla_bla_chow`, `bla_bla_chow__s2`, `PC_Generator__s22`, `PC_Generator__s45`, `roy_d`, `Q__s6`; match rule: exact-or-substring on `base_identity`). Filter Phase A's per-frame predictions on `teams_real_all_dev` to chronic-6 identities. Compute three numbers per ckpt:
   - chronic-6 FPR
   - per-identity FPR breakdown
   - `pc_generator` cluster (subset: `PC_Generator__s22` + `PC_Generator__s45`) failure rate

   Close criterion (3-tier per session decision):
   1. Slot 1 chronic-6 FPR ≤ P8A baseline FPR
   2. Slot 1 chronic-6 FPR ≤ 8% absolute
   3. Slot 1 chronic-6 FPR < Slot 2 by ≥ 10pp absolute (the GroupDRO ablation)

### When Phase C finishes (ETA: similar to Phase A; smaller suite count so likely earlier)

6. **Phase F4 — HDTF cross-substrate**: read `gs://training-job-outputs/test_results/teams_promotion_contract/p1-pe-hdtf-scorecard-2026-05-07/promotion_contract/promotion_winner.json`. Close criterion: `proper_real_*` FPR ≤ 5%. **Caveat**: the HDTF manifest was expanded from 4,968 to 7,304 videos in commit `30007e0`; PA's prior 7.87% number on `proper_visomaster_enhanced_teams` (memory `project_pa_does_not_generalize_to_hdtf_2026-05-05.md`) was on the smaller manifest, NOT this one. F4 is an absolute-gate reading; cross-packet comparisons need re-baselining.

### Phase F — synthesis & decision

After A + B1 + B2 + C + D land:
1. **Pass/fail/partial table** with one row per close-criterion (F1, F2, F3, F4, F5) × one column per ckpt. Mark Phase E weight-delta as a sub-row (instrumented diagnostic, NOT a gate).
2. **Slot 1 vs Slot 2 ablation read** — on each gate, compute the BUNDLE − PAIRRANK_ONLY delta. Memory `anti_shortcut_bundle_decomposition` requires this; the question is "does GroupDRO add value over pair-rank alone?" Do NOT answer this question without the data — Phase A's 50-row `teams_real_dor_dev` plus the chronic-6 per-identity breakdown are the load-bearing evidence.
3. **P1 vs PD comparison** — pull `analysis/pd_scorecard_artifacts_2026-05-06/unified_scorecard_simple.csv` and compare on `visomaster_enhanced_macro_dev` / `teams_fake_lockbox` / `teams_real_dor_dev`. PD's verdict is "conclusions NOT yet drawn per user instruction" so you may need to re-run the F2/F3 audit on PD ckpts in parallel for matched comparison.
4. **Append §12 entry to `NEXT_STEPS_PLAN_2026-05-06.md`** with verdict table + ablation read + comparison + decision.
5. **Revise §1-11 of NEXT_STEPS_PLAN** if the verdict materially changes priors.
6. **Update `OPEN_LOOPS.md` via `tools/regenerate_open_loops.py`** for any newly resolved or moved entries (e.g. `contract-policy-bug-fix-not-committed` component 3 should close to `resolved` if Phase A ran via the recall-floor path).
7. **Memory entries**: ask the user before writing any. Do not unilaterally save numerical findings as memory (provenance-pinning convention is being established as part of `eval-manifests-version-pinning` open loop).

---

## Decision points the user reserves — do not presume

These are flagged because the previous handoff's drift came from agents presuming verdict-shape. Each has been raised in this session and the user has *not* given a directive yet:

1. **Phase A.5 stale-path fix vs accept partial coverage**. Open loop `grouped-manifest-v2-stale-paths` proposes regenerating the manifest. The agent has not been authorized to do this. Phase A.5's 4-of-9-suite coverage stands until the user decides.

2. **Memory provenance convention** (e.g. citing manifest_version + git_sha alongside numerical claims). Discussed in session retro Tier 3; not yet adopted as policy.

3. **Tier 2/Tier 3 doc-gap fixes** (verdict_template, AGENTS.md additions, preflight_launch.sh, lockbox tagging extension). Tier 1 done; Tier 2/3 deferred until after P1 verdict so the verdict drives the requirements.

4. **Whether to extend the contract suite with diagnostic substrates** (open loop `diagnostic-substrates-not-in-contract`). Possible follow-up packet, not a decision yet.

5. **Whether to interpret the dor probe numbers**. The agent compiled the data; the user explicitly asked the agent NOT to embed an interpretation in this handoff. The numbers stand for the next agent or the user to read independently.

6. **Whether P1 promotes** (the actual verdict). 4-of-4 close criterion in YAML headers; verdict is for Phase F + user, not pre-data.

---

## Reproducibility — quickest path to refresh state

If picking this up from cold:

```bash
# 1. Branch + image
cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training
git status                   # should be clean except WIP files unrelated to P1
git log --oneline -10        # confirm last commit is `080beee` or later
cat VERSION                  # 1.3.270

# 2. Vertex job state
gcloud ai custom-jobs describe 7995519158412378112 --region=us-east1 --project=train-cvit2 --format='value(state)'
gcloud ai custom-jobs describe 524047376604725248  --region=us-east1 --project=train-cvit2 --format='value(state)'

# 3. Phase E (if you want to verify the qkv-residual numbers)
cd analysis/p1_pe_eval_2026-05-07/weight_delta
python3 compute_weight_delta.py     # ~3 min, reuses cache

# 4. Dor probe (if you want to verify the 180-frame numbers)
cd ../dor_invariance_2026-05-07
python3 run_inference.py            # ~5 min, reuses Phase E cache

# 5. Phase A.5 partial (if you want to verify the 4 valid suites; will hit stale paths on the 5 broken ones)
cd ../diagnostic_substrates
python3 run_inference.py            # ~14 min; 5,536 decode warnings expected
```

All five `run_*.py` scripts are deterministic given the same ckpts + frames; results should match `outputs/*.csv` to 1e-4.

---

## Cross-references

**Plan**: `docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md` §3 (substrates), §6 (advisor pair-rank rationale), §8.2 P1 (close criterion), §12 (chronological log; this handoff's events appear under `2026-05-07 — Slots 1+2 finished overnight; P1 evaluation in flight`).

**Threads**:
- `threads/eval_substrate_layering.md` (NEW; the three-layer model + 4 open loops)
- `threads/contract_policy_bug.md` (updated 2026-05-07; `in-progress`)
- `threads/promotion_contract_evolution.md` (load-bearing context for "trainer composite is not deployment-grade")
- `threads/value_composite_semantics.md` (what val_holdout AUC can and cannot tell us)
- `threads/anti_shortcut_bundle_decomposition.md` (slot-1-vs-slot-2 ablation discipline)
- `threads/in_proj_svd_gradient_bug.md` (Phase E context)
- `threads/processing_signature_shortcut.md` (dor probe context — this is where `dor-real-webcam-false-flag-no-virtual-bg` was first identified as the camera-signature anchor)

**Memory anchors**:
- `project_promotion_contract.md` — lockbox-anchored contract is the canonical readout
- `project_contract_policy_bug.md` — the recall-floor τ-collapse failure mode (caveat: this memory's "FIXED IN WORKING TREE, uncommitted" framing is now stale; scorer was committed `974e033` 2026-04-29; OPEN_LOOPS entry has been updated)
- `project_signature_shortcut_finding.md` — slot-07 / RLP6_04 anchor numbers
- `project_p8a_breakthrough.md` — P8A anchor numbers + the `apply_svd_to_in_proj` correction block
- `project_in_proj_svd_gradient_bug.md` — the bug fix `2feea58`
- `project_class_sep_not_predictive.md` — class_separation peak does not predict scorecard outcomes (relevant if you read the W&B summary's `train/collapse/class_separation`)
- `project_train_auc_not_valid_promotion_signal.md` — train AUC drops while operational fake recall rises; the same warning applies to val_holdout AUC

**Code paths**:
- `arena/score_teams_promotion_contract.py:281,282,291` — v3-fix scorer defaults (committed `974e033`)
- `arena/run_target_domain_validation_sequential.py:992-1001` — runner CLI flags (`--promotion_target_*`)
- `arena/launch_teams_promotion_contract.sh` — local launcher (this session's `ad070d3` + `7f81e7a` edits)
- `detectors/effort_detector.py:813-870` — SVD residual setup (`apply_svd_to_in_proj`, `unfreeze_final_proj`, etc.)
- `data/sources/combined_paired.py:3503` + `detectors/effort_detector.py:1073` — pair_rank_loss wiring (the lever P1 tests; per `b50f245` commit message)

**Authoritative artifacts produced this session**:
- `analysis/p1_pe_eval_2026-05-07/outputs/weight_delta_*.csv` (Phase E)
- `analysis/p1_pe_eval_2026-05-07/diagnostic_substrates/{per_suite_comparison_tau_0.5.csv, scores_full.csv ⊆ grouped_manifest_v2_with_p1.csv}` (Phase A.5 partial)
- `analysis/p1_pe_eval_2026-05-07/dor_invariance_2026-05-07/{per_variant_fpr.csv, scores_full.csv, manifest.csv}` (dor probe)
- `arena/checkpoint_maps/teams_target_domain.p1_pe_pair_rank_2026-05-07.yaml` (in image 1.3.270)
- `infra/cloudbuild/vertex_job_template_scorecard.yaml` (500GB scorecard template)

---

## End-of-session checklist

- [x] Phase E run + result CSVs committed (script committed; CSVs are gitignored per repo convention)
- [x] Phase A.5 partial run + result CSVs (script committed; CSVs gitignored)
- [x] Dor probe run + result CSVs (script and manifest committed below; CSVs gitignored)
- [x] Pre-launch eval-infra commits (5 commits: ad070d3, 7f81e7a, 5dccfa4, 30007e0, 683b7d3, 78b7fe1, 6eba42b, 79a2a7c, 080beee)
- [x] Open loops updated (1 moved, 4 new) + OPEN_LOOPS.md regenerated
- [x] §12 entry in NEXT_STEPS_PLAN
- [x] This handoff doc
- [ ] Phase A scorecard finishes + verdict read (you)
- [ ] Phase C scorecard finishes + verdict read (you)
- [ ] Phase B1, B2, D scripts written + run (you)
- [ ] Phase F synthesis + plan revision (you)
