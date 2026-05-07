# P1 (PE_PAIR_RANK_DRO) — Raw Results Document, 2026-05-07

**Status**: factual-only. No interpretation, no verdict. For agents/users to read independently.
**Companion doc**: `PHASE_F_SYNTHESIS_TEMPLATE_FACTS.md` has the structured pass/fail framing; this doc has the underlying numbers in one place.

---

## 1. Vertex job outcomes

| Job | Phase | Started (UTC) | Ended (UTC) | Duration | State |
|---|---|---|---|---:|---|
| `7995519158412378112` | A — 29-suite contract scorecard | 2026-05-07 08:24:59 | 2026-05-07 14:40:03 | 6h 15m | `JOB_STATE_SUCCEEDED` |
| `524047376604725248` | C — HDTF cross-substrate (16-suite proper_data_future) | 2026-05-07 08:24:39 | 2026-05-07 15:34:29 | 7h 09m | `JOB_STATE_FAILED` |

Phase C error: `replica workerpool0-0 exited with a non-zero status of 1`. The per-suite diagnostic_scorecard phase **completed** before the failure (8 ckpts × 16 suites = 128 rows present). The `promotion_contract/` directory was never written — failure occurred during the contract scoring step that comes after per-suite scoring. Root cause not yet investigated.

**Artifacts pulled to local disk:**
- Phase A: `analysis/p1_pe_eval_2026-05-07/scorecard/{promotion_winner.json, selected_threshold_scorecard.csv, threshold_grid.csv, promotion_contract.json}`
- Phase C: `analysis/p1_pe_eval_2026-05-07/hdtf/{scorecard.csv, scorecard.json, scorecard.wide.csv, scorecard.int8_delta.csv}` — diagnostic only, no promotion_contract verdict

---

## 2. Phase A — Promotion contract verdict

### 2.1 Winner from `promotion_winner.json`

```
checkpoint_key:      P1_PAIRRANK_PERIODIC_STEP500
checkpoint_path:     gs://training-job-outputs/best_checkpoints/s2mp5fxm/periodic_effort_20260506_step500_auc0.9848_eer0.0500.pth
selected_threshold:  0.76772
promotion_rank:      1
threshold_candidate_count: 5592

dev_fake_macro_recall:        0.353828
dev_primary_real_fpr:         0.060867
dev_worst_real_stress_fpr:    0.099215

lockbox_fake_recall:          0.70751   (n_videos=253)
lockbox_real_fpr:             0.018369  (n_videos=1361)

teams_fake_all_dev__fake_recall:               0.50934
visomaster_enhanced_macro_dev__fake_recall:    0.21636
deeplive_enhanced_dev__fake_recall:            0.33578
```

Contract config (from `promotion_winner.json` `contract` block):
- `dev_fake_suites`: `teams_fake_all_dev`, `visomaster_enhanced_macro_dev`, `deeplive_enhanced_dev`
- `dev_real_stress_suites`: `teams_real_poor_quality_dev`, `teams_real_lighting_extreme_dev`
- `dev_real_suite`: `teams_real_all_dev`
- `lockbox_fake_suite`: `teams_fake_all_lockbox`
- `lockbox_real_suite`: `teams_real_all_lockbox`
- `target_real_fpr`: 0.07
- `target_stress_fpr`: 0.10
- `target_fake_recall_min`: 0.30 (recall floor — set by the launcher's `--promotion_target_fake_recall_min 0.30` flag, which is the v3-fix to the contract policy bug)

### 2.2 Per-checkpoint × suite — REAL FPR at calibrated τ

Source: `scorecard/selected_threshold_scorecard.csv`. All FPR values at the ckpt's contract-selected τ. Lower is better (real-FPR target ≤7%, stress-FPR target ≤10%).

| ckpt | τ | teams_real_all_dev | teams_real_all_lockbox | teams_real_dor_dev | teams_real_lighting_extreme_dev | teams_real_poor_quality_dev |
|---|---:|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.9156 | 0.0695 | 0.0184 | 0.0800 | 0.0685 | 0.0260 |
| E2B_TOP_N_STEP3200 | 0.7108 | 0.0667 | 0.0235 | 0.1200 | 0.0999 | 0.0813 |
| P1_BUNDLE_PERIODIC_STEP500 | 0.9919 | 0.0670 | 0.0331 | 0.0400 | 0.0999 | 0.0455 |
| P1_BUNDLE_TOP_N_STEP3750 | 0.9994 | 0.0529 | 0.0132 | 0.0600 | 0.0999 | 0.0141 |
| P1_BUNDLE_TOP_N_STEP4000 | 0.9989 | 0.0535 | 0.0206 | 0.0800 | 0.0999 | 0.0163 |
| P1_PAIRRANK_PERIODIC_STEP500 | 0.7677 | 0.0609 | 0.0184 | 0.1600 | 0.0992 | 0.0347 |
| P1_PAIRRANK_TOP_N_STEP6000 | 0.9878 | 0.0596 | 0.0198 | 0.0800 | 0.0992 | 0.0249 |
| P1_PAIRRANK_TOP_N_STEP6750 | 0.9900 | 0.0596 | 0.0257 | 0.0400 | 0.0999 | 0.0260 |

Suite sizes (real-only): all_dev n=3253; all_lockbox n=1361; dor_dev n=50; lighting_extreme_dev n=1401; poor_quality_dev n=923.

### 2.3 Per-checkpoint × suite — FAKE RECALL at calibrated τ

Higher is better. F1 close criterion: `teams_fake_all_lockbox` recall ≥ 0.90.

| ckpt | τ | teams_fake_all_dev | teams_fake_all_lockbox | visomaster_enhanced_macro_dev | deeplive_enhanced_dev |
|---|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.9156 | 0.5255 | 0.3874 | 0.1345 | 0.2385 |
| E2B_TOP_N_STEP3200 | 0.7108 | 0.6783 | 0.6285 | 0.0509 | 0.7963 |
| P1_BUNDLE_PERIODIC_STEP500 | 0.9919 | 0.2196 | **0.8261** | 0.0055 | 0.0000 |
| P1_BUNDLE_TOP_N_STEP3750 | 0.9994 | 0.4097 | 0.1542 | 0.0727 | 0.0936 |
| P1_BUNDLE_TOP_N_STEP4000 | 0.9989 | 0.4774 | 0.2609 | 0.1582 | 0.2183 |
| P1_PAIRRANK_PERIODIC_STEP500 | 0.7677 | 0.5089 | **0.7075** | 0.2164 | 0.3358 |
| P1_PAIRRANK_TOP_N_STEP6000 | 0.9878 | 0.5372 | 0.3241 | 0.1073 | 0.3725 |
| P1_PAIRRANK_TOP_N_STEP6750 | 0.9900 | 0.5073 | 0.4862 | 0.1473 | 0.3706 |

Suite sizes (fake-only): teams_fake_all_dev n=2409; teams_fake_all_lockbox n=253; visomaster_enhanced_macro_dev n=550; deeplive_enhanced_dev n=545.

**No ckpt clears F1 (lockbox recall ≥ 90%).** Best lockbox recall: P1_BUNDLE_PERIODIC_STEP500 at 82.6%.

### 2.4 Selected-threshold pattern

| ckpt | selected τ | category |
|---|---:|---|
| E2B_TOP_N_STEP3200 | 0.7108 | low (~0.7) |
| P1_PAIRRANK_PERIODIC_STEP500 | 0.7677 | low (~0.7-0.8) |
| P8A_REFERENCE_STEP5000 | 0.9156 | mid (~0.9) |
| P1_PAIRRANK_TOP_N_STEP6000 | 0.9878 | high (~0.99) |
| P1_PAIRRANK_TOP_N_STEP6750 | 0.9900 | high (~0.99) |
| P1_BUNDLE_PERIODIC_STEP500 | 0.9919 | high (~0.99) |
| P1_BUNDLE_TOP_N_STEP4000 | 0.9989 | very high (~0.999) |
| P1_BUNDLE_TOP_N_STEP3750 | 0.9994 | very high (~0.999) |

The selected τ for both BUNDLE_TOP_N_STEP3750 and STEP4000 is in the 0.999x regime — historically associated with the contract-policy τ-tail collapse bug (memory `project_contract_policy_bug.md`). Note the `target_fake_recall_min: 0.30` floor was set; for those two ckpts dev_fake_macro_recall is 0.192 and 0.285 respectively (computed as the mean of teams_fake_all_dev, visomaster_enhanced_macro_dev, deeplive_enhanced_dev recalls). That is below the floor — so τ selection for those ckpts may not have been gated by the floor at all, or the floor logic has different semantics than the flag name suggests. Worth checking against `arena/score_teams_promotion_contract.py` and `threshold_grid.csv` to confirm.

---

## 3. Phase C — HDTF cross-substrate (DIAGNOSTIC ONLY, τ=0.5)

Phase C ran 8 ckpts × 16 HDTF suites at τ=0.5, then failed before the contract step. **F4 (≤5% FPR) is not directly readable from this data** — F4 needs FPR at the calibrated τ from Phase A, not τ=0.5. To get F4 we'd need to apply Phase A's τ values to the HDTF per-frame `reports/` (which exist at GCS but were not pulled — see §6).

What the τ=0.5 numbers below show is the **uncalibrated** baseline FPR/recall on HDTF substrates, useful for comparing ckpt behavior across HDTF subtypes.

### 3.1 HDTF real-FPR at τ=0.5

Suite sizes: teams_dev n=1444; teams_lockbox n=382; clean_dev n=1443; clean_lockbox n=382.

| ckpt | proper_real_teams_dev | proper_real_teams_lockbox | proper_real_clean_dev | proper_real_clean_lockbox |
|---|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.0097 | 0.0131 | 0.0042 | 0.0000 |
| E2B_TOP_N_STEP3200 | 0.0021 | 0.0000 | 0.0028 | 0.0000 |
| P1_BUNDLE_PERIODIC_STEP500 | **0.0727** | **0.0681** | **0.2502** | **0.2016** |
| P1_BUNDLE_TOP_N_STEP3750 | 0.0035 | 0.0026 | 0.0180 | 0.0262 |
| P1_BUNDLE_TOP_N_STEP4000 | 0.0035 | 0.0000 | 0.0111 | 0.0262 |
| P1_PAIRRANK_PERIODIC_STEP500 | 0.0048 | 0.0000 | 0.0152 | 0.0209 |
| P1_PAIRRANK_TOP_N_STEP6000 | 0.0028 | 0.0000 | 0.0055 | 0.0105 |
| P1_PAIRRANK_TOP_N_STEP6750 | 0.0042 | 0.0052 | 0.0083 | 0.0209 |

**Note**: at τ=0.5, BUNDLE_PERIODIC_STEP500 has 25% FPR on `proper_real_clean_dev` and 20% on `proper_real_clean_lockbox`. At its calibrated τ=0.9919, both should drop substantially. The other 7 ckpts already meet F4 (≤5%) at τ=0.5 on every HDTF real suite.

### 3.2 HDTF fake-recall at τ=0.5

Suite sizes: teams_alldev n=1444; teams_alllb n=382; clean_alldev n=1442; clean_alllb n=382; visomaster_teams_dev n=262; visomaster_clean_dev n=262; visomaster_enhanced_teams_dev n=1182; visomaster_enhanced_clean_dev n=1180.

Higher is better.

| ckpt | fake_teams_alldev | fake_teams_alllb | fake_clean_alldev | fake_clean_alllb | viso_teams_dev | viso_clean_dev | viso_enh_teams_dev | viso_enh_clean_dev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.9370 | 0.9503 | 0.9827 | 0.9817 | 0.9427 | 0.9847 | 0.9357 | 0.9822 |
| E2B_TOP_N_STEP3200 | 0.1482 | 0.1492 | 0.8239 | 0.8403 | 0.5038 | 0.9656 | 0.0694 | 0.7924 |
| P1_BUNDLE_PERIODIC_STEP500 | 0.9744 | 0.9843 | 0.9979 | 1.0000 | 0.9847 | 1.0000 | 0.9721 | 0.9975 |
| P1_BUNDLE_TOP_N_STEP3750 | 0.5402 | 0.5864 | 0.9938 | 0.9974 | 0.7481 | 1.0000 | 0.4941 | 0.9924 |
| P1_BUNDLE_TOP_N_STEP4000 | 0.4965 | 0.5393 | 0.9924 | 0.9974 | 0.7252 | 1.0000 | 0.4459 | 0.9907 |
| P1_PAIRRANK_PERIODIC_STEP500 | 0.6482 | 0.6466 | 0.9889 | 0.9948 | 0.7176 | 0.9886 | 0.6328 | 0.9890 |
| P1_PAIRRANK_TOP_N_STEP6000 | 0.4972 | 0.5262 | 0.9882 | 0.9921 | 0.7748 | 1.0000 | 0.4357 | 0.9856 |
| P1_PAIRRANK_TOP_N_STEP6750 | 0.6219 | 0.6623 | 0.9917 | 0.9948 | 0.8092 | 1.0000 | 0.5804 | 0.9898 |

### 3.3 Cross-substrate comparison — HDTF vs production

Same model, same checkpoint, two different substrates. P8A on the canonical "viso enhanced + teams transport" question:

| substrate | suite | ckpt | metric | value |
|---|---|---|---|---:|
| HDTF (τ=0.5) | proper_visomaster_enhanced_teams_dev (n=1182) | P8A | fake_recall | 0.9357 |
| Production (calibrated τ=0.9156) | visomaster_enhanced_macro_dev (n=550) | P8A | fake_recall | 0.1345 |

That's a 7× recall gap on the same model on what is nominally the same kind of attack. Memory `project_job_b_findings_universal_vs_trajectory_2026-05-04.md` documented this from Job B's RLP6_04 run; the same gap is now confirmed for P8A on the post-30007e0 expanded HDTF manifest.

---

## 4. Pre-existing prep work (already in PHASE_F_SYNTHESIS_TEMPLATE_FACTS.md)

- **Phase E weight-delta** — `outputs/weight_delta_verdict.csv`. 8 ckpts. qkv/out_proj ratio sits at 0.0715-0.0796 across all P1 ckpts; qkv/mlp at 0.0300-0.0330. No `apply_svd_to_in_proj=False` baseline → ratios are relative-only.
- **Phase A.5 partial** — 4 valid suites of 9 attempted (5 invalid due to stale `gs://local/...` paths in `grouped_manifest_v2.csv`).
- **Dor invariance probe (180 frames)** — `dor_invariance_2026-05-07/per_variant_fpr.csv` + `axis_decoupling_trajectory.csv`. P8A combined-FPR=0.383, E2B=0.128, BUNDLE_step4000=0.250, PAIRRANK_step6750=0.422. Sharpness raw_r trajectory: BUNDLE [−0.272, −0.342, −0.317] across step500/3750/4000; PAIRRANK [−0.517, −0.600, −0.631] across step500/6000/6750. P8A baseline raw_r=−0.644.
- **W&B logging gap** — `trainer/trainer.py:1727` silently discards diagnostic loss components when `use_group_dro=true`. BUNDLE's `pair_rank_loss` and DRO-loss scalar magnitudes are absent from W&B history. PAIRRANK_ONLY logs `train/loss/pair_rank_loss` directly (median 0.113, max 0.919, fired throughout).

---

## 5. F1-F5 close criterion — pass/fail/partial readout

Definitions from `experiments/phase2_round13/R13_P1_BUNDLE_FT_FROM_P8A.yaml` header:

- **F1**: `teams_fake_all_lockbox` recall ≥ 0.90 at FPR ≤ 0.10.
- **F2**: pair-rank metric — fraction `fake_score > real_score` on previously missed fakes ≥ 30% on ≥ 2 of 6 paired lanes; worst-group recall lift ≥ 20% on chronic / dor-drift cohorts. **Requires Phase A `reports/` to compute** — pulled per-suite `frames_report.csv` to local but B1 audit script not yet run.
- **F3**: no untargeted axis (is_webcam, face_area_fraction, min_dim, color_b_dev) amplifies +50%. **Requires the 5-axis CPU audit** (`run_audit.py` template; ckpts commented out pending Phase A; not yet run).
- **F4**: HDTF cross-substrate FPR ≤ 0.05. **Phase C failed before producing calibrated-τ reads.** At τ=0.5 (uncalibrated diagnostic), 7 of 8 ckpts already pass on all 4 HDTF real suites; BUNDLE_PERIODIC_STEP500 fails at 25% on clean_dev / 20% on clean_lockbox (τ=0.5).
- **F5** (BUNDLE only): chronic-FP `pc_generator` cluster failure rate (P8A baseline 0.520) drops by ≥ 0.10 absolute. **Requires Phase D chronic-6 filter** (`phase_d/run_chronic_filter.py` ready; needs Phase A reports run through it).

| ckpt | F1 (lockbox≥90%) | F2 (pair-rank lift) | F3 (5-axis audit) | F4 (HDTF≤5% FPR) | F5 (pc_generator↓0.10) |
|---|---|---|---|---|---|
| P8A_REFERENCE_STEP5000 | FAIL (38.7%) | n/a (baseline) | TBD | likely PASS at calibrated τ | n/a |
| E2B_TOP_N_STEP3200 | FAIL (62.9%) | n/a (baseline) | TBD | likely PASS at calibrated τ | n/a |
| P1_BUNDLE_PERIODIC_STEP500 | FAIL (82.6%) — closest | TBD | TBD | likely FAIL even at calibrated τ (25% FPR at τ=0.5) | TBD |
| P1_BUNDLE_TOP_N_STEP3750 | FAIL (15.4%) | TBD | TBD | likely PASS | TBD |
| P1_BUNDLE_TOP_N_STEP4000 | FAIL (26.1%) | TBD | TBD | likely PASS | TBD |
| P1_PAIRRANK_PERIODIC_STEP500 | FAIL (70.8%) — winner | TBD | TBD | likely PASS | n/a |
| P1_PAIRRANK_TOP_N_STEP6000 | FAIL (32.4%) | TBD | TBD | likely PASS | n/a |
| P1_PAIRRANK_TOP_N_STEP6750 | FAIL (48.6%) | TBD | TBD | likely PASS | n/a |

"Likely PASS / FAIL" on F4 are **inferences from the τ=0.5 diagnostic**, not the actual verdict. The verdict requires applying calibrated τ to the HDTF reports.

---

## 6. What's NOT YET DONE

- **F2 audit** — `analysis/p1_pe_eval_2026-05-07/run_audit.py` template exists; ckpts commented out pending Phase A reports. Reports are in GCS at `…/p1-pe-pair-rank-scorecard-2026-05-07/reports/`; not yet pulled.
- **F3 5-axis audit** — same as F2.
- **F4 calibrated-τ HDTF FPR** — Phase C `reports/` exist in GCS at `…/p1-pe-hdtf-scorecard-2026-05-07/reports/`. Need to apply Phase A's per-ckpt τ to those reports to get F4-grade FPR. Phase C job FAILED so the contract step never ran; we can do the same locally given the per-frame data.
- **F5 chronic-6** — `phase_d/run_chronic_filter.py` ready; needs `teams_real_all_dev_<ckpt>_frames_report.csv` from Phase A reports/ pulled to disk.
- **Phase C failure root-cause** — error message is generic. Need to inspect Vertex job logs.
- **PD comparison** — `analysis/pd_scorecard_artifacts_2026-05-06/unified_scorecard_simple.csv` not yet joined.

---

## 7. Cross-references (load-bearing)

- Handoff: `docs/relaunch_handoffs/HANDOFF_P1_EVAL_IN_FLIGHT_2026-05-07.md`
- Synthesis template: `analysis/p1_pe_eval_2026-05-07/PHASE_F_SYNTHESIS_TEMPLATE_FACTS.md`
- Promotion contract policy: memory `project_contract_policy_bug.md`, `project_promotion_contract.md`
- E2B is production deployment: memory `project_deployment_is_e2b_2026-05-06.md`
- HDTF substrate vs production substrate gap: memory `project_job_b_findings_universal_vs_trajectory_2026-05-04.md`
- in_proj-SVD gradient bug: memory `project_in_proj_svd_gradient_bug.md`
- Slot-1-vs-Slot-2 ablation discipline: `threads/anti_shortcut_bundle_decomposition.md`
- W&B logging trainer bug: `trainer/trainer.py:1718-1727`, `effort_detector.py:1345-1366`
