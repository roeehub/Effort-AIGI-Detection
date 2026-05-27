# P2-D step3000 — HDTF Phase C FACTS (2026-05-08)

> **Status: factual-only.** No interpretation. Forbidden words: succeeds, fails,
> wins, promotes, deployment-grade.
>
> **Check (b)** of the IQ-deconvolution program's pre-Stage-2a checks. GPU.
> The Vertex job's per-suite evaluation step completed; the contract
> aggregation step crashed on a known bug — see §1.2.
>
> **Companion FACTS docs**:
> - [`P2_PHASE_A_VERDICT_FACTS_2026-05-08.md`](../p2_eval_2026-05-08/P2_PHASE_A_VERDICT_FACTS_2026-05-08.md) — Phase A verdict (v2 substrate)
> - [`P2_DEEPER_ANALYSIS_FACTS_2026-05-08.md`](../p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_FACTS_2026-05-08.md) — J1-J5
> - [`IQ_DECOMP_FACTS_2026-05-08.md`](../iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md) — Stage 1 R² probe
> - [`IQ_PERLAYER_PROBE_FACTS_2026-05-08.md`](../iq_perlayer_probe_2026-05-08/IQ_PERLAYER_PROBE_FACTS_2026-05-08.md) — check (a)
> - [`DOR_ENCODER_AXIS_FACTS_2026-05-08.md`](../dor_encoder_axis_2026-05-08/DOR_ENCODER_AXIS_FACTS_2026-05-08.md) — check (c)

---

## 1. Setup

### 1.1 Job

- **Vertex `6632089555598049280`** — display name `p2-d-step3000-hdtf-2026-05-08`,
  region `us-east1`, image `1.3.273`. Started 2026-05-08T16:31:59Z, terminated
  2026-05-08T19:22:50Z (2h 50m).
- **State at terminus**: `JOB_STATE_FAILED` (replica exit code 1 — see §1.2).
- **Suite manifest**: `arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml`
  (16 HDTF suites: 4 real × 12 fake).
- **Checkpoint set**: 3 ckpts from
  `arena/checkpoint_maps/teams_target_domain.p2_scratch_2026-05-08.yaml` —
  `P8A_REFERENCE_STEP5000`, `E2B_TOP_N_STEP3200`,
  `P2_D_FOURIER_PERIODIC_STEP3000`.
- **Promotion contract policy**: `target_real_fpr=0.07`,
  `target_stress_fpr=0.10`, `target_fake_recall_min=0.30`.

### 1.2 Failure root cause

All 48 (suite × ckpt) per-frame reports + the diagnostic_scorecard outputs
(τ=0.5 sidecar, 4 files) wrote to GCS. The downstream
`score_teams_promotion_contract.py:322` (`_load_video_scores`) raised a
404 trying to read `teams_real_all_dev_p8a_reference_step5000_videos_report.csv`
— a Phase A suite name (`teams_real_all_*`). The HDTF run wrote
`proper_real_teams_*` (Phase C suite names). The contract scoring script's
suite-name map is keyed on Phase A naming and does not handle Phase C
HDTF naming.

This closes the open loop `phase-c-hdtf-promotion-contract-failure` per
`docs/packet_retrospectives/STATE.md` "High-severity / load-bearing open
loops" — same root cause as the 2026-05-07 P1 PE Phase C failure.

The `promotion_contract/` GCS directory was therefore not written. The
contract verdict is reconstructed locally in §3 below by applying each
ckpt's Phase A τ (from `P2_PHASE_A_VERDICT_FACTS_2026-05-08.md` §1) to its
HDTF per-frame reports.

### 1.3 Reconstruction method

Driver: `build_hdtf_verdict.py`. For each (ckpt × suite):

1. Load `raw_reports/<suite>_<ckpt>_frames_report.csv` (schema:
   `method, label, video_id, frame_path, frame_prob, group_key, family_key`).
2. Apply the ckpt's Phase A contract τ to `frame_prob`:
   - τ_P8A = 0.9156, τ_E2B = 0.7108, τ_P2D = 0.4600.
3. Compute frame-level `real_fpr = mean(frame_prob > τ)` for the 4 real
   suites (label==0) or `fake_recall` for the 12 fake suites (label==1).
4. Also compute video-level via mean-of-frame aggregation per `video_id`
   (parity with cloud `diagnostic_scorecard` τ=0.5 sidecar, which is
   video-level).

Outputs at `outputs/hdtf_per_cell_at_phase_a_tau.csv` (48 rows) and
`outputs/hdtf_summary.json` (per-ckpt + macro aggregates).

---

## 2. Diagnostic scorecard (cloud-written, τ=0.5)

Source: `scorecard.wide.csv` (intact in `diagnostic_scorecard/`). Video-level
real_fpr / fake_recall at fixed τ=0.5. Reported here as a sanity baseline
prior to the contract-τ readout in §3.

### 2.1 Real suites — `real_fpr_at_0p5` (video level)

| suite | n_videos | P8A | E2B | P2D |
|---|---:|---:|---:|---:|
| proper_real_teams_dev | 1444 | 0.0097 | 0.0021 | 0.0000 |
| proper_real_teams_lockbox | 382 | 0.0131 | 0.0000 | 0.0000 |
| proper_real_clean_dev | 1443 | 0.0042 | 0.0028 | 0.0069 |
| proper_real_clean_lockbox | 382 | 0.0000 | 0.0000 | 0.0183 |

### 2.2 Fake suites — `fake_recall_at_0p5` (video level)

| suite | n_videos | P8A | E2B | P2D |
|---|---:|---:|---:|---:|
| proper_fake_teams_all_dev | 1444 | 0.937 | 0.148 | 0.089 |
| proper_fake_teams_all_lockbox | 382 | 0.950 | 0.149 | 0.068 |
| proper_visomaster_teams_dev | 262 | 0.943 | 0.504 | 0.206 |
| proper_visomaster_teams_lockbox | 80 | 0.950 | 0.363 | 0.150 |
| proper_visomaster_enhanced_teams_dev | 1182 | 0.936 | 0.069 | 0.063 |
| proper_visomaster_enhanced_teams_lockbox | 302 | 0.950 | 0.093 | 0.046 |
| proper_fake_clean_all_dev | 1442 | 0.983 | 0.824 | 0.962 |
| proper_fake_clean_all_lockbox | 382 | 0.982 | 0.840 | 0.963 |
| proper_visomaster_clean_dev | 262 | 0.985 | 0.966 | 0.931 |
| proper_visomaster_clean_lockbox | 80 | 0.988 | 0.925 | 0.875 |
| proper_visomaster_enhanced_clean_dev | 1180 | 0.982 | 0.792 | 0.969 |
| proper_visomaster_enhanced_clean_lockbox | 302 | 0.980 | 0.818 | 0.987 |

Note: τ=0.5 is below P8A's contract τ (0.916) and E2B's (0.711), so P8A and
E2B both score more aggressively at τ=0.5 than at their respective contract
τ. P2D's contract τ (0.460) is closest to 0.5, so P2D's τ=0.5 numbers are
nearest its operating point.

---

## 3. Contract-τ verdict (reconstructed; frame-level)

Each ckpt's Phase A contract τ applied to its HDTF per-frame reports.
Frame-level rates. Source: `outputs/hdtf_per_cell_at_phase_a_tau.csv`.

### 3.1 Real suites — `real_fpr` at Phase A τ

| suite | n_frames | P8A τ=0.916 | E2B τ=0.711 | P2D τ=0.460 |
|---|---:|---:|---:|---:|
| proper_real_teams_dev | 11,553 | 0.0079 | 0.0015 | 0.0008 |
| proper_real_teams_lockbox | 3,058 | 0.0088 | 0.0013 | 0.0010 |
| proper_real_clean_dev | 11,545 | 0.0026 | 0.0016 | 0.0186 |
| proper_real_clean_lockbox | 3,055 | 0.0000 | 0.0013 | 0.0236 |

### 3.2 Fake suites — `fake_recall` at Phase A τ

| suite | n_frames | P8A τ=0.916 | E2B τ=0.711 | P2D τ=0.460 |
|---|---:|---:|---:|---:|
| proper_fake_teams_all_dev | 11,540 | 0.8346 | 0.1237 | 0.1197 |
| proper_fake_teams_all_lockbox | 3,054 | 0.8518 | 0.1073 | 0.1083 |
| proper_visomaster_teams_dev | 2,094 | 0.8678 | 0.4070 | 0.2414 |
| proper_visomaster_teams_lockbox | 640 | 0.8406 | 0.2703 | 0.1734 |
| proper_visomaster_enhanced_teams_dev | 9,453 | 0.8272 | 0.0609 | 0.0927 |
| proper_visomaster_enhanced_teams_lockbox | 2,415 | 0.8547 | 0.0642 | 0.0911 |
| proper_fake_clean_all_dev | 11,529 | 0.9456 | 0.7419 | 0.9568 |
| proper_fake_clean_all_lockbox | 3,055 | 0.9493 | 0.7634 | 0.9558 |
| proper_visomaster_clean_dev | 2,094 | 0.9394 | 0.9327 | 0.9265 |
| proper_visomaster_clean_lockbox | 640 | 0.9422 | 0.9172 | 0.8703 |
| proper_visomaster_enhanced_clean_dev | 9,438 | 0.9469 | 0.6996 | 0.9636 |
| proper_visomaster_enhanced_clean_lockbox | 2,414 | 0.9512 | 0.7223 | 0.9785 |

### 3.3 Macro aggregates (frame-level, contract τ)

| metric | P8A τ=0.916 | E2B τ=0.711 | P2D τ=0.460 |
|---|---:|---:|---:|
| macro_real_fpr (mean over 4 real suites) | 0.0048 | 0.0014 | 0.0110 |
| macro_fake_recall (mean over 12 fake suites) | 0.8959 | 0.4842 | 0.5399 |
| worst_real_fpr (max over 4) | 0.0088 | 0.0016 | 0.0236 |
| worst_fake_recall (min over 12) | 0.8272 | 0.0609 | 0.0911 |

---

## 4. Cross-comparison vs Phase A (v2 substrate)

Phase A τ-selected scorecard from `P2_PHASE_A_VERDICT_FACTS_2026-05-08.md` §3.
Same 3 ckpts, same Phase A τ. Compares the v2 substrate (Phase A) to the
HDTF substrate (Phase C, this doc).

### 4.1 Macro fake_recall comparison

| ckpt | Phase A `dev_fake_macro_recall` (v2) | Phase C macro_fake_recall (HDTF, frame) |
|---|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.300 | 0.896 |
| E2B_TOP_N_STEP3200 | 0.508 | 0.484 |
| P2_D_FOURIER_PERIODIC_STEP3000 | 0.530 | 0.540 |

Cross-substrate Δ (HDTF macro − v2 macro):
- P8A: +0.596
- E2B: −0.024
- P2D: +0.010

### 4.2 Real_fpr comparison (lockbox)

| ckpt | Phase A `lockbox_real_fpr` (v2) | Phase C `proper_real_teams_lockbox` real_fpr (HDTF) |
|---|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.018 | 0.009 |
| E2B_TOP_N_STEP3200 | 0.024 | 0.001 |
| P2_D_FOURIER_PERIODIC_STEP3000 | 0.164 | 0.001 |

P2D's `lockbox_real_fpr` regression on Phase A v2 substrate (0.164,
9.1× E2B) does not appear on `proper_real_teams_lockbox` (HDTF), where
P2D's frame-level real_fpr is 0.001.

### 4.3 Per-suite viso recall comparison (the §6 question 1 from handoff)

D step3000's Phase A v2 substrate viso lift (per
`P2_PHASE_A_VERDICT_FACTS_2026-05-08.md` §3): visomaster_enhanced_macro_dev
fake_recall 0.166 (vs P8A 0.135, vs E2B 0.051).

D step3000's HDTF readout for the closest analog suites:

| HDTF suite | n_frames | P8A | E2B | P2D |
|---|---:|---:|---:|---:|
| proper_visomaster_enhanced_clean_dev | 9438 | 0.947 | 0.700 | 0.964 |
| proper_visomaster_enhanced_clean_lockbox | 2414 | 0.951 | 0.722 | 0.979 |
| proper_visomaster_enhanced_teams_dev | 9453 | 0.827 | 0.061 | 0.093 |
| proper_visomaster_enhanced_teams_lockbox | 2415 | 0.855 | 0.064 | 0.091 |

Direct numerical observations:
- On `proper_visomaster_enhanced_clean_*` (HDTF, no Teams transport),
  P2D's recall is at or above P8A's: 0.964 vs 0.947 (dev), 0.979 vs 0.951
  (lockbox).
- On `proper_visomaster_enhanced_teams_*` (HDTF, Teams transport applied),
  P2D's recall is 0.091-0.093 vs P8A 0.827-0.855 (a delta of 0.73-0.76).
- E2B and P2D have similar magnitudes on the `_teams_*` variants
  (E2B 0.061-0.064, P2D 0.091-0.093).

### 4.4 Worst-case fake recall

| ckpt | min recall over 12 fake suites | which suite | n_frames |
|---|---:|---|---:|
| P8A_REFERENCE_STEP5000 | 0.827 | proper_visomaster_enhanced_teams_dev | 9453 |
| E2B_TOP_N_STEP3200 | 0.061 | proper_visomaster_enhanced_teams_dev | 9453 |
| P2_D_FOURIER_PERIODIC_STEP3000 | 0.091 | proper_visomaster_enhanced_teams_lockbox | 2415 |

---

## 5. Cross-reference to IQ_DECOMP_FACTS § HDTF rows

`IQ_DECOMP_FACTS_2026-05-08.md` §2.1-2.2 reports the IQ R² regression on
HDTF cells (P8A and E2B only; P2D HDTF cells were absent because Phase C
had not run for P2-D as of that doc's authorship). With this doc's data,
the P2D HDTF cells can now be added to that decomposition; that addition
is left for a future update of `IQ_DECOMP_FACTS_2026-05-08.md`.

For reference, the HDTF substrate IQ characterization (from
`IQ_ATLAS_FACTS_2026-05-08.md` §2.5 and §6):
- HDTF reals: lap_var p50 = 122.5–209.9; min_dim p50 = 224.
- HDTF fakes: lap_var p50 = 140.3–208.7; min_dim p50 = 224.
- HDTF is the only substrate without IQ artifacts on either side of the
  real/fake split. R² of `score ~ IQ` is uniformly low across the 4 HDTF
  cells (0.016-0.090 for P8A; 0.035-0.090 for E2B).

The HDTF substrate's IQ-cleanness means HDTF-side recall measures the
content channel directly (per the IQ R² Stage 1 reading); the per-suite
recalls in §3.2 above are therefore content-channel readouts on the
clean-transport variants, while the `_teams` transport variants apply
the Teams capture pipeline on top of the HDTF content.

---

## 6. Comparison transport split (clean vs teams) by ckpt

Reorganization of §3.2: per-ckpt mean over the 6 clean fakes vs the 6
teams fakes.

| ckpt | mean fake_recall on 6 clean | mean fake_recall on 6 teams | Δ (clean − teams) |
|---|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.946 | 0.846 | +0.100 |
| E2B_TOP_N_STEP3200 | 0.796 | 0.172 | +0.624 |
| P2_D_FOURIER_PERIODIC_STEP3000 | 0.942 | 0.137 | +0.805 |

Per-ckpt clean-vs-teams gap is +0.10 for P8A, +0.62 for E2B, +0.81 for P2D.

---

## 7. Open observations (factual)

The following are observable from §2-§6. No interpretation; no ranking;
no recommendation.

1. **HDTF macro_fake_recall ordering at contract τ** (§3.3): P8A > P2D > E2B
   (0.896 / 0.540 / 0.484). On Phase A v2 dev substrate the same metric
   ordering was P2D > E2B > P8A (0.530 / 0.508 / 0.300).
2. **HDTF clean-transport recall**: all three ckpts at 0.79-0.98 across all
   6 clean fake suites (§3.2).
3. **HDTF teams-transport recall**: P8A at 0.83-0.87; E2B and P2D at
   0.06-0.41 (§3.2).
4. **P2D vs E2B on HDTF teams-transport viso suites** (§4.3): E2B mean 0.063,
   P2D mean 0.092; on `_teams` viso (raw, not enhanced): E2B 0.27-0.41,
   P2D 0.17-0.24.
5. **P2D vs P8A on HDTF clean-transport visomaster_enhanced** (§4.3): P2D
   0.964/0.979, P8A 0.947/0.951 — P2D's frame-level recall is 0.013-0.028
   higher than P8A's on these 2 cells.
6. **lockbox_real_fpr regression observed in Phase A v2 (P2D 0.164 vs E2B
   0.024) does not reproduce on HDTF** (`proper_real_teams_lockbox`: P2D
   0.001 vs E2B 0.001) — different substrate (§4.2).
7. **P8A's HDTF macro_fake_recall (0.896) exceeds its Phase A v2 macro
   (0.300) by 0.596** at the same contract τ (0.916). E2B's HDTF macro
   (0.484) is within 0.024 of its Phase A v2 macro (0.508). P2D's HDTF
   macro (0.540) is within 0.010 of its Phase A v2 macro (0.530).

(Caller may extend / contest. These are starting points only.)

---

## 8. Coverage caveats

1. **The contract aggregator never ran on this HDTF substrate.** §3 is a
   local reconstruction applying Phase A τ to HDTF per-frame reports. It
   does NOT include the contract's lex policy + recall floor mechanism;
   the τ search occurred on Phase A v2, not on HDTF. A native Phase C
   contract verdict (with HDTF-recalibrated τ) would require fixing the
   suite-name map bug in `score_teams_promotion_contract.py`.
2. **Frame-level vs video-level**: §2 (cloud τ=0.5 sidecar) is video-level;
   §3 is frame-level (driver was simpler to write at frame level given the
   raw-CSV frame schema). The per-cell numerical magnitudes differ because
   of intra-video score correlation. The §6 transport-split structure is
   stable to this choice (verified in `outputs/hdtf_summary.json` under
   `_macro_*_video` keys).
3. **`P2_D_FOURIER_PERIODIC_STEP3000` is the single P2-D ckpt examined.**
   Phase A J4 (per-identity) reports D step8000 / step19000 with very
   different real_fpr/recall patterns. HDTF was not measured for the other
   D variants.
4. **Suite manifest**:
   `arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml`
   is an HDTF-on-HDTF source corpus. The "viso" in the suite names
   (`proper_visomaster_*`) refers to the visomaster face-swap method
   applied to HDTF source material, not the v2 visomaster_enhanced_macro_dev
   that Phase A measured.

---

## 9. Cross-references

- Phase A verdict: `analysis/p2_eval_2026-05-08/P2_PHASE_A_VERDICT_FACTS_2026-05-08.md`.
- J1-J5: `analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_FACTS_2026-05-08.md`.
- Stage 1 IQ R² (HDTF was P8A+E2B only): `analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md`.
- Atlas (HDTF IQ characterization): `analysis/iq_data_atlas_2026-05-08/IQ_ATLAS_FACTS_2026-05-08.md`.
- Phase C precedent failure: `docs/packet_retrospectives/STATE.md` open
  loop `phase-c-hdtf-promotion-contract-failure` (P1 PE 2026-05-07).
- Memory: `project_pa_does_not_generalize_to_hdtf_2026-05-05.md`,
  `project_data_domain_taxonomy.md`,
  `project_job_b_findings_universal_vs_trajectory_2026-05-04.md`,
  `project_deployment_is_e2b_2026-05-06.md`.

---

## 10. Artifacts

- `build_hdtf_verdict.py` — driver (idempotent).
- `raw_reports/` — 48 frame-level CSVs (gitignored, regenerable from GCS).
- `outputs/hdtf_per_cell_at_phase_a_tau.csv` — 48-row per-cell table.
- `outputs/hdtf_summary.json` — per-ckpt + macro aggregates.
- `scorecard.csv`, `scorecard.wide.csv`, `scorecard.json`,
  `scorecard.int8_delta.csv` — cloud-written diagnostic_scorecard
  (τ=0.5, video-level).
- GCS: `gs://training-job-outputs/test_results/teams_promotion_contract/p2-d-step3000-hdtf-2026-05-08/`
  (per-frame reports + diagnostic_scorecard intact; promotion_contract
  dir not written per §1.2).
