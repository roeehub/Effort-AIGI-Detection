# Facts pack for an independent second-opinion review — 2026-05-06

> **Purpose**: this is a deliberately bias-stripped factual snapshot for an outside reviewer. It contains data, citations, and explicit open questions. It does NOT contain recommendations, ranked technique lists, or "what to do next" framings. The reviewer is asked to read this + the cited raw artifacts and form their own opinion before being shown any of the project's prior synthesis.
>
> **Scope rule**: every claim below cites an analysis directory, a CSV/JSON file, a memory entry by name, or a thread doc. If you find a sentence here that asserts a recommendation or a "should" without a citation, it's a bug — flag it.

## 1. The system, the user goal, the deployed substrate

- **Project**: DeepfakeBench/training — face-swap manipulation detection on Microsoft Teams video. CLIP-B16 (or scratch-CLIP-B16) backbone + arcface or CE head; trained on a mixture of `deeplive`, `visomaster`, `df40`, and proper-data lanes. Detector loads via `arena/model_arena.py::load_model`.
- **User goal (verbatim from earlier sessions)**: a single deployed model that holds **fake recall ≥ 90% on target methods (deeplive, visomaster, teams) AND real-side FPR ≤ 5% AND robustness across capture conditions** (lighting / camera / codec / color). All three pillars are load-bearing — memory `project_success_criteria.md`. Single τ at deployment; per-mode τ is NOT deployable because Teams does not surface capture mode at inference (memory `feedback_per_mode_tau_not_deployable.md`).
- **Deployed model identity (as of 2026-05-06)**: confirmed E2B (or near-clone) — Pearson r = +1.000 between deployment scores in `gs://live-fakes-teams-prod/real/session_20260506_125113/metadata/frame_tags.json` and local CPU inference of `E2B_TOP_N_STEP3200` from `gs://training-job-outputs/best_checkpoints/rmat8lwx/`. See `analysis/xinhe_cross_camera_audit_2026-05-06/outputs/MODEL_SCORES_FINDINGS.md` and memory `project_deployment_is_e2b_2026-05-06.md`.

## 2. Data sources currently available — inventory

The project has accumulated multiple frame-source buckets / suites. The identity-browser dataset at `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv` (14,626 rows after today's append) is the operational consolidated view; per-suite breakdown:

| suite | n | label class | source bucket | notes |
|---|---:|---|---|---|
| `teams_real_all_dev` | 4,295 | real | `teams-faces-data-test-2914-fake-4420-real-feb-28` | dev real frames (training/eval substrate) |
| `teams_real_all_lockbox` | 1,418 | real | same bucket | held-out real lockbox |
| `teams_fake_all_dev` | 1,620 | fake | same bucket | dev fake frames |
| `teams_fake_all_lockbox` | 425 | fake | same bucket | held-out fake lockbox |
| `teams_real_dor_dev` | 50 | real | same bucket | Dor sub-pool of dev real |
| `visomaster_v2_dor` | 2,073 | fake | `visomaster-enhanced-face-cropped-v2` | "v2" = Dor identity × 16 swap-model families; PA's data lever sourced from this bucket; see memory `project_v2_substrate_is_dor_diverse_swap.md` |
| `live_reals_teams_prod` | 677 | real | `live-fakes-teams-prod/real/` | production live captures: roee_tester (323), tester_roee (173), royd_real (181) |
| `live_fakes_teams_prod` | 1,675 | fake | `live-fakes-teams-prod/fake/session_20260414_112354/` | production live fakes: 11 xinhe-fake variants (no-glasses + glasses subtypes) + 6 xiang-fake variants + 3 dor_fake_deeplive_enhanced |
| `dor_evening` | 324 | real | `live-fakes-teams-prod/real/...` | Dor real, evening session (single identity) |
| `dor_morning` | 244 | real | `live-fakes-teams-prod/real/...` | Dor real, morning session (single identity) |
| `dor_fake_local` | 605 | fake | local | Dor as victim across 12 fake-method variants (vladi_regular/enhanced, bill_gates_regular, chinese_regular/enhanced, elone_regular/enhanced, face_1_regular/enhanced, mark_zukerberk_regular/enhanced, trump_regular) |
| `team_sanity_may5` | 210 | real | `real-teams-dor-roee` | 5 identities (Dor, Roee, Noyn, Xiang, Xinhe) — one-day controlled sanity capture |
| `extra` (real + fake mix) | 918 | both | local | extra_roy_d (236 real), extra_xinghe (19 real + 366 fake), extra_xiang (224 real + 73 fake) |
| `xinhe_may6_falseflag` | 92 | real | `live-fakes-teams-prod/real/session_20260506_125113/xinhe-may6-real-false-flag-1/` | **NEW 2026-05-06** captured today 11:06 UTC; deployment scores 0.83-0.93 RED on most; appended to identity-browser manifest |

Each row in `grouped_manifest_v2.csv` carries pre-computed columns: `score_P8A`, `score_E2B`, `score_PA_3800` (scores from cached frame reports + today's local CPU inference for the may6 set), `face_area_ratio`, `face_size`, `quality`, `is_lockbox`. Score CSV pointers per ckpt are in `analysis/identity_browser_2026-05-05/data/*_scores_*.csv`.

## 3. Per-suite per-checkpoint frame-level metrics on the consolidated dataset

Computed by joining the score columns above; raw numbers, no τ-policy adjustment. **Lower-is-better for real suites (FPR), higher-is-better for fake suites (recall).** Single-τ is the score column ≥ 0.5 default; per-suite τ-recalibration is NOT applied here.

### Real suites — FPR@0.5 (full pool, no chronic-6 exclusion)

| suite | n | P8A | E2B | PA_3800 |
|---|---:|---:|---:|---:|
| `teams_real_all_dev` | 4,295 | 12.9% | 10.7% | 13.5% |
| `teams_real_all_lockbox` | 1,418 | 6.8% | 6.3% | 18.1% |
| `teams_real_dor_dev` | 50 | 30.0% | 24.0% | 76.0% |
| `dor_morning` | 244 | 10.7% | 3.3% | 22.1% |
| `dor_evening` | 324 | 0.0% | 0.0% | 1.5% |
| `live_reals_teams_prod` | 677 | 0.4% | 1.2% | 1.3% |
| `team_sanity_may5` | 210 | 1.0% | 1.0% | 0.5% |
| `extra` (real subset) | 479 | 7.3% | 14.0% | 17.5% |
| `xinhe_may6_falseflag` (NEW today) | 92 | **0.0%** | **57.6%** | 16.3% |

### Fake suites — recall@0.5

| suite | n | P8A | E2B | PA_3800 |
|---|---:|---:|---:|---:|
| `teams_fake_all_dev` | 1,620 | 93.9% | 94.8% | 96.4% |
| `teams_fake_all_lockbox` | 425 | 66.4% | 83.3% | 79.3% |
| `visomaster_v2_dor` | 2,073 | 73.8% | 56.0% | 25.2% |
| `live_fakes_teams_prod` | 1,675 | 75.3% | 81.1% | 50.4% |
| `dor_fake_local` | 605 | 92.6% | 97.0% | 98.7% |
| `extra` (fake subset) | 439 | 89.7% | 94.8% | 57.4% |

### Matched-domain AUC (production-distribution audit) — `live_*_teams_prod` only

Pairing 1,675 prod fakes vs 677 prod reals:
| ckpt | AUC | recall@FPR=5% | recall@FPR=10% |
|---|---:|---:|---:|
| P8A | 0.994 | 98.1% | 99.5% |
| E2B | 0.988 | 91.3% | 96.6% |
| PA_3800 | 0.952 | 67.8% | 83.0% |

### Per-identity recall on the 11 xinhe-fake-* variants in `live_fakes_teams_prod`

| variant | n | P8A | E2B | PA |
|---|---:|---:|---:|---:|
| xinhe-fake-1 (no glasses) | 111 | 28.8% | 47.7% | 9.0% |
| xinhe-fake-2 (no glasses) | 86 | 16.3% | 51.2% | 1.2% |
| xinhe-fake-3 (no glasses) | 133 | 42.9% | 42.9% | 18.0% |
| xinhe-fake-4 | 104 | 67.3% | 94.2% | 51.9% |
| xinhe-fake-5 | 88 | 60.2% | 88.6% | 33.0% |
| xinhe-fake-6 | 82 | 81.7% | 67.1% | 15.9% |
| xinhe-fake-7 | 66 | 100.0% | 100.0% | 57.6% |
| xinhe-fake-8 | 107 | 85.0% | 98.1% | 29.9% |
| xinhe-fake-8-glasses | 123 | 80.5% | 91.1% | 29.3% |
| xinhe-fake-9-glasses | 60 | 61.7% | 96.7% | 36.7% |
| xinhe-fake-10-glasses | 73 | 76.7% | 97.3% | 37.0% |
| xinhe-fake-11-glasses | 44 | 65.9% | 95.5% | 59.1% |

xiang-fake-* variants (different person, same setup): all ≥ 96% recall on all ckpts.

### Per-identity FPR on the chronic-6 reals

Identities documented in memory `project_chronic_offenders_partition_per_ckpt_2026-05-04.md`. Their FPR varies dramatically per ckpt:
| identity | n (across suites) | P8A FPR | E2B FPR | PA FPR |
|---|---:|---:|---:|---:|
| `extra_roy_d` | 236 | 12.7% | 26.3% | 32.6% |
| `bla_bla_chow` (variants) | ~1,030 | varies; see analysis/job_11_identity_audit_2026-05-04/ | | |
| `PC_Generator__s22` | varies | 0.91 mean P8A vs 0.06 mean E3 (memory cited) | | |
| `Q__s6` | varies | varies | | |

Detailed per-identity tables are in `analysis/job_11_identity_audit_2026-05-04/outputs/per_identity_pivot.csv`.

## 4. Today's seven CPU probes — raw results

All directories under `analysis/`. Each includes a per-row CSV; the headline numbers below are the load-bearing measurements only.

### Probe 1 — `xinhe_cross_camera_audit_2026-05-06/`

92 may6_falseflag + 60 may5_correct frames (both file-prefixed `Generator PC` — same nominal participant/hardware, captured 2026-05-06 vs the team_sanity_may5 session). 24 IQ features extracted per frame (Laplacian variance full + face, luma mean+std, saturation mean+std, Sobel edge, HF-energy ratio, face-area-fraction, image dimensions, color channel stats). 5-fold CV logistic regression (n_jobs=1, balanced) on may6 (1) vs may5 (0).

Raw results (`outputs/axis_comparison.csv`, `falseflag_classifier.json`):
- 24-feature multivariate LR: **5-fold CV AUC = 1.0000** (every fold).
- Top discriminators by Cohen's d: `sat_std` d=−5.95 (single-axis AUC=1.0; max(may6)=44.98 < min(may5)=55.85), `sobel_mean_face` d=−4.33, `face_area` d=+4.31, `lap_var_face` d=−2.65 (face HF-energy roughly halved), `width`/`height` d≈4.3 (single-axis AUC=1.0).
- Within-may6 (n=92) Pearson r vs `deploy_score`: `lap_var_face` +0.331, `r_mean` −0.378, `r_std` +0.306, `luma_mean` −0.275, `sat_std` −0.234.

### Probe 2 — `dor_drift_mechanism_2026-05-06/`

820 real Dor frames sampled across 7 recording sessions (`dor_evening`, `dor_morning`, `team_sanity_may5__Dor`, `teams_real_all_lockbox.dor_shkedi`, `teams_real_all_lockbox.real_dor`, `teams_real_all_dev.dor_shkedi__s16`, `teams_real_dor_dev.dor_shkedi`). 200/session sampling cap. IQ axes computed per frame.

Raw drift magnitudes (low = `dor_evening`, high = `teams_real_dor_dev.dor_shkedi`):
| ckpt | low mean | high mean | Δ | ratio |
|---|---:|---:|---:|---:|
| P8A | 0.0148 | 0.331 | 0.316 | 22.4× |
| E2B | 0.0213 | 0.300 | 0.279 | 14.1× |
| PA_3800 | 0.1194 | 0.665 | 0.546 | 5.6× |

Endpoint-union Ridge (alpha=1.0) `score_ckpt ~ IQ_axes` (`outputs/drift_attribution.csv`, `regression_p8a.json`):
- Predicted/total drift on P8A = 0.285/0.316 = **90.2% explained** by named pixel-domain axes; named-axis residual = 9.8%. E2B 87%; PA 82%.
- Per-axis attribution to drift on P8A: `min_dim` (resolution) 53%, `color_b_dev` (LAB B-channel cast) 27%, `edge_mag` (Sobel mean) 44%, sharpness COUNTER-DRIVES (high-quality crisp frames push score up).

### Probe 3 — `amp_vs_phase_probe_2026-05-06/`

5,000 frames stratified across `teams_fake_all_dev` (label=1) + `teams_real_all_dev` (label=0), 5-fold StratifiedGroupKFold by video_id. 224×224 grayscale FFT; 16 radial × 8 angular bins = 128 features per spectrum half. Logistic regression (C=1.0, balanced).

Raw 5-fold mean AUCs (`outputs/amp_phase_aucs.csv`):
- Amplitude features: **0.946 ± 0.005**
- Phase features: **0.861 ± 0.008**
- Pixel-baseline (pooled grayscale stats): 0.671
- Amp shuffle-control: 0.497 (sanity)
- Top amplitude bands by |coef|: radial 8-13 dominate; mid-band radial 11-12 has strongest individual coefficients.
- Top phase bands by |coef|: radial 9-12 dominate.

### Probe 4 — `paired_feature_consistency_2026-05-06/`

275 paired (raw, teams) viso fake frame pairs from cached features at `analysis/clip_vs_p8a_viso_2026-05-03/outputs/p8a__features.npz` and `clip_b16_raw__features.npz`. P8A is frozen-feature on the 1100-frame viso eval substrate.

Per-pair P8A cosine-distance distribution (`outputs/pair_distances.csv`):
- 47% < 0.05 (tight invariant)
- 21% in [0.05, 0.15]
- 22% in [0.15, 0.30]
- 11% > 0.30 (max 0.92)

CLIP-B16-raw baseline on same pairs: 6% / 93% / 0.7% / 0% (no >0.30 tail).

Pearson r and OLS R² (`outputs/regression_score_on_feat.json`):
- r(P8A pair-feat-dist, P8A pair-score-delta) = **+0.697**, OLS R² = 0.485, p < 1e-40.
- r(P8A pair-feat-dist, E2B pair-score-delta) = -0.089, p = 0.14 (null).
- E2B mean score raw vs teams: 0.086 vs 0.172 — **sign reversed** (Wilcoxon p = 0.0019). See memory `project_pair_loss_premise_refuted_2026-05-04.md` for prior characterization.

### Probe 5 — `xinhe_cross_camera_audit_2026-05-06/outputs/MODEL_SCORES_FINDINGS.md`

Local CPU inference (P8A, E2B, PA_3800) on 152 may6+may5 frames, identical preprocessing path via `arena.model_arena.load_model`. ~2 minutes Mac CPU.

| ckpt | may5 mean / FPR@0.5 | may6 mean / FPR@0.5 | population AUC | r vs deploy_score |
|---|---|---|---:|---:|
| P8A | 0.021 / 0.000 | 0.021 / **0.000** | 0.619 | +0.359 |
| E2B | 0.053 / 0.017 | 0.511 / **0.576** | 0.892 | **+1.000** |
| PA_3800 | 0.054 / 0.000 | 0.269 / 0.163 | 0.883 | +0.757 |

Within-may6 IQ-axis correlations:
| axis | P8A | E2B | PA_3800 |
|---|---:|---:|---:|
| `lap_var_face` | +0.111 | +0.331 | +0.479 |
| `luma_mean` | −0.109 | −0.275 | −0.277 |
| `sat_std` | −0.054 | −0.234 | −0.246 |
| `hf_ratio_face` | +0.111 | +0.230 | +0.362 |

### Probe 6 — `fourier_band_overlap_2026-05-06/`

Per-band single-feature logistic AUC for two tasks:
- Task A (shortcut): may6 (1) vs may5 (0), 152 frames
- Task B (manipulation signal): teams_fake_all_dev (1) vs teams_real_all_dev (0), 400 frames sampled fresh

Per-band AUC (`outputs/per_band_aucs.csv`):
| band | shortcut AUC | signal AUC | Δ |
|---:|---:|---:|---:|
| 0 | 0.655 | 0.629 | +0.03 |
| 1 | 0.691 | 0.471 | +0.22 |
| 2 | 0.846 | 0.693 | +0.15 |
| 3 | 0.913 | 0.732 | +0.18 |
| 4 | 0.845 | 0.738 | +0.11 |
| 5 | 0.628 | 0.721 | -0.09 |
| 6 | 0.614 | 0.696 | -0.08 |
| 7 | 0.889 | 0.677 | +0.21 |
| 8 | 0.981 | 0.659 | +0.32 |
| 9 | 0.981 | 0.638 | +0.34 |
| 10 | 0.903 | 0.616 | +0.29 |
| 11 | 0.827 | 0.589 | +0.24 |
| 12 | 0.967 | 0.523 | +0.44 |
| 13 | 0.974 | 0.456 | +0.52 |
| 14 | 0.825 | 0.543 | +0.28 |
| 15 | 0.632 | 0.545 | +0.09 |

### Probe 7 — `per_layer_p8a_e2b_pa_2026-05-06/`

[CLS] features extracted at all 12 OpenCLIP-B16 transformer resblocks via forward hook. P8A, E2B, PA_3800 on 152 may6+may5 frames + 400 teams_*_dev frames. 5-fold CV logistic regression (balanced, C=1.0) per layer per task.

`outputs/per_layer_aucs.csv` (excerpts):

| layer | P8A shortcut | E2B shortcut | PA shortcut | P8A fake | E2B fake | PA fake |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | 1.000 | 0.888 | 0.889 | 0.888 |
| 1 | 1.000 | 1.000 | 1.000 | 0.965 | 0.963 | 0.958 |
| 2 | 1.000 | 1.000 | 1.000 | 0.986 | 0.984 | 0.982 |
| 4 | 1.000 | 1.000 | 1.000 | 0.995 | 0.996 | 0.996 |
| 6 | 1.000 | 1.000 | 1.000 | 0.995 | 0.994 | 0.993 |
| 8 | 1.000 | 1.000 | 1.000 | 0.998 | 0.996 | 0.997 |
| 9 | 1.000 | 1.000 | 1.000 | 0.998 | 0.998 | 0.998 |
| 11 | 1.000 | 1.000 | 1.000 | 0.996 | 0.995 | 0.996 |

`outputs/per_layer_cosine.csv` — cos(P8A, E2B) on the 152 shortcut-substrate frames:
| layer | cos p10 | cos p50 | frac<0.95 | frac<0.90 | frac<0.80 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.998 | 0.998 | 0% | 0% | 0% |
| 5 | 0.979 | 0.981 | 0% | 0% | 0% |
| 6 | 0.946 | 0.954 | 24% | 0% | 0% |
| 9 | 0.899 | 0.925 | 89% | 11% | 0% |
| 10 | 0.805 | 0.855 | 100% | 97% | 8% |
| 11 | 0.185 | 0.320 | 100% | 100% | 100% |

Cos(E2B, PA_3800) at layer 11: p50=0.772 (PA was FT-from-E2B). Cos(P8A, PA_3800) at layer 11: p50=0.351 (similar magnitude to P8A↔E2B).

## 5. Prior R13 attempts and their bottom-line outcomes (factual ledger)

Each row is an attempted training recipe, what it changed from a baseline, and the dispositive measurement on the promotion contract scorecard (or equivalent). No interpretation column.

| packet | date | base | single-lever delta | scorecard outcome | citation |
|---|---|---|---|---|---|
| RLP1-RLP6 | 04-19 to 04-24 | various | hint lanes, proper-data, arcface margin, gate alignment | RLP6_04 wins value_composite=0.9006 → 2-camera test reveals camera-signature shortcut; loop `shortcut-deployment-block` opens | `packets/RLP{1,2,3,5,6}.md` |
| P8A | 04-25 | RLP7_02 | unfreeze visual.proj + ln_post + apply MLP-SVD | breaks anchor ceiling Δ=−0.188; regresses fake recall −13.6pp aggregate | `packets/P8A.md`, `project_p8a_breakthrough.md` |
| P9-P12 | 04-26 to 04-28 | P8A | recipe tuning (soften, codec_hedge, HEAVY aug); P11 codec_hedge +30% trainer-side anchor | C3 codec_hedge: NOT a Phase D promotion candidate (lockbox readout fails) | `packets/P{9,11,12}.md` |
| P13 | 04-28 | scratch CLIP | anchor-aware loss + pipeline-random + face_scale_jitter@0.25 | γ verdict: cross-domain capability collapsed; viso 5.5%, deeplive 48.8%, modern_v2 FPR 30.2% | `packets/P13.md` |
| P14 (3 sister variants) | 04-29-30 | P8A_step5000 | jitter@0.50 ALONE wins trainer composite 0.661; bundle and DATA_FIX both <0.130 | leader `mclioexb`: zero τ in 5549-pt grid clears contract | `packets/P14.md`, `project_mclioexb_does_not_promote_2026-04-30.md` |
| P15 | 04-30 | bundle | DANN/GRL on quality-domain head, λ=0.20 static | bundle drag; weaker than jitter-isolated | `packets/P15.md` |
| P16 | 04-30 | E2B base | data-axis at fw=2.0 | does NOT promote, ranks 2-9 below P8A | `project_p16_data_axis_does_not_promote_2026-04-30.md` |
| P17 | 05-01 | various | layer-3 readout heads (ArcFace + LINEAR + others) | trained head systematically destroys substrate-invariance by step ~1000 | `project_p17_trained_head_destroys_substrate_invariance.md` |
| P18 | 05-01-02 | E2B | 12-class method-conditional GRL | corrective probe verdict: GRL preserves FT-induced regression, doesn't add invariance; P8A still wins | `project_p18_diagnostics_complete_2026-05-02.md`, `project_p18_d_contract_p8a_wins_no_floor.md` |
| P22 | 05-02 | E2B | pipeline_randomization aug curriculum | step1k robust winner; step8k score variance collapsed 140×; ensemble (P8A + P22 step1k min) gives 18× viso, 51× deeplive | `project_p22_cpu_followups_reframe_2026-05-02.md` |
| S1/S2/S3 | 05-03 | P22 step1k | training-cap, earlier base, viso fw=8.0 — single-lever P22 variants | viso ceiling unbroken; S2 step600 wins teams_fake_lockbox at 91.5% (joint FPR=10%) | `project_s1_s2_s3_2026-05-03.md` |
| E1/E2B/E3 | 05-03 | scratch + CE | B16 scratch (E2B), L14 scratch (E3) | E2B breaks deeplive ceiling (87.5% recall@FPR=10%) but viso REGRESSES 27%→7%; L14 doesn't break viso | `project_e2b_breaks_deeplive_ceiling.md`, `project_l14_does_not_break_viso_ceiling.md` |
| Job 7 | 05-04 | P8A frozen features | head-only retrain (6 head variants) | all 6 over-fire 80-92% on lockbox reals; refuted | `project_job7_head_retrain_REFUTED_2026-05-04.md` |
| PA | 05-04-05 | E2B | visomaster_enhanced + visomaster_teams_enhanced data at fw=4.0 | F4 v2 viso 72.4% (best in R13); HDTF cross-substrate 7.87% (collapse) — substrate-bound; doesn't generalize | `packets/PA.md`, `project_pa_does_not_generalize_to_hdtf_2026-05-05.md` |
| PC | 05-04-05 | E2B | data + Teams codec aug | codec aug HURTS viso recall by 35-50pp on F4 vs PA | `packets/PC.md`, `project_pc_codec_aug_hurts_viso_2026-05-05.md` |
| PD (in flight) | 05-05-06 | E2B | correlation_penalty (Pearson) on sharpness/luma/face_area, λ=1.0 | scorecard running 2026-05-06 (`pd-corr-penalty-scorecard-2026-05-06`); training-half complete; verdict pending | `packets/PD.md`, `threads/correlation_penalty_loss.md` |

Cross-cutting bug + caveat list to read before interpreting any pre-2026-04-26 number:
- `INTER_AREA → INTER_LINEAR` preprocessing parity bug (memory `project_p8a_breakthrough.md` cites the WS-P0 fix at commit `855871e`; pre-fix retro-scores have silent kernel drift).
- `apply_svd_to_in_proj` silent zero-gradient bug pre-`2feea58` 2026-04-26 (memory `project_in_proj_svd_gradient_bug.md`).
- `quality_enhancement` family-routing bug: pre-2026-05-05 R13 packets trained on ~5,120 / ~18,880 deeplive_enhanced_fake frames (~27%) mislabeled (memory `project_quality_enhancement_routing_2026-05-05.md`). PD is the first post-fix packet.
- Contract-policy v3 fix (recall floor) is in working tree but uncommitted (open loop `contract-policy-bug-fix-not-committed`).

## 6. Currently open structural questions (with explicit non-resolution)

These are open questions in the `OPEN_LOOPS.md` register. Each links to its owning thread:

- `shortcut-deployment-block` (critical, in-progress) — `dor-real-webcam-false-flag-no-virtual-bg ≤ 0.30` AND `lockbox_fake_recall ≥ 0.60` at single τ holding `teams_ood_real` FPR ≤ 5%. Not met by any candidate to date.
- `corr-penalty-deployment-grade-verdict-pending` (high, in-progress) — PD scorecard verdict on 4/4 close criterion (lockbox recall ≥ 90% at FPR ≤ 10%; shortcut weakening ≥ 30% on ≥ 2 of 5 axes; no untargeted axis +50%; HDTF cross-substrate FPR ≤ 5%).
- `corr-penalty-frozen-head-shifting-axes-not-targeted` (medium) — frozen-head prototype showed shifting onto `face_area` and `is_webcam`; `face_area` added to encoder-FT penalty axes; `is_webcam` not deployable.
- `deployment-vs-p8a-substrate-tradeoff-not-quantified` (high, opened today) — written quantification of substrate-classes where P8A vs E2B trade off, with disposition.
- `face-size-label-leak` (high) — flip rate from face-size-targeted intervention not yet ≤ 10%.
- `eval-production-crop-tightness-mismatch` (high) — eval substrate has looser crop than production; unquantified delta.
- `sharpness-metric-computed-on-full-image-not-face` (high) — full-image Laplacian, not face-crop; downstream FPR-by-quartile reports partially confounded.
- `frame-level-vs-clip-level-scorer-mismatch` (medium) — clip-level recall vs frame-level AUC reconciliation under corrected contract policy.
- `enhanced-vs-unenhanced-val-pool-confound` (medium, in-progress) — enhanced-proper validation pool composition + dose matching.
- `quality-enhancement-strategy-misrouted-fix-pending` (medium, in-progress) — fix landed 2026-05-05 in code; image rebuilt; cross-packet contamination caveat applies to all pre-fix packets.

Full list with close criteria: `docs/packet_retrospectives/OPEN_LOOPS.md` (29 open / 5 in-progress / 10 resolved / 1 superseded after today's regenerate).

## 7. What today's measurements specifically do AND do not say (caveats)

- **Probes 1-7 are all measurements, not interventions.** They say what is true of fixed ckpts on fixed data. They do not say what would happen if a new training recipe were applied.
- **Identity overlap matters.** `teams_*_dev` substrates have identity overlap with training. Frame-level AUC numbers on those substrates are partially identity-leakage; the layer-by-layer trajectory and the relative ordering are the meaningful reads, not absolute peak values.
- **Pair feature analysis (Probe 4) used cached P8A features only** — E2B and PA frozen features are not cached. Cross-encoder claims (E2B's pair geometry) are inferred from score-side measurements, not measured directly on E2B encodings.
- **The Fourier-band probe (Probe 6) uses single-feature univariate AUC**, not multivariate. A model has access to combinations; "band X has low signal AUC alone" doesn't imply "removing band X amplitude during training preserves the model's manipulation discrimination." This is a univariate measurement that bounds, but does not predict, the effect of an intervention.
- **The shortcut readability at AUC=1.0 at every layer (Probe 7) is on a small substrate (152 frames) with high feature dimensionality (768)**. The value is consistent across all 12 layers and all 3 ckpts, which makes regularization-driven overfitting an unlikely explanation, but a larger substrate would tighten the claim.

## 8. Pointers to raw artifacts the reviewer should consult

- All probe outputs: `analysis/{xinhe_cross_camera_audit,dor_drift_mechanism,amp_vs_phase_probe,paired_feature_consistency,fourier_band_overlap,per_layer_p8a_e2b_pa}_2026-05-06/outputs/*.csv,*.json`
- Identity-browser dataset: `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv` (with HTML at `index.html`)
- TIMELINE: `docs/packet_retrospectives/TIMELINE.md` — append-only chronological master index
- OPEN_LOOPS: `docs/packet_retrospectives/OPEN_LOOPS.md` — mechanically generated; do not hand-edit
- Cross-cutting threads: `docs/packet_retrospectives/threads/*.md`
- Per-packet retros: `docs/packet_retrospectives/packets/*.md`
- Prior R13 plan log: `april-26-training-master-plan-v2.LOG.md`

## What the reviewer should explicitly NOT read in the first pass

To preserve independent judgment:

- `docs/relaunch_handoffs/ANTI_SHORTCUT_TECHNIQUES_RANKED_2026-05-06.md` — contains a 1-9 ranked list of next-step recommendations. Read AFTER forming your own view.
- `docs/packet_retrospectives/threads/processing_signature_shortcut.md` — the **2026-05-06 afternoon update** sub-section contains "Operational implications" sub-blocks under Probes 3, 4, 6, 7 and a "Joint reading across the four probes" section with explicit recipe recommendations. The data tables in those sub-sections are factually clean; the prose around them is interpretive.
- Any `FINDINGS.md` file inside the probe directories (only Probes 3 and 7 have one) — they synthesize and recommend.
- Memory entries: the 2026-05-06 entries have been refactored to facts + open questions, but the older entries (pre-2026-05-06) carry various prior agents' interpretations. They're useful as historical context and citation, but they are not bias-free.

## Reviewer's question

Given the data above and the user's stated three-pillar goal (fake recall ≥ 90% on target methods AND real-side FPR ≤ 5% AND robustness across capture conditions), what would you propose as the next 1-3 packets and why? Show your reasoning. Be explicit about:
- What you'd run first and what verdict would flip the plan.
- Confidence interval on whether the resulting model would generalize across the substrates listed in §2.
- What information you'd want that isn't in this pack (so the user can prioritize a probe to gather it).
- Anything in §5 (prior R13 attempts) you think we should revisit with hindsight from §3-4.

You're asked to be honest about uncertainty and to flag assumptions.
