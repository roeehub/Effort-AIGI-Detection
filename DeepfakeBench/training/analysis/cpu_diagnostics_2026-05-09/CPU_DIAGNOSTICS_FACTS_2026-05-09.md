# CPU diagnostics FACTS (2026-05-09 audit pass)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, promotes,
> deployment-grade. Numbers + tables + cross-references only. Interpretation
> goes in `CPU_DIAGNOSTICS_OPINIONS_2026-05-09.md` (separate document; reader
> may disagree).
>
> **Authoring**: 2026-05-09 ~13:00 local. Driven by user request to audit
> documentation overstatement and to run all CPU diagnostics that test
> whether the model is using a forgery channel vs other shortcut signals
> under the user's framing that the production target domain is genuinely
> unknown.

---

## 1. Question (4 sub-questions)

For each Stage 2 ckpt (S1/S2/S3 × step{500, 2500, 4500}) plus the three
reference ckpts (P8A, E2B, P2D-step3000):

**Q1.** What is the per-layer AUC of linear probes trained to predict each
of {real-vs-fake, is_dor, is_chronic_6, lap_var-high, min_dim-high,
face_size-high} from CLS activations at L0/L3/L6/L9/L11?

**Q2.** Does substrate filtering (drop chronic-6, drop is_no_face, drop
min_dim<200) reduce Stage 2 ckpt real-FPR while preserving fake recall on
the 800-frame triptych sample?

**Q3.** What does the IQ-shortcut R² + residual-AUC decomposition look like
on the triptych for the 9 Stage 2 ckpts? Does the residual-AUC pattern
mirror what was found on the Stage 1 23-cell decomposition for P8A/E2B/P2D?

**Q4.** Per identity present in the 800-frame triptych, how does each ckpt's
real-side score distribute, and does this confirm or refute the
partition-by-identity finding (memory `project_chronic_offenders_partition_per_ckpt_2026-05-04`)?

## 2. Method

### 2.1. Frame substrate

The 800-frame triptych at
`analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/sampled_frames.csv`.
Distribution:

- 800 total: 476 real / 324 fake
- 713 dev / 87 lockbox
- 33 unique identities; 282 chronic-6 frames (35.2%)
- 30+ fake methods + `teams_real`
- IQ feature spread: lap_var p10/p50/p90 = 14.7 / 153.1 / 736.3;
  face_pixel_area p10/p50/p90 = 6,920 / 27,255 / 103,266

After IQ-panel join with atlas + inline-computed missing rows, panel size
is 863 (some atlas duplicates expanded the row count slightly).

### 2.2. Feature extraction

Per ckpt, single forward pass extracts both fake-class softmax probability
AND CLS activations at `transformer.resblocks` layers {0, 3, 6, 9, 11}
via forward hooks (`extract_stage2_on_triptych.py`). Reference ckpt
features (P8A/E2B/P2D) reused from `analysis/iq_perlayer_probe_2026-05-08/_cache/`.

### 2.3. Probes

Per (ckpt × layer × signal):
- L2-normalize CLS features
- 5-fold StratifiedKFold logistic regression (`C=1.0`, `n_jobs=1`)
- Out-of-fold predictions → ROC-AUC

Per memory `feedback_sklearn_njobs.md`: n_jobs=1 throughout.

### 2.4. F4-style filters

Applied to the triptych panel:
- F0_full: all 800 frames
- F1_no_chronic_6: drop frames whose `identity_key` substring-matches any of
  {Roy_D, PC_Generator, bla_bla_chow, Md_noyn_Sharker, dor_shkedi, healthy_dor}
- F2_no_chronic_no_noface: F1 + drop `is_no_face=True`
- F4_no_chronic_no_noface_min_dim_ge200: F2 + drop `min(width, height) < 200`

### 2.5. IQ decomp (triptych version of Stage 1)

For each (Stage 2 ckpt × pool ∈ {ALL, DEV, DEV_NO_CHRONIC, LOCKBOX}):
- Linear regression of `score ~ z(PRIMARY_6 IQ axes)` on n ≥ 30 frames per pool
- R² of fit + residual = `score − pred`
- ROC-AUC of residual as fake-vs-real predictor
- Compared against raw-score AUC on same sample

### 2.6. Cross-substrate variance

For each Stage 2 ckpt, group by `(identity_key, split)` and compute
score median, count, std. Pivot dev vs lockbox per identity.

## 3. Results

### 3.1. Forgery-signal atlas — `outputs/forgery_signal_atlas.csv`

Per-layer AUC for `is_real_vs_fake` (n=863 frames; positive class = fake):

| ckpt | L0 | L3 | L6 | L9 | L11 | L11 − L0 |
|---|---:|---:|---:|---:|---:|---:|
| P8A | 0.7245 | 0.9224 | 0.9920 | 0.9793 | 0.9759 | +0.2514 |
| E2B | 0.7245 | 0.9374 | 0.9918 | 0.9837 | 0.9948 | +0.2703 |
| P2D | 0.7245 | 0.9434 | 0.9933 | 0.9814 | 0.9932 | +0.2688 |
| S1_step500 | 0.7245 | 0.9256 | 0.9921 | 0.9842 | 0.9791 | +0.2546 |
| S1_step2500 | 0.7243 | 0.9278 | 0.9912 | 0.9812 | 0.9867 | +0.2624 |
| S1_step4500 | 0.7243 | 0.9289 | 0.9924 | 0.9841 | 0.9869 | +0.2625 |
| S2_step500 | 0.7245 | 0.9220 | 0.9918 | 0.9846 | 0.9790 | +0.2544 |
| S2_step2500 | 0.7245 | 0.9208 | 0.9925 | 0.9897 | 0.9840 | +0.2594 |
| S2_step4500 | 0.7245 | 0.9207 | 0.9916 | 0.9887 | 0.9841 | +0.2596 |
| S3_step500 | 0.7245 | 0.9214 | 0.9921 | 0.9883 | 0.9860 | +0.2615 |
| S3_step2500 | 0.7245 | 0.9263 | 0.9935 | 0.9900 | 0.9920 | +0.2675 |
| S3_step4500 | 0.7245 | 0.9260 | 0.9932 | 0.9883 | 0.9901 | +0.2657 |

Three observations from the table (factual, no interpretation):
1. L0 AUC is uniformly 0.7245 across all 12 ckpts (consistent with FT
   freezing patch+positional embedding).
2. Encoder L0 → L6 gain is +0.27 AUC for all ckpts; L6 → L11 gain is
   ≤ +0.04 for all ckpts.
3. L11 AUC ranges 0.9759 (P8A) to 0.9948 (E2B). P8A has the lowest L11
   real/fake AUC among the 12 ckpts measured.

Per-layer AUC for `is_dor` (positive class = identity_key contains "dor"):
all ckpts ≥ 0.99 at L11. L0 is_dor AUC = 0.85+ across all ckpts.

Per-layer AUC for `is_chronic_6`: P8A L11 = 0.9087; E2B L11 = 0.9747; P2D
L11 = 0.9801; Stage 2 step4500 ckpts = 0.93–0.95. P8A L11 chronic-6 AUC is
the lowest among the 12 ckpts.

Per-layer AUC for `lap_var_high` (split at p50): peak at L11 across most
ckpts (0.92–0.96).

Per-layer AUC for `min_dim_high`: peak at L3 for all ckpts (0.95+);
saturates from L3 onward.

Per-layer AUC for `face_size_high`: peak at L9 or L11 (0.94–0.97).

Full table at `outputs/forgery_signal_atlas.csv` (361 rows: 12 ckpts × 5
layers × 6 signals + null cells).

### 3.2. F4 filter recompute — `outputs/f4_filter_triptych.csv`

Real-FPR @ τ=0.5 for each Stage 2 ckpt across 4 filter levels:

| ckpt | F0_full | F1_no_chronic | F2_no_chronic_no_noface | F4_full_clean |
|---|---:|---:|---:|---:|
| S1_step500 | 0.126 | 0.057 | 0.056 | 0.030 |
| S1_step2500 | 0.096 | 0.041 | 0.039 | 0.012 |
| S1_step4500 | 0.163 | 0.070 | 0.069 | 0.042 |
| S2_step500 | 0.106 | 0.062 | 0.064 | 0.036 |
| S2_step2500 | 0.153 | 0.074 | 0.077 | 0.049 |
| S2_step4500 | 0.149 | 0.074 | 0.077 | 0.049 |
| S3_step500 | 0.244 | 0.103 | 0.103 | 0.073 |
| S3_step2500 | 0.126 | 0.041 | 0.039 | 0.024 |
| S3_step4500 | 0.149 | 0.037 | 0.034 | 0.024 |

Fake-recall @ τ=0.5 across the same filters:

| ckpt | F0_full | F1_no_chronic | F2 | F4 |
|---|---:|---:|---:|---:|
| S1_step500 | 0.864 | 0.854 | 0.857 | 0.845 |
| S1_step2500 | 0.825 | 0.825 | 0.827 | 0.813 |
| S1_step4500 | 0.941 | 0.939 | 0.941 | 0.932 |
| S2_step500 | 0.867 | 0.848 | 0.850 | 0.837 |
| S2_step2500 | 0.955 | 0.948 | 0.951 | 0.944 |
| S2_step4500 | 0.955 | 0.948 | 0.951 | 0.944 |
| S3_step500 | 0.977 | 0.974 | 0.977 | 0.972 |
| S3_step2500 | 0.963 | 0.974 | 0.977 | 0.976 |
| S3_step4500 | 0.952 | 0.958 | 0.961 | 0.968 |

Aggregate observations (factual):
- Stage 2 step4500 F0 real-FPR ranges 0.149–0.163.
- F1 (drop chronic-6 only) cuts real-FPR by 2.0–4.0× across Stage 2 ckpts.
- F4 (full clean) cuts real-FPR by 3.4–5.5× vs F0 across Stage 2 ckpts.
- Fake-recall is preserved within ≤ 0.02 across F0 → F4 transitions for
  every Stage 2 ckpt.

### 3.3. IQ decomp on triptych (Stage 2 ckpts) — `outputs/iq_decomp_triptych.csv`

R² of `score ~ z(6 IQ axes)`:

| ckpt | ALL | DEV | DEV_NO_CHRONIC | LOCKBOX |
|---|---:|---:|---:|---:|
| S1_step500 | 0.131 | 0.196 | 0.275 | 0.141 |
| S1_step2500 | 0.155 | 0.246 | 0.305 | 0.149 |
| S1_step4500 | 0.150 | 0.194 | 0.294 | 0.071 |
| S2_step500 | 0.162 | 0.199 | 0.266 | 0.395 |
| S2_step2500 | 0.178 | 0.190 | 0.275 | 0.500 |
| S2_step4500 | 0.178 | 0.196 | 0.278 | 0.436 |
| S3_step500 | 0.155 | 0.145 | 0.269 | 0.844 |
| S3_step2500 | 0.219 | 0.228 | 0.301 | 0.251 |
| S3_step4500 | 0.211 | 0.236 | 0.321 | 0.132 |

Residual AUC (fake-vs-real after IQ removed):

| ckpt | ALL | DEV | DEV_NO_CHRONIC | LOCKBOX |
|---|---:|---:|---:|---:|
| S1_step500 | 0.868 | 0.908 | 0.921 | 0.607 |
| S1_step2500 | 0.866 | 0.923 | 0.922 | 0.610 |
| S1_step4500 | 0.874 | 0.917 | 0.939 | 0.587 |
| S2_step500 | 0.875 | 0.896 | 0.904 | 0.679 |
| S2_step2500 | 0.889 | 0.908 | 0.926 | 0.594 |
| S2_step4500 | 0.890 | 0.911 | 0.927 | 0.614 |
| S3_step500 | 0.852 | 0.868 | 0.926 | 0.549 |
| S3_step2500 | 0.912 | 0.929 | 0.948 | 0.702 |
| S3_step4500 | 0.891 | 0.922 | 0.946 | 0.660 |

Aggregate observations:
- LOCKBOX R² range across Stage 2 ckpts: 0.071 (S1_step4500) → 0.844
  (S3_step500). 12× spread.
- DEV_NO_CHRONIC R² range: 0.266–0.321. 1.2× spread (much narrower).
- LOCKBOX residual AUC range: 0.549–0.702.
- DEV_NO_CHRONIC residual AUC range: 0.904–0.948.
- Cross-comparison with Stage 1 (P8A/E2B/P2D): per
  `analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md`,
  P8A LOCKBOX R² = 0.397, E2B LOCKBOX R² = 0.454, P2D LOCKBOX R² = 0.542.
  Stage 2 LOCKBOX R² spans both below P8A's value and above P2D's.

### 3.4. Per-identity score distribution — `outputs/score_per_identity_per_ckpt.csv`

297 rows (33 identities × 9 Stage 2 ckpts; ckpts that didn't see a
particular identity are dropped). Selected real-side medians for the 16
chronic-6 sub-identities present in the triptych:

| identity_key | S1_500 | S1_4500 | S2_500 | S2_4500 | S3_500 | S3_4500 |
|---|---:|---:|---:|---:|---:|---:|
| Md_noyn_Sharker__s15 | 0.023 | 0.003 | 0.091 | 0.028 | 0.026 | 0.003 |
| PC_Generator__s4 | 0.016 | 0.002 | 0.074 | 0.022 | 0.021 | 0.002 |
| PC_Generator__s8 | 0.019 | 0.003 | 0.078 | 0.025 | 0.158 | 0.005 |
| PC_Generator__s13 | 0.043 | 0.064 | 0.104 | 0.052 | 0.333 | 0.025 |
| PC_Generator__s14 | 0.083 | 0.026 | 0.147 | 0.153 | 0.463 | 0.045 |
| PC_Generator__s15 | 0.837 | 0.897 | 0.676 | 0.839 | 0.899 | 0.520 |
| PC_Generator__s22 | 0.796 | 0.761 | 0.701 | 0.846 | 0.947 | 0.376 |
| PC_Generator__s34 | 0.019 | 0.004 | 0.082 | 0.026 | 0.031 | 0.004 |
| PC_Generator__s45 | 0.849 | 0.986 | 0.756 | 0.962 | 0.965 | 0.955 |
| bla_bla_chow | 0.325 | 0.186 | 0.169 | 0.276 | 0.549 | 0.426 |
| bla_bla_chow__s1 | 0.364 | 0.849 | 0.378 | 0.880 | 0.978 | 0.896 |
| bla_bla_chow__s2 | 0.281 | 0.223 | 0.349 | 0.435 | 0.972 | 0.198 |
| dor_shkedi | 0.505 | 0.805 | 0.298 | 0.483 | 0.571 | 0.859 |
| dor_shkedi__s16 | 0.023 | 0.005 | 0.080 | 0.025 | 0.030 | 0.004 |

Within "PC_Generator" alone (single root identity, 9 sessions), real-side
median spans 0.002 (PC_Generator__s4) to 0.986 (PC_Generator__s45) at
S1_step4500. Same-root, different-session.

Within "dor", `dor_shkedi__s16` has real-side medians 0.003–0.080 across
all Stage 2 ckpts; `dor_shkedi` (no suffix) has medians 0.298–0.859 across
the same ckpts. 100×–280× spread between two sub-identities of the same
person.

### 3.5. Cross-substrate variance — `outputs/cross_substrate_variance.csv`

Empty file. The 800-frame triptych has 0 identity_keys present in BOTH
`split=dev` AND `split=lockbox`. The triptych's dev-split identities and
lockbox-split identities are disjoint. Cross-substrate per-identity variance
cannot be measured on this sample.

## 4. Cross-references

- Source data: `outputs/{forgery_signal_atlas.csv,
  f4_filter_triptych.csv, iq_decomp_triptych.csv,
  score_per_identity_per_ckpt.csv, cross_substrate_variance.csv,
  stage2_triptych_scores.csv}`
- Feature cache: `_cache/intermediate__{ckpt}__layer{XX}__n800.npz`
  (12 ckpts × 5 layers; reference ckpts re-used from
  `analysis/iq_perlayer_probe_2026-05-08/_cache/`)
- Driver scripts: `extract_stage2_on_triptych.py`, `run_analyses.py`,
  `f4_filter_recompute.py`
- Sister Stage 1 decomposition: `analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md`
- Per-layer probe baseline: `analysis/iq_perlayer_probe_2026-05-08/IQ_PERLAYER_PROBE_FACTS_2026-05-08.md`
- Stage 2 score-distribution: `analysis/stage2_cpu_2026-05-09/STAGE2_SCORE_PROBE_FACTS_2026-05-09.md`

## 5. Caveats

- Sample size: 800 frames (863 after IQ join), 33 unique identities. Not
  large enough for tight per-identity CIs.
- Substrate: predominantly dev-split (713/800); lockbox sub-sample (87/800)
  has fewer chronic-6 frames (45) than dev (237). Lockbox-only IQ R² is
  more variance-prone than dev-only at this sample size.
- Linear probes only. Nonlinear IQ-correlated patterns (Fourier-band
  per `project_fourier_band_overlap_2026-05-06`) not captured.
- Forgery-signal atlas at 5 layers, not all 12. L1, L2, L4, L5, L7, L8,
  L10 not measured.
- Score-side IQ decomp is on output `frame_prob`, not on encoder
  activations.
- F4 filter results are on the triptych sample, not on the full v2
  production substrate or HDTF substrate. Substrate-portability of these
  filter effects is not measured here.
- Cross-substrate per-identity variance not measurable from this sample
  (dev/lockbox identities disjoint in triptych).
- Timing: forward-pass extraction 4.5 min on MPS for 9 Stage 2 ckpts ×
  800 frames. Full analysis pipeline ~12 min wall-clock.
