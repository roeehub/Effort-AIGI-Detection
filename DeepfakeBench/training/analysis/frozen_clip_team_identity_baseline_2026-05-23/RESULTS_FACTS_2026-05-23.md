# Frozen-CLIP team-identity baseline — RESULTS FACTS

Generated 2026-05-23. Factual readout only (per `docs/packet_retrospectives/AGENTS.md` §"Eval-folder authoring contract" — forbidden words: "succeeds", "fails", "wins", "promotes", "deployment-grade", "breakthrough", "ceiling-breaking"). Interpretive content lives in `AGENT_PROPOSAL_2026-05-23.md`.

> **Question**: Does a properly-trained linear/MLP head on frozen-OpenCLIP-B16-DataComp.XL L11 features approximate the 5 FT'd ckpts (P8A, E2B, T5C, Slot A v2 CLS, Slot A v2 face-pool) on the team-identity deploy cohort?
>
> **Inputs**: 5,941 deploy-relevant team-identity frames (1,821 real + 4,120 fake-attack across 5 humans: Roee_Windows / dor / Noyn / Xiang / Xinhe) from the expanded readout at `analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_frame_full.csv`.

---

## 0. Scope and method

### 0.1 Encoder
OpenCLIP ViT-B-16 DataComp.XL (`weights/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin`), forward hook on `visual.transformer.resblocks[11]`, CLS token (index 0). Preprocessing: 224x224 cv2 INTER_LINEAR resize + CLIP mean/std normalize, MPS-batched at 32.

This is the same encoder all FT'd ckpts (P8A, E2B, T5C, Slot A v2) started from, before fine-tuning their final-block + projection + linear head.

### 0.2 Training corpora
- **DEV** (Option A, primary): 2,000 dev_real + 2,000 dev_fake sampled per D8's seed=42 protocol from `analysis/lockbox_tagging/full_tags_2026-04-27.parquet`. Same exact sample as D8.
- **DEV+LB** (informational upper bound): DEV + 414 lockbox_real + 425 lockbox_fake. The 839 lockbox frames are NOT in the team-identity deploy cohort (lockbox is a separate eval substrate), so this is a fair stronger training pool that adds the lockbox-domain reals to the head's training data.
- **OPTB** (Option B): 3,000 real + 3,000 fake stratified-by-method sample of the actual training manifest `frame_properties.parquet` (1,409,040 rows; deep-live-cam, simswap, neuraltextures, FF++ variants, AVSpeech, celeb_synthesis, ...). Random seed=42.
- **OPTB+DEV**: OPTB ∪ DEV.

### 0.3 Heads
- **LR**: `LogisticRegression(C=1.0, max_iter=2000, n_jobs=1, solver='lbfgs')` per D8's pattern.
- **MLP**: `MLPClassifier(hidden_layer_sizes=(256,), max_iter=200, early_stopping=True, n_iter_no_change=10, validation_fraction=0.15, random_state=42)`.
- Constraints: `n_jobs=1` per `feedback_sklearn_njobs.md`; deterministic seed=42; sklearn CPU only (MPS only for feature extraction).

### 0.4 τ-modes
Two complementary calibration schemes:
- **Cross-ckpt constant-τ** (per `project_deployment_three_modes_slot_a_v2_2026-05-21`): mode A 0.535, mode B 0.78, mode C 0.87. Inherits the expanded-readout's caveat that these are calibrated for Slot A v2 CLS-pool; the same numeric τ across ckpts is NOT constant FPR.
- **Per-head team-FPR-calibrated** (apples-to-apples): τ set such that fraction(team-real ≥ τ) = {0.02, 0.05, 0.10}. Each head's τ is computed on the 1,821 team-real frames. This is the cleanest cross-head comparison.

### 0.5 Decoupling note: training pool vs eval pool overlap
- DEV pool (4,000 frames sampled from `teams-faces-data-test-2914-fake-4420-real-feb-28`) shares the bucket with parts of the team-identity real-cohort (Noyn 210 real + some dor cohorts). The exact frame-level overlap was not deduplicated. This is a leakage caveat for the LR_DEV / MLP_DEV results — see §6.
- LB pool (839 frames, mostly `teams_capture_cam_test_s33/35` per `project_lockbox_tagging`) shares NO frames with the team-identity cohort (different videos, different sessions).
- OPTB pool (training corpus, e.g. AVSpeech / celeb_synthesis / FF++ / deep-live-cam) has NO known frame-level overlap with the team-identity cohort; it covers different sources entirely.

---

## 1. Coverage

### 1.1 CLIP feature extraction (team-identity)

| Quantity | Value |
|---|---:|
| Team-identity cohort total (deploy_relevant=True) | 5,941 frames |
| Local-resolved | 1,547 frames |
| GCS-downloaded | 4,394 frames |
| Extraction failures | 0 |
| Extraction wall time | 1,053 sec (~17.5 min) |
| Cache file | `outputs/clip_frozen_l11__team_identity_n5941.npz` (16.9 MB) |

### 1.2 CLIP feature extraction (Option B training corpus)

| Quantity | Value |
|---|---:|
| OPTB sample requested | 3,000 real + 3,000 fake |
| Stratified by method (frame_properties.parquet 1.4M rows) | seed=42 |
| Extracted | 6,000 frames (3,000 real + 3,000 fake) |
| Failures | 0 |
| Extraction wall time | 1,753 sec (~29 min) |
| Cache file | `outputs/clip_frozen_l11__training_optionB_n6000.npz` (17.2 MB) |

OPTB method breakdown (top 10 per-side): real-side dominated by `external_youtube_avspeech`, `celeb_synthesis`, `phase1_real`, `dfdc_real`, `real_social_12_09`, `vcd`; fake-side dominated by `deep-live-cam_fake`, `celeb_synthesis`, `dfdc_fake`, `simswap`, `inswap`, `faceswapff`, `neuraltextures`, `deepfakes`, `face2face`, `faceshifter`, etc.

### 1.3 D8 cache reuse

D8's `outputs/clip_frozen_l11__n4839.npz` (4,000 dev + 839 lockbox CLIP-frozen L11 features) was reused without re-extraction. Same OpenCLIP B16 DataComp.XL weights, same preprocessing.

---

## 2. Head training summary

Source: `outputs/training_summary_v2.csv`.

| Head | Corpus | n_train | Train AUC | Dev AUC | Lockbox AUC | τ@dev5% | LB FPR@τ | LB recall@τ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LR_DEV | DEV | 4,000 | 1.0000 | 1.0000 | 0.7829 | 0.0201 | 0.8309 | 1.0000 |
| MLP_DEV | DEV | 4,000 | 1.0000 | 1.0000 | 0.6078 | 0.0587 | 0.8188 | 0.9741 |
| LR_DEV_LB | DEV+LB | 4,839 | 1.0000 | 1.0000 | 1.0000 | 0.0194 | 0.0242 | 1.0000 |
| MLP_DEV_LB | DEV+LB | 4,839 | 1.0000 | 1.0000 | 1.0000 | 0.0491 | 0.0145 | 1.0000 |
| LR_OPTB | OPTB | 6,000 | 0.9783 | 0.8183 | **0.3452** | 0.8570 | 0.5628 | 0.1176 |
| MLP_OPTB | OPTB | 6,000 | 0.9940 | 0.7969 | **0.5162** | 0.8620 | 0.5411 | 0.5059 |
| LR_OPTB_DEV | OPTB+DEV | 10,000 | 0.9876 | 0.9997 | **0.4560** | 0.1087 | 0.7222 | 0.9365 |
| MLP_OPTB_DEV | OPTB+DEV | 10,000 | 0.9986 | 1.0000 | **0.3661** | 0.0011 | 0.9058 | 0.9741 |

### 2.0.1 OPTB transfer-AUC observation

OPTB heads (LR / MLP trained on 6K training-corpus frames) get **lockbox AUC 0.345–0.516** — WORSE than DEV heads (0.61–0.78) and worse than chance for LR_OPTB. OPTB+DEV combined heads (10K total) also degrade vs DEV alone (lockbox AUC 0.36–0.46). The OPTB training data (DF40 / FF++ / deep-live-cam / AVSpeech) does not transfer to the Teams-substrate lockbox distribution; adding it to DEV ACTIVELY HURTS the head's lockbox transfer because the larger OPTB pool dominates the loss.

### 2.1 Cross-check vs D8
D8's `P8A_dev_unweighted` (LR on P8A's trained-encoder L11 features) had DEV→LOCKBOX transfer AUC 0.5865. This run's `LR_DEV` (LR on frozen-CLIP L11 features, identical sample) has DEV→LOCKBOX transfer AUC **0.7829**. Frozen-CLIP features deliver +0.19 AUC over the trained-P8A-encoder features on the same train/test split. D8's strongest weighted reference (estimator-C-KLIEP) was 0.6792; frozen-CLIP unweighted matches/exceeds the strongest D8 weighted head.

---

## 3. Joint-calibrated comparison @ team-real-FPR = 5%

Source: `outputs/joint_calibrated_summary.csv`. Per-head τ set so that fraction(team-real ≥ τ) = 0.05 on the 1,821 deploy-relevant real frames. Fake recall reported per-human on fake-attack cohorts (dor n=2,443; Xinhe n=1,099; Xiang n=578). Higher recall = better at catching attacks at the same effective real-side FPR.

Ranked by `min_recall` descending (the binding fake-side gate per `project_team_identities_multi_labeled_2026-05-23`).

| Head | Mean real prob | τ | dor recall | Xinhe recall | Xiang recall | min recall | mean recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| P8A | 0.047 | 0.239 | 0.870 | 0.786 | 0.936 | **0.786** | 0.864 |
| E2B | 0.067 | 0.455 | 0.700 | 0.813 | 0.945 | 0.700 | 0.819 |
| SlotAv2_FACE | 0.445 | 0.653 | 0.562 | 0.905 | 0.988 | 0.562 | 0.818 |
| SlotAv2_CLS | 0.140 | 0.461 | 0.622 | 0.546 | 0.964 | 0.546 | 0.710 |
| T5C (production) | 0.210 | 0.657 | 0.759 | 0.379 | 0.948 | 0.379 | 0.695 |
| MLP_DEV_LB | 0.205 | 0.913 | 0.527 | 0.205 | 0.465 | 0.205 | 0.399 |
| MLP_DEV | 0.274 | 0.975 | 0.292 | 0.160 | 0.159 | 0.160 | 0.204 |
| LR_DEV_LB | 0.203 | 0.975 | 0.422 | 0.153 | 0.381 | 0.153 | 0.318 |
| **LR_DEV** | 0.255 | 0.993 | 0.289 | 0.055 | 0.197 | **0.055** | 0.180 |
| MLP_OPTB_DEV | 0.212 | 0.999 | 0.171 | 0.037 | 0.303 | 0.037 | 0.170 |
| LR_OPTB_DEV | 0.332 | 0.995 | 0.160 | 0.018 | 0.268 | 0.018 | 0.149 |
| MLP_OPTB | 0.296 | 0.986 | 0.092 | 0.014 | 0.005 | 0.005 | 0.037 |
| LR_OPTB | 0.382 | 0.993 | 0.071 | 0.002 | 0.000 | **0.000** | 0.024 |

### 3.0.1 OPTB head observation

Heads trained on the actual training corpus (OPTB; 6K stratified-by-method sample of `frame_properties.parquet`) perform STRICTLY WORSE than DEV-trained heads on every per-human metric. LR_OPTB gets 0.0% Xiang recall and 0.2% Xinhe recall at 5% team-FPR; MLP_OPTB gets 0.5% Xiang and 1.4% Xinhe. OPTB+DEV combined (10K) is also strictly worse than DEV alone (LR_OPTB_DEV min recall 0.018 vs LR_DEV 0.055).

This is because the OPTB training data (DF40 / FF++ / deep-live-cam / AVSpeech) covers different fake-method families than the Teams-substrate cohorts (`live_prod__xinhe-fake-{1..11}`, `Xiang_Xiang2_Feng_fake`, etc.). The frozen-CLIP head trained on OPTB learns FF++-style separators that don't transfer to Teams-substrate attacks. Adding more diverse OPTB data ACTIVELY HURTS team-identity transfer because the OPTB loss dominates the smaller DEV signal.

### 3.1 Absolute gaps (current-production T5C vs frozen-CLIP baselines, min recall)

| Comparison | dor Δ | Xinhe Δ | Xiang Δ | min Δ |
|---|---:|---:|---:|---:|
| T5C − LR_DEV         | +0.470 | +0.324 | +0.751 | +0.324 |
| T5C − MLP_DEV        | +0.467 | +0.219 | +0.789 | +0.219 |
| T5C − LR_DEV_LB      | +0.337 | +0.226 | +0.567 | +0.226 |
| T5C − MLP_DEV_LB     | +0.232 | +0.174 | +0.483 | +0.174 |

T5C dominates the best frozen-CLIP head (MLP_DEV_LB) by **17-48 percentage points** per-human at the same effective 5% team-real-FPR.

### 3.2 Best FT'd ckpt (P8A) vs frozen-CLIP

| Comparison | dor Δ | Xinhe Δ | Xiang Δ | min Δ |
|---|---:|---:|---:|---:|
| P8A − LR_DEV         | +0.581 | +0.731 | +0.739 | +0.731 |
| P8A − MLP_DEV_LB     | +0.343 | +0.581 | +0.471 | +0.471 |

### 3.3 Option B vs Option A delta (training-corpus head vs DEV-substrate head, joint-cal team-FPR=5%)

| Head pair (best-vs-best) | dor Δ | Xinhe Δ | Xiang Δ | min Δ | mean Δ |
|---|---:|---:|---:|---:|---:|
| LR_OPTB        − LR_DEV         | −0.218 | −0.053 | −0.197 | −0.055 | −0.156 |
| MLP_OPTB       − MLP_DEV        | −0.200 | −0.146 | −0.154 | −0.154 | −0.167 |
| LR_OPTB_DEV    − LR_DEV         | −0.129 | −0.037 | +0.071 | −0.037 | −0.031 |
| MLP_OPTB_DEV   − MLP_DEV        | −0.121 | −0.123 | +0.144 | −0.122 | −0.034 |
| LR_OPTB_DEV    − LR_DEV_LB      | −0.262 | −0.135 | −0.113 | −0.135 | −0.169 |
| MLP_OPTB_DEV   − MLP_DEV_LB     | −0.356 | −0.168 | −0.162 | −0.168 | −0.229 |

Sign convention: negative means Option B head has LOWER fake recall than Option A head at the same effective 5% team-real-FPR. Every Option B vs Option A pairing is net-negative on `min_recall`. The two combined-corpus heads (OPTB+DEV) get +7-14pp on Xiang but lose 12-36pp on dor and 4-17pp on Xinhe.

### 3.4 Headline question (did Option B close the gap to FT'd?)

| Metric | Best Option A (MLP_DEV_LB) | Best Option B (MLP_OPTB_DEV) | Best FT'd (P8A) | OPTB-vs-OPTA Δ | OPTB-vs-P8A gap remaining |
|---|---:|---:|---:|---:|---:|
| min recall @ 5% team-FPR | 0.205 | 0.037 | 0.786 | −0.168 | 0.749 |
| mean recall @ 5% team-FPR | 0.399 | 0.170 | 0.864 | −0.229 | 0.694 |
| dor recall | 0.527 | 0.171 | 0.870 | −0.356 | 0.699 |
| Xinhe recall | 0.205 | 0.037 | 0.786 | −0.168 | 0.749 |
| Xiang recall | 0.465 | 0.303 | 0.936 | −0.162 | 0.633 |

Adding training-corpus diversity (3K real + 3K fake stratified-by-method sample of `frame_properties.parquet`, either standalone OPTB or OPTB+DEV combined) MOVED THE FROZEN-CLIP BASELINE DOWN on every per-human metric vs the best Option A head. The gap to P8A on `min_recall` widened from 0.581 (Option A) to 0.749 (Option B). The OPTB lockbox-transfer AUCs (0.345-0.516; LR_OPTB below chance) corroborate that the OPTB-trained head learned FF++/DF40-style decision boundaries that do not transfer to the Teams-substrate distribution covered by lockbox and team-identity reals.

---

## 4. Constant-τ comparison @ mode B (τ=0.78)

Same table as the FT'd-ckpts expanded readout's §3, with all 8 frozen-CLIP rows appended. Source: `outputs/comparison_table_v2.csv` (v1 `comparison_table.csv` has Option A only). Rows ranked by `team_min_fake_recall_B` descending.

| Ckpt | Team-agg FPR @B | Team-max FPR @B | Team-min fake recall @B | Roee_Win FPR | dor FPR | Xinhe FPR | Xiang FPR | Noyn FPR | dor fake recall | Xinhe fake recall | Xiang fake recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P8A | 0.012 | 0.016 | 0.487 | 0.009 | 0.016 | 0.000 | 0.014 | 0.005 | 0.745 | 0.487 | 0.815 |
| E2B | 0.015 | 0.036 | 0.473 | 0.000 | 0.011 | 0.000 | 0.036 | 0.000 | 0.473 | 0.674 | 0.869 |
| FrozenCLIP_LR_OPTB_DEV | 0.258 | 0.710 | 0.351 | 0.018 | 0.710 | 0.000 | 0.034 | 0.014 | 0.531 | 0.351 | 0.920 |
| FrozenCLIP_MLP_OPTB_DEV | 0.177 | 0.479 | 0.334 | 0.003 | 0.479 | 0.000 | 0.033 | 0.024 | 0.334 | 0.343 | 0.784 |
| SlotAv2_CLS | 0.010 | 0.024 | 0.293 | 0.000 | 0.008 | 0.000 | 0.024 | 0.000 | 0.414 | 0.293 | 0.830 |
| SlotAv2_FACE | 0.002 | 0.003 | 0.252 | 0.000 | 0.002 | 0.000 | 0.003 | 0.000 | 0.252 | 0.495 | 0.830 |
| T5C | 0.024 | 0.052 | 0.247 | 0.000 | 0.052 | 0.000 | 0.021 | 0.000 | 0.628 | 0.247 | 0.907 |
| FrozenCLIP_MLP_DEV | 0.161 | 0.426 | 0.234 | 0.000 | 0.426 | 0.000 | 0.038 | 0.033 | 0.698 | 0.234 | 0.661 |
| FrozenCLIP_MLP_DEV_LB | 0.100 | 0.208 | 0.222 | 0.000 | 0.208 | 0.000 | 0.088 | 0.014 | 0.716 | 0.222 | 0.687 |
| FrozenCLIP_LR_DEV_LB | 0.137 | 0.268 | 0.216 | 0.000 | 0.268 | 0.000 | 0.136 | 0.024 | 0.727 | 0.216 | 0.775 |
| FrozenCLIP_LR_DEV | 0.191 | 0.473 | 0.214 | 0.000 | 0.473 | 0.000 | 0.077 | 0.043 | 0.723 | 0.214 | 0.656 |
| FrozenCLIP_MLP_OPTB | 0.213 | 0.581 | 0.139 | 0.003 | 0.581 | 0.000 | 0.024 | 0.062 | 0.210 | 0.139 | 0.256 |
| FrozenCLIP_LR_OPTB | 0.291 | 0.826 | 0.098 | 0.030 | 0.826 | 0.000 | 0.009 | 0.014 | 0.264 | 0.098 | 0.339 |

### 4.1 Score-distribution drift caveat

Each row above uses τ=0.78 directly. Frozen-CLIP heads have a different score distribution from the FT'd ckpts (mean team-real prob 0.20-0.45 for frozen-CLIP vs 0.05-0.21 for FT'd ckpts at the calibration scale used; OPTB heads sit at the top of this range, e.g. LR_OPTB mean 0.382). At τ=0.78, the frozen-CLIP heads are scoring closer to the right tail of their natural distribution, so the FPRs are inflated relative to the FT'd ckpts' calibrated mode B. The joint-calibrated comparison in §3 corrects for this.

---

## 5. Per-cohort decomposition

### 5.0 Xinhe-fake decomposition @ joint-cal team-FPR=5%

Xinhe-fake totals 1,099 frames across two cohorts: `extra_xinghe_fake` (100 frames, the older non-Teams sample) and `session_20260414_112354` (999 frames, the `live_prod__xinhe-fake-{1..11-glasses}` Teams-substrate attacks).

| Cohort | n | P8A | E2B | T5C | SlotA-CLS | SlotA-FACE | LR_DEV | MLP_DEV | LR_DEV_LB | MLP_DEV_LB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| extra_xinghe_fake | 100 | 0.940 | 0.960 | 0.790 | 0.920 | 0.990 | 0.010 | 0.050 | 0.040 | 0.040 |
| session_20260414_112354 (live_prod) | 999 | 0.771 | 0.799 | 0.338 | 0.509 | 0.897 | 0.059 | 0.171 | 0.164 | 0.221 |

The `extra_xinghe_fake` cohort gives an extreme gap: FT'd ckpts catch 79-99% of these attacks, frozen-CLIP catches 1-5%. On `live_prod` Teams-substrate Xinhe-fake (the harder cohort), the gap is smaller but still 30-77pp.

### 5.1 dor-real decomposition (worst cohort per human, mode B τ=0.78)

The frozen-CLIP `dor` FPR (0.208-0.473 depending on head) is concentrated in two cohorts: `dor_morning` (n=150) and `dor_shkedi__s16` (chronic). The team-cohort proxy `team_may5__Dor` (n=30) shows 0% FPR across all baselines.

| Cohort | n | LR_DEV FPR@B | LR_DEV_LB FPR@B | MLP_DEV_LB FPR@B |
|---|---:|---:|---:|---:|
| dor_evening | 150 | 0.293 | 0.227 | 0.187 |
| dor_morning | 150 | 0.727 | 0.633 | 0.567 |
| dor_shkedi__s16 (chronic) | 147 | 0.891 | 0.204 | 0.048 |
| dor_shkedi (full) | 31 | 0.000 | 0.000 | 0.000 |
| real_dor | 109 | 0.083 | 0.064 | 0.083 |
| team_may5__Dor | 30 | 0.000 | 0.000 | 0.000 |

**Roee_Windows** (300 frames `tester tester__*` + 30 frames `team_may5__Roee`): 0% FPR across every frozen-CLIP head at mode B.

The LR_DEV → LR_DEV_LB → MLP_DEV_LB direction shows that adding lockbox-real frames to the training pool cuts `dor_shkedi__s16` FPR from 0.891 → 0.048 (because lockbox training data presumably contains this chronic-FP identity). The other two `dor_morning` and `dor_evening` cohorts move less (0.727 → 0.567 / 0.293 → 0.187), suggesting they're not in lockbox training data.

---

## 6. Caveats and limitations

1. **Frame-level overlap not deduplicated**. The DEV training pool (2,000 dev_real from `teams-faces-data-test-2914-fake-4420-real-feb-28`) shares the source bucket with parts of the team-identity real cohort (Noyn 210 + some dor cohorts). Whether any specific frames overlap was not checked. The LR_DEV / MLP_DEV results are therefore an UPPER BOUND on frozen-CLIP head performance from this sampling protocol; if there is overlap the true performance is lower.

2. **LR_DEV_LB / MLP_DEV_LB include the 839 lockbox frames in training**. These frames are not in the team-identity deploy cohort, but the lockbox bucket is similar in capture style. These heads are informational only; they bound what frozen-CLIP can do with lockbox-domain reals in training. Real production would not have lockbox in training.

3. **τ-mode portability**. The cross-ckpt constant-τ values 0.535 / 0.78 / 0.87 were calibrated for Slot A v2 CLS-pool on a 9-suite scorecard. Using the same numeric τ for frozen-CLIP heads (which have a different score distribution; mean team-real prob 0.20-0.27 vs 0.05-0.21 for FT'd ckpts) gives inflated FPR. The §3 joint-calibrated comparison corrects for this.

4. **Sample composition mismatch within OPTB**. Method-stratified sampling means OPTB contains ~30% deep-live-cam variants, ~13% celeb_synthesis, ~12% AVSpeech reals, etc. The FT'd ckpts trained on these methods at much higher epoch coverage. Even Option B is a tiny fraction (6K / 1.4M = 0.4%) of what FT'd ckpts saw. OPTB heads got lockbox AUC 0.345-0.516 (vs DEV's 0.61-0.78) and per-human fake recall <= 9.2% at team-FPR=5%, strictly worse than DEV-trained heads. Whether a larger OPTB sample would change this depends on what fraction of the 1.4M training corpus carries Teams-substrate signal; the method-distribution shows `external_youtube_avspeech`, `celeb_synthesis`, `deep-live-cam_fake`, FF++ variants dominate — Teams-substrate cohorts (`live_prod__*`, `visomaster_v2_*`) are NOT in the training manifest at all (those data lanes were added separately in later FT'd-data revisions). So even a 100% OPTB sample (1.4M frames) would not contain the Teams-fake substrate. The DEV pool (which IS sampled from the Teams-substrate dev set) is the more aligned baseline; combining DEV + OPTB hurt rather than helped (LR_OPTB_DEV min recall 0.018 vs LR_DEV 0.055), suggesting OPTB-style training data overwrites the Teams-substrate signal in the head's loss surface.

5. **Single-head architecture sweep**. LR and one MLP (256u, 1 hidden layer) were tried. Deeper/wider MLPs, gradient boosting on CLIP features, or ensembling were not explored. The fair-baseline question "what is the best frozen-CLIP head" has a higher ceiling than measured here.

6. **Mac-Roee informational read excluded**. Per `project_team_identities_multi_labeled_2026-05-23`, Mac-captured Roee frames are out-of-scope for deploy. They are also excluded from the 5,941 deploy-relevant cohort.

7. **No identity-blocking inside DEV pool**. dor_shkedi__s16 is present in both training (dev_real) and the team-identity cohort (`dor` real-side cohort). When LR_DEV gets 0.891 FPR on dor_shkedi__s16, this is on out-of-training-frames of the same identity. The LR_DEV_LB drop to 0.204 reflects the lockbox training pool including this identity at higher frame coverage.

8. **xinhe_may6_falseflag (92 frames at `analysis/xinhe_cross_camera_audit_2026-05-06/raw/may6/`) is NOT in this readout**. That cohort is mentioned in the brief as historical context (E2B 57.6% vs P8A 0% FPR) but lives outside `grouped_manifest_v2.csv`. Per `project_team_identities_multi_labeled_2026-05-23` the team-cohort proxy `team_may5__Xinhe` (60 frames) is in scope here and shows 0% FPR across every head (FT and frozen-CLIP alike).

---

## 7. Artifacts

- `outputs/clip_frozen_l11__team_identity_n5941.npz` — 5,941 × 768 frozen-CLIP L11 CLS features for the team-identity cohort + frame-paths/labels/humans/roles.
- `outputs/clip_frozen_l11__training_optionB_n6000.npz` — Option B training-corpus feature cache (6,000 frames; 17.2 MB).
- `outputs/per_frame_baseline.csv` / `_v2.csv` — per-frame baseline probabilities, one column per head (v2 has all 8 heads including OPTB).
- `outputs/per_human_baseline.csv` / `_v2.csv` — per-(head × human × role) metrics at modes A/B/C (v2 has all 8 heads).
- `outputs/per_human_combined_with_baselines.csv` / `_v2.csv` — FT'd ckpts + frozen-CLIP baselines in one file (v2 has all 8 baselines).
- `outputs/comparison_table.csv` / `.md` — Option-A-only constant-τ comparison table at mode B.
- `outputs/comparison_table_v2.csv` / `_constanttau.md` — constant-τ comparison at modes A/B/C with all 8 baselines.
- `outputs/comparison_table_v2.md` — joint-calibrated comparison at team-FPR=5% with all 8 baselines.
- `outputs/joint_calibrated_summary.csv` — per-(head × team-FPR-target) full table with all 13 ckpts/heads.
- `outputs/training_summary.csv` / `_v2.csv` — per-head train/dev/lockbox AUCs + dev-cal-5% τ (v2 has 8 head rows).
- `outputs/_extract.log` / `_extract_optionb.log` / `_train_score.log` / `_train_score_v2.log` / `_finalize_optb.log` — full run logs.
- `scripts/extract_clip_features.py` — frozen-CLIP feature extraction (team-identity cohort).
- `scripts/extract_option_b_training.py` — frozen-CLIP feature extraction (training-corpus sample).
- `scripts/train_and_score.py` / `_v2.py` — head training + scoring + comparison-table builders.
- `scripts/finalize_option_b.py` — extends per-human + constant-τ comparison tables to include OPTB heads.

All CPU-only ($0); no Vertex jobs, no image builds, no production code modifications.

---

## 8. Throughput notes

- Team-identity feature extraction (5,941 frames; 1,547 local + 4,394 GCS): 1,053 sec (~17.5 min) on MPS at ~5.7 fps.
- Option B feature extraction (6,000 frames, all GCS): 1,753 sec (~29 min) at ~3.4 fps.
- Head training + scoring (v1, 4 heads): 52 sec.
- Head training + scoring (v2 with OPTB heads, 8 heads): 248 sec (~4 min). MLP_OPTB+DEV fit was the slowest (~114 sec) on the 10K combined pool.
- Finalize Option B (per-human + constant-τ tables): 0.1 sec.
- Total wall clock: ~50 min CLIP feature extraction + ~5 min head training + ~0.1 sec finalize = ~55 min.
