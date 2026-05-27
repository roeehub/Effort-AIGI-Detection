# R13 Forward Plan — 2026-04-29

> Compressed from `docs/packet_retrospectives/plans/MASTER_PLAN_2026-04-29.md` (1527 lines). Target audience: Codex adversarial review via `/claudex:plan --from-draft --rounds 2`. Citation rule: every load-bearing claim cites `file:line` or `file:section`; uncited claims are tagged `[uncited]`. Source of truth: `docs/packet_retrospectives/` as of 2026-04-29 afternoon.

## Changelog

- **2026-04-29 round-2 review (security + data-integrity).** Findings escalated from outside the wiki's tracked surfaces:
  - **§8.2 / §9 P3:** contract v3 commit must ship safe defaults *or* CI guard as part of the same landing, not as follow-up. Update every launch wrapper. (high)
  - **§8.8 / §9 P5:** identity-split is no longer "user judgment alone" — frozen train/val/eval manifests split by identity + capture/session + source bucket, manifest-overlap test as launch-blocker, source-coverage proof for the enhanced-through-Teams conjunction. (high + medium)
  - **§10.11 NEW:** W&B API key currently exported env / hardcoded launch-wrapper fallbacks; rotate, remove defaults, move to Secret Manager / Cloud Build `secretEnv`. (high)
  - **§9 P6 / §10.7:** P15 GRL needs pre-launch label-by-domain-by-source count gate; domain 3 currently empty in training. (medium)
  - **§10.12 NEW:** scorecard outputs are overwritable GCS objects with no immutable run manifest or COMPLETE sentinel. (medium)
  - **§10.13 NEW:** checkpoint upload retry+checksum; keep local until confirmed. (medium)
  - **§12.5:** image-currency guard fail-closed for paid launches; currently fail-open on missing gcloud / `SKIP_IMAGE_CURRENCY_CHECK=1`. (medium)
  - **§9 P1:** Move 1 disposition — if used as launch gate, requires grouped splits + direct enhanced-Teams source + held-out scorecard-matching target. (low)

---

## 0. Orientation

**Three-layer ship-readiness model:**

- **L1 — Contract scorecard.** v3 fix must commit + image-rebuild + scorecard-verify. Mechanical; awaits user. Loop `contract-policy-bug-fix-not-committed`, high.
- **L2 — Headline-metric magnitude (bucket gap).** `R13_P14_DATA_FIX.yaml` wires missing eval-distribution. YAML-only; ~$70 / 1-2 days; preconditioned by Move 1. Loop `p14-data-fix-not-launched`, medium.
- **L3 — Residual shortcut at Axis 3.** P14_FT (in flight) tests P13 anti-shortcut interventions on a working cross-domain base; P15 GRL is structurally-different next lever, contingent on P14 verdict. Loop `shortcut-deployment-block`, critical, in-progress.

All three layers must close together for ship-readiness. Decision tree in §12 sequences them.

**Confidence labels used throughout:** **HIGH** (measured directly), **MEDIUM** (one-two cites, heuristic), **LOW** (substrate-conditional or wiki-flagged-open).

---

## 1. Situation

Load-bearing one-liner (`STATE_2026-04-29.md:9-11`): **frame-level strong on cross-domain fakes; contract-recall weak under legacy policy; separately-fixable bucket gap on headline metric; residual camera-signature shortcut as deployment block.**

### 1.1. Frame-level AUC reframing — HIGH confidence

Cached `P8A_step5000` predictions, no τ (`STATE_2026-04-29.md:74-77`, memory `project_p8a_frame_level_auc_2026-04-29.md`):

| Suite | Frame-level AUC |
|---|---:|
| `visomaster_enhanced_macro_dev` | **0.7527** |
| `deeplive_enhanced_dev` | **0.8614** |
| `teams_fake_all_dev` | **0.9106** |

Reframing (`processing_signature_shortcut.md:51-58`): "low recall" is **scorer artifact**, not model failure. The contract τ-policy operated where the model's confident-fake region sat below threshold. Through Slices 4-6 the framing was "model has shortcut, underperforms cross-domain"; Slice 7 reframes as "substantial cross-domain signal, contract reading near-zero recall via τ-tail-collapse."

**Operational status of frame-level AUC (revised, Codex round 1, MEDIUM):** frame-level AUC is a *sanity check*, not a launch/ship gate. All P-series promotion decisions, lift gates, and §8.1 close criteria remain on **clip-level** corrected-policy contract scorecards. Reasons: (i) frame-level metric is dominated by majority-class frames within a clip; clip-level deployment surface is what matters; (ii) the frame-vs-clip reconciliation loop (§7.1) is open — translating frame-level "0.91 AUC" to "ship-ready clip recall" requires a derivation we haven't done. Until §7.1 closes, citing frame-level AUC as a quantitative target (e.g., "expected viso lift 13.6% → 24% based on AUC ceiling") is *directional only*; the actual lift gate is the clip-level scorecard under corrected policy with the §10.5 group-split holdout.

**The shortcut is real (Axis 3 Δ ~0.5 on best P13) but was NOT the proximate cause of the headline failure** (`processing_signature_shortcut.md:61`). Model less broken than headline metrics suggested — not deployment-ready: corrected viso 13.6% << 90% gate (`STATE_2026-04-29.md:84`); shortcut Δ=0.52 still triggers block (`P13.md:14`); P13 from-scratch modern_v2 FPR explodes to 30.2% (`P13.md:73`).

### 1.2. Contract recall: artifact vs model

| Suite | Buggy policy (τ ≈ 0.99) | Corrected (τ ≈ 0.92, recall floor 0.30) |
|---|---:|---:|
| `visomaster_enhanced_macro_dev` | 1.1% | **13.6%** |
| `deeplive_enhanced_dev` | 1.6% | **23.9%** |
| `teams_fake_all_dev` | ~5% | **52.6%** |
| `lockbox_real_fpr` | 0.4% | 1.8% (within 5% budget) |

Source: `STATE_2026-04-29.md:80-86`, `contract_policy_bug.md:25-66`. **Every P-series scorecard read pre-fix is a lower bound on actual deployment performance** (`STATE_2026-04-29.md:87`).

### 1.3. Bucket gap — HIGH confidence

`visomaster_enhanced_macro_dev` reads `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/` (550 GAN-enhanced viso recaptured through Teams). Training viso reads `gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_*` with `tiers: null` — **zero enhancers, no Teams transport** (`viso_bucket_gap.md:15`).

Deeplive control: `deeplive_enhanced_dev` from same eval bucket; deeplive *training* includes `edge_cases_enhanced` / `minimal_processing_enhanced` AND `deeplive_teams_*` family. Training distribution ≈ eval distribution; viso's does not (`viso_bucket_gap.md:16`). Predicts the persistent ~3-5× viso-vs-deeplive recall gap.

Magnitude: **~2× in delta-from-1**, not 3-5× (`viso_bucket_gap.md:19`). `(1 − 0.75) ≈ 2 × (1 − 0.86)`. Substrate-matched ceiling for L2 lift: viso ≈ deeplive's ~24%, **not** 90%.

Decomposition: ~**two-thirds bucket-gap, one-third shortcut** (`viso_bucket_gap.md:29`, MEDIUM — heuristic, consistent with AUC ceiling). L2 closure is yaml-only; `R13_P14_DATA_FIX.yaml` drafted, not launched.

### 1.4. Camera-signature shortcut — HIGH confidence on production

2026-04-24 controlled 2-camera test (`processing_signature_shortcut.md:14`): same person, same lighting, swap camera, score flips 0.02 ↔ 0.94 on Dor (laptop vs webcam) and 0.01 ↔ 0.90 on Roee (Windows vs Mac). `dor_shkedi` vs `real_dor` flip (`processing_signature_shortcut.md:13`): same person, different `identity_key`, mean prob_fake = 0.457 vs 0.038.

P8A breaks P7 ceiling at single-variable cost (Δ=−0.188 vs RLP6_04 on `dor-real-webcam-no-VBG`; `processing_signature_shortcut.md:20-21`) but at −13.6 pp aggregate fake recall on `teams_fake_all_dev` (`processing_signature_shortcut.md:23`). ~80% of misses below τ-recoverable band (separability collapse).

P13 anti-shortcut moved Axis 3 directionally (P13_step18000 = 0.520 vs P8A 0.659, ~14pp; `processing_signature_shortcut.md:67-69`) but still 3.5× the 0.15 gate. From-scratch trade collapsed cross-domain (viso 5.5%, deeplive 48.8% at τ=0.5; `P13.md:73`).

### 1.5. In-flight / drafted / uncommitted

**In flight** (`STATE_2026-04-29.md:18-19`):
- `R13_P14_FT_FROM_P8A.yaml` — Day-5 fallback per Plan v6; FT-from-P8A_REFERENCE_step5000 + 3 P13 anti-shortcut interventions. Vertex (us-east1 → us-west4 fallback). ~$60. ~24h.

**Drafted, not launched** (`STATE_2026-04-29.md:22-25`):
- `R13_P14_DATA_FIX.yaml` — bucket-gap closure; wires `VisoMasterEnhancedSample` into `combined_paired.py` as `visomaster_enhanced_fake`. Untracked. ~$70 / 1-2d. **Wiring fully built** (`viso_bucket_gap.md:17`). **Source-coverage caveat (Codex round 1):** the yaml enables `visomaster_enhanced` (enhancers, no Teams transport) + `visomaster_hints_teams` (Teams transport, ~404, no enhancers) — i.e. the *union* of two single-axis sources. The eval substrate `teams-faces-data-test-2914-...` is the *conjunction* (GAN-enhanced viso recaptured through Teams). A separate source `visomaster_teams_enhanced` already wired at `combined_paired.py:704` (`discover_visomaster_teams_enhanced_samples` at `data/sources/visomaster.py:1044`) is the direct conjunction match and is **NOT enabled** in the current yaml. See §7.3.1 for resolution path before launch.
- `R13_P15_GRL_FROM_P8A.yaml` — quality-domain DANN. Drafted; smoke-test required. 5-line yaml change from P14. ~$60. Contingent on P14 verdict (`P15.md:60-66`).
- **Move 1** — frozen-feature linear probe of P8A on eval bucket vs training bucket viso. ~$0 / 1-2h. Pre-spend gate. Not yet executed.

**Working tree, uncommitted** (`STATE_2026-04-29.md:28-41`):
- Contract policy v3 fix: +207 net lines across `arena/score_teams_promotion_contract.py` (98), `arena/run_target_domain_validation_sequential.py` (19), `tests/test_score_teams_promotion_contract.py` (125, 2 new tests). Adds `target_fake_recall_min=0.30`; bumps FPR budgets 0.02/0.05 → 0.07/0.10; tier-based sort. 6 contract tests pass locally. Re-scoring P13 lifts P8A rank 4→1 (τ=0.916, viso 1.1%→13.6%). **Third "fix" attempt in 6 days, uncommitted.** (`contract_policy_bug.md:25-66`)
- `tests/test_visomaster_enhanced_wiring.py` (3 wiring tests, untracked).
- `R13_P14_DATA_FIX.yaml`, `R13_P14_FT_FROM_P8A.yaml`, `R13_P15_GRL_FROM_P8A.yaml` (untracked).
- Smaller patches: `feat_norm_reg_lambda` knob, `path_exclude_contains` filter, `entrypoint.sh` subcommands, `launch_batch_inference.sh` cd-vs-SCRIPT_DIR fix, `CLAUDE.md` US-region paragraph.

### 1.6. Eval-substrate caveats added 2026-04-29 PM

Three audit findings affect every FPR/recall reading (`STATE_2026-04-29.md:54-64`):

1. **Eval-vs-production crop-tightness mismatch** — eval frames carry more background context than production crops do (`eval_production_crop_tightness_gap.md:14-22`). Structurally upstream of camera-signature shortcut + face-size leak + webcam FPR. **Translation from eval FPR to production FPR is suspect.**
2. **`sharpness_laplacian` computed on full image, not face** (`sharpness_metric_bug.md:14-26`, `analysis/lockbox_tagging/layers/quality.py:80-82`). Per-quartile FPR analyses indexed on this column are partially confounded.
3. **Eval substrate has data-quality issues** (`eval_substrate_data_hygiene.md`): is_no_face slice (n=219) is data degeneracy not model weakness; substrate contains 99×110-pixel source crops passing existing 1k px² floor but failing any reasonable `min(width,height) ≥ 200` bound.

**All FPR numbers carry these caveats.** Modern_v2 already excludes is_no_face but not the resolution floor or the crop-tightness gap. **Frame-level AUC reframing survives** (caveats apply uniformly to baseline-vs-corrected-policy comparisons, `STATE_2026-04-29.md:88-89`); what shifts is "eval-FPR X" → "production-FPR X" without translation audit.

---

## 2. Strengths

**2.1. Cross-domain frame-level signal is real and substantial.** Frame-level AUCs (§1.1) are dispositive that contract scorer's "viso recall = 1.1%" was measurement artifact. Operational rule encoded: *"sanity-check operating-point recall against frame-level AUC before concluding a checkpoint is bad"* (`processing_signature_shortcut.md:63`). HIGH.

**2.2. P8A breakthrough on camera-signature anchor** (`processing_signature_shortcut.md:18-25`):
- Anchor pool `dor-real-webcam-false-flag-no-virtual-bg` Δ = **−0.188** vs RLP6_04 (best P7 was −0.097).
- Roee-mac Δ = **−0.399** (best P7 was −0.228).
- All 3 real-correct pools moved closer to zero (no FPR regression).
- `lockbox_real_fpr`: 0.441% → 0.147% (`processing_signature_shortcut.md:167`).
- Single-variable delta from RLP7_02: `unfreeze_final_proj=true`, `unfreeze_final_ln=true`, `apply_svd_to_mlp=true`.

Cost: −13.6 pp aggregate `teams_fake_all_dev` recall at default τ. **Caveat (2026-04-26):** P8A's 3-lever claim was wrong — `apply_svd_to_in_proj=true` was a no-op pre-fix; actual lever set was MLP-SVD + visual.proj + ln_post (memory `project_p8a_breakthrough.md:24-31`).

**2.3. Modern_lockbox_v2 0.71% FPR — data hygiene at P8A baseline** (`webcam_fpr_dominance.md:31-46`):

| Filter | n_real | n_fake | FPR @ τ=0.5 | FPR @ τ=0.9741 |
|---|---:|---:|---:|---:|
| baseline | 414 | 425 | 21.98% | 4.59% |
| v2_recommended | 281 | 367 | 6.76% | **0.71%** |

v2 = `clip_capture_mode ∉ {webcam, screen} AND face_area_ratio ≥ 0.10 AND not is_pose_extreme AND not is_no_face`. **FPR side of 90/5 solved by data hygiene at P8A baseline.** Recall side remains gating.

**Caveats (compound):** v2 is `dor_shkedi`-skewed (211/281 = 75%; `webcam_fpr_dominance.md:50`); two highest-FPR identities (Chikara, PC_Generator__s15) almost-fully filtered by webcam-drop — survivor bias. Plus crop-tightness gap may not make v2 a strict production-FPR floor (`webcam_fpr_dominance.md:80-88`).

**2.4. Operational discipline accumulated:**
- Pre-launch image-currency guard (commit `ad76cd8`, `scripts/launch/check_image_currency.sh`).
- `tests/test_train_sweep_reapply_allowlist.py` (commit `c366026`) — first generalized regression-test guard for wandb-flattening.
- 6 contract tests pass locally for v3 fix (`contract_policy_bug.md:54-57`).
- **Caveat (Codex round 1):** `tests/test_unpaired_reals_and_grl.py` currently fails at the augmentation-pipeline import stage (4 tests). The blanket "unit-tested" claim for the GRL infra in §10.7 was overstated; first 9 tests pass, downstream pipeline/yaml tests do not. Fix is gating for P15 launch (see §10.7).

**2.5. Identity diversity adequate — refutes "more identities" recipe** (`identity_audit.md:18-26`):

| Bucket | Source | Samples | Identities | In P13? |
|---|---|---:|---:|:---:|
| `live-deepfake-methods-real-and-fake-frames-cropped` | viso | 960 | **429** | ✅ |
| same | deeplive | ~5,800 | **1,916** | ✅ |
| `live-...-teams` | teams | 2,626 | 1,306 | ✅ partial |
| `live-...-teams-v2` | viso_teams_v2_companion | 1,994 | 997 | ❌ |
| `visomaster-enhanced-face-cropped-v2` | viso enhanced | 2,073 | many | ❌ |
| `hdtf_visomaster_cropped_frames` + `_teams` | proper-data 2026-04-19 | 1,322 | 705 | ❌ |

Active training: ~430 viso + ~1,900 deeplive. **+2,300 unused identities** in eval-distribution-matching buckets. Right axis is *pipeline transport*, not identity count (`identity_audit.md:32`).

**2.6. L2 wiring fully built** (`viso_bucket_gap.md:17`):
- `data/sources/visomaster.py:605-890` — `VisoMasterEnhancedSample`.
- `arena/build_visomaster_enhanced_v2_manifest.py` + `arena/manifests/visomaster_enhanced_v2_manifest_2026-04-13.json` (2,073 frames, 9 swap × 8 enhancers).
- `data/sources/combined_paired.py:646` — `create_unified_samples_from_visomaster_enhanced` with `QUALITY_DOMAIN_MAP["visomaster_enhanced"] = 2`.
- `train_sweep.py` re-applies whole `combined_paired` block (no flattening trap for nested children).
- `tests/test_visomaster_enhanced_wiring.py` (3 tests, untracked).

Blocker: yaml enables — `visomaster_enhanced.enabled: true` + `visomaster_hints_teams.enabled: true` + family weights. **Pre-launch correction required (Codex round 1, §7.3.1):** decide whether to additionally enable `visomaster_teams_enhanced.enabled: true` (the direct enhancer×Teams conjunction source) or empirically prove the union of the two single-axis sources reproduces the eval substrate's joint distribution before retraining. Default recommendation: enable the conjunction source.

**2.7. Real-vs-artifact discipline.** RLP6_04 84.2% lockbox-fake-recall partially shortcut-driven (`processing_signature_shortcut.md:13-14`); Slice-7 reframing keeps both (model has signal AND contract reads artifact) without inflating either.

---

## 3. Weaknesses and confounds

**3.1. Contract scorer τ-tail collapse** (`contract_policy_bug.md:7`). Lexicographic τ-selection minimizes `dev_primary_real_fpr` first, no recall budget. On sharp-prediction model (Effort, p90 ≈ [0.94, 0.99]), τ snaps to ~0.995, recall craters ~10-30× vs trainer W&B. Bug is **structural** in deployment artifact (lines 456-479). Recurring since 2026-04-23. Three "fix" attempts; none committed. Pattern: **design lands as analysis, doesn't propagate to deployment artifact** (`contract_policy_bug.md:74`).

**3.2. Sharpness-metric full-image bug** (`sharpness_metric_bug.md:14-19`). `sharpness_laplacian` at `analysis/lockbox_tagging/layers/quality.py:82` runs on full grayscale, not face crop. Reproduction: `dor_shkedi__frame_000059` full=760.7 / face=484.1 / **bg=844.9**; `PC_Generator__s15_1754.4_frame_058496` full=555.1 / face=443.6 / **bg=1018.1**. 2026-04-27 "very sharp >402: 45% FPR" finding mixes two populations (sharp faces in sharp envs vs soft faces in busy/compressed crops where bg drives laplacian).

**3.3. Eval-vs-production crop-tightness gap** (`eval_production_crop_tightness_gap.md:14-22`). Structurally most consequential audit. Eval frames carry more background context. Implications: pipeline-signal cues get larger eval surface; eval FPR overstates production FPR for shortcut-driven flags; eval recall might over/under-state production recall depending on signal locus; close criterion of `shortcut-deployment-block` is on eval substrate. **Cheapest closure: re-crop sample of lockbox at production tightness, re-score** (`eval_production_crop_tightness_gap.md:42`). 2-camera test was production-side, so shortcut is real on production; eval substrate amplification compounds rather than replaces.

**3.4. In_proj-SVD retroactive attribution problem** (`in_proj_svd_gradient_bug.md:5,13-14`). Every R12g/RLP/P-* run with `apply_svd_to_in_proj=true` trained q/k/v residuals on **zero classification gradient** until commit `2feea58` (2026-04-26). `forward_pre_hook` did `module.in_proj_weight.data.copy_(...)` — `.data.copy_` skips autograd. Grad audit (`analysis/probe_battery_2026-04-26/grad_audit.py`): on OLD, 9/9 in_proj residuals had grad=None or zero. Post-fix: 9/9 in_proj + 3/3 `SVDResidualLinear` receive non-zero grad.

Implications:
- Memory `project_p8a_breakthrough.md:24-31` carries CORRECTION block.
- Memory `project_shortcut_is_upstream.md:33-40` carries CAVEAT block.
- Loop `apply-svd-in-proj-attribution-revision-needed` (open, high) tracks unfinished retroactive sweep.

**3.5. Modern_v2 dor_shkedi-skew** (`webcam_fpr_dominance.md:50-57`). 211/281 = 75% of v2 reals are dor_shkedi.

| Identity | n | px² | FPR |
|---|---:|---:|---:|
| bla_bla_chow__s1 | 68 | 45k | 1.47% |
| dor_shkedi | 275 | 22k | 10.55% |
| Chikara_Takahashi__s22 | 42 | 17k | 83.3% |
| PC_Generator__s15 | 29 | 2.3k | **89.7%** |

Per-identity breakdown **mandatory** before treating v2 as load-bearing.

**3.6. Source-bucket linear probe says P8A is NOT shortcut-clean.** 10-class probe, chance=0.10, gate ≤0.25 (`processing_signature_shortcut.md:31-35`):
- RLP6_04 control: 0.97 test_acc — saturates.
- P8A_REFERENCE_STEP5000: **0.461** — 4.6× chance, FAIL.
- C3_CODEC_HEDGE_VC_STEP2000: 0.416 — 4.2× chance, FAIL.

P8A's 65% τ=0.5 lockbox fake recall is **partially shortcut-attributable**. Codec aug pulled some signal out (~9% relative reduction) but didn't clear gate.

**3.7. Crop-tightness sweep flips 53% of frames.** 47-frame × 5-tightness sweep: 25/47 (53%) flipped predicted label across t∈[0.7, 1.5] (`face_size_label_leak.md:15-18`). FAIL regime sits in "FAKE valley" at native crop. Same shortcut from different axis as camera-signature (memory `project_face_size_label_leak.md`).

**3.8. AUC ceiling caps L2 expected lift** (`viso_bucket_gap.md:19`). L2 closure plausibly lifts viso to ~24%, **not** 90%. Gap from 24% → 90% is what shortcut + architectural-class interventions must close. Quantitative bound, not a claim L2 is unimportant.

**3.9. 70% of cross-pool FPR gap remains post per-camera calibration.** WS-P1 verdict: 30.1% closure on average across {5,10,15,25}% target FPRs. Per-target {5%: 0, 10%: 0.333, 15%: 0.286, 25%: 0.583}. **Per-camera calibration is complement, not cure.** Needs training-side fix (`OPEN_LOOPS.md` `residual-70-pct-fpr-gap-no-lever-past-rlp7`).

**3.10. Frame-level AUCs themselves carry eval-substrate caveats** (`STATE_2026-04-29.md:88-89`). Computed on same substrate flagged by §1.6. Slice 7 reframing correct directionally; absolute interpretation as production floor inherits caveats.

---

## 4. Whack-a-mole patterns

### 4.1. Contract-policy bug — three fixes in 6 days, none committed

**The pattern this knowledge base was built to prevent surfaces here in concentrated form** (`contract_policy_bug.md:3`).

- **Attempt #1 (2026-04-23 slot-07 sanity).** First live evidence — slot-07 sanity carrier hits τ=0.9953, dev real FPR=0 across 3 suites, recall craters (`teams_fake_all_dev=27.6%`, `viso_enhanced=1.6%`, `deeplive_enhanced=0.0%`, `lockbox_fake=15.8%`, `dev_fake_macro=9.7%`); trainer-side `value_composite=0.7736`. **Filed as "expected sanity, FPR axis passes."** Bug existed for 4+ days before framed as bug. (`contract_policy_bug.md:15`)
- **Attempt #2 (2026-04-27 codec-hedge readout, lost to /tmp wipe).** All 4 ckpts hit τ ≈ 0.97-0.99 regime. Agent surfaces PCP with 3 forks; user authorizes (ii) "fix the contract policy bug first." +61/-16 patch added `target_real_fpr` / `target_stress_fpr` budget infra. Variants written to `analysis/policy_reruns_2026-04-27/{default,recall_floor_30}/`. **Not committed. Lost to /tmp wipe across machine change** (`contract_policy_bug.md:17,19`).
- **Attempt #3 (2026-04-29 v3 fix).** `target_fake_recall_min=0.30` + `target_real_fpr=0.07` + `target_stress_fpr=0.10`. Re-scoring P13: P8A rank 4→1, τ=0.992→0.916, viso 1.1%→13.6%, deeplive 1.6%→23.9%, teams_fake ~5%→52.6%, lockbox 0.4%→1.8%. 6 contract tests pass. **Working tree, uncommitted as of today.** (`contract_policy_bug.md:23`)

Diff structure (`contract_policy_bug.md:39-50`):
1. `_threshold_sort_key` (lines 456-479): tier-based sort (tier 0/1/2 by budget+floor); default `target_fake_recall_min=0.0` preserves backcompat.
2. `_promotion_summary_sort_key` (lines 510-531): cross-ckpt ranker tiers — without it, `P13_step2000` was crowned rank-1 with 3.6% lockbox fake recall.
3. `arena/run_target_domain_validation_sequential.py` (+19): three CLI args — `--promotion_target_real_fpr`, `--promotion_target_stress_fpr`, `--promotion_target_fake_recall_min`, plus `--promotion_readout_only_suites`.

**Why fixes didn't stick:** #1 misclassified; #2 lost to /tmp wipe + no explicit "commit now"; #3 in tree without commit. Structural failure mode: **design lands but doesn't propagate to deployment artifact** (`contract_policy_bug.md:120-122`). Wiki's loop `contract-policy-bug-fix-not-committed` is the surface that should prevent attempt #4.

### 4.2. Wandb-side hygiene pattern recurrence — 3 instances

- **Slice 3 (`872502c`, 2026-04-22).** `wandb.config.get('value_composite')` returns None despite yaml. 8-line allowlist patch. Memory rule named, **no test guard added.**
- **Slice 5 (`f366368`, 2026-04-26).** Different wandb surface — artifact name >128 chars. Pattern recurs.
- **Slice 6 (`c366026`, 2026-04-28).** `anchor_aware`/`face_scale_jitter`/`periodic_saves` silently DISABLED on P13 first launch (`wandb_flattening.md:17-22`). ~$3.50 burned across cancelled jobs; first wrong-layer fix attempt (trainer-side `_to_plain_dict()`) lost ~$1.40. Pattern: **flattening is sensitive to depth of new key relative to existing allowlist coverage** (`wandb_flattening.md:35`). Nested feature added under existing top-level allowlisted block silently works; new top-level block silently fails.

**Slice 6 fix structurally complete:** `c366026` adds 3 new keys to `train_sweep.py:283-313` mirroring Slice-3 pattern + `tests/test_train_sweep_reapply_allowlist.py` fails CI if any `TRAINER_NESTED_KEYS` not re-applied. Slice-3 loop `train-sweep-allowlist-not-closed-under-schema-changes` was prematurely marked resolved without closure mechanism; stays resolved post-Slice-6 because closure mechanism (test guard) now exists.

Broader loop `wandb-side-surface-hygiene-not-systematic` (medium) stays open — covers artifact-name + future flattening surfaces.

### 4.3. In_proj-SVD silent zero-gradient bug

Different bug class (autograd graph break) but **same failure mode**: lever was silently no-op; only explicit audit caught it (`in_proj_svd_gradient_bug.md:74`). Discovery: 2026-04-26 static review of `_install_svd_in_proj_routing` caught `forward_pre_hook` + `.data.copy_`. Fix: commit `2feea58`. Regression guard: commit `7af72b1` adds anchor-pool monitor + grad_audit.

Phase C overnight (2026-04-26): C.1 (with fix) vs C-ablation (`apply_svd_to_in_proj: false`) **tied within noise** on canonical P10_SYM-on-P8A recipe (`in_proj_svd_gradient_bug.md:21-22`). Lever exists but not exercised by this recipe. Loop `in-proj-svd-residual-capacity-not-exercised` (medium) tracks whether different recipe surfaces measurable lever.

### 4.4. Meta-pattern: "design lands, ship discipline fails"

Wiki names this in `silent-feature-failures-pattern` (medium, `wandb_flattening.md:69-83`), 3 instances:

1. `apply_svd_to_in_proj` zero-gradient.
2. `anchor_aware`/`face_scale_jitter`/`periodic_saves` silent DISABLED.
3. `periodic_saves` silent no-checkpoint (P12_HEAVY_LONG dud).

Closure: **single pre-launch CI/smoke/lint step verifying every yaml-declared trainer feature emits "ENABLED" log line at trainer init.**

---

## 5. Marked fixed, probably aren't

**5.1. `periodic_saves` silent failure.** Patched `c7dc828` (no `isinstance(dict)` defense). Silently failed P12_HEAVY_LONG. Closed via Slice 6 `c366026`. Loop status: closed for periodic_saves specifically; OPEN as `silent-feature-failures-pattern` meta-loop. **Recurrence risk:** new nested-dict block in different code path could repeat. A code path bypassing `train_sweep.py` (direct trainer invocation) wouldn't benefit from test guard.

**5.2. Slice-3 `train-sweep-allowlist-not-closed-under-schema-changes`.** Marked resolved with mechanical fix `872502c`. Recurred 6 days later in Slice 6 (3 missing keys). Stays resolved post-Slice-6 because test guard added. **Case study in why `resolved` carries weight only with closure mechanism.**

**5.3. In_proj-SVD lever post-fix.** Bug fixed `2feea58`; 9/9 residuals receive grad. **But not exercised by canonical recipe** — Phase C tied within noise. Loop `in-proj-svd-residual-capacity-not-exercised` open; FT-only ceiling claim from P7/RLP7_08 imprecise (measured with broken in_proj-SVD). Full per-packet sweep not done — loop `apply-svd-in-proj-attribution-revision-needed` (high) tracks.

**5.4. Recurrent meta-pattern: closure-mechanism-vs-mechanical-patch.** Status `resolved` means little without closure mechanism. Slice-3 → Slice-6 trajectory canonical example. **Implication for v3 contract fix:** runner default is `target_fake_recall_min=0.0` (`contract_policy_bug.md:50`); future operator without explicit `--promotion_target_fake_recall_min 0.30` re-fires bug regime. Closure (CI test asserting runner refuses τ ≥ 0.99 unless floor explicit) not in place. **ESCALATED to in-scope blocker (Codex round 1, HIGH):** v3 fix as drafted ships the bug latent — any operator omitting the flag silently re-enters τ-tail collapse. Required as part of Priority 3 commit (see §8.2 expanded close criterion): EITHER (a) flip runner default to `target_fake_recall_min=0.30` and require explicit `--legacy_no_recall_floor` to opt out, OR (b) add fail-fast assertion that rejects any selected `τ ≥ 0.99` paired with `floor=0` and surfaces "policy regression — set --promotion_target_fake_recall_min explicitly". Plus a CI test that exercises the assertion path.

---

## 6. Wrong paths and dead-end conclusions

**6.1. "N too small" identity-diversity hypothesis (refuted).** Mental model off by 2 orders of magnitude. Hard counts: 429 viso + 1,916 deeplive in active training; +2,300 in unused buckets including eval-distribution-matching one. Right axis: pipeline transport, not raw identity count. Noticed 2026-04-29 ~10:00 CEST after P13 γ verdict, when user pushed back and asked agent to count buckets (`viso_bucket_gap.md:15`). Hypothesis was detecting right surface (wrong distribution), wrong axis (`identity_audit.md:27`).

**6.2. "FT-only ceiling" reading from before in_proj-SVD fix.** P7 ceiling at ~−0.10 attributed to "head + attention-only SVD reach + frozen visual.proj/ln_post" — reading in_proj-SVD residual capacity as load-bearing. Wrong: every R12g/RLP/P-* with `apply_svd_to_in_proj=true` trained on zero classification gradient. P8A's anchor breakthrough came from MLP-SVD + unfrozen `visual.proj` + unfrozen `ln_post` ALONE. Noticed 2026-04-26 static review. Memory CORRECTION/CAVEAT blocks added. Qualitative narrative survives; parameter accounting shifts. Phase C suggests in_proj-SVD lever small relative to others in canonical recipe.

**6.3. `dor_shkedi` / `real_dor` flip misread initially.** Slot-07 sanity at τ=0.9953 with viso 1.6% / lockbox 15.8% / dev_fake_macro 9.7% filed as "expected sanity, FPR axis passes." Was first live evidence of contract-policy τ-tail-collapse interlocked with camera-signature shortcut. Diagnosis 4+ days late. Proximate cause of wiki's anti-whack-a-mole protocol.

**6.4. Hints-as-supervision (closed; negative result).** RLP1 (2026-04-19) tested hints ladder. Result: 8/8 matched holdouts Δ −0.0077..−0.0115. Closed Slice-1 loop as negative.

**6.5. Codec_hedge as Phase D promotion candidate (downgraded).** 2026-04-26 Phase C: codec_hedge +30% best_anchor/composite vs canonical C.1. 2026-04-27 lockbox-anchored eval: P8A reference dominates BOTH lockbox axes vs all 3 C-slate ckpts; all 4 hit τ ≈ 0.97-0.99 regime; source-bucket probe on C3 = 0.416 (FAIL). Trainer-side metrics not deployment-grade. Operational rule encoded.

**6.6. Inference-time stack as recipe lever (exhausted).** 2026-04-28 13:30 Option A: 85 configs (30 single + 45 noisy-OR + 10 weighted-noisy-OR). **All 85 fail 90/5.** Best ensemble: `p8a_reference + p11_heavy_step1000` isotonic-noisy-OR, τ=0.846 → viso 31.3%, deeplive 58.2%, teams_fake 75.0%, modern_v2 FPR 4.98%, teams_real_all_dev 10.9% (fails 7% cap). AUC(viso fake vs modern_v2 real) caps ~0.65 across all configs. **Calibration moves probabilities, not AUC.** Motivated Plan v4 substrate-redesign track.

**6.7. Pattern.** Each wrong path detected real signal, pinned on wrong cause. Each took ≥1 packet to resolve. **Wiki's job is to prevent re-derivation.**

---

## 7. Paths to investigate further

**7.1. Frame-level vs clip-level scorer reconciliation.** Test: corrected contract policy with floor 0.30 produces clip-level recalls within ~5pp of `(1 − threshold-implied-FNR)` from frame-level distribution, OR contract scorer documented as measuring something genuinely different. Loop `frame-level-vs-clip-level-scorer-mismatch` (open, medium; `viso_bucket_gap.md:73-79`). Cost: docs + analysis; no Vertex. Informs every future scorecard read. Bucket-gap finding makes this materially more important.

**7.2. Eval-vs-production crop-tightness quantification (Move 1.5).** Cheapest test: re-crop sample of lockbox at production tightness, re-score `P8A_step5000`. If distribution shifts materially, gap is load-bearing. Loop `eval-production-crop-tightness-mismatch` (open, high; `eval_production_crop_tightness_gap.md:62-69`). Cost: hours; no Vertex. Drafted as Move 1.5 companion to Move 1.

**7.3. P14_DATA_FIX (drafted, not launched).** Wires `VisoMasterEnhancedSample` (`data/sources/visomaster.py:605-890`) into `combined_paired.py` as `visomaster_enhanced_fake`. Loop `p14-data-fix-not-launched` (medium; `viso_bucket_gap.md:61-70`). Cost: ~$70 / 1-2d retrain + ~$10 / 3h scorecard. State: drafted (`R13_P14_DATA_FIX.yaml`, untracked); tests drafted (untracked). Move 1 = pre-spend gate. Expected lift: 13.6% → ~24% (substrate-matched ceiling). NOT to 90%.

**7.3.1. P14_DATA_FIX source-coverage correction (Codex round 1, HIGH).** The yaml enables `visomaster_enhanced` + `visomaster_hints_teams`, the *union* of single-axis sources. The eval substrate is the *conjunction* (enhancer×Teams). Source `visomaster_teams_enhanced` already wired at `combined_paired.py:704` is the direct conjunction match but is **not** enabled. Resolution before launch (pick one):

- **Path A (default recommendation):** add `visomaster_teams_enhanced.enabled: true` + family weight to yaml; smoke load 16 samples; verify discover path returns nonzero count; per-family-count log.
- **Path B:** prove empirically — compute distributional similarity (e.g. backbone-feature MMD or simple cosine to eval-substrate mean) for each of {`visomaster_enhanced`, `visomaster_hints_teams`, union, `visomaster_teams_enhanced`} vs the eval suite. If union ≥ conjunction within ε, proceed without `visomaster_teams_enhanced`; else enable it.

Either way, gate launch on a manifest-overlap test asserting no train/eval row collision (see §10.5 expanded spec).

**7.4. P15 GRL (drafted, smoke test required, additional pre-launch gates per Codex round 1, MEDIUM).** 5-line yaml from P14: `use_quality_domain_head: true` + `quality_domain_loss_weight: 0.20` + `quality_domain_require_labels: true` + `quality_domain_count: 4` + `quality_head_hidden_dim: 128`. Seed 1501 (`P15.md:38-46`).

**Pre-launch gates (added Codex round 1):**
1. **Label×domain×source coverage table** emitted at trainer init: rows = `(source_family, fake_or_real)`, cols = quality_domain ∈ {0,1,2,3}, values = sample counts. Fail-fast log if any (label, domain) cell is zero AND `quality_domain_require_labels=true`.
2. **Domain 3 disposition.** Per the §7.4 mapping table, no training source maps to domain 3 (`social_media`); only `youtube` OOD-monitor uses it. Decide before launch (pick one): (a) drop `quality_domain_count` to 3 and remap domain indices, OR (b) treat domain 3 as monitor-only with explicit `quality_domain_loss_mask` excluding it from gradient, OR (c) leave as-is and accept that the GRL head will never see domain-3 supervision (the modal current state — document it explicitly).
3. **Label/domain confound check.** Domains 0–2 partially correlate with the deepfake-source family (df40↔domain0, deeplive_teams↔domain1, deeplive↔domain2). If quality-domain accuracy on a frozen-feature probe ≈ source-family accuracy, GRL is gradient-reversing source-family signal — exactly the cross-domain capacity it's supposed to preserve. **Pre-launch:** train a 2-line frozen-feature logistic from P8A features → quality_domain labels; report accuracy. If > 0.85 across all (source_family → domain) pairs, flag confound and either rebalance domains or document risk explicitly.
4. **Smoke run requirement.** Launch a 200-step smoke job that prints `Quality domain head ENABLED ...` AND emits per-step `quality_domain_loss` to W&B; gate full launch on (a) log line present, (b) loss is finite and nonzero, (c) gradient norm on quality head ≠ 0.

Domain mapping (`P15.md:24-32`):

| Domain | Sources |
|---|---|
| 0 clean_academic | `df40` |
| 1 webcam_codec | `external` (VCD), **`deeplive_teams`** |
| 2 studio_capture | `deeplive`, `visomaster` |
| 3 social_media | (none in training; `youtube` OOD-monitor only) |

Teams traffic = domain 1, same distribution that drives modern_v2 FPR. Backbone penalized when features identify "this is webcam_codec" — exactly the shortcut breakdown P13_step18000 demonstrated (modern_v2 FPR exploded 30%+ because backbone tied "webcam look" → "fake"). Cost: ~$60 / 8h. Decision logic on P14 verdict (α: don't / β: do / γ: do; `P15.md:60-66`).

**Slice-7 reframing (`P15.md:55-66`):** GRL on training-only pipelines does not teach model the eval pipeline; expected gain higher *after* bucket gap closes. **P15 still useful but no longer cheapest next move** — contingent on P14 verdict + Moves 1-3.

**7.5. Sharpness-metric re-derivation on face crops.** ~15 lines in `analysis/lockbox_tagging/layers/quality.py:82` + re-tag `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` + audit downstream. Loop (open, high; `sharpness_metric_bug.md:69-74`).

**7.6. Source-image-resolution floor on eval substrate.** Test: `min(width, height) >= 200` floor on canonical ckpt. Compose with v2 or report side-by-side. 99×110 frames pass `face_pixel_area > 1k` but fail any reasonable resolution bound (`eval_substrate_data_hygiene.md:18-19`). Loop (open, medium; `eval_substrate_data_hygiene.md:65-70`).

**7.7. Canonical priority sequence** (`viso_bucket_gap.md` § 6, `P15.md:60-66`):

1. Move 1 — linear probe (~$0).
2. Move 2 — wire enhanced viso (P14_DATA_FIX, ~$70).
3. Move 3 — superpose anti-shortcut on top of P14_DATA_FIX.
4. Move 4 — same-identity pair training using proper-data wave.
5. Move 5 — only if Moves 2-4 still leave Axis 3 Δ > 0.30: revive P15 GRL with broader pipeline labels.

---

## 8. Open-loop close-path map

### 8.1. `shortcut-deployment-block` (in-progress, critical)

**Prerequisite:** contract-policy v3 commit (8.2) so close criterion's `lockbox_fake_recall ≥ 0.60` half is readable.

**What to do:** chain canonical priority sequence — P14_FT verdict (in flight) → Move 1 → Move 2 (P14_DATA_FIX) → Move 3 (anti-shortcut on top) → if Δ > 0.30: Move 5 (P15 GRL).

**Cost:** ~$60 P14_FT (in flight) + ~$70 P14_DATA_FIX + ~$60 P15 = ~$190 across sequence.

**Signal:** `dor-real-webcam-false-flag-no-virtual-bg` mean prob_fake ≤ 0.30 AND `lockbox_fake_recall ≥ 0.60` at τ holding `teams_ood_real` FPR ≤ 5%.

**Sub-loop disposition (`shortcut-block-criterion-may-be-scorer-artifact-bound`, medium):** record whether `lockbox_fake_recall ≥ 0.60` half is reaffirmed under corrected contract policy or revised to frame-level-AUC version. **Do not change criterion silently** (`processing_signature_shortcut.md:147-156`).

### 8.2. `contract-policy-bug-fix-not-committed` (open, high)

**Prereq:** none — working tree diff is ready.

**Do:** commit +207 net diff + **default-safety patch (Codex round 1)**: flip runner default to `target_fake_recall_min=0.30` (or add fail-fast assertion + CI test per §5.4) so future scorecards cannot silently re-enter τ-tail collapse + image rebuild (`./dev.sh build-prod -y` auto-bumps VERSION) + scorecard run on a representative checkpoint (e.g., P8A_REFERENCE_step5000). All four required; do NOT commit v3 without the default-safety patch — otherwise the bug ships latent.

**Cost:** mechanical commit + ~3h Vertex.

**Signal:** scorecard documents τ via recall-floor path, not legacy minimize-FPR-only.

**Owner:** user. **Wiki built specifically to prevent attempt #4** (`contract_policy_bug.md:115-122`).

**Standing pattern:** every future P-series scorecard read requires this in image; otherwise τ-tail-collapse fires.

**Round-2 escalation — launch-wrapper audit (high).** A safe runner default still leaves stale wrappers as a footgun if any wrapper passes `target_fake_recall_min=0.0` explicitly or pins a legacy budget. Before commit, audit and update every authoritative invocation surface to either pass the new flags or rely on the new defaults: `arena/run_target_domain_validation_sequential.py` callers, `scripts/launch/launch_batch_inference.sh`, any cron / Cloud Build / CI invocation, the local helpers under `analysis/policy_reruns_*`. List each wrapper path in the commit message so a future audit can verify coverage.

### 8.3. `fpr-minimization-no-budget-tau-collapse` (in-progress, high)

**Paired with 8.2; closes together when v3 lands on main.** This loop = bug itself; 8.2 = commit-discipline failure.

**Close criterion:** runner enforces recall floor or τ ceiling preventing `selected_threshold ≈ 0.995` silently.

**Durability note (revised, Codex round 1, HIGH):** v3 runner default = `target_fake_recall_min=0.0` (`contract_policy_bug.md:50`) — escalated from "future-slice" to **commit-blocker**. Required before v3 lands on main: flip default to 0.30 OR add fail-fast assertion that refuses τ ≥ 0.99 with floor=0, plus CI test exercising the assertion. Otherwise v3 ships the bug latent and 8.2 closure is cosmetic.

### 8.4. `apply-svd-in-proj-attribution-revision-needed` (open, high)

**Prereq:** none.

**Do:** either (a) re-train every prior R12g/RLP/P-* with `apply_svd_to_in_proj=true` from matched fork point with fixed routing + per-packet delta table, OR (b) annotate every retro with bug-class CAVEAT.

**Cost:** (b) ~hours/packet × 6-10 retros. (a) $X00s + days.

**Recommendation:** **(b) for all packets EXCEPT P8A and upstream-shortcut reasoning** (already have CORRECTION/CAVEAT in memory). Phase C tied-within-noise suggests lever is small in canonical recipe; absolute attribution corrections likely don't shift qualitative narratives.

**Signal:** every published retro carries the CAVEAT.

### 8.5. `face-size-label-leak` (open, high)

**Prereq:** P14 verdict so post-`c366026` candidate (with `face_scale_jitter` actually live) can be re-tagged.

**Do:** re-measure per-method face-pixel-area on post-P14 candidate; require Cohen's d on dev_fake vs dev_real ≤ 0.10 AND tightness-sweep flips ≤ 10% in FAIL regime.

**Cost:** post-P14 re-tag + small inference run.

**Note:** parallel to 8.10 (resolution floor) — different mechanisms.

### 8.6. `eval-production-crop-tightness-mismatch` (open, high)

**Prereq:** none.

**Do:** re-crop lockbox sample at production tightness, re-score P8A_step5000 (Move 1.5; 7.2). Either (a) quantify delta + document translation rule, OR (b) replace canonical reporting surface with re-cropped substrate.

**Cost:** hours; no Vertex.

**Signal:** if score distribution shifts materially, gap is load-bearing; revise close criteria for `shortcut-deployment-block` per disposition tracked by `shortcut-block-criterion-may-be-scorer-artifact-bound`.

### 8.7. `sharpness-metric-computed-on-full-image-not-face` (open, high)

**Prereq:** none.

**Do:** ~15 lines in `analysis/lockbox_tagging/layers/quality.py:82` + re-tag parquet + audit downstream consumers (esp. 2026-04-27 per-quartile FPR table).

**Cost:** short session.

**Signal:** every per-quartile FPR analysis indexed on sharpness either rerun or annotated as confounded.

**Sibling `very-sharp-fp-slice-mixes-two-populations`** (medium) closes naturally.

### 8.8. `p14-data-fix-not-launched` (open, medium)

**Prereq:** Move 1 (linear probe, ~$0) + user authorization + **frozen split-manifest artifact (round-2 escalation)**.

**Do:** commit `R13_P14_DATA_FIX.yaml` + `tests/test_visomaster_enhanced_wiring.py`; smoke load enhanced viso samples; retrain; contract scorecard with corrected policy.

**Round-2 escalation — split as committed artifact, not user judgment alone (high).** The drafted YAML's generic holdout block is insufficient: `visomaster_enhanced_macro_dev` is 550 videos all from one Teams-recapture session, so naive frame-split or even loose identity-split contaminates the deployment proxy. Required before launch:

- **Frozen train / val / eval manifests** committed alongside the yaml, split jointly by identity AND capture-session AND source bucket (single axis is insufficient — same-session same-identity reappears across enhancers).
- **Manifest-overlap test in `tests/test_visomaster_enhanced_wiring.py`** that fails if any pair of (train, val, eval) shares any of {identity_key, session_id, source bucket path}. Test must run in CI and locally before the launch script proceeds.
- **Source-coverage proof per §7.3.1.** Either Path A (`visomaster_teams_enhanced.enabled: true`) committed in the yaml, or Path B (recorded distributional similarity proof) committed under `analysis/p14_source_coverage_*` before launch. Launch script must check for at least one of these.
- **Split policy + manifest hashes recorded in the run's immutable manifest** (see §10.12) so a future audit can reconstruct what train/eval looked like for this run.

**Cost:** ~$70 / 1-2d retrain + ~$10 / 3h scorecard. ~$0 / 1-2h Move 1 pre-gate. ~hours for split-manifest authoring + overlap-test wiring.

**Signal:** `visomaster_enhanced_macro_dev` recall lifted materially above P8A baseline (≥ 24% target under corrected policy), with the manifest-overlap test green and the source-coverage proof committed.

**Owner:** user (split policy choice); agent-actionable (manifest authoring + overlap test once policy chosen).

### 8.9. `webcam-mode-fpr-dominance-headline-misleading` (open, medium)

**Prereq:** modern_v2 filter audit (Plan v4 Track H per `webcam_fpr_dominance.md:114`).

**Do:** contract scorecard reports v2-filtered lockbox FPR as first-class metric alongside baseline; per-identity breakdown as hard sub-gate ("no single v2 identity FPR > 30%").

**Cost:** contract-scorer integration pass.

**Signal:** candidate with 5% baseline lockbox FPR but 0.7% v2 FPR is read as "deployment-FPR ≤ 5%."

### 8.10. `silent-feature-failures-pattern` (open, medium)

**Prereq:** none.

**Do:** single pre-launch CI/smoke/lint step verifying every yaml-declared trainer feature emits "ENABLED" log line at trainer init. For every nested-dict block in launched yaml that trainer reads via `self.config.get('<block>')`, runtime check fails fast (and CI test fails offline) if silently fall-through-disabled.

**Cost:** small CI script + tests.

**Signal:** any future "added-but-not-firing" instance fails fast at submission, not after $X spent on cancelled jobs.

### 8.11. Medium-tier follow-ups

- **`shortcut-block-criterion-may-be-scorer-artifact-bound`**: written disposition — reaffirm or revise close criterion under corrected policy.
- **`source-image-resolution-floor-not-applied-to-eval`**: small filter + rerun on canonical ckpt; fold into v2 or side-by-side.
- **`frame-level-vs-clip-level-scorer-mismatch`**: doc + analysis pass.
- **`split-mode-delta-unquantified`** (low): re-run RLP1_01 with `hash_stable` on same data snapshot.
- **`deploy-server-preprocessing-drift`** (medium): deployment server `http://34.16.217.28:8999` verified through `tests/test_inference_train_preprocessing_parity`-style guard; smoke probe reproduces post-fix anchor-pool number within ±0.02.
- **`residual-70-pct-fpr-gap-no-lever-past-rlp7`** (medium): Packet-7+ readout reports per-pool FPR spread closing ≥70% of cross-pool gap; OR documented next-lever proposal filed.
- **`per-identity-reducer-not-a-contract-gate`** (low): contract hard-gates `max(per-identity real_fpr) ≤ <budget>`.
- **`prior-leader-rescore-sweep`** (low): every pre-fix leader (RLP1_01, RLP2_02, RLP3_05, RLP3.5_02, RLP5_07, RLP6_04 — only RLP6_04 done) re-scored through `INTER_LINEAR` on matched eval suite.
- **`is-no-face-slice-is-data-degeneracy`** (low): 219 frames removed/relocated; 2026-04-27 "FN lift 2.09×" reading documented as data-degeneracy.

---

## 9. Recommended experiment slate

### Priority 1 — Move 1: Frozen-feature linear probe

**Source:** `viso_bucket_gap.md:25`, `STATE_2026-04-29.md:166`.

**Tests:** is P8A's eval-bucket viso AUC materially lower than train-bucket viso AUC, *not* attributable to identity / session / source-bucket shortcut?

Frozen-feature linear probe of P8A on eval bucket (`gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/`) vs training bucket (`live-...-cropped` viso).

**Protocol — revised, Codex round 1, MEDIUM:**
- **Grouped train/test split** by `(identity, capture_session)`; never share a group across the probe split. Without this, the probe trivially fits identity / camera-signature shortcuts and reports inflated train-bucket AUC.
- **Bucket variants probed (4-way, not 2-way):** {(a) training bucket viso, (b) eval bucket viso (the canonical gap), (c) `visomaster_teams_enhanced` direct conjunction source — gates §7.3.1 Path-A vs Path-B decision, (d) eval substrate re-cropped at production tightness — composes Move 1 with Move 1.5}.
- **Per-class report:** AUC, balanced accuracy, *and* a control logistic regression on identity labels alone (chance baseline) — if identity-only AUC ≥ 0.90, the probe is dominantly fitting identity, not bucket signal; rerun with stricter group exclusion or larger holdout.

**Cost:** ~$0 / 1-2h (now 3-4h with the 4 variants + identity control).

**Outcomes (revised):**
- **Diagnostic POSITIVE (necessary, not sufficient):** train-bucket AUC > 0.80 AND eval-bucket AUC ~0.65 AND identity-only control AUC < 0.70 → bucket gap is real *and* not identity-confounded → proceed to §7.3.1 Path-A/B decision → P14_DATA_FIX commit (Priority 5).
- **Diagnostic NEGATIVE:** bucket AUCs similar → gap not dominant in feature space → elevate P15 GRL.
- **Diagnostic AMBIGUOUS:** identity-only control ≥ 0.70 → probe results are confounded; do NOT use as P14_DATA_FIX gate. Re-run with stricter group split, or treat Move 1 as inconclusive and proceed on independent eval evidence.

**Decision:** the probe is *diagnostic*, not *dispositive*. POSITIVE result is a *necessary precondition* for spending $70 on P14_DATA_FIX, but the actual lift gate remains the held-out viso-enhanced+Teams group-split scorecard from §10.5(5b). Do not declare bucket-gap closed on probe alone.

**User decision required:** authorize Move 1 with the revised protocol. ~$0; informational; gates (not seals) P14_DATA_FIX investment.

### Priority 2 — Move 1.5: Re-crop at production tightness

**Source:** `STATE_2026-04-29.md:166`, `eval_production_crop_tightness_gap.md:69`.

**Tests:** does re-cropping lockbox sample at production tightness shift score distribution materially?

**Cost:** hours; no Vertex.

**Outcomes:**
- **Success (load-bearing):** distribution shifts → revise close criteria for `shortcut-deployment-block`; eval FPR readouts gain translation caveat.
- **Failure (marginal):** distribution stable → eval substrate is honest production proxy; current FPR numbers stand.

**Decision:** informational; runs in parallel to Move 1 → P14_DATA_FIX chain.

**User decision:** authorize Move 1.5. ~$0; informational on meta-risk.

### Priority 3 — Commit + ship contract-policy v3 fix

**Source:** `contract_policy_bug.md:108-122`.

**Tests:** does v3 fix produce representative scorecard with non-degenerate τ?

**Cost:** mechanical commit + image rebuild (auto-bumps VERSION) + ~3h Vertex run.

**Outcomes:**
- **Success:** scorecard documents τ via recall-floor path; P8A_REFERENCE_step5000 becomes rank 1 with τ=0.916, viso 13.6%, deeplive 23.9%, teams_fake 52.6%.
- **Failure:** re-scored numbers diverge from local re-score; investigate before launching subsequent P-series scorecards.

**Decision:** closes loops 8.2 and 8.3 by construction. **Standing precondition for every P-series scorecard read after this point.**

**User decision:** authorize commit + image rebuild + scorecard run. Mechanical.

### Priority 4 — P14_FT_FROM_P8A (in flight)

**Source:** `P14.md:22-36`, `STATE_2026-04-29.md:19-21`.

**Tests:** do P13's anti-shortcut interventions land additively on a working cross-domain base (FT-from-P8A_REFERENCE_step5000)?

Spec: init from P8A_REFERENCE_STEP5000; apply all 3 P13 anti-shortcut interventions; light FT (small LR, ~2K-4K steps to avoid catastrophic forgetting); validate against Day-4 contract on Day-6.

**Cost:** ~$60 / ~8h on A100 us-east1 → us-west4 fallback (running).

**Outcomes (`P14.md:62-67`):**
- **α (passes triple-axis):** ship P14, save GRL.
- **β (Axis 3 Δ ∈ [0.15, 0.45], cross-domain ≥ P8A):** launch P15.
- **γ (Axis 3 Δ ≥ 0.45 OR cross-domain < P8A by > 8pp):** launch P15.

**Decision:** verdict drives P15 launch (Priority 6). Verdict will likely look like P13's even if bundle works (bucket gap unchanged) — see `viso_bucket_gap.md` § 3.

### Priority 5 — P14_DATA_FIX (drafted, contingent on Move 1 success)

**Source:** `P14.md:38-51`.

**Tests:** does bucket-gap closure lift `visomaster_enhanced_macro_dev` recall ≥ 24% under corrected policy?

Spec: add `visomaster_enhanced_fake` family wiring in `combined_paired.py` (uses existing `VisoMasterEnhancedSample`); update yaml to enable `visomaster.enhanced_enabled: true` + family weights; image rebuild; smoke test (16 samples, shapes/labels); training (~$60 / ~8h FT-from-P8A); scorecard (~$10 / ~3h with `--promotion_target_fake_recall_min 0.30`).

**Cost:** ~$70 / 1-2d + ~$10 / 3h scorecard.

**Outcomes:**
- **Success:** viso ≥ 24% under corrected policy → L2 closed; pivot to L3.
- **Failure:** viso < 18% → bucket gap not dominant; reconsider heuristic; downgrade L2.

**Decision:**
- Success + α → ship P14, save GRL.
- Success + β → launch P15 (further Axis 3 movement).
- Success + γ → launch P15 as substrate's last cheap structural lever.
- Failure → reconsider; elevate P15 OR investigate Move 4.

**User decision:** authorize P14_DATA_FIX commit + launch (contingent on Move 1). ~$70 / 1-2d.

**Eval-split policy decision (load-bearing, concrete spec in §10.5):** if `R13_P14_DATA_FIX.yaml` wires eval bucket directly into training, `visomaster_enhanced_macro_dev` is 550 videos from one Teams session; if landed as-is, stops being deployment proxy. **Implement the §10.5 group-split spec verbatim** (joint key `(identity, session, source_bucket)`, ≥30% group holdout, manifest-overlap test, trainer-init intersection assertion, dual reporting). User judgment narrowed to approving the holdout fraction.

### Priority 6 — P15 GRL (drafted, contingent on P14 verdict)

**Source:** `P15.md:9-16`, `docs/relaunch_handoffs/P15_GRL_READINESS_NOTE_2026-04-29.md`.

**Tests:** does direct anti-shortcut supervision (DANN) break Axis 3 ceiling input-space perturbations couldn't?

Five-line yaml from P14_FT: `use_quality_domain_head: true` + `quality_domain_loss_weight: 0.20` + `quality_domain_require_labels: true` + `quality_domain_count: 4` + `quality_head_hidden_dim: 128`. Seed 1501.

**Cost:** ~$60 / ~8h.

**Outcomes (`P15.md:85-89`):**
- **Best:** Axis 3 Δ < P14 AND viso/deeplive recall ≥ P8A. First candidate where direct anti-shortcut supervision acts on working cross-domain base.
- **Modal (β):** static λ=0.20 likely moves Δ another 0.05-0.15, still above 0.15 gate, cross-domain preserved by FT init. Licenses P16 (ramped λ) or escalation to spectral/dual-stream.
- **Worst:** catastrophic cross-domain degradation (similar to P13_FROM_SCRATCH but different cause: backbone forced to throw away too much). **Mitigation:** kill switch step 1500 if viso recall drops > 8pp below P8A.

**Decision:**
- α (P14_FT passes): don't launch; ship P14.
- β/γ: launch P15.

**User decision:** authorize P15 launch per α/β/γ matrix. ~$60 / 8h.

**Round-2 escalation — pre-launch label-by-domain-by-source count gate (medium).** Existing pre-launch checklist (`P15.md:71-76`) only greps "ENABLED" log lines, not domain *populations*. P15's quality-domain mapping leaves **domain 3 (`social_media`) empty in training** (YouTube is OOD-monitor only) and may have other domains where a single source dominates — a GRL classifier on an empty or single-source domain optimizes a spurious objective while the smoke check passes. Add a smoke-time gate that emits and asserts, before the first optimizer step:
- Per-`quality_domain` × per-`source` × per-`label` (real/fake) sample count over the first ~256-batch warmup window.
- **Hard fail** if any active domain has zero training samples, or if any active domain has >90% concentration in a single source (label-confound surrogate).
- **Hard fail** if `quality_domain_loss` is identically zero across the warmup window (head not connected) or `> 1e3` (numeric blowup).
- Decide before launch: drop domain 3 (`quality_domain_count: 3`) or populate it from existing training surface; do not ship the count=4 yaml with an empty domain.

### Priority 7 — Substrate hygiene cleanup

**Source:** `sharpness_metric_bug.md:69-74`, `eval_substrate_data_hygiene.md:65-70`.

**Tests:** do these change any headline FPR/recall conclusion?

Three fixes: sharpness-metric on face crops; source-image-resolution floor; is_no_face slice removal/relocation.

**Cost:** short session.

**Outcomes:** most likely headline numbers don't move materially but downstream interpretations clarified (very-sharp-FP splits 2 populations; resolution floor catches degenerate inputs; is_no_face documented as data degeneracy). Less likely: a slice splits revealing previously-hidden failure mode worth packet-level intervention.

**Decision:** parallel to main chain. Closes 3 medium loops by construction.

**User decision:** prioritize relative to next packet, or accept partial confound.

### Priority 8 — Move 4 (same-identity pair training using proper-data wave)

**Source:** `STATE_2026-04-29.md:170`, `viso_bucket_gap.md:84-85`.

**Tests:** same-identity contrastive on `(clean[id_X], teams[id_X])` pairs — does explicit pairing help model distinguish "what should look the same vs different"?

2026-04-19 proper-data wave: 705 paired clean+teams identities; companion_bucket plumbing exists (`data/sources/visomaster.py:748,1085,1105,1176,1523`); `_load_merged_teams_enhanced_paired` exists.

**Cost:** TBD (yaml + smoke + ~$70 retrain).

**Position:** post-P14_DATA_FIX, post-P15. Move 4 in canonical sequence.

**Decision:** revival contingent on L2 closure + L3 status after P15 verdict.

### Priority 9 — Substrate-redesign last-resort

If P14_DATA_FIX + P15 + Move 4 still leave Axis 3 Δ > 0.30:
- Move 5: P15 GRL with broader pipeline label set the new data enables (`P15.md:64`).
- Representation loss work (`OPEN_LOOPS.md` `residual-70-pct-fpr-gap-no-lever-past-rlp7`).
- Architectural change (spectral/dual-stream features per `P15.md:88`).

Deferred. Wiki names no slate beyond P15 + Move 4; further levers speculative until bucket-gap closure status known.

---

## 10. Risk register

**10.1. Move 1 risk.** Probe is on eval-substrate features. If crop-tightness mismatch is large, signal might not translate. **Mitigation:** pair with Move 1.5. **Severity:** medium.

**10.2. Move 1.5 risk.** Small re-crop sample might not be representative. v2 already excludes some slices. **Mitigation:** stratify by `clip_capture_mode`; document sampling. **Severity:** low.

**10.3. Contract v3 commit risk.** Image rebuild + scorecard ~3h Vertex; selected τ might surprise. **Mitigation:** local re-score at `/tmp/p13_repolicy/recall_floor_30/` is reference; Vertex should reproduce within ε. **Severity:** low-medium. **Op note:** memory `feedback_promotion_contract_launch.md` requires `WANDB_API_KEY`/`WANDB_ENTITY`/`WANDB_PROJECT` exported (unlike `launch_batch_inference.sh`).

**10.4. P14_FT risk.** From-scratch trade may persist in light FT; cross-domain might not preserve. Anti-shortcut might land but not move Axis 3 enough (modal β; `P15.md:88`). **Mitigation:** kill switch on viso regression; α/β/γ structure permissive. **Severity:** medium.

**10.5. P14_DATA_FIX risk — eval split policy.** If `R13_P14_DATA_FIX.yaml` wires `visomaster_enhanced_macro_dev` directly into training, suite stops being deployment proxy. 550 videos all from one session. **Mitigation (concrete spec, Codex round 1, HIGH — replaces prior "user judgment, not delegable" placeholder):**

1. **Freeze the eval suite.** `visomaster_enhanced_macro_dev` v_2026-04-29 snapshot becomes immutable for the duration of P14_DATA_FIX evaluation; the manifest hash is recorded in the launch artifact.
2. **Group split.** Joint group key = `(identity, capture_session, source_bucket)`; minimum 30% of *groups* (not frames) held out for eval. Identity-only splits insufficient when same identity recaptured across sessions.
3. **Manifest-overlap pre-flight test** (extends `tests/test_visomaster_enhanced_wiring.py`): assert `set(train_groups) ∩ set(eval_groups) == ∅`, assert no shared `(identity, frame_id)` rows, assert no shared `(source_uri)` rows.
4. **Per-identity intersection assertion at trainer init** — fail-fast log if any train identity appears in eval-suite identity list, by intersecting with `arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json`.
5. **Held-out reporting.** Scorecard reports recall on (a) frozen `visomaster_enhanced_macro_dev` and (b) held-out viso-enhanced+Teams group-split internal eval; both must clear; (b) is the contamination canary.

User judgment now reduced to: approve the 30% holdout fraction (vs 20% / 40%) and confirm the joint-group key. Implementation is mechanical. **Severity:** high if any step skipped.

**10.6. P14_DATA_FIX risk — expected lift might not realize.** "Two-thirds bucket-gap" decomposition is heuristic. Substrate-matched ceiling ~24% bounded above by deeplive AUC; actual lift might be less. **Mitigation:** Move 1 = pre-spend gate. **Severity:** medium.

**10.7. P15 GRL risk.** First production exercise of GRL infra (`P15.md:50`). `detectors/effort_detector.py:236-274`, `combined_paired.py:65-82`, `effort_detector.py:906-958` cited as unit-tested (`tests/test_unpaired_reals_and_grl.py`) but never end-to-end on Vertex. **Test-coverage caveat (Codex round 1, MEDIUM, verified 2026-04-29):** `pytest tests/test_unpaired_reals_and_grl.py` currently fails at `TestVCDTargetedPreset.test_vcd_targeted_in_presets` with `ImportError: attempted relative import with no known parent package` originating in `data/augmentations/pipelines.py:21` — and the failure cascades to other tests in that file. The first 9 unit tests pass; downstream presets/yaml-instantiation tests do not. **Cited "unit-tested" claim is overstated** for the GRL-yaml + augmentation-pipeline integration paths.

**Pre-launch blocker (added):** before P15 Vertex launch, fix the relative-import path on `data/augmentations/pipelines.py` (likely needs package init or test-side `sys.path` shim) and require ALL tests in `tests/test_unpaired_reals_and_grl.py` to pass green; otherwise the "production-ready GRL path" claim is invalid and §7.4 smoke-run gate is the only safety net.

**Mitigation:** smoke test with explicit log-line greps (`P15.md:71-76`):
- `Quality domain head ENABLED with 4 domains, hidden_dim=128, loss_weight=0.2, require_labels=True`
- `Applied anchor_aware: enabled=True weight=5.0 target_mean_prob=0.1`
- `Applied face_scale_jitter: enabled=True scale_limit=0.25`
- `Applied periodic_saves: enabled=True step_list=[500, 1000, ..., 4000]`

Plus kill switch step 1500. **Severity:** medium.

**10.8. P15 GRL risk — λ static at 0.20.** `GradientReversalLayer.set_lambda(val)` exists; no caller in `trainer/trainer.py`. Ramping 0→target requires ~10-line trainer patch (`P15.md:51-52`). Static-λ is deliberate first-cut; ramp is planned P16. **Severity:** low.

**10.9. Substrate-redesign overshoot risk.** P14_DATA_FIX + P15 + corrections might still leave Axis 3 Δ > 0.30. Wiki slate beyond P15 (Move 4, Move 5) speculative. **Mitigation:** Move 4 → Move 5 → representation loss → architectural change. **Severity:** medium.

**10.10. META-RISK — entire reframing wrong directionally.** What if "low recall = scorer artifact" is false and model is genuinely weak cross-domain?

**Counter-evidence load-bearing:**
- Frame-level AUCs computed directly on cached predictions (`STATE_2026-04-29.md:74-77`).
- Numbers consistent across `(1 − AUC)` margin inversion and corrected-policy operating-point recall.
- Three independent τ-policies (legacy ~0.99, calibrated 5%-FPR, τ=0.5 diagnostic) tell consistent stories on P13 verdict (`P13.md:64`).

**But:** AUCs computed on same eval substrate flagged in §1.6. Treating frame-level AUC 0.91 on teams_fake as strict production-deployment floor inherits caveats (`STATE_2026-04-29.md:88-89`). If eval-vs-production crop-tightness delta large, reframing partially survives but absolute magnitude shifts.

**Mitigation:** Move 1.5 is cleanest test. Materially-shifting distribution → reframing partially overstated; stable → reframing stands.

**Severity if fires:** entire forward plan mis-priced (L2 expected lift overstated; L3 shortcut more dispositive than wiki implies).

Wiki stance: directionally correct, substrate has additional load-bearing caveats Slice 7 didn't surface. Corrected-policy numbers load-bearing for relative comparisons; absolute production-floor interpretation shifts.

**Codex round 1 sharpening:** any forward decision quoting frame-level AUC as a quantitative target (substrate-matched ceiling 24%, two-thirds bucket-gap heuristic, etc.) is calibrated against a metric whose translation to clip-level deployment readout is unresolved. §7.1 reconciliation must close before treating these targets as anything more than priors. Until then, P14_DATA_FIX success/failure is decided on clip-level scorecard recall, not on whether AUC moves toward 0.91.

**10.11. Secret handling — W&B credentials (round 2, HIGH).** `WANDB_API_KEY` is treated as exported launch env (memory `feedback_promotion_contract_launch.md`); some launch / Cloud Build paths additionally carry hardcoded API-key fallbacks or inject the key into generated job YAML, where it can land in build logs, GCS-cached artifacts, or `gcloud ai custom-jobs describe` output. Same pattern likely applies to any GCP service-account JSON staged into images. **Mitigation:**
- Treat the currently-exported key as compromised; rotate it.
- Remove any hardcoded fallbacks from `scripts/launch/*` and `cloudbuild*.yaml`; route through GCP Secret Manager or Cloud Build `secretEnv` so the value is never written into artifact YAML or persisted logs.
- Audit `entrypoint.sh` and the image-build path for env-leak surfaces (`env`, `set -x`, `echo $WANDB_*`).
- Add a CI check that fails if `WANDB_API_KEY` (or anything matching the W&B key prefix) appears in tracked files or generated job specs.

**Severity:** high (credential exposure compounds across every launch and is hard to remediate after-the-fact).

**10.12. Scorecard / promotion-contract output immutability (round 2, MEDIUM).** Scorecard runs currently write multiple GCS objects under predictable, overwritable paths with no immutable run manifest, no GCS generation precondition, no COMPLETE sentinel. A retry, mid-run crash, or reused `--job-name` can leave the path with mixed-vintage objects (some from run A, some from run B) that look authoritative to a downstream reader. **Mitigation:**
- Each promotion-contract / scorecard run writes under a unique immutable run ID (e.g., `runs/<timestamp>-<short-sha>-<random>/`).
- Run manifest committed at run start: git SHA, image digest, config + manifest hashes, full CLI flag set including `--promotion_target_fake_recall_min` (closes the §8.2 audit trail).
- Intermediate writes go to `tmp/...` with `x-goog-if-generation-match: 0` preconditions; finalize via copy on completion.
- Publish a `_COMPLETE` sentinel only after all expected files verify (size + hash). Downstream readers check for `_COMPLETE` before consuming.

**Severity:** medium.

**10.13. Checkpoint upload data-loss under transient failure (round 2, MEDIUM).** Trainer's checkpoint code path logs upload failures and deletes the local file post-attempt while training continues, so a transient GCS outage during a `periodic_*.pth` upload silently loses that step's checkpoint. Compounds with the periodic_saves silent-failure history (§5.1) — same evidence surface, different failure mode. **Mitigation:**
- Retry with checksum (md5/crc32c) verification on upload; fail-closed on persistent verify-mismatch.
- Keep local checkpoint until upload confirmed (treat upload as sink-write, not fire-and-forget).
- For required uploads (best, periodic at gate steps), fail the trainer job on persistent upload failure rather than continuing — a job that "completed" without its required checkpoints is a worse outcome than a visibly-failed one.

**Severity:** medium (worst case: P14_FT or P14_DATA_FIX completes, gate-step checkpoint is gone, must re-spend).

---

## 11. Open questions for the user

Per memory `feedback_decision_points.md`: present recommendation + tradeoffs, wait for explicit pick.

### 11.1. User-decision (judgment calls)

1. **Authorize Move 1?** ($0, 1-2h) — gates P14_DATA_FIX investment. **Recommendation (`STATE_2026-04-29.md:172`):** yes; cheapest decisive step on largest pending uncertainty.
2. **Authorize contract v3 commit + image rebuild + representative scorecard?** Closes 8.2 + 8.3 by construction. Wiki built specifically to prevent attempt #4. **Recommendation:** yes.
3. **Authorize P14_DATA_FIX commit + launch?** Contingent on Move 1. ~$70 / 1-2d. **Recommendation (`STATE_2026-04-29.md:169`):** if Move 1 confirms bucket gap dispositive, yes.
4. **Authorize P15 launch?** Per α/β/γ matrix on P14 verdict. ~$60 / 8h. **Recommendation (`P15.md:60-66`):** β/γ → launch; α → don't.
5. **Eval-split policy for P14_DATA_FIX:** §10.5 specifies the mechanical implementation (joint group key, ≥30% holdout, manifest-overlap test, trainer-init assertion, dual reporting). User judgment narrowed to: pick holdout fraction (recommend 30%) and confirm joint group key includes `capture_session` (recommend yes — eval suite is one Teams session).
6. **Cancel in-flight P14_FT once Move 1 lands?** Per memory `feedback_no_cancelling_vertex_jobs.md`, default no — destructive class. P14_FT tests separate hypothesis ("can bundle land additively on working cross-domain base?"); defensible $60 to let finish even if Move 1 redirects.
7. **Authorize Move 1.5 (re-crop)?** ~$0, hours; informational on meta-risk. **Recommendation (`STATE_2026-04-29.md:166`):** yes; runs parallel to Move 1.
8. **Sharpness-metric fix priority** — block on it before next packet, or accept partial confound? **Recommendation:** do not block — fold into routine analytics maintenance pass alongside resolution floor + is_no_face cleanup (Priority 7).
9. **Source-image-resolution floor disposition** — fold into v2 or report side-by-side? **Recommendation:** side-by-side first; fold once delta documented.
10. **Disposition on `shortcut-block-criterion-may-be-scorer-artifact-bound`** — reaffirm or revise close criterion under corrected policy? **Recommendation:** defer until P14 verdict + Move 1.5; needs both inputs.

### 11.2. Agent-actionable (standing approval)

- Run regenerator script after thread updates (`tools/regenerate_open_loops.py` per `AGENTS.md:101-107`).
- Re-tag parquet after sharpness fix.
- Document cross-thread caveats as new findings land.
- Update memory frontmatter `last_verified` after each session (`AGENTS.md:89-99`).
- Verify working-tree state matches threads' "fix in progress" annotations before any code change (`AGENTS.md:50-58`).

### 11.3. Decision matrix (`STATE_2026-04-29.md:128-138`)

| Decision | Recommendation | User action |
|---|---|---|
| Authorize Move 1 (linear probe) | Run before P14_DATA_FIX commit; ~$0 | OK / not OK |
| Authorize contract v3 commit + image rebuild | Commit per close criterion | OK / push back |
| Authorize P14_DATA_FIX (assuming Move 1 confirms) | Highest-priority remaining unbuilt experiment | OK / not OK / change scope |
| Authorize P15 launch (post P14 verdict) | β/γ → launch per matrix; α → don't | OK / not OK |
| Eval-split policy for P14_DATA_FIX | Identity-split out held-out portion | User judgment (not delegable) |
| Cancel in-flight P14_FT once Move 1 lands? | Tests separate hypothesis; defensible $60 | OK / cancel |

---

## 12. Decision tree

### 12.1. Inputs

Branches off two independent signals:
- **Move 1 verdict** (success/failure) — gates P14_DATA_FIX commit.
- **P14_FT verdict** (α/β/γ) — gates P15 launch.

Plus parallel precondition gating **everything** that uses contract scorecards:
- **Contract v3 commit + image rebuild + scorecard verify** — must precede any P-series scorecard read.

### 12.2. Tree

```
Contract v3 commit (Priority 3)
├── DEFERRED: every P-series scorecard inherits τ-tail-collapse regime; manual τ=0.5 fallback per session
└── DONE: scorecards produce non-degenerate τ; P-series readouts trustworthy
    │
    └── (parallel) Move 1 (Priority 1) + Move 1.5 (Priority 2)
        │
        ├── Move 1 SUCCESS (train-bucket AUC >> eval-bucket AUC)
        │   │
        │   └── commit + launch P14_DATA_FIX (Priority 5)
        │       │
        │       ├── P14_DATA_FIX SUCCESS (viso recall ≥ 24% under corrected policy)
        │       │   │
        │       │   ├── P14_FT verdict α: ship P14_FT, save GRL for next round
        │       │   ├── P14_FT verdict β: launch P15 GRL with goal of further Axis 3 movement
        │       │   └── P14_FT verdict γ: launch P15 GRL as substrate's last cheap structural lever
        │       │
        │       └── P14_DATA_FIX FAILURE (viso recall < 18%)
        │           │
        │           ├── reconsider: bucket gap not dominant; heuristic 2/3 share overstated
        │           ├── elevate P15 priority (GRL bet now has higher expected gain)
        │           └── consider Move 4 (paired-transport contrastive) as alternative data lever
        │
        └── Move 1 FAILURE (bucket-bucket AUC similar)
            │
            ├── downgrade bucket-gap finding to "real but not dominant"
            ├── elevate P15 GRL to top priority (post-P14_FT verdict)
            └── (orthogonal) consider Move 1.5 scope-extension
                │
                ├── Move 1.5 SUCCESS (score distribution shifts under re-crop)
                │   └── crop-tightness gap is load-bearing
                │       └── revise `shortcut-deployment-block` close criterion to production-translation-aware version
                │
                └── Move 1.5 FAILURE (distribution stable under re-crop)
                    └── eval substrate is honest deployment proxy
                        └── focus moves to L3 (residual shortcut) entirely
```

### 12.3. Side-channels and gates

- **P14_FT verdict drives P15 launch independent of Move 1.** P14_FT in flight at end of Slice 7; verdict ~24h regardless of Move 1 status. α/β/γ matrix at `P15.md:60-66` is canonical.
- **Move 4 sits below P14_DATA_FIX and P15 in canonical priority** (`viso_bucket_gap.md` § 6). Data-axis lever if P14_DATA_FIX delivers substrate-matched ceiling but residual shortcut at Axis 3 still fires modern_v2 FPR > 5%.
- **Substrate hygiene (Priority 7) parallel** to main tree. Doesn't gate; provides cleaner readouts for downstream substrate-level decisions.

### 12.4. Naming disambiguation (Codex round 1, LOW)

The decision tree uses three artifact names sharing the "P14" prefix; they refer to *distinct* yamls, runs, scorecards, and verdicts. **Do not let them collapse to "P14" in launch checklists, scorecard outputs, or W&B run names.** Canonical mapping:

| Short | Yaml | Status | Tests | Verdict drives |
|---|---|---|---|---|
| `P14_FT` | `R13_P14_FT_FROM_P8A.yaml` | in-flight (Day-5 fallback FT-only) | α/β/γ matrix at `P15.md:60-66` | P15 launch decision |
| `P14_DATA_FIX` | `R13_P14_DATA_FIX.yaml` | drafted (bucket-gap retrain) | success/failure on §10.5(5b) held-out group split | L2 closure |
| `P14` (umbrella) | — | informal shorthand | — | **DO NOT USE in artifacts**; always disambiguate |

Launch-script hygiene: every `--job-name`, every yaml filename, every scorecard `--output-path`, every W&B run name must contain either `P14_FT` or `P14_DATA_FIX` (never bare `P14`). Add to pre-launch checklist.

### 12.5. Standing operational discipline

Every node assumes:
- Working-tree-diff check before any commit (`AGENTS.md:50-58`).
- Image-currency check before any Vertex launch (commit `ad76cd8`, `scripts/launch/check_image_currency.sh`). **Round-2 escalation (medium): fail-closed for promotion + paid training launches.** The current guard is fail-open on missing `gcloud` answers, missing local files, untagged images, or `SKIP_IMAGE_CURRENCY_CHECK=1` — under retry pressure this can ship stale code/config. For paid runs (training + promotion-contract scorecards), require an explicit audited override that records image digest and config hashes; refuse launch when any of those signals is missing rather than warning. Diagnostic / dry-run paths may stay fail-open.
- US-region preference: `us-west4` ↔ `us-east1` ↔ `us-central1` only; non-US requires explicit user authorization (`CLAUDE.md`).
- 30-minute pending threshold for region switch; never cancel original until replacement RUNNING.
- WANDB env vars exported before any contract-scorecard launch (memory `feedback_promotion_contract_launch.md`). **Per §10.11 (round 2):** prefer Secret Manager / Cloud Build `secretEnv` over a long-lived exported key once the rotation lands.
- No cancellation of in-flight Vertex jobs without explicit user authorization (memory `feedback_no_cancelling_vertex_jobs.md`).
- **Run-manifest + COMPLETE sentinel (§10.12)** for every promotion-contract / scorecard write; downstream readers check `_COMPLETE` before consuming.

These are conditions, not nodes.

### 12.6. Modal expected end-to-end path

1. **User approves Move 1 + Move 1.5.** Parallel; ~$0; hours.
2. **Move 1 succeeds** (train-bucket viso AUC > 0.80, eval-bucket ~0.65). Bucket gap dispositive.
3. **Move 1.5 informational.** Either confirms eval substrate honest proxy (status quo for `shortcut-deployment-block` criterion) or flags translation gap (caveat added).
4. **User approves contract v3 commit.** Mechanical; ~3h Vertex validates.
5. **User approves P14_DATA_FIX commit + launch.** ~$70 / 1-2d. Identity-split done; eval suite preserved.
6. **P14_FT verdict lands** (parallel; ~24h from launch). Modal: β. Anti-shortcut land additively but Axis 3 Δ stays in [0.15, 0.45]; cross-domain ≥ P8A.
7. **User approves P15 launch** (β branch). ~$60 / 8h.
8. **P15 GRL verdict lands.** Modal: β again. Δ moves another 0.05-0.15, still above 0.15 gate, cross-domain preserved. Licenses P16 (ramped λ) or escalation to spectral/dual-stream.
9. **P14_DATA_FIX scorecard lands.** Viso recall ≈ 24% under corrected policy. L2 closed.
10. **Substrate hygiene parallel.** Sharpness fix, resolution floor, is_no_face dispositioning. Headline numbers don't move materially; downstream interpretations clarified.

End state: contract v3 in image; P14_DATA_FIX bucket-gap closed (viso ~24%); P14_FT verdict β; P15 verdict β; P16 drafted; Move 4 in queue. Axis 3 Δ residual ~0.30-0.40; modern_v2 FPR still elevated relative to P8A baseline. Deployment block not yet closed; structural anti-shortcut work continues.

### 12.7. Unhappy paths

- **Move 1 fails:** bucket gap not dominant. Plan reverts to "P15 GRL top priority, possibly broader pipeline labels." 6 weeks of recipe-tuning partially redeemed (recipes under-powered against non-bucket-gap-dominant cause; P11/P13 anti-shortcut directionally correct).
- **P14_FT verdict α:** ship P14, no P15. Plan compresses.
- **P14_DATA_FIX fails:** heuristic 2/3 share overstated. Bucket gap real-but-marginal OR substrate-matched ceiling argument wrong. Plan moves to Move 4 + parallel substrate-redesign.
- **P15 catastrophically degrades cross-domain:** kill switch step 1500 fires. Mitigation: lower loss weight or remove `quality_domain_require_labels` safety belt. P16 with ramped λ next bet.
- **Meta-risk fires:** Move 1.5 surfaces if score distribution shifts materially under production-tightness re-crop. Plan re-prices: L2 expected lift downgraded; L3 shortcut elevated; modern_v2 FPR readouts gain translation caveat.

---

## 13. Contradictions found in the wiki

**No load-bearing contradictions found** after a careful read of source-of-truth surfaces (AGENTS.md, STATE_2026-04-29.md, OPEN_LOOPS.md, TIMELINE.md, 11 specified threads, P13/P14/P15 retros).

Five candidates audited and resolved as not-contradictions:

1. **Open-loop count discrepancy.** STATE says "19 open + 3 in-progress + 7 resolved" (`STATE_2026-04-29.md:122`); OPEN_LOOPS shows "Open: 24, In progress: 3, Resolved: 7." Reconciliation: STATE notes "audit adds 5 new open loops"; 19+5=24. Not a contradiction.
2. **`viso_bucket_gap.md` 2/3-bucket-gap heuristic vs frame-level AUC ceiling.** Heuristic decomposition (`viso_bucket_gap.md:29`) coexists with AUC ceiling (`viso_bucket_gap.md:19`). Heuristic is residual headline-metric gap decomposition; AUC is upper bound on absolute viso recall under bucket-fix. Consistent.
3. **`shortcut-deployment-block` close criterion measured on eval substrate vs `eval-production-crop-tightness-mismatch` loop.** Acknowledged via paired sub-loop `shortcut-block-criterion-may-be-scorer-artifact-bound` (`processing_signature_shortcut.md:147-156`). Explicit cross-thread caveat, not contradiction.
4. **P13 retro "post-fix relaunch" timing.** `P13.md:9` references "2026-04-28 EOD (post-`c366026` relaunch) → 2026-04-29 02:30 UTC (Day-4 verdict γ)." TIMELINE 2026-04-28 EOD entry confirms post-`c366026` relaunch fired all 3 new yaml blocks live; verdict 02:30 UTC 2026-04-29. Consistent.
5. **`silent-feature-failures-pattern` first_seen 2026-04-28** but in_proj-SVD bug (instance #1) fixed 2026-04-26. Resolution: meta-loop authored 2026-04-28 retroactively grouping prior instances; `first_seen` is consolidation date, not any particular instance date. Not a contradiction.

Wiki appears coherent. Codex pass + user compression may surface load-bearing contradictions this read missed; if so, list here in numbered update with citations to disagreeing surfaces.

---

## Appendix B: Citation index

Every load-bearing claim traces to one of these surfaces.

### B.1. Wiki orienting surfaces

- `docs/packet_retrospectives/AGENTS.md` (read/update protocol; lines 42-110)
- `docs/packet_retrospectives/STATE_2026-04-29.md` (orienting; :74-77 frame-level AUC, :80-86 corrected-policy recall, :91-100 P13 verdict, :104-122 top-priority loops, :128-138 decisions deferred, :161-170 recommended next actions)
- `docs/packet_retrospectives/OPEN_LOOPS.md` (regenerated 2026-04-29; mechanical inventory)
- `docs/packet_retrospectives/TIMELINE.md` (chronological master index, append-only)

### B.2. Threads cited

- `threads/contract_policy_bug.md` (flagship; :7-23 attempts #1-3, :29-66 v3 fix, :108-122 close criterion + whack-a-mole)
- `threads/processing_signature_shortcut.md` (:5 question, :13-16 origin, :18-26 P8A breakthrough, :28-43 codec_hedge + face-size, :47-76 frame-level AUC reframing + P13 γ verdict, :91-97 audit caveat, :147-169 close criterion sub-loops)
- `threads/eval_production_crop_tightness_gap.md` (most consequential audit; :14-22 mechanism, :42 close path, :62-69 open loop)
- `threads/sharpness_metric_bug.md` (:14-26 mechanism, :30-44 stance, :69-85 open loops)
- `threads/eval_substrate_data_hygiene.md` (:17-21 audit, :28-35 stance, :56-72 open loops)
- `threads/viso_bucket_gap.md` (:15-19 dispositive, :29-35 stance, :61-79 open loops + frame-level AUC reconciliation, :81-87 cross-thread refs)
- `threads/identity_audit.md` (:18-26 dispositive identity-count table, :30-34 stance, :47 evidence)
- `threads/face_size_label_leak.md` (:15-37 evidence, :43-58 stance + Slice 7 update, :87-94 open loop)
- `threads/webcam_fpr_dominance.md` (:17-50 v2 sweep, :64-72 stance, :76-88 audit caveats, :107-117 open loop)
- `threads/in_proj_svd_gradient_bug.md` (:13-22 discovery + Phase C, :23-30 stance, :53-69 open loops)
- `threads/wandb_flattening.md` (:17-37 Slice 6, :39-46 stance, :69-94 open loops)

### B.3. Packet retros

- `packets/P13.md` (status :7-16, results :54-93, conclusions :95-114)
- `packets/P14.md` (:6-16 status, :22-56 two variants spec + Move 1 pre-spend gate)
- `packets/P15.md` (:7-16 status, :20-66 readiness + decision logic)

### B.4. Memory entries (cited via wiki)

- `project_p8a_frame_level_auc_2026-04-29.md` (Slice-7 reframing anchor)
- `project_contract_policy_bug.md` (whack-a-mole anchor)
- `project_signature_shortcut_finding.md` (Slice 4 origin)
- `project_p8a_breakthrough.md` (CORRECTION 2026-04-26)
- `project_shortcut_is_upstream.md` (CAVEAT 2026-04-26)
- `project_face_size_label_leak.md` (Slice 6 origin)
- `project_lockbox_fpr_dominated_by_webcam_mode.md` (Slice 6 webcam-mode)
- `project_wandb_flattens_nested_dicts.md` (rule + 14-key allowlist)
- `project_in_proj_svd_gradient_bug.md`
- `project_data_inventory_identity_diversity.md` (Slice 7 audit)
- `project_clean_teams_same_identity.md` (Slice 7 paired-transport)
- `project_viso_train_eval_bucket_gap.md` (Slice 7 dispositive)
- `project_promotion_contract.md`
- `feedback_decision_points.md`
- `feedback_no_cancelling_vertex_jobs.md`
- `feedback_promotion_contract_launch.md`
- `reference_image_rebuild.md`

### B.5. External handoffs (cited via wiki)

- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md`
- `docs/relaunch_handoffs/P13_DAY4_VERDICT_2026-04-29.md`
- `docs/relaunch_handoffs/TRAINING_EVAL_VISO_BUCKET_GAP_2026-04-29.md`
- `docs/relaunch_handoffs/P15_GRL_READINESS_NOTE_2026-04-29.md`
- `docs/relaunch_handoffs/WANDB_FLATTENING_BUG_HANDOFF_2026-04-28.md`

### B.6. Source files (cited via wiki)

- `arena/score_teams_promotion_contract.py:456-479` (`_threshold_sort_key`)
- `arena/score_teams_promotion_contract.py:510-531` (`_promotion_summary_sort_key`)
- `arena/run_target_domain_validation_sequential.py` (+19 CLI args)
- `arena/checkpoint_maps/teams_target_domain.r13_rlp5_07_sanity_2026-04-23.yaml`
- `data/sources/visomaster.py:605-890` (`VisoMasterEnhancedSample`)
- `data/sources/combined_paired.py:646` (`create_unified_samples_from_visomaster_enhanced`)
- `data/sources/combined_paired.py:65-82` (`QUALITY_DOMAIN_MAP`)
- `train_sweep.py:172-282` (pre-`c366026` allowlist; 11 blocks)
- `train_sweep.py:283-313` (post-`c366026` allowlist; 14 keys)
- `train_sweep.py:322-349` (P15 quality-domain fail-fast guards)
- `tests/test_train_sweep_reapply_allowlist.py` (Slice 6 regression guard)
- `tests/test_score_teams_promotion_contract.py` (working-tree v3; +125 lines)
- `tests/test_visomaster_enhanced_wiring.py` (untracked; 3 wiring tests)
- `tests/test_unpaired_reals_and_grl.py` (P15 unit)
- `analysis/lockbox_tagging/layers/quality.py:80-82` (sharpness bug)
- `analysis/lockbox_tagging/full_tags_2026-04-27.parquet`
- `analysis/modern_lockbox_v2_2026-04-27/build_modern_subset.py`
- `analysis/probe_battery_2026-04-26/grad_audit.py` (in_proj-SVD regression guard)
- `detectors/effort_detector.py:236-274` (quality-domain head + GRL hook)
- `detectors/effort_detector.py:906-958` (trainer loss aggregation for GRL)
- `scripts/launch/check_image_currency.sh` (commit `ad76cd8`)
- `experiments/phase2_round13/R13_P13_FROM_SCRATCH.yaml` (commit `cab2909`)
- `experiments/phase2_round13/R13_P14_FT_FROM_P8A.yaml` (in flight)
- `experiments/phase2_round13/R13_P14_DATA_FIX.yaml` (untracked)
- `experiments/phase2_round13/R13_P15_GRL_FROM_P8A.yaml` (untracked)

### B.7. Commits cited

- `c7dc828` (periodic_saves trainer + yaml; no allowlist)
- `cab2909` (P13 anti-shortcut interventions)
- `c366026` (wandb-flattening fix; 3 keys + test guard)
- `6c9a320` (image 1.3.226 bump)
- `2feea58` (in_proj-SVD silent-zero-grad fix)
- `7af72b1` (anchor-pool monitor + grad_audit + manifest_overlap)
- `ac83ba1` (VERSION 1.3.216, first image with in_proj-SVD fix)
- `872502c` (Slice 3 train_sweep value_composite allowlist)
- `f366368` (Slice 5 wandb artifact name >128)
- `855871e` (WS-P0 preprocessing parity)
- `deac44e` (WS-P1 calibration probe + WS-P2.b per-identity reducer)
- `ad76cd8` (pre-launch image-currency guard)
- `2c9778b` (P10 anti-shortcut packet authored; 6 GRL yamls drafted, never launched)

### B.8. Citation discipline note

This document was compressed from `MASTER_PLAN_2026-04-29.md` (1527 lines), which was written from the wiki only. No source files re-surveyed independently of wiki citations. Where claims required file:line check (e.g., contract-policy v3 line numbers), the wiki's cited surface was used directly. Claims without a wiki cite should be considered inferential and flagged on the next pass.

---

*End. Source: `docs/packet_retrospectives/` as of 2026-04-29 PM. Compressed 2026-04-29 from `MASTER_PLAN_2026-04-29.md`. Companion long-form audit-trail at `docs/packet_retrospectives/plans/MASTER_PLAN_2026-04-29.md`.*

---

## Changelog

### Round 1 — Codex adversarial review, 2026-04-29

Findings file: `.claude/claudex/20260429-174012-ee98e2/findings-round-1.md`.

**Accepted (8 of 8):**

1. **HIGH — P14_DATA_FIX source coverage mismatch.** Yaml enables `visomaster_enhanced` (enhancers, no Teams) + `visomaster_hints_teams` (Teams, no enhancers) — the *union* of two single-axis sources. The eval substrate is the *conjunction* (enhancer×Teams). A separate source `visomaster_teams_enhanced` is already wired at `combined_paired.py:704` (`discover_visomaster_teams_enhanced_samples` at `data/sources/visomaster.py:1044`) but **not enabled** in the yaml. **Resolution:** added §7.3.1 with two paths — Path A (default): enable `visomaster_teams_enhanced` directly; Path B: empirically prove union ≈ conjunction via feature-similarity probe. Updated §1.5, §2.6, Priority 5 references. Verified by direct grep of the yaml and `combined_paired.py`.

2. **HIGH — Recall floor disabled by default.** v3 contract fix as drafted preserves backcompat default `target_fake_recall_min=0.0`; any future scorecard omitting `--promotion_target_fake_recall_min 0.30` silently re-enters τ-tail-collapse regime. **Resolution:** escalated from "future-slice" to commit-blocker. Updated §5.4 / §8.3 / Priority 3 to require either default-flip to 0.30 (with `--legacy_no_recall_floor` opt-out) OR fail-fast assertion + CI test, before v3 lands on main.

3. **HIGH — Eval-split policy was a placeholder.** Prior plan said "user judgment, not delegable" without spec; an implementer could leak eval substrate into training by frame-split. **Resolution:** §10.5 now specifies joint group key `(identity, capture_session, source_bucket)`, ≥30% group holdout, manifest-overlap pre-flight test, trainer-init intersection assertion against `arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json`, and dual reporting (frozen public eval + held-out internal canary). User judgment narrowed to holdout fraction + group-key confirmation. Mirrored into Priority 5 and §11.1(5).

4. **MEDIUM — Move 1 probe protocol underspecified.** Original probe could trivially fit identity / camera-signature shortcut and report inflated train-bucket AUC. **Resolution:** Priority 1 now requires grouped train/test by `(identity, capture_session)`, 4-way bucket variant (training viso, eval viso, `visomaster_teams_enhanced` direct, production-tightness re-crop), an identity-only control, and explicit downgrade from "dispositive" to "necessary-not-sufficient diagnostic." Actual lift gate is held-out group-split scorecard from §10.5(5b).

5. **MEDIUM — P15 GRL pre-launch coverage gates missing.** Domain 3 (`social_media`) has no training source; quality-domain labels likely confound source-family labels; no requirement to verify per-domain-by-source counts pre-launch. **Resolution:** §7.4 adds (1) label×domain×source coverage table at trainer init, (2) explicit domain-3 disposition decision (drop to 3 / mask / accept-as-monitor-only), (3) frozen-feature confound check (probe quality-domain labels from P8A features and compare to source-family accuracy), (4) 200-step smoke run gating full launch on nonzero `quality_domain_loss` and gradient norm.

6. **MEDIUM — GRL test claim overstated.** Plan cited `tests/test_unpaired_reals_and_grl.py` as production-readiness coverage, but `pytest tests/test_unpaired_reals_and_grl.py` fails at `data/augmentations/pipelines.py:21` with `ImportError: attempted relative import with no known parent package`; first 9 unit tests pass, downstream presets/yaml-instantiation tests fail. **Verified directly with pytest.** **Resolution:** §10.7 adds caveat + pre-launch blocker requiring all tests in that file pass green before P15 Vertex launch; §2.4 amended with the caveat.

7. **MEDIUM — Frame-level AUC mis-cast as launch gate.** Plan's "13.6% → ~24%" lift target derived from frame-level AUC ceiling, but frame↔clip reconciliation loop (§7.1) is open. **Resolution:** §1.1 + §10.10 + Priority 1 now explicitly mark frame-level AUC as *sanity check only*; all promotion / lift / close-criterion decisions remain on clip-level corrected-policy contract scorecards; frame-level numbers are directional priors, not quantitative gates.

8. **LOW — Naming overload.** "P14", "P14_FT", "P14_DATA_FIX" used interchangeably in decision branches. **Resolution:** new §12.4 "Naming disambiguation" with canonical mapping table + launch-script hygiene rule (every `--job-name`, yaml filename, scorecard `--output-path`, and W&B run name must contain `P14_FT` or `P14_DATA_FIX`, never bare `P14`); §12.5–12.7 renumbered accordingly.

**Rejected:** none. All 8 findings were either dispositively right (1, 2, 3, 6 verified directly) or material-on-balance (4, 5, 7, 8 — addressed even where the original framing had partial defense, because the cost of wrong is high and the cost of being explicit is low).

**Net effect on plan:** P14_DATA_FIX is now correctly framed as gated on (a) source-coverage decision in §7.3.1, (b) §10.5 group-split implementation, (c) clip-level scorecard not frame-level AUC. Contract v3 commit is now correctly framed as gated on default-safety patch. P15 launch is now correctly framed as gated on test-suite green + per-domain coverage table + smoke-run nonzero-loss check.
