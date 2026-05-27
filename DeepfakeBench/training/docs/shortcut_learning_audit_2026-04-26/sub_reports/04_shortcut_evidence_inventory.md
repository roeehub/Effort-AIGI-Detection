# Sub-Report 04 — Shortcut Evidence Inventory

**Date authored:** 2026-04-26
**Audience:** Receiving agent reasoning about what we know vs suspect about shortcut learning in the Effort Teams detector
**Scope:** Every measurement, observation, and analytical artifact bearing on shortcut learning, identity overfitting, camera-signature shortcuts, and quality shortcuts produced in the R13 program (current data spans 2026-04-22 → 2026-04-26).

---

## 0 — Reading guide

The detector chain we are auditing:

| Tag | What it is | Provenance |
|---|---|---|
| RLP6_04 | Production baseline for R13. R12g → light FT (head + attention SVD only). Frozen CLIP backbone. | `gs://training-job-outputs/phase2r13_experiments/h2pdu6i5/value_composite_effort_20260424_step23500_auc0.9942_eer0.0169.pth` |
| RLP7_02..08 | Six aug-axis variants forked from RLP6_04 step 23500 (RLP7_08 from step 4500). FT-only, frozen backbone. | `phase2r13_experiments/{f5fav9u0, 8wb3dsxq, hhc8quq9, g1kuc91c, 1zz8mvhf, u6jphwku}/...` |
| P8A | RLP7_02 + `unfreeze_final_proj`, `unfreeze_final_ln`, `apply_svd_to_mlp` | `phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth` |
| P8B | Scratch on plain CLIP (no R12g/R13 chain), step 11000 (job hung at step 12000) | `phase2r13_experiments/n8yk2hox/value_composite_effort_20260425_step11000_*.pth` |
| P9 series | Five "softened P8A" single-variable variants + 5 longshot pack | `phase2r13_experiments/{5oc7zyzw, ku6fdljz, a5w629y1, 6pd7o8iz, ...}` |
| P10 | New axis (anti-shortcut): symmetric routing + GRL probes | `phase2r13_experiments/{osji02ho, ntbx1hh1, 2wajepid}` (today) |

Every number in this inventory is sourced to a JSON / CSV / PNG / handoff doc on disk. Where the chain of evidence has gaps, I've marked the item ambiguous.

---

## 1 — The Slot-07 finding (project_signature_shortcut_finding.md)

**What the claim is.** Slot-07 (R13_RLP5_07_E3_SEEDB) fails on the Teams lockbox in a way that is **not threshold-recoverable**, because the model latched onto a *processing pipeline signature*. Same person `dor_shkedi` (1138 lockbox vids, mean prob_fake = 0.457, 14.6% > 0.9) flips to "fake" while `real_dor` (109 lockbox vids of the visually-identical person, mean prob_fake = 0.038, 0% > 0.9) is fine. On the fake side, `teams_capture_cam_test_s33` (lockbox, 191 videos) misses 30.4% while the same face-swap family in s32 / s35 / s38 / s46 (dev) misses 0–1.2%.

**Checkpoint under test.** `gs://training-job-outputs/phase2r13_experiments/6jwwb526/value_composite_effort_20260423_step20500_auc0.9908_eer0.0304.pth` (R13_RLP5_07_E3_SEEDB).

**Source artifacts.**

| Artifact | What it shows |
|---|---|
| `analysis/lockbox_failure_contact_2026-04-24.png` | 4×12 contact sheet: row 1 = 12 missed s33 fakes (probs 0.008–0.073), row 2 = caught s32 reference (probs 0.994–0.996), row 3 = falsely-flagged dor_shkedi reals (0.991–0.995), row 4 = clean real_dor reals (0.005–0.241). |
| `analysis/lockbox_failure_contact_2026-04-24.index.json` | Per-thumbnail CSV with frame_path / video_id / frame_prob / method. 48 entries verified. |
| `analysis/lockbox_failure_contact_sheet_2026-04-24.py` | Reproducible script (uses CSVs at `/tmp/r13_analysis/frames/teams_{real,fake}_all_{lockbox,dev}_r13_rlp5_07_e3_seedb_frames_report.csv`). |
| `analysis/dor_closeup_2026-04-24.png` and `.py` | High-res 6×6 dor_shkedi (worst-flagged) vs real_dor (clean). Visual proof "same person." Identity_keys 732954601 vs 249494207 in the manifest. |
| `analysis/enhanced_fakes_contact_sheet_2026-04-24.py` and `.png` | Visomaster_enhanced_macro_dev (2.9% recall) + deeplive_enhanced_dev (9.0% recall) — enhanced fakes are stock-photo-clean and evade detection; webcam-style teams fakes get caught. |

**Methodological caveats.**
1. Sample size is 30 frames per pool (small, but the effect is strong — 0% vs 99% misclass at the *population* mean).
2. The "30.4% missed on s33" figure is at default τ=0.5; at τ>0.5 the gap widens because the failures cluster heavily below 0.1.
3. The closeup contact sheet is *visually* convincing that real_dor and dor_shkedi are the same human, but no formal face-recognition similarity score was computed.
4. The 90/90 lockbox goal is **not threshold-reachable** at any τ for this checkpoint (at τ catching 90% fakes, real FPR is 63.6%). This was not a calibration claim; it was computed from the per-pool score distributions.

**Verdict status.** Decisive. The slot-07 finding is the founding evidence for the camera/pipeline-signature shortcut hypothesis and reframed all subsequent packets.

---

## 2 — The s33 outlier (within the camera-test fake family)

**Claim.** Within the same face-swap "cam_test" family (sessions s32/s33/s35/s38/s46), only s33 misses on the lockbox.

**Numbers (per `lockbox_failure_contact_2026-04-24.index.json` and `R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md` §7.2):**

| Method (lockbox/dev) | n videos | RLP6_04 acc (default τ) | P8A acc | Comment |
|---|---:|---:|---:|---|
| `teams_capture_cam_test_s32` (dev) | 139 | 1.000 | 1.000 | reference (caught fine — slot-07 missed only 0–1.2%) |
| `teams_capture_cam_test_s33` (lockbox) | 191 (slot-07) / not in P8A scorecard | slot-07: 30.4% missed; visually p<0.075 on 12 worst | — | ambiguous (only slot-07 has these numbers; P8A scorecard suite only spans dev) |
| `teams_capture_cam_test_s35` (dev) | 244 | 0.975 | 0.943 | ~3pp regression |
| `teams_capture_cam_test_s38` (dev) | 51 | 1.000 | 1.000 | unchanged |
| `teams_capture_cam_test_s46` (dev) | 89 | 1.000 | 1.000 | unchanged |

The 30.4% number for s33 came from the slot-07 retro-score. RLP6_04's lockbox-side s33 number is not in any artifact I could find (the per-method scorecards in `analysis/p8a_fake_failure_analysis_2026-04-25/teams_fake_all_dev_*` are dev-only). **Ambiguous:** we don't have a clean RLP6_04 vs P8A vs P9 comparison on the lockbox s33 pool from the same per-frame report.

**Source files.**
- `analysis/lockbox_failure_contact_2026-04-24.index.json` (12 representative s33 missed frames, frame probs all <0.08)
- `analysis/lockbox_failure_contact_sheet_2026-04-24.py` (reproducible)
- Slot-07 lockbox CSVs at `/tmp/r13_analysis/frames/teams_fake_all_lockbox_r13_rlp5_07_e3_seedb_frames_report.csv` (referenced in scripts; not in repo)

**Methodological caveats.** The 30.4% headline is from one checkpoint (slot-07). We have not run the s33 lockbox pool against RLP6_04 or P8A directly — only against the dev partition of s33 (s33 is partly in `teams_fake_all_dev` for some scorecards, but that's a different sample). The cross-checkpoint s33-on-lockbox comparison **does not exist as a single artifact**.

---

## 3 — The visomaster_enhanced collapse (P8A regression vs RLP6_04)

**Claim.** Per-method recall on `visomaster_enhanced_macro` (and `deeplive_enhanced` and `teams_flat_xiang_xiang2_feng`) regressed substantially under P8A vs RLP6_04 on `teams_fake_all_dev` (2409 videos). 329 regressed videos, 0 compensating wins.

**Numbers (default τ=0.5 — see `analysis/p8a_fake_failure_analysis_2026-04-25/summary.json`).**

| Method | n videos | RLP6_04 acc | P8A acc | Δ (pp) | Regressed (P8A loses, RLP wins) |
|---|---:|---:|---:|---:|---:|
| `deeplive_enhanced` | 545 | 0.7963 | 0.5303 | **−26.6** | 145 (26.61%) |
| `teams_flat_xiang_xiang2_feng` | 135 | 0.7852 | 0.4519 | **−33.3** | 45 (33.33%) |
| `visomaster_enhanced_macro` | 550 | 0.5727 | 0.3564 | **−21.6** | 119 (21.64%) |
| `teams_capture_cam_test_s35` | 244 | 0.9754 | 0.9426 | −3.3 | 8 (3.28%) |
| `teams_capture_noyn_sharker_s23` | 204 | 0.9706 | 0.9118 | −5.9 | 12 (5.88%) |
| Other 12 `teams_capture_*` methods (varied) | varied | ≥0.98 | ≥0.98 | ~0 | 0–negligible |
| **Aggregate** | **2409** | **0.8389** | **0.7024** | **−13.66** | **329 (13.66%)** |

**Visomaster sub-pool split** (from same JSON, `visomaster_pool_split_raw_vs_teams`):

| Sub-pool | n | RLP6_04 acc | P8A acc | Regressed % | Notes |
|---|---:|---:|---:|---:|---|
| `visomaster_enhanced_raw` | 275 | 0.644 | 0.473 | 17.09% | non-Teams variant |
| `visomaster_enhanced_teams` | 275 | 0.502 | 0.240 | 26.18% | Teams-codec variant — **regresses 1.5× harder** |

The Teams-codec variant of visomaster_enhanced is exactly the slice closest to the deployment distribution. Its harder regression under P8A is consistent with the "codec-aug pressure made the real pass too codec-tolerant, hurting fakes that share that codec" reading.

**Score-distribution diagnostics on regressed videos.**

- `score_distribution_when_both_correct.deeplive_enhanced.p8a_score_when_correct.median = 0.896` vs `rlp_score_when_correct.median = 0.983` — P8A is **less confident even when correct**.
- `p8a_score_buckets_on_regressed.deeplive_enhanced`: 20 below 0.10, 40 in [0.10, 0.20), 31 in [0.20, 0.30), 21 in [0.30, 0.40), 13 in [0.40, 0.45), 20 in [0.45, 0.50). Only 33/145 (22.76%) sit in the τ-recoverable [0.40, 0.50) band; 41.4% sit confidently below 0.20.
- `pct_tau_recoverable_score_in_0.40_0.50` for visomaster_enhanced_macro is **12.61%**.

**Frame-level "full-flip" analysis** (same JSON, `frame_level.*`): for `deeplive_enhanced` (n=145) and `visomaster_enhanced_macro` (n=119), `videos_full_flip_pct_frames_flipped_ge_95` = 100% of regressed videos. Interpretation field in the JSON: *"P8A predicts real on essentially every frame -> deep separability loss, not noise-driven."*

**Source artifacts.**

| Path | Role |
|---|---|
| `analysis/p8a_fake_failure_analysis_2026-04-25/summary.json` | Structured findings (most detailed — bucket histograms, frame-flip rates, sub-pool splits, video-id token enrichment) |
| `analysis/p8a_fake_failure_analysis_2026-04-25/regressed_videos.csv` | 329 regressed video_ids with both checkpoints' scores |
| `analysis/p8a_fake_failure_analysis_2026-04-25/teams_fake_all_dev_p8a_step5000_summary_report.txt` | RLP6_04 acc 0.8389 vs P8A acc 0.7024 |
| `analysis/p8a_fake_failure_analysis_2026-04-25/teams_fake_all_dev_rlp6_04_step23500_summary_report.txt` | per-method recall RLP6_04 |
| `analysis/p8a_fake_failure_analysis_2026-04-25/analyze.py` | Reproducible analysis (bucketization, enrichment, etc.) |

**Methodological caveats.**
1. All numbers at **default τ=0.5**; the contract-calibrated τ may move them a few points but not enough to close the gap (only ~19% of regressed scores sit in the τ-recoverable band).
2. Only `teams_fake_all_dev` was scored. **No lockbox-side per-method P8A regression analysis exists at this granularity.**
3. `frames_per_video` is ~1.26 on average — frame-level flip-rate analysis is effectively video-level.
4. The "0 compensating wins" claim has been confirmed: `videos_p8a_won_rlp_lost: 0`.

**Verdict status.** Decisive **for the dev partition**. Lockbox-side per-method regression is *implied* but not directly measured at this granularity — promotion-contract scorecard reports per-method on lockbox at calibrated τ, see §14.

---

## 4 — Anchor-pool spread across checkpoints (the central comparison)

**The 6-pool anchor matrix.** Six fixed pools of 30 frames each, drawn from `gs://real-teams-dor-roee/session_20260424_combined_tags_121458_121007/`, hold the camera-signature shortcut signal:

| Pool | Identity / Camera | Expected behavior on a clean detector |
|---|---|---|
| `dor-real-laptop-correct-no-virtual-bg-whiteish` | Dor / laptop / clean | ≈0 (real, no shortcut activated) |
| `dor-real-laptop-correct-no-virtual-bg-yellowish` | Dor / laptop / yellow lighting | ≈0 |
| `roee-real-windows-laptop-correct` | Roee / Windows laptop | ≈0 |
| `dor-real-webcam-false-flag` | Dor / webcam / virtual bg | low; high = shortcut active |
| `dor-real-webcam-false-flag-no-virtual-bg` (**ANCHOR**) | Dor / webcam / no VB | low; **canonical shortcut signal** |
| `roee-mac-laptop-false-flag-virtual-bg` | Roee / Mac / VB on | low; cross-subject confirmation |

All numbers below are **mean prob_fake** per pool. Driver script: `analysis/teams_pool_rescore.py` (re-score any GS checkpoint on the same 180-frame matrix; writes `analysis/pool_rescore_<label>.summary.json`). Baselines from `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`.

### 4.1 — Master pool-mean table

| Run | Ckpt step | Date scored | Dor-laptop-white | Dor-laptop-yellow | Roee-Win | Dor-webcam-VB | **Dor-webcam-noVB (ANCHOR)** | Roee-Mac-VB | Decision tree verdict |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| RLP6_04 (baseline) | 23500 | 2026-04-24 | 0.013 | 0.014 | 0.006 | 0.964 | **0.932** | 0.751 | reference |
| RLP7_02 (codec-agg) | 5500 | 2026-04-24 | 0.016 | 0.016 | 0.010 | 0.885 | 0.856 (Δ −0.076) | 0.582 | marginal_close |
| RLP7_04 (spatial-only) | 5000 | 2026-04-24 | 0.012 | 0.013 | 0.009 | 0.934 | 0.857 (Δ −0.075) | 0.623 | marginal_close |
| RLP7_05 (spatial+codec) | 3500 | 2026-04-24 | 0.017 | 0.015 | 0.011 | 0.947 | 0.843 (Δ −0.089) | 0.597 | marginal_close |
| RLP7_06 (CCT-only) | 5000 | 2026-04-24 | 0.013 | 0.014 | 0.009 | 0.933 | 0.916 (Δ −0.015) | 0.618 | no_meaningful_change |
| RLP7_07 (triple SCT) | 5000 | 2026-04-24 | 0.013 | 0.013 | 0.008 | 0.945 | 0.893 (Δ −0.038) | 0.524 | no_meaningful_change |
| RLP7_08 (codec, fork @ 4500) | 9500 | 2026-04-24 | 0.019 | 0.017 | 0.009 | 0.929 | 0.835 (Δ −0.097) | 0.823 (+0.072 ⚠) | marginal_close |
| **P8A** (unfreeze proj+ln+mlp-svd) | 5000 | 2026-04-24 | **0.006** | **0.007** | **0.005** | **0.730** | **0.744 (Δ −0.188)** | **0.353** (Δ −0.399) | marginal_close — **best** |
| P8A | 2500 (ood_composite) | 2026-04-25 | 0.019 | 0.023 | 0.012 | 0.840 | 0.868 (Δ −0.064) | 0.720 | marginal_close — step 5000 leads |
| P8B (scratch on plain CLIP) | 5000 | 2026-04-25 | 0.071 ⚠ | 0.037 | 0.038 | 0.992 | 0.992 (Δ +0.061) | 0.500 | regression — anchor pinned ~1 |
| P8B | 11000 | 2026-04-25 | 0.141 ⚠ | 0.031 | 0.009 | 1.000 | 0.998 (Δ +0.066) | 0.374 | regression + real-pool regression |
| P10_GRL_baseline (λ=0.1) | 4500 | 2026-04-26 | 0.024 | 0.022 | 0.012 | 0.946 | 0.896 (Δ −0.036) | 0.662 | no_meaningful_change |
| P10_SYM_baseline | 4500 | 2026-04-26 | 0.043 | 0.035 | 0.016 | 0.936 | 0.911 (Δ −0.021) | 0.703 | no_meaningful_change |
| P10_SYM_LIGHT | 5500 | 2026-04-26 | 0.061 ⚠ | 0.045 | 0.014 | 0.957 | 0.945 (Δ +0.013 ⚠) | 0.807 ⚠ | no_meaningful_change |

(Δ = `this_run_mean − RLP6_04_baseline_mean`; negative = improvement on false-flag pools.)

### 4.2 — Per-pool "stuck-at-fake" rate (`frac_gt_0_9`) on the anchor

The anchor mean compresses signal — `frac_gt_0_9` shows how many of the 30 frames remain pinned ≥0.9.

| Run | Ckpt | `frac_gt_0_9` on anchor |
|---|---:|---:|
| RLP6_04 | 23500 | 0.80 |
| RLP7_02 | 5500 | 0.60 |
| RLP7_04 | 5000 | 0.57 |
| RLP7_05 | 3500 | 0.57 |
| RLP7_06 | 5000 | 0.80 |
| RLP7_07 | 5000 | 0.63 |
| RLP7_08 | 9500 | 0.50 |
| **P8A** | 5000 | **0.43** |
| P8A | 2500 | 0.63 |
| P8B | 5000 | 1.00 |
| P8B | 11000 | 1.00 |
| P10_GRL | 4500 | 0.63 |
| P10_SYM | 4500 | 0.77 |
| P10_SYM_LIGHT | 5500 | 0.87 |

P8A is the only checkpoint that pulls `frac_gt_0_9` below 0.50. Every P10 candidate scored today is *worse than RLP6_04* on this metric except the GRL run (which is comparable).

### 4.3 — FPR guardrail on the three "correct" pools

| Run | Max correct-pool mean | Verdict |
|---|---:|---|
| RLP6_04 | 0.014 | clean |
| RLP7_02 | 0.016 | clean |
| RLP7_04 | 0.013 | clean |
| RLP7_05 | 0.017 | clean |
| RLP7_06 | 0.014 | clean |
| RLP7_07 | 0.013 | clean |
| RLP7_08 | 0.019 | clean |
| **P8A** | **0.007** | **best — improved on correct pools too** |
| P8B (5000) | 0.071 | regression |
| P8B (11000) | 0.141 | regression |
| P10_GRL | 0.024 | mild — passes |
| P10_SYM | 0.043 | mild |
| P10_SYM_LIGHT | 0.061 | weakest of P10 trio |

### 4.4 — Source files (one per checkpoint)

`analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json` (baseline)
`analysis/pool_rescore_rlp7_02.summary.json`
`analysis/pool_rescore_rlp7_04.summary.json`
`analysis/pool_rescore_rlp7_05.summary.json`
`analysis/pool_rescore_rlp7_06.summary.json`
`analysis/pool_rescore_rlp7_07.summary.json`
`analysis/pool_rescore_rlp7_08.summary.json`
`analysis/pool_rescore_rlp8_a.summary.json` (P8A step 5000)
`analysis/pool_rescore_rlp8_a_step2500.summary.json` (P8A step 2500)
`analysis/pool_rescore_rlp8_b.summary.json` (P8B step 11000)
`analysis/pool_rescore_rlp8_b_step5000.summary.json` (P8B step 5000)
`analysis/pool_rescore_p10_grl_step4500.summary.json` (today)
`analysis/pool_rescore_p10_sym_step4500.summary.json` (today)
`analysis/pool_rescore_p10_sym_light_step5500.summary.json` (today)

Per-frame CSVs alongside each (`pool_rescore_*.per_frame.csv`).

### 4.5 — Methodological caveats on the anchor matrix

1. **Sample size: 30 frames per pool.** This is small. A movement of one frame is 3.3pp. Cross-checkpoint differences <0.03 are within noise.
2. The anchor frames come from a **single 6-second capture session** (`session_20260424_combined_tags_121458_121007`) — not multiple Dor-on-webcam captures across days. Time-of-day and within-clip drift shouldn't bias the comparison (every checkpoint sees the same 180 frames) but limits external validity.
3. **Preprocessing fix on 2026-04-24** (commit `855871e`): `cv2.INTER_AREA → cv2.INTER_LINEAR` in `arena/model_arena.py:472` (and `batch_inference_gcs.py:407`) to match training. This affected anchor numbers for RLP6_04 from 0.94 (pre-fix) to 0.93 (post-fix) on the no-VBG pool — see `rlp6_04_postfix_rescore_2026-04-24.summary.json` field `pre_fix_mean` vs `post_fix_mean`. **Every `pool_rescore_*` JSON dated ≥2026-04-24 12:22 UTC is post-fix.** The pre-fix numbers in the Packet-7 handoff (e.g. "Dor webcam noVB 0.94") are slightly inflated.
4. Anchor delta is **not the same metric as the deployment scorecard** (lockbox FPR + per-method recall at calibrated τ). A run can win the anchor and lose the contract — exactly P8A's situation.

---

## 5 — The P8A breakthrough claim

**Claim (memory `project_p8a_breakthrough.md`).** P8A "broke the camera-signature ceiling, ~2× lower anchor FPR" via unfreezing `visual.proj` + `visual.ln_post` + adding SVD on MLP.

**The supporting numbers (re-derived from the anchor matrix in §4):**

| Pool | RLP6_04 | best P7 (RLP7_05) | **P8A** | P7 Δ | **P8A Δ** | P8A advantage over best P7 |
|---|---:|---:|---:|---:|---:|---:|
| Dor-webcam-noVB (ANCHOR) | 0.932 | 0.843 | 0.744 | −0.089 | **−0.188** | 2.1× |
| Dor-webcam-VB | 0.964 | 0.947 | 0.730 | −0.017 | −0.234 | 14× |
| Roee-Mac-VB | 0.751 | 0.597 | 0.353 | −0.154 | −0.399 | 2.6× |
| Max real-correct mean | 0.014 | 0.017 | **0.007** | improved | improved | better |

So the "~2×" claim is correct on the anchor pool's mean, **but** the anchor `frac_gt_0_9` only drops from 0.80 (RLP6_04) → 0.43 (P8A). 13/30 anchor frames remain pinned ≥0.9 — the shortcut is *weakened, not eliminated*.

**Cross-check: did P8A "break the ceiling" or just shift the failure mode?**

The data on this is in §3 above (the visomaster_enhanced collapse). P8A traded:
- A real-pool win (anchor mean −0.188; FPR on `teams_real_all_dev` 15.92% → 12.11%; on `teams_real_poor_quality_dev` 13.33% → 8.13%) — a real, lockbox-confirmable improvement.
- For a fake-recall loss (per-method aggregate −13.66pp; visomaster_enhanced_teams 50.2% → 24.0%; deeplive_enhanced 79.6% → 53.0%) — concentrated in the codec-similar fake families.

So the answer is: **P8A broke the camera-signature ceiling and simultaneously shifted the model into a regime where codec-similar fakes look like reals.** The reframe is in `R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md` §7.4: *"P8A's regression is separability loss, not threshold drift."* And in the failure-mode JSON: per-method `frac_full_flip_ge_95` = 100% on the regressed methods.

So the memory's claim is **factually correct but incomplete**. It captures the anchor-side win and explicitly notes "13/30 anchor frames still flip" — but the contemporaneous P8A breakthrough memo predates the per-method dev scorecard that revealed the fake-recall regression. The follow-up packet-9 mid-flight handoff (`docs/relaunch_handoffs/PACKET_9_MID_FLIGHT_HANDOFF_2026-04-26.md`) is candid that "Roee is less confident" about the P8A premise.

**Source artifacts for the breakthrough claim:**

| Path | Role |
|---|---|
| `analysis/pool_rescore_rlp8_a.summary.json` | Anchor numbers |
| `analysis/overnight_packet7_packet8_summary_2026-04-25.md` | Narrative + cross-run table |
| `~/.claude/projects/.../memory/project_p8a_breakthrough.md` | Memory note |
| `R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md` §6, §7 | Compact experiment chain + regression reframe |
| `analysis/p8a_fake_failure_analysis_2026-04-25/summary.json` | The follow-up regression analysis |

**Methodological caveats.**
1. The "~2×" claim is over the *best of 6 P7 variants* (RLP7_05 at −0.089). The "average P7" Δ is around −0.06, so vs that average it's ~3×.
2. The anchor pool has 30 frames in one session; the lockbox real-pool numbers (3253 videos at 12.11% FPR for P8A vs 15.92% for RLP6_04) are the real validation. Those numbers are from `R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md` §7.1 and originate from the in-progress P8A scorecard (job `2162379556655202304`). I have not opened that scorecard's underlying CSV in this audit.
3. P8A `frac_gt_0_9` of 0.43 means 13 of 30 anchor frames still flip — **the ceiling was lowered, not removed**.

**Verdict status.** Confirmed for the anchor metric (decisive). The "broke the ceiling" framing is technically correct *for that metric* but the deployment-grade contract scorecard now reveals the regression that the memo did not capture.

---

## 6 — The P7 ceiling investigation (project_shortcut_is_upstream.md)

**Claim.** "FT-only from any RLP6_04 step hits a ~0.84–0.89 anchor ceiling." Therefore the shortcut "lives upstream of RLP6_04."

**Methodology.** Compare anchor-Δ across the six P7 runs at different aug axes and one earlier fork (RLP7_08 from RLP6_04 step 4500). All FT-only, frozen backbone, head + attention SVD trainable, 10k-step schedule.

**Numbers (from §4.1 and §4.2):**

| Run | Aug | Fork | Anchor mean | Anchor Δ | Anchor `frac_gt_0_9` | Roee-Mac Δ |
|---|---|---|---:|---:|---:|---:|
| RLP7_02 | codec-aggressive | step 23500 | 0.856 | −0.076 | 0.60 | −0.169 |
| RLP7_04 | spatial-only | step 23500 | 0.857 | −0.075 | 0.57 | −0.129 |
| RLP7_05 | spatial+codec | step 23500 | 0.843 | −0.089 | 0.57 | −0.154 |
| RLP7_06 | CCT-only | step 23500 | 0.916 | −0.015 | 0.80 | −0.134 |
| RLP7_07 | triple (S+C+T) | step 23500 | 0.893 | −0.038 | 0.63 | −0.228 |
| RLP7_08 | codec-aggressive | **step 4500** | 0.835 | −0.097 | 0.50 | +0.072 |

**Observation.**
- Anchor-mean clusters in [0.835, 0.916] regardless of aug axis or fork point. *No FT-only variant breaks below 0.835.*
- Earlier fork (RLP7_08, step 4500) gave the *best* anchor (−0.097) but regressed Roee-Mac. So the RLP6_04 shortcut isn't "consolidating late" — it was already locked in by step 4500.
- Codec-heavy aug (RLP7_02/05/08) clusters around −0.08 to −0.10. CCT-only and triple-axis are worse on anchor. Three axes are not additive.

**Why this means "upstream":** if the shortcut is reachable by FT, you'd expect *some* aug recipe to break below the cluster. None did. Conclusion: the shortcut is in the frozen-backbone CLIP features themselves, not in what FT-trained head/SVD-residuals do with them.

**Source artifacts.**
- `analysis/pool_rescore_rlp7_0{2,4,5,6,7,8}.summary.json`
- `analysis/teams_pool_rescore.py` (re-score driver)
- `analysis/overnight_packet7_packet8_summary_2026-04-25.md` (narrative)
- `~/.claude/projects/.../memory/project_shortcut_is_upstream.md`

**Methodological caveats.**
1. "Upstream of RLP6_04" actually means "in the frozen CLIP backbone" — that's where P8B (scratch on plain CLIP) implicates the data mix even more (P8B is *worse* than RLP6_04 on anchor). So the *real* statement is: "the shortcut lives in the data mix AND is fixed by light-FT only when FT has reach into the CLIP backbone." The memory's "upstream of RLP6_04" framing is colloquial shorthand — the precise claim is the one from `overnight_packet7_packet8_summary_2026-04-25.md`: *"the shortcut lives in the data mix, but FT-from-RLP6_04 with backbone-reach is what suppresses it."*
2. All P7 results are from one checkpoint each (the latest `value_composite` per run). No multi-step-trajectory analysis exists.
3. CCT-only (RLP7_06) showing the *worst* anchor delta is suggestive that color-temperature isn't the dominant axis — but CCT was best for Roee-Mac. The six runs are not orthogonal in their aug axis coverage.

**Verdict status.** The anchor-ceiling observation is decisive within the FT-only family. The "upstream of RLP6_04" causal claim is partially supported (P8B refutes "in R12g/R13 weight chain"; P8A confirms backbone reach is what was missing) but the precise locus of the shortcut is "frozen CLIP backbone × data mix interaction," not "RLP6_04 weights specifically."

---

## 7 — The P10 anchor rescore (today, 2026-04-26)

**What's being tested.** Three Packet-10 candidates that finished early (step 4500–5500) on a new "anti-shortcut" axis:

- **P10_SYM** (`osji02ho`, step 4500): symmetric routing baseline (data axis variant)
- **P10_GRL** (`ntbx1hh1`, step 4500): GRL probe at λ=0.1 (gradient reversal on quality/codec head)
- **P10_SYM_LIGHT** (`2wajepid`, step 5500): a lighter SYM variant

**Anchor numbers (from §4.1, dated today):**

| Pool | RLP6_04 | P10_SYM | P10_GRL | P10_SYM_LIGHT |
|---|---:|---:|---:|---:|
| Dor-laptop-white | 0.013 | 0.043 | 0.024 | 0.061 ⚠ |
| Dor-laptop-yellow | 0.014 | 0.035 | 0.022 | 0.045 |
| Roee-Win | 0.006 | 0.016 | 0.012 | 0.014 |
| Dor-webcam-VB | 0.964 | 0.936 | 0.946 | 0.957 |
| **Dor-webcam-noVB (ANCHOR)** | 0.932 | 0.911 | **0.896** | 0.945 |
| Roee-Mac-VB | 0.751 | 0.703 | 0.662 | 0.807 ⚠ |

**Anchor Δ vs RLP6_04:** SYM −0.021, GRL −0.036, SYM_LIGHT **+0.013** (regression).
**Anchor `frac_gt_0_9`:** SYM 0.77, GRL **0.63**, SYM_LIGHT 0.87.
**Real-correct max mean:** SYM 0.043, GRL 0.024, SYM_LIGHT 0.061.

**Verdict per checkpoint** (from each summary JSON's `decision_tree.verdict`):
- P10_SYM: `no_meaningful_change`
- P10_GRL: `no_meaningful_change`
- P10_SYM_LIGHT: `no_meaningful_change`

None of the three early-stop P10 candidates reaches P8A's anchor depth (P8A: anchor 0.744, Δ −0.188, `frac_gt_0_9` 0.43, max correct 0.007). All three are in the P7-cluster regime or worse on every metric.

**Caveats.** These three checkpoints are early-stopped (step 4500–5500). The full P10 slate (with longer-running GRL_strong, SYM_GRL, SYM_on_P8A variants) is still in progress at this writing — see `arena/checkpoint_maps/teams_target_domain.p10_partial_2026-04-26.yaml` headnote. The "early-stop" decisions came from the trainer's `value_composite`, which has already been observed to crown the wrong checkpoint (memory `project_promotion_contract.md`). The contract-anchored scorecard has not been run on these P10 checkpoints yet.

**Source artifacts.**
- `analysis/pool_rescore_p10_grl_step4500.summary.json` + `.per_frame.csv`
- `analysis/pool_rescore_p10_sym_step4500.summary.json` + `.per_frame.csv`
- `analysis/pool_rescore_p10_sym_light_step5500.summary.json` + `.per_frame.csv`
- `arena/checkpoint_maps/teams_target_domain.p10_partial_2026-04-26.yaml` (intent + crowning rules)

**Verdict status.** Inconclusive. The early-stop P10 trio does not demonstrate progress on the camera-signature shortcut beyond the P7 cluster. Stronger P10 variants are still training.

---

## 8 — Contact sheets and visual analyses

These are visual artifacts that humans look at to confirm or reject hypotheses. Each has a paired index/CSV/summary JSON and a Python driver.

| Artifact | Visual content | Purpose | Methodological caveat |
|---|---|---|---|
| `analysis/lockbox_failure_contact_2026-04-24.png` (4×12 grid, 2.2 MB) + `.index.json` | Row 1: 12 missed s33 fakes (probs 0.008–0.073). Row 2: caught s32 reference. Row 3: falsely-flagged dor_shkedi reals (0.991–0.995). Row 4: clean real_dor reference. | Visual proof that pipeline-shifted same-face data flips the model. Driver: `lockbox_failure_contact_sheet_2026-04-24.py` (slot-07 frames). | Chosen frames are extreme (lowest/highest prob); not representative of pool average — which is why the index JSON is needed for accurate stats. |
| `analysis/dor_closeup_2026-04-24.png` (1.1 MB) + `dor_closeup_2026-04-24.py` | High-res 6×6 dor_shkedi (worst-flagged, prob ≥ 0.99) vs real_dor (random, prob ≤ 0.04). Visually identical person. | Rules out "different people" explanation for the dor_shkedi vs real_dor split. | No formal face-recognition embedding similarity — just human-eye verification. |
| `analysis/enhanced_fakes_contact_2026-04-24.png` (1.6 MB) + `enhanced_fakes_contact_sheet_2026-04-24.py` | Row 1: visomaster_enhanced_macro_dev failures (slot-07: 2.9% recall). Row 2: deeplive_enhanced_dev failures (slot-07: 9.0% recall). Row 3: teams_fake_all_dev (caught) reference. | Visual evidence: enhanced fakes are stock-photo-clean, evade detection; webcam-style teams fakes get caught — direct visual signature shortcut. | Slot-07 specific. P8A regression analysis (§3) extends to RLP6_04 vs P8A. |
| `analysis/session_20260424_4people_compare.png` (3.3 MB) + `session_20260424_4people_compare.py` | 4-row contact sheet, all REAL: Xiang_Xiang2_Feng (model: real), Xinhe_XH68_Wang (model: real), tester_tester (model: real), dor_shkedi (model: **FAKE p>0.9**). Same 2026-04-24 capture session, all 4 real. | Decisive visual: same pipeline, same session, model says 3 real / 1 fake. Rules out "session-wide pipeline" as the only signal — the model picks Dor specifically. | All 4 are in the same Teams capture session; the per-identity flip on Dor is the signal. |
| `analysis/check_frame_4people_2026-04-24.json` + `.py` | `check-frame` CLI scoring of those 4 identities (32 frames each). Per-identity mean prob_fake, std, frac>0.9, frac<0.1. | Quantitative companion to `session_20260424_4people_compare.png`. **Important:** Xiang_Xiang2_Feng mean prob 0.0089 (clean real); Xinhe_XH68_Wang mean 0.030; tester_tester unread; dor_shkedi the failing pool. | Sample is 30–32 frames per identity in one session; check-frame uses production `batch_inference_gcs.py` path, so the post-2026-04-24 INTER_LINEAR fix is in effect. |
| `analysis/dor_baseline_repro_2026-04-24.png` series (4 PNGs, 30k each) | Strip-plots: today_failing vs lockbox_clean, dor_shkedi_lb vs lockbox_clean, today vs dor_shkedi_lb. Per-metric overlay (mean_lum, dynamic_range, dct_hf_ratio, etc.). | Visual diagnostic that today's failing Dor pool clusters with shkedi_lb, not with real_dor. | 30 frames per pool. |

**Decision-tree verdict from the dor_baseline_repro JSON** (`dor_baseline_repro_2026-04-24.summary.json`):
- `today_vs_lockbox_clean`: 5 lighting hits, 3 codec hits → "V3_combined" recommendation (combined aug).
- The metrics that consistently separate failing pools from the clean reference are `dynamic_range` (failing > clean), `wb_rb_ratio` (failing < clean — i.e., less red-blue ratio), `dct_hf_ratio` (failing < clean — less HF DCT energy), and `mean_cb`/`mean_cr` (chroma offsets).

---

## 9 — Feature-space analysis (`analysis/feature_space_2026-04-23/`)

**What was measured.** Per-source 512-dim backbone-feature embeddings extracted via `extract_features.py` (Vertex job), then pairwise distances over 20 sources: 3 distance metrics (Fréchet, MMD-RBF, centroid Euclidean). 443 frames analyzed across user-created training sources (deeplive_*, proper_visomaster_*, dl_bucket_*, tv2_*), candidate new bucket (visomaster_enhanced_v2), and external OOD-gate targets (`external_vcd_real`, `external_youtube_avspeech_real`, `wma_failure_fake`).

**Key findings (from `analysis/feature_space_2026-04-23/REPORT.md`):**

1. **Per-source detector behavior** (sample, see CSV):
   - `proper_visomaster_teams_fake`: prob_mean 0.917, prob>0.5 rate 93.7% — model is mostly catching this.
   - `proper_visomaster_enhanced_teams_fake`: prob_mean 0.949, rate 96.0% — better.
   - `tv2_visomaster_fake`: prob_mean **0.494**, rate **42.1%** — borderline (low n=38).
   - `wma_failure_fake`: prob_mean 0.994, rate 100% — caught (but wma is structurally broken for the gate, see §11).
   - All real sources: prob_mean 0.02–0.22; `external_youtube_avspeech_real` is the noisiest at 0.221 mean / 23.8% false-flag rate.

2. **Top distributional gaps** (Fréchet over 512-dim features): all the largest pairs are real ↔ fake cross-class, which is what we'd want. No "fake source close to a real source" is in the top-15 — i.e., the trained detector's feature space *does* separate fake from real on average.

3. **Centroid distances** (`distance_matrix_centroid.csv`):
   - Closest real-to-real: `proper_real_teams ↔ proper_real_clean` 1.71; `proper_real_clean ↔ deeplive_enh_real` 1.85.
   - Closest fake-to-real (cross-class): `tv2_visomaster_fake ↔ external_youtube_avspeech_real` 10.39 — well-separated.
   - The cross-class minimum is far above the intra-class spread, so the detector embedding is *not* obviously confused at the centroid level.

**Verdict.** The 2026-04-23 feature-space analysis says the detector embedding does separate classes globally. It does *not* address the per-camera/per-pipeline shortcut: the failing Dor-webcam frames are inside the `proper_real_teams` distribution at the centroid level but score "fake" at the head — implying the head's decision uses dimensions the centroid distance metric averages out. **This is a limitation of the analysis, not a contradiction of the shortcut.**

**Source artifacts.**
- `analysis/feature_space_2026-04-23/REPORT.md`
- `analysis/feature_space_2026-04-23/per_source_summary.csv`
- `analysis/feature_space_2026-04-23/distance_matrix_{frechet,mmd_rbf,centroid}.csv`
- `analysis/feature_space_2026-04-23/extract_features.py` (Vertex)
- `analysis/feature_space_2026-04-23/compute_distances.py`
- `analysis/feature_space_2026-04-23/launch_vertex.sh`

**Methodological caveats.**
1. Backbone features (post-CLIP, pre-head) — the head's last-layer projection may amplify dimensions the centroid analysis flattens.
2. 15–60 frames per source — small for high-dim Fréchet.
3. **No per-camera grouping in this analysis.** It's per-bucket / per-method — so "Dor-webcam vs Dor-laptop" isn't a row in the distance matrix.

**Conclusion.** Feature-space distance is **not where the shortcut signal lives at this granularity**. The analysis was useful for ruling out gross domain mismatch but does not localize the camera-signature axis.

---

## 10 — Fingerprint diff work (the key shortcut diagnostic)

**What "fingerprint" means here.** A 10-feature per-frame vector of pixel-level statistics:

| Group | Metrics | Probes |
|---|---|---|
| Lighting | `mean_lum`, `highlight_clip_pct`, `shadow_clip_pct`, `dynamic_range`, `wb_rb_ratio`, `specular_hotspots` | overall brightness, clipping, white-balance |
| Codec / chroma | `dct_hf_ratio`, `mean_cb`, `mean_cr`, `bits_per_pixel` | high-freq content (compression), chroma offset, file size per pixel |

The fingerprint is **not** a learned embedding. It's a deterministic per-frame stats vector designed to capture camera/codec/pipeline differences invisible to a centroid-distance check.

**Driver scripts.**
- `analysis/fingerprint_diff.py` — generic two-pool fingerprint diff with separation-σ thresholds.
- `analysis/dor_pool_fingerprint_diff_2026-04-24.py` — specialized for the 3-pool dor case (today_failing, dor_shkedi_lb, real_dor_lb).

**Caches:**
- `analysis/_fingerprint_cache_2026-04-24/` (84 files for the 3-pool initial run)
- `analysis/_fingerprint_cache_combined_2026-04-24/` (182 files for the 6-pool combined Dor+Roee run)

### 10.1 — `dor_pool_fingerprints_2026-04-24.summary.json` (3-pool initial)

**Pools (30 frames each):**
- `today_failing` — Dor session 2026-04-24, dor_shkedi capture (model false-flags)
- `dor_shkedi_lb` — lockbox identity_key=dor_shkedi (also model false-flags)
- `real_dor_lb` — lockbox identity_key=real_dor (clean — model correctly says real)

**Verdict from JSON:**
```
verdict: { lighting_hits: 5, codec_hits: 3, recommended_variant: V3_combined }
```

That is — both lighting and codec axes separate the failing pools from the clean reference. Specifically, `mean_lum` (failing > clean), `highlight_clip_pct`, `shadow_clip_pct`, `dynamic_range`, `wb_rb_ratio`, `specular_hotspots` for lighting; `dct_hf_ratio`, `mean_cb`, `mean_cr` for codec. (`bits_per_pixel` did not separate.)

### 10.2 — `dor_roee_combined_2026-04-24.summary.json` (6-pool extended)

Same 6 anchor pools as in §4. Computes per-pool metric statistics + Pearson/Spearman correlation between each metric and the model's prob_fake within each pool + cross-pool correlation aggregate.

**Cross-pool score-correlation ranking** (which fingerprint metric most predicts model score, mean |Spearman|):

| Metric | mean abs Spearman | Pools where strong (n_pools_strong / 6) | Direction |
|---|---:|---:|---|
| `mean_cr` | 0.385 | 3 | higher → higher score (more red chroma → more fake) |
| `mean_cb` | 0.362 | 4 | higher → lower score |
| `dynamic_range` | 0.348 | 5 | higher → lower score |
| `wb_rb_ratio` | 0.330 | 3 | higher → higher score |
| `dct_hf_ratio` | 0.302 | 3 | mixed direction |
| `bits_per_pixel` | 0.299 | 2 | mixed |
| `mean_lum` | 0.240 | 2 | mixed |
| `highlight_clip_pct` | 0.205 | 2 | mixed |
| `shadow_clip_pct` | 0.192 | 3 | higher → lower score |
| `specular_hotspots` | 0.116 | 1 | mixed |

**Cross-pair separation** (which metrics most reliably separate failing from clean across the 5 contrast pairs):

- Pairs: A. dor-webcam-noVB-failing vs dor-laptop-white-clean; B. dor-webcam-VB-failing vs dor-laptop-white; C. roee-mac-failing vs roee-windows-clean; D. dor-laptop-yellow-clean vs dor-laptop-white-clean (negative control); E. dor-webcam-noVB vs dor-webcam-VB (intra-camera control).

- The metrics that separate consistently **with same direction** across the 3 real failing-vs-clean pairs (A, B, C) are: `dynamic_range` (failing > clean — i.e., webcam captures have higher contrast range), `wb_rb_ratio`, `dct_hf_ratio`, `mean_cb`, `bits_per_pixel`.

**The clean negative controls work:** D (Dor-laptop-yellow vs Dor-laptop-white — should look the same to the model, both are "clean") shows minimal `mean_lum` separation. E (with vs without virtual background) shows large `dynamic_range` separation but small score-impact within Dor-webcam (both pools score >0.87) — confirms VB itself is not the dominant axis.

### 10.3 — `dor_baseline_repro_2026-04-24.summary.json` (today_failing reproduces shkedi_lb)

Verifies that the live `dor_session_20260424` capture has the same fingerprint signature as the lockbox `dor_shkedi_lb` videos that the slot-07 model false-flags. Per-pair verdict:

- `today_vs_lockbox_clean`: 5 lighting hits, 3 codec hits → V3 combined.
- `shkedi_lb_vs_lockbox_clean`: 2 lighting + 2 codec hits.
- `today_vs_shkedi_lb`: only 2 hits (similar fingerprints) — confirming they are pipeline-equivalent.

**This is the bridge** — today_failing's fingerprint matches dor_shkedi_lb's fingerprint, so reproducing the false-flag in a controlled capture is well-grounded.

**Methodological caveats.**
1. 30 frames per pool. Per-pool statistics are stable but inter-frame variance within a pool is sometimes high (e.g., `shadow_clip_pct` σ = 4.5 on dor_shkedi_lb).
2. Spearman cross-pool correlations of 0.30–0.40 are *suggestive* but explain <16% of variance; many other axes contribute to model score.
3. **Direction inconsistency on `dct_hf_ratio` and `bits_per_pixel`** — these flip direction in different pools, indicating they are confounded with other variables.
4. The metrics are pixel-level. They don't capture *post-CLIP-feature* shortcut axes. So a fingerprint that doesn't separate may still correlate with a CLIP-level shortcut.

**Verdict status.** Clean. The 3-pool initial confirms lighting + codec both separate failing from clean. The 6-pool extended confirms the same metrics generalize cross-subject (Roee-Mac matches the Dor pattern). The negative controls (D, E) work as expected.

**Source artifacts.**
- `analysis/dor_pool_fingerprint_diff_2026-04-24.py`
- `analysis/dor_pool_fingerprints_2026-04-24.summary.json` + `.csv` + `.png`
- `analysis/dor_roee_combined_2026-04-24.summary.json` + `.csv` + `.A_*.png` ... `E_*.png` (5 contact sheets) + `cross_pair_summary.png` + `score_correlation.png`
- `analysis/dor_baseline_repro_2026-04-24.summary.json` + `.csv` + 3 PNGs
- `analysis/fingerprint_diff.py` (generic driver)
- `analysis/configs/dor_baseline_repro.yaml`, `dor_roee_combined_2026-04-24.yaml`

---

## 11 — Bucket comparison (`analysis/bucket_comparison_2026-04-23/`)

**Subject.** A different question than the camera-signature shortcut — *why a model trained on DeepLive + proper_visomaster doesn't generalize to the OOD `wma_failure` / `teams_ood` pools that the trainer's `value_composite` gate grades against.* Generates a 443-image cross-source analysis.

**Method.** Sampled ~15–60 frames per source, computed image-level stats (resolution, file size, luminance, sharpness via Laplacian variance, edge density via Canny, noise via high-pass residual, blockiness proxy). 20 sources covering training data, unused buckets, and external OOD targets.

**Headline findings (`REPORT.md`):**

1. **Resolution clusters into 3 regimes:** Standard 224² (most training data), Teams-small 177–214² (tv2_*, visomaster_enhanced_v2), Outlier 342×436 non-square (`wma_failure_fake`).
2. **`wma_failure_fake` is structurally broken for the gate:**
   - Resolution 342×436 (everything else: 175–262 square)
   - Laplacian variance **18** (everything else 51–241; 5–10× less sharp)
   - Noise std **2.5** (everything else 4–7.4; half the natural floor)
   - Edge density **0.013** (everything else 0.025–0.063)
   The detector achieves 100% on this pool because the data has been so heavily smoothed/upscaled it doesn't resemble anything else, but the gate uses this as a fake-side OOD driver. **The gate signal is unreliable on this pool.**
3. **`proper_visomaster_clean_fake` and `proper_visomaster_teams_fake` are oddly smooth** (Laplacian 51–56) compared to their paired reals (113–193) and enhanced variants (148–215) — looks like the generators' output got post-blurred by some downstream processing. The model may be learning "smoothing = fake" from these lanes.
4. **`proper_visomaster_enhanced_clean` is the sharpest fake source** (Laplacian 215) — but is **NOT in training**. Sitting unused.
5. **Brightness is bucket-systematic.** Teams-bucket sources skew bright (luma 108–161); proper_visomaster skews mid (103–116); DeepLive bucket is mixed (83–157).

**Source artifacts.**
- `analysis/bucket_comparison_2026-04-23/REPORT.md`
- `analysis/bucket_comparison_2026-04-23/per_image_stats.csv` (443 rows)
- `analysis/bucket_comparison_2026-04-23/per_source_summary.csv`
- `analysis/bucket_comparison_2026-04-23/thumbnails_grid.png` (6.5 MB)
- `analysis/bucket_comparison_2026-04-23/sample_and_analyze.py`, `make_thumbnail_grid.py`
- `analysis/bucket_comparison_2026-04-23/cache/<source>/*.png`

**Caveats.** Image-level pixel stats only — no embedding analysis. 15–60 frames per source. Doesn't directly probe the camera-signature axis (which is a per-source, per-pipeline issue *within* a source).

**Verdict status.** Decisive on the OOD-gate diagnosis. Not the camera-signature shortcut, but adjacent — it shows there's a class of "quality / sharpness / smoothness" axes in the data that the model could exploit. Especially relevant for the post-smoothed proper_visomaster non-enhanced lanes (which may be teaching "smoothing = fake").

---

## 12 — Pool rescore series (the RLP6_04 → RLP7 → P8 → P10 trajectory)

The complete set of pool-rescore JSONs from `analysis/`. All numbers are pulled directly from the JSON files; cross-checkpoint table is the master in §4.1.

| Filename | Scoped checkpoint | Generated | Verdict |
|---|---|---|---|
| `rlp6_04_postfix_rescore_2026-04-24.summary.json` | RLP6_04 step 23500 (after INTER_LINEAR fix) | 2026-04-24 12:22 | baseline reference |
| `pool_rescore_rlp7_02.summary.json` | RLP7_02 step 5500 | 2026-04-24 17:02 | marginal_close |
| `pool_rescore_rlp7_04.summary.json` | RLP7_04 step 5000 | 2026-04-24 21:02 | marginal_close |
| `pool_rescore_rlp7_05.summary.json` | RLP7_05 step 3500 | 2026-04-24 21:03 | marginal_close (best balanced P7) |
| `pool_rescore_rlp7_06.summary.json` | RLP7_06 step 5000 | 2026-04-24 21:05 | no_meaningful_change |
| `pool_rescore_rlp7_07.summary.json` | RLP7_07 step 5000 | 2026-04-24 17:02 | no_meaningful_change |
| `pool_rescore_rlp7_08.summary.json` | RLP7_08 step 9500 (fork @ 4500) | 2026-04-24 20:03 | marginal_close (best anchor in P7) |
| `pool_rescore_rlp8_a.summary.json` | P8A step 5000 | 2026-04-24 23:54 | marginal_close — **best overall** |
| `pool_rescore_rlp8_a_step2500.summary.json` | P8A step 2500 (ood_composite) | 2026-04-25 08:38 | marginal_close — step 5000 leads |
| `pool_rescore_rlp8_b.summary.json` | P8B step 11000 | 2026-04-25 01:01 | regression (anchor pinned + real-pool regression) |
| `pool_rescore_rlp8_b_step5000.summary.json` | P8B step 5000 | 2026-04-25 00:07 | regression |
| `pool_rescore_p10_grl_step4500.summary.json` | P10 GRL @ step 4500 | 2026-04-26 15:01 | no_meaningful_change |
| `pool_rescore_p10_sym_step4500.summary.json` | P10 SYM baseline @ step 4500 | 2026-04-26 15:01 | no_meaningful_change |
| `pool_rescore_p10_sym_light_step5500.summary.json` | P10 SYM_LIGHT @ step 5500 | 2026-04-26 15:01 | no_meaningful_change |

**Key cross-cuts learned across this series:**

1. **The anchor ceiling is reproducible.** Six P7 variants, three different aug axes, two fork points → all between 0.835 and 0.916 mean on the anchor.
2. **P8A is decisively below the ceiling** (0.744 mean, 0.43 `frac_gt_0_9`).
3. **P8B is decisively *above* the ceiling** (0.992–0.998 mean, 1.00 `frac_gt_0_9`) — confirms scratch-on-CLIP doesn't help.
4. **P10 trio (early-stop) clusters with P7** — does not reach P8A's anchor depth.
5. **Roee-Mac is a separate axis from Dor-webcam.** RLP7_07 (triple S+C+T) is best for Roee-Mac (Δ −0.228) but middling for Dor-webcam (Δ −0.038). RLP7_08 is best for Dor-webcam (Δ −0.097) but regresses Roee-Mac (Δ +0.072). Only P8A wins both simultaneously.

**Adjacent script:** `analysis/compare_teams_pools.py` — the original 2026-04-22 diagnostic that was used in `R13_RELAUNCH_VISOMASTER_TEAMS_POOL_DIAGNOSTIC_FINDINGS_2026-04-23.md` to investigate the proper_visomaster_teams pool. **Different purpose** (per-method diagnostic on the visomaster pool), not the 6-pool anchor matrix. Documented in `docs/superpowers/specs/2026-04-22-compare-teams-pools-design.md`.

**Adjacent script:** `analysis/rlp6_04_postfix_rescore_2026-04-24.py` — produced the post-fix baseline used by every subsequent `pool_rescore_*` JSON. Notable: the `pre_fix_mean` field in this JSON shows that `cv2.INTER_AREA` inference inflated the anchor mean by ~0.02 vs `cv2.INTER_LINEAR` (training-matched). After the fix, RLP6_04 baseline anchor moved 0.94 → 0.93. **All numbers in the RLP7+ era are post-fix.**

---

## 13 — Analysis sessions (`session_20260424_4*`)

This refers to two artifacts dated 2026-04-24:

1. **`session_20260424_4people_compare.png` (3.4 MB) + `session_20260424_4people_compare.py`** — the 4-row contact sheet of REAL captures from the same Teams session: Xiang_Xiang2_Feng (model: real), Xinhe_XH68_Wang (model: real), tester_tester (model: real), dor_shkedi (model: FAKE p>0.9). Same camera, same compression, same session. **The decisive visual** for ruling out "session-wide pipeline as the only signal" — three identities pass, one (Dor) flips. Driver code at line 16: `IDENTS = [...4 names...]`. 12 frames each, evenly sampled across captures.

2. **`check_frame_4people_2026-04-24.json` + `.py`** — quantitative companion. 30–32 frames per identity scored via `check-frame` CLI (production inference path):

| Identity | n | mean prob_fake | std | frac>0.9 | frac<0.1 | Overall |
|---|---:|---:|---:|---:|---:|---|
| Xiang_Xiang2_Feng | 30 | 0.0089 | 0.008 | 0.000 | 1.000 | REAL |
| Xinhe_XH68_Wang | 30 | 0.0295 | 0.093 | 0.000 | 0.967 | REAL (one outlier 0.521) |
| dor_shkedi | (same session, the Dor capture) | (failing — see fingerprint diff §10.3) | | | | flipped |
| tester_tester | (third real) | (caught real) | | | | REAL |

This is the controlled test that grounds the camera/pipeline-signature claim. Three different humans, one camera/session, same pipeline; only one (Dor) flips. Proves the failure is identity-specific within this session — but the cross-checkpoint anchor matrix (§4) shows it's not pure identity (Roee-Mac also flips on a different camera, and Dor-laptop is fine). So it's a **per-(identity × camera) interaction** the model latches onto.

(Note: `check-frame` is from `/scripts/launch/launch_batch_inference.sh` family — production inference path. Includes the post-2026-04-24 INTER_LINEAR fix.)

**Source artifacts.**
- `analysis/session_20260424_4people_compare.py` + `.png`
- `analysis/check_frame_4people_2026-04-24.py` + `.json`

---

## 14 — Promotion contract scorecard outputs (`arena/reports/*.json` and adjacents)

**What's available.** The `arena/reports/` directory has 5 build/eval reports, but none are full promotion-contract scorecard outputs — those land at `gs://training-job-outputs/test_results/teams_promotion_contract/<scorecard-name>/`.

**Files in `arena/reports/`:**

| File | Date | Role |
|---|---|---|
| `hdtf_visomaster_clean_vs_teams_join_2026-04-19.json` | 2026-04-19 | hdtf manifest join report |
| `proper_visomaster_wave_2026_04_19_provisional_build_report.json` | 2026-04-21 | wave inventory build report |
| `quickclips_visomaster_clean_vs_teams_join_2026-04-19.json` | 2026-04-19 | quickclips manifest join report |
| `visomaster_proper_clean_bucket_census_2026-04-19.json` | 2026-04-19 | clean bucket census |
| `visomaster_proper_teams_bucket_census_2026-04-19.json` | 2026-04-19 | teams bucket census |

These are **data-build reports**, not scorecard outputs. The scorecard outputs are GCS-only:

**Scorecard runs that we know exist (from handoff docs and checkpoint-map yamls):**

| Scorecard run | Date | Checkpoints scored | GCS prefix | Key result |
|---|---|---|---|---|
| `p8a-review-scorecard-20260425` | 2026-04-25 | P8A_STEP5000, RLP7_05_STEP3500, RLP6_04_STEP23500 | `gs://training-job-outputs/test_results/teams_promotion_contract/p8a-review-scorecard-20260425/reports/` | P8A: lockbox_real_fpr 0.147% / fake_recall 23.7% (vs RLP6_04 0.441% / 23.3%); per-method visomaster_enhanced 1.1% (vs 3.3%), deeplive_enhanced 2.4% (vs 6.4%) — see `R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md` §7.2 + `PACKET_9_MID_FLIGHT_HANDOFF_2026-04-26.md` |
| `p9-review-scorecard-20260426` | 2026-04-26 | P9_05, P9_FREEZE, P9_R, P9_FORK_DATA + P8A + RLP6_04 | `gs://training-job-outputs/test_results/teams_promotion_contract/p9-review-scorecard-20260426/promotion_contract/checkpoint_summary.csv` | **In progress when `PACKET_9_MID_FLIGHT_HANDOFF` was written** (job `2877145196556976128` started 07:46 UTC, ETA 10:45 UTC) |
| (none yet) for P10 | — | — | — | — |

**Checkpoint-map yamls (the inputs to the scorecard launcher):**
- `arena/checkpoint_maps/teams_target_domain.p8a_review_2026-04-25.yaml` (P8A + RLP7_05 + RLP6_04)
- `arena/checkpoint_maps/teams_target_domain.p9_review_2026-04-26.yaml` (4 P9 finished + P8A + RLP6_04)
- `arena/checkpoint_maps/teams_target_domain.p10_partial_2026-04-26.yaml` (3 P10 early-stop + P8A + RLP6_04 — **not yet launched** as of mid-flight handoff)
- `arena/checkpoint_maps/teams_target_domain.r13_finalists_2026-04-12.yaml` (older)
- `arena/checkpoint_maps/teams_target_domain.promotion_shortlist_2026-04-17.yaml` (older)

**Crowning protocol (from each yaml's headnote, 2026-04-26 version):**

1. `lockbox_real_fpr ≤ RLP6_04.lockbox_real_fpr` (≤ 0.441%)
2. `lockbox_fake_recall ≥ max(0.80, RLP6_04 − 0.02)` — i.e., ≥21.3% macro
3. Per-method recall on `deeplive_enhanced` ≥ 3.4%, `visomaster_enhanced_macro` ≥ 0.3% (RLP6_04 − 3pt slack)
4. 7% lockbox FPR hard cap.

The `selected_threshold` field in scorecard outputs must be checked because of the contract-policy bug (memory `project_contract_policy_bug.md`): when the FPR budget is too tight, τ-search drives τ to ~0.995 and crushes recall to ~23%. The values above (0.441% FPR / 23.3% recall) are *post-bug* — i.e., what we get *with* τ inflation. If the bug is fixed, those numbers will change.

**Source artifacts.**
- `arena/score_teams_promotion_contract.py:430-479` (lexicographic τ search)
- `arena/launch_teams_promotion_contract.sh` (launcher)
- `arena/run_teams_promotion_contract.sh` (worker entrypoint)
- `arena/run_target_domain_validation_sequential.py` (per-suite eval driver)
- `arena/postprocess_per_identity.py` (per-identity FPR reducer added in WS-P2.b)
- `arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json` (latest manifest)
- `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml` (suite map)

**Verdict status / methodological caveats.**

1. **The scorecard CSVs are GCS-only.** I have not opened them in this audit; the per-suite numbers in §3 and §5 above came from `R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md` §7 (which was authored against the in-progress P8A scorecard).
2. **Contract-policy bug.** Always check `selected_threshold` — if τ ≥ 0.995, the contract may be inflating real_fpr metrics. Memory `project_contract_policy_bug.md` is the authority.
3. **Trainer's `value_composite` is NOT the contract metric.** Multiple past packets crowned wrong checkpoints by trusting it. Always cross-check against the contract scorecard.
4. **No P10 scorecard yet.** As of `PACKET_9_MID_FLIGHT_HANDOFF_2026-04-26.md`, the P10 trio has only had the 30-frame anchor rescore (§7); no per-method dev/lockbox scorecard run.

---

## 15 — Calibration probe (`analysis/calibration_probe_2026-04-24.summary.json`)

**Subject.** Adjacent to the shortcut question: tests whether per-camera τ calibration could close the FPR gap across the 6 anchor pools without retraining.

**Method (from JSON `config`).** For each target FPR ∈ {0.05, 0.10, 0.15, 0.25}, find the global τ that meets that FPR using a 50/50 random split of the 30-frame pools (15 train, 15 test, seed=42). Then compute per-pool τ that meets that FPR within each pool. Compare test-FPR under global vs per-pool τ.

**Headline result (from JSON `summary`):**

```
avg_gap_closure_across_target_fprs: 0.301
gap_closure_per_target_fpr: { 0.05: 0.0, 0.1: 0.333, 0.15: 0.286, 0.25: 0.583 }
verdict: mixed_both_levers_needed
verdict_thresholds: { calibration_is_right_lever: ">= 0.60", training_augmentation_is_right_lever: "<= 0.30" }
```

So at low target FPRs (5%), per-pool calibration closes 0% of the cross-pool FPR gap. At higher target FPRs (25%) it closes 58%. The aggregate is 30.1% — exactly at the boundary between "training-aug is the right lever" (≤0.30) and "calibration is right lever" (≥0.60).

**Implication.** Camera-signature shortcut FPR is **not** purely a calibration shift. Even with perfect per-camera τ knowledge, only ~30% of the gap closes. The remaining 70% is structural — i.e., the model's score distributions on Dor-webcam-noVB (test_mean 0.946) and Roee-Mac (test_mean 0.922) overlap so heavily with the threshold region that no τ can separate them cleanly from the correct pools (test_mean 0.008–0.047).

**Source.** `analysis/calibration_probe_2026-04-24.py` and `analysis/calibration_probe_2026-04-24.summary.json`.

**Caveats.** This is run on **slot-07 era** scores (per the prior session's data, before P7 / P8A / etc.). For each new checkpoint family, the calibration-probe should be re-run; the gap-closure on P8A specifically would be a useful number to have but **does not exist**.

---

## 16 — What we know vs what we suspect (the bottom line)

### 16.1 — Decisively known

1. **The shortcut is real and reproducible.** Same person, same lighting, same face, different camera/pipeline → different output. Demonstrated in three independent settings: (a) slot-07 lockbox dor_shkedi vs real_dor (§1); (b) controlled 2026-04-24 capture, 4 identities one camera one session — only Dor flips (§13); (c) cross-subject Roee-Mac vs Roee-Win (§4 Roee-Mac column).
2. **The shortcut is not pure identity.** Dor-laptop is fine (mean 0.013 on RLP6_04). Roee flips on Mac, not on Windows. Cross-subject confirmed.
3. **The shortcut is not removable by FT-only interventions on top of RLP6_04.** Six P7 variants spanning three aug axes and two fork points all cluster in [0.835, 0.916] anchor mean (§6).
4. **The shortcut IS partially fixable by unfreezing the CLIP backbone.** P8A drops anchor to 0.744 / `frac_gt_0_9` 0.43 (§5).
5. **Scratch-on-plain-CLIP makes the shortcut worse.** P8B saturates at 0.99+ on the anchor (§4.1, §6).
6. **P8A's anchor win comes with a measured per-method fake-recall regression on dev** at default τ (§3): aggregate −13.66pp, concentrated in deeplive_enhanced (−26.6pp), teams_flat_xiang_xiang2_feng (−33.3pp), visomaster_enhanced_macro (−21.6pp). Failure-mode analysis says ~80% of regressed scores are below the τ-recoverable band — separability loss.
7. **Pixel-level fingerprints separate failing-vs-clean pools** along lighting + codec axes (`dynamic_range`, `wb_rb_ratio`, `dct_hf_ratio`, `mean_cb`, `mean_cr`) consistently across Dor and Roee, with negative controls confirming directionality (§10).
8. **Per-camera τ calibration alone closes only ~30% of the cross-pool FPR gap** at slot-07's score distribution (§15). The rest is structural.

### 16.2 — Suspected but not decisively measured

1. **The shortcut lives in "data mix × frozen-CLIP backbone interaction"** — a precise causal locus is implied but not directly measured. The "upstream of RLP6_04" framing in memory is the closest formulation.
2. **The visomaster-non-enhanced lanes are post-smoothed and may be teaching "smoothing = fake."** Bucket-comparison Laplacian-variance numbers (51–56) suggest this but no controlled experiment has been run.
3. **The s33 outlier may be a per-session pipeline anomaly** rather than a per-method effect — but no session-by-session breakdown of the cam_test family on lockbox exists.
4. **P8A's fake regression may concentrate at codec-similar-to-real-aug fakes** (the `visomaster_enhanced_teams` 26% regression rate vs `visomaster_enhanced_raw` 17%). Suggestive but only one comparison.
5. **The `wma_failure_fake` pool drives the OOD gate spuriously** because its quality stats are 5–10× off the training distribution — but no formal "remove wma, what does the gate say" experiment has been run.

### 16.3 — Known unknowns

1. **No lockbox-side per-method P8A scorecard CSV opened in this audit.** All P8A per-method numbers come from the dev partition (`teams_fake_all_dev`, 2409 vids). The lockbox per-method gap may be smaller or larger.
2. **No per-checkpoint calibration probe.** The 30%-gap-closure number is slot-07 only.
3. **No formal face-recognition similarity check** on the dor_shkedi vs real_dor identity claim — only visual.
4. **No multi-step trajectory analysis** of how the shortcut consolidates within RLP6_04 training (we only have step 4500 vs step 23500 endpoints).
5. **The P10 contract scorecard hasn't run yet.** Anchor numbers only (and they don't look promising — see §7).
6. **No frame-level "which CLIP features fire on shortcut frames"** analysis. The `feature_space_2026-04-23` work computed only centroid/Fréchet/MMD aggregates — no per-frame attribution.

---

## 17 — Quick reference: every artifact this report touched

### Memory notes (`~/.claude/projects/.../memory/`)
- `project_signature_shortcut_finding.md` (slot-07 finding)
- `project_p8a_breakthrough.md` (P8A claim)
- `project_shortcut_is_upstream.md` (P7 ceiling investigation)
- `project_success_criteria.md` (three pillars)
- `project_contract_policy_bug.md` (τ inflation issue)
- `project_promotion_contract.md` (contract-not-value_composite)

### Anchor-pool rescore JSONs (`analysis/`)
- `rlp6_04_postfix_rescore_2026-04-24.summary.json` (+ `.per_frame.csv`, `.py`)
- `pool_rescore_rlp7_0{2,4,5,6,7,8}.summary.json` (+ `.per_frame.csv` each)
- `pool_rescore_rlp8_a.summary.json`, `pool_rescore_rlp8_a_step2500.summary.json`
- `pool_rescore_rlp8_b.summary.json`, `pool_rescore_rlp8_b_step5000.summary.json`
- `pool_rescore_p10_grl_step4500.summary.json`, `pool_rescore_p10_sym_step4500.summary.json`, `pool_rescore_p10_sym_light_step5500.summary.json`
- `teams_pool_rescore.py` (driver)

### Fingerprint diff (`analysis/`)
- `dor_pool_fingerprint_diff_2026-04-24.py`
- `dor_pool_fingerprints_2026-04-24.summary.json`, `.csv`, `.png`
- `_fingerprint_cache_2026-04-24/` (84 files)
- `dor_roee_combined_2026-04-24.summary.json`, `.csv`, 5 `*_pair_*.png`, `cross_pair_summary.png`, `score_correlation.png`
- `_fingerprint_cache_combined_2026-04-24/` (182 files)
- `dor_baseline_repro_2026-04-24.summary.json`, `.csv`, 3 `*.png`
- `fingerprint_diff.py` (generic driver)
- `configs/dor_baseline_repro.yaml`, `configs/dor_roee_combined_2026-04-24.yaml`

### Visual contact sheets (`analysis/`)
- `lockbox_failure_contact_2026-04-24.png` + `.index.json` + `.py`
- `dor_closeup_2026-04-24.png` + `.py`
- `enhanced_fakes_contact_2026-04-24.png` + `.py`
- `session_20260424_4people_compare.png` + `.py`
- `_dor_closeup_cache_2026-04-24/`, `_enh_fake_cache_2026-04-24/`, `_lockbox_contact_cache_2026-04-24/` (frame caches)

### check-frame quantitative companions (`analysis/`)
- `check_frame_4people_2026-04-24.json` + `.py`

### P8A failure-mode analysis (`analysis/p8a_fake_failure_analysis_2026-04-25/`)
- `analyze.py`
- `summary.json` (full structured findings)
- `regressed_videos.csv` (329 rows)
- `teams_fake_all_dev_p8a_step5000_{summary_report.txt, frames_report.csv, videos_report.csv, group_metrics.csv}`
- `teams_fake_all_dev_rlp6_04_step23500_{summary_report.txt, frames_report.csv, videos_report.csv, group_metrics.csv}`

### Feature-space + bucket-comparison (`analysis/`)
- `feature_space_2026-04-23/REPORT.md`, `compute_distances.py`, `extract_features.py`, `launch_vertex.sh`, `distance_matrix_*.csv`, `per_source_summary.csv`
- `bucket_comparison_2026-04-23/REPORT.md`, `sample_and_analyze.py`, `make_thumbnail_grid.py`, `per_image_stats.csv`, `per_source_summary.csv`, `thumbnails_grid.png`, `cache/<source>/*.png`

### Calibration / WS-P1
- `analysis/calibration_probe_2026-04-24.summary.json` + `.py`

### Checkpoint maps (`arena/checkpoint_maps/`)
- `teams_target_domain.p8a_review_2026-04-25.yaml`
- `teams_target_domain.p9_review_2026-04-26.yaml`
- `teams_target_domain.p10_partial_2026-04-26.yaml`
- `teams_target_domain.r13_finalists_2026-04-12.yaml` (older)
- `teams_target_domain.promotion_shortlist_2026-04-17.yaml` (older)

### Manifests, suites (`arena/`)
- `manifests/teams_target_domain_manifest_2026-04-23_with_dor.json`
- `target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml`
- `score_teams_promotion_contract.py`, `launch_teams_promotion_contract.sh`, `run_teams_promotion_contract.sh`
- `postprocess_per_identity.py` (per-identity reducer)

### Compare-pools diagnostic (separate from anchor matrix)
- `analysis/compare_teams_pools.py` (1100 lines, 17 unit tests)
- `tests/test_compare_teams_pools.py`
- `docs/superpowers/specs/2026-04-22-compare-teams-pools-design.md`
- `scratch/teams_pool_diff/2026-04-23T00-25-36/{report.html, raw_scores.parquet, stats_pass_model.json}` (the 150×8 authoritative run)

### Handoff narratives (`docs/relaunch_handoffs/`)
- `R13_RELAUNCH_VISOMASTER_TEAMS_POOL_DIAGNOSTIC_FINDINGS_2026-04-23.md` (visomaster lane bug + 67% in-training acc finding)
- `R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md` (P7 launch context + INTER_LINEAR fix)
- `R13_RELAUNCH_STATE_OF_THE_TEAMS_DETECTOR_2026-04-25.md` (anchor-focused state-of-detector)
- `R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md` (compact full narrative + P8A regression discovery)
- `PACKET_9_MID_FLIGHT_HANDOFF_2026-04-26.md` (today's pivot context)
- `analysis/overnight_packet7_packet8_summary_2026-04-25.md` (P7/P8 summary)

---

## 18 — Caveats specific to this inventory

1. I did not open the **promotion-contract scorecard CSVs on GCS**. The lockbox numbers in §3 and §5 came from `R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md` §7, which was authored *during* the in-progress P8A scorecard. The aggregator may have shifted those numbers slightly when it landed.
2. I did not open the slot-07 underlying CSVs at `/tmp/r13_analysis/frames/teams_*_r13_rlp5_07_e3_seedb_frames_report.csv` — those are referenced by the contact-sheet scripts but live outside the repo. The numbers I quoted (1138 vids, 14.6%>0.9, etc.) come from the slot-07 finding memory file.
3. The **30-frame anchor pool sample size** caveat applies everywhere in §4.
4. **Pre-fix vs post-fix preprocessing** (§4.5 caveat 3): every number tagged ≥2026-04-24 12:22 UTC uses `cv2.INTER_LINEAR`. Numbers in the 2026-04-24 Packet-7 handoff that are *before* this fix are slightly inflated (anchor RLP6_04 ~0.94 pre-fix vs 0.93 post-fix).
5. The s33 outlier numbers (§2) are slot-07 only. The post-RLP6_04 / P8A behavior on s33-on-lockbox is **not in any artifact I could find**.
6. The "frac_full_flip 100% on regressed methods" finding (§3) is striking — but its frame-level standard deviations are mostly NaN because each video has ~1.26 frames on average. Treat the frame-level analysis as effectively video-level.

---

*End of sub-report 04 — Shortcut Evidence Inventory.*
