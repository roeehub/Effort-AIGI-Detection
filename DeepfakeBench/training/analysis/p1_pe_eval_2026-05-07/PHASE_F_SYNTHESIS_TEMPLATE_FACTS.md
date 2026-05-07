# Phase F Synthesis — P1 (PE_PAIR_RANK_DRO)

**Status**: SKELETON — fill in once Phase A + Phase C land.

| field | value |
|---|---|
| Packet ID | R13_P1 (PE_PAIR_RANK_DRO) |
| Slot 1 (BUNDLE: pair-rank + GroupDRO) | W&B `tznuar61` / Vertex `7995519158412378112` |
| Slot 2 (PAIRRANK_ONLY ablation) | W&B `s2mp5fxm` / Vertex `524047376604725248` |
| Expected fill-in | 2026-05-08 (after Phase A 29-suite + Phase C HDTF scorecards land) |
| Handoff | `docs/relaunch_handoffs/HANDOFF_P1_EVAL_IN_FLIGHT_2026-05-07.md` |
| YAML (BUNDLE) | `experiments/phase2_round13/R13_P1_BUNDLE_FT_FROM_P8A.yaml` |
| YAML (PAIRRANK_ONLY) | `experiments/phase2_round13/R13_P1_PAIRRANK_ONLY_FT_FROM_P8A.yaml` |

---

## 1. Pass / fail / partial table — close criteria F1-F5 × 8 ckpts

Close criteria copied from YAML headers (PAIRRANK_ONLY F1-F4; BUNDLE F5):

- **F1**: lockbox fake recall >=90% at FPR <=10%.
- **F2**: pair-rank metric — fraction `fake_score > real_score` on previously missed fakes lifts >=30% on at least 2 of 6 paired lanes; worst-group recall lift >=20% on chronic / dor-drift cohorts.
- **F3**: no untargeted axis (is_webcam, face_area_fraction, min_dim, color_b_dev) amplifies +50%.
- **F4**: HDTF cross-substrate FPR <=5%.
- **F5** (BUNDLE only): chronic-FP `pc_generator` cluster failure rate (P8A baseline 0.520) drops by >=0.10 absolute via the GroupDRO chronic_flag term.

| ckpt | F1 | F2 | F3 | F4 | F5 |
|---|---|---|---|---|---|
| `p8a_reference_step5000` | TBD | TBD | TBD | TBD | TBD |
| `e2b_top_n_step3200` | TBD | TBD | TBD | TBD | TBD |
| `p1_bundle_step500` | TBD | TBD | TBD | TBD | TBD |
| `p1_bundle_step3750` | TBD | TBD | TBD | TBD | TBD |
| `p1_bundle_step4000` | TBD | TBD | TBD | TBD | TBD |
| `p1_pairrank_step500` | TBD | TBD | TBD | TBD | n/a |
| `p1_pairrank_step6000` | TBD | TBD | TBD | TBD | n/a |
| `p1_pairrank_step6750` | TBD | TBD | TBD | TBD | n/a |

Cell legend: `pass` / `fail` / `partial` / `TBD`. Ckpt list pulled from `arena/checkpoint_maps/teams_target_domain.p1_pe_pair_rank_2026-05-07.yaml` (the map Phase A actually scored).

---

## 2. Slot-1 vs Slot-2 ablation — BUNDLE minus PAIRRANK_ONLY delta

Question (per memory `anti_shortcut_bundle_decomposition`): **what tradeoff did GroupDRO buy?** — NOT "did GroupDRO add anything". A negative delta on one criterion + positive delta on another is the expected ablation read.

Pair the matched-step ckpts. Slot 1 BUNDLE seed=5072; Slot 2 PAIRRANK_ONLY seed=5071 — interpret single-seed delta with caution.

| pair | F1 delta | F2 delta | F3 delta | F4 delta | F5 delta |
|---|---|---|---|---|---|
| step500 BUNDLE − step500 PAIRRANK_ONLY (matched-step) | TBD | TBD | TBD | TBD | TBD |
| step3750 BUNDLE − step6000 PAIRRANK_ONLY (mid-trajectory; ckpts not matched-step) | TBD | TBD | TBD | TBD | TBD |
| step4000 BUNDLE − step6750 PAIRRANK_ONLY (final-vs-final; differ by 2750 steps) | TBD | TBD | TBD | TBD | TBD |

Note: PAIRRANK_ONLY early-stopped at epoch 4 step 2750 (W&B `s2mp5fxm`); BUNDLE early-stopped at epoch 3 step 2750 (W&B `tznuar61`). Top-N ckpts at step6000/6750 (PAIRRANK) and step3750/4000 (BUNDLE) sit past the early-stop point — they reflect post-EarlyStop continued evaluation, not "training kept going past stop". step500 is the only true matched-step pair.

---

## 3. Phase E weight-delta — diagnostic sub-row (NOT a gate)

Source: `analysis/p1_pe_eval_2026-05-07/outputs/weight_delta_verdict.csv`. Frobenius-norm delta from CLIP-DataComp-XL pretrained for the 12 ViT-B-16 resblocks.

| ckpt | qkv_mean_fnorm | out_proj_mean_fnorm | mlp_mean_fnorm | qkv/out_proj | qkv/mlp | in Phase A scope? |
|---|---:|---:|---:|---:|---:|:---:|
| `p1_bundle_step500` | 0.00804 | 0.10099 | 0.24492 | 0.0796 | 0.0328 | yes |
| `p1_bundle_step1000` | 0.01245 | 0.17415 | 0.39837 | 0.0715 | 0.0313 | no (trajectory-only) |
| `p1_bundle_step3750` | 0.01918 | 0.25287 | 0.58366 | 0.0759 | 0.0329 | yes |
| `p1_bundle_step4000` | 0.01918 | 0.25279 | 0.58472 | 0.0759 | 0.0328 | yes |
| `p1_pairrank_step500` | 0.00784 | 0.10213 | 0.24878 | 0.0767 | 0.0315 | yes |
| `p1_pairrank_step1000` | 0.01225 | 0.17109 | 0.40821 | 0.0716 | 0.0300 | no (trajectory-only) |
| `p1_pairrank_step6000` | 0.01902 | 0.25739 | 0.58893 | 0.0739 | 0.0323 | yes |
| `p1_pairrank_step6750` | 0.02032 | 0.26811 | 0.61628 | 0.0758 | 0.0330 | yes |

Phase E now overlaps fully with the 6 P1 ckpts in Phase A scope (step500/3750/4000 BUNDLE, step500/6000/6750 PAIRRANK_ONLY); the 2 step1000 rows are trajectory-only and have no scorecard counterpart to attach to.

Read framework (per handoff §"Phase E"): qkv/out_proj << 1 and qkv/mlp << 1 indicates `apply_svd_to_in_proj` lever is being used at low magnitude relative to the unfrozen out_proj + MLP-SVD parameters. Across all 8 P1 ckpts the qkv/out_proj ratio sits in [0.0715, 0.0796] and qkv/mlp in [0.0300, 0.0330] — the in_proj-SVD lever is active and consistent across both arms × all training steps.

**Caveats**:

1. **No `apply_svd_to_in_proj=False` baseline** in this packet. Ratios are relative-only — they show "lever is active and consistent across BUNDLE vs PAIRRANK_ONLY" but cannot answer "is the lever load-bearing for the F1-F5 outcomes?". A future packet with the lever toggled off as the only variable is the only way to answer that.

2. **W&B logging gap on BUNDLE diagnostic loss components.** When `use_group_dro: true`, `trainer/trainer.py:1718-1727` extracts only `per_sample_losses_dict['overall']` and replaces the dict via `calculate_group_dro_loss()` — silently discarding the 11+ diagnostic scalars from `effort_detector.py:1345-1366` (including `pair_rank_loss` at line 1364, `cls_loss`, `corr_penalty_loss`, `feat_norm_loss`). The W&B log loop at `trainer.py:1847-1857` only sees the DRO mixin's 4-key return dict (`{overall, group_weights, group_losses_ema, group_dro_in_warmup}`).

   - **PAIRRANK_ONLY (`s2mp5fxm`)** logs `train/loss/pair_rank_loss` directly (median 0.113 / max 0.919 / final-step 0.025 — fired throughout, no collapse).
   - **BUNDLE (`tznuar61`)** has no scalar magnitude logged for either `pair_rank_loss` or any `group_dro_*` loss component. Only `train/loss/group_dro_in_warmup` (binary flag, transitions 1→0 at step 50 as designed) and `train/diagnostic/{group_losses_ema,group_weights}` (histograms) are present.
   - One-line fix: `trainer.py:1727` — `losses.update({k: v for k, v in per_sample_losses_dict.items() if k != 'overall'})`. Affects every `use_group_dro=True` run since this code path was introduced.
   - **Implication for F2/F5 interpretation**: BUNDLE's pair-rank lever activation is unverifiable from W&B history alone. To verify, replay a few batches against the BUNDLE checkpoint with the same data loader seed (cannot be reconstructed from the checkpoint itself — depends on per-batch `pair_id` composition).

---

## 4. Phase A.5 — partial coverage on diagnostic substrates (4 valid + 5 invalid)

Source: `analysis/p1_pe_eval_2026-05-07/diagnostic_substrates/per_suite_comparison_tau_0.5.csv`. Inferred at tau=0.5 via Mac CPU substrate.

### 4 valid suites (use these in synthesis)

| suite | n | label | score_P8A | score_E2B | score_PA_3800 | score_P1_BUNDLE_step4000 | score_P1_PAIRRANK_step6750 |
|---|---:|---|---:|---:|---:|---:|---:|
| `xinhe_may6_falseflag` | 92 | real (FPR) | 0.0000 | 0.5761 | 0.1630 | 0.0978 | 0.1304 |
| `live_fakes_teams_prod` | 1675 | fake (recall) | 0.7534 | 0.8113 | 0.5039 | 0.7409 | 0.7887 |
| `visomaster_v2_dor` | 2073 | fake (recall) | 0.7381 | 0.5596 | 0.2523 | 0.6522 | 0.7771 |
| `team_sanity_may5` | 210 | real (FPR) | 0.0095 | 0.0095 | 0.0048 | 0.0095 | 0.0143 |

### 5 invalid suites — DO NOT USE

| suite | n | reason |
|---|---:|---|
| `live_reals_teams_prod` | 677 | bucket layout migration — paths gone |
| `dor_evening` | 324 | `gs://local/...` placeholders |
| `dor_morning` | 244 | `gs://local/...` placeholders |
| `dor_fake_local` | 605 | `gs://local/...` placeholders |
| `extra` | 918 | `gs://local/...` placeholders |

Constant-score artifacts (`score_*_FPR=1.0`, `score_P1_PAIRRANK_FPR=0.0` etc.) on these 5 suites are zero-tensor head-bias readings, NOT model judgments. Do not interpret them in any verdict synthesis. See open loop `grouped-manifest-v2-stale-paths` in `threads/eval_substrate_layering.md`.

---

## 5. Dor invariance probe — fresh 6-variant substrate (180 frames)

Source: `analysis/p1_pe_eval_2026-05-07/dor_invariance_2026-05-07/per_variant_fpr.csv`. All frames are reals (lower FPR is better). tau=0.5.

| variant | n | score_P8A_FPR | score_E2B_FPR | score_P1_BUNDLE_step4000_FPR | score_P1_PAIRRANK_step6750_FPR |
|---|---:|---:|---:|---:|---:|
| `dor_laptop_whiteish` | 30 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `dor_laptop_yellowish` | 30 | 0.0000 | 0.0000 | 0.3000 | 0.2000 |
| `dor_session_0411` | 30 | 0.0333 | 0.0333 | 0.0333 | 0.0333 |
| `dor_session_0424` | 30 | 0.7000 | 0.3667 | 0.6000 | 0.8667 |
| `dor_webcam_no_vbg` | 30 | 0.8000 | 0.3667 | 0.3333 | 0.7667 |
| `dor_webcam_with_vbg` | 30 | 0.7667 | 0.0000 | 0.2333 | 0.6667 |
| **Combined (180)** | 180 | 0.3833 | 0.1278 | 0.2500 | 0.4222 |

**Factual observation (flagged for user attention, not interpreted here)**: on this fresh dor probe, E2B combined FPR = 0.1278 vs P8A combined FPR = 0.3833 — E2B is ~3x lower than P8A. This contradicts the cohort-diagnosis-2026-05-06 reading that gave P8A the dor-invariance edge on `dor_evening` / `dor_morning`. The dor_evening / dor_morning suites sit in the 5-suite invalid set above (stale `gs://local/...` placeholders), so the cohort-diagnosis numbers cannot be re-verified on this Mac. Phase F should call the user's attention to this substrate-mismatch and let them adjudicate which substrate is the deployment-relevant dor proxy.

---

## 5b. Dor probe — IQ-axis trajectory across all 6 P1 ckpts

Source: `analysis/p1_pe_eval_2026-05-07/dor_invariance_2026-05-07/axis_decoupling_trajectory.csv` and `axis_correlation_summary.csv`. n=180 (combined across 6 dor variants). Axes: `sharpness` (Laplacian variance), `min_dim` (min(W, H)), `color_b_dev` (B-channel std). Pearson r between per-frame axis value and per-frame raw score; "delta_r" = correlation of (score_ckpt − score_P8A) against the axis.

For reference: P8A raw-r combined = `sharpness=−0.644`, `min_dim=+0.447`, `color_b_dev=` (low). E2B raw-r combined = `sharpness=−0.379`, `min_dim=−0.017`.

| arm | step | sharpness raw_r | sharpness delta_r | min_dim raw_r | min_dim delta_r | color_b_dev raw_r | color_b_dev delta_r |
|---|---:|---:|---:|---:|---:|---:|---:|
| BUNDLE | 500 | −0.272 | +0.451 | +0.338 | −0.151 | −0.016 | −0.398 |
| BUNDLE | 3750 | −0.342 | +0.439 | +0.192 | −0.354 | −0.045 | −0.463 |
| BUNDLE | 4000 | −0.317 | +0.476 | +0.181 | −0.369 | −0.044 | −0.446 |
| PAIRRANK | 500 | −0.517 | +0.557 | +0.298 | −0.462 | +0.218 | −0.342 |
| PAIRRANK | 6000 | −0.600 | +0.156 | +0.354 | −0.237 | +0.243 | −0.210 |
| PAIRRANK | 6750 | −0.631 | +0.044 | +0.407 | −0.096 | +0.196 | −0.288 |

Extended per-variant FPR @ tau=0.5 (source: `per_variant_fpr_extended.csv`; all reals, lower is better):

| variant | n | BUNDLE step500 | BUNDLE step3750 | BUNDLE step4000 | PAIRRANK step500 | PAIRRANK step6000 | PAIRRANK step6750 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dor_laptop_whiteish` | 30 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `dor_laptop_yellowish` | 30 | 0.933 | 0.400 | 0.300 | 0.067 | 0.067 | 0.200 |
| `dor_session_0411` | 30 | 0.033 | 0.033 | 0.033 | 0.033 | 0.033 | 0.033 |
| `dor_session_0424` | 30 | 1.000 | 0.667 | 0.600 | 0.367 | 0.867 | 0.867 |
| `dor_webcam_no_vbg` | 30 | 1.000 | 0.433 | 0.333 | 0.567 | 0.700 | 0.767 |
| `dor_webcam_with_vbg` | 30 | 1.000 | 0.300 | 0.233 | 0.433 | 0.533 | 0.667 |

**Factual observations (flagged for user attention, NOT interpreted here):**

1. BUNDLE sharpness raw_r is in [−0.342, −0.272] across all three steps; PAIRRANK sharpness raw_r is in [−0.631, −0.517] across its three steps. The two arms occupy different bands at every measured step including step500. P8A baseline raw_r = −0.644.
2. PAIRRANK_ONLY combined-FPR per-variant: dor_session_0424 0.367 → 0.867 across step500 → step6750; dor_webcam_with_vbg 0.433 → 0.667. PAIRRANK_step500 has lower FPR than P8A on three variants (0_0424, no_vbg, with_vbg); PAIRRANK_step6750 has higher FPR than P8A on one (0_0424) and lower on two.
3. BUNDLE_step500 has FPR ≥ 0.93 on three variants (yellowish, 0_0424, no_vbg, with_vbg) while simultaneously showing the lowest sharpness raw_r magnitude (0.272). BUNDLE step4000 FPR drops to ≤ 0.60 on those variants while sharpness raw_r stays in the same band (0.317).
4. The ablation question (slot-1 vs slot-2) cannot be read off step500 alone — both arms are in different regimes there. Phase A's selected_threshold scorecard is the load-bearing arbiter; the table above is supporting context only.

---

## 6. P1 vs PD comparison — placeholder

Reference: `analysis/pd_scorecard_artifacts_2026-05-06/unified_scorecard_simple.csv`. PD ran the same 29-suite contract that Phase A is currently running for P1.

When Phase A lands, fill this section with:

- Per-suite recall / FPR P1_BUNDLE vs the strongest PD ckpt at FPR-matched tau.
- Whether P1 reproduces or refutes the PD finding `project_dor_drift_named_axes_2026-05-06` (PD captured 90% of dor drift via named pixel-domain IQ axes; does P1 capture more / less / similar?).
- Net structural read: did the corr_penalty=OFF + pair_rank+GroupDRO swap recover the substrate invariance that PD lost while keeping PD's IQ-shortcut suppression?

| suite | PD best | P1_BUNDLE_step4000 | P1_PAIRRANK_step6750 | delta vs PD |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD |

---

## 7. Verdict

**VERDICT DEFERRED — user makes the call.**

The numbers in sections 1-6 stand for the user to read independently. Do NOT write a verdict here. Do NOT advocate for promotion / non-promotion. Do NOT pre-pick a winning slot. The agent's job in Phase F is to surface the pass/fail/partial cells truthfully and flag attention-points (e.g. the dor probe E2B-vs-P8A reversal); the user does the synthesis.

If the user requests a verdict after reading, draft it as a follow-up section under this header — never as a replacement for this paragraph.

---

## 8. Open loops to close on verdict

| loop id | closes if | source |
|---|---|---|
| `contract-policy-bug-fix-not-committed` (component 3) | Phase A scorecards ran with the `--promotion-target-real-fpr 0.07 --promotion-target-stress-fpr 0.10 --promotion-target-fake-recall-min 0.30` recall-floor flags and reports show non-collapsed tau tails | `threads/contract_policy_bug.md` |
| `grouped-manifest-v2-stale-paths` | only closes if the 5 invalid Phase A.5 suites are re-run against a fixed manifest; does NOT close from Phase F itself | `threads/eval_substrate_layering.md` |
| `diagnostic-substrates-not-in-contract` | Phase A.5 partial-coverage results judged sufficient by user, OR contract is regenerated with the diagnostic substrates included | `threads/eval_substrate_layering.md` |

Any new open loops surfaced by Phase F (e.g. unexpected behavior in F2 pair-rank lift, untargeted-axis amplification in F3, HDTF cross-substrate regression in F4) should be appended to this section AND added to `OPEN_LOOPS.md` via `tools/regenerate_open_loops.py`.
