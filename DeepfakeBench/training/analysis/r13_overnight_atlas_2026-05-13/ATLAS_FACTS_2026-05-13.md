# L11 atlas inv_mean per-substrate recompute for R13 overnight candidates — FACTS (2026-05-13)

> **Status: factual-only.** No interpretation. Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably.
> Numbers + tables only. Interpretation belongs in `AGENT_PROPOSAL_<date>.md`.
>
> **Scope**: CPU diagnostic. Recomputes A3's per-substrate L11 LR-probe `inv_mean` on the 800-frame triptych for one representative ckpt per slot of the 2026-05-13 overnight R13 GPU batch (per packet retro [`packets/U_SLOTS_2026-05-13.md`](../../docs/packet_retrospectives/packets/U_SLOTS_2026-05-13.md)). P8A_REFERENCE_STEP5000 included for baseline parity verification against the prior A3 measurement.
>
> **Inputs**:
> - Triptych manifest: `analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/sampled_frames.csv` rows 0..799.
> - 5 L11 feature caches (extracted this session): `analysis/r13_overnight_atlas_2026-05-13/_features/intermediate__{P8A_REF, SLOT1..4}__layer11__n800.npz`.
> - IQ panel: `analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet`.
> - Atlas reference (A3): `analysis/cpu_diagnostics_2026-05-11_a3_atlas_composition/ATLAS_COMPOSITION_FACTS_2026-05-11.md`.
> - Ckpt map (paths to candidate ckpts): `arena/checkpoint_maps/teams_target_domain.r13_overnight_2026-05-13.yaml`.
>
> **Scripts**:
> - `analysis/r13_overnight_atlas_2026-05-13/run_atlas.py` — feature extraction + per-substrate inv_mean (mirrors A3's `_run_a3_atlas_composition.py`).
>
> **Device**: MPS (Apple Silicon) for feature extraction; CPU for sklearn 5-fold CV LR probes.

---

## 1. Question

For each of the 4 R13 overnight candidate ckpts (one representative per slot), recompute the A3 per-substrate L11 `inv_mean` on the standing 800-frame triptych and quantify Δ vs P8A_REFERENCE_STEP5000.

## 2. Method

Mirrors A3 exactly (see `ATLAS_COMPOSITION_FACTS_2026-05-11.md` §2).

- Panel: first 800 rows of `sampled_frames.csv` (the same triptych A3 used).
- Substrate columns: `split ∈ {dev, lockbox}` from the manifest; `is_chronic_6 = identity_key matches one of {Roy_D, PC_Generator, bla_bla_chow, Md_noyn_Sharker, dor_shkedi, healthy_dor}`; `is_dor = "dor" in identity_key.lower()`.
- `inv_mean = forgery_AUC − mean(shortcut_AUCs)` where shortcuts = {`is_dor`, `is_chronic_6`, `lap_var_high`, `min_dim_high`, `face_size_high`}.
- All 6 probes are 5-fold stratified-CV LR on L2-normalised L11 CLS features (`solver=lbfgs, C=1.0, n_jobs=1`). Folds reduce to `min(5, n_pos, n_neg)` when small; below 3 returns NaN.
- Shortcut binarisation thresholds (`lap_var_high`, `min_dim_high`, `face_size_high`) are pool-medians on the FULL 800-frame panel (matches A3); subset masks apply only to ROWS, not thresholds.
- Slices: `full_n800`, `dev_only`, `lockbox_only`, `chronic_6_only`, `non_chronic_only`.
- For `chronic_6_only` and `non_chronic_only`, `is_chronic_6` probe is degenerate within the slice and is reported as NaN; `mean_shortcut_AUC` averages over the remaining 4 axes (matches A3).
- Feature extraction: build `EffortDetector` with SVD architecture from the ckpt's saved `model_config`; for LoRA-trained ckpts (Slots 1, 2), wrap LoRA on `attn.{in_proj, out_proj}` + `mlp.{c_fc, c_proj}` at resblocks 10 and 11 (rank=16, alpha=32, per [`experiments/phase2_round13/R13_LORA_L10_L11_2026-05-13.yaml`](../../experiments/phase2_round13/R13_LORA_L10_L11_2026-05-13.yaml) lines 141-146) BEFORE state-dict load. CLIP normalisation, 224×224 bicubic-resize via cv2 INTER_LINEAR (matching the A3 extraction pattern).

## 3. Candidate ckpts

| Slot | label | ckpt | training packet | LoRA wrap |
|---|---|---|---|:--:|
| 1 | SLOT1_LORA_P8A_top_n_step2000 | `top_n_effort_20260513_step2000_auc0.9929_eer0.0229.pth` (`gf6l06rf`) | R13_LORA_L10_L11 (P8A_step5000 base, frozen SVD + LoRA on resblocks 10-11) | yes |
| 2 | SLOT2_LORA_T5C_periodic_step1500 | `periodic_effort_20260513_step1500_auc0.9941_eer0.0198.pth` (`912kd88q`) | R13_LORA_T5C_L10_L11 (T5C_step3500 base, same LoRA recipe) | yes |
| 3 | SLOT3_T5C_JITTER030_top_n_step4500 | `top_n_effort_20260513_step4500_auc0.9923_eer0.0304.pth` (`502dcznh`) | R13_SLOT3_T5C_PLUS_JITTER030 (T5C_step3500 base + face_scale_jitter scale_limit=0.30; multi-axis-L11-GRL hidden_dim=1024 inherited) | no |
| 4 | SLOT4_B16SC_FOURIER_top_n_step10000 | `top_n_effort_20260513_step10000_auc0.9902_eer0.0259.pth` (`qrpf5dtr`) | R13_SLOT4_B16_SCRATCH_FOURIER (B16-scratch + fourier_aug bands_randomize [8-13], bands_preserve [5, 6], phase preserved, p_apply=0.5) | no |

Anchor: `P8A_REFERENCE_STEP5000` (`value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth`) — included to verify baseline parity against A3's published numbers.

## 4. Feature-extraction smoke results

For each ckpt: build model, optionally wrap LoRA, load state dict, run a 1-frame forward smoke, then full-cohort L11 CLS extraction on the 800-frame triptych.

| Ckpt | smoke `pooler_output` shape | smoke finite? | n_valid frames | features shape | n non-finite |
|---|---|:--:|---:|---|---:|
| P8A_REF (cached, reused from prior A3 / iq_perlayer cache) | n/a | n/a | 800 | (800, 768) | 0 |
| SLOT1_LORA_P8A_top_n_step2000 | (1, 512) | yes | 800 | (800, 768) | 0 |
| SLOT2_LORA_T5C_periodic_step1500 | (1, 512) | yes | 800 | (800, 768) | 0 |
| SLOT3_T5C_JITTER030_top_n_step4500 | (1, 512) | yes | 800 | (800, 768) | 0 |
| SLOT4_B16SC_FOURIER_top_n_step10000 | (1, 512) | yes | 800 | (800, 768) | 0 |

LoRA wrap diagnostic for the two LoRA ckpts (Slots 1 + 2): 8 modules wrapped (resblocks {10, 11} × {attn.in_proj, attn.out_proj, mlp.c_fc, mlp.c_proj}); 393,216 LoRA parameters of 235,277,952 total backbone params (0.167% — matches the smoke result from `scripts/smoke_lora_wiring_2026-05-12.py`).

State-dict load `unexpected_keys` for ckpts that carry multi-axis-GRL bottleneck/head parameters (Slot 3 inherits T5C's MAGRL): 18 dropped (`multi_axis_grl_block.bottleneck.*`, `multi_axis_grl_block.heads.*`); these are head-side parameters, do not affect L11 CLS extraction.

## 5. Substrate composition of the 800-frame triptych (reproduction of A3 §3)

| Slice | n | n_real | n_fake | n_chronic_6 | n_dor | n_identities |
|---|---:|---:|---:|---:|---:|---:|
| split=dev | 713 | 436 | 277 | 237 | 73 | 28 |
| split=lockbox | 87 | 40 | 47 | 45 | 25 | 5 |
| is_chronic_6=1 | 282 | 241 | 41 | 282 | 65 | 6 |
| is_chronic_6=0 | 518 | 235 | 283 | 0 | 33 | 27 |
| all | 800 | 476 | 324 | 282 | 98 | 33 |

(Identical to A3 §3.1 — same panel construction; verifies the join logic reproduces.)

## 6. Per-substrate forgery_AUC, mean_shortcut_AUC, inv_mean

Source: `per_substrate_inv_mean.csv`.

### 6.1 forgery_AUC

| Slice | P8A_REF | SLOT1_LORA_P8A | SLOT2_LORA_T5C | SLOT3_T5C_JITTER030 | SLOT4_B16SC_FOURIER |
|---|---:|---:|---:|---:|---:|
| full_n800 | 0.9738 | 0.9760 | 0.9891 | 0.9853 | 0.9916 |
| dev_only | 0.9828 | 0.9840 | 0.9936 | 0.9936 | 0.9952 |
| lockbox_only (n=87) | 0.9489 | 0.9404 | 0.9787 | 0.9803 | 0.9989 |
| non_chronic_only | 0.9698 | 0.9756 | 0.9944 | 0.9941 | 0.9974 |
| chronic_6_only (n=282) | 0.9915 | 0.9950 | 0.9811 | 0.9695 | 0.9846 |

### 6.2 mean_shortcut_AUC

| Slice | P8A_REF | SLOT1_LORA_P8A | SLOT2_LORA_T5C | SLOT3_T5C_JITTER030 | SLOT4_B16SC_FOURIER |
|---|---:|---:|---:|---:|---:|
| full_n800 | 0.9425 | 0.9372 | 0.9572 | 0.9488 | 0.9655 |
| dev_only | 0.9318 | 0.9284 | 0.9576 | 0.9484 | 0.9655 |
| lockbox_only (n=87) | 0.9679 | 0.9477 | 0.9862 | 0.9929 | 0.9954 |
| non_chronic_only | 0.9508 | 0.9431 | 0.9613 | 0.9560 | 0.9697 |
| chronic_6_only (n=282) | 0.9470 | 0.9531 | 0.9617 | 0.9432 | 0.9655 |

### 6.3 inv_mean = forgery_AUC − mean_shortcut_AUC

| Slice | P8A_REF | SLOT1_LORA_P8A | SLOT2_LORA_T5C | SLOT3_T5C_JITTER030 | SLOT4_B16SC_FOURIER |
|---|---:|---:|---:|---:|---:|
| full_n800 | +0.0313 | +0.0388 | +0.0319 | +0.0365 | +0.0261 |
| dev_only | +0.0509 | +0.0556 | +0.0360 | +0.0453 | +0.0297 |
| lockbox_only (n=87) | −0.0189 | −0.0073 | −0.0074 | −0.0126 | +0.0035 |
| non_chronic_only | +0.0191 | +0.0325 | +0.0331 | +0.0381 | +0.0276 |
| chronic_6_only (n=282) | +0.0445 | +0.0419 | +0.0194 | +0.0263 | +0.0191 |

### 6.4 Δ inv_mean vs P8A_REFERENCE_STEP5000

Source: `delta_inv_mean_vs_p8a.csv`.

| Slice | SLOT1_LORA_P8A | SLOT2_LORA_T5C | SLOT3_T5C_JITTER030 | SLOT4_B16SC_FOURIER |
|---|---:|---:|---:|---:|
| full_n800 | +0.0075 | +0.0006 | +0.0052 | −0.0052 |
| dev_only | +0.0047 | −0.0150 | −0.0057 | −0.0212 |
| lockbox_only (n=87) | +0.0117 | +0.0115 | +0.0064 | +0.0224 |
| non_chronic_only | +0.0134 | +0.0140 | +0.0190 | +0.0086 |
| **chronic_6_only (n=282)** | **−0.0026** | **−0.0252** | **−0.0182** | **−0.0254** |

## 7. Reference comparison to prior atlas measurements (chronic_6_only slice)

Pulled from the cited sources. Same triptych, same method, same probe — only the ckpts differ.

| Ckpt (training packet) | chronic_6 forgery_AUC | chronic_6 inv_mean | Δ vs P8A_REF | Source |
|---|---:|---:|---:|---|
| P8A_REFERENCE_STEP5000 (anchor) | 0.9915 | +0.0445 | 0 (by definition) | A3 §4.1 / §4.2; reproduced this session §6.3 |
| T4_LAMBDA1_TOP_N_STEP10500 (T4-λ1 full FT, prior packet) | 0.9561 | +0.0130 | −0.0315 | A3 §4.2 / §4.3 |
| T5C_PERIODIC_STEP3500 (T5C classifier hidden_dim=1024, prior packet) | (per memory `project_t5c_chronic6_partial_recovery_2026-05-12`) | (per memory) | +0.0224 | memory entry |
| SLOT1_LORA_P8A_top_n_step2000 (LoRA r=16 α=32, frozen P8A base) | 0.9950 | +0.0419 | **−0.0026** | this report §6.3-6.4 |
| SLOT2_LORA_T5C_periodic_step1500 (LoRA r=16 α=32, frozen T5C_step3500 base) | 0.9811 | +0.0194 | **−0.0252** | this report §6.3-6.4 |
| SLOT3_T5C_JITTER030_top_n_step4500 (T5C + face_scale_jitter @0.30) | 0.9695 | +0.0263 | **−0.0182** | this report §6.3-6.4 |
| SLOT4_B16SC_FOURIER_top_n_step10000 (B16-scratch + Fourier bands 8-13) | 0.9846 | +0.0191 | **−0.0254** | this report §6.3-6.4 |

## 8. Output artifacts

- `per_substrate_inv_mean.csv` — full per-ckpt × per-slice probe results (5 ckpts × 5 slices, 25 rows; columns: forgery_auc, mean_shortcut_auc, inv_mean, + per-shortcut AUCs).
- `delta_inv_mean_vs_p8a.csv` — Δ inv_mean for the 4 candidates vs P8A_REF baseline (5 slices × 4 candidate columns + P8A_REF zero-column).
- `_features/intermediate__{P8A_REF, SLOT1..4}__layer11__n800.npz` — L11 CLS feature caches (numpy NPZ; `features` (800, 768) + `valid_idx` (800,) per ckpt).
- `run_atlas.py` — extract + probe pipeline. Reproducible from cached ckpts; `--phase {extract, probe, all}`.

## 9. Direct observations

1. **Baseline parity reproduced**: P8A_REF `chronic_6_only` inv_mean = +0.0445 in this run; A3 §4.2 reports +0.0445 (3-decimal match). Other slices: full_n800 +0.0313 (A3: +0.0313), dev_only +0.0509 (A3: +0.0509), lockbox_only −0.0189 (A3: −0.0189), non_chronic_only +0.0191 (A3: +0.0191). All 5 slices match A3 to 4 decimal places.
2. **All 4 candidates produce finite-valued L11 features on all 800 triptych frames**, including both LoRA-wrapped ckpts. Zero non-finite values across all 5 ckpt caches.
3. **Slot 1 (LoRA-P8A top_n_step2000) chronic_6_only inv_mean = +0.0419**; Δ = −0.0026 vs P8A_REF baseline (essentially flat within 5-fold CV noise). Slot 1 also has the highest chronic_6 forgery_AUC of the 5 ckpts (0.9950 vs P8A_REF 0.9915).
4. **Slot 2 (LoRA-T5C periodic_step1500) chronic_6_only inv_mean = +0.0194**; Δ = −0.0252. Lower forgery_AUC (0.9811) than P8A_REF (0.9915) on chronic_6 cohort. inv_mean is within ~0.003 of the T5C_step3500 base measurement (+0.0224 per memory).
5. **Slot 3 (T5C+jitter030 top_n_step4500) chronic_6_only inv_mean = +0.0263**; Δ = −0.0182. Lowest chronic_6 forgery_AUC of all 5 ckpts at 0.9695. Slot 3 inherits T5C_step3500 base; T5C_step3500's chronic_6 inv_mean was +0.0224 (memory ref); Slot 3 lands at +0.0263 (Δ vs its own base ≈ +0.004, vs P8A −0.0182).
6. **Slot 4 (B16-scratch+Fourier top_n_step10000) chronic_6_only inv_mean = +0.0191**; Δ = −0.0254. The slot is the only ckpt in this report trained from scratch (B16 ImageNet init, no P8A or T5C base loaded). chronic_6 inv_mean is positive (i.e., forgery_AUC > mean_shortcut_AUC) and non-zero.
7. **Lockbox-slice direction (sign of inv_mean) flips for Slot 4 only**: Slot 4 = +0.0035 (above zero); P8A_REF = −0.0189, Slot 1 = −0.0073, Slot 2 = −0.0074, Slot 3 = −0.0126 (all below zero). Slot 4's lockbox forgery_AUC (0.9989) is the highest of the 5 ckpts on that 87-frame slice.
8. **Non-chronic-slice direction**: all 4 candidates have higher inv_mean than P8A_REF in `non_chronic_only` (+0.0086 to +0.0190). On the same slice, all 4 also have higher forgery_AUC (≥ 0.9756, vs P8A 0.9698) AND lower mean_shortcut_AUC than P8A only for Slot 1 (0.9431 vs P8A 0.9508); the other 3 have higher mean_shortcut on non_chronic.
9. **dev_only Δ inv_mean direction is mixed**: Slot 1 +0.0047 (above P8A), Slot 2 −0.0150, Slot 3 −0.0057, Slot 4 −0.0212. The mixed direction on dev_only is driven by Slots 2-4 having higher dev_only mean_shortcut_AUC (0.9484 to 0.9655) than P8A (0.9318) while their forgery_AUC also increased (0.9936 to 0.9952 vs P8A 0.9828).
10. **chronic_6 slice rank-ordering (by inv_mean, descending)**: P8A_REF (+0.0445) > Slot 1 (+0.0419) > Slot 3 (+0.0263) > Slot 2 (+0.0194) > Slot 4 (+0.0191). All 4 candidates score below P8A_REF on this single slice; the same 4 candidates score above P8A_REF on `non_chronic_only` (rank order Slot 3 > Slot 2 > Slot 1 > Slot 4 > P8A).
11. **Slot 1 LoRA vs Slot 2 LoRA differ only in base**: same LoRA recipe (rank=16, alpha=32, layers [10, 11]); base is P8A_step5000 (Slot 1) vs T5C_step3500 (Slot 2). chronic_6 inv_mean Δ vs P8A_REF: Slot 1 −0.0026 vs Slot 2 −0.0252 (gap = 0.0226 absolute on the chronic_6 slice).
12. **Slot 4 (scratch) produces non-zero chronic_6 inv_mean = +0.0191** — same order of magnitude as the FT'd T5C-base candidates (Slot 2: +0.0194; Slot 3: +0.0263). Slot 4's full_n800 forgery_AUC is the highest of the 5 (0.9916, vs P8A 0.9738) but its mean_shortcut_AUC is also the highest (0.9655, vs P8A 0.9425), so its full_n800 inv_mean (+0.0261) is below P8A's (+0.0313).

## 10. Caveats

1. **5-fold CV variance not quantified** — single random_state=0 LR fit per slice (matches A3); repeat-fold variance not measured here. A3 §10 estimates ~±0.005 noise on AUC values per slice from this protocol.
2. **n=87 lockbox-only slice is small** — for `chronic_6` shortcut probe inside this slice, n_negatives is small (45 chronic-6 of 87) and the 5-fold CV becomes 3-fold or less on the shortcut probes. lockbox cells should be read with that variance in mind; full lockbox suite AUC (n=1843) is a more stable reference (see A3 §6).
3. **T5C_PERIODIC_STEP3500 chronic_6 inv_mean = +0.0224 (cited in §7)** — pulled from memory entry `project_t5c_chronic6_partial_recovery_2026-05-12.md`. T5C step3500 was not re-extracted this session; the comparison in §7 relies on the memory's published number being from the same triptych + same probe definition. T4_L1_step10500 chronic_6 inv_mean (+0.0130 → Δ −0.0315) is from A3 §4.2 / §4.3 directly.
4. **The `inv_mean` probe is feature-level, not head-level.** As A3 §10 (caveat 3) notes, "LR probes recover feature-level separability while trained-head AUC reflects whether the learned linear classifier actually exploits that separability at the deployed parameters." This report does not measure head-level behaviour; the promotion-contract scorecard (Vertex `1806724447428673536`) is the head-level read.
5. **MPS / CPU consistency**: feature extraction ran on MPS; probe fitting on CPU. Confirmed all features are finite. Float-precision differences between MPS extraction and a hypothetical CPU re-extraction are typically ≤ 1e-5 per dimension; not measured this session.
6. **No HDTF / cross-substrate probe in this report**. The 800-frame triptych is 89.1% dev / 10.9% lockbox (per §5). Cross-substrate generalisation (e.g., HDTF proper viso, may6 production frames) is out of scope; the present measurement is encoder-level invariance on the triptych's substrate.

---

## Summary (self-contained, 5 bullets)

- Bullet 1: Baseline parity with A3 reproduced to 4 decimals on all 5 P8A_REF slices (chronic_6 inv_mean = +0.0445 here vs +0.0445 in A3 §4.2). All 4 candidate ckpts loaded + extracted with finite outputs on all 800 triptych frames; LoRA-wrapped ckpts (Slots 1, 2) loaded with 8 wrapped modules and 393,216 LoRA params each.
- Bullet 2: chronic_6 inv_mean Δ vs P8A_REF: Slot 1 (LoRA-P8A) −0.0026; Slot 2 (LoRA-T5C) −0.0252; Slot 3 (T5C+jitter030) −0.0182; Slot 4 (B16-scratch+Fourier) −0.0254. Reference comparisons: T4_L1_step10500 (full FT from P8A) = −0.0315 per A3; T5C_step3500 (T4 + classifier 1024) = +0.0224 absolute per memory.
- Bullet 3: chronic_6 forgery_AUC ranking (descending): Slot 1 0.9950 > P8A_REF 0.9915 > Slot 4 0.9846 > Slot 2 0.9811 > Slot 3 0.9695. Slot 1 produces the only chronic_6 forgery_AUC above P8A_REF. On `non_chronic_only`, all 4 candidates have higher forgery_AUC than P8A_REF (Slot 4 highest at 0.9974).
- Bullet 4: lockbox-slice sign-of-inv_mean flips only for Slot 4 (+0.0035, above zero); P8A_REF and Slots 1-3 are all below zero on the 87-frame lockbox slice (range −0.0189 to −0.0073). Slot 4 also has the highest lockbox forgery_AUC (0.9989). dev_only Δ direction is mixed: only Slot 1 is above P8A.
- Bullet 5: Caveats: (a) 5-fold CV variance ~±0.005 not measured per-cell this session; (b) n=87 lockbox slice produces ≤3-fold CV for some shortcuts; (c) T5C_step3500 comparison value (+0.0224) is from memory, not re-extracted this session; (d) probe is feature-level — head-level scorecard verdict lives in the parallel promotion-contract scorecard (Vertex `1806724447428673536`).
