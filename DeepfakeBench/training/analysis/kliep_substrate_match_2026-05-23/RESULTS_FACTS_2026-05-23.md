# KLIEP substrate-matching check — RESULTS FACTS

Generated 2026-05-23. Factual readout only. Interpretive content in `AGENT_PROPOSAL_2026-05-23.md`.

> **Question**: How distinguishable are the various data distributions in frozen-CLIP feature space — specifically, is OPTB (the training-corpus sample) a good substrate-match for team-identity deploy distribution?
>
> **Method**: For each pair of pools, train a logistic-regression discriminator on frozen-CLIP L11 (768-d) features. 5-fold stratified CV with seed=42. Report accuracy + AUC + Kanamori-style effective-sample-size (ESS = (Σw)²/Σw² where w = p/(1−p) is the LR-derived density ratio).
>
> **Inputs**: OPTB cache (6,000 frames, 3000 real + 3000 fake) + team-identity cache (5,941 frames, 1,821 deploy-relevant real + 4,120 deploy-relevant fake). Both at `analysis/frozen_clip_team_identity_baseline_2026-05-23/outputs/`.

---

## 0. Interpretation key

| CV accuracy | Interpretation |
|---|---|
| ≤55% | near-identical distributions |
| 55-65% | well-matched |
| 65-75% | moderately-mismatched |
| 75-85% | well-mismatched |
| >85% | highly-mismatched |

ESS-on-source = effective fraction of source samples after importance weighting toward target distribution. Lower = more of the source pool is far from target.

---

## 1. Aggregate distribution comparisons

| Comparison | n_A | n_B | CV accuracy | CV AUC | ESS_on_A | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| **OPTB-real ↔ team-identity-real** | 3000 | 1821 | **0.999 ± 0.001** | **1.000** | **0.031** | **highly-mismatched** |
| **OPTB-fake ↔ team-identity-fake** | 3000 | 4120 | **1.000 ± 0.000** | **1.000** | **0.027** | **highly-mismatched** |
| OPTB-real ↔ OPTB-fake (sanity) | 3000 | 3000 | 0.867 | — | — | well-mismatched (label signal in CLIP) |
| team-identity-real ↔ team-identity-fake (sanity) | 1821 | 4120 | 0.997 | — | — | highly-mismatched (very strong label signal in CLIP for this cohort) |

Key observations:
- **OPTB and team-identity are essentially disjoint in frozen-CLIP space** (CV accuracy ≥0.999 on both real and fake sides; AUC 1.000)
- ESS_on_OPTB = 0.031 (real) / 0.027 (fake) — only ~3% of OPTB frames would survive a re-weighting toward team-identity distribution
- The OPTB-real ↔ OPTB-fake discrimination is *easier* than the OPTB ↔ team-identity discrimination only relative to itself (0.867 vs 0.999): the substrate-axis dominates the label-axis at the population level
- The team-identity-real ↔ team-identity-fake CV accuracy of 0.997 confirms there is a real-vs-fake signal in frozen-CLIP for this cohort

---

## 2. OPTB-real vs each team-human's real cohort

Per-human breakdown. All pairs are highly mismatched.

| Comparison | n_team_human | CV accuracy | CV AUC | ESS_on_OPTB |
|---|---:|---:|---:|---:|
| OPTB-real ↔ Noyn real | 210 | 1.000 ± 0.000 | 1.000 | 0.028 |
| OPTB-real ↔ Roee_Windows real | 330 | 1.000 ± 0.000 | 1.000 | 0.032 |
| OPTB-real ↔ Xiang real | 582 | 0.999 ± 0.001 | 1.000 | 0.031 |
| OPTB-real ↔ Xinhe real | 79 | 1.000 ± 0.000 | 1.000 | 0.010 |
| OPTB-real ↔ dor real | 620 | 0.999 ± 0.002 | 1.000 | 0.041 |

ESS_on_OPTB varies slightly per team-human (0.010 for Xinhe to 0.041 for dor) — implies dor real frames are very slightly less far from OPTB than Xinhe real frames are, in terms of which OPTB frames would re-weight to match.

---

## 3. Intra-team-identity: between-human reals

All pairs are highly mismatched at CV accuracy ≥0.999, AUC 1.000.

| Pair | CV accuracy | Interpretation |
|---|---:|---|
| Noyn ↔ Roee_Windows | 1.000 | highly-mismatched |
| Noyn ↔ Xiang | 1.000 | highly-mismatched |
| Noyn ↔ Xinhe | 1.000 | highly-mismatched |
| Noyn ↔ dor | 1.000 | highly-mismatched |
| Roee_Windows ↔ Xiang | 1.000 | highly-mismatched |
| Roee_Windows ↔ Xinhe | 1.000 | highly-mismatched |
| Roee_Windows ↔ dor | 0.999 | highly-mismatched |
| Xiang ↔ Xinhe | 1.000 | highly-mismatched |
| Xiang ↔ dor | 1.000 | highly-mismatched |
| Xinhe ↔ dor | 1.000 | highly-mismatched |

Each team-human's real cohort is a distinct sub-distribution in frozen-CLIP L11 space. None of the 10 pairwise comparisons drops below 0.999 CV accuracy.

---

## 4. Substrate-match policy thresholds

Based on these readouts, proposed CV-accuracy thresholds for any future dataset-acquisition decision (e.g., the 7000-webcam dataset in `TRAINING_DIRECTIONS_OPTIONS §3.III B.III.3`):

| CV accuracy of (new_data ↔ team-identity) | Recommended action |
|---|---|
| ≤ 65% | Well-matched. Train on it. |
| 65 - 80% | Moderately mismatched. Train on a KLIEP-weighted subset only; expect modest lift. |
| 80 - 95% | Highly mismatched. Use ONLY the KLIEP-selected subset (typically ESS-fraction ~5-15%); risk of net-negative on team-identity transfer. |
| > 95% | Disjoint distribution. Do not train on it without substrate-aware filtering at the frame level; treating as additional data is likely to hurt (see OPTB result). |

For reference: OPTB ↔ team-identity is 99.9% — in the "do not train" range. The frozen-CLIP baseline §1.5.6 measured concrete harm at this distance: best Option B head min recall 0.037 vs best Option A head 0.205 (Δ −0.168).

---

## 5. Extension: lockbox + dev comparisons (added 2026-05-23 PM follow-up)

Reconstructed D8's dev/lockbox split (replay_dev_lockbox_paths, seed=42) and extracted lockbox-real subset from the cached `clip_frozen_l11__n4839.npz` (positional: first 2000 = dev_real, next 2000 = dev_fake, then 839 lockbox in parquet row order).

| Comparison | n_A | n_B | CV accuracy | CV AUC | ESS_on_A | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| OPTB_real ↔ lockbox_real | 3000 | 414 | 1.0000 | 1.0000 | 0.038 | highly-mismatched |
| **team_id_real ↔ lockbox_real** | 1821 | 414 | **0.9808** ± 0.007 | 0.9916 | 0.012 | highly-mismatched (slightly lower) |
| dev_real ↔ lockbox_real | 2000 | 414 | 0.9996 | 1.0000 | 0.080 | highly-mismatched |
| **dev_real ↔ team_id_real** | 2000 | 1821 | **0.8739** ± 0.008 | 0.9651 | 0.133 | **highly-mismatched but LOWEST CV acc** |
| dev_real ↔ OPTB_real | 2000 | 3000 | 0.9988 | 1.0000 | 0.068 | highly-mismatched |

### 5.1 Substrate-distance ordering

Sorted by CV accuracy ascending (lower = closer pools):

1. dev_real ↔ team_id_real: 0.874 — closest pair among any tested
2. team_id_real ↔ lockbox_real: 0.981
3. OPTB_real ↔ lockbox_real: 1.000
4. dev_real ↔ lockbox_real: 1.000
5. dev_real ↔ OPTB_real: 0.999

The **dev pool is genuinely closer to team-identity than any other pool tested**. lockbox is closer to team-identity than to dev (despite both being non-OPTB), suggesting team-identity overlaps somewhat with both dev and lockbox capture-pipelines.

OPTB is maximally distant from every pool — confirms `frozen_clip_team_identity_baseline §1.5.6` finding that OPTB-trained heads have lockbox AUC 0.345-0.516 (near or below chance).

### 5.2 Implications for substrate-match policy thresholds

Updated context for the §4 thresholds:
- "Well-matched" (≤65%): nothing in the available pool set qualifies. dev↔team_id at 87% is the closest achievable today.
- The existing dev pool (used to train the contract heads) sits at the boundary between "moderately mismatched" (65-80%) and "well mismatched" (80-95%) — explains why training on dev produces deploy-relevant heads but with substantial transfer gap.
- Any new webcam acquisition should aim to BEAT dev's 87% — i.e., achieve <80% CV accuracy vs team-identity to materially help.

## 6. Artifacts

- `outputs/discriminator_results.csv` — 19 pairwise comparisons (4 aggregate + 5 OPTB-vs-team-human + 10 intra-team-human)
- `outputs/discriminator_results_with_lockbox.csv` — 5 follow-up comparisons including lockbox + dev
- `scripts/substrate_match.py` — driver

Wall: ~24 min main run + ~3 min lockbox follow-up.

---

## 6. Caveats

1. **D8 lockbox cache** (4839 frames) was not used in this readout — the cache lacks labels/paths and was not re-extracted. Adding "OPTB ↔ lockbox" and "team-identity ↔ lockbox" comparisons would round out the picture but is not required for the OPTB-vs-team-identity policy question.
2. **The 100% CV accuracy on intra-team-human comparisons is suspicious-good and warrants permutation testing.** Likely true (the same person captured under different camera/lighting/session conditions can be highly distinguishable in CLIP space), but the 1.000 number is at the ceiling.
3. **ESS estimate via LR-derived density ratio** is a coarse proxy for KLIEP — KLIEP itself uses iterative density-ratio fitting. The LR approach is sufficient for the "are these populations distinguishable" policy question but not for fine ESS calibration.
4. **OPTB is a 6,000-frame stratified sample of the 1.4M-frame training corpus**. A larger sample might be slightly closer to team-identity (more chance of in-distribution outliers), but the OPTB method-distribution dominates (mostly FF++/AVSpeech/deep-live-cam variants), and these methods are systematically different substrates from Teams-substrate captures.
5. **Frozen-CLIP L11 is one feature space**. The OPTB-vs-team-id mismatch could be slightly different at deeper transformer layers; not measured here.
