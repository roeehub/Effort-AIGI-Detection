# CPU Diagnostic Synthesis — 2026-05-10 Phase 1

> **Status**: FACTS document. Ran 2026-05-10 to inform next training packet.
> Driven by user reframe: "no shortcut, robust model, picks up forgery signal,
> not confused by other conditions, strong fake recall across the board."
>
> **Companion docs**:
> - `analysis/cpu_diagnostics_2026-05-10/outputs/job_*.csv` (CSVs per job)
> - `analysis/cpu_diagnostics_2026-05-10/outputs/forgery_signal_atlas_extended.csv` (atlas)
> - Scripts at `analysis/cpu_diagnostics_2026-05-10/scripts/`

---

## Executive summary

**6 CPU diagnostic jobs ran on 2026-05-10 to characterize step1500/step2500 vs P8A/E2B and inform the next training packet.** Key findings:

1. **The encoder is the bottleneck**, not the head. At L11, ALL ckpts have BOTH forgery and shortcut signals extractable at >0.94 AUC. The HEAD chooses what to use, but the encoder co-mingles forgery with shortcuts. NONE of the 5 ckpts measured (P8A, E2B, T3 step1500, T3 step2500) achieve `inv_max = forgery_AUC - max_shortcut_AUC > 0` at any layer. P8A is closest (-0.019).

2. **T3 produced encoder-level change (not head-only).** P8A→step1500 cosine at L11 dropped from 1.0 to 0.55 — substantial rotation. step1500→step2500 stayed at 0.89 (consolidating, not diverging further). T3 isn't a head-recalibration; it shifted L11 representation. But the shift was AWAY from P8A (cos 0.55) and ALSO AWAY from E2B (negative cos -0.18) — T3 found a NEW direction.

3. **viso is harder than deeplive structurally.** Viso fakes have lap_var p50 ~30-75; deeplive fakes have lap_var p50 ~240. Viso is intrinsically blurry at the data level. The model's "high-Lap=real" tendency hurts viso recall regardless of which ckpt. Direct mitigation: train the model to attend to local forgery cues (swap-region) instead of global IQ.

4. **P8A's fragility is mostly gate-shielded.** 9/11 unique catastrophic FPs are on low-resolution sessions (Q__s6 min_dim=95, Md_noyn_Sharker__s15 min_dim=165 with 5% medium-gate retention). Production IQ gate would abstain on these. Only 2/11 (Test_Cam__s73) pass medium gate.

5. **step2500's 60 unique FPs partially pass the gate.** PC_Generator__s14 (15 FPs at min_dim=193, 32% gate retention) + Test_Cam__s41 (3 FPs, 53% retention) + Test_Cam__s76 (3 FPs, 41% retention) — ~21 FPs pass medium gate. step2500 has REAL post-gate fragility.

6. **Face-scale jitter (P14 winner MCLIOEXB) does/does-not weaken face_size shortcut at L11** — see §6 below for the empirical answer once atlas extension lands.

---

## 1. Forgery vs shortcut at L11 — the encoder bottleneck

L11 LR-probe AUCs on the 800-frame triptych (extended forgery signal atlas):

| ckpt | forgery (real_vs_fake) | is_dor | is_chronic_6 | lap_var_high | min_dim_high | face_size_high |
|---|---:|---:|---:|---:|---:|---:|
| **E2B** | **0.995** | 0.999 | 0.975 | 0.956 | 0.974 | 0.972 |
| **T3_S1_step1500** | 0.992 | 0.997 | 0.945 | 0.951 | 0.977 | 0.968 |
| **T3_S1_step2500** | 0.991 | 0.999 | 0.956 | 0.946 | 0.976 | 0.973 |
| **P8A** | **0.976** | 0.995 | 0.909 | 0.923 | 0.971 | 0.949 |

**Reading**:
- All ckpts: forgery and shortcut signals BOTH extractable at >0.92 AUC from L11 features.
- No ckpt achieves `forgery > max_shortcut` (inv_max > 0).
- P8A is uniformly LOWEST on both forgery AND shortcut (less expressive L11). Inv_mean at L11: P8A 0.027 (best of 4), T3 step1500 0.024, T3 step2500 0.021, E2B 0.020 (worst).
- **The chronic-6 leakage at L11 is the cleanest invariance proxy**: P8A 0.909 < T3 step1500 0.945 < T3 step2500 0.956 < E2B 0.975. P8A's celebrated "substrate-invariance" IS measurable but small.

**Layer trajectory** (forgery AUC by layer):

| layer | E2B | P8A | T3_step1500 | T3_step2500 |
|---:|---:|---:|---:|---:|
| 0 | 0.724 | 0.725 | 0.724 | 0.725 |
| 3 | 0.937 | 0.922 | 0.922 | 0.924 |
| 6 | 0.992 | 0.992 | 0.991 | 0.992 |
| 9 | 0.984 | 0.979 | 0.987 | 0.988 |
| 11 | 0.995 | 0.976 | 0.992 | 0.991 |

**Reading**: Forgery saturates at L6 for all ckpts. From L6 onward, forgery AUC barely changes; what changes is HOW MUCH SHORTCUT the encoder also encodes. Adding depth doesn't add forgery; it adds expressivity for both forgery AND shortcut.

Source: `outputs/forgery_signal_atlas_extended.csv`, `outputs/job_h_invariance_trajectory.csv`.

---

## 2. Encoder-vs-head: where ckpts diverge

Pairwise L11 cosine similarity on the 800-frame triptych (median):

| Pair | L0 | L3 | L6 | L9 | **L11** |
|---|---:|---:|---:|---:|---:|
| P8A vs E2B | 0.998 | 0.992 | 0.949 | 0.920 | **0.168** |
| P8A vs T3_step1500 | 1.000 | 0.998 | 0.988 | 0.982 | **0.552** |
| P8A vs T3_step2500 | 1.000 | 0.997 | 0.986 | 0.982 | **0.487** |
| E2B vs T3_step1500 | 0.999 | 0.994 | 0.959 | 0.932 | **−0.178** |
| E2B vs T3_step2500 | 0.999 | 0.993 | 0.958 | 0.936 | **−0.174** |
| T3_step1500 vs T3_step2500 | 1.000 | 0.999 | 0.994 | 0.992 | **0.893** |

**Reading**:
- **All ckpts agree through L6** (cos > 0.94). The "shared representation" extends through layer 6.
- **Divergence accelerates L9→L11**: cosine drops sharply at L11 across all ckpts. L11 is the head region — different training regimes produce orthogonal L11 directions.
- **T3 stayed closer to P8A (0.55) than to E2B (−0.18)** at L11. T3 inherited P8A's L11 subspace and rotated within it; it did NOT converge with E2B.
- **step1500 and step2500 are ~89% similar** at L11 — they're consolidations of the same direction, not different representations.

**Implication**: the L11 representation has multiple "valid" subspaces. P8A, E2B, and T3 each found different L11 subspaces with similar forgery_AUC but different shortcut_AUCs. The encoder is the load-bearing component for invariance; choosing the L11 direction matters.

Source: `outputs/job_h_layer_cosine_p8a_e2b.csv` + cosine matrix above.

---

## 3. Catastrophic-FP profile — gate-shielding analysis

(Job C, `outputs/job_c2_session_iq_profiles.csv`)

### P8A — 11 unique catastrophic-FP frames

| identity__session | n_FPs | chronic | atlas n | lap_var p50 | min_dim p50 | medium_gate retention |
|---|---:|:---:|---:|---:|---:|---:|
| Q__s6 | 7 | N | 9 | 645 | **95** | **0%** (gate-shielded) |
| Md_noyn_Sharker__s15 | 2 | Y | 240 | 461 | 165 | 5% (gate-shielded) |
| Test_Cam__s73 | 2 | Y | 98 | 189 | 294 | 95% (NOT shielded) |

**9 of 11 P8A unique FPs are gate-shielded** (low-resolution sessions). Only the 2 Test_Cam__s73 frames pass medium gate. P8A's deployment risk on broad cohort = ~2 catastrophic FPs out of 4564 reals = 0.04% post-gate.

### step2500 — 60 unique catastrophic-FP frames (partial atlas join)

| identity__session | n_FPs | chronic | atlas n | lap_var p50 | min_dim p50 | medium_gate retention |
|---|---:|:---:|---:|---:|---:|---:|
| PC_Generator__s14 | 15 | Y | 53 | 216 | **193** | **32%** |
| bla_bla_chow__s2 | 10 | Y | 100 | 55 | 144 | 0% (shielded) |
| Test_Cam__s41 | 3 | Y | 228 | 213 | 212 | **53%** |
| Test_Cam__s76 | 3 | Y | 94 | 94 | 398 | **41%** |
| Test_Cam__s73 | 2 | Y | 98 | 189 | 294 | 95% |
| Roy_D + dor + others | 27 | mixed | various | mixed | mixed | mostly 0% but some 100% |

**~21 step2500 unique FPs would PASS the medium gate** (rough estimate from joined atlas data — actual may be higher counting non-joined frames). step2500 has real post-gate fragility on chronic-6 sessions.

### shared_all_4 (where every ckpt > 0.9)

| identity__session | n_FPs | chronic | retention |
|---|---:|:---:|---:|
| bla_bla_chow__s2 | 9 | Y | 0% |
| Roy_D + others | 8 | mixed | mostly 0% |

Universal failure modes are mostly gate-shielded.

**Headline**: P8A's "broad cohort fragility" (Job B) shrinks to ~2 frames after IQ gate. step2500's shrinks to ~21+ frames after IQ gate. step1500 has 0 unique catastrophic FPs at any threshold.

---

## 4. Viso-vs-deeplive — viso is structurally blurrier

(Job E, `outputs/job_e_viso_vs_deeplive_summary.csv`)

Per-ckpt recall at τ=0.5 + IQ profile of caught vs missed fake frames:

| ckpt | viso recall@0.5 | deeplive recall@0.5 | viso missed lap_var p50 | deeplive missed lap_var p50 |
|---|---:|---:|---:|---:|
| P8A | 35.6% | 53.0% | **28** (very blurry) | 246 |
| E2B | 8.4% | 94.5% | 40 | 237 |
| T3_step1500 | 25.6% | 69.5% | 32 | 240 |
| T3_step2500 | 65.3% | 96.5% | 26 | 101 |

**Structural finding**: viso fakes have lap_var p50 ~30-75 across all ckpts; deeplive fakes have lap_var p50 ~240. Viso is **3-10× blurrier** than deeplive at the data level.

Reading: Viso is intrinsically a low-IQ method. The model's IQ-shortcut tendency (low Lap correlated with real-or-not depending on training) hurts viso recall. Direct mitigation = force attention to LOCAL forgery cues independent of global blurriness.

The fact that step2500 lifted viso recall to 65% (vs P8A 36%) at τ=0.5 means **the data-axis lever (drop high-Lap teams reals) DID help break the IQ-shortcut on viso**. But step2500 paid for it with shortcut-leakage elsewhere (chronic-6 over-fire).

---

## 5. step2500 representation — what it cost to lift viso

(Synthesizing Jobs A + H)

step2500 vs P8A trade:
- **Gained**: forgery_AUC at L11 +0.015 (0.976 → 0.991), F4 viso recall +12pp, F4 deeplive saturation
- **Cost**: chronic-6 AUC at L11 +0.047 (0.909 → 0.956), is_dor AUC +0.004 (0.995 → 0.999), per-axis FPR spread +8pp
- **Cosine with P8A**: dropped to 0.487 at L11 (representation rotation)

step1500 vs P8A trade:
- **Gained**: forgery_AUC at L11 +0.016 (0.976 → 0.992), F4 viso recall +6pp
- **Cost**: chronic-6 AUC +0.036 (0.909 → 0.945), is_dor AUC +0.002, per-axis spread +4pp
- **Cosine with P8A**: 0.552 at L11

**Reading**: T3 traded shortcut-leakage for forgery-saturation. The trade was net-positive at step1500 (modest gains, modest costs); at step2500 the gains and costs both grew.

---

## 6. Does face_scale_jitter weaken the face-size shortcut at L11?

[PENDING — atlas extension running with MCLIOEXB (P14 jitter winner). Will populate after extension lands.]

If MCLIOEXB face_size_high AUC at L11 < P8A's 0.949 → jitter weakens face-size shortcut at encoder → T4 face_scale_jitter directly addresses the right shortcut.

If MCLIOEXB face_size_high AUC ≈ P8A's → jitter operates at the head, not the encoder → T4 lever is at a different mechanism than the structural one.

---

## 7. The user's "from-scratch / curriculum" hypothesis

User reframe: "I'm still as always suspicious with the idea of fine-tuning from P8A And I do believe that once we find the right recipe, we should be able to train either from scratch or from scratch via curriculum learning - And even better model."

**Quantitative support from this session's data**:

1. **The encoder-level invariance bottleneck is shared across the FT chain.** P8A → S1/S2/S3 → T3 all have similar inv_mean at L11 (0.020-0.027). FT can't escape this because P8A's encoder ALREADY co-mingles forgery + shortcut, and FT inherits.

2. **E2B (from-scratch+CE) has the WORST L11 invariance** (inv_mean 0.020) — naive from-scratch+CE doesn't solve the problem either; in fact CE training amplifies shortcut-extractability.

3. **Implication**: The right recipe needs an EXPLICIT invariance objective, not just CE on real/fake. Pure-from-scratch+CE isn't sufficient (E2B proves that). Pure-FT-from-P8A inherits the shortcut leakage. Both have failed the goal.

4. **Concrete target metric**: inv_mean = forgery_AUC − mean(shortcut_AUCs) at L11 > 0.10. Currently best is 0.027 (P8A). Need a ~4× lift on this metric.

---

## 8. What the data tells us about the next training packet

Three structurally distinct paths, in increasing radicalism:

### Path α — Preserve P8A encoder, train new L11 + head only with multi-axis adversarial GRL

**Thesis**: P8A's encoder through L9 is invariance-friendly (we just don't have the metric for that yet — but cos > 0.92 with E2B through L9 supports it). Freeze L0-L9; train L10-L11 + head with: (a) standard CE loss, (b) reverse-gradient classifiers for {is_dor, is_chronic, lap_var quartile, min_dim quartile, color_a quartile} at L11.

**Cost**: ~$30-50 GPU. Smaller than full FT because L0-L9 frozen.

**Pre-test (CPU)**: Use existing P8A L9 features → train a small head-only classifier with multi-axis GRL → measure inv_mean on held-out frames. If inv_mean > 0.10 achievable, GPU launch is justified.

**Pillar 3 mechanism**: forces L11 to be uninformative about shortcut axes by construction.

### Path β — From-scratch B16 + multi-axis adversarial GRL at L11, no FT base

**Thesis**: Start from CLIP-B16 pretrained (NOT from P8A), train end-to-end with: (a) CE loss on real/fake, (b) multi-axis GRL at L11. The GRL directly punishes shortcut-extractability throughout training.

**Cost**: ~$70-100 GPU (full training run).

**Pre-test (CPU)**: Same as Path α — head-only multi-axis GRL on CLIP-B16 frozen features at L11. If achievable invariance is competitive, scratch+GRL is justified.

### Path γ — Curriculum from-scratch

**Thesis**: Stage 1 train on IQ-balanced data (force the model to learn across all IQ-axis quartiles, not the natural-distribution skew). Stage 2 train on forgery-localization tasks (force attention to swap region with auxiliary loss). Stage 3 fine-tune on Teams substrate.

**Cost**: ~$150-200 GPU (3 stages).

**Pre-test (CPU)**: Construct IQ-balanced subset; verify the IQ shortcut R² drops on this subset (vs full data). Then prototype Stage 1 training run on the balanced subset and measure L11 invariance after a few epochs.

**Risk**: Curriculum design is hard. Wrong stage criteria → no improvement.

### Path δ — Forgery-localization auxiliary loss

**Thesis**: For training data with known swap regions (DF40 pairs, HDTF, v2), add an auxiliary loss that forces attention masks (Grad-CAM or attention-rollout) to align with swap regions. Forces local forgery attention.

**Cost**: ~$70-100 GPU + infra investment (mask generation pipeline).

**Pre-test (CPU)**: Compute Grad-CAM masks for P8A on a sample of v2 fakes. Measure mask-vs-swap-region IoU. If IoU is already high, the mechanism is in place; if low, this lever has substantial room.

---

## 9. What I'd recommend doing FIRST

CPU-cheap before any GPU spend:

1. **Validate the multi-axis-GRL thesis** (Path α/β pre-test): use existing P8A L9 features, train head-only classifier + multi-axis GRL, measure achievable inv_mean. ~3 hours CPU. Decisive for Path α/β feasibility.

2. **Score MCLIOEXB layer atlas** (running now): does face_scale_jitter weaken the face_size shortcut at the encoder level? Decisive for whether T4 face_scale_jitter is a structural or head-level lever.

3. **Job G (counterfactual face-scale at inference)** — apply face_scale_jitter perturbation at inference and measure score destabilization. Tests whether ckpts are face-scale-stable currently. ~2 hours.

4. **Skip Job D** (step3500/4500 trajectory) — the step1500→step2500 progression already shows shortcut-leakage growing with training; step3500/4500 likely continue this trend.

After Phase 1: pick Path α/β/γ/δ based on probe outcomes, formalize thesis, run.

---

## Source files

- Driver scripts: `analysis/cpu_diagnostics_2026-05-10/scripts/` (job_c, job_e, job_h, extract_t3_features, extend_forgery_atlas, extend_atlas_with_extras)
- Atlas: `outputs/forgery_signal_atlas_extended.csv` (14 ckpts × 5 layers × 6 signals = 420 rows)
- Per-ckpt summaries: `outputs/job_*_*.csv`
- This synthesis: `analysis/cpu_diagnostics_2026-05-10/SYNTHESIS_FACTS_2026-05-10.md`
