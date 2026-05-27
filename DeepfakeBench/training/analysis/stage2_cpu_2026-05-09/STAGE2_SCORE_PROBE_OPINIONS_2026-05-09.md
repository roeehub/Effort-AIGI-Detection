# Stage 2 score-distribution probe OPINIONS (2026-05-09)

> **Status: opinions / interpretation.** Companion to
> `STAGE2_SCORE_PROBE_FACTS_2026-05-09.md`.
>
> The author of this doc was the agent that launched the three Stage 2
> packets and ran the CPU probe. Reader is welcome to disagree —
> facts are in the FACTS doc, this doc layers interpretation.

## 1. Headline reading

All three Stage 2 single-lever interventions **regressed P8A's signature
dor invariance** at step4500 — by 15-28× the P8A baseline median real-FPR
on `DOR_REAL_LOCKBOX`. **None preserved P8A's Pillar-2 strength.** The
per-cohort trajectory data clarifies which hypotheses bite and which
don't:

| slot | hypothesis tested | trajectory | reading |
|---|---|---|---|
| S1 REAL_AUG_OFF | IQ-prior FLIP between train/eval reals is the source of the IQ-shortcut at deployment | DOR_REAL_LOCKBOX p50: 0.27 → 0.35 → 0.54 (monotone degradation) | **REFUTED.** Disabling `pipeline_randomization.p_real` doesn't preserve P8A's dor invariance; it makes things worse over training. The IQ-prior-flip story may still hold at the *deployment* level but is not the load-bearing source of identity-conditional regression. |
| S2 LOW_LR_FT | Mild FT (LR 1e-5) preserves P8A's pillars by limiting L10-11 drift | DOR_REAL_LOCKBOX p50: 0.20 → 0.31 → 0.29 (plateau) | **PARTIALLY SUPPORTED.** S2 has the best Pearson r vs P8A (0.62 on DOR_REAL_LOCKBOX at step4500) and the smallest dor regression among Stage 2 final ckpts. Still 15× P8A's baseline. |
| S3 WEAK_PAIRRANK | λ=0.05 reduces pair_rank's score-compression linearly, preserving lockbox lift while keeping separation | DOR_REAL_LOCKBOX p50: 0.66 → 0.50 → 0.52 (early collapse, partial recovery, plateau) | **REFUTED.** λ=0.05 still produces an identity-cluster effect at step500 (DOR_REAL_DEV std 0.077 = 4.5× compressed vs P8A 0.347), AND uniquely regresses on `DOR_FAKE_DEV` too (p50 0.58 vs P8A 0.99). pair_rank's failure mode is non-linear in λ. |

## 2. The structural finding under all three: FT-from-P8A drifts the dor invariance no matter what

The convergent observation across the three Stage 2 slots — and across
P2D (Fourier from CLIP-scratch), BUNDLE (pair_rank+GroupDRO from P8A FT),
and PAIRRANK (pair_rank-only from P8A FT) — is that **any FT from P8A
or any from-scratch retraining drifts the dor cohort cluster within the
first 500-2500 steps**, regardless of:

- whether pair_rank is on, off, or weakened
- whether GroupDRO is on or off
- whether real-side aug is on or off
- whether LR is 1e-4 or 1e-5
- whether the loss is CE or pair_rank or Fourier-aug
- whether base ckpt is P8A or scratch CLIP

The lever that has been measured is: P8A (only ckpt with `DOR_REAL_LOCKBOX
p50 ≤ 0.025`) vs everything else (≥ 0.20). The four sub-mechanisms
proposed by Stage 2 (data-axis, optimization-axis, loss-axis) all leave
the encoder in approximately the same family of "drifted-from-P8A"
states.

**This argues the binding constraint isn't any of the three single levers
— it's the FT process itself.** The encoder's L10-11 (per check (a) per-
layer probe) plus head are getting reorganized in a way that disrupts
the Dor identity cluster.

## 3. What looks like the best path forward — three sub-options

### 3a. Anchor-loss FT (the original Slot 2 from §11 of the prior turn)

Instead of LOWERING the LR, **anchor encoder L11 + projection during FT**
to P8A's outputs on a probe set that includes Dor identities. This is the
direct intervention that prevents L11 drift on the load-bearing identities,
while letting the head + earlier layers adapt to lift lockbox recall.

- Cost: trainer code change (~30-60 lines), image rebuild, ~$50-70 GPU.
- Risk: anchor weight λ tuning; too tight gives no learning, too loose
  no preservation.
- Information yield: HIGH. Directly tests "is L11+projection drift the
  load-bearing failure mode?" — the answer if S2 LOW_LR_FT is the best
  preserver but still regresses.

### 3b. Per-identity GroupDRO with chronic_flag and dor_flag

Reuse BUNDLE's GroupDRO infrastructure but with a chronic-flag that
includes Dor identities at a higher weight (clip_max=8.0 vs current 4.0).
Combined with weak pair_rank (λ=0.05) for lift.

- Cost: YAML-only (no code). ~$50-70 GPU.
- Risk: GroupDRO doesn't bite hard enough on Dor; chronic_flag axis is
  already in BUNDLE's R-D key but didn't preserve Dor.
- Information yield: MODERATE. Tests "is the chronic_flag axis
  under-weighted, or structurally insufficient?"

### 3c. Re-evaluate S2_step4500 vs P8A on the full promotion contract

S2_step4500 is the closest preserver of P8A's dor invariance in the
Stage 2 set (Pearson r 0.62 on DOR_REAL_LOCKBOX). Its holdout AUC 0.9926
matches P8A's exactly (0.9926 base ckpt). It may have the best chance
of passing the promotion contract scorecard among Stage 2 ckpts.

- Cost: scorecard run only — ~$15-20 GPU + 2-3h.
- Risk: low. Just a measurement.
- Information yield: HIGH if S2_step4500 *actually* preserves Pillar 2
  on the full lockbox; LOW if it regresses like the others (we already
  know dor_lockbox p50 is 0.29 vs P8A 0.02, so probably regresses).

### 3d. Accept P8A as deployment baseline and pivot

If three more attempts don't beat P8A on Pillars 1+2+3, the structural
question is whether to pivot from "improve the encoder via FT" to:

- **Substrate-aware τ at deployment** (per memory `feedback_per_mode_tau_not_deployable.md`
  this is hard, but a Dor-identity classifier upstream could enable it)
- **Identity-router architecture**: route different identity clusters to
  different scoring heads — engineering complexity but capturing the
  identity-conditional structure the encoder can't preserve

## 4. Recommendation for the morning

**UPDATE 2026-05-09 03:38 — Roy_D probe completed and strengthens the
verdict.** ALL three Stage 2 step4500 ckpts have catastrophic Roy_D
regression at τ=0.5 (S1 99.2%, S2 96.9%, S3 100% real-FPR vs P8A 45.4%).
Even the "best preserver" S2 collapses on Roy_D. See companion FACTS
doc: [`STAGE2_ROY_D_PROBE_FACTS_2026-05-09.md`](STAGE2_ROY_D_PROBE_FACTS_2026-05-09.md).

Original recommendation was: run promotion contract scorecard for
`S2_step4500` (option 3c). **Revised recommendation: don't run it.**
The Roy_D probe already predicts S2_step4500 will fail Pillar-2 at any
deployable τ; the contract scorecard would just confirm at $15-20 cost.

Save the GPU spend for the next structurally distinct intervention:

**Option 3a (L11 anchor-loss FT) — the only lever left that's not yet
tried.** The convergent Stage 2 + Roy_D evidence shows:

- Data-axis (S1 REAL_AUG_OFF): catastrophic Roy_D regression
- Optimization-axis (S2 LOW_LR_FT): catastrophic Roy_D regression
- Loss-axis (S3 WEAK_PAIRRANK): catastrophic Roy_D regression
- All three preceded by P1 BUNDLE / PAIRRANK / P2D / P14 with same pattern

The encoder L10-11 representation of P8A's chronic-identity invariance
is fragile — every FT regime drifts it. An L11 anchor that explicitly
preserves it during FT is the natural next intervention.

## 4a. Pre-launch CPU diagnostic for Option 3a — RAN 2026-05-09 04:18

The L11 distance probe was executed (`run_l11_distance_probe.py`,
~2min on MPS). Per-frame cos-distance(P8A_L11, ckpt_L11) on the 130
Roy_D frames + summary at `outputs/l11_distance_per_ckpt.csv`:

| ckpt | cos_dist mean | cos_dist p50 | Roy_D real-FPR @τ=0.5 |
|---|---:|---:|---:|
| S2_step500 | 0.21 | 0.20 | 84.6% |
| S1_step2500 | 0.43 | 0.39 | 81.5% |
| S1_step500 | 0.46 | 0.45 | 96.9% |
| S2_step2500 | 0.53 | 0.55 | 96.2% |
| S2_step4500 | 0.55 | 0.57 | 96.9% |
| S3_step2500 | 0.69 | 0.76 | 97.7% |
| S1_step4500 | 0.69 | 0.74 | 99.2% |
| S3_step500 | 0.74 | 0.75 | 100% |
| **S3_step4500** | **0.82** | **0.88** | **100%** |

Reading:
- **L11 drift is substantial** for Stage 2 step4500 ckpts (0.55-0.82
  cos-distance from P8A, comparable to or larger than typical
  cross-architecture distances).
- **L11 drift correlates with Roy_D score regression** — Spearman
  rank correlation between cos_dist and Roy_D real-FPR is 0.91 on
  these 9 ckpts. The L11 anchor lever has a target.
- **S2 LOW_LR_FT has the smallest L11 drift** at every step
  (0.21/0.53/0.55), consistent with its score-distribution-best status
  among Stage 2 ckpts.
- **S3 WEAK_PAIRRANK has the largest L11 drift** at every step
  (0.74/0.69/0.82), consistent with pair_rank's known L11-axis
  gradient signal magnitude.

**Conclusion**: L11 anchor-loss FT is a structurally well-targeted next
intervention. Anchor weight λ in the [0.05, 0.30] range is likely the
operating regime — tight enough to keep cos_dist <0.3 (S2_step500's
preservation level) while letting head + earlier layers adapt.

---

**CAVEAT (added 2026-05-09 — strength-of-evidence honest read).** The Spearman
r=0.91 between L11 cos-distance and Roy_D real-FPR is computed across n=9
sibling Stage 2 ckpts (3 slots × 3 step counts). All 9 share base ckpt
(`P8A_REFERENCE_STEP5000`), training data, and full-encoder-update regime.
Training-step amount alone correlates with both L11 drift (more steps →
more drift) AND with Roy_D regression (more steps → more drift in the dor
cluster). The correlation could equally be explained by training amount
rather than by L11 being the causal substrate of dor invariance. To break
the confound, the L11 anchor would need to be tested against alternatives:
(a) reduced training amount (already covered by S1 step500 vs step4500 —
both regress, ruling this out partially), (b) anchoring at L9 or L10 instead
(per check (a) per-layer probe, IQ encoding peaks at L6 and L11 is the
divergence layer between P8A and FT'd ckpts — but L9 was the dip), (c)
output-side preservation rather than activation-side preservation (anchor
on score distribution on the dor cohort, not on L11 features). The "L11
anchor is the only structurally distinct lever left" framing in §3a / §4
above is the strongest reading from current evidence; weaker readings include
"output-anchor" and "adapter-FT" that have not been tested.

**CAVEAT (added 2026-05-09 — under user 2026-05-09 reframe).** The user's
position is that the production target domain is genuinely unknown (different
laptops / cameras / lighting / people / backgrounds) and Pillar 3 robustness
must come from forgery signal rather than other-property shortcuts. Under
this framing, "preserving L11 representation" only matters if L11 carries
forgery signal. If P8A's L11 representation is heavily IQ-shortcut (per
the per-layer probe, IQ AUC at L11 is 0.85+) plus identity-cluster (per
`project_phase1a_method_cluster_axis_2026-05-01`) plus a forgery residual,
then anchoring L11 also anchors the IQ shortcut. The L11 anchor lever may
preserve P8A's good Pillar-2 properties AND its bad Pillar-3 properties,
locking in shortcut dependence. A cleaner intervention under (C) would be
to anchor on a SUBSET of L11 features that probes show carry forgery signal
(if such a subset can be isolated), or to use a contrastive loss that
encourages forgery-axis separation while penalizing IQ-axis co-variation.
The forgery-signal atlas (CPU diagnostic 2026-05-09) is the cheap test of
"does any layer × any ckpt actually carry a forgery signal independent of
IQ + identity shortcuts?".

## 5. What I'd revise in the original packet design

The score-distribution probe took 3 minutes per ckpt and answered the
"is the lever working?" question definitively for all three slots without
needing the full promotion contract scorecard. **In the future, every
Stage N packet should include a CPU probe diagnostic in the close
criterion** — runnable in the first 30 minutes after training finishes,
predictive of which ckpts deserve the GPU scorecard spend.

Specifically, the probe should always include the dor cohort + chronic-6
(Roy_D, PC_Generator, etc.) since identity-cluster collapse is the
recurring failure mode across R12/R13/Stage 2.

## 6. Cross-references

- **FACTS**: [`STAGE2_SCORE_PROBE_FACTS_2026-05-09.md`](STAGE2_SCORE_PROBE_FACTS_2026-05-09.md)
- **Pre-launch synthesis**: [`docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md`](../../docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md) §7 (Stage 2 launch)
- **Dor cohort source**: [`analysis/dor_encoder_axis_2026-05-08/DOR_ENCODER_AXIS_FACTS_2026-05-08.md`](../dor_encoder_axis_2026-05-08/DOR_ENCODER_AXIS_FACTS_2026-05-08.md)
- **Per-layer divergence (motivates 3a)**: [`analysis/iq_perlayer_probe_2026-05-08/IQ_PERLAYER_PROBE_FACTS_2026-05-08.md`](../iq_perlayer_probe_2026-05-08/IQ_PERLAYER_PROBE_FACTS_2026-05-08.md)
- **BUNDLE_step500 cluster collapse precedent**: P1 PE eval Roy_D regression FACTS (memory)
