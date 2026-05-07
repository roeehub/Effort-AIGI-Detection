# Thread: Pair-rank loss collateral on non-paired-lane identities

> **Template contract**: this thread follows `thread_template.md`. Status of the load-bearing claim is **hypothesis-tentative as of 2026-05-07** — there is one strong piece of indirect evidence (counter-theory probe) and one mechanism story that fits, but no causal proof. A reviewer should weight the "Current stance" with skepticism; alternate explanations are listed in §"What changed our mind".

## The question

Does the `pair_rank_loss` lever (per-pair invariance: push fake_score above paired_real_score within a pair) cause collateral damage on real frames belonging to identities NOT in any paired training lane? Specifically: when pair_rank is added on top of a P8A FT base, does the model's representation generalize the score-up direction from the paired training lanes to identities the model has previously handled correctly?

This thread exists because of the Roy_D regression observed in P1: a chronic-list identity that P8A handled at 29% FPR was inflated to 78-93% FPR by every P1 ckpt. The regression has a specific feature-axis signature (`color_b_dev` Δr=+0.71 mirror of P8A's r=−0.71) AND is shared between the BUNDLE and PAIRRANK arms (Wilcoxon p=0.875). The shared-mechanism finding refutes a simpler "GroupDRO balloon effect" hypothesis the agent initially proposed — but a positive mechanism claim has not been proven.

## Initial belief

The agent who ran P1's evaluation initially proposed that GroupDRO with `chronic_flag` had a "balloon effect": it suppressed the targeted chronic group (PC_Generator FPR 62.9% → 0%) at the cost of inflating non-target identities (Roy_D 29% → 93%). This framing was published in the agent's 2026-05-07 evening message: *"GroupDRO has a balloon effect — it inflates non-target identities like Roy_D as a side-effect of suppressing PC_Generator"*. Source: `analysis/p1_pe_eval_2026-05-07/AGENT_PROPOSAL_2026-05-07.md` §4 ("The error I made earlier — the GroupDRO balloon hypothesis").

## What changed our mind

- **2026-05-07 (counter-theory probe, refutation)**: A direct comparison of per-identity Δ FPR between BUNDLE arm (with GroupDRO+chronic_flag) and PAIRRANK arm (without) on 9 non-target identities showed mean |Δ| BUNDLE = 0.0058, mean |Δ| PAIRRANK = 0.0080. **Wilcoxon signed-rank p=0.875**. BUNDLE has greater non-target regression than PAIRRANK on only 3 of 9 identities. Roy_D specifically: BUNDLE Δ +0.638/+0.577/+0.562 across step500/3750/4000; PAIRRANK Δ +0.492/+0.523/+0.485 across step500/6000/6750 — similar magnitude across both arms (`analysis/p1_pe_eval_2026-05-07/counter_theory_per_identity_deltas.csv`). The GroupDRO-balloon hypothesis is empirically refuted on this data.
- **2026-05-07 (axis attribution)**: Roy_D's regression has a specific axis signature. P8A's r(score, color_b_dev) on roy_d = −0.714 (highest single-axis r magnitude observed for P8A on this identity). For all 6 P1 ckpts, r(Δscore vs P8A, color_b_dev) = +0.582 to +0.714 — the Δ-r magnitude on `color_b_dev` matches P8A's r magnitude. On `min_dim` and `sharpness` the Δ-r magnitudes are <0.30 and <0.13 respectively. **The arithmetic implication: P1 ckpts unwind exactly the `color_b_dev`-related signal P8A learned on roy_d.** Source: `analysis/p1_pe_eval_2026-05-07/roy_d_regression/roy_d_with_axes.csv` (130 frames × 4 axes × 8 ckpts).
- **2026-05-07 (paired-lane coverage)**: F2(a) audit found that of the 6 yaml-named paired training lanes (df40, deeplive, visomaster_v1_base, visomaster_enhanced, visomaster_teams_enhanced, deeplive_teams), only `deeplive_teams` has any Phase A eval-substrate proxy. Roy_D is NOT in any paired training lane. Source: `analysis/p1_pe_eval_2026-05-07/f2_pair_rank/F2_PAIR_RANK_FACTS_2026-05-07.md` Caveats §1, §5.

## Current stance (2026-05-07)

**Hypothesis-tentative**: `pair_rank_loss` is the load-bearing lever for the Roy_D-class regression. Both arms (BUNDLE = pair_rank + GroupDRO + chronic_flag; PAIRRANK_ONLY = pair_rank only) regress on Roy_D with similar magnitude, ruling out GroupDRO-specific causes. The most parsimonious shared mechanism is pair_rank itself: it fires on tight `(sample_id, frame_idx)` pairs in 6 paired training lanes and pushes fake-class score above real-class score within each pair. On non-paired-lane real frames (like Roy_D), the model's representation may generalize the score-up direction — particularly when the FT erases a fine-grained learned association (P8A's color_b_dev=>real signal on roy_d).

**This stance is hypothesis-tentative**, not proven, because:
1. The shared regression in BUNDLE and PAIRRANK is consistent with multiple shared causes besides pair_rank — including the post-`2feea58` SVD-gradient activation, anchor_aware penalty, stability loss, or a generic FT-from-P8A direction. The counter-theory probe rules out GroupDRO; it does NOT prove pair_rank specifically.
2. The color_b_dev mechanism on Roy_D is correlation-only at 130 frames. We do not know whether Roy_D is in P8A's training set (memorization unwound by FT) or held-out (generalization erased by FT).
3. F2(a) is structurally not testable on Phase A substrate (1 of 6 lanes represented, baseline saturated), so we cannot directly measure pair_rank's effect on the paired lanes.

**Three counter-experiments that would falsify this stance** (each is a CPU job; cost-balanced; runnable independently):

1. **From-scratch CLIP+pair_rank run with no P8A intermediary**: if the resulting model also regresses on Roy_D, the FT-from-P8A path is not the cause; pair_rank itself is.
2. **PAIRRANK-only τ-sweep on Roy_D**: if Roy_D regression vanishes at lower τ for PAIRRANK_ONLY but not for BUNDLE, the regression is τ-dependent rather than feature-representation-dependent — pulls the rug from the "pair_rank pulls scores up" claim.
3. **Cross-substrate Roy_D color_b_dev test**: re-run the axis attribution on a different real-only substrate where Roy_D is absent. If P8A's color_b_dev=>real association is roy_d-specific (not present on other identities), the "P8A learned a useful identity-specific feature" framing weakens; the regression is more about a statistical artifact of roy_d being unusual.

## Packet timeline

- [P1](../packets/P1.md) — Roy_D regression first observed (2026-05-07). The thread is opened by P1's evaluation. F1+F4+F5 PASS for BUNDLE arm; Roy_D regression is the one substantial cost. The mechanism hypothesis is staked but not proven; this thread tracks the open question.

## Evidence locations

- `analysis/p1_pe_eval_2026-05-07/counter_theory_per_identity_deltas.csv` — per-identity Δ FPR vs P8A for both arms; Wilcoxon p=0.875 on non-targets.
- `analysis/p1_pe_eval_2026-05-07/roy_d_regression/roy_d_with_axes.csv` — 130 roy_d frames × 4 axes × 8 ckpts; Δr on color_b_dev.
- `analysis/p1_pe_eval_2026-05-07/roy_d_regression/roy_d_per_frame_scores.csv` — 1040 rows (8 ckpts × 130 frames) raw score data.
- `analysis/p1_pe_eval_2026-05-07/f2_pair_rank/F2_PAIR_RANK_FACTS_2026-05-07.md` — paired-lane coverage gap (5 of 6 yaml-named lanes have no Phase A proxy).
- `analysis/p1_pe_eval_2026-05-07/AGENT_PROPOSAL_2026-05-07.md` §3 (proposed mechanism), §4 (retraction log for prior GroupDRO-balloon hypothesis), §7 (counter-experiments).
- Memory: `project_signature_shortcut_finding.md` — frames the pair-rank lever class; precursor context for why this thread matters.
- Memory: `project_in_proj_svd_gradient_bug.md` — the post-`2feea58` codepath that P1 was the first to test under pair_rank/GroupDRO.

## Open loops

### Open loop: pair-rank-non-paired-lane-collateral
status: open
severity: high
first_seen: 2026-05-07
last_verified: 2026-05-07
close_criterion: a follow-up CPU or training experiment establishes whether pair_rank_loss is the specific lever responsible for non-paired-lane real-side regressions. Either: (a) a from-scratch CLIP+pair_rank run reproduces the Roy_D-class regression on its own substrate (confirms pair_rank), OR (b) a controlled FT-from-P8A run with pair_rank disabled reaches similar fake recall without the regression (refutes pair_rank as the exclusive cause).

The agent who opened this loop has staked a position in `analysis/p1_pe_eval_2026-05-07/AGENT_PROPOSAL_2026-05-07.md` §3 and §6. A reviewer should treat the staked position as hypothesis-tentative; the counter-experiments in `AGENT_PROPOSAL` §7 are the falsification tests. None of those counter-experiments have been run.

### Open loop: roy-d-color-b-dev-mechanism
status: open
severity: medium
first_seen: 2026-05-07
last_verified: 2026-05-07
close_criterion: determine whether P8A's r(score, color_b_dev) = −0.714 on roy_d is (a) generalization (roy_d is held-out from P8A's train set; P8A learned a useful identity-invariant feature), or (b) memorization (roy_d is in train set; P8A's association is set-specific). The (a) interpretation makes the P1 regression more concerning; (b) makes it less so. A grep of the train manifest for Roy_D would close the loop cheaply.
