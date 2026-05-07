# Thread: Anti-shortcut bundle decomposition

> **2026-04-30 finding** — when stacking multiple anti-shortcut interventions in one yaml without a single-lever ablation, the resulting bundle can be net-negative against its single load-bearing component. The P14 bundle (anchor_aware + pipeline_random + face_scale_jitter@0.25) scored value_composite=0.116 on FT-from-P8A; the sister variant with anchor_aware + pipeline_random DISABLED and jitter strengthened to scale_limit=0.50 scored 0.661 — a 5.7× lift from a strict subset of the bundle's interventions. Memory `project_face_scale_jitter_load_bearing.md` is the auto-memory anchor; this thread is the deliberated synthesis and the discipline rule that drops out.

## The question

When the team adds N anti-shortcut interventions to a training recipe in one packet, how do we know whether each intervention is contributing a positive lift, contributing nothing, or actively hurting? The P12/P13/P14 series stacked anti-shortcut interventions cumulatively — anchor-aware loss, pipeline-randomization aug, face-scale jitter — without isolating any lever. P14_FT empirically demonstrated that the bundle was net-negative vs its single load-bearing component (jitter alone at higher strength). The question this thread answers: **what discipline prevents the next packet from re-doing this — assuming a bundle works because each lever was independently motivated?**

## Initial belief

Through P10 → P14 the team treated each new anti-shortcut intervention as **additive on top of prior interventions in the same packet**, motivated by independent evidence: anchor-aware loss directly penalizes the false-flag pool; pipeline-randomization perturbs codec/JPEG/gamma signatures; face-scale jitter breaks the face-pixel-area shortcut. Each was independently motivated by a documented shortcut axis (memory `project_signature_shortcut_finding.md`, `project_face_size_label_leak.md`, `project_lockbox_fpr_dominated_by_webcam_mode.md`). The implicit assumption was that the levers are *orthogonal* — they attack different shortcut axes, so they should compose additively. P11_HEAVY → P13_FROM_SCRATCH → P14_FT each accumulated more levers without reverting prior interventions to test whether the prior was still pulling its weight.

The closest the team got to bundle-decomposition discipline pre-2026-04-30 was the **commit-isolation discipline forged 2026-04-26** (`detectors/effort_detector.py` working tree split: in_proj-SVD fix + feat_norm-reg pulled apart into two commits). That discipline applied to *code patches*, not *training recipes*. The transferable principle did not get articulated until the 2026-04-30 jitter-isolated result.

## What changed our mind

- **2026-04-30 — Sister-variant ablation `R13_P14_FACE_SCALE_JITTER_ISOLATED.yaml` added to the overnight slate at user request.** The yaml was deliberately constructed as a single-variable delta from `R13_P14_FT_FROM_P8A.yaml`: anchor_aware ENABLED → DISABLED, pipeline_randomization ENABLED → DISABLED, face_scale_jitter scale_limit 0.25 → 0.50. Seed 2273 (vs P14_FT 737, P15 1501) to avoid wandb collision. The motivating question recorded in the yaml header: *"Is face-scale-jitter the LOAD-BEARING lever in the P14 bundle, or is the lift mostly anchor_aware + pipeline_random?"*

- **2026-04-30 — Result: jitter@0.50 alone scores value_composite=0.661 vs the bundle's 0.116** (5.7×) (`gs://training-job-outputs/phase2r13_experiments/mclioexb/`, run id `mclioexb`, wandb summary). Cross-method generalization was preserved by the FT init (other_fakes_tpr=0.591 vs the bundle's 0.020). DATA_FIX (bundle + visomaster_teams_enhanced fw=8.0) scored 0.126, statistically tied with the bundle. P15 (bundle + GRL λ=0.20) scored 0.516 — second to jitter-isolated, materially above the bundle.

- **2026-04-30 — Two interpretations of the bundle being net-negative are consistent with the observation; both probably contribute** (`docs/packet_retrospectives/packets/P14.md` § "Headline finding"). (1) Jitter at 0.25 is undertrained against the face-pixel-area shortcut; the bundle's other interventions don't compensate, so the bundle's jitter half is doing all the lifting in a regime where it can't yet bite. Jitter at 0.50 (matching the t∈[0.7, 1.5] tightness range that flipped 53% of frames in the 04-27 audit per [`face_size_label_leak`](face_size_label_leak.md)) finally has enough range to bite. (2) anchor_aware + pipeline_randomization on top of jitter create regularization conflict in the FT-from-P8A regime — too many soft constraints simultaneously lead to a narrow specialization where most fake-method generalization collapses (other_fakes_tpr 0.020 in the bundle vs 0.591 in jitter-isolated).

- **The 5.7× gap is a discipline signal, not just a recipe finding.** A 5.7× value_composite gap from a strict subset of a bundle's interventions means the bundle as a whole was actively hurting in this regime — not "less than the sum of its parts" but "less than its strongest single part". Without the sister-variant ablation, the team would have shipped P14_FT's 0.116 reading as the verdict on the entire P14 anti-shortcut hypothesis. The bundle hypothesis's failure mode would have been mis-attributed to "anti-shortcut interventions don't work on FT-from-P8A" rather than the correct attribution: "this specific bundle composition was wrong; one of its components alone is the lever".

## Current stance (2026-04-30)

When stacking anti-shortcut interventions in a single training packet, one of two disciplines is required:

1. **Single-lever ablation in the same packet**: at least one slot in the packet runs only the new intervention without prior interventions, holding everything else fixed. The 5.7× jitter-isolated vs bundle gap is a strong existence proof that this discipline must apply at the *intervention* level, not just the *hyperparameter* level — anchor_aware on/off is a different axis than anchor_aware weight 5.0 vs 2.0.

2. **Sequential build via consecutive packets**: packet N introduces intervention K; packet N+1 stacks K + intervention K+1; the per-packet leader board carries the cumulative bundle's verdict, and a "remove K" ablation is queued for any K that gets superseded. Less efficient slot-wise but easier to administer.

The user's preference (memory `feedback_decision_points.md` — "user reserves judgment calls at decision points") tilts toward option 1 in a single packet when the slot budget allows. P14 had budget for 2 slots originally (FT + DATA_FIX); the sister variant expanded to 3, and the marginal cost (~$60-70) was small relative to the discipline value of the empirical decomposition.

The **operational rule** going forward: if the next anti-shortcut packet stacks more than one lever, at least one slot must be the strongest single lever alone, holding the FT init + data + LR fixed. The bundle's other levers may be retained, but their additive contribution is always measurable against the single-lever baseline. Threads with active anti-shortcut interventions ([`processing_signature_shortcut`](processing_signature_shortcut.md), [`face_size_label_leak`](face_size_label_leak.md), [`webcam_fpr_dominance`](webcam_fpr_dominance.md), [`calibration_vs_training_aug`](calibration_vs_training_aug.md)) all interlock with this rule.

A cross-cutting corollary: **when a bundle fails, the failure mode "one of the bundled levers cancels another's gradient" is at least as plausible as "the bundled levers are insufficient"**. The narrative bias before 2026-04-30 was toward the latter ("we need more interventions to beat the shortcut"); the empirical evidence after 2026-04-30 says the former is at least as common in the FT-from-P8A regime. Successor packets should shrink before they grow.

## Packet timeline

- [P10](../packets/P10.md) — symmetric router + GRL slate drafted; only the symmetric-router half ran (Phase C). First multi-intervention packet, but the GRL half was deferred so de-facto a single-lever packet.
- [P11](../packets/P11.md) — first packet to land multiple training-time interventions in distinct slots (MILD vs HEAVY ctx_scale; HEAVY_DEEPLIVE; WEBCAM_HARDEN). The portfolio shape is the precursor to bundle-decomposition discipline, but slots tested *parameters* of one intervention, not *whether each intervention contributes*.
- [P13](../packets/P13.md) — first packet to deliberately stack three anti-shortcut classes (anchor_aware + pipeline_random + face_scale_jitter); the wandb-flattening bug initially silenced two of them; even post-fix, the bundle was tested only as a stack, not against subset-ablations. The bundle's verdict γ was attributed to "anti-shortcut interventions on a from-scratch substrate are insufficient" — an attribution P14_FT inherited unchallenged.
- [P14](../packets/P14.md) — bundle inherited from P13 + FT-from-P8A regime; sister variant `R13_P14_FACE_SCALE_JITTER_ISOLATED` added to the launch slate at user request, providing the first single-lever subset ablation in the bundle's lifecycle. The sister won by 5.7×, which is what generated this thread.
- [P15](../packets/P15.md) — GRL on top of the same bundle; second-place result (0.516) consistent with this thread's finding that the bundle is dragging *whatever else is layered on top of it*. The natural follow-up is "P15 + jitter@0.50 - bundle".
- [P1](../packets/P1.md) (2026-05-06 → 2026-05-07) — first packet to follow the bundle-decomposition discipline DELIBERATELY: BUNDLE = pair_rank + GroupDRO + chronic_flag; PAIRRANK_ONLY = pair_rank only. Matched-step ablation (BUNDLE_step500 vs PAIRRANK_step500) shows different tradeoffs on different axes — BUNDLE wins lockbox recall (82.6% vs 70.8%) but PAIRRANK wins `dev_fake_macro_recall` (0.354 vs 0.075). The slot-1-vs-slot-2 question "did GroupDRO add value over pair_rank alone?" answers: **yes on F5 chronic-FP** (BUNDLE pc_generator FPR 0% vs PAIRRANK 7-14%), **no on dev_fake_macro** (PAIRRANK is the contract rank-1 winner). **Critically**: the agent's initial mechanism claim ("GroupDRO has balloon effect on non-target chronic identities") was REFUTED by a counter-theory probe (Wilcoxon p=0.875 between BUNDLE and PAIRRANK arms on 9 non-target identities). The shared regression on Roy_D (29% → 78-93% across both arms) suggests the load-bearing lever for both gains AND collateral damage is `pair_rank_loss` itself, not GroupDRO. New thread [`pair_rank_collateral`](pair_rank_collateral.md) tracks the open hypothesis. Source: `analysis/p1_pe_eval_2026-05-07/AGENT_PROPOSAL_2026-05-07.md` §4 (retraction), `counter_theory_per_identity_deltas.csv`.

## Evidence locations

- `experiments/phase2_round13/R13_P14_FT_FROM_P8A.yaml` — the bundle: anchor_aware ENABLED + pipeline_random ENABLED + face_scale_jitter@0.25.
- `experiments/phase2_round13/R13_P14_FACE_SCALE_JITTER_ISOLATED.yaml` — the single-lever ablation: anchor_aware DISABLED + pipeline_random DISABLED + face_scale_jitter@0.50. Header docstring captures the motivating question.
- `experiments/phase2_round13/R13_P14_DATA_FIX.yaml` — bundle + visomaster_teams_enhanced fw=8.0 (Layer-2 closure variant); ran 2026-04-30 with the bundle inherited; collapsed cross-method generalization (other_fakes_tpr=0.047). Reinforces the bundle-drag finding.
- `experiments/phase2_round13/R13_P15_GRL_FROM_P8A.yaml` — bundle + GRL λ=0.20; ran 2026-04-30 with the bundle inherited; second-place at 0.516.
- W&B summaries: `dtect-vision/phase2-experiments/runs/{mclioexb,w5tky6ss,xan4dfto}` — the three overnight runs' value_composite + cross-method TPR breakdown.
- Memory: `project_face_scale_jitter_load_bearing.md` (auto-memory anchor; lever-attribution + the discipline rule).
- Memory: `project_signature_shortcut_finding.md`, `project_face_size_label_leak.md`, `project_lockbox_fpr_dominated_by_webcam_mode.md` — the three independently-motivated shortcut axes that motivated the bundle's three interventions; the bundle-decomposition finding refines (does not refute) any of them.
- Master plan log: not yet authored as a session entry post-2026-04-30 (PLAN.md is the active plan surface; LOG entry pending the next session).

## Open loops

### Open loop: anti-shortcut-bundle-needs-single-lever-discipline
status: open
severity: medium
first_seen: 2026-04-30
last_verified: 2026-04-30
close_criterion: the next anti-shortcut packet that stacks more than one intervention is structured with at least one single-lever ablation slot (the strongest lever alone, FT init + data + LR fixed) AND the packet retro records whether the bundle is net-additive vs the single-lever baseline. The discipline either becomes a `BUILD_SCAFFOLD.md`-style operational rule for future packets, OR a counter-example (a stacked bundle that demonstrably beats its single-strongest component on a deployment-relevant axis) is filed and this loop is resolved as superseded.

The discipline is a behavioral rule, not a technical fix. Closing it requires either (a) a packet that follows the rule and a retro that documents the comparison, or (b) explicit user override authorizing bundle-only packets going forward (e.g., if a packet-N-bundle is so well-motivated that the slot cost of the ablation outweighs the discipline value). Either close path generates evidence; the loop closes on writing-it-down.

### Open loop: bundle-failure-mode-attribution-revision
status: open
severity: low
first_seen: 2026-04-30
last_verified: 2026-04-30
close_criterion: the P13 retro's γ-verdict attribution ("anti-shortcut interventions on a from-scratch substrate are insufficient") is either reaffirmed by a from-scratch single-lever-jitter-only run, OR explicitly revised in the P13 retro to "anti-shortcut bundle as composed in P13 was insufficient on from-scratch; whether the single-lever jitter@0.50 would have been enough is unknown without re-running."

The 2026-04-30 finding does not directly invalidate P13's γ verdict (different FT regime, different baseline), but it does invalidate the *generalization* of P13's verdict across all "anti-shortcut on from-scratch" framings. A future agent reading the P13 retro should not extrapolate "anti-shortcut doesn't work on from-scratch" without flagging that P13 only tested the bundle. The cheapest close path is annotating P13.md with a 2026-04-30 caveat block; the more rigorous close is a from-scratch jitter@0.50 single-lever run, which is probably not worth the spend given the P14-regime evidence already in hand.

## Cross-thread refs

- [`face_size_label_leak`](face_size_label_leak.md) — the load-bearing shortcut axis that jitter@0.50 attacks. The face-size leak's close criterion was the lever that won the bundle decomposition; the bundle-drag finding is the *complement* of that thread's empirical lever-validation.
- [`processing_signature_shortcut`](processing_signature_shortcut.md) — the broader shortcut narrative; bundle decomposition is the discipline rule that drops out when *multiple* shortcut axes' interventions are stacked. The `shortcut-deployment-block` (critical, in-progress) loop in that thread interlocks: if P16-style packets follow the discipline and find a stack that beats jitter@0.50 isolated, that would be the next inflection point on the deployment-block close criterion.
- [`webcam_fpr_dominance`](webcam_fpr_dominance.md) — anchor_aware's specific job was the webcam-FPR axis; the bundle drag may be partially anchor_aware-specific. A useful follow-up: P14 + anchor_aware ONLY (no pipeline_random, no jitter) to disentangle anchor_aware's contribution from the bundle's compound effect. Slot cost ~$60.
- [`calibration_vs_training_aug`](calibration_vs_training_aug.md) — the WS-P1 finding that training-aug is the dominant cross-pool FPR lever (vs calibration) is consistent with this thread's stance that the right training-time intervention is high-leverage. Calibration is a complement, not a substitute for finding the right single lever.
- [`viso_bucket_gap`](viso_bucket_gap.md) — the DATA_FIX failure (Layer-2 closure with the bundle on top of it) is reframed by this thread: DATA_FIX did not get a fair test because the bundle was dragging it. A clean DATA_FIX test would be FT-from-P8A + jitter@0.50 + viso_teams_enhanced (no anchor_aware, no pipeline_random); whether *that* configuration would have closed the bucket-gap close criterion is unknown.
