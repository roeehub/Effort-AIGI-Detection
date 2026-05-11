# Agent Proposal — T4 packet interpretation (2026-05-11)

> **Status: OPINION, not RECORD.** Reviewer is invited to disagree. Each load-bearing claim cites specific numbers from `RESULTS_FACTS_2026-05-11.md` (`§N.M`).
>
> **Authoring**: written 2026-05-11 by the agent resuming from `HANDOFF.md` after the T4 promotion-contract scorecard completed. The agent also wrote the FACTS doc and ran the post-hoc CPU diagnostics.
>
> **Required disclaimer per convention**: past framings in this project have been demonstrably wrong (e.g., the original P15/P18 GRL packets predicted invariance lift that did not materialize; the cyclic-λ T5-A packet was hypothesized to capture attractor moments and was refuted same-night). This doc adds another candidate framing to that ledger; treat it skeptically.

---

## 1. Executive summary

The T4 packet did not place a new ckpt above P8A on the v3-fix contract (FACTS §3.1). Among the 7 ckpts scored, P8A is rank-1; T4_LAMBDA1_TOP_N_STEP10500 ranks 3.

The patterns I see in the per-suite + AUC + score-distribution data are consistent with **substrate-overfit between dev and lockbox**: T4_L1_step10500 lifts AUC on every dev cell (+0.037 to +0.082) while dropping lockbox AUC by 0.174 (FACTS §7.1). The dev-recall lift is concentrated almost entirely on `deeplive_enhanced_dev` (+0.286 vs +0.057, +0.007 on the other dev fake cells; FACTS §4.2). Score percentiles on lockbox compress from the wide P8A range to a narrow mid-range under T4 (FACTS §6.1, §6.2), with the fake-vs-real median ratio collapsing from 51.2 to 1.68 (§6.4).

The implications I draw — that T5-B / T5-C as currently scoped don't address this, that the L11 inv_mean atlas does not predict deployment behavior, that the encoder + forgery-head pipeline is the right unit-of-analysis — are interpretations of these facts. I assign them MEDIUM confidence, and §5 names the framings I retract from my own session.

## 2. High-confidence claims

These are direct paraphrases of FACTS, not interpretations:

1. P8A is contract rank-1; no T4 ckpt is rank-1. **Citation**: FACTS §3.1.
2. T4_L1_step10500's dev_fake_macro_recall lift (+0.117 absolute vs P8A) is +0.286 on `deeplive_enhanced_dev` and ≤ +0.057 on the other two dev fake cells. **Citation**: FACTS §4.2.
3. T4_L1_step10500's lockbox AUC (0.7619) is 0.174 absolute below P8A's (0.9355). **Citation**: FACTS §7.1.
4. T4_L1_step10500's dev AUCs are 0.037–0.082 absolute above P8A's across all 3 dev fake cells. **Citation**: FACTS §7.1.
5. No τ in [0.50, 0.99] gives T4_L1_step10500 both lower lockbox_real_fpr AND higher lockbox_fake_recall than P8A simultaneously. **Citation**: FACTS §5.2.
6. T4_L1_step10500 lockbox p50_fake / p50_real ratio is 1.68 vs P8A's 51.2. **Citation**: FACTS §6.4.
7. AUC is invariant under any monotonic calibration (including isotonic); no rank-preserving rescaling lifts T4_L1_step10500's lockbox AUC ceiling of 0.7619. **Citation**: FACTS §8.3.

## 3. The mechanism I propose (caveated)

**Claim** (MEDIUM confidence): T4 produced **substrate-overfit between dev and lockbox** — the encoder + forgery-head pipeline learned features that separate dev fakes from dev reals more sharply than P8A's pipeline, at the direct expense of fake-vs-real separation on lockbox.

**What this mechanism would explain**:
- The simultaneous +ΔAUC on dev and −ΔAUC on lockbox (claims #3 and #4).
- The dev recall lift's concentration on `deeplive_enhanced_dev` (claim #2): one dev cell carries the lift, consistent with a feature direction specific to dev's deeplive distribution rather than to deepfakes generally.
- The score compression on lockbox (claim #6): if the encoder/head pair maps lockbox into a narrow region of feature space (because the discriminative directions it learned are absent there), both reals and fakes end up scoring near the same value.

**What this mechanism does NOT explain**:
- Why the L11 inv_mean atlas (LR-probe on cached features, +18% vs ceiling) moved in the opposite direction from lockbox AUC. The mechanism above predicts the LR-probe should also show lockbox-specific degradation; instead the probe is averaged over a shared triptych frame set and shows a small positive shift. This is a real unresolved tension between the probe metric and deployment AUC.
- Whether the substrate gap is specifically dev↔lockbox or would also appear dev↔HDTF or dev↔production-may6. We have no T4 numbers on those substrates yet.

**What I am unsure about**:
- Whether the same pattern would appear with a different GRL configuration (λ, axes, attachment layer, classifier capacity). The N=2 ckpts compared (T4_L1_step10500 + T4_L2_step1500) both show compression, but the sample is small and the parameter sweep is narrow.
- Whether the "forgery head learning residual substrate-correlated features" sub-mechanism is the load-bearing one, or whether the encoder itself encodes substrate differently and the head is doing its standard work. I have no direct measurement that distinguishes these.

## 4. Confidence-tiered claims

| Claim | Confidence | Citation |
|---|---|---|
| P8A is rank-1; no T4 ckpt displaces it | HIGH | FACTS §3.1 |
| Lockbox AUC drops 0.174 for T4_L1_step10500 vs P8A | HIGH | FACTS §7.1 |
| Calibration cannot lift lockbox AUC above 0.7619 | HIGH | FACTS §8.3 |
| Dev recall lift is concentrated on deeplive_enhanced_dev | HIGH | FACTS §4.2 |
| T4_L1_step10500 score distribution on lockbox is narrower than P8A's | HIGH | FACTS §6.1, §6.2 |
| The dev↑ + lockbox↓ pattern reflects substrate-overfit | MEDIUM | inference from claims #3, #4 |
| The forgery head specifically learned residual substrate features | LOW | speculative sub-mechanism in §3 |
| T5-B (multi-layer attach) would compound this | LOW | depends on mechanism in §3 being correct |
| T5-C (bigger GRL classifier) would compound this | LOW | depends on mechanism in §3 being correct |
| Atlas L11 inv_mean does not predict deployment behavior | LOW (N=1 ckpt, N=1 packet) | inference from §3 + FACTS §7.1 |

## 5. Self-correction log

During this session I authored an earlier version of this diagnostic (uncommitted) that overstated several claims. The version-1 framings, the retractions, and the lessons:

| Retracted framing | What I said | Why it was overconfident | Where the correction lives now |
|---|---|---|---|
| "The forgery head learned to use residual substrate-correlated features" (asserted as established mechanism) | TL;DR of uncommitted T4_DIAGNOSTIC_2026-05-11.md | I have no probe that measures which sub-component learned what; I inferred from AUC pattern + score compression, but that pattern is consistent with several mechanisms. | Reframed as MEDIUM/LOW-confidence speculation in §3; tagged LOW in §4. |
| "Atlas L11 inv_mean is NOT a deployment north-star" (broad prescriptive) | Reframe section of T4_DIAGNOSTIC_2026-05-11.md | Generalization from N=1 ckpt + N=1 packet. The atlas-vs-deployment tension is real but data-poor. | Replaced with tiered claim "Atlas L11 inv_mean does not predict deployment behavior" — LOW confidence (§4). |
| "T5-B and T5-C as scoped don't address the diagnosed root cause" (presented as conclusion) | Implication section of T4_DIAGNOSTIC_2026-05-11.md | This claim is downstream of the LOW-confidence mechanism in §3. Phrased as conclusion in version-1; it's actually a conditional opinion. | Marked LOW in §4; reframed as conditional in §6 next-steps. |
| Recommendation block ("Stop optimizing atlas inv_mean as the north-star") | Recommendation section of T4_DIAGNOSTIC_2026-05-11.md | Strong prescriptive on the basis of a LOW-confidence chain. | Reframed in §6 as one of several next-step options the user can weigh, not a prescription. |
| Memory entry: "stop using inv_mean as success criterion; T5-B/T5-C as scoped intensify the diagnosed problem" | `~/.claude/projects/.../memory/project_t4_substrate_overfit_inv_mean_misleading_2026-05-11.md` | Same overreach as the doc-level claims; propagates into future agent contexts. | Memory entry rewritten 2026-05-11 to FACTS-leaning, with the mechanism explicitly tagged as opinion. |

**Lesson for future me / next agent**: when one diagnostic produces a story that explains everything tidily, treat it with suspicion. The FACTS doc should be written first and the OPINION doc second so that the OPINION is a *response* to facts rather than a *narrative* the facts get fit into. I did this in reverse — wrote the diagnostic, then constructed the FACTS doc — which let speculative framings carry over.

## 6. Proposed next steps (options, not prescriptions)

The user picked "diagnose first" from the earlier menu; the diagnostic above is in. Next moves are the user's call. Possibilities I see, by lever class:

### Investigation (CPU-only, $0)

1. **Cross-substrate AUC for T4_L1_step10500**: measure AUC on HDTF (per `project_job_b_findings_universal_vs_trajectory_2026-05-04.md`) and on may6 production frames (per `project_xinhe_may6_falseflag_2026-05-06.md`). If dev↑/lockbox↓ pattern repeats on those substrates, the substrate-overfit framing gains support. If lockbox is the unique loser, the framing is over-general.
2. **L11 feature probe on T4_L1_step10500 features for lockbox specifically**: re-run the 5-fold LR probe used in the atlas, but use lockbox frames as the probe set. Tests whether the encoder's lockbox feature representation has lost shortcut signal AND lost forgery-discriminative signal, or just one.
3. **Per-frame disagreement audit**: of the lockbox fakes T4 misses that P8A catches, what are their characteristics? Are they the same chronic-6 cohort? This is the §9 (Job 9) pattern from `project_job9_disagreement_audit_2026-05-04.md`.

### Training (GPU spend)

4. **T5-C (hidden_dim=1024)** — if user judges the mechanism in §3 is wrong (e.g., the issue is GRL-classifier weakness, not substrate overfit), T5-C remains a clean test of the bigger-classifier hypothesis. ~5h training, ~$70. The pre-test 1 from the handoff showed hidden_dim scan; would extend it.
5. **T5-B (multi-layer L6+L11 attach)** — same caveat: if §3 mechanism is wrong, T5-B is the multi-layer test the handoff proposed. ~2h engineering, ~5h training, ~$70.
6. **Two-stage training** — train encoder with GRL on dev, freeze, then train forgery head on a substrate-diverse (dev+held-out) set. Targets the head/encoder conflict. ~4h engineering, ~5h training, ~$70.

### Status quo

7. **Park T4/T5-A/T5-B/T5-C, keep P8A as the deployment baseline.** Cheapest. Honest if the user judges the multi-axis-L11-GRL program has been adequately tested for N=1 packet + 1 follow-up.

## 7. Counter-experiments that would falsify the mechanism in §3

If the substrate-overfit framing is wrong, one of these would show it:

1. **Cross-substrate AUC** (option #1 above): if T4_L1_step10500's HDTF AUC is ABOVE P8A's, "lift on dev / loss on lockbox" was lockbox-specific, not dev-vs-everything-else.
2. **Synthetic substrate test**: re-render lockbox fakes with dev-substrate processing and re-score. If T4_L1_step10500's recall on the re-rendered fakes ≈ its recall on dev fakes, the encoder *can* discriminate the underlying manipulation when substrate is held constant; the head/encoder pair just couldn't bridge substrates.
3. **Linear-probe on T4 features for lockbox fakes-vs-reals**: if a frozen-T4-feature linear probe gives lockbox AUC ≈ 0.93 (matching P8A), the encoder retained separation power and only the head failed. If linear probe AUC ≈ 0.76 (matching the trained-head AUC), the encoder lost separation power.

## 8. Open questions I cannot answer from this data

- Is T4_L1_step10500's lockbox AUC drop unique to lockbox, or also visible on HDTF and may6? (Critical for whether the framing in §3 is substrate-general or lockbox-specific.)
- Does the dev_recall lift on `deeplive_enhanced_dev` reflect (a) the encoder learning a deeplive-specific feature, (b) the head exploiting a shortcut in the dev's deeplive subset, or (c) a real generalizable signal that just happens to fire on dev's deeplive? Need cross-substrate deeplive numbers.
- Why does the atlas L11 inv_mean show +18% for T4_L1_step10500 while lockbox AUC shows −0.17? The atlas LR-probe runs on cached features from a triptych that includes lockbox-ish frames; this should have aligned the two metrics directionally, but they diverged. Plausible: the triptych frame mix is dev-leaning, so the atlas measures dev features primarily. Untested.

## 9. Suggested CPU jobs not yet run

These would update my belief regardless of which direction they point:

1. **Cross-substrate AUC matrix**: T4_L1_step10500 + P8A + E2B × {dev, lockbox, HDTF, may6}. ~1h CPU. Direct test of §3.
2. **Frozen-encoder linear-probe on lockbox**: described in §7 #3. ~1h CPU.
3. **Triptych composition audit**: enumerate which substrates the atlas frame mix comes from. Tests whether the atlas measurement is dev-biased. ~30 min, just reading manifests.
4. **Per-frame catch-disagreement on lockbox**: §9 pattern, T4 vs P8A. ~1h CPU.

## 10. Reviewer guidance

If you (the reviewer) are picking this up cold:

1. Read `RESULTS_FACTS_2026-05-11.md` first. Form your own view on what the AUC + score-distribution numbers imply.
2. Only then read §3 of this doc (the proposed mechanism). Compare against your independent view.
3. If you disagree with the mechanism, the counter-experiments in §7 are designed to be cheap-ish ways to discriminate; pick the one that most cleanly distinguishes our two views.
4. The self-correction log in §5 is the most important section of this doc — it names the framings I retracted from my own version-1, and the *category* of error (one-tidy-story overconfidence). Future agents should treat it as guidance for what NOT to do when reading their own diagnostics.
