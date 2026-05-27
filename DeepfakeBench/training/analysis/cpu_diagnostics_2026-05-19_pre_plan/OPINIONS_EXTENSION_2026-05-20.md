# OPINIONS EXTENSION — Pre-plan synthesis update 2026-05-20

> **OPINIONS doc.** Per `AGENT_GUIDE.md` Rule 5: every interpretive claim is one
> agent's reading. Past framings (mine and others') have been demonstrably
> wrong — this doc updates yesterday's `OPINIONS_2026-05-19.md` with
> findings from extension Jobs K/A-ext/B-ext/L/N. FACTS live in
> `FACTS_EXTENSION_2026-05-20.md`. Confidence calibration revised explicitly.

## §1 — The headline correction (high confidence)

Yesterday I wrote that **P8A vs T5C are operationally orthogonal on lockbox
reals (Pearson 0.10–0.34) and the ensemble path is structurally live** — and
proposed Path B (per-IQ-quartile router) as the inference-side lever. I also
proposed Path A: ship T5C step3500 under an amended tiebreak.

**Both proposals are now superseded by a simpler, higher-confidence move:**
ship **Slot A v2 (anchor_aware on T5C step3500 base, W&B `hp35c51p`,
step3500)** — it is rank-1 under 6 of 8 reasonable tiebreaks (vs T5C's 5
yesterday), has lockbox_real_fpr 0.0191 (just 0.07pp above P8A), and catches
**77% more lockbox fakes than P8A** (0.688 vs 0.387). It also reduces the
binding dor_shkedi over-fire rate by **6×** vs Slot β at deployment τ.

I missed this ckpt yesterday because the pre-plan analysis only loaded the
overnight scorecard (5 ckpts) — not the auto-mode scorecard (6 ckpts) that
landed 2026-05-16T22:46 UTC and includes Slot A v2 + Slot B real_rebal. The
auto-mode scorecard was launched after the overnight one and isn't in any of
the dated docs under `analysis/reschain_grl6_eval_2026-05-16/`. Yesterday's
Slot β-centric framing was working from a 5-ckpt landscape that excluded
the strongest candidate.

## §2 — Why Slot A v2 dominates (confidence: HIGH)

**Mechanism**. Slot A v2 = T5C step3500 + anchor_aware penalty (weight=5.0,
target=0.10, samples=16, pool=`dor-real-webcam-false-flag-no-virtual-bg`).
The anchor pool's content IS the binding failure mode that Slot β couldn't
solve. Concretely:

| metric | T5C s3500 | Slot β s3500 (6-axis GRL) | Slot A v2 s3500 (anchor_aware) |
|---|---:|---:|---:|
| dev_fake_macro_recall | 0.459 | 0.545 | 0.438 |
| viso_enhanced recall | 0.138 | **0.235** | 0.167 |
| lockbox_real_fpr | 0.0279 | 0.0882 | **0.0191** |
| lockbox_fake_recall | 0.660 | 0.541 | **0.688** |
| deeplive_enhanced | 0.626 | 0.738 | 0.552 |
| dor_shkedi lockbox FPR | 0.027 | 0.099 | **0.016** |

Reading: Slot β trades viso lift for dor regression. Slot A v2 trades a tiny
viso lift (vs T5C) for **better-than-P8A behavior across all three pillars
except real-FPR-by-0.07pp**. The single number that prevents rank-1 in
v3-fix is the 0.07pp tiebreak gap on lockbox_real_fpr — exactly the
"tiebreak is load-bearing" finding from yesterday's Job B.

**Confidence ~85%** that Slot A v2 is the operationally correct deployment
candidate under any reasonable cost-ratio interpretation. The +30pp absolute
lockbox_fake_recall lift over P8A is large; the +0.07pp lockbox_real_fpr cost
is essentially noise.

## §3 — The ensemble path is real but not a free lunch (confidence: MEDIUM-HIGH)

Job L's ensemble simulation:

- **T5C OR Slot A v2 s3500** (lockbox_real_fpr 0.0353, lockbox_fake_recall 0.7365, composite k=1 = **0.701**) is the strongest readout in the entire 9-ckpt × ensemble panel. Beats single Slot A v2 by +0.033 on composite k=1; recall lift +0.049 at FPR cost +0.016.
- **P8A AND Slot A v2 s3500** (FPR 0.0042, recall 0.395) is the lowest-FPR rule that retains P8A-class behavior. Useful for environments where false positives are dispositive.
- **P8A AND Slot A v2 s1500** (FPR 0.00071) is the lowest FPR observed across all rules; recall 0.355.

**Composite-k regime matters**:
- k ≤ 2 (FN cost ≥ ½ × FP cost): top OR-ensembles win.
- k = 5–10 (FP cost dominates): single Slot A v2 s3500 wins on composite (because OR-ensemble FPR scales linearly with the penalty while recall gain is bounded).

**Confidence ~65%** that an OR-ensemble (T5C ∨ Slot A v2 s3500) is operationally
preferable to single Slot A v2. Lower than for the single-ckpt recommendation
because:
1. Adds inference-time complexity (two models served simultaneously).
2. The OR rule's FPR (0.035) is +0.017 above the single Slot A v2 — production
   cost analysis for that gap is unclear.
3. The within-T5C-family ensemble (T5C ∨ Slot A v2) likely has high correlation
   on errors (both share T5C step3500 base) — gain may not generalize
   beyond the current lockbox panel. Yesterday's Pearson on lockbox reals
   showed T5C-family pairwise 0.90+ — the OR-ensemble gain comes from the
   non-shared decision-boundary region, not from independent error modes.

## §4 — What this means for tonight's 3-slot plan (confidence: MEDIUM-HIGH)

Yesterday's 3 GPU slots (planned but not launched):

1. **Slot 1: Slot β + anchor_aware stacked** — was hypothesized to recover the dor over-fire. With Slot A v2's data, we can now see that anchor_aware ALONE (no 6-axis GRL) already drops dor over-fires from 116 (Slot β) to 19 (Slot A v2). Adding the GRL on top is now a "does it stack additively or interfere?" question — second-tier interest. **Lower priority but still informative.**

2. **Slot 2: LoRA-L8-L9 ablation** — independent of Slot A v2 finding. Same priority as yesterday.

3. **Slot 3: 5-axis sister (drop luma)** — independent of Slot A v2 finding. Same priority as yesterday.

**Three concrete amendments to consider**:

(a) **Drop Slot 1 in favor of an additivity test of Slot A v2 + Slot β's GRL extension** — i.e., test whether stacking the 6-axis GRL on top of an anchor_aware base (rather than the original direction of stacking anchor_aware on top of Slot β recipe) preserves Slot β's viso lift while keeping Slot A v2's dor FPR. Same yaml work, different scientific framing.

(b) **Add a Slot 4: ship-mode evaluation packet for Slot A v2** — full 29-suite scorecard re-evaluation of Slot A v2 step3500 to confirm the contract numbers hold under the canonical (not minimal-9) suite manifest. Cost ~$15-25 scorecard run, not a training packet. This is the lowest-risk path to confirming Slot A v2 ship-readiness.

(c) **Defer all 3 training slots tonight; do (b) first** — if Slot A v2 is the deployment candidate, validating it on the canonical 29-suite manifest is the prerequisite. Tonight's planned training slots become tomorrow's launches once the validation lands.

## §5 — Open questions (calibration check)

| Question | My current best answer | Confidence |
|---|---|---:|
| Is Slot A v2 step3500 the deployment-grade ckpt? | YES under any cost-ratio k ≤ 5 | HIGH (~85%) |
| Does the T5C OR Slot A v2 OR-ensemble generalize beyond the lockbox panel? | UNKNOWN — needs holdout cohort | LOW (~35%) |
| Will Slot 1 (anchor + 6-axis GRL stack) beat Slot A v2 alone? | UNLIKELY — stacking risk + Slot β's GRL extension didn't help dor | LOW (~25%) |
| Will Slot 3 (5-axis no-luma) identify the Slot β viso-lift driver? | YES regardless of outcome | HIGH (~80%) — same as yesterday |
| Will Slot 2 (LoRA L8-L9) produce a deployment-grade candidate? | UNLIKELY — likely useful ablation only | LOW-MEDIUM (~30%) |
| Is the "P8A vs T5C-family orthogonality" finding still load-bearing for ensemble design? | YES — but the within-family ensembles (T5C ∨ Slot A v2) are also strong | MEDIUM (~50%) |

## §6 — Recommended next-decision pathway

**If user prioritizes shipping**:
1. Re-run scorecard on Slot A v2 step3500 against the canonical 29-suite manifest (`arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml`) under v3-fix policy. ~$15-25 GPU. ~2-3h wall-clock.
2. If contract numbers hold, ship Slot A v2 step3500 with amended tiebreak (any composite k ≤ 5 — or simply `lockbox_fake_recall_desc`) — promotes Slot A v2 over P8A.
3. The 3 tonight-slots become next-priority research; not blockers on deployment.

**If user prioritizes mechanism understanding**:
1. Launch Slots 1+2+3 tonight as planned. Each closes an open question.
2. Re-evaluate after the new ckpts land + scorecard.

**If user prioritizes maximum value per GPU dollar**:
1. Run the 29-suite scorecard validation of Slot A v2 first (~$15-25).
2. Use the remaining ~$60-90 budget for whichever of the 3 tonight-slots is most informative AFTER the Slot A v2 validation lands.

## §7 — Self-correction log (Rule 5)

**Framings I held yesterday that are now updated**:

1. "Slot A v2 is rank-2 by 0.07pp tiebreak per memory `project_band_shortcut_ood_hypothesis_2026-05-16`" — yesterday I treated this as established context but did not verify it against the per-frame data. Today's Job A extended **confirms** the numbers and **strengthens** the case: at the contract τ, the dor over-fire reduction is 6× vs Slot β.

2. "P8A and T5C are operationally orthogonal — ensemble path is live" — still TRUE on the original 5-ckpt panel. But the ENSEMBLE path is now de-prioritized because **Slot A v2 alone catches +30pp more lockbox fakes than P8A** without requiring inference-time ensembling.

3. "The §4.2 pocket spec is refuted" — UNCHANGED. Slot A v2's mechanism works via anchor_aware (training-time supervision on chronic-FP pool), not via data ingestion that the §4.2 spec was meant to enable. The data-ingestion lever remains structurally live as a separate path but is no longer the primary recommendation.

4. "Slot β's viso lift is bounded by dor_shkedi over-fire concentration" — CORRECT, and now empirically resolved: anchor_aware on the dor pool produces the dor-FPR reduction that Slot β's 6-axis GRL did not achieve. Adding axes to GRL is NOT the right mechanism for the dor failure; explicit training-time penalty on the false-flag pool IS.

5. "Path A (ship T5C step3500)" — SUPERSEDED. Slot A v2 step3500 dominates T5C step3500 on the same contract surface (lockbox_real_fpr 0.0191 vs 0.0279; lockbox_fake_recall 0.688 vs 0.660; viso 0.167 vs 0.138).

**Framings NOT updated by this analysis**:

- The "tiebreak is load-bearing" finding is reinforced — Slot A v2 changes the deployment landscape even more decisively than yesterday's T5C analysis suggested.
- The per-IQ-quartile decomposition (Job B yesterday) and the chronic-FP partition findings (Job E yesterday) remain valid.

## §8 — Outstanding work that would update these views

1. **Slot A v2 cross-suite validation** — does the auto-mode scorecard's 9-suite manifest result hold on the canonical 29-suite manifest? Per the memory `project_t6_t7_t5c_scorecard_2026-05-12` headline numbers, T5C's rank-3 was on the 29-suite manifest, so the 9-suite is a known reduced surface. The reduced surface CAN hide regressions on suites that don't load in the 9-suite manifest.

2. **Slot A v2 per-suite breakout** at the contract τ on the canonical 29-suite manifest. Specifically: does Slot A v2 retain the dor invariance under stress conditions (`teams_real_lighting_extreme_dev`, `teams_real_poor_quality_dev`)?

3. **Re-evaluation of yesterday's tonight-slot plan** in light of Slot A v2's emergence. Specifically: is Slot 1 (anchor + 6-axis GRL stack) still the right Slot 1, or should we test additivity from a Slot A v2 base instead?

I have not run any of these tonight — they require GPU budget. Recommend they be the first three GPU spends after user-approval of the amended deployment direction.
