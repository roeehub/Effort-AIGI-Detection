# Synthesis and Ranked Recommendations

**Date:** April 15, 2026  
**Purpose:** unify the repo analysis, experiment memory, data audit, target-domain findings, whitepaper gap analysis, and literature review into one honest decision memo

## Executive Verdict

The next step should **not** be “another clever mixed-source weight sweep.”

The research converges on a sharper conclusion:

1. The current system is still **structurally under-training the exact failure mode** you care about: enhanced deepfakes after Microsoft Teams processing.
2. The current sampler and merged-source design make several nominal weights look more meaningful than they really are.
3. The current evaluation and checkpoint-selection story is still too fragmented for a low-FP deployment objective.
4. The most likely path to a meaningful jump is:
   - tighten the selection lane,
   - split the merged source so the target-domain truth is explicit,
   - run one high-purity target-domain fine-tune,
   - run one nuisance-invariance ablation,
   - and add a deployment-side calibration / temporal decision sweep.

If you spend the next round on more small family-weight tweaks inside the current sampler, the expected return is low.

## The Most Important Convergent Findings

### 1. The Track A merged source does not actually teach the full target condition

This is the strongest and most repeated finding across the reports:

- `visomaster_teams_enhanced` is mostly `clean_fallback`, not true Teams companion.
- only `54 / 997` merged rows are `teams_v2_companion`;
- in train, only `42 / 837` merged rows are true Teams companion;
- with `p_original = 0.5`, expected true Teams fake exposure from this source is tiny;
- the loader never emits a fake frame that is both **enhanced** and **Teams-processed**.

That means the current Track A path can improve fake recall through better enhanced-fake supervision, but it cannot fully solve the true `enhanced-after-Teams` problem because the exact fake condition is not present in the emitted training pairs.

Support:

- [04_data_composition_and_curriculum_opportunities.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15/04_data_composition_and_curriculum_opportunities.md)
- [05_target_domain_gap_and_teams_enhanced_structural_findings.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15/05_target_domain_gap_and_teams_enhanced_structural_findings.md)

### 2. Weight tuning is weaker than it looks under the current sampler

The current identity-balanced paired-object sampler means:

- most identities never present a true cross-family choice;
- only `39` train identities create real cross-family competition;
- real-family weights mostly do not affect paired-object selection;
- DF40 still lands much harder in the actual diet than its nominal `0.15` fake weight suggests.

So the current system is not in a regime where another careful set of small weight edits is likely to unlock a big qualitative change.

Support:

- [04_data_composition_and_curriculum_opportunities.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15/04_data_composition_and_curriculum_opportunities.md)
- [01_repo_system_map.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15/01_repo_system_map.md)

### 3. Track A is a branch result, not a promotion result

Track A clearly did something useful:

- better target fake recall,
- much better `visomaster_enhanced_macro` recall.

But it also clearly did something harmful:

- worse real Teams false positive rate on dev,
- worse real Teams false positive rate on lockbox.

That makes Track A an informative branch, not a replacement for `R12_G`.

Support:

- [05_target_domain_gap_and_teams_enhanced_structural_findings.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15/05_target_domain_gap_and_teams_enhanced_structural_findings.md)
- [03_experiment_memory_and_run_forensics.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15/03_experiment_memory_and_run_forensics.md)

### 4. The repo still has unclosed nuisance-invariance gaps

Two repo-specific issues stand out:

- lighting robustness is still incomplete;
- spatial/crop robustness is still weak.

Worse, there is a credible config-path reason to believe the intended `GammaUp` path in current R13 configs may never have been active because the key names were wrong.

That means some augmentation conclusions from recent runs are not yet trustworthy enough to close the question.

Support:

- [01_repo_system_map.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15/01_repo_system_map.md)
- [05_target_domain_gap_and_teams_enhanced_structural_findings.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15/05_target_domain_gap_and_teams_enhanced_structural_findings.md)
- [07_literature_review_stability_calibration_and_low_fp.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15/07_literature_review_stability_calibration_and_low_fp.md)

### 5. The paper is still directionally useful, but not operationally sufficient

The whitepaper's core lesson still holds:

- preserve pretrained semantics,
- learn the fake residual carefully,
- do not blindly fully fine-tune away the prior.

But the paper is not enough for:

- Teams transport effects,
- enhanced-through-Teams,
- low-FP decisioning,
- temporal instability,
- or the smaller B/16 regime under deployment pressure.

Support:

- [02_whitepaper_to_repo_gap_analysis.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15/02_whitepaper_to_repo_gap_analysis.md)

### 6. Low false positives require a decision-system upgrade, not only a training upgrade

The repo already knows:

- score scale drifts badly across domains,
- threshold `0.5` is not semantically stable,
- frame scores can flicker,
- and deployment wants asymmetric cost.

So low-FP improvement is partly a calibration and temporal decision problem, not only a better checkpoint problem.

Support:

- [07_literature_review_stability_calibration_and_low_fp.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15/07_literature_review_stability_calibration_and_low_fp.md)
- [01_repo_system_map.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15/01_repo_system_map.md)

## Ranked Next Moves

## Rank 1: Freeze the evaluation lane and shortlist the real candidates

Before another expensive round, compare a fixed shortlist on one frozen target-domain lane:

- `R12_G`
- best current Track A checkpoint
- `R13_E`
- `R13_FT10`

Judge them with:

- the same frozen Track C suite,
- the same threshold policy,
- and ideally one calibration sweep rather than raw `0.5` only.

Why this is first:

- the project already has multiple plausible contenders;
- the reports show that “winner” depends too much on which lane is used;
- you should not launch another training round blind if one of the later real-boost branches is already the best tradeoff.

## Rank 2: Split the merged source by truth, not by convenience

Do not keep treating `visomaster_teams_enhanced` as one family.

At minimum split it into:

- true Teams companion rows,
- clean fallback rows.

Preferably split it further into:

- true Teams + original branch,
- true Teams + enhanced branch,
- clean fallback + original branch,
- clean fallback + enhanced branch.

Why this is second:

- it directly fixes the biggest hidden structural dilution;
- it makes future weight choices honest;
- it sets up a curriculum that can actually emphasize the deployment domain instead of merely naming it.

## Rank 3: Run one high-purity target-domain fine-tune

If you are allowed only one new training experiment soon, make it this one.

Use a strong initializer (`R12_G` or a later scorecard-base FT lane) and train on as pure a target-domain set as you can assemble:

- Teams real,
- Teams fake,
- actual enhanced-through-Teams captures when available,
- avoid mixing `clean_fallback` inside the supposedly Teams-enhanced lane.

This run is diagnostic as much as it is optimization:

- if it helps a lot, realism was the main blocker;
- if it helps only a little, nuisance invariance and decision logic become even more central.

## Rank 4: Run one invariance ablation with verified active keys

On a stable baseline, change only the nuisance robustness pieces:

- fix `gamma_up` key names,
- raise spatial shift / spatial augmentation to the already-proposed stronger regime,
- keep everything else matched.

Measure:

- real Teams FPR,
- lockbox real Teams FPR,
- score variance under YOLO operating-point changes,
- fake recall on the relevant enhanced slices.

Why this is fourth:

- it is low-confound;
- it attacks a clear known weakness;
- it avoids the “maybe the transform never ran” ambiguity.

## Rank 5: Add a deployment-side calibration and temporal decision sweep

This can happen in parallel with training work.

Compare on frozen checkpoints:

- raw score at `0.5`,
- calibrated score,
- calibrated + short-window EMA/median,
- calibrated + hysteresis,
- calibrated + perturbation-disagreement gate.

Why this matters:

- your business objective is low FP;
- the literature and repo history both say raw probabilities are not portable;
- this is one of the few levers that can reduce false positives without waiting for new model training.

## What I Would Explicitly Not Do Next

### 1. Do not spend the next round on more small family-weight tweaks

The sampler semantics make that low probability unless the source structure changes first.

### 2. Do not trust the current merged source name

Until it is split by status, it is too easy to overestimate true Teams coverage.

### 3. Do not re-promote generic stability lambda as the main new story

The repo already explored that lane enough to know it is not the highest-confidence bet.

### 4. Do not sink time into a universal Teams simulator before the real-data ablation

Track B already showed the old simulator was too coarse, and the literature does not support synthetic degradation as a full substitute for the deployment domain.

### 5. Do not make major architecture changes before fixing data truth and selection truth

The reports do not justify “bigger model first” as the next best move. Capacity can matter, but the stronger immediate blockers are structural and evaluative.

## If Only One Training Experiment Is Allowed

Run the **high-purity target-domain fine-tune**.

Why this one:

- it directly tests the strongest current hypothesis;
- it avoids the current merged-source artifact;
- it gives the cleanest read on whether more real enhanced-through-Teams data is the real unlock.

## If Only One Non-Training Project Is Allowed

Run the **frozen checkpoint calibration + temporal decision sweep** on the shortlist.

Why this one:

- it could lower false positives fastest;
- it gives cleaner selection pressure for the next training round;
- it may reveal that one existing branch is already much more viable than the current narrative suggests.

## Honest Bottom Line

The project is not one good loss term away from being solved.

It is closer to this:

- the current B/16 system is good enough to keep pushing;
- but the next gain will come from **making the target-domain truth explicit** and **making the selection logic honest**;
- only after that will additional weighting, curriculum, or capacity changes be interpretable.

The strongest technical belief coming out of this research round is:

**The current main bottleneck is not lack of ambition. It is hidden structural dilution of the exact target condition.**
