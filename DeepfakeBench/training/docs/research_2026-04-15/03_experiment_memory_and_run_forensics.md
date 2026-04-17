# Experiment Memory And Run Forensics

## Scope

This report reconstructs what actually improved the project from `R8` through the current `R13` branches, using repo docs as the primary record and a small set of WandB summary spot-checks on decisive runs only.

The main goal is not to restate round names. It is to separate:

- real lessons from one-off stories,
- generic OOD progress from Microsoft Teams/live-target progress,
- and comparable metrics from metrics that only look comparable.

## Method And Guardrails

- Primary written sources reviewed: `R8_EXPERIMENT_REPORT`, `PHASE2_SUMMARY_R8_R11`, `R95_FINAL_REPORT`, `WINNING_RUNS_REGISTRY`, `R12_EXPERIMENT_PLAN`, `R12_POST_LAUNCH_PLAN`, `VISOMASTER_ENHANCED_INTEGRATION`, `ENHANCED_TEAMS_GAP_REPORT`, `TEAMS_TARGET_DOMAIN_UPGRADE_PLAN`, `TRACK_A/B/C` docs, and `SPATIAL_AUGMENTATION_PROPOSAL_2026-04-14`.
- Targeted WandB summary checks were used only on runs that materially change the story: `R8_E`, `R9_A`, `R9_D`, `R95_B`, `R95_D`, `R10_C`, `R11_F`, `R11_G`, `R12_A`, `R12_B`, `R12_E`, `R12_G`, `R13_FT2`, `R13_A` Track A, `R13_TB1`, `R13_E`, and `R13_FT10`.
- When repo docs and WandB disagree, I treat that as a forensic finding, not something to smooth over.

## Phase Clusters

| Phase | Backbone / training mode | Data regime | Representative runs | Forensic read |
|---|---|---|---|---|
| `R8` | B16, scratch vs FT | target-heavy rebalance, no Teams training yet | `R8_E`, `R8_A`, `R8_C`, `R8_B` | Big step forward came from data composition and scratch reset, not loss tricks. Some later docs preserve an interim VCD-real high-water mark rather than the final summary. |
| `R9` / `R9.5` | B16, mostly FT from `R8_E`, one scratch probe | first Teams-aware regime, Teams v1, early codec-sim ideas, stability bug then bug-fix | `R9_A`, `R9_D`, `R9_F`, `R95_B`, `R95_D` | The accidental `lambda=0` setting was genuinely better. FT learned a Teams-aware boundary, but repeated FT also collapsed hard-method diversity. |
| `R10` / `R11` | B16 FT and scratch, plus L14 probe | Teams v2 expansion, larger target-domain fake corpus | `R10_C`, `R10_G`, `R11_F`, `R11_G`, `R11_E` | B16 Teams adaptation plateaued. L14 proved capacity matters, but B16 scratch still preserved some hard-method signal that FT lost. |
| `R12` / `R12.5` | B16 scratch and FT | compound lighting/context aug, Teams OOD added, GRL and `k=64` tested | `R12_A`, `R12_B`, `R12_E`, `R12_G` | No-GRL scratch runs became the strongest B16 family. Late convergence and checkpoint selection mattered more than most round summaries admit. `R12.5` is still a plan, not a result. |
| Early `R13` enhanced FT | B16 FT from `R12_G` | enhanced VisoMaster data added, still mostly generic training metrics | `R13_FT1/2/3` | These runs improved generic OOD/composite quickly, but did not prove live enhanced-through-Teams robustness. |
| `R13` Track A / B / later FT | B16 scratch and FT | merged Teams-enhanced source, sidecar Teams-sim redesign, real-boost follow-ons | `R13_A`, `R13_TB1`, `R13_E`, `R13_FT10` | Track A clearly helps target fake recall. Track B did not beat Track A. Later FT follow-ons look strong in training metrics, but equivalent frozen target-domain scorecard evidence is still missing. |

## Cross-Round Comparability Rules

- `R8` to `R11` OOD AUC is mostly comparable because the monitored OOD suite was the older `{youtube, VCD real, WMA fake}` set.
- `R12+` OOD AUC is **not directly comparable** to `R8-R11` OOD AUC because `teams_ood_real` and `teams_ood_fake` were added.
- `Track A` holdout metrics are **not** cleanly comparable to older rounds because the source mix and holdout population changed. The Track A docs are correct to prefer frozen target-domain scorecards, or at least the same `ood_composite` lane, over raw holdout AUC.
- Several docs mix three different objects without saying so: interim snapshots, best-checkpoint values, and end-of-run summaries. `R8` and `R12` are the worst offenders.

## What Consistently Helped

- **Data-regime changes plus scratch retraining** helped whenever the task definition changed materially. `R8_E` beat the R8 FT family on deployment-relevant real data; `R95_D` preserved hard-method signal that FT lost; the `R12` scratch family overtook the `R12` FT family; `R13_E` recovered much more useful Track-A-like behavior than the first Track A scratch baseline.
- **Adding genuinely relevant domain data** helped more than clever losses. `R8` target-heavy balancing fixed VisoMaster weakness. `R9` Teams data created the first usable Teams-aware checkpoint. `Track A` merged Teams-enhanced sourcing clearly improved fake-side target behavior on the frozen Teams scorecard, especially `visomaster_enhanced_macro_dev`.
- **OOD-aware checkpoint selection** was a real improvement, not bookkeeping. `R12_A` had its best real/OOD tradeoff early and then overfit holdout. `R12_G` looked mediocre early and only became the best B16 recipe if judged late and on the right lane. `Track A` only makes sense when read through `ood_composite` plus the frozen target-domain scorecard, not through raw holdout.
- **Simple B16 recipes beat fancy B16 recipes.** The durable B16 winners keep reusing the same pattern: no stability loss, no label smoothing, no GRL in the promoted winner, no Group DRO dependence, and strong emphasis on source mix plus augmentation coverage.
- **For Teams/live target behavior specifically, real-source weighting on the real side matters.** The best post-Track-A follow-ons checked in WandB are `R13_E` scratch real-boost and `R13_FT10` merged-source real-boost plus `p_original=0.7`, which is notable because both are explicitly trying to repair Track A's real-side regression rather than just improve fake recall.

## What Consistently Hurt Or Plateaued

- **Stability regularization and label smoothing** repeatedly hurt the metrics that matter. `R9.5` is the cleanest result in the repo: lower `lambda` was better, `lambda=0` was best, and the fix made jitter and OOD worse rather than better.
- **B16 fine-tuning for Teams adaptation plateaued** across multiple rounds. `R10_C`, `R10_G`, and the B16 `R11` FT family all landed in a narrow band: acceptable, but not a breakthrough on VCD-real, Teams-real, or facedancer. `R12` FT runs plateaued almost immediately. The exception is later `R13` FT, but that was mainly a fast way to absorb new fake-family data, not proof that FT solved the live Teams problem.
- **Synthetic Teams augmentation has not earned promotion.** The old R9-era Teams simulator was never a stable main-line answer, and the refreshed Track B sidecar (`R13_TB1`) still lost to Track A on comparable training-side OOD/composite.
- **More fake diversity without enough real-side compensation** often hurt the real-video side. This pattern shows up in the VCD/Teams-real regressions between `R8` and `R10/R11`, and again in Track A: fake recall went up, but frozen-scorecard real Teams FPR got worse.
- **B16 add-on sophistication had diminishing returns.** `k=64` was sometimes respectable but never decisive, Group DRO was flat, GRL never produced a promotion-safe overall tradeoff, and the cost of chasing these knobs was usually more narrative than gain.
- **Round `12.5` has not helped yet because it is unfinished.** The lighting ideas may be good, but there is no completed run evidence in the repo to count as an experiment-memory success.

## Findings That Are Less Certain Than They Sound

- **"B16 has a hard capacity ceiling" is directionally true, but too absolute.** `R11_G` shows larger capacity solves major problems, but later B16 runs (`R12_G`, `R13_FT10`, `R13_E`) moved well past some earlier "ceiling" narratives. The stronger claim is narrower: B16 is fragile when asked to simultaneously minimize real-side Teams false positives and retain hard fake-method coverage.
- **`R8_E`'s famous `82.1%` VCD-real number is not the clean final story.** That value appears in the mid-run R8 report; the WandB final summary for `R8_E` is lower (`78.5%`). The round was still a success, but the exact number often quoted is snapshot-dependent.
- **Several `R12` summaries are stale or selection-lane-mismatched.** Older registry text reports `R12_G` around `OOD AUC 0.9607 / composite 0.9738`, while later docs and WandB summarize roughly `0.9789 / 0.9869`. `R12_A` has the same problem. This means some "R12 vs earlier rounds" stories are comparing different checkpoints or different frozen moments.
- **"GRL didn't help" is too clean as a scientific statement.** It is fair as a promotion decision, but not as a mechanistic conclusion. `R12_E` had decent real-side summary metrics in WandB while being poor overall because it sacrificed holdout/hard-method performance. The honest lesson is: GRL never yielded a winner, not that it provided zero signal anywhere.
- **"Track A is the replacement for `R12_G`" is unsupported.** The frozen Track C scorecard shows a trade: better fake recall and much better `visomaster_enhanced_macro`, but worse real Teams FPR. That is a useful branch result, not a full promotion result.
- **"R13 FT fixed enhanced-through-Teams" is not established.** Early FT runs like `R13_FT2` looked strong on in-training OOD, including Teams-OOD summaries, but the dedicated gap report still recorded manual failure on enhanced-through-Teams frames. The repo does not yet contain the corresponding frozen scorecard evidence for the later FT7-FT10 family either.
- **"R13 used GammaUp / new lighting push" may be partly fictional.** The spatial-augmentation proposal makes a credible case that several R13 YAMLs used `gamma_up_p` rather than `context_variation_gamma_up_p`, which may mean the intended transform was silently off. Until that is verified, augmentation conclusions that depend on GammaUp being active should be treated as provisional.

## The 3 Highest-Value Unfinished Questions

1. **Which post-`R12` candidate actually wins the live Teams objective on one frozen scorecard lane?**  
   The shortlist is no longer just `R12_G` versus first Track A. It should include at least `R12_G`, `R13_E`, `R13_FT10`, and the best Track A checkpoint, all judged on the same frozen target-domain suite with the same threshold policy. Right now the repo has enough evidence to know the branches are trading off different slices, but not enough to name the winner.

2. **Is the enhanced-through-Teams gap mostly a missing-data problem or a training-regime problem?**  
   The strongest evidence currently points to missing combined-domain data. Clean enhanced training is not enough. Old Teams simulation is not enough. Track A merged sourcing helps but still hurts real-side FPR. The highest-value next experiment family is the one that directly compares real captured enhanced-through-Teams data, merged-source training, and simulator-based substitutes under the same scorecard.

3. **Do spatial and lighting augmentations reduce real Teams false positives once the configuration path is trustworthy?**  
   This remains unfinished because `R12.5` never closed the loop, and there is a plausible `gamma_up` key mismatch in `R13`. Before spending another round on weighting lore, the project needs one clean augmentation verdict with verified active transforms, frozen evaluation, and an explicit read on real-Teams FPR stability.

## Bottom Line

The strongest durable lesson is still the old one: this project improves when it gets closer to the real target domain in data and evaluation, not when it adds more training cleverness.

The strongest B16 result for generic robustness is the late `R12_G` family. The strongest evidence for target-fake improvement is the Track A family. The strongest evidence that capacity still matters is `R11_G`. The strongest evidence that the central Teams/live problem is still unsolved is that every branch that improves fake recall has, so far, either hurt real Teams FPR or lacks a frozen scorecard proving otherwise.
