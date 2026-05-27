# R13 Relaunch Packet 1 Live Monitoring Record - 2026-04-20

## Scope

This file is the living monitoring record for the live W&B project:

- `dtect-vision/phase2r13-rlp1-overnight-20260420`

This update was refreshed from the live runs on:

- `2026-04-20 22:58:52 CEST`

It is meant to be updated again as the packet progresses. The reference packet
intent still lives in:

- `DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_MONITORING_HANDOFF_2026-04-20.md`

## Freshness And Limits

- `4` FT runs are now `finished`: `01`, `02`, `04`, and `07`
- `3` FT runs are still marked `running` even though they have passed the
  nominal `10000` FT budget: `03` at `10682`, `05` at `10499`, and `06` at
  `10499`
- the scratch hedge is still `running` at `10567 / 30000` (`35.2%`)
- every arm now has a written `best_ood_composite` checkpoint summary, so the
  current review can use best-checkpoint metrics rather than only early
  snapshots
- some finished FT runs stopped at `9000` or `9500`, so late-packet review
  must use best-checkpoint metrics, not assume a clean `10000` terminal eval
- conclusions below are much stronger than the mid-packet read, but are still
  not fully final until the remaining live arms close cleanly

This matters because the packet was explicitly designed to care about the
target-domain direction, not just early in-training holdout behavior.

## Late-Packet Review (~23:00 CEST / 4 FT Runs Finished)

- **Observed fact:** `RLP1_01` finished and remains the packet leader.
  Best OOD-composite is `0.98915` at step `6000`
  (`holdout 0.99628`, `OOD 0.98213`), and its latest terminal-ish snapshot is
  still excellent (`holdout 0.99540`, `OOD 0.98183`, `composite 0.98857`).
- **Inference:** nothing in the packet has displaced the clean honest no-hints
  FT baseline on the main combined metric.
- **Recommendation:** `RLP1_01` stays the burden-of-proof control.

- **Observed fact:** `RLP1_04` finished as the best proper-data arm and the
  only serious challenger to `RLP1_01`. Best OOD-composite is `0.98773` at
  step `8000` (`holdout 0.98868`, `OOD 0.98677`), only `0.00143` behind the
  packet leader. Relative to `RLP1_01`, it gives up `0.00760` holdout but buys
  `0.00464` OOD.
- **Inference:** unenhanced proper-data is the only add-on that is buying real
  target-domain lift without breaking the overall packet balance.
- **Recommendation:** `RLP1_04` remains the main proper-data reference arm for
  the next packet discussion.

- **Observed fact:** `RLP1_02` finished at `9500` and still failed to justify
  the baseline hints. Best OOD-composite is `0.98524` at step `9000`, which is
  `0.00391` behind `RLP1_01`. Versus `RLP1_01`, the hints bought only
  `+0.00085` OOD while costing `-0.00877` holdout at their respective best
  composite checkpoints.
- **Inference:** the retained hints remain a net drag, not a hidden win.
- **Recommendation:** keep the no-hints baseline as the default FT control.

- **Observed fact:** `RLP1_07` finished at `9000` and did not rescue the
  full-proper sidecar story. Best OOD-composite is `0.98500` at step `7000`,
  and the run softened to `0.98470` by its latest snapshot. It still trails
  `RLP1_05`'s current best composite of `0.98542`.
- **Inference:** the Teams-shadow sidecar adds complexity without producing a
  durable packet-level gain.
- **Recommendation:** do not prioritize sidecars unless a later frozen
  scorecard clearly rescues them.

- **Observed fact:** among the remaining live FT arms, `RLP1_05` is the
  strongest full-proper run. Its best OOD-composite is `0.98542` at step
  `8000`, ahead of `RLP1_06` (`0.98492`) and `RLP1_07` (`0.98500`), but still
  `0.00231` behind `RLP1_04`. Relative to `RLP1_04`, the full-proper packet
  buys `+0.00168` OOD but pays `-0.00628` holdout.
- **Inference:** the full-proper family is getting real OOD lift, but the
  holdout tax is still too large.
- **Recommendation:** if the full-proper family survives, `RLP1_05` is the
  mainline reference, not its sidecars.

- **Observed fact:** `RLP1_06` has improved gradually into step `10000` and is
  currently near its best checkpoint, but its best OOD-composite is still
  `0.00050` below `RLP1_05`.
- **Inference:** truthful GammaUp looks like, at most, a tiny hedge rather than
  a meaningful packet shift.
- **Recommendation:** do not treat `RLP1_06` as a distinct win unless an
  external scorecard later makes that tiny gap matter.

- **Observed fact:** `RLP1_03` has now passed `10000` and is basically flat
  since step `7000`. Best OOD-composite is `0.98442` at step `7000`; latest is
  `0.98410`.
- **Inference:** the extra Teams-hints slice still does not repair the weak
  hint ladder.
- **Recommendation:** there is still no evidence-based case to expand the hint
  family.

- **Observed fact:** `RLP1_08` scratch produced the packet's highest best OOD
  AUC (`0.99167` at step `8000`) but only `0.97182` holdout at that same
  checkpoint, yielding a best OOD-composite of only `0.98164`. Its latest
  composite has fallen further to `0.97399`.
- **Inference:** scratch is interesting as an OOD-sensitive hedge, but it is
  not competitive on the packet's actual balance metric.
- **Recommendation:** keep it alive if budget allows, but not as a current
  promotion candidate.

## Best-Checkpoint Ranking

This is the main ranking that matters right now. It uses each arm's recorded
`best_ood_composite` checkpoint rather than the latest raw step.

| Slot | State | Best step | Best holdout AUC | Best OOD AUC | Best OOD composite | Gap to `01` | Review read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `01` | `finished` | `6000` | `0.99628` | `0.98213` | `0.98915` | `0.00000` | clear overall winner |
| `04` | `finished` | `8000` | `0.98868` | `0.98677` | `0.98773` | `0.00143` | best proper-data arm |
| `05` | `running` | `8000` | `0.98240` | `0.98845` | `0.98542` | `0.00374` | strongest full-proper mainline |
| `02` | `finished` | `9000` | `0.98751` | `0.98298` | `0.98524` | `0.00391` | hints still negative |
| `07` | `finished` | `7000` | `0.98179` | `0.98824` | `0.98500` | `0.00415` | sidecar not justified |
| `06` | `running` | `10000` | `0.98191` | `0.98795` | `0.98492` | `0.00423` | tiny GammaUp hedge only |
| `03` | `running` | `7000` | `0.98515` | `0.98369` | `0.98442` | `0.00474` | Teams hints still mixed/weak |
| `08` | `running` | `8000` | `0.97182` | `0.99167` | `0.98164` | `0.00751` | OOD spike, poor overall balance |

## Late-Packet Shape

- `RLP1_01`: peaked early at step `6000`, then softened slightly while still
  remaining the strongest run in the packet.
- `RLP1_02`: improved into the late run and effectively plateaued around step
  `9000`, but never threatened `RLP1_01`.
- `RLP1_04`: improved through step `8000` and then basically held its gain
  into the finish.
- `RLP1_07`: peaked at step `7000` and then softened, which weakens the case
  that the Teams-shadow sidecar is giving a robust lift.
- `RLP1_03`: peaked at step `7000` and has been flat-to-slightly-soft since.
- `RLP1_05`: improved through step `8000`, then faded modestly; still the best
  current full-proper mainline.
- `RLP1_06`: ground upward slowly into step `10000`, but not by enough to pass
  `RLP1_05`.
- `RLP1_08`: highly volatile; it spiked at step `8000`, then gave back a large
  part of that gain by the current read.

## Current Operational Read

- no summary-level collapse signal is visible: `train/collapse/is_constant_output = 0`
  and `train/params_with_grad = 145` across the packet where logged
- every arm now has a written `best_ood_composite/gcs_path`, so checkpoint
  capture is functioning
- the W&B / launcher stop boundary is not perfectly aligned: some FT runs are
  `finished` at `9000` or `9500`, while `03/05/06` are still `running` beyond
  `10000`; interpret the packet by best checkpoint metrics, not nominal step
  budget alone
- the two major interpretation caveats are unchanged:
  larger-than-documented proper-data counts in the live packet, and the legacy
  global-shuffle identity split that makes packet-1 comparisons directional
  rather than perfectly frozen A/Bs

## Packet Idea Refresher

The packet is testing three things in order:

1. which data packet is strongest under the honest relaunch contract
2. whether one light WT-C nuisance sidecar helps the strongest full-proper FT arm
3. whether a scratch hedge is worth keeping alive

The eight runs are:

- `RLP1_01`: clean honest FT baseline, no hints, no proper-data
- `RLP1_02`: `RLP1_01` plus `480` baseline hints
- `RLP1_03`: `RLP1_02` plus `202` Teams-played hints
- `RLP1_04`: `RLP1_03` plus unenhanced proper-data; live startup showed `684`
  proper fake rows (`342 clean + 342 teams`)
- `RLP1_05`: `RLP1_03` plus full proper-data; live startup showed `3186`
  proper fake rows (`342 + 1251 + 1251 + 342`)
- `RLP1_06`: `RLP1_05` plus truthful GammaUp
- `RLP1_07`: `RLP1_05` plus light Teams-shadow augmentation
- `RLP1_08`: scratch schedule on the same live data packet as `RLP1_05`

So the intended comparison chain is still:

- `01 -> 02 -> 03 -> 04 -> 05`
- then `05 -> 06`, `05 -> 07`, and `05 -> 08`

## Mid-Packet Conclusion (~7k / first two OOD reads)

- **Observed fact:** `RLP1_01` is still the clear best overall balance run.
  Current holdout AUC is `0.99446`, latest OOD AUC is `0.98213`, and latest
  OOD-composite is `0.98915`.
- **Inference:** the clean no-hints baseline remains the burden-of-proof winner
  even after OOD has arrived.
- **Recommendation:** treat `RLP1_01` as the current packet leader, not just
  the early-stage leader.

- **Observed fact:** `RLP1_04` is still the best balanced proper-data arm.
  Current holdout AUC is `0.98755`, latest OOD AUC is `0.98618`, and latest
  composite is `0.98731`.
- **Inference:** unenhanced proper-data remains the healthiest proper-data
  intervention overall.
- **Recommendation:** keep `RLP1_04` as the main proper-data reference arm for
  packet-2 decisions.

- **Observed fact:** the full-proper family (`05/06/07`) looks better on OOD
  than it did on early holdout alone, but not enough to win the packet.
  Latest OOD AUCs are `0.98678`, `0.98725`, and `0.98643`, but latest
  composites are only `0.98479`, `0.98449`, and `0.98430`, all still below
  `RLP1_04` and well below `RLP1_01`.
- **Inference:** OOD softens the earlier skepticism about full proper-data, but
  it does not reverse it.
- **Recommendation:** do not treat full proper-data as a clear promotion win
  yet; it still looks like a tradeoff between stronger OOD and weaker holdout.

- **Observed fact:** the hint ladder is still weak. `RLP1_02` currently has
  holdout AUC `0.98753`, OOD AUC `0.98093`, composite `0.98372`. `RLP1_03`
  has holdout AUC `0.98515`, OOD AUC `0.98218`, composite `0.98320`.
- **Inference:** Teams hints may help OOD a bit relative to baseline hints, but
  that gain is not large enough to repay the holdout damage.
- **Recommendation:** keep the hint family on a short leash. It still is not
  making a strong case for expansion.

- **Observed fact:** the WT-C sidecars are not producing a clear win. `RLP1_05`
  currently has the best composite inside the full-proper trio (`0.98479`);
  `RLP1_06` has the best OOD AUC (`0.98725`) but weaker holdout
  (`0.98202`); `RLP1_07` no longer leads the trio on the main summary metrics.
- **Inference:** neither GammaUp nor Teams shadow is earning extra complexity
  yet.
- **Recommendation:** if one sidecar survives, it should be because later OOD
  or frozen scorecard evidence clearly rescues it, not because of this mid-run
  snapshot.

- **Observed fact:** `RLP1_08` is still clearly behind the FT family on
  holdout (`0.97035`) and composite (`0.97126`), but its latest OOD AUC
  (`0.98264`) is not dead.
- **Inference:** scratch is alive, but it is not yet a serious challenger.
- **Recommendation:** keep the scratch hedge running, but do not let it distort
  packet-level decisions at this stage.

## Current Status Snapshot

Latest holdout / in-dist numbers are from the `6500` to `7000` window. Latest
OOD / composite numbers are still from the `6000` evaluation.

| Slot | Step | Progress | Val in-dist AUC | Val holdout AUC | Latest OOD AUC | Latest OOD composite | Current read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `01` | `7000` | `70.0%` | `0.98592` | `0.99446` | `0.98213` | `0.98915` | best overall balance |
| `02` | `7000` | `70.0%` | `0.99026` | `0.98753` | `0.98093` | `0.98372` | hints-only still weak |
| `03` | `7000` | `70.0%` | `0.99143` | `0.98515` | `0.98218` | `0.98320` | Teams hints still mixed |
| `04` | `7000` | `70.0%` | `0.98443` | `0.98755` | `0.98618` | `0.98731` | best proper-data balance |
| `05` | `7000` | `70.0%` | `0.98696` | `0.98290` | `0.98678` | `0.98479` | full proper best mainline |
| `06` | `7000` | `70.0%` | `0.98696` | `0.98202` | `0.98725` | `0.98449` | best OOD, weaker holdout |
| `07` | `7000` | `70.0%` | `0.98701` | `0.98179` | `0.98643` | `0.98430` | no clear sidecar gain |
| `08` | `7000` | `23.3%` | `0.97775` | `0.97035` | `0.98264` | `0.97126` | scratch alive, still behind |

## Path Through The First Two OOD Gates

This table is the most important path view so far. It shows whether first OOD
readout at `5000` was confirmed or weakened at `6000`, then where current
holdout ended up by about `7000`.

| Slot | Holdout @5k | Holdout @6k | Holdout @current | OOD @5k | OOD @6k | Composite @5k | Composite @6k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `01` | `0.99611` | `0.99628` | `0.99446` | `0.98036` | `0.98213` | `0.98817` | `0.98915` |
| `02` | `0.98716` | `0.98652` | `0.98753` | `0.98106` | `0.98093` | `0.98410` | `0.98372` |
| `03` | `0.98380` | `0.98422` | `0.98515` | `0.98370` | `0.98218` | `0.98375` | `0.98320` |
| `04` | `0.98569` | `0.98844` | `0.98755` | `0.98503` | `0.98618` | `0.98536` | `0.98731` |
| `05` | `0.98228` | `0.98282` | `0.98290` | `0.98743` | `0.98678` | `0.98485` | `0.98479` |
| `06` | `0.98116` | `0.98174` | `0.98202` | `0.98757` | `0.98725` | `0.98435` | `0.98449` |
| `07` | `0.98239` | `0.98218` | `0.98179` | `0.98669` | `0.98643` | `0.98454` | `0.98430` |
| `08` | `0.96791` | `0.96014` | `0.97035` | `0.98268` | `0.98264` | `0.97524` | `0.97126` |

## Current Comparison Read

### Best current holdout

- `01`: `0.99446`
- `04`: `0.98755`
- `02`: `0.98753`

### Best current OOD

- `06`: `0.98725`
- `05`: `0.98678`
- `07`: `0.98643`
- `04`: `0.98618`

### Best current composite

- `01`: `0.98915`
- `04`: `0.98731`
- `05`: `0.98479`

The important critical read is:

- `RLP1_01` is still the packet leader
- `RLP1_04` is still the best proper-data compromise
- full proper-data is getting real OOD benefit, but not enough overall
- `RLP1_05` now looks better than its sidecars on the main balance metric
- the hint family is still not convincing

Everything below this point is the earlier `08:08 CEST` pre-OOD snapshot,
retained for audit and historical comparison.

## Early Packet Conclusion

- **Observed fact:** `RLP1_01` is the strongest run by a wide margin so far. It is the best current holdout-AUC run at `0.99635`, and it beat `RLP1_02` at every matched evaluation by `0.0077` to `0.0115` AUC.
- **Inference:** the clean no-hints baseline is much stronger than the packet design implicitly expected.
- **Recommendation:** treat `RLP1_01` as the burden-of-proof baseline. Every add-on family now has to beat it, not merely look plausible.

- **Observed fact:** the hint ladder is weak so far. `RLP1_02` trails `RLP1_01` at every matched eval, and `RLP1_03` trails `RLP1_02` at 7 of 8 matched evals.
- **Inference:** the retained hint packet is not helping the current holdout objective, and the extra Teams-played hint slice looks worse, not better.
- **Recommendation:** unless first OOD readouts clearly reverse this story, do not expand the hint-heavy family in packet 2.

- **Observed fact:** `RLP1_04` is the best proper-data arm so far. It beat `RLP1_03` at every matched holdout checkpoint and is currently the second-best run overall.
- **Inference:** explicit unenhanced proper-data looks materially healthier than extra weak-signal hints.
- **Recommendation:** keep `RLP1_04` as the main proper-data reference arm for the next readout.

- **Observed fact:** `RLP1_05` underperforms `RLP1_04` at every matched holdout checkpoint so far. `RLP1_06` and `RLP1_07` recover only small fractions of that gap.
- **Inference:** the full proper-data snapshot, especially the enhanced portion, is not earning its extra complexity yet.
- **Recommendation:** if OOD does not rescue this family, narrow or rebalance the enhanced proper-data lanes rather than assuming "more proper-data" is automatically better.

- **Observed fact:** `RLP1_08` is still far behind the FT family, but its curve is steeply upward from a cold start: holdout AUC `0.5467 -> 0.9531` by step `3000`, then a small slip to `0.9497` at step `3500`.
- **Inference:** scratch is alive, but it is nowhere near mature enough to judge against the FT family yet.
- **Recommendation:** do not kill the scratch hedge early, but do not let its current numbers distort packet-level conclusions either.

## Critical Startup Findings

### 1. No obvious dead run or collapse signal

- all 8 runs are alive in W&B
- expected source families are present where expected
- no run shows a summary-level missing-lane failure
- latest training-health summaries show `train/collapse/is_constant_output = 0` and `train/params_with_grad = 145` across the packet

So far, the packet looks like a real comparison, not a debugging failure.

### 2. The live proper-data packet is larger than the handoff says

This is the biggest monitoring finding so far.

The packet docs describe the retained proper fake counts as:

- unenhanced proper fake rows: `442` = `221 + 221`
- full proper fake rows: `2412` = `221 + 985 + 985 + 221`

The live runs are not using those counts.

From live W&B discovery summaries:

| Arm | Planned proper fake rows | Live proper fake rows | Live lane breakdown |
| --- | ---: | ---: | --- |
| `RLP1_04` | `442` | `684` | `342 clean + 342 teams` |
| `RLP1_05/06/07/08` | `2412` | `3186` | `342 clean + 1251 enhanced clean + 1251 enhanced teams + 342 teams` |

This also changes the live total packet sizes:

| Arm | Planned total rows | Live total rows |
| --- | ---: | ---: |
| `RLP1_04` | `9036` | `9278` |
| `RLP1_05/06/07/08` | `11006` | `11780` |

- **Observed fact:** the live proper-data runs are materially larger than the written handoff and experiment-plan counts.
- **Inference:** the `03 -> 04` and especially `04 -> 05` comparisons are bigger interventions than documented.
- **Recommendation:** use the live counts above for all future monitoring and packet-2 planning. The packet docs need a count correction.

The current in-repo proper-data artifacts support the live story more than the
handoff story. The live training packet is clearly not running on the smaller
`221 / 985` lane counts described in the written packet docs.

### 3. Packet-1 arm comparisons are directionally useful, but not a perfectly frozen A/B ladder

- the current live packet used the legacy global-shuffle identity split
- that means later arms can move some existing identities between
  `train` / `val` / `test` when the arm adds new identities
- so `01 -> 05` still teaches something real, but it is not a perfectly frozen
  shared holdout slice in the strictest sense
- future reruns and packet-2 comparisons should use
  `combined_paired.identity_split_mode: "hash_stable"` so packet growth does
  not move the evaluation slice underneath the comparison

## Status Snapshot

| Slot | Run | Step | Progress | Val in-dist AUC | Val holdout AUC | Holdout F1 at in-dist thresh | Current read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `01` | `R13_RLP1_01_FT_WTB1_no_hints_live_0419-2241` / `40xok4cb` | `4471` | `44.7%` | `0.98598` | `0.99635` | `0.98008` | clear packet leader |
| `02` | `R13_RLP1_02_FT_WTB2_hints_only_live_0419-2242` / `vj7b4z21` | `4051` | `40.5%` | `0.98937` | `0.98725` | `0.95544` | hints-only is not paying |
| `03` | `R13_RLP1_03_FT_WTB3_hints_plus_teams_hints_live_0419-2241` / `dxrt0cjs` | `4088` | `40.9%` | `0.99054` | `0.98258` | `0.94326` | extra Teams hints look worse |
| `04` | `R13_RLP1_04_FT_WTB3_plus_proper_unenhanced_live_0419-2242` / `omqszujc` | `4036` | `40.4%` | `0.98286` | `0.98752` | `0.95682` | best proper-data arm so far |
| `05` | `R13_RLP1_05_FT_WTB3_plus_proper_full_live_0419-2241` / `qbwyyen0` | `3999` | `40.0%` | `0.98671` | `0.98077` | `0.95389` | full proper-data is underwhelming |
| `06` | `R13_RLP1_06_FT_WTB3_plus_proper_full_truthful_gammaup_0419-2242` / `djyg3dfb` | `3999` | `40.0%` | `0.98674` | `0.98105` | `0.95527` | tiny lift vs `05`, not decisive |
| `07` | `R13_RLP1_07_FT_WTB3_plus_proper_full_teams_shadow_0419-2242` / `pdduz69x` | `4000` | `40.0%` | `0.98711` | `0.98157` | `0.95827` | best of the `05/06/07` trio |
| `08` | `R13_RLP1_08_SCRATCH_WTB3_plus_proper_full_live_0419-2242` / `wofj4hlp` | `4000` | `13.3%` | `0.96342` | `0.94969` | `0.88365` | rising fast, still clearly behind |

## Important Shape In The Current Metrics

The current in-dist versus holdout split is already telling a useful story.

- **Observed fact:** `RLP1_03` has the strongest current `val_in_dist` AUC in the FT family (`0.99054`) but a much weaker current holdout AUC (`0.98258`).
- **Inference:** Teams-hint-heavy packets may be learning something that helps the easier in-dist slice without helping the harder holdout slice.
- **Recommendation:** do not let in-dist wins rescue `RLP1_03` if holdout and later OOD keep disagreeing.

- **Observed fact:** `RLP1_04` has lower in-dist AUC than `RLP1_02`, `RLP1_03`, `RLP1_05`, `RLP1_06`, and `RLP1_07`, but stronger holdout than all of them.
- **Inference:** unenhanced proper-data may be more target-aligned even if it does not maximize the easier validation slice.
- **Recommendation:** keep tracking the holdout-versus-OOD relationship for `RLP1_04`; it looks more interesting than its in-dist number alone suggests.

## Comparison Chain

### `01` vs `02`

- **Observed fact:** `RLP1_01` beat `RLP1_02` at all 8 matched holdout checkpoints. The gap ranged from `0.0077` to `0.0115` AUC.
- **Inference:** the baseline hint packet is a net negative so far.
- **Recommendation:** the default packet should stay no-hints unless later OOD evidence proves a hidden benefit.

### `02` vs `03`

- **Observed fact:** `RLP1_03` beat `RLP1_02` only once in 8 matched holdout checkpoints, and it currently trails by `0.00468` AUC. Its holdout-at-in-dist-threshold F1 is also worse (`0.9433` vs `0.9554`).
- **Inference:** the extra Teams-played hint slice is not improving the packet on the current holdout objective.
- **Recommendation:** do not expand the Teams-hints path unless OOD clearly says the holdout read is misleading.

### `03` vs `04`

- **Observed fact:** `RLP1_04` beat `RLP1_03` at all 8 matched holdout checkpoints by `0.0017` to `0.0070` AUC.
- **Inference:** explicit unenhanced proper-data is more useful than more weak-signal hints.
- **Recommendation:** keep `RLP1_04` as the main packet-2 proper-data anchor.

### `04` vs `05`

- **Observed fact:** `RLP1_05` lost to `RLP1_04` at every matched holdout checkpoint so far by `0.0063` to `0.0140` AUC.
- **Inference:** the full proper-data snapshot is not helping early holdout behavior, and the enhanced portion is the main suspect.
- **Recommendation:** if OOD does not reverse this, narrow or rebalance the enhanced proper-data lanes before giving this family more budget.

This comparison should be read with one extra caution:

- `RLP1_05` is not merely "the planned full proper snapshot"
- it is a **larger-than-documented** full proper snapshot
- so the current underperformance is attached to the live larger packet, not to the smaller packet described in the handoff

### `05` vs `06`

- **Observed fact:** `RLP1_06` beat `RLP1_05` on 5 of 7 matched holdout checkpoints, but the margins are tiny and its EER path is mixed.
- **Inference:** truthful GammaUp may be a small hedge, not a meaningful packet shift.
- **Recommendation:** keep it alive through first OOD, but do not overread the current advantage.

### `05` vs `07`

- **Observed fact:** `RLP1_07` is currently the strongest member of the `05/06/07` family on both holdout AUC and holdout-at-in-dist-threshold F1, but its checkpoint-by-checkpoint edge over `RLP1_05` is inconsistent.
- **Inference:** Teams shadow is slightly more interesting than GammaUp right now, but still not convincingly so.
- **Recommendation:** if one WT-C-style sidecar survives to the next discussion, `RLP1_07` has the best current case, but it is not a confirmed winner.

### `05` vs `08`

- **Observed fact:** `RLP1_08` trails `RLP1_05` at every matched holdout checkpoint so far, but its curve rose dramatically from `0.5467` at step `500` to `0.9531` at step `3000`.
- **Inference:** scratch is behaving like a still-forming run, not a dead run.
- **Recommendation:** defer serious judgment on scratch until at least the first OOD pass and a later FT-versus-scratch comparison window.

## Per-Run Notes

### `RLP1_01`

- **Run:** `R13_RLP1_01_FT_WTB1_no_hints_live_0419-2241` / `40xok4cb`
- **Hypothesis:** the clean honest no-hints baseline should be the control, not necessarily the winner.
- **Schedule facts:** FT run, `10000` steps, first OOD at `5000`.
- **Startup sanity:** counts matched the written non-proper baseline exactly: total `7912`, no hints, no proper-data.
- **Path so far:** holdout AUC has stayed in a very tight high band: `0.9942, 0.9958, 0.9942, 0.9951, 0.9958, 0.9943, 0.9943, 0.9963`.
- **Best checkpoint behavior:** best current holdout checkpoint is step `4000`, and it is also the packet best so far.
- **Interim verdict:** strongest run in the packet by a large margin.
- **What this teaches:** the honest baseline is stronger than expected, so every extra data or augmentation lane has to justify itself against a very strong control.

### `RLP1_02`

- **Run:** `R13_RLP1_02_FT_WTB2_hints_only_live_0419-2242` / `vj7b4z21`
- **Hypothesis:** adding the small retained hint slice might help over the clean baseline.
- **Schedule facts:** FT run, `10000` steps, first OOD at `5000`.
- **Startup sanity:** counts matched the written hints-only arm: total `8392`, with `480` hint rows visible in the live source summary.
- **Path so far:** the curve is stable but second-tier: `0.9841 -> 0.9873`, always below `RLP1_01`.
- **Best checkpoint behavior:** best current holdout checkpoint is step `4000`, but it still trails `RLP1_01` by about `0.0091` AUC.
- **Interim verdict:** baseline hints are not helping enough to justify themselves.
- **What this teaches:** the retained hints look more like drag than leverage on the current holdout objective.

### `RLP1_03`

- **Run:** `R13_RLP1_03_FT_WTB3_hints_plus_teams_hints_live_0419-2241` / `dxrt0cjs`
- **Hypothesis:** the extra Teams-played hint slice might improve over baseline hints.
- **Schedule facts:** FT run, `10000` steps, first OOD at `5000`.
- **Startup sanity:** live counts matched the written hint-family totals: total `8594`, with `480` baseline hints and `202` Teams hints.
- **Path so far:** early holdout rise was weak, peak came at step `2000`, and the later path drifted back down to `0.9826`.
- **Best checkpoint behavior:** best holdout checkpoint is only `0.9843`, which is still below `RLP1_02` at most matched steps.
- **Interim verdict:** extra Teams hints look actively unhelpful so far.
- **What this teaches:** the current Teams-hint residue may improve easier in-dist metrics without improving the harder holdout slice that matters more.

### `RLP1_04`

- **Run:** `R13_RLP1_04_FT_WTB3_plus_proper_unenhanced_live_0419-2242` / `omqszujc`
- **Hypothesis:** explicit unenhanced proper-data should beat the hints-only packet.
- **Schedule facts:** FT run, `10000` steps, first OOD at `5000`.
- **Startup sanity:** this arm loaded the correct lane families, but not the documented packet size. Live proper-data discovery is `684`, not the planned `442`.
- **Path so far:** holdout AUC improved at every broad stage except a modest dip at `2500`, then reached `0.9880` at `3500` before a tiny slip to `0.9875` at `4000`.
- **Best checkpoint behavior:** current best checkpoint is step `3500`, and it is the best non-baseline FT result in the packet.
- **Interim verdict:** best proper-data arm so far; strongest challenger to `RLP1_01`.
- **What this teaches:** explicit proper-data looks useful, but the live ladder step is larger than planned, so downstream interpretation must use live counts.

### `RLP1_05`

- **Run:** `R13_RLP1_05_FT_WTB3_plus_proper_full_live_0419-2241` / `qbwyyen0`
- **Hypothesis:** the full proper-data snapshot should improve on the unenhanced proper-data arm.
- **Schedule facts:** FT run, `10000` steps, first OOD at `5000`.
- **Startup sanity:** live proper-data discovery is `3186`, not the planned `2412`. This is a materially larger intervention than the packet docs say.
- **Path so far:** the holdout curve is steadily upward from `0.9716` to `0.9808`, but it is below `RLP1_04` at every matched checkpoint.
- **Best checkpoint behavior:** best current holdout checkpoint is step `3500`; there is no sign yet that it is closing the gap to `RLP1_04`.
- **Interim verdict:** the full proper-data snapshot is underperforming.
- **What this teaches:** enhanced proper-data is not automatically beneficial; right now it looks like dilution, imbalance, noise, or all three.

### `RLP1_06`

- **Run:** `R13_RLP1_06_FT_WTB3_plus_proper_full_truthful_gammaup_0419-2242` / `djyg3dfb`
- **Hypothesis:** truthful GammaUp might help the strongest full-proper FT packet.
- **Schedule facts:** FT run, `10000` steps, first OOD at `5000`.
- **Startup sanity:** same live data packet size as `RLP1_05`; no sign of a missing family.
- **Path so far:** slightly better than `RLP1_05` on most holdout checkpoints, but the edge is small and not cleanly monotonic.
- **Best checkpoint behavior:** best current holdout checkpoint is `0.9810` at `3500`.
- **Interim verdict:** small possible lift over `RLP1_05`, not enough to call a real win yet.
- **What this teaches:** GammaUp may be a nuisance hedge worth watching, but it is nowhere close to a decisive packet change.

### `RLP1_07`

- **Run:** `R13_RLP1_07_FT_WTB3_plus_proper_full_teams_shadow_0419-2242` / `pdduz69x`
- **Hypothesis:** a light Teams-shadow sidecar might help the strongest full-proper FT packet.
- **Schedule facts:** FT run, `10000` steps, first OOD at `5000`.
- **Startup sanity:** same live data packet size as `RLP1_05`; no sign of a missing family.
- **Path so far:** mixed versus `RLP1_05`, but it currently has the best holdout AUC and best holdout-at-in-dist-threshold F1 inside the `05/06/07` trio.
- **Best checkpoint behavior:** best current holdout checkpoint is `0.9816` at `3500`.
- **Interim verdict:** current sidecar front-runner, but still not close to `RLP1_04`.
- **What this teaches:** if one light nuisance overlay survives to the next packet discussion, this is the one with the best current claim.

### `RLP1_08`

- **Run:** `R13_RLP1_08_SCRATCH_WTB3_plus_proper_full_live_0419-2242` / `wofj4hlp`
- **Hypothesis:** scratch on the strongest expected packet might outperform the FT family once it has enough time.
- **Schedule facts:** scratch run, `30000` steps, first OOD at `5000`.
- **Startup sanity:** same live data packet size as `RLP1_05`; no sign of loader collapse.
- **Path so far:** holdout AUC rose from `0.5467` to `0.8604` to `0.9332` to `0.9531`, then dipped to `0.9497` at the latest checkpoint.
- **Best checkpoint behavior:** best current holdout checkpoint is step `3000`, not the latest step.
- **Interim verdict:** clearly behind all FT arms, but clearly still learning.
- **What this teaches:** scratch cannot be compared honestly to FT yet; the only honest next check is later in training and after OOD begins.

## What Looks Most Important To Watch Next

1. First OOD readout at FT step `5000`.
2. Whether `RLP1_01` still holds its lead once OOD and OOD-composite exist.
3. Whether `RLP1_04` keeps beating the full proper-data family on the more target-oriented metrics.
4. Whether `RLP1_07` keeps a sidecar edge over `RLP1_05` and `RLP1_06`, or whether that lead disappears as variance.
5. Whether `RLP1_08` keeps climbing after step `4000`, or flattens into a permanently non-competitive scratch hedge.

## Interim Packet Lean

If a forced pre-OOD read had to be made right now:

- strongest overall arm: `RLP1_01`
- strongest proper-data arm: `RLP1_04`
- strongest full-proper sidecar arm: `RLP1_07`, but only narrowly
- weakest family claim: extra retained hints, especially the added Teams-hint slice
- most important unresolved question: whether OOD reverses the holdout story for any of the proper-data-heavy runs

That is still not enough to declare a packet winner. The packet has not yet
reached the stage it was mainly designed to answer.
