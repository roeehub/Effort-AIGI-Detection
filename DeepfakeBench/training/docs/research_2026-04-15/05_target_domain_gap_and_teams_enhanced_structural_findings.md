# Target-Domain Gap and Teams-Enhanced Structural Findings

## Scope

This report is based on:

- `docs/ENHANCED_TEAMS_GAP_REPORT.md`
- `docs/TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md`
- `docs/TRACK_A_HANDOFF_2026-04-06.md`
- `docs/TRACK_B_MATCHED_PAIR_REPORT_2026-04-07.md`
- `docs/TRACK_C_SCORECARD_RUNBOOK_2026-04-07.md`
- `docs/SPATIAL_AUGMENTATION_PROPOSAL_2026-04-14.md`
- `docs/LIGHTING_ROBUSTNESS_REPORT.md`
- `docs/R12_POST_LAUNCH_PLAN.md`
- `experiments/phase2_round12/R12_G_scratch_seed_control.yaml`
- `experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`
- `experiments/phase2_round13/R13_TB1_trackB_family_split_sidecar.yaml`
- `arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json`
- `arena/target_domain_suites.teams_manifest.breakdown_2026-04-07.yaml`
- `data/sources/visomaster.py`
- `data/sources/combined_paired.py`

The conclusion is not that Track A failed. It did solve part of the fake-side problem. The conclusion is that the current system is still structurally mis-specified for the exact deployment problem: low-FP detection of live Teams deepfakes, especially enhanced ones.

## 1. Enhanced-through-Teams failures are real, and the current training path still under-models them

The March gap report already identified the core issue: enhanced fakes passed through Teams were failing badly even though clean enhanced crops were nearly solved in training and holdout. The stated root cause was that the model learned clean enhancement cues, but Teams washed out or altered the cues it relied on.

Track B later confirmed that this is not a vague intuition. On matched pairs:

- ordinary current Teams slices were not uniformly blur-dominant
- the enhanced-through-Teams slice was materially different
- for clean enhanced VisoMaster fake -> Teams-enhanced fake, sharpness dropped `23.2%` and high-frequency energy dropped `81.8%`

That means `enhanced + Teams` is not just “enhanced data plus some generic call compression.” It is its own mode.

The strongest structural finding is in the loader itself. `load_visomaster_teams_enhanced_frames()` does this:

- real frames always come from the resolved companion bucket
- fake frames come either from the companion fake branch when `fake_branch="original"`
- or from the clean enhanced bucket when an enhancer branch is chosen

So the merged Track A source never emits a fake frame that is both:

1. enhanced, and
2. passed through Teams

For the `54` true `teams_v2` companion samples, the two possibilities are:

- Teams real + Teams original fake
- Teams real + clean enhanced fake

That second case is not the target condition. It is a cross-domain pair.

## 2. Why Track A improved fake recall but regressed real Teams FPR versus R12_G

Track C gives the cleanest readout:

| Metric | `R12_G` | `TRACK_A_CANDIDATE` |
|---|---:|---:|
| `teams_real_all_dev` FPR | `0.1980` | `0.2355` |
| `teams_real_poor_quality_dev` FPR | `0.1679` | `0.2319` |
| `teams_real_all_lockbox` FPR | `0.6334` | `0.7252` |
| `teams_fake_all_dev` recall | `0.8510` | `0.8896` |
| `visomaster_enhanced_macro_dev` recall | `0.4527` | `0.7400` |
| `deeplive_enhanced_dev` recall | `0.9706` | `0.9138` |

This is a real tradeoff, not noise. The real-side regression is visible on both dev and lockbox. The fake-side gain is large, but concentrated.

What most likely happened:

- Track A added strong fake-side pressure exactly where R12_G was weak: `visomaster_enhanced_fake` got explicit weight `3.5`, and the merged source injected many more enhanced-family exposures.
- The gain was narrow rather than universal. The scorecard explicitly says `teams_capture_cam_test_dev`, `teams_capture_pc_generator_dev`, and `teams_capture_test_cam_dev` were unchanged or effectively tied. The large fake gain came mostly from `visomaster_enhanced_macro_dev`.
- Real Teams invariance did not improve enough. Lighting robustness was already known to be weak, and the current R13 configs likely never enabled the intended `GammaUp` path because they set `gamma_up_p` / `gamma_up_range`, while the augmentation code only accepts `context_variation_gamma_up_p` / `context_variation_gamma_up_range`.
- Spatial robustness is also still weak. Current Track A keeps `context_variation_shift: 0.04` and `context_variation_individual_p: 0.15`, which is too little given the observed crop-boundary sensitivity.

There is also an important negative finding in the config diff:

- `R12_G` used `deeplive_teams_real: 4.0`, `realpool_real: 2.0`
- `R13_A` kept `deeplive_teams_real: 4.0` and raised `realpool_real` to `2.5`

So the real Teams FPR regression did not happen because reals were simply starved. Real weighting was not reduced. The more likely cause is that the fake side became sharper on the known enhanced gap while the real Teams nuisance factors remained structurally under-covered.

## 3. What Track B taught, and what it did not teach

Track B taught three important things:

- The old single blur-heavy `TeamsCodecSimulation` is not a valid general Teams model. Direction-match accuracy was only `37.5%` on `teams_v1_real`, `50.0%` on `teams_v1_fake`, `37.5%` on `teams_v2_real`, and `50.0%` on `teams_v2_fake`.
- The enhanced-through-Teams slice is genuinely different from ordinary Teams slices, and closer to the blur/HF-loss story than the ordinary buckets are.
- A family-conditional policy is more plausible than one global Teams simulator. `family_split` was the best sidecar candidate at the image-space level.

What Track B did not teach:

- It did not prove that synthetic Teams augmentation improves the actual low-FP deployment objective.
- The matched-pair study was only `186` frame pairs, and the enhanced slice covered only `3` base sample IDs.
- The training-side sidecar ablation (`R13_TB1`) regressed mixed OOD composite by `0.0081` versus Track A, and it was explicitly not yet scored on the frozen Track C suite.

So Track B successfully killed a bad assumption: “one Teams simulator will solve this.” It did not yet produce a promotion-safe policy.

## 4. What Track C taught, and what it did not teach

Track C taught:

- The Track A fake gain is real.
- The Track A real Teams FPR regression is also real.
- The April lane flipped the earlier March fine-tune tradeoff: old FT checkpoints reduced real Teams false positives but lost fake recall, while Track A improved fake recall and worsened real Teams FPR.

Track C also exposed an evaluation limitation that matters a lot:

- the scorecard uses `fake_splits: dev`
- fake lockbox exists in the frozen manifest, but the suites do not score it
- `visomaster_enhanced_macro` has `550` dev videos and `0` lockbox
- `deeplive_enhanced` has `545` dev videos and `0` lockbox

So Track C currently does not tell us whether enhanced-fake gains generalize to unseen target-domain sessions. It only tells us that the gain exists on the dev side of the frozen target-domain set.

Real-side Track C is much stronger:

- `teams_real_all` has `3253` dev and `1361` lockbox videos

So the real-FPR regression should be treated as a stronger and more reliable signal than the enhanced-fake gain, because it is supported by an actual lockbox slice. One caveat: `teams_real_poor_quality_lockbox` has only `22` videos, so that one slice alone is too small for heavy interpretation.

Track C also does not measure:

- temporal stability on near-identical adjacent frames
- crop-perturbation sensitivity
- threshold calibration for low-FP deployment

Those are all directly relevant to the Teams use case.

## 5. The current merged Teams-enhanced source is useful, but not structurally sufficient

The repo documentation is explicit:

- `999` enhanced base sample IDs were audited
- `997` were training-eligible
- only `54` resolved to true `teams_v2` companions
- `943` resolved to `clean_fallback`

That means `94.6%` of the merged source is not true Teams companion data.

This by itself would already make the source insufficient for solving the exact target problem. But the loader logic makes the mismatch sharper:

- branch selection is `p_original = 0.5`
- the original branch uses the resolved companion fake
- enhancer branches use the clean enhanced bucket

Therefore, for the `54` `teams_v2` rows:

- only about half of exposures use the actual Teams fake branch
- the other half use clean enhanced fake frames against Teams real frames

In expected terms, only about `27 / 997` merged-source exposures are true Teams fake branch exposures under the default mix. That is roughly `2.7%` of the merged-source family, and still `0%` of the exact enhanced-through-Teams fake condition.

This is the most important structural answer in this report:

**The current merged source is training-usable, but it is not a direct training source for enhanced-through-Teams fakes.**

It can improve the model by forcing it to learn enhanced fakes and by exposing some Teams-domain reals/original fakes, but it cannot fully solve the real problem because the exact fake condition is missing.

## 6. Bottleneck ranking

### 6.1 Primary bottleneck: data realism

This is the main bottleneck.

- The exact target condition is not being trained directly.
- The merged source does not emit enhanced-through-Teams fake frames.
- Track B showed that the synthetic Teams path is not yet faithful enough to stand in for the missing condition.

### 6.2 Secondary bottleneck: augmentation and invariance

This is the next most likely bottleneck.

- lighting sensitivity is a known production problem
- intended `GammaUp` likely never ran in current R13 configs
- spatial perturbation is still too weak for the observed crop sensitivity
- Track A fake gains came without real Teams invariance gains

### 6.3 Third bottleneck: quantity of true target-domain paired data

This is separate from realism.

- `54` true `teams_v2` companion IDs is not much
- enhanced target-domain evaluation has no lockbox
- ordinary Teams real lockbox is decent, but target-domain enhanced fake coverage is still shallow

### 6.4 Fourth bottleneck: weighting

Weighting matters, but it is not the root cause.

- real weights were not reduced from `R12_G` to Track A
- yet real Teams FPR worsened
- several drafted follow-on configs tweak `realboost` and `p_original`, which may improve the tradeoff, but they cannot manufacture the missing `enhanced + Teams` fake condition

### 6.5 Fifth bottleneck: decision policy

Decision policy is operationally important, especially for low FP, but it is not the main training bottleneck.

- production threshold mismatch was already documented as severe
- calibration and temporal smoothing will matter
- but calibration cannot recover fake evidence that the model never learned, and it cannot explain the large Track A gain on `visomaster_enhanced_macro_dev`

### 6.6 Lowest-evidence bottleneck right now: model capacity

Capacity may matter later, but it is not the best current explanation.

- there is prior evidence that `k=64` can help some robustness axes
- but there is no target-domain evidence here showing that capacity is the reason Track A improved recall yet regressed real Teams FPR
- the current failure looks more like domain mismatch plus weak nuisance invariance than a raw representation ceiling

## Most Likely Structural Bottlenecks Right Now

1. The model still is not trained on the exact `enhanced fake after Teams processing` condition. This is the single biggest structural miss.
2. The current merged Track A source is mostly `clean_fallback`, and even on true `teams_v2` rows it mixes in clean enhanced fake frames rather than enhanced-through-Teams fake frames.
3. Real Teams nuisance factors are still under-covered: lighting, white balance, compression mode changes, and crop geometry.
4. The current Teams simulator is not trustworthy enough to substitute for the missing real data.
5. The scorecard is strong for real Teams FPR but still weak for lockbox enhanced-fake generalization, so the fake-side win is easier to over-read than the real-side regression.
6. Weighting and threshold tuning are downstream levers. They can move the tradeoff, but they are unlikely to solve the core structural miss by themselves.

## If Only 2 Experiments Could Be Run Next

### 1. Run one high-purity target-domain fine-tune, not another mixed-source weight sweep

Use `R12_G` or the scorecard-base FT lane as the initializer, and train on:

- real Teams reals
- actual Teams-passthrough fakes
- actual enhanced-through-Teams captures if available
- no `clean_fallback` inside the supposed Teams-enhanced lane

If there are not enough real enhanced-through-Teams captures yet, do not pretend the merged source solves that gap. Keep this experiment diagnostic and honest: pure target domain, small run, score it immediately on Track C.

Why this should be one of the two:

- it directly tests whether realism is the blocker
- it avoids the current cross-domain pairing artifact
- it gives a clean answer that weighting sweeps cannot give

### 2. Run one invariance-focused ablation on the same baseline

Keep the data mix fixed and change only:

- fix `gamma_up_p` to `context_variation_gamma_up_p`
- fix `gamma_up_range` to `context_variation_gamma_up_range`
- raise spatial jitter to the already-proposed regime (`context_variation_shift: 0.08`, `context_variation_individual_p: 0.40`)

Then evaluate:

- `teams_real_all_dev`
- `teams_real_all_lockbox`
- `teams_real_poor_quality_dev`
- frame-to-frame variance on the same Teams clips under small YOLO crop changes

Why this should be the other experiment:

- it targets the exact real-side failure mode and instability you care about
- it is low-confound because it does not rely on new data assumptions
- it tests the most plausible non-data explanation before spending time on capacity or broad weight searches

What I would not spend one of the two runs on:

- another blind `p_original` sweep
- another blind real/fake weight sweep
- promoting Track B augmentation into main line
- a `k=64` target-domain run before realism and invariance are tested cleanly
