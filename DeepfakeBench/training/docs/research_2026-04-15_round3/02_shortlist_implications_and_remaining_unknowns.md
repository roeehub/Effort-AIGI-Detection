# Shortlist Implications And Remaining Unknowns

## 1. What is now closed

The Track 1 sub-question "Are `FT8` and `FT10` real runtime-distinct evidence?" can move from plausible to established:

- `R13_FT8_FP32` should be treated as functionally duplicate of `R13_FT7_FP32`
- `R13_FT10_FP32` should be treated as functionally duplicate of `R13_FT9_FP32`

That means the reduced shortlist from Round 2 is now strengthened by remote-runtime evidence, not just by local-viewer inference.

Keep:

- `R12_G_FP32`
- `R13_A_STEP15500`
- `R13_E_BESTSOFAR`
- `R13_FT7_FP32`
- `R13_FT9_FP32`

Do not spend new promotion-measurement bandwidth on:

- `R13_FT8_FP32`
- `R13_FT10_FP32`

## 2. What remains open after this package

- No accessible calibrated target-domain shortlist scorecard was found locally in this runtime.
- No accessible raw per-video shortlist prediction dump was found locally either.
- Fixed threshold `0.5` remains an invalid promotion contract.
- `teams_fake_all_lockbox` is still missing from the frozen Teams suite YAML, so even the frozen promotion lane remains incomplete until that suite is rerun.

So the calibrated winner question remains open, but the `FT8/FT10` branch question does not.

## 3. Current literature alignment

Current literature still supports the two repo-side lessons that matter here:

- identity-aware generalization remains fragile when detectors learn identity-correlated shortcuts instead of manipulation cues
- reducing brittle spatial dependence remains a live route to better robustness under shift

Most relevant here:

- `Implicit Identity Leakage: The Stumbling Block to Improving Deepfake Detection Generalization`
- `Reduced Spatial Dependency for More General Video-level Deepfake Detection`

Repo implication:

- a branch that only renames real-side weights without changing actual identity competition is not strong scientific evidence
- explicit sampler/curriculum redesign remains more plausible than more small weight nudges

## 4. Immediate next move after this package

If the next round can execute only one measurement track, it should be:

1. freeze the five-run shortlist above
2. add `teams_fake_all_lockbox` to the promotion suite
3. score only those five checkpoints under a calibrated low-FP contract
4. ignore `FT8/FT10` unless a future repo change gives real-side weights a real sampling path

## 5. Status

- Established:
  - remote runtime loaded some external VCD reals
  - `FT8/FT10` still do not constitute meaningful extra sampler evidence
- Plausible:
  - calibrated shortlist measurement can now exclude `FT8/FT10` with no scientific loss
- Still unknown:
  - the calibrated winner
  - the deployment-safe threshold outcome
  - whether a true sampler redesign can materially improve Teams real safety
