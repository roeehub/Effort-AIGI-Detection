# Enhanced + Teams: Detection Gap Report

**Date:** March 15, 2026  
**Context:** FT1/FT2/FT3 fine-tune runs from R12_G champion checkpoint

---

## 1. Problem: Model Fails on Enhanced Faces Through Teams

Manual testing of ~30 frames that were enhanced (face restoration) and then passed through Microsoft Teams showed **complete failure** on FT2 (our current best fine-tune, composite 0.9873). The model confidently misclassifies these as real.

This is despite FT2 having **8,788 enhanced samples** in training (7,450 train / 889 val_in_dist / 416 holdout) with near-perfect validation accuracy on clean enhanced crops:
- Holdout: 100% on all 8 enhancers
- val_in_dist: 100% on 7/8 enhancers (gpen-256 at 97.2%)

**Root cause:** The model learned to detect enhanced faces from clean crops only. It never saw the combination of enhancement + Teams compression. The Teams codec washes away the enhancement artifacts the model relies on.

---

## 2. Action Item: Verify TeamsCodecSimulation Was Not Used in Recent Runs

The `TeamsCodecSimulation` augmentation exists in the codebase (`data/augmentations/teams_simulation.py`) and is wired into the pipeline. However, it appears that **no R13 experiment config enabled it** — the `teams_codec_simulation` YAML block was only present in R9_F and R9_G configs.

**TODO:** Confirm that the currently running R13 experiments (both scratch and FT runs) do not have `teams_codec_simulation.enabled: true` in their active configs. If confirmed, all R13 training used only real Teams-passthrough data (the `deeplive_teams_*` family) and clean enhanced crops — never synthetic Teams degradation applied to enhanced samples.

---

## 3. Path Forward: OBS Coordinated Capture for Enhanced Data

The synthetic `TeamsCodecSimulation` is a 4-stage approximation (brightness boost → blur → JPEG → deblocking) validated on only 18–50 matched pairs. It models the "blur mode" of Teams but misses the content-adaptive sharpening mode and other WebRTC dynamics. Given that enhanced faces interact with Teams' own enhancement/denoising in unpredictable ways, **we should capture real Teams-passthrough data using the OBS coordinated sender/receiver pipeline.**

This requires:
- The enhanced videos are on a separate machine
- The OBS coordinated sender (`obs_coordinated_sender.py`) needs to be set up on that machine and pointed at the enhanced video files
- The receiver (`receiver_server.py`) captures the Teams output on the other end
- This is a non-trivial setup — will be handled in a separate session

---

## 4. Current Training Status

R13 FT1/FT2/FT3 have completed. Additional runs may be in progress. Given this finding, runs that don't address the enhanced+Teams gap may need to be re-evaluated or terminated, since the core issue is a missing data combination rather than a hyperparameter tuning problem.
