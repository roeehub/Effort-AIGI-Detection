# Round 3 Research Index

This round did not try to name the calibrated deployment winner.

It closed a narrower Track 1 uncertainty that Round 2 left open because only local discovery truth was available:

> Are `R13_FT8` and `R13_FT10` real runtime-distinct evidence, or are they mostly renamed duplicates of `R13_FT7` and `R13_FT9`?

This round did four concrete things:

1. Used live W&B run truth for `FT7/FT8/FT9/FT10` instead of only local discovery-cache approximations.
2. Proved that the remote R13 runtime did load a nonzero external VCD real lane.
3. Traced the current identity sampler to show why the `FT8/FT10` "realboost" weights still do not materially change paired-sample selection.
4. Froze the shortlist implication: `FT8` and `FT10` should no longer be treated as open scientific contenders under the current runtime.

## Read this round in this order

1. `01_remote_runtime_realboost_and_duplicate_truth.md`
2. `02_shortlist_implications_and_remaining_unknowns.md`
3. `03_next_agent_handoff_prompt.md`
4. `04_internal_instability_mitigation_summary_for_parallel_review.md`

## What changed versus Round 2

- Round 2's local statement that the local viewer/cache path had no discovered external reals remains true locally, but it was not the full remote-runtime truth for the actual `R13_FT7/8/9/10` runs.
- The remote `FT7/FT8/FT9/FT10` runs all discovered the same external VCD lane:
  - `5056` frames
  - `158` identities
  - `63` identities routed into the configured `external_training_reals` split
  - `63` unpaired external-real samples before the global split
  - `55` external train samples after the global identity split
- That new remote truth does **not** rehabilitate `FT8` or `FT10`.
  - paired-sample selection is weighted by fake family only
  - `FT8/FT10` changed only real-family weights
  - the remaining `external_real` change is also effectively inactive because each external identity contributes one unpaired sample
- The repo still lacks an accessible calibrated shortlist scorecard in this runtime.
  This round closes the `FT8/FT10` duplication question, not the calibrated winner question.

## Evidence base used this round

- W&B project `dtect-vision/phase2r13-experiments`
  - `w4n9ejic` -> `R13_FT7_trackA_merged_base`
  - `fpdcvzhf` -> `R13_FT8_trackA_merged_realboost`
  - `irzf5ymv` -> `R13_FT9_trackA_merged_porig70`
  - `ctcz09ko` -> `R13_FT10_trackA_merged_realboost_porig70`
- Downloaded `output.log` files for all four runs
- Repo/runtime truth
  - `data/sources/combined_paired.py`
  - `utils/grouping.py`
  - `train_sweep.py`
- Prior package context
  - `research_2026-04-15_round2/01_evaluation_contract_and_shortlist.md`
  - `research_2026-04-15_round2/03_sampler_curriculum_leverage.md`
  - `research_2026-04-15/04_data_composition_and_curriculum_opportunities.md`

## What this round does not claim

- It does **not** claim a calibrated promotion winner.
- It does **not** claim the missing enhanced-through-Teams fake condition was fixed.
- It does **not** claim `FT7` or `FT9` beats `R12_G` on the deployment contract.
- It does **not** claim current threshold-`0.5` tables are promotion-safe.
