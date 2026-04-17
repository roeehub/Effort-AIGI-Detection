You are continuing after `research_2026-04-15_round3`.

Do **not** reopen the `FT8/FT10` question unless you find stronger evidence that one of these changed:

- `data/sources/combined_paired.py::_sample_family_for_sampling()`
- `data/sources/combined_paired.py::_discover_external_training_reals()`
- the remote training config so that external reals become overlapping paired identities or otherwise gain real sampler leverage

Assume the following unless disproven by stronger evidence:

- remote `FT7/FT8/FT9/FT10` all loaded the same external VCD lane:
  - `5056` frames
  - `158` identities
  - `63` configured external-training identities
  - `55` external train samples after the global split
- `FT8` is functionally duplicate of `FT7` under the current identity sampler
- `FT10` is functionally duplicate of `FT9` under the current identity sampler
- the reduced shortlist is now established:
  - `R12_G_FP32`
  - `R13_A_STEP15500`
  - `R13_E_BESTSOFAR`
  - `R13_FT7_FP32`
  - `R13_FT9_FP32`

Highest-value unresolved question now:

- Which of those five checkpoints wins the calibrated low-FP Teams promotion contract?

Needed evidence for the next round:

- accessible shortlist scorecards or raw prediction artifacts
- explicit addition of `teams_fake_all_lockbox` to the promotion suite
- calibrated dev-threshold sweep plus frozen-threshold lockbox readout

What not to do next:

- do not spend calibrated promotion budget on `FT8/FT10`
- do not treat training `best/auc` as deployment evidence
- do not treat threshold `0.5` Teams tables as a final promotion contract
