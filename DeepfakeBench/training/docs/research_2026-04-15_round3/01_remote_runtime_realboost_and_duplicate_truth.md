# Remote Runtime Realboost And Duplicate Truth

## 1. Question

Round 2 left one explicit Track 1 sub-question open:

> Are `R13_FT8` and `R13_FT10` real runtime-distinct evidence, or mostly renamed duplicates of `R13_FT7` and `R13_FT9`?

This file resolves that question using remote W&B run truth plus current sampler code.

## 2. Remote run truth that was not available last round

Accessible remote training runs:

| Alias | W&B run | State | Branch description |
| --- | --- | --- | --- |
| `R13_FT7` | `w4n9ejic` | finished | merged-source baseline |
| `R13_FT8` | `fpdcvzhf` | finished | merged-source realboost |
| `R13_FT9` | `irzf5ymv` | finished | merged-source `p_original=0.7` |
| `R13_FT10` | `ctcz09ko` | finished | merged-source realboost + `p_original=0.7` |

All four downloaded `output.log` files show the same external real discovery:

- `5056` frames discovered from `gs://effort-collected-data/real/VCD`
- `158` unique external identities
- `63 / 158` identities routed into the configured `external_training_reals` identity split
- `63` unpaired external-real samples, `945` total frames, added before the global split
- after the global identity split:
  - `55` train external samples
  - `4` val external samples
  - `4` test external samples

All four runs also show the same post-discovery pool sizes:

- train: `12613` samples
- val: `1420` samples
- test: `702` samples
- train unique identities: `6816`
- split seed: `737`
- run seed in W&B summary: `1024`

That is the strongest direct correction to the earlier local-only wording:

- local viewer/cache truth: `.viewer_cache/discovery/external_reals.json` was empty
- actual remote training truth for these four R13 runs: a nonzero external VCD lane was present

So "external reals absent" was a **local discovery statement**, not the final remote-runtime statement for these finalists.

## 3. Reporting mismatch that matters

There is a second source-name/runtime mismatch in the current W&B summary path.

`train_sweep.py` writes `wandb.run.summary["data/source_counts"]` using only:

- `df40`
- `deeplive`
- `visomaster`
- `total`

It does **not** include:

- `visomaster_teams_enhanced`
- `deeplive_teams`
- `external`

That means `data/source_counts` is not authoritative for full runtime composition, even though `total` includes those extra lanes.

For these runs, the safer source-composition fields are:

- downloaded `output.log` source distributions
- `data/family_counts`
- `data/train_family_counts`

## 4. Why `FT8` realboost is still a sampler no-op in the current runtime

### What `FT8` changed in YAML

Relative to `FT7`, `FT8` changes only these sampling weights:

- `deeplive_teams_real`: `4.0 -> 5.5`
- `realpool_real`: `2.5 -> 3.0`
- `external_real`: `2.5 -> 3.5`

Relative to `FT9`, `FT10` applies the same realboost block while keeping the same `p_original = 0.7`.

### What the sampler actually does

Current identity-weighted sampling in `data/sources/combined_paired.py` has two crucial properties:

1. `_sample_family_for_sampling()` prioritizes the **fake family** for paired samples.
2. `_discover_external_training_reals()` emits exactly **one** `UnifiedUnpairedRealSample` per external identity.

Those two facts collapse almost all `realboost` leverage:

- paired samples do **not** consult `deeplive_teams_real` or `realpool_real` during identity-weighted selection
- unpaired external identities each have one candidate sample, so changing `external_real` weight cannot change the within-identity choice

That means the `FT8` weight changes have no meaningful path to alter the identity-balanced training diet relative to `FT7`.

The same logic applies to `FT10` relative to `FT9`.

## 5. Remote evidence that matches the no-op conclusion

### `FT7` vs `FT8`

Remote runtime evidence is identical on the data side:

- same external discovery counts
- same family counts
- same train family counts
- same train source counts from `output.log`

Training-side summaries also line up:

| Run | `best/auc` | `best/eer` | `best/eer_threshold` |
| --- | ---: | ---: | ---: |
| `FT7` | `0.9906345966772848` | `0.02524271844660194` | `0.4483183026313782` |
| `FT8` | `0.9906345966772848` | `0.02524271844660194` | `0.44840240478515625` |

### `FT9` vs `FT10`

Again, remote runtime evidence is identical on the data side:

- same external discovery counts
- same family counts
- same train family counts
- same train source counts from `output.log`

Training-side summaries are nearly indistinguishable:

| Run | `best/auc` | `best/eer` | `best/eer_threshold` |
| --- | ---: | ---: | ---: |
| `FT9` | `0.991542525797776` | `0.021359223300970873` | `0.4455662369728088` |
| `FT10` | `0.9915654147672` | `0.019417475728155335` | `0.4454794526100158` |

These are **not** deployment metrics and should not be used for promotion.

They are only supporting evidence that the `realboost` block did not create a materially different data regime.

## 6. What this means scientifically

### Established

- The remote `FT7/FT8/FT9/FT10` runs did load a nonzero external VCD real lane.
- That lane does **not** rescue `FT8` or `FT10` as distinct sampler interventions.
- Under the current sampler, `FT8` is functionally duplicate of `FT7` for identity-balanced sample selection.
- Under the current sampler, `FT10` is functionally duplicate of `FT9` for identity-balanced sample selection.
- `realboost` in these experiment names is a config-name/runtime mismatch under the present identity sampler.

### Plausible

- The tiny `FT9` vs `FT10` training delta is optimizer noise, checkpoint timing, or incidental stochasticity, not evidence that the `FT10` realboost block taught a new real-side lesson.
- Any real gain from external VCD reals would require a sampler or curriculum that actually changes exposure, not only larger real-family weights.

### Still unknown

- whether calibrated `R13_FT7` or calibrated `R13_FT9` beats calibrated `R12_G`
- whether a true real-side curriculum using these external VCD identities would reduce Teams real false positives
- whether a future sampler rewrite could turn the configured external real lane into a meaningful intervention
