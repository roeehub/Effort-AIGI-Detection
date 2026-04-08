# Teams Augmentation Side-Track Handoff

**Date:** April 6, 2026  
**Purpose:** kickstart a separate, non-blocking agent track focused on a more faithful Microsoft Teams augmentation path without disturbing the main-line Track A / Track C work

## 0. Current Main-Line State

This side-track doc was drafted during the same implementation thread that:

- landed the resolver-driven Track A source path;
- added the Track C / S2 manifest builder and validation plumbing; and
- produced dedicated main-line handoff docs.

The main-line state is now:

- the first Track A smoke has already **passed** on Vertex;
- the next main-line Track A step is the first full-length launch;
- Track C manifest tooling is already in place and should remain undisturbed.
- the frozen Teams target-domain manifest now exists at:
  - `DeepfakeBench/training/arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json`

Before doing any side-track work, read:

- `DeepfakeBench/training/docs/TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md`
- `DeepfakeBench/training/docs/TRACK_A_HANDOFF_2026-04-06.md`
- `DeepfakeBench/training/docs/TRACK_A_RUNTIME_RUNBOOK_2026-04-06.md`
- `DeepfakeBench/training/docs/SMOKE_RUNNING_AGENT_HANDOFF_2026-04-06.md`

## 0.1 Coordination Update (April 7, 2026)

Current ownership after regroup:

- `Agent C1` owns Track C scorecard export and is currently in progress there.
  - write scope:
    - `DeepfakeBench/training/arena/run_target_domain_validation_sequential.py`
    - `DeepfakeBench/training/arena/build_teams_target_domain_suites.py`
    - frozen / breakdown target-domain suite manifests
    - related Track C tests / scorecard helpers
- Track A is now in progress on the main line.
  - live job:
    - display name: `exp-R13_A_trackA_teams_enhanced-20260407-160710`
    - Vertex job id: `7406731712530481152`
  - best next move:
    - wait for the first full-length checkpoint
    - then hand off to the one-off arena path
- Track B is now in progress.
  - first completed deliverables:
    - `DeepfakeBench/training/tools/analyze_teams_matched_pairs.py`
    - `DeepfakeBench/training/docs/TRACK_B_MATCHED_PAIR_REPORT_2026-04-07.md`
  - first local rerun artifacts:
    - `/tmp/teams_matched_pairs_2026-04-07.json`
    - `/tmp/teams_matched_pairs_2026-04-07.csv`
- current recommendation from that first rerun:
    - keep all work sidecar
    - do not promote the old single-mode `TeamsCodecSimulation` into active `phase2_round13` configs
  - best next move:
    - Track B has now produced a first best sidecar candidate:
      - `teams_codec_simulation.policy: "family_split"`
    - if Track B continues, test that candidate in a dedicated sidecar ablation
    - keep all work sidecar and avoid active `phase2_round13` configs

Do not overlap with `Agent C1` files if you pick up Track A or Track B in a
separate worktree.

## 0.2 Current Track B Readout (April 7, 2026)

The first deterministic current-bucket rerun is now complete.

Quick readout:

- ordinary current Teams v1/v2 slices were **not** uniformly blur-dominant
- the true enhanced-through-Teams slice remained the most blur / HF-loss-like
- the current single `TeamsCodecSimulation` matched direction poorly on the ordinary slices and only partially on the enhanced slice

Key numbers from `DeepfakeBench/training/docs/TRACK_B_MATCHED_PAIR_REPORT_2026-04-07.md`:

- `teams_v1_real` simulator direction match: `37.5%`
- `teams_v1_fake` simulator direction match: `50.0%`
- `teams_v2_real` simulator direction match: `37.5%`
- `teams_v2_fake` simulator direction match: `50.0%`
- `visomaster_enhanced_to_teams` simulator direction match: `62.5%`

Safe interpretation:

- Track B has earned a **do not promote current simulator blindly** conclusion
- Track B has **not** yet earned a main-line augmentation change
- the current best sidecar candidate is now:
  - `teams_codec_simulation.policy: "family_split"`
- the next justified step is a dedicated sidecar ablation of that policy, not a silent re-enable of the old preset

## 1. Scope And Hard Guardrails

This side-track exists to investigate and, if useful, prototype a more realistic Teams augmentation path.

It is **not** the main line of work.

Hard guardrails:

- Do **not** relaunch the historical Track A smoke; it already cleared the runtime gate.
- Do **not** edit the main-line Track A configs unless there is an explicit decision to test a new augmentation policy.
- Do **not** disturb the frozen-manifest / target-domain evaluation work.
- Prefer analysis, reports, experimental transforms, or clearly sidecar configs over edits to active production-path files.
- If code experiments are created later, they should be isolated behind new names or new experimental YAMLs, not silently swapped into the current R13 path.

Main-line files to avoid touching unless explicitly approved:

- `DeepfakeBench/training/experiments/phase2_round13/R13_SMOKE_trackA_teams_enhanced.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`
- `DeepfakeBench/training/arena/build_teams_target_domain_manifest.py`
- `DeepfakeBench/training/arena/target_domain_suites.teams_manifest.template.yaml`
- `DeepfakeBench/training/arena/prefix_rules.teams_manifest.template.yaml`

## 2. Why This Track Exists

The current best understanding is that the model fails on **enhanced faces after Teams passthrough** because training has mostly exposed it to:

- clean enhanced crops, and
- real Teams passthrough data from other families,

but not enough of the exact combined condition:

- `enhancement + Teams codec / call processing`

This is already documented in `ENHANCED_TEAMS_GAP_REPORT.md`:

- manual testing showed complete failure on enhanced-then-Teams frames;
- the likely root cause is that Teams washes away the enhancement artifacts the model learned to rely on;
- the repo already warns that the old Teams simulator is only an approximation.

## 3. What The Main Line Looks Like Today

### 3.1 Active R13 augmentation path

The active Track A config uses:

- `augmentation.version: "quality_targeted_family"`
- `augmentation.strength: "vcd_targeted"`
- context variation overrides (gamma / brightness / contrast / framing / CCT)

Important negative fact:

- no `phase2_round13` config currently enables `augmentation.teams_codec_simulation`

So the active R13 path is **not** explicitly applying the Teams-specific simulator to non-Teams samples.

Operational status update:

- the smoke gate for this path already passed;
- the next main-line action is the first real full-length Track A launch, not another smoke;
- the smoke checkpoint itself should not be treated as a quality result.

### 3.2 What the active router still does

Even without `TeamsCodecSimulation`, the `vcd_targeted` preset still injects a generic `VideoCodecSimulation` via `webcam_codec_p`.

That generic transform was designed for:

- VCD / webcam / video-call style codec fingerprints,
- flatter PSD slope,
- extra high-frequency codec noise,
- generic block quantization artifacts

This is useful for robustness, but it is not a faithful Teams model.

### 3.3 What Teams samples themselves get

The current family router is intentionally conservative for real Teams data:

- Teams passthrough families get only light flip + mild brightness/contrast jitter.

That part is reasonable. Real Teams-native data should mostly preserve its authentic fingerprint.

### 3.4 What the new merged enhanced source really contains

The resolver-driven `visomaster_teams_enhanced` source is useful, but it is **not** mostly true Teams-paired data:

- `999` enhanced base sample IDs were audited
- only `54` resolve to full `teams_v2` companion
- `943` resolve to `clean_companion_only`
- `2` are missing companion data

Implication:

- the new merged source is hybrid and training-usable,
- but it does **not** solve the augmentation question by itself,
- because the overwhelming majority of enhanced rows still do not have a full Teams fake companion.

## 4. Core Augmentation Insights

### 4.1 Current augmentation is mainly anti-shortcut robustness, not Teams emulation

The active R13 augmentation stack is good at:

- breaking quality shortcuts,
- broadening lighting/exposure/color/framing variation,
- making fake and real families span wider quality ranges

It is **not** a strong approximation of the Microsoft Teams pipeline.

The current stack is mostly:

- JPEG compression,
- blur,
- downscale,
- generic codec simulation,
- noise,
- color / lighting transforms

Those are useful ingredients, but not a data-backed Teams model on their own.

### 4.2 The explicit Teams simulator is still provisional

`TeamsCodecSimulation` currently models:

- brightness increase,
- contrast increase,
- blur / smoothing,
- JPEG-like compression,
- bilateral deblocking,
- optional chroma blur

That is better aligned with matched Teams pairs than the older bucket-level assumptions, but the repo already says:

- the first bucket-level measurements were wrong,
- matched-pair validation was required,
- Teams behavior is at least partly bimodal,
- one fixed transform is only an approximation

So the existing Teams simulator should be treated as a starting point, not ground truth.

### 4.3 The generic webcam codec transform and Teams transform point in different directions

This matters:

- `VideoCodecSimulation` adds coherent codec noise and quantization artifacts
- matched-pair Teams analysis found the dominant mode often **reduces** sharpness, noise, and high-frequency energy

That means the active R13 stack likely mixes:

- one transform family aimed at webcam/VCD robustness, and
- a different, disabled transform family meant to mimic Teams smoothing

This is probably useful for general robustness, but it is not the same as explicitly teaching the model what Teams does to enhanced fakes.

### 4.4 The current simulator was not validated on the exact failure case

The existing validation path is directionally helpful, but limited:

- it uses the older Teams bucket, not the newer `teams-v2` bucket
- it validates on matched real frames
- it does not directly target enhanced/fake content

That means it is weak evidence for the exact problem we care about:

- `enhanced fake -> Teams -> detector failure`

### 4.5 The external report supports a richer model than we currently use

The external research report is consistent with a broader view of Teams as:

- adaptive resolution / FPS / bitrate ladder,
- H.264-dominant camera-video pipeline,
- content-adaptive pre/post processing,
- network-state-dependent behavior,
- possible freeze / frame-drop regimes,
- telemetry-visible outcomes rather than one fixed pixel transform

For a **frame-based face detector**, the practical implication is:

- prioritize the **post-decode visual outcomes** the model actually sees,
- not the transport internals themselves

Useful targets for augmentation:

- scale / bitrate-linked softness or quantization,
- denoise / deblock / sharpen modes,
- brightness / contrast / color-temperature drift,
- multimodal behavior rather than one single preset

Less important for this detector:

- literal SRTP / ICE / RTP simulation by itself

## 5. Recommended Improvement Ladder

### 5.1 Trivial

These are analysis-heavy and should not threaten the main line:

1. Re-run matched-pair analysis on the **current** buckets, especially `teams-v2`.
2. Separate analysis by:
   - real vs fake
   - enhanced vs non-enhanced
   - old Teams bucket vs v2
3. Confirm whether the current data still looks like:
   - one dominant blur/smoothing mode, or
   - at least two distinct modes
4. Produce a new augmentation report before changing any training path.
5. Stop treating `vcd_targeted` as if it were already “Teams simulation”.

### 5.2 Medium

These are plausible sidecar contributions if the analysis supports them:

1. Replace the single fixed Teams transform with a **2-mode mixture**:
   - blur/smoothing mode
   - sharpen/cleanup mode
2. Apply Teams simulation **selectively by family** instead of globally.
3. Focus the synthetic effort on the highest-value gap:
   - `visomaster_enhanced_fake`
   - possibly `deeplive_enhanced_fake`
   - especially rows that currently fall back to clean companions
4. Build an **offline cached Teamsified dataset** for selected enhanced families instead of relying only on on-the-fly random transforms.
5. Add pair-consistency evaluation:
   - the same frame before and after Teams should preserve the label,
   - and the model should not swing wildly in logit space.

### 5.3 Difficult

These are real projects, not quick tweaks:

1. Collect more real enhanced-through-Teams data using the OBS coordinated sender/receiver path.
2. Build a video-level emulator with coupled state:
   - bitrate,
   - resolution,
   - frame rate,
   - call-state changes,
   - freeze / loss regimes
3. Fit augmentation distributions from telemetry plus matched pairs.

These may be worth doing, but they should not be the first move.

## 6. How To Use Before/After Teams Data Correctly

The repo already learned the correct lesson:

- bucket-level comparisons are unreliable,
- matched pairs are the real source of truth

The separate agent should treat the matched pairs as the primary calibration asset.

### 6.1 Primary validation uses

Use matched before/after pairs to:

1. Fit image-space deltas:
   - sharpness
   - high-frequency energy
   - noise estimate
   - brightness
   - contrast
   - blockiness
   - compressibility / bpp
   - chroma behavior if useful
2. Detect whether Teams behavior is:
   - single-mode,
   - bimodal,
   - or family/content-conditional
3. Compare candidate augmentations against real Teams outcomes.

### 6.2 Best use of the enhanced data that already passed through Teams

Yes, this data should absolutely be used to validate or reinforce correctness.

Recommended use:

- treat the true enhanced-through-Teams pairs as the highest-value lockbox for this augmentation question
- do **not** spend them all on fitting
- keep a held-out subset for “does the synthetic policy actually move closer to the real enhanced Teams distribution?”

Practical split idea:

- use ordinary Teams matched pairs to fit a generic Teams appearance model
- use a dev subset of the true enhanced-through-Teams pairs to tune enhanced-specific policy choices
- keep a small enhanced-through-Teams lockbox untouched for final comparison

### 6.3 Model-space validation is as important as image-space validation

The goal is not just to make images look Teams-like.

The better validation question is:

- does `original -> synthetic Teams` move detector behavior in the same direction as `original -> real Teams`?

Useful checks:

- logit shift magnitude
- fake-score drop on enhanced fakes
- label consistency on real pairs
- method/enhancer-specific behavior

## 7. Recommended Side-Track Execution Order

1. Read:
   - `DeepfakeBench/training/docs/ENHANCED_TEAMS_GAP_REPORT.md`
   - `DeepfakeBench/training/docs/TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md`
   - `DeepfakeBench/training/data/augmentations/teams_simulation.py`
   - `DeepfakeBench/training/data/augmentations/pipelines.py`
   - `DeepfakeBench/training/tests/test_teams_simulation.py`
2. Produce an updated matched-pair report on current data.
3. Decide whether the evidence supports:
   - one better global Teams transform,
   - a two-mode mixture,
   - or family-conditional policy
4. If experimentation is justified, keep it sidecar:
   - new transform names
   - new experiment YAMLs
   - no silent edits to active R13 configs
5. Only after that, decide whether the side-track has earned a proper training ablation.

## 8. What A Useful Side-Track Deliverable Looks Like

A successful contribution from the separate agent would be:

1. A new report using matched pairs from current buckets.
2. A clear recommendation among:
   - keep current setup,
   - re-enable old `TeamsCodecSimulation`,
   - replace it with a mixture model,
   - or build offline cached Teamsified enhanced data
3. Evidence that the recommendation is justified by:
   - pair metrics,
   - model-space behavior,
   - and the real enhanced-through-Teams examples
4. Zero disruption to the active smoke / Track A / manifest path.

## 9. Bottom Line

The current augmentation stack is useful, but it is not yet a convincing “true to source” Microsoft Teams emulator.

The safest next step is **not** to start patching active training configs.

The safest next step is:

- re-measure on matched current data,
- use the real before/after Teams pairs as the calibration backbone,
- treat enhanced-through-Teams examples as the key validation asset,
- and keep any augmentation experiments fully sidecar until they earn promotion into the main line.
