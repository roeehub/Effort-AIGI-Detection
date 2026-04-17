# WT-B Weak-Signal Draft Runbook

## Status

This package is a `draft-only weak-signal plan blocked on source integration`.

It is not a runnable same-day training family in the tracked tree.

## Why WT-B Is Blocked

1. WT-A froze the lane semantics, but did not land tracked training-side
   `visomaster_hints` / `visomaster_hints_teams` support.
2. In this WT-B worktree, `DeepfakeBench/training/data/sources/` is absent from
   the tracked tree, so there is no tracked code path to inspect or extend
   inside WT-B ownership.
3. Searching the tracked training tree shows `visomaster_hints` and
   `visomaster_hints_teams` only in docs, viewer code, and the WT-A truth
   artifact, not in tracked training-source code.
4. The strongest available dry-run check on this host stops immediately because
   `python3` lacks `PyYAML`, so the existing repo dry-run path cannot be used
   here as a config-load validator.

## Scope Refinement

The original three-arm family still makes sense, but only as a future explicit
hint-lane family.

WT-B therefore refined the package in two ways:

1. Keep the three arms:
   - `no_hints`
   - `hints_only`
   - `hints_plus_teams_hints`
2. Make the family a conservative fine-tune draft anchored on the
   `R13_FT11_trackA_scorecard_base` shape instead of using scratch,
   `realboost`, `teamsonly`, or `p_original` side conditions.

Why this base:

- scratch adds variance that would blur a weak-signal question
- `realboost` remains questionable because local external reals are not proved
- `teamsonly` and `p_original` operate on the mixed `visomaster_teams_enhanced`
  lane rather than on explicit hint lanes
- the WT-B question is whether explicit retained hints help, not whether mixed
  VTE branch choices help

## Draft Family Shape

All three draft manifests keep these base choices constant:

- `df40`, `deeplive`, and direct `teams`
- conservative scorecard-base fine-tune budget
- no `realboost`
- no `teamsonly`
- no `p_original` sweep

All three draft manifests intentionally remove these confounds:

- old-semantics `visomaster`
- mixed diagnostic `visomaster_teams_enhanced`

This means the family is planned to answer only one question:

- what weak-signal value remains once the old bad-data mass is replaced with
  explicit retained hint lanes and nothing else changes

## Draft Manifests

### `R13_WTB1_weak_signal_no_hints_draft_only`

- keeps the fine-tune base
- keeps `df40`, `deeplive`, and `teams`
- removes both explicit hint lanes
- represents the future no-hints control

### `R13_WTB2_weak_signal_hints_only_draft_only`

- same base as WTB1
- adds future `combined_paired.visomaster_hints`
- keeps future `combined_paired.visomaster_hints_teams` off
- isolates retained baseline hints only

### `R13_WTB3_weak_signal_hints_plus_teams_draft_only`

- same base as WTB1
- adds future `combined_paired.visomaster_hints`
- adds future `combined_paired.visomaster_hints_teams`
- isolates the incremental effect of the Teams-played retained hints

## What These Drafts Do Not Mean

- not `policy-corrected`
- not target-domain realism
- not a promotion family
- not proof that the current tracked runtime accepts these lane names
- not permission to reinterpret `visomaster_teams_enhanced` as an exact
  enhanced-through-Teams lane

## What Must Land Before These Become Runnable

1. WT-A-owned tracked source integration for:
   - `combined_paired.visomaster_hints`
   - `combined_paired.visomaster_hints_teams`
2. An explicit tracked config-key contract for those blocks.
3. A config-load or dry-run pass in an environment that has `PyYAML` and the
   tracked training runtime available.

Only after those three steps should anyone convert these manifests from
draft-only planning files into runnable training configs.
