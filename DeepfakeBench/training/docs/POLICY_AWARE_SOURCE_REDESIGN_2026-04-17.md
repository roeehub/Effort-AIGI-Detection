# Policy-Aware Source Redesign

## Purpose

Freeze the post-April-17 lane semantics for training-side discussion and define
what a policy-aware redesign must look like before any same-day retrain can be
described honestly.

This document is intentionally stricter than the older Track A docs. It treats
lane naming as part of model truth, not just a viewer convenience.

Update 2026-04-18:

- at that point, the local WT-B runtime had explicit `combined_paired.visomaster_hints`
  and `combined_paired.visomaster_hints_teams` support plus a clean Teams
  policy filter in `training/data/sources/combined_paired.py`
- at that point, that implementation still lived in the ignored local `training/data/sources/`
  tree and still depends on the local April 17 policy packet under
  `training/debug/`
- so the tracked-tree warning below remains relevant for merge / portability
  questions even though the local workspace is no longer blocked in the same way

Update 2026-04-19:

- the April 17 manifest / summary / report / upload-audit packet now has a
  tracked canonical home under `training/policy/visomaster_bad_data/`
- the repo-level `data/` ignore rule now makes an explicit exception for
  `DeepfakeBench/training/data/**/*.py`, so the runtime package is visible to
  Git instead of being trapped in an ignored subtree
- the remaining portability gap is now commit / merge follow-through plus a
  real smoke run, not missing raw policy artifacts

## What The Re-Investigation Changed

### 1. Current checked-in training is still not proved policy-aware

The reviewed training path still consumes the old source names:

- `combined_paired.visomaster`
- `combined_paired.teams`
- `combined_paired.visomaster_teams_enhanced`

The April 17 bad-data overlay is only proved in the checked-in viewer path
today. The viewer imports `visomaster_policy.py` and relabels retained rows into
`visomaster_hints` / `visomaster_hints_teams`; the reviewed training loader path
does not have an equivalent checked-in proof.

Conclusion:

- any currently checked-in Track A / merged / teamsonly retrain is still
  `old-semantics`

### 2. The repo-tracked tree still does not carry most of `training/data/`

The runtime source files reviewed under
`DeepfakeBench/training/data/sources/{combined_paired,visomaster}.py` no longer
sit behind an ignore-rule wall in this repo layout, but they still are not part
of the Git tree on `teams-relaunch-root-2026-04-17` until the now-visible
runtime package is committed.

Implication:

- WT-A can merge back authoritative truth artifacts and redesign docs now
- a merge-safe tracked source integration still requires committing the runtime
  package now that the ignore-rule blocker is gone

### 3. The raw April 17 policy packet now has a tracked canonical copy

A canonical copy of the April 17 packet now lives under
`DeepfakeBench/training/policy/visomaster_bad_data/`, including the manifest,
summary, policy report, and upload audit.

Implication:

- the missing-packet blocker is gone for exact corrected count reconstruction
- full reproducibility still depends on getting the runtime package committed
  alongside those tracked policy artifacts

## Frozen Lane Semantics

| Lane | Kind | What it means now | What it must not be called |
| --- | --- | --- | --- |
| `visomaster_hints` | weak-signal lane | retained historical bad VisoMaster baseline rows | clean VisoMaster supervision, method-faithful supervision, target-domain lane |
| `visomaster_hints_teams` | weak-signal lane | retained historical bad Teams-played VisoMaster rows | clean Teams target-domain lane, clean per-method Teams lane |
| `teams` | mixed lane | direct Teams data after removing the policy-retained weak rows into a separate lane | monolithic pure Teams lane |
| `visomaster_teams_enhanced` | mixed diagnostic lane | mostly `clean_fallback`, tiny weak-signal Teams-companion subset, structurally missing true Teams-enhanced fake emission | exact enhanced-through-Teams lane |

## What We Can Prove Today

### Current training-path truth

- current checked-in training is **not** proved policy-aware
- the direct Teams lane is contaminated under old semantics
- the merged VTE lane still hides weak-signal Teams overlap inside a mostly
  `clean_fallback` source
- the loader still cannot emit `Teams real + Teams enhanced fake`

### Promotion-relevant corrected count freeze

The checked-in artifact
`relaunch_handoffs/WT-A_policy_truth_artifact_2026-04-17.json` freezes the
evidence-backed corrected truth that can be stated exactly from the tracked
tree:

- old total paired objects: `15,532`
- corrected total paired objects: `10,388`
- old train pairs: `13,294`
- corrected train pairs: `8,932`
- corrected retained hint train pairs: `571`
  - `396` baseline hints
  - `175` Teams hints
- corrected retained hint total pairs: `682`
  - `480` baseline hints
  - `202` Teams hints

### Direct Teams contamination freeze

- old direct Teams total: `1,348` pairs
- bad VisoMaster-through-Teams before policy: `237`
- retained weak-signal Teams hints after policy: `202`
- removed after policy: `35`
- clean direct Teams total after policy: `1,111`

Train split:

- old direct Teams train: `1,135`
- bad VisoMaster-through-Teams before policy: `198`
- retained weak-signal Teams hints after policy: `175`
- removed after policy: `23`
- clean direct Teams train after policy: `937`

### Merged VTE contamination freeze

- total VTE pairs: `997`
  - `54` `teams_v2_companion`
  - `943` `clean_fallback`
- train VTE pairs: `837`
  - `42` `teams_v2_companion`
  - `795` `clean_fallback`

The `teams_v2_companion` subset is not an extra clean target-domain pool. It is
an overlap subset of the same weak-signal Teams-played VisoMaster family.

## Same-Day Retrain Labels

Use these labels literally:

| Label | When it is allowed | Today |
| --- | --- | --- |
| `policy-corrected` | only after the training path itself consumes explicit corrected lane names and no old `visomaster` semantics remain | not available today |
| `weak-signal-only` | a future config that trains on explicit `visomaster_hints` / `visomaster_hints_teams` weak-signal lanes and describes them honestly | possible in principle, but not checked in |
| `old-semantics` | any current checked-in Track A / merged / teamsonly config | all same-day checked-in retrains |

So the current answer is simple:

- do **not** call any same-day checked-in retrain `policy-corrected`
- do **not** call any same-day checked-in retrain a true target-domain
  VisoMaster run
- call currently checked-in same-day retrains `old-semantics`

## Recommended Source Redesign

Do not silently reinterpret old source names. Add explicit opt-in lanes instead.

### New explicit weak-signal lanes

- `combined_paired.visomaster_hints`
- `combined_paired.visomaster_hints_teams`

Required semantics:

- method should collapse to the lane meaning, not old per-method VisoMaster
  claims
- fake-family weighting may temporarily alias to the old family weights
  (`visomaster_fake` and `deeplive_teams_fake`) if we need YAML compatibility,
  but the source names themselves must stay explicit
- reporting must show these lanes separately from clean `teams`

### What not to do

- do not silently auto-relabel `combined_paired.visomaster` behind the same old
  name
- do not keep mixing retained Teams hints inside the direct `teams` lane while
  calling it clean target-domain supervision
- do not keep calling `visomaster_teams_enhanced` an exact target-domain lane

## Verification Hook

The checked-in tool `DeepfakeBench/training/tools/wt_a_policy_truth.py` renders
and validates the frozen artifact without external dependencies:

```bash
python3 DeepfakeBench/training/tools/wt_a_policy_truth.py --validate
```

That tool validates:

- current training classification (`old-semantics`)
- corrected train/total count arithmetic
- direct Teams contamination arithmetic
- family-level VTE branch exposure sums

## Remaining Follow-Up

1. Commit the now-visible `training/data/**/*.py` runtime tree, or the minimum
   agreed subset, so the explicit hint-lane loader path becomes tracked code.
2. Run a real config-load or short training smoke with the tracked policy
   bundle.
3. After those two steps, call the package `weak-signal-only` or
   `policy-corrected` only where the tracked runtime path really matches that
   claim.
