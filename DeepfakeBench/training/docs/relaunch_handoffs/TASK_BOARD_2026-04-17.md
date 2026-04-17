# Relaunch Task Board

Shared coordination board for `teams-relaunch-root-2026-04-17`.

Allowed statuses:

- `free`
- `claimed`
- `blocked`
- `analysis-done`
- `merged`

Dependency rule:

- `WT-B` stays blocked until `WT-A` merges a lane-semantics freeze into `teams-relaunch-root-2026-04-17`.

## Tracks

| Track | Priority | Status | Owner | Branch | Worktree | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| WT-A | 1 | free |  | wt-a-data-truth-2026-04-17 | ../wt-a-data-truth-2026-04-17 | Must go first; freezes lane semantics |
| WT-E | 2 | free |  | wt-e-promotion-contract-2026-04-17 | ../wt-e-promotion-contract-2026-04-17 | Keep aligned with WT-A and WT-F |
| WT-D | 3 | free |  | wt-d-decision-system-2026-04-17 | ../wt-d-decision-system-2026-04-17 | Frozen-checkpoint and low-FP analysis |
| WT-C | 4 | free |  | wt-c-nuisance-2026-04-17 | ../wt-c-nuisance-2026-04-17 | Coordinate measurement claims with WT-D |
| WT-F | 5 | free |  | wt-f-proper-data-schema-2026-04-17 | ../wt-f-proper-data-schema-2026-04-17 | Future proper-data schema and manifests |
| WT-B | 6 | blocked |  | wt-b-weak-signal-ablations-2026-04-17 | ../wt-b-weak-signal-ablations-2026-04-17 | Blocked on WT-A merge |

## Claim Log

Append one line per state change.

| Time | Track | New Status | Owner | Summary |
| --- | --- | --- | --- | --- |
| 2026-04-17 | WT-B | blocked | root setup | Waiting for WT-A lane-semantics freeze |
