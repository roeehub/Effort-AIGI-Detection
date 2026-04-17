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
| WT-A | 1 | claimed | roeedar/codex-wt-a | wt-a-data-truth-2026-04-17 | ../wt-a-data-truth-2026-04-17 | Claimed 2026-04-17 18:53 CEST; re-investigating lane semantics and policy truth |
| WT-E | 2 | claimed | roeedar/codex-wt-e | wt-e-promotion-contract-2026-04-17 | ../wt-e-promotion-contract-2026-04-17 | Claimed 2026-04-17 18:53:43 CEST; keep aligned with WT-A and WT-F |
| WT-D | 3 | claimed | roeedar/codex-wt-d | wt-d-decision-system-2026-04-17 | ../wt-d-decision-system-2026-04-17 | Claimed 2026-04-17 18:54:13 CEST; re-investigating low-FP decision-layer leverage and stability |
| WT-C | 4 | claimed | roeedar/codex-wt-c | wt-c-nuisance-2026-04-17 | ../wt-c-nuisance-2026-04-17 | Claimed 2026-04-17 18:54:18 CEST; re-investigating runtime augmentation truth and narrow nuisance sidecars |
| WT-F | 5 | claimed | roeedar/codex-wt-f | wt-f-proper-data-schema-2026-04-17 | ../wt-f-proper-data-schema-2026-04-17 | Claimed 2026-04-17 18:54:43 CEST; re-investigating proper-data schema, provenance, and future eval slices |
| WT-B | 6 | blocked |  | wt-b-weak-signal-ablations-2026-04-17 | ../wt-b-weak-signal-ablations-2026-04-17 | Blocked on WT-A merge |

## Claim Log

Append one line per state change.

| Time | Track | New Status | Owner | Summary |
| --- | --- | --- | --- | --- |
| 2026-04-17 18:53 CEST | WT-A | claimed | roeedar/codex-wt-a | Claimed WT-A; root commit before worktree creation |
| 2026-04-17 | WT-B | blocked | root setup | Waiting for WT-A lane-semantics freeze |
| 2026-04-17 18:53:43 CEST | WT-E | claimed | roeedar/codex-wt-e | Claimed WT-E; root commit before worktree creation |
| 2026-04-17 18:54:18 CEST | WT-C | claimed | roeedar/codex-wt-c | Claimed WT-C; root commit before worktree creation |
| 2026-04-17 18:54:13 CEST | WT-D | claimed | roeedar/codex-wt-d | Claimed WT-D; root commit before worktree creation |
| 2026-04-17 18:54:43 CEST | WT-F | claimed | roeedar/codex-wt-f | Claimed WT-F; root commit before worktree creation |
