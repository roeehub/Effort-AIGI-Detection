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
| WT-A | 1 | merged | roeedar/codex-wt-a | wt-a-data-truth-2026-04-17 | ../wt-a-data-truth-2026-04-17 | Merged 2026-04-17 19:05 CEST; policy truth freeze and lane semantics landed |
| WT-E | 2 | merged | roeedar/codex-wt-e | wt-e-promotion-contract-2026-04-17 | ../wt-e-promotion-contract-2026-04-17 | Merged 2026-04-17 19:05:04 CEST; authoritative calibrated promotion path added |
| WT-D | 3 | merged | roeedar/codex-wt-d | wt-d-decision-system-2026-04-17 | ../wt-d-decision-system-2026-04-17 | Merged 2026-04-17 19:09:08 CEST; report-driven threshold, abstain, temporal, and hysteresis tooling landed |
| WT-C | 4 | merged | roeedar/codex-wt-c | wt-c-nuisance-2026-04-17 | ../wt-c-nuisance-2026-04-17 | Merged 2026-04-17 19:07:22 CEST; GammaUp runtime truth, sidecars, and tests landed; WT-D owns measurement |
| WT-F | 5 | merged | roeedar/codex-wt-f | wt-f-proper-data-schema-2026-04-17 | ../wt-f-proper-data-schema-2026-04-17 | Merged 2026-04-17 19:06:14 CEST; proper-data schema, inventory template, and future eval slice contract landed |
| WT-B | 6 | blocked | roeedar/codex-wt-b | wt-b-weak-signal-ablations-2026-04-17 | ../wt-b-weak-signal-ablations-2026-04-17 | Blocked 2026-04-17 20:41:39 CEST after merge `5533985`; draft-only package landed, but tracked training-side visomaster_hints integration is still missing |

## Claim Log

Append one line per state change.

| Time | Track | New Status | Owner | Summary |
| --- | --- | --- | --- | --- |
| 2026-04-17 18:53 CEST | WT-A | claimed | roeedar/codex-wt-a | Claimed WT-A; root commit before worktree creation |
| 2026-04-17 | WT-B | blocked | root setup | Waiting for WT-A lane-semantics freeze |
| 2026-04-17 18:53:43 CEST | WT-E | claimed | roeedar/codex-wt-e | Claimed WT-E; root commit before worktree creation |
| 2026-04-17 19:05:04 CEST | WT-E | merged | roeedar/codex-wt-e | Merged authoritative promotion contract tooling and runbooks |
| 2026-04-17 18:54:18 CEST | WT-C | claimed | roeedar/codex-wt-c | Claimed WT-C; root commit before worktree creation |
| 2026-04-17 18:54:13 CEST | WT-D | claimed | roeedar/codex-wt-d | Claimed WT-D; root commit before worktree creation |
| 2026-04-17 18:54:43 CEST | WT-F | claimed | roeedar/codex-wt-f | Claimed WT-F; root commit before worktree creation |
| 2026-04-17 19:05 CEST | WT-A | merged | roeedar/codex-wt-a | Merged policy truth artifact, lane freeze doc, and WT-A handoff |
| 2026-04-17 19:05 CEST | WT-B | free | root follow-up | WT-A lane-semantics freeze merged; WT-B may start |
| 2026-04-17 19:06:14 CEST | WT-F | merged | roeedar/codex-wt-f | Merged proper-data schema docs, provenance-first manifest builder, and future suite template |
| 2026-04-17 19:07:22 CEST | WT-C | merged | roeedar/codex-wt-c | Merged GammaUp runtime-truth plumbing, sidecar YAMLs, and WT-C handoff |
| 2026-04-17 19:09:08 CEST | WT-D | merged | roeedar/codex-wt-d | Merged decision-system analysis tooling, stability docs, and WT-D handoff |
| 2026-04-17 20:36:32 CEST | WT-B | claimed | roeedar/codex-wt-b | Claimed WT-B; verifying whether the tracked tree can honestly express explicit weak-signal hint lanes |
| 2026-04-17 20:41:39 CEST | WT-B | blocked | roeedar/codex-wt-b | Merged draft-only weak-signal package; tracked `visomaster_hints` / `visomaster_hints_teams` source integration is still missing |
