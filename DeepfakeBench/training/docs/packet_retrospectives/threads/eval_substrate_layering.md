# Thread: eval substrate layering — contract vs diagnostic vs HDTF

> **Read protocol**: see `AGENTS.md`. **Update protocol**: see `AGENTS.md`. Run `tools/regenerate_open_loops.py` after editing any `### Open loop:` block.

## The question

What eval substrate covers what failure mode, and which substrates are *not* covered by the canonical promotion-contract scorecard? Eval has accreted three layers — the 29-suite contract (frozen 2026-04-23), the HDTF cross-substrate manifest (provisional, 2026-04-19), and a growing set of diagnostic substrates that postdate the contract freeze and live only in `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv`. Without a doc-level map, packet eval plans miss the substrates that motivated their own packet.

## Initial belief

The 29-suite contract scorecard at `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml` is the canonical readout for promotion. Anything outside it is "informational" and either (a) gets folded back in periodically or (b) lives in `OPEN_LOOPS` if structurally important. This implicit assumption was never written down.

## What changed our mind

- **2026-05-06 — production false-flag at xinhe-may6.** Plan `NEXT_STEPS_PLAN_2026-05-06.md:9` opens with the may6 false-flag (92 frames, deployment scores 0.83-0.93 captured 2026-05-06 11:06 UTC) as the literal session trigger. The frames are scored in `analysis/xinhe_cross_camera_audit_2026-05-06/` and added to `grouped_manifest_v2.csv` as suite `xinhe_may6_falseflag` — but **not** to the contract suite. A scorecard run on the contract has no signal on may6.
- **2026-05-07 — P1 eval design caught the gap mid-flight.** A first-draft eval plan for P1 (PE_PAIR_RANK_DRO) Phase A scored against the 29-suite contract; the user pointed out that every substrate that drove this week's investigation (`xinhe_may6_falseflag`, `live_*_teams_prod` per-variant, `dor_evening`/`dor_morning`, `team_sanity_may5`, `dor_fake_local`, `extra`, `visomaster_v2_dor`) is in `grouped_manifest_v2.csv`, not in the contract. Without a Phase A.5 over those substrates, P1's verdict on the failure modes that motivated the packet would be unmeasured. Phase A.5 was added; the gap surfaced as this thread.
- **The contract itself drifted unstaged.** Audit on 2026-05-07 found `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml` had `9` suites in HEAD vs `29` in working tree (uncommitted since `b50f245`). Diff is a 2026-05-04 expansion adding read-only diagnostic-only sub-suites (poor_quality_lockbox, lighting_extreme_lockbox, capture-mode mini-slices). Image `1.3.270` baked the working-tree (29-suite) version. The *contract* itself was a moving target undocumented in `git log`.
- **HDTF manifest similarly drifted.** `arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json` grew from 4,968 videos (HEAD) to 7,304 videos (working tree, +47%) since `b50f245`. PA's prior 7.87% on HDTF was computed against the smaller manifest; the larger version isn't comparable without re-baselining.

## Current stance (2026-05-07)

Eval has three layers; every packet-eval plan must address all three:

1. **Contract scorecard** — `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml` (29 suites as of 2026-05-07; 9 in HEAD prior to today's commit). Promotion-grade, runs on Vertex GPU via `arena/launch_teams_promotion_contract.sh`. Authority for τ calibration + lockbox readout per memory `project_promotion_contract.md`. Substrates: `teams_real_all_*`, `teams_fake_all_*`, plus diagnostic-only sub-suites for poor_quality / lighting_extreme / capture-mode mini-slices added 2026-05-04.

2. **HDTF cross-substrate scorecard** — `arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml` + `arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json` (16 suites, 7,304 videos as of 2026-05-07). Substrate-shift gate (F4 in P1's close criterion). PA passed F1 on v2 substrate then collapsed to 7.87% recall on HDTF (memory `project_pa_does_not_generalize_to_hdtf_2026-05-05.md`); F4 catches the substrate-bound walkback class.

3. **Diagnostic substrates (not in either scorecard)** — `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv` (14,626 frames, 14 suites). Includes the substrates that motivated the current week's investigation. Scored locally on Mac CPU via the `arena.GCSFrameDataset` pattern (template at `analysis/xinhe_cross_camera_audit_2026-05-06/run_local_inference.py`, P1 adaptation at `analysis/p1_pe_eval_2026-05-07/diagnostic_substrates/run_inference.py`). Existing columns: `score_P8A`, `score_E2B`, `score_PA_3800`. Substrates not in either scorecard:

   | suite | n | rationale |
   |---|---:|---|
   | `xinhe_may6_falseflag` | 92 | 2026-05-06 production false-flag (the session trigger) |
   | `live_fakes_teams_prod` | 1,675 | per-variant xinhe-fake-1..11 recall in plan §3.4 |
   | `live_reals_teams_prod` | 677 | matched-domain real prod baseline |
   | `dor_morning` / `dor_evening` | 244 / 324 | substrate-invariance markers (P8A 10.7% / 0.0% FPR) |
   | `team_sanity_may5` | 210 | session-baseline reals |
   | `dor_fake_local` | 605 | local Dor swap variants |
   | `extra` | 918 | misc real + fake aggregation (xinhe / xiang / fnh) |
   | `visomaster_v2_dor` | 2,073 | viso-on-Dor diversity probe |

   Total diagnostic frames: 6,818. None in contract or HDTF.

**Operational rule for eval plans**: every packet's eval plan must include three explicit phases — (A) contract scorecard, (C) HDTF cross-substrate, (A.5) diagnostic substrates — OR explain why a phase is omitted. The default is "include all three." The diagnostic-substrates phase is CPU-only and adds no GPU cost; omitting it is a documentation choice, not an economy.

**Long-term direction**: bake the diagnostic substrates into a new contract-suite extension `target_domain_suites.teams_promotion_contract_2026-05-07_extended.yaml` so they're scored on the GPU scorecard and don't require parallel CPU runs per packet. Tracked as a separate follow-up packet decision; in flight 2026-05-07.

## Packet timeline

- [PD](../packets/PD.md) — first packet to need diagnostic substrates beyond contract; scorecard ran on contract only (`pd_scorecard_artifacts_2026-05-06/`); diagnostic substrates handled separately via `xinhe_cross_camera_audit_2026-05-06/`. The split-eval pattern was implicit but undocumented.
- [P1](../packets/P1.md) — first packet to formally split eval into Phase A (contract), A.5 (diagnostic substrates), C (HDTF), making the layering explicit. Surfaced this thread.

## Evidence locations

- `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml` — contract suite (29 suites in working tree as of 2026-05-07).
- `arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json` — HDTF manifest (7,304 videos).
- `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv` — diagnostic substrate manifest (14 suites, 14,626 frames).
- `analysis/xinhe_cross_camera_audit_2026-05-06/run_local_inference.py` — CPU-inference template for diagnostic substrates.
- `analysis/p1_pe_eval_2026-05-07/diagnostic_substrates/run_inference.py` — P1 adaptation, scoring on `grouped_manifest_v2.csv` non-contract substrates.
- `docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md:47-100` — plan §3.1 / §3.2 tabulating all three layers' per-suite numbers in one table.
- Memory: `project_promotion_contract.md` (contract is canonical readout); `project_pa_does_not_generalize_to_hdtf_2026-05-05.md` (HDTF is the substrate-shift gate); `project_deployment_is_e2b_2026-05-06.md` + `project_xinhe_may6_falseflag_2026-05-06.md` (the production-failure substrates that motivated the diagnostic layer).

## Open loops

### Open loop: diagnostic-substrates-not-in-contract
status: open
severity: high
first_seen: 2026-05-07
last_verified: 2026-05-07
close_criterion: either (a) the canonical contract suite manifest is extended with the 9 diagnostic substrates listed in this thread (`xinhe_may6_falseflag`, `live_*_teams_prod`, `dor_evening`/`dor_morning`, `team_sanity_may5`, `dor_fake_local`, `extra`, `visomaster_v2_dor`) so the GPU scorecard scores them automatically, OR (b) a `verdict_template.md` is authored that mandates a `Phase A.5 — diagnostic substrates` step in every packet retro and is referenced by `AGENTS.md` as a pre-launch checklist item. Either path closes the loop; (a) is the more durable fix.

The gap surfaced 2026-05-07 during P1 eval-plan design: an initial draft scored only against the 29-suite contract, missing every substrate (including `xinhe_may6_falseflag`, the literal session trigger) that motivated the packet. Plan §3.1 / §3.2 tabulates the 14-suite reality but no doc enforces it for future packets. Without closure, P2/P3+ eval plans risk re-omitting the layer.

### Open loop: eval-manifests-version-pinning
status: open
severity: high
first_seen: 2026-05-07
last_verified: 2026-05-07
close_criterion: every eval manifest under `arena/manifests/` is either (a) committed with a date-pinned filename (`..._wave_<YYYY-MM-DD>.json` or `..._wave_<YYYY-MM-DD>_v2.json`) so cross-version comparisons are explicit, OR (b) the manifest filename embeds a content-hash that the scorer/runner records alongside its results so a stale comparison is auto-flagged. A pre-launch lint at `tools/lint/preflight_launch.sh` fails the launch if `git status arena/manifests/` is non-empty.

The gap surfaced 2026-05-07 during P1 Phase C design: `arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json` had grown from 4,968 videos (HEAD) to 7,304 videos (working tree) since `b50f245` without a commit. PA's 7.87% HDTF baseline was computed against the smaller manifest; the larger one isn't comparable. Same risk on `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml` (9→29 suites, also unstaged). Without the lint, every future scorecard launch risks running against a different substrate than the one prior numbers cite.

### Open loop: grouped-manifest-v2-stale-paths
status: open
severity: high
first_seen: 2026-05-07
last_verified: 2026-05-07
close_criterion: `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv` is regenerated against current GCS state (bucket layout has migrated to `session_<timestamp>/...` for `live-fakes-teams-prod` and the `roee_tester_real_2026-03-24/` folder no longer exists), AND every row whose `frame_path` starts with `gs://local/...` is either (a) marked with an explicit `is_local: True` flag so scoring scripts can branch (download from local mirror or skip), OR (b) re-pointed to a real GCS URI. Phase A.5 verifies by re-running the broken suites (`live_reals_teams_prod`, `dor_evening`, `dor_morning`, `dor_fake_local`, `extra` — 2,768 frames total) without zero-tensor decode failures.

The gap surfaced 2026-05-07 P1 Phase A.5: 5,536 of 6,818 inferences silently returned zero-tensor outputs because the manifest references stale paths. Two distinct stale-path classes:
- **`gs://local/...` placeholders** (4 suites, 2,091 frames): files live on a different machine. `extract_paired_features.py` flagged this for a different artifact (Phase 0h, 32% of `pair_gaps.csv`); the same class infects `grouped_manifest_v2.csv`.
- **bucket layout migration** (1 suite, 677 frames): `live_reals_teams_prod` references `gs://live-fakes-teams-prod/real/roee_tester_real_2026-03-24/...` which has been replaced by `gs://live-fakes-teams-prod/real/session_20260324_174822/...`. Existing `score_P8A` / `score_E2B` / `score_PA_3800` columns in the manifest were populated against the old paths and may also be against stale data.

Until closed, any CPU-side inference using `grouped_manifest_v2.csv` returns garbage scores on these 5 suites. Phase A.5's verdict on may6 / live_fakes / viso / team_sanity is unaffected (those 4 suites scored cleanly on 4,050 frames); the dor invariance check (`dor_evening` / `dor_morning`) is the load-bearing missing piece.

### Open loop: open-loops-stale-state-claims
status: open
severity: medium
first_seen: 2026-05-07
last_verified: 2026-05-07
close_criterion: `tools/regenerate_open_loops.py` is extended to (a) flag entries whose `last_verified` is more than 30 days old AND whose `close_criterion` text contains state-claim keywords (`uncommitted`, `in working tree`, `not yet committed`, `pending commit`), AND (b) auto-check those claims against current `git status` / `git log` output where possible. Output is a warning section in `OPEN_LOOPS.md` listing entries that need re-verification.

The gap surfaced 2026-05-07 during P1 launch prep: the `contract-policy-bug-fix-not-committed` entry described the v3-fix scorer code as "uncommitted in working tree" with `last_verified: 2026-04-30`. Audit found the patch had been committed as `974e033` on 2026-04-29 — the entry text was stale by a week. The first-draft eval plan inherited this stale framing. Without the auto-check, drift between OPEN_LOOPS text and tree state is invisible until an agent does an explicit audit.

## Cross-thread refs

- [`promotion_contract_evolution`](promotion_contract_evolution.md) — defines the contract layer's authority. This thread documents that the contract is a *layer*, not the only readout.
- [`contract_policy_bug`](contract_policy_bug.md) — the contract's policy-fix discipline. The `open-loops-stale-state-claims` loop in this thread generalizes that thread's recurring "claim doesn't match tree state" pattern.
- [`processing_signature_shortcut`](processing_signature_shortcut.md) — the may6 falseflag substrate is part of the broader shortcut-deployment-block surface; this thread is the eval-coverage half, that thread is the model-failure half.
