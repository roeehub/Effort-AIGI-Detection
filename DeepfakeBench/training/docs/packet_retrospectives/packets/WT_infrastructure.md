# Packet WT-infrastructure  ·  Parallel-worktree relaunch prep — six tracks that enabled RLP1+

> **Template contract**: fill every section. If a section truly has no content, write `*(none — reason)*` rather than deleting the heading. Keep the status-card table intact.

## Status card

| Field | Value |
|---|---|
| Dates | 2026-04-17 → 2026-04-19 |
| Worktrees | WT-A, WT-B, WT-C, WT-D, WT-E, WT-F |
| Headline lever | parallel infrastructure delivery — data policy, weak-signal plumbing, augmentation plumbing, decision layer, promotion contract, proper-data schema + loader |
| Leader slot | *(n/a — infrastructure packet)* |
| Leader metric | *(n/a — infrastructure packet; no metrics ranked)* |
| Verdict | ✅ confirmed — all six landed, all six still in production use |
| Next-packet decision | gate-opened RLP1: WT-B hints ladder + WT-F `proper_data` became the two axes of the first real packet |
| Themes touched | [promotion_contract_evolution](../threads/promotion_contract_evolution.md) (WT-C/D/E foundation); [gate_alignment_story](../threads/gate_alignment_story.md) (WT-A source of hint-vs-clean split; WT-F exposed Teams gap); [preprocessing_parity_bug](../threads/preprocessing_parity_bug.md) (light — WT-E scaffolded the arena scoring path that later got `INTER_LINEAR`-fixed) |

## Configuration

- **Prior state.** April-6 Track A/B/C stack had landed (resolver-driven Teams-enhanced loader, matched-pair report, scorecard runbook) but assumed old VisoMaster lane semantics. See `docs/TRACK_A_HANDOFF_2026-04-06.md:1-40`, `docs/TRACK_B_MATCHED_PAIR_REPORT_2026-04-07.md`, `docs/TRACK_C_SCORECARD_RUNBOOK_2026-04-07.md`.
- **Trigger.** April-17 VisoMaster bad-data audit: 4904 sample IDs → ignored, 480 → `visomaster hints`, 202 → `visomaster hints (teams)`, 3 → delete-only; 237/1348 direct-Teams `pair_complete` rows flagged as bad VisoMaster-through-Teams. See `docs/VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md:36-60`.
- **Coordination.** Six parallel git worktrees off `teams-relaunch-root-2026-04-17`, each agent-owned, tracked on a shared board; `WT-B` gated behind `WT-A`'s freeze. See `docs/relaunch_handoffs/TASK_BOARD_2026-04-17.md:15-49`.
- **Not a training packet.** No yamls ran, no metrics produced. Deliverables: code, tracked artifacts, suite manifests, checkpoint maps, tests, runbooks — plus the April-19 `combined_paired.proper_data` loader extension.

## Results at the time

Merge timeline (`docs/relaunch_handoffs/TASK_BOARD_2026-04-17.md:21-26`): WT-A + WT-E merged 04-17 19:05 CEST; WT-F 19:06; WT-C 19:07; WT-D 19:09; WT-B merged 04-19 12:31 after its draft-only April-17 package was rebuilt once WT-A unblocked the loader path. See each WT's handoff doc for the full "what landed" list.

Remote smoke proofs (`docs/relaunch_handoffs/WT_B_AND_NEW_DATA_READINESS_2026-04-19.md:57-118`):
- **WT-B startup smoke**: Vertex `9012653657048481792` / W&B `8bjcadyr`, `JOB_STATE_SUCCEEDED`, `visomaster_hints_fake=480`, `visomaster_hints_teams_fake=202`, `deeplive_teams_fake=1111`.
- **WT-B integration smoke**: Vertex `2921535161029885952` / W&B `g6f8fovi`, `val_in_dist/overall/auc=0.97534` at step 100.
- **Proper-data startup smoke** (April-19 extension, post-merge): Vertex `2064725331922649088` / W&B `bxay0n66`, `ProperData: enabled=True -> 128 samples`, nonzero counts in all four `proper_visomaster_*` lanes.

## Conclusions drawn in-session

### WT-A — data-policy truth freeze (`docs/relaunch_handoffs/WT-A_2026-04-17.md`)

**Goal**: freeze the April-17 bad-data policy into mergeable truth so every downstream track reasons against the same composition numbers. **Landed**: authoritative JSON at `docs/relaunch_handoffs/WT-A_policy_truth_artifact_2026-04-17.json`, stdlib validator at `tools/wt_a_policy_truth.py`, focused test, lane-semantics redesign doc `docs/POLICY_AWARE_SOURCE_REDESIGN_2026-04-17.md`. Pinned corrected totals (`WT-A_2026-04-17.md:45-52`): 10,388 paired objects; 8,932 train; 571 retained hint train pairs; 1,111 / 937 clean direct-Teams. **Still used by**: every subsequent packet — RLP1 hints ladder, RLP2 hint removal, RLP6 gate-alignment plan. **Caveat**: WT-A deliberately refused a tracked-tree loader rewrite (`DeepfakeBench/training/data/` was repo-ignored); exact corrected `val`/`test` hint splits still depend on the missing raw April-17 policy packet (`WT-A_2026-04-17.md:62-65`). WT-B then owned that loader gap on April-18.

### WT-B — weak-signal ablation smoke test (`docs/relaunch_handoffs/WT-B_2026-04-18.md`)

**Goal**: three-arm ablation (`no_hints` → `hints_only` → `hints + teams_hints`) testing whether retained weak-signal residue helps the target-domain gate. **Landed**: runtime support for `combined_paired.visomaster_hints`, `combined_paired.visomaster_hints_teams`, `combined_paired.teams.apply_bad_data_policy` (`WT-B_2026-04-18.md:25-43`); runnable `R13_WTB{1,2,3}_*.yaml`; launcher startup + integration smokes; tracked policy bundle at `training/policy/visomaster_bad_data/`. **Still used by**: RLP1 tested hints as arms `02`/`03`; RLP2 dropped hints from the main slate; RLP3+ kept hints off by default. **Caveat**: the April-17 WT-B attempt shipped draft-only, blocked on WT-A (`WT-B_2026-04-17.md:28-55`) — runnable package only landed April-18 after a `.gitignore` surgery made `data/sources/*.py` reviewable. WT-B's smoke seeded the hypothesis RLP1/RLP2 then confirmed failed: hints are a net drag.

### WT-C — augmentation plumbing truth (`docs/relaunch_handoffs/WT-C_2026-04-17.md`)

**Goal**: audit live nuisance-invariance behavior, correct dead config intent (the `gamma_up_p` name was never consumed), add truthful sidecars without editing active Track-A baselines. **Landed**: optional lighting key surface on the first-preset allowlist in `data/augmentations/pipelines.py` (`WT-C_2026-04-17.md:62-73`); Teams-passthrough special-aug knobs, inert unless `teams_passthrough_special_aug_enabled: true`; three sidecars (`R13_WTC{1,2,3}_*.yaml`); `46 passed`. **Still used by**: RLP1 tested sidecars as arms `06`/`07`; both lost to base arm `05`; RLP2 deferred them. Runtime plumbing stays live; sidecar yamls are dormant. **Caveat**: WT-C surfaced a two-layer drop — renamed key + first-preset allowlist filter — that had silently killed intended GammaUp config in prior packets. Pre-WT-C augmentation claims should be read as dead-knob drift.

### WT-D — threshold sweep / decision layer (`docs/relaunch_handoffs/WT-D_2026-04-17.md`)

**Goal**: report-driven decision tooling (threshold sweeps, abstain bands, temporal aggregation, hysteresis) to rank the five-checkpoint shortlist under the low-FP contract without a retrain. **Landed**: `tools/teams_video_policy_analysis.py`, `tools/teams_frame_policy_analysis.py`, decision-report utility, two focused tests; minimal-baseline scoping doc at `docs/WT_D_MINIMAL_BASELINE_PACKET_2026-04-17.md:60-93`. **Still used by**: the τ readout and policy-summary CSVs — this is the machinery the current promotion-contract policy bug (`τ → ~0.995`) surfaces through. **Caveat**: merge-time honesty, *"this runtime still cannot name a calibrated five-checkpoint promotion winner because the shortlist report artifacts are not locally accessible here"* (`WT-D_2026-04-17.md:42-44`). The policy bug that crushes fake recall lives in the FPR-minimization wrapper, not WT-D's primitives. See [../threads/promotion_contract_evolution.md](../threads/promotion_contract_evolution.md).

### WT-E — promotion contract runbook (`docs/relaunch_handoffs/WT-E_2026-04-17.md`)

**Goal**: convert the Track-C scaffolding into an authoritative calibrated promotion path; demote fixed-`0.5` scorecards to diagnostic-only. **Landed**: `arena/target_domain_suites.teams_promotion_contract_2026-04-17.yaml` (eight suites incl. previously-missing `teams_fake_all_lockbox`); `arena/checkpoint_maps/teams_target_domain.promotion_shortlist_2026-04-17.yaml` (five-checkpoint shortlist: `R12_G_FP32`, `R13_A_STEP15500`, `R13_E_BESTSOFAR`, `R13_FT7_FP32`, `R13_FT9_FP32`); local + Vertex wrappers; stale fake-lockbox claim corrected in `research_2026-04-15_round2/01_evaluation_contract_and_shortlist.md`. **Still used by**: every promotion readout from RLP3 forward; `promotion_winner.json` and `checkpoint_summary.csv` are the only promotion-authoritative artifacts. **Caveat**: WT-E shipped the machinery but deferred emitting `promotion_winner.json` on the shortlist. The FPR-minimization-no-budget policy bug surfaced during later real use; always inspect `selected_threshold` before trusting the scorecard.

### WT-F — proper-data schema + loader (`docs/relaunch_handoffs/WT-F_2026-04-17.md`)

**Goal**: canonical `proper_*` schema for Teams-parallel clean/Teams captures; stop new HDTF/quickclips buckets from being folded into legacy `visomaster_hints` or `visomaster_teams_enhanced`. **Landed at merge**: schema doc, provenance/inventory rules, inventory template, future-manifest builder, suite template with six canonical exact lanes (`proper_real_{clean,teams}`, `proper_visomaster_{clean,enhanced_clean,teams,enhanced_teams}`) (`WT-F_2026-04-17.md:26-37`). **April-19 follow-up**: actual `combined_paired.proper_data` runtime loader (`data/sources/proper_data.py`), identity grouping keyed on WT-F `split_group_id`, loader-integrity logging, provisional inventory/manifest/suite under a strict `16/16` contract — see `docs/relaunch_handoffs/NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md:340-510`. **Still used by**: RLP1 arms `04`/`05`; RLP2 promoted unenhanced proper-data to main bet; RLP3+ built its proper-data thesis here. The `hash_stable` identity-split mode shipped here (`RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md:406-416`) is the split contract from RLP2 onward. **Caveat**: provisional snapshot built under incomplete Teams propagation (~30% more expected, `NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md:108-111`); RLP1 live counts (684 / 3186 fake rows for `04` / `05..08`) exceeded planning prose, driving the rule: cite builder reports and startup W&B summaries, not prose counts (`RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md:388-407`).

- **Session IDs**: *(none load-bearing — infrastructure scored only through downstream packet use; no single convmem session carries a canonical conclusion for the WT cluster; handoffs above are the primary record.)*

## Retrospective (as of 2026-04-24)

- **All six tracks landed and remain in production use.** No WT was reversed.
- **WT-A is the single most load-bearing artifact.** Every gate-alignment claim from RLP6 forward — the `worst_pool_fpr` correction to real-pool-only, the `avspeech` evaluation-hygiene arc — resolves against WT-A's bookkeeping. See [../threads/gate_alignment_story.md](../threads/gate_alignment_story.md).
- **WT-B hypothesis failed cleanly.** RLP1 showed hints as a net drag (control `RLP1_01` beat `RLP1_02`/`RLP1_03` across all 8 matched holdout checkpoints). Load-bearing negative: hints eliminated as a lever without re-litigation.
- **WT-C sidecars are dormant.** GammaUp and Teams-shadow lost in RLP1 on top of a dominated base; nothing later challenged it. Runtime plumbing fix stays live.
- **WT-D tooling carries the τ readout today.** The τ → ~0.995 policy bug lives in the FPR-minimization wrapper, not WT-D's decision primitives. See [../threads/promotion_contract_evolution.md](../threads/promotion_contract_evolution.md).
- **WT-E is the baseline of the contract-evolution thread.** The five-checkpoint shortlist and eight-suite manifest remain the promotion-authoritative inputs. The deferral WT-E flagged is what the FPR bug later dragged into the open.
- **WT-F's `hash_stable` split-mode is the RLP1→RLP2 boundary artifact.** RLP1 used legacy `shuffle`; RLP2+ used `hash_stable`. RLP2's fresh-control drop (−0.00253 vs RLP1_01) has an unquantified split-mode component RLP2 couldn't separate from mutable-source drift. See RLP1.md and RLP2.md.
- **Preprocessing-parity note.** These worktrees predate commit `855871e` (2026-04-24, `cv2.INTER_AREA → cv2.INTER_LINEAR` in `arena/model_arena.py:472` and `batch_inference_gcs.py:407`). The WT-* artifacts carry no numbers so the bug doesn't invalidate them — but WT-E scaffolded the arena retro-score path that later got the fix, and every pre-fix promotion-contract scorecard ran on drifted preprocessing. See [../threads/preprocessing_parity_bug.md](../threads/preprocessing_parity_bug.md).

## Source files

- **Handoffs**: `docs/relaunch_handoffs/TASK_BOARD_2026-04-17.md:15-49`; `WT-A_2026-04-17.md:45-65`; `WT-B_2026-04-17.md` (draft-only) + `WT-B_2026-04-18.md:25-90`; `WT-C_2026-04-17.md:62-160`; `WT-D_2026-04-17.md:20-56`; `WT-E_2026-04-17.md:8-70`; `WT-F_2026-04-17.md:8-87`; `WT_B_AND_NEW_DATA_READINESS_2026-04-19.md:57-147`; `NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md:340-510`; `RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md:168-416`. Policy source: `docs/VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md:36-149`. Runbooks: `docs/R13_WTB_WEAK_SIGNAL_RUNBOOK_2026-04-18.md`, `docs/R13_WTB_WEAK_SIGNAL_DRAFT_RUNBOOK_2026-04-17.md` (superseded), `docs/WT_D_MINIMAL_BASELINE_PACKET_2026-04-17.md:19-93`. Pre-WT lineage: `docs/TRACK_A_HANDOFF_2026-04-06.md`, `docs/TRACK_B_MATCHED_PAIR_REPORT_2026-04-07.md`, `docs/TRACK_C_SCORECARD_RUNBOOK_2026-04-07.md`, `docs/TRACK_C_PROMOTION_CONTRACT_RUNBOOK_2026-04-17.md:1-80`.
- **Yamls**: WT-B — `experiments/phase2_round13/R13_WTB{1,2,3}_*.yaml`, `R13_SMOKE_WTB3_*.yaml`, `R13_STARTUP_SMOKE_WTB3_*.yaml`; WT-C — `R13_WTC{1,2,3}_*_sidecar.yaml`; WT-F (April-19) — `R13_STARTUP_SMOKE_PROPER_DATA_WTF_PROVISIONAL.yaml`. WT-A / WT-D / WT-E landed no training yamls; WT-E shipped suite-manifest and checkpoint-map yamls under `arena/`.
- **Scorecards / analysis**: `arena/target_domain_suites.teams_promotion_contract_2026-04-17.yaml`; `arena/checkpoint_maps/teams_target_domain.promotion_shortlist_2026-04-17.yaml`; `policy/visomaster_bad_data/`; `docs/relaunch_handoffs/WT-A_policy_truth_artifact_2026-04-17.json`; provisional proper-data artifacts under `arena/inventories/`, `arena/manifests/`, `arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml`, `arena/reports/proper_visomaster_wave_2026_04_19_provisional_build_report.json`.
- **Memory pointers**: `project_promotion_contract.md` (WT-E + WT-D authoritative τ readout); `project_contract_policy_bug.md` (FPR-minimization bug lives in the WT-E/WT-D contract layer); `feedback_promotion_contract_launch.md` (launcher env-var requirement for WT-E wrapper).
