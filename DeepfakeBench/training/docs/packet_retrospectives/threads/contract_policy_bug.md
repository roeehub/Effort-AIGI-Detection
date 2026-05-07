# Thread: Contract policy bug — the FPR-minimization no-budget τ-tail-collapse

> **Flagship thread for this wiki**. The pattern this whole knowledge base was built to prevent surfaces here in concentrated form: **the same contract-policy bug was diagnosed and "fixed" three times in 6 days, with all three fixes either lost or uncommitted**. Memory `project_contract_policy_bug.md` (originSession `f8af57d7-46c1-41b1-a6de-2a32f4bd62fb`) is the auto-memory anchor; this thread is the deliberated synthesis.

## The question

The lexicographic τ-selection in `arena/score_teams_promotion_contract.py:_threshold_sort_key` minimizes `dev_primary_real_fpr` first **with no recall budget**. On a sharp-prediction model (Effort with confident-fake p90 ≈ [0.94, 0.99]), τ snaps to ~0.995 and reported recall craters by ~10–30× vs the trainer's W&B numbers. The same scorecard "passes" because no recall floor blocks promotion. The question this thread answers: **why has the fix been re-discovered three times across 6 days without ever being committed**, and what observable surface would prevent a fourth rediscovery?

## Initial belief

WT-D / WT-E (Slice 1) shipped the lexicographic-τ + eight-suite-scorecard machinery on the assumption that "minimize FPR with no budget" was a sane policy because the eight-suite-manifest gate would impose effective constraints upstream. WT-E (`docs/relaunch_handoffs/WT-E_2026-04-17.md:55-63`) explicitly **deferred** emitting `promotion_winner.json` on the five-checkpoint shortlist, and WT-D (`WT-D_2026-04-17.md:42-44`) admitted *"this runtime still cannot name a calibrated five-checkpoint promotion winner because the shortlist report artifacts are not locally accessible here."* The structural bug was thus present in merged code from 2026-04-17 onward, but was unobserved because no calibrated readout had been produced yet.

## What changed our mind

- **2026-04-23 — Attempt #1 (first live evidence; first rediscovery).** Slot-07 sanity check on the calibrated promotion contract is the first time the τ-tail-collapse fires in the wild on a real artifact (carrier `arena/checkpoint_maps/teams_target_domain.r13_rlp5_07_sanity_2026-04-23.yaml`). Selected `τ = 0.9953`, dev real FPR = 0.0 across all 3 real suites, but recall craters: `teams_fake_all_dev = 27.6%`, `visomaster_enhanced_macro_dev = 1.6%`, `deeplive_enhanced_dev = 0.0%`, `lockbox_fake_recall = 15.8%`, `dev_fake_macro_recall = 9.7%`. **Trainer-side `value_composite=0.7736` vs contract-side macro recall 9.7% on the same checkpoint, same data, different τ.** The bug is not yet diagnosed in-session — the readout was filed as "expected sanity, slot-07 passes the contract on FPR axis" and the diagnostic energy moves to RLP6 gate-alignment ([promotion_contract_evolution.md](promotion_contract_evolution.md) Slice-4 evidence block; memory `project_contract_policy_bug.md` "concrete evidence — slot-07 sanity check 2026-04-23" block). **No fix lands.** The first live evidence existed for 4+ days before anyone framed it as a bug.

- **2026-04-27 09:50 UTC — Attempt #2 (codec-hedge readout surfaces the symptom; PCP filed; not actioned in-session).** The codec_hedge A.2-style validation (`april-26-training-master-plan-v2.LOG.md:390-422`) ran 4 checkpoints and **all 4 hit the contract policy bug regime (τ ≈ 0.97–0.99)**. The agent surfaces a PLAN CHANGE PROPOSAL with three forks: (i) raise the FPR gate, (ii) **fix the contract policy bug first** ("Without this fix, no calibrated number will be trustworthy"), or (iii) commit to the §9 "no candidate" branch. User authorizes fork (ii) end-to-end revalidation.

- **2026-04-27 14:55 UTC — Attempt #2 (continued; the load-bearing dual-attempt session).** The user-authorized fork-(ii) work re-scores the codec_hedge reports under an uncommitted policy fix in `arena/score_teams_promotion_contract.py` (`+61/-16`) — the patch added the `target_real_fpr` / `target_stress_fpr` budget infrastructure with a tier-based sort (budget-OK > budget-OK-but-floor-failed > budget-violated) (`april-26-training-master-plan-v2.LOG.md:425-495`). Two variants written to `analysis/policy_reruns_2026-04-27/{default,recall_floor_30}/`: under `recall_floor=0.30`, P8A's macro_recall (0.300) sits exactly at the floor → byte-identical to the no-floor "default" variant, so the floor isn't binding for this slate. The agent reports *"Policy fix is end-to-end validated under the 7%/10% FPR budget. Tier-based sort produces deterministic, plan-aligned τ selection. The contract-policy-bug regime is closed for this configuration. Defaults bumped at the source level so future runs auto-use the corrected policy."* **The fix is not committed in this session.** The agent's own next-step note (`:489-494`) explicitly leaves staging for the user: *"If user wants the policy fix committed: stage `arena/score_teams_promotion_contract.py` plus the four untracked files in `analysis/probe_battery_2026-04-26/` …"*. **The agent then changes machines / sessions; the smoke-tested fix is lost to /tmp wipe** — confirmed by the 2026-04-29 session header (memory `project_contract_policy_bug.md` "Status 2026-04-29: FIXED IN WORKING TREE, uncommitted" — the fix had to be re-derived at attempt #3). The **same** agent re-establishes the result in fork (ii) durable-storage form a few hours later in the same LOG entry. So 2026-04-27 contains both a transient and a durable form of attempt #2; neither is committed.

- **2026-04-27 17:26 → 2026-04-28 13:30 (Slice 6 spans).** The contract-policy bug is **explicitly cited as a known regime** in every subsequent contract scorecard readout: codec-hedge (04-27 09:50, τ=0.97-0.99), Phase A.2 v3 retro from Slice 5 (memory `project_contract_policy_bug.md` reaffirmed), P11 day-2 inference (04-28 11:53, τ ≈ 0.99 across all 3 P11 step-1000 checkpoints + P8A reference). Each readout falls back to **diagnostic τ=0.5** for promotion thinking. The whack-a-mole pattern has now turned into **"always check `selected_threshold`; if τ ≈ 0.99, defer to τ=0.5"** — operator discipline absorbing what a contract-yaml-level fix should have prevented. **No fix lands in Slice 6.**

- **2026-04-29 — Attempt #3 (the design that becomes the v3 fix; memory says "fixed", working tree says "uncommitted").** Memory `project_contract_policy_bug.md` describes the post-2026-04-29 state: `target_fake_recall_min=0.30` + `target_real_fpr=0.07` + `target_stress_fpr=0.10` is the corrected policy; test coverage added in `tests/test_score_teams_promotion_contract.py` (2 new tests, 4 contract tests pass total). **Re-scoring P13 reports under the new policy** moves headline numbers dramatically: P8A_REFERENCE_STEP5000 becomes rank 1 (was rank 4 under old policy); selected τ 0.916 (was ~0.992); viso_enhanced_macro_dev recall 1.1% → **13.6%**; deeplive_enhanced_dev recall 1.6% → **23.9%**; teams_fake_all_dev recall ~5% → **52.6%**; lockbox real FPR 0.4% → 1.8% (still well within the 5% budget). **The fix is in the working tree, uncommitted as of today.** Verified by `git diff --stat`: `arena/score_teams_promotion_contract.py | 98 +++++++++++++---` and `tests/test_score_teams_promotion_contract.py | 125 +++++++++++++++++++++` show 207 net added lines across the two files (working-tree state, 2026-04-29). The implementation lives at `arena/score_teams_promotion_contract.py:456-479` (`_threshold_sort_key`) — the patch adds `budget_active` and `recall_floor_active` flags, gates τ selection on both, and pushes budget/floor violations to the bottom of the sort. Slice 7 is the slice that captures the v3 design conversation.

### Slice 7 — V3 fix design and uncommitted state (2026-04-29)

This subsection documents the precise state of attempt #3 as of end-of-Slice-7. **The fix is designed, tested, validated, and uncommitted.** A future agent reading this thread should not re-derive any of this; they should commit the existing diff.

#### What is in the working tree

Verified via `git diff --stat`:

```
arena/run_target_domain_validation_sequential.py                |  19 +
arena/score_teams_promotion_contract.py                          |  98 ++++++++++++++---
tests/test_score_teams_promotion_contract.py                     | 125 +++++++++++++++++++++
```

The patch consists of three logically-distinct changes that ship together:

1. **`arena/score_teams_promotion_contract.py:_threshold_sort_key` (lines 456-479).** The within-checkpoint τ selector. Adds:
   - `budget_active` flag (True when `target_real_fpr < 1.0` AND `target_stress_fpr < 1.0`).
   - `recall_floor_active` flag (True when `target_fake_recall_min > 0.0`).
   - Three-tier sort: tier 0 satisfies both budgets, tier 1 satisfies budgets but fails recall floor, tier 2 violates budgets. Within each tier: maximize `dev_fake_macro_recall`, then prefer higher τ (more conservative), then prefer lower primary FPR.
   - Default `target_fake_recall_min = 0.0` preserves backward compatibility (legacy budget-only behavior when not explicitly set).
   - Defaults bumped: `target_real_fpr = 0.07`, `target_stress_fpr = 0.10` (operational acceptance level per 2026-04-27 codec_hedge readout). The prior 0.02/0.05 budgets were themselves unreachable below τ ≈ 0.99 on a sharp-prediction model, so the recall-floor without the budget bump would not have closed the regime.

2. **`arena/score_teams_promotion_contract.py:_promotion_summary_sort_key` (cross-checkpoint ranker, lines 510-531).** This is a **second** sort point that the prior fix attempts had not addressed. Without tiering at the cross-ckpt summary level, the cross-ckpt ranker picks the lowest-`lockbox_real_fpr` ckpt regardless of how degenerate its recall is — which is exactly how `P13_step2000` (3.6% lockbox fake recall, 2.9% dev fake macro recall) was crowned rank-1 on 2026-04-29 (`docs/relaunch_handoffs/P13_DAY4_VERDICT_2026-04-29.md` § "τ-Degeneracy Caveat"). The patch adds tier 0 / tier 1 by recall-floor satisfaction at the summary layer too; the load-bearing P8A_REFERENCE rank-1 promotion is mediated by this second sort point as well as the within-ckpt one.

3. **`arena/run_target_domain_validation_sequential.py` (+19 lines).** Adds three new CLI args to the validation runner: `--promotion_target_real_fpr` (default 0.02 to preserve runner backward compat — note this differs from the contract's bumped 0.07 default), `--promotion_target_stress_fpr` (default 0.05), `--promotion_target_fake_recall_min` (default 0.0). Plus `--promotion_readout_only_suites` (default `teams_real_dor_dev`). These plumb through to `ContractConfig` at the validation runner's contract-yaml emission point. **The default 0.0 floor in the runner is intentional**: the runner is the in-flight scorecard producer, where the operator is expected to pass the floor explicitly via `--promotion_target_fake_recall_min 0.30`. Memory `feedback_promotion_contract_launch.md` is the reference for the launch invocation pattern.

#### What tests exist

- `tests/test_score_teams_promotion_contract.py:test_promotion_summary_sort_key_demotes_low_recall_with_floor` — unit test on the cross-ckpt ranker. Constructs a "degenerate" ckpt (lockbox_real_fpr=0, lockbox_fake_recall=0, dev_fake_macro_recall=0.05) and a "healthy" ckpt (lockbox_real_fpr=0.05, lockbox_fake_recall=0.50, dev_fake_macro_recall=0.50). Asserts: (a) without floor, the degenerate wins on lower lockbox_real_fpr; (b) with floor=0.30, the healthy wins despite higher lockbox_real_fpr (degenerate sits in tier 1, healthy in tier 0).
- `tests/test_score_teams_promotion_contract.py:test_recall_floor_changes_winner_in_score_promotion_contract` — end-to-end integration test. Same fixtures, two policies. Without the recall floor the degenerate ckpt wins (lower lockbox_real_fpr); with the floor the healthy ckpt wins (only it satisfies the floor). Confirms the within-ckpt + cross-ckpt sort points work together.
- 2 new tests + 4 prior contract tests = **6 tests total, all passing locally per memory `project_contract_policy_bug.md`**.

#### What is NOT yet committed

- The 207-net-added-line diff across the two `arena/` files + the test file. **None on `teams-relaunch-root-2026-04-17` as of 2026-04-29.**
- Any in-image artifact of the fix. Per memory `reference_image_rebuild.md`: an image rebuild (`./dev.sh build-prod -y`, auto-bumps VERSION patch) is required for the fix to ship to a Vertex contract scorecard run. As of 2026-04-29, VERSION is at 1.3.229 (working tree, also uncommitted) — image build status not verified end-to-end with the fix in.
- A re-scored P-series scorecard run (e.g. P8A reference step 5000) explicitly documented as having selected τ via the recall-floor path. Memory cites the in-place P13 re-score (`/tmp/p13_repolicy/recall_floor_30/`) showing P8A becoming rank-1 with τ=0.916, but a durable artifact + scorecard run does not exist.

#### What would close the loop

The structured open-loop block below names the close criterion. In one sentence: **commit the diff, rebuild the image, run a representative P-series scorecard with `--promotion_target_fake_recall_min 0.30`, document the τ selection.** The first half is mechanical; the second is a ~3-hour Vertex run. The reason this has not happened across three "fix" attempts in 6 days is documented in the memory and in this thread's narrative — design is correct, ship-discipline is the failure mode.

## Current stance (2026-04-29)

The bug is structural in `arena/score_teams_promotion_contract.py:_threshold_sort_key` (lines 456-479) and has been recurring since at least 2026-04-23. Three "fix" attempts in 6 days have not produced a committed change to the deployment artifact:

1. **2026-04-23**: symptom observed (slot-07 sanity), not framed as a bug.
2. **2026-04-27**: fix designed, smoke-tested, end-to-end validated locally + on durable storage; the working-tree state was lost in a /tmp wipe between sessions (memory `project_contract_policy_bug.md` traceback confirms attempt #3 had to re-derive what attempt #2 had).
3. **2026-04-29 (Slice 7)**: fix re-derived with the recall-floor design, with regression tests; **uncommitted in working tree as of today**. The diff is +207 net lines across `arena/score_teams_promotion_contract.py`, `arena/run_target_domain_validation_sequential.py`, and `tests/test_score_teams_promotion_contract.py`. Six contract tests pass locally. Re-scored P13 reports under the corrected policy lift P8A from rank-4 to rank-1; viso recall 1.1% → 13.6%, deeplive 1.6% → 23.9%, teams_fake ~5% → 52.6%; lockbox real FPR 0.4% → 1.8% (still well within the 5% budget). See "Slice 7 — V3 fix design and uncommitted state" subsection above for the full present-tense status: what is in the working tree, what tests exist, what is not yet committed, and what would close the loop.

The pattern is **"design lands but doesn't ship to deployment artifact"**. Every readout post-2026-04-23 inspects `selected_threshold` first and falls back to τ=0.5 if it's in the τ-tail-collapse band — operator discipline absorbing what a code commit would resolve. **The wiki's anti-whack-a-mole purpose is exactly this case**: the open loop `contract-policy-bug-fix-not-committed` (below) is the surface that should now prevent attempt #4. The frame-level AUC reframing in Slice 7 ([`processing_signature_shortcut`](processing_signature_shortcut.md), memory `project_p8a_frame_level_auc_2026-04-29.md`) makes the load-bearing case material: the bug had been hiding ~10-30× of the model's actual cross-domain recall under the buggy τ-policy. Every P-series scorecard read pre-fix is a *lower bound* on actual deployment performance.

## Packet timeline

- [WT-infrastructure](../packets/WT_infrastructure.md) — WT-D / WT-E (2026-04-17) shipped the `_threshold_sort_key` machinery without the recall budget; the structural bug landed here.
- [RLP3](../packets/RLP3.md), [RLP3.5](../packets/RLP3_5.md) — retro τ values 0.96-0.97; not in the 0.995 collapse band; symptom not yet visible.
- [RLP5](../packets/RLP5.md) — slot-07 sanity (2026-04-23) is the **first live evidence** of the τ-tail-collapse. Filed as "expected" in-session.
- [RLP6](../packets/RLP6.md) — RLP6_04 with `value_composite=0.9006` is contract-legal under the corrected gate but unshippable (lockbox 90/90 not threshold-reachable). The contract-policy bug masks the deployment block in current scorecards; both must close together.
- [P8A](../packets/P8A.md) — 2026-04-25 P8A scorecard τ=0.991, RLP6_04 τ=0.992 (per `PACKET_9_MID_FLIGHT_HANDOFF_2026-04-26.md:84-91`). Symptom keeps firing.
- [P9](../packets/P9.md) — pre-launch crowning protocol explicitly excludes `value_composite`; first packet to gate on lockbox per-method recall floors as a workaround.
- [P10](../packets/P10.md) — Phase A.2 v3 retro at τ ≈ 0.99 (`april-26-training-master-plan-v2.LOG.md:340-343`); recommendation "defer to diagnostic τ=0.5 readout" is operator discipline absorbing the unfixed bug.
- [P11](../packets/P11.md) — 2026-04-28 11:53 P11 step-1000 inference verdict ran with the unfixed contract; the codec-hedge fork-(ii) revalidation (2026-04-27) produced an uncommitted recall-floor variant of the policy fix. **Attempt #2 lives here.**
- [P12](../packets/P12.md) — P12_HEAVY_LONG dud + `periodic_saves` silent failure; doesn't directly touch the contract policy but demonstrates the same "added-but-not-firing" pattern at the trainer layer (see [`periodic_saves` open loop](#open-loop-periodic-saves-silent-failure) below).
- *(Slice 7)* — Attempt #3 (recall floor design, the v3 fix) — that's where the test coverage lands and the headline P8A re-rank to rank-1 happens.

## Evidence locations

- `arena/score_teams_promotion_contract.py:456-479` — `_threshold_sort_key` (the bug; the fix in the working tree adds `budget_active` / `recall_floor_active` gates + tier-based sort)
- `tests/test_score_teams_promotion_contract.py` — working-tree test additions for the recall-floor regression test (uncommitted)
- `arena/checkpoint_maps/teams_target_domain.r13_rlp5_07_sanity_2026-04-23.yaml` — first-evidence carrier (the slot-07 sanity readout)
- `analysis/policy_reruns_2026-04-27/{default,recall_floor_30}/` — attempt #2 durable-storage variants (codec-hedge revalidation under the policy fix)
- Master plan LOG sub-sections:
  - `april-26-training-master-plan-v2.LOG.md:354-389` — 2026-04-27 07:18 UTC probe + codec_hedge launch
  - `:390-422` — 2026-04-27 09:50 UTC codec-hedge readout + PCP fork (i)/(ii)/(iii)
  - `:425-495` — 2026-04-27 14:55 UTC fork (ii) end-to-end revalidation (attempt #2 dual form)
  - `:1275-1336` — 2026-04-28 11:53 P11 verdict β + symptom continues firing
- Handoffs: *(none direct in Slice 6 — the contract-policy bug is referenced across multiple handoffs but no handoff is dedicated to it; the WANDB_FLATTENING handoff is contemporaneous and demonstrates the parallel "added-but-not-firing" pattern)*
- Memory: `project_contract_policy_bug.md` — auto-memory anchor with the current `target_fake_recall_min=0.30` policy and the pre-fix slot-07 sanity numbers preserved; `project_promotion_contract.md` — the lockbox-anchored deployment readout the contract is supposed to gate
- Threads: [`promotion_contract_evolution`](promotion_contract_evolution.md) — the broader contract-evolution arc; this thread is the bug-class-specific deep-dive

## Open loops

### Open loop: contract-policy-bug-fix-not-committed
status: resolved
severity: high
first_seen: 2026-04-23
last_verified: 2026-05-07
close_criterion: the recall-floor + budget-aware τ-selection patch is (1) committed to `teams-relaunch-root-2026-04-17`, (2) the image is rebuilt with the fix in, and (3) a contract scorecard run on a representative recent checkpoint is invoked with `--promotion_target_fake_recall_min 0.30` AND the resulting scorecard is documented as having selected τ via the recall-floor path (not via the legacy no-budget minimize-FPR-only path). All three components are required. **2026-04-30 evening update**: the `mclioexb` scorecard run did NOT exercise the v3 fix — the launcher (`arena/launch_teams_promotion_contract.sh`) omitted `--promotion_target_fake_recall_min`, so the scorecard ran the DEFAULT policy (τ=0.975, recall=0.13). Component 3 still NOT MET. **2026-05-07 update — components 1 and 2 are MET; component 3 is in flight.** Audit of the tree (commit `974e033` 2026-04-29) shows the scorer code with `target_real_fpr=0.07 / target_stress_fpr=0.10 / target_fake_recall_min=0.70` defaults was committed a week ago; the runner with `--promotion_target_*` CLI args was committed at the same time. The actual missing piece was the launcher: `arena/launch_teams_promotion_contract.sh` did not pass these flags through. Today's commits `ad070d3` (passing the flags + setting v3 defaults `0.07 / 0.10 / 0.30`) and `7f81e7a` (canonicalizing the 500GB scorecard template to prevent disk-exhaustion) close the launcher gap. Image `1.3.270` (commit `5dccfa4`, Cloud Build `40cf4d7c-4b8b-4af7-81ca-0991f2083450`) bakes everything in. The Phase A scorecard run for P1 (Vertex job `7995519158412378112`, us-east1, submitted 2026-05-07T08:19:20Z) explicitly invokes `--promotion_target_fake_recall_min 0.30` — verifiable in the gcloud `containerSpec.args` trace. Component 3 closes when the resulting `promotion_winner.json` shows τ selected via the recall-floor path; verdict pending Phase A finish (~12:30 UTC).

The whack-a-mole pattern: same bug, four "fix" attempts, **first three were not committed** to the deployment artifact:

1. **2026-04-23 (attempt #1)** — symptom observed (slot-07 sanity, τ=0.9953, recall 9.7%); not framed as a bug; no fix.
2. **2026-04-27 (attempt #2)** — fix designed, smoke-tested, end-to-end validated locally + on durable storage (`analysis/policy_reruns_2026-04-27/`); transient version lost to /tmp wipe; durable version not committed; same agent re-derives within hours, same session.
3. **2026-04-29 (attempt #3, Slice 7)** — fix re-derived with `target_fake_recall_min=0.30` recall-floor design + 2 new tests + budget defaults bumped 0.02/0.05 → 0.07/0.10 + cross-ckpt summary sort tiering; the runner gets new CLI args. **Committed as `974e033` ("Promotion contract: default recall floor to 0.70 (Block A)") on 2026-04-29 20:57:12 +02:00**, but the entry above remained worded as "uncommitted in working tree" until today's audit. Source-of-truth drift: the OPEN_LOOPS text was stale by a week.
4. **2026-05-07 (attempt #4)** — launcher gap closed. The scorer + runner had been committed since attempt #3, but the launcher was passing default flags (legacy `0.02 / 0.05 / 0.70`) instead of the v3 design (`0.07 / 0.10 / 0.30`). Today's commits `ad070d3` + `7f81e7a` thread the flags through the launcher; image `1.3.270` rebakes; Phase A scorecard exercises them.

5. **2026-05-07 evening (closure)** — Phase A scorecard SUCCEEDED 2026-05-07 14:40 UTC. `promotion_winner.json` shows `selected_threshold = 0.76772` for `P1_PAIRRANK_PERIODIC_STEP500` (rank-1) — well below the 0.99x τ-tail collapse band; the recall-floor tier mechanism worked as designed. The forensic on `arena/score_teams_promotion_contract.py:460-510` (`_threshold_sort_key`) confirmed the floor is **TIERED, not GATED**: tier 0 = budget-OK + recall ≥ floor; tier 1 = budget-OK + recall < floor; tier 2 = budget-violated. For ckpts with at least one tier-0 candidate (PAIRRANK_step500 has 117), the contract correctly selects from tier 0. For ckpts with NO tier-0 candidates anywhere in the threshold grid (BUNDLE_step3750/step4000 have 0 of ~5300 grid points achieving recall ≥ 0.30 within FPR ≤ 0.07), the contract correctly demotes them to tier-1 selection at the highest available τ — a property of those ckpts' degenerate ROC, not a contract bug. **All three components of the close criterion are MET**: (1) commit `974e033` (2026-04-29) shipped the recall-floor scorer + CLI args, (2) image `1.3.270` (2026-05-07) bakes the flagged launcher in, (3) the Phase A `promotion_winner.json` documents τ=0.768 selected via the tier-0 recall-floor path. Loop closes. Source: `analysis/p1_pe_eval_2026-05-07/scorecard/promotion_winner.json`, `analysis/p1_pe_eval_2026-05-07/roc_degeneracy/ROC_DEGENERACY_FACTS_2026-05-07.md`.

The structural failure mode this thread should henceforth track is **NOT** "fix is uncommitted" (resolved 2026-04-29) but **OPEN_LOOPS source-of-truth drift**: an entry with a `last_verified` date can describe tree state that no longer matches reality. The `regenerate_open_loops.py` tool propagates the entry text but does not re-verify file-state claims (e.g., "uncommitted", "in working tree"). See follow-up loop `open-loops-stale-state-claims` for the systemic fix.

This loop is **paired with `fpr-minimization-no-budget-tau-collapse`** in [`promotion_contract_evolution`](promotion_contract_evolution.md) — that loop tracks the bug itself; this loop tracks the *commit-discipline* failure mode. They close together when component 3 reads positively on the Phase A scorecard.

### Cross-thread refs

- [`promotion_contract_evolution`](promotion_contract_evolution.md) — the parent thread; the `fpr-minimization-no-budget-tau-collapse` open loop is the bug-itself surface, this thread is the bug-class deep-dive. Both loops are "in-progress" / "open" because the fix is not committed.
- [`processing_signature_shortcut`](processing_signature_shortcut.md) — the contract-policy bug currently *masks* the deployment block (the camera/ISP shortcut). A contract fix without a representation fix surfaces the shortcut at the lockbox readout. The two threads' open loops `shortcut-deployment-block` (critical) + `contract-policy-bug-fix-not-committed` (high) must both close for ship-readiness.
- [`wandb_flattening`](wandb_flattening.md) and [`wandb_yaml_propagation_bugs`](wandb_yaml_propagation_bugs.md) — share the meta-pattern "added-but-not-firing" / "fixed-but-not-shipped" with this thread. Slice 6 saw all three patterns recur in concentrated form.
