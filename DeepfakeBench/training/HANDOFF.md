# Handoff — Phase 1 + Phase 2 done. P18 yamls drafted, code-blocked.

**Generated**: 2026-05-01 19:30 CEST (updated; P17 verdict generated 18:00).
**Branch**: `teams-relaunch-root-2026-04-17`
**Status**: P17 chapter complete. Phase 1 diagnostic battery (CPU only) done. Phase 2 readiness work done; contract v3 verify on Vertex (job `5427016015462531072` us-east1, RUNNING since 19:18, ETA ~3h). P18 (next packet) yamls drafted, blocked on ~½ day code changes. No Phase 3 launch yet — pending user authorization on the code work.

> **READ FIRST**: [`docs/relaunch_handoffs/PHASE1_2_COMPLETE_STATUS_2026-05-01.md`](docs/relaunch_handoffs/PHASE1_2_COMPLETE_STATUS_2026-05-01.md) — comprehensive status, all decisions, what to do next.
>
> Tier-1 supporting docs (in order):
> - [`docs/relaunch_handoffs/P17_FINAL_VERDICT_2026-05-01.md`](docs/relaunch_handoffs/P17_FINAL_VERDICT_2026-05-01.md) — canonical P17 record.
> - [`docs/relaunch_handoffs/PHASE1A_SUBSTRATE_DIRECTION_FINDING_2026-05-01.md`](docs/relaunch_handoffs/PHASE1A_SUBSTRATE_DIRECTION_FINDING_2026-05-01.md) — capture-mode is the wrong axis; identity/method-cluster (`is_dor_shkedi`/`is_deeplive_enhanced`) is right.
> - [`docs/relaunch_handoffs/PHASE1_SYNTHESIS_2026-05-01.md`](docs/relaunch_handoffs/PHASE1_SYNTHESIS_2026-05-01.md) — three-probe synthesis (1A + Move 1 + Move 1.5 + 2B).
> - [`docs/relaunch_handoffs/PHASE2D_CODE_CHANGE_CHECKLIST_2026-05-01.md`](docs/relaunch_handoffs/PHASE2D_CODE_CHANGE_CHECKLIST_2026-05-01.md) — what code has to land before P18 launch.
>
> Memory entry: `project_p17_trained_head_destroys_substrate_invariance.md` (P17 result; future memory should also capture the Phase 1A finding).

## Onboarding reading list (for an agent picking this up cold)

This handoff is scoped to the latest concluded experiment (P17). To reason about *next moves* you need broader context. Read in this order:

### Tier 1 — Must read (~10 min)
1. `HANDOFF.md` (this file) — current state + decisions pending.
2. `docs/relaunch_handoffs/P17_FINAL_VERDICT_2026-05-01.md` — latest verdict, full data, candidate next-move categories.
3. `CLAUDE.md` (root) — operational rules (region preference, capacity playbook).
4. `~/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/MEMORY.md` — persistent project memory index. Load-bearing pointers for current state: `project_p17_*`, `project_p16_*`, `project_p14_*`/`project_mclioexb_*`, `project_p8a_*`, `project_signature_shortcut_finding`, `project_face_size_label_leak`, `project_eval_production_crop_tightness_gap`, `project_visomaster_hints_lanes_bad_data`.

### Tier 2 — Strategic context (~20 min)
5. `PLAN.md` (root, 2026-04-29) — current R13 Forward Plan with citation discipline. P15/P16/P17 packets were instances of executing parts of it.
6. `docs/relaunch_handoffs/P16_DATA_AXIS_VERDICT_2026-04-30.md` — last verdict before P17. Data-axis lever does not promote. Establishes the "single-lever failure" pattern.
7. `docs/relaunch_handoffs/P13_DAY4_VERDICT_2026-04-29.md` — earlier verdict in the chain. Recurring pattern of trainer-side levers failing the deployment-grade test.
8. `docs/relaunch_handoffs/P15_GRL_READINESS_NOTE_2026-04-29.md` — readiness for substrate-adversarial training (Option B in the P17 verdict). Already partially scoped.
9. `docs/relaunch_handoffs/TRAINING_EVAL_VISO_BUCKET_GAP_2026-04-29.md` — concrete substrate-axis evidence for one upstream axis the P17 verdict says now needs addressing.

### Tier 3 — Deep dive (on demand)
10. `docs/relaunch_handoffs/OVERNIGHT_LAYER3_FINDING_2026-05-01.md` and `P17_TRAINED_HEAD_FINDING_2026-05-01.md` — P17's precursor docs. Superseded but contain methodology depth.
11. `docs/relaunch_handoffs/R13_RELAUNCH_TRAINING_SOURCE_OF_TRUTH_2026-04-21.md` — original R13 wiring/architecture reference.
12. `analysis/intermediate_layer_probe_2026-04-30/trajectory_and_direction_2026-05-01.py` + `outputs/*.json` — the actual P17 analysis code and raw data, if reproducing or extending.

## TL;DR

P17 closed the layer-X readout chapter. Phase 1's CPU diagnostic battery + Phase 2's readiness work re-pointed the next-packet design:

1. **The trained head's modal axis is `is_dor_shkedi`/`is_deeplive_enhanced` cluster, NOT capture-mode.** P15 GRL @ static λ=0.20 didn't bite because it was reversing a gradient the encoder wasn't using. Ramped λ on capture-mode is also doomed.
2. **Eval substrate FPR underestimates production FPR by ~10–13 pp pooled, +40 pp on webcam.** Move 1.5 (190 frames) and 2B retag (839 frames) replicate at scale; modern_v2 absorbs to +2.14 pp; `PC_Generator__s15` reals jump 20.69%→89.66% under tight crops.
3. **The viso bucket gap is identity-confounded.** Move 1 grouped probe: bucket AUC 0.916 vs identity-only AUC 0.987 → AMBIGUOUS verdict. P14_DATA_FIX-style bucket lever has no probe support.
4. **Contract policy v3 already committed** (974e033, 2026-04-29; default flipped to 0.70). Image at 1.3.239 includes it. Vertex verify in flight (~3h ETA).
5. **P18 yamls drafted** (`R13_P18_METHOD_DOMAIN_GRL.yaml` treatment + `R13_P18_NO_GRL_CONTROL.yaml` control) — uses the 12-bucket method-domain map from Phase 2C audit, splits `deeplive` into basic/enhanced/teams, isolates the `is_deeplive_enhanced` axis Phase 1A pinpointed. **Blocked on ~½ day code changes** in `combined_paired.py`, `effort_detector.py`, plus a test-import fix per PLAN.md §10.7.

**Recommended next session**: do the code work in `PHASE2D_CODE_CHANGE_CHECKLIST_2026-05-01.md`, run pre-launch CPU smoke gates, then submit P18 + control to Vertex (~$120 / 1 day).

## Completed this session

- [x] Diagnosed and fixed three bugs in P17 launch (yaml-schedule-keys, frozen-svd-residuals, hidden_size override). Image rebuilt to 1.3.239.
- [x] Launched 2 ArcFace arms (L3 + L4) — both finished cleanly at step 10000.
- [x] Launched 2 LINEAR companion arms (L3 + L4) — cancelled at step ~1450 after probe verdict.
- [x] Wrote and ran trained-head probe on ArcFace + LINEAR ckpts.
- [x] Wrote trajectory + bootstrap + direction analysis (`trajectory_and_direction_2026-05-01.py`).
- [x] **Decisive findings**:
  - ArcFace L3 lockbox AUC trajectory: ep1 = 0.7229 → step 1687 = 0.0745 → step 2088 = 0.1867
  - LINEAR L3 lockbox AUC trajectory: ep1 = 0.7117 → step 1044 = 0.0431 → step 1285 = 0.0660
  - Pearson r between trained-head and fresh-LR scores on lockbox: −0.20 to −0.41 (all trained ckpts)
  - Cosine sim between fresh-LR direction and trained-head decision direction: +0.03 to +0.09 (orthogonal)
  - Bootstrap 95% CIs on all 6 trained ckpts entirely below 0.5
- [x] Cancelled both LINEAR Vertex jobs (`6468843621113135104`, `5009677341845094400`) — user authorized.
- [x] Wrote memory entry + MEMORY.md pointer.
- [x] Wrote `docs/relaunch_handoffs/P17_FINAL_VERDICT_2026-05-01.md` (full picture for cold reading).
- [x] Updated this HANDOFF.md.

## Pending decisions for the user (NOT to be made unilaterally)

1. **Strategic pivot direction.** Three candidate categories in `P17_FINAL_VERDICT_2026-05-01.md` §"What's NOT yet ruled out":
   - **A.** Fix the data signal directly (substrate-balanced training data, substrate-leaky data removal)
   - **B.** Substrate-adversarial training (P15 GRL territory)
   - **C.** Substrate-matched eval infrastructure (cheapest, gives honest measurement)

   My read: **C → B → A** sequencing. C is cheap and gives us deployment-relevant visibility. B addresses root cause architecturally. A is the long arc.

2. **Whether to commit the working tree.** Per project pattern, kept uncommitted. P17 spans many files (detector, trainer, configs, analysis scripts, docs). Would be a large commit if so.

3. **Whether to free up `/tmp/p17_ckpts/` (~9 GB).** Local cache only; downloadable from GCS at any time.

4. **ArcFace promotion-contract scorecard** — held by user. Reversible. Probe says lockbox AUC of 0.187 means no τ in the contract grid will rescue a model that ranks fakes BELOW reals on lockbox; saved ~$5-15 by holding.

5. **(Optional)** Pipeline confidence boost: replicate trainer's val-in-dist AUC via `effort_detector.py` to resolve the 0.65-vs-0.74 within-dev discrepancy. Not load-bearing for the verdict.

## Files modified / created this session

```
A  analysis/intermediate_layer_probe_2026-04-30/eval_trained_heads_2026-05-01.py (orig probe, modified to support LINEAR)
A  analysis/intermediate_layer_probe_2026-04-30/trajectory_and_direction_2026-05-01.py (new full validation)
A  analysis/intermediate_layer_probe_2026-04-30/outputs/trained_head_eval_2026-05-01.json
A  analysis/intermediate_layer_probe_2026-04-30/outputs/trajectory_and_direction_2026-05-01.json
A  analysis/intermediate_layer_probe_2026-04-30/outputs/trajectory_and_direction_2026-05-01.csv
A  experiments/phase2_round13/R13_P17_LAYER3_HEAD_LINEAR.yaml
A  experiments/phase2_round13/R13_P17_LAYER4_HEAD_LINEAR.yaml
A  docs/relaunch_handoffs/P17_FINAL_VERDICT_2026-05-01.md   (canonical)
A  docs/relaunch_handoffs/P17_TRAINED_HEAD_FINDING_2026-05-01.md   (precursor)
A  docs/relaunch_handoffs/OVERNIGHT_LAYER3_FINDING_2026-05-01.md   (precursor)
M  utils/config_helpers.py   (apply_wandb_backbone_params: honor explicit hidden_size from yaml)
M  tests/test_p17_layer3_head_wiring.py   (regression test)
M  VERSION  (1.3.238 → 1.3.239)
M  HANDOFF.md   (this file)

Memory:
A  ~/.claude/.../memory/project_p17_trained_head_destroys_substrate_invariance.md
M  ~/.claude/.../memory/MEMORY.md   (added pointer)
```

Working tree is uncommitted per project pattern — do NOT commit without explicit user OK.

## Job IDs and W&B run IDs (for reference)

| Arm | Vertex job ID | W&B run | Final state |
|---|---|---|---|
| L3 ArcFace | `2429114755361800192` | `melp4mol` | DONE — best step 2088, val AUC 0.7390, lockbox AUC 0.187 |
| L4 ArcFace | `3168830994157404160` | `faheakaf` | DONE — best step 2410, val AUC 0.7634, lockbox not probed (cache lacks layer-4) |
| L3 LINEAR | `6468843621113135104` | `nqvfz44v` | CANCELLED — best step 1285, val AUC 0.7277, lockbox AUC 0.066 |
| L4 LINEAR | `5009677341845094400` | `mbd951b8` | CANCELLED — last seen step ~1450, val AUC 0.74, lockbox not probed |

## Active session schedule (cron)

The earlier session-only wakeup at 17:23 CEST fired and was reconciled (its premise — "ArcFace stuck at AUC 0.55" — had been overtaken by events; verdict already in via the probe). No active wakeups remain.
