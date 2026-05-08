# Scorecard Guide — when to use which mode

> **Status**: AUTHORITATIVE. This is the operating manual for promotion-contract scorecards. Edits to the canonical anchor list or the mode-to-suite mapping require updating both this doc AND `arena/launch_teams_promotion_contract.sh` AND the relevant memory entries in the same commit.
>
> **Audience**: any agent assembling a scorecard. The default action of "run the full scorecard" is often wasteful — pick a mode that matches the question being asked.
>
> **Where it sits**: top-level wiki, sibling of [`MODEL_GOALS.md`](MODEL_GOALS.md), [`AGENTS.md`](AGENTS.md), [`AGENT_GUIDE.md`](AGENT_GUIDE.md).

---

## The three modes

The launcher `arena/launch_teams_promotion_contract.sh` accepts `--mode {full|iterative|trajectory}`. Each mode answers a specific question, and includes a different (ckpt, suite) cross-product.

| mode | answers | ckpts | suites | wall-time on A100×1 |
|---|---|---:|---:|---|
| **iterative** | "Did this packet move the needle?" | 3 (anchors + 1 candidate) | 27 (drops 2 lockbox-stress) | ~3-4h |
| **trajectory** | "Within the winning packet, which ckpt is the operating point?" | 2 + N (anchors + N trajectory points) | 29 (full) | scales with N |
| **full** | "What's the cross-packet promotion verdict?" | all in the map | 29 (full) | scales with map size |

`launch_teams_promotion_contract.sh --mode <X>` defaults `--suite-manifest` and assembles `--checkpoints` accordingly. Anchors are appended automatically in iterative + trajectory modes.

---

## Mode 1 — `iterative`

**Use this when**: a single packet has finished training and you want to know if its candidate ckpt(s) move the needle vs. the production anchors.

**Command shape**:
```bash
bash arena/launch_teams_promotion_contract.sh \
    --mode iterative \
    --candidate P2_D_FOURIER_PERIODIC_STEP3000 \
    --checkpoint-map arena/checkpoint_maps/teams_target_domain.<packet>_<date>.yaml \
    --job-name <packet>-iter-<date>
```

**What it scores**: 3 ckpts × 27 suites = 81 (suite × ckpt) pairs.
- 2 production anchors (P8A_REFERENCE_STEP5000 + E2B_TOP_N_STEP3200) auto-included
- 1 candidate ckpt — the canary-best ckpt from the packet
- 27 suites (the FULL suite set minus 2 lockbox-stress diagnostics whose dev counterparts already drive τ)

**What it preserves**:
- All τ-deciding suites (Tier 1)
- Both lockbox readout suites (Tier 2)
- All 17 per-identity diagnostic suites — load-bearing for chronic-FP detection per memory `project_p1_pe_eval` (Roy_D regression detection) and `project_chronic_offenders_partition_per_ckpt_2026-05-04` (different ckpts handle different identities; macro can hide regressions)

**What it sacrifices**:
- 2 lockbox-stress suites (`teams_real_poor_quality_lockbox`, `teams_real_lighting_extreme_lockbox`). Their dev counterparts already drive τ; the lockbox versions add a small cross-substrate sanity check that is valuable at promotion-verdict time but redundant for "did this move the needle?"
- Trajectory information — only one ckpt per packet is scored. Use trajectory mode after an iterative win.

**Picking the candidate**: the canary-best ckpt is the right default. Per-frame canary readouts (`trainer/mixins/canary_probe.py`) include `lockbox_recall_at_FPR_10pct`, `score_p95_on_reals`, chronic-6 means. If multiple canary fires show distinct operating points, pick the one with the highest `lockbox_recall_at_FPR_10pct` AND `score_p95_on_reals < 0.95` (low-saturation regime — see Slot D step3000 selection in `analysis/p2_eval_2026-05-08/d1_d4_cpu/D5_CKPT_MAPPING_FACTS_2026-05-08.md`).

**Decision after iterative**:
- **Pass**: candidate beats E2B on `dev_fake_macro_recall` AND doesn't regress P8A's chronic-FP behavior. Launch trajectory mode to find operating point. Optionally launch full mode for cross-packet verdict.
- **Fail**: stop. Don't burn compute on trajectory readouts for a candidate that didn't move the needle.

---

## Mode 2 — `trajectory`

**Use this when**: an iterative scorecard has produced a passing candidate, and you need to read out multiple trajectory points (early/mid/late ckpts from the same training run) to find the operating point.

**Command shape**:
```bash
bash arena/launch_teams_promotion_contract.sh \
    --mode trajectory \
    --checkpoints P2_D_FOURIER_PERIODIC_STEP3000,P2_D_FOURIER_PERIODIC_STEP8000,P2_D_FOURIER_TOP_N_STEP19000 \
    --checkpoint-map arena/checkpoint_maps/teams_target_domain.<packet>_<date>.yaml \
    --job-name <packet>-trajectory-<date>
```

**What it scores**: (2 anchors + N candidates) × 29 suites. N=3 typical (early/nominal-cap/late).

**Why FULL suite manifest**: at this stage we are characterizing operating points; the lockbox-stress diagnostics are useful sanity checks for cross-substrate consistency.

**Why this is conditional on iterative success**: trajectory readouts are expensive and only useful when there's a winner whose operating point matters. Don't run trajectory pre-iterative — it's the same anti-pattern as ensemble-as-first-resort.

---

## Mode 3 — `full`

**Use this when**:
- Final promotion-verdict run for a candidate that has cleared iterative + trajectory.
- Cross-packet comparison batch (e.g. "compare P-A, P-B, P-C against E2B+P8A on the same suite manifest").
- Authoritative scorecard for documenting a packet retrospective.

**Command shape**:
```bash
bash arena/launch_teams_promotion_contract.sh \
    --checkpoint-map arena/checkpoint_maps/teams_target_domain.<packet>_<date>.yaml
```

(`--mode full` is the default; can be omitted.)

**What it scores**: ALL ckpts in the map × 29 full suites.

**Cost**: scales linearly with map size. The 2026-05-08 P2 scorecard (7 ckpts) is ~8h.

---

## The anchor list (single source of truth)

The production anchors live in **two** places that MUST stay in sync:

1. `arena/launch_teams_promotion_contract.sh` — `PRODUCTION_ANCHORS` bash array.
2. `arena/checkpoint_maps/teams_target_domain.<packet>_<date>.yaml` — every checkpoint map MUST include both anchors as the first two entries.

When the production deployment changes (e.g. E2B_TOP_N_STEP3200 → some new ckpt), update:
- `PRODUCTION_ANCHORS` in the launcher
- The default checkpoint map (or fix the user's flow to ensure new maps include the new anchors)
- Memory `project_deployment_is_e2b_2026-05-06.md`
- `MODEL_GOALS.md` § Deployment

**Same commit**, no exceptions. Anchor drift across these locations causes silent comparison-against-stale-baseline bugs.

---

## What this guide does NOT do

This guide controls **what to score**, not **how to interpret the score**. For interpretation:

- **Pillar definitions + promotion bar**: `MODEL_GOALS.md` § Three pillars + Promotion bar.
- **τ calibration policy**: `score_teams_promotion_contract.py:268-302` (lex policy + recall floor).
- **Per-identity diagnostic interpretation**: memory `project_chronic_offenders_partition_per_ckpt_2026-05-04`, `project_p1_pe_eval` (Roy_D regression case study).
- **F-suite vs full-substrate framing**: memory `project_job14_substrate_clean_2026-05-04`, `project_f1_recall_results_2026-05-04`.

---

## Operational pitfalls

### Image-currency check

The launcher runs `scripts/launch/check_image_currency.sh` before submitting. If you've edited the suite manifest yaml or the checkpoint map, the file mtime will be newer than the image push time, and the check will block the launch. This is by design — yamls are baked into the image at `/workspace/`, and a launch would otherwise use a stale baked-in copy.

To proceed: `./dev.sh build-prod -y` (~17min Cloud Build, auto-bumps VERSION patch). Then re-launch.

Override with `SKIP_IMAGE_CURRENCY_CHECK=1` only when debugging the launcher itself.

### Anchor presence in checkpoint maps

Iterative mode auto-adds anchors to `--checkpoints`, but the anchor aliases (`P8A_REFERENCE_STEP5000`, `E2B_TOP_N_STEP3200`) MUST be defined in the checkpoint map yaml. If not, the runner will fail with an unresolved-key error.

Convention: every checkpoint map should start with both anchors as the first two entries. See `arena/checkpoint_maps/teams_target_domain.p2_scratch_2026-05-08.yaml` for the canonical structure.

### Don't second-guess iterative without per-identity

Memory `project_chronic_offenders_partition_per_ckpt_2026-05-04` and `project_p1_pe_eval` document a specific failure mode: a candidate looks net-positive on macro suites while regressing on a single chronic identity (P1's Roy_D case: 29% → 78-93% FPR, hidden in the macro until per-identity surfaced it). The iterative manifest INCLUDES per-identity diagnostics for this exact reason. **Do not strip per-identity from iterative without re-litigating the analysis at `analysis/p2_eval_2026-05-08/CANARY_RESOLUTION_FACTS_2026-05-08.md` and the prior debate.**

### When in doubt, run full

If the question being asked doesn't fit cleanly into iterative or trajectory — for example, a one-off "let me compare these 4 ckpts from different packets" — just use `--mode full` (the default). Mode discipline is for the **iteration loop**, not for one-off measurements.

---

## Revision history

- **2026-05-08** — initial authorship after debate documented in conversation thread. Modes implemented in `launch_teams_promotion_contract.sh` rev `[commit-hash-tbd]`. Iterative manifest at `arena/target_domain_suites.teams_promotion_contract_iterative_2026-05-08.yaml`. Default suite manifest path updated from `2026-04-17.yaml` → `2026-04-23_with_dor.yaml` (the latter is what's actually been used in practice; the old default was stale).
