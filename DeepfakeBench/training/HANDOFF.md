# Handoff: Codec restoration triple — VERDICT IN — Slot A v2 still the deploy ckpt

**Generated**: 2026-05-21 (verdict landed UTC ~03:30; original handoff written ~00:30)
**Branch**: `teams-relaunch-root-2026-04-17`
**Status**: All 3 packets SUCCEEDED. Verdict scored. **None of the 3 packets are deployment upgrades — Slot A v2 step3500 remains the best.**

## ⚡ Morning bottom line

Read `analysis/codec_restoration_gates_2026-05-21/VERDICT_FACTS_2026-05-21.md` first.

- **Codec aug DOES bite** the natural-experiment transport axis (37-60% Δ reduction across the 3 packets).
- **None of them close the gap** to the deployment-grade criterion (|Δ| ≤ 0.05). Best closure: Slot 2 at Δ=-0.067 (60% reduction); criterion would need ≤ 0.050.
- **All 3 regress** on may6 production-drift vs Slot A v2 base (Slot A v2 = 0.043 → packets 0.065-0.120).
- **Decision-tree branch: NUANCED D** — mechanism activity confirmed, single-lever insufficient.

| ckpt | anchor | codec_p | Δ (criterion ≤ 0.05) | Δ reduction | may6 frac>0.5 (vs ref 0.043) |
|---|:---:|---:|---:|---:|---:|
| **Slot A v2 step3500 (ref)** | ON | 0.00 | -0.168 | — | **0.043 (best)** |
| Slot 1 | ON | 0.40 | -0.083 | -51% | 0.065 |
| Slot 2 | ON | 0.20 | -0.067 (best) | -60% | 0.087 |
| Slot 3 | OFF | 0.40 | -0.106 | -37% | 0.120 (regressed) |

## Tonight's question

After the 2026-05-20 7-job CPU evidence batch established that:
- The 0.07pp `lockbox_real_fpr` tiebreak that keeps P8A rank-1 over Slot A v2 is **inside sampling noise** (Job B 95% CI [−0.008, +0.010] covers 0, P=0.519);
- Slot A v2's anchor mechanism is **decoupled from the transport-shortcut mechanism** (Job A: 2026-05-19 natural-experiment Δ identical to T5C to 4 decimal places);
- Slot A v2 preserves P8A-level may6 production-drift survival (Job F: 4/92 false-flags, vs P8A 0/92 within MPS noise; T5C 16/92, E2B 53/92);

...the program's next move is to address the **per-Teams-account transport-shortcut** while preserving Slot A v2's chronic-FP gains.

The audit found that **P22 (2026-05-02) deliberately replaced `teams_codec_sim` with `pipeline_randomization`** in the augmentation stack, won 3× dev fake_macro_recall at the time, but introduced an invisible per-account transport-flip cost. T3 → T4 → T5C → Slot A v2 all inherit P22's swap.

Tonight tests: **does restoring `teams_codec_sim_p=0.40` ON TOP OF `pipeline_randomization` close the transport gap?**

## What was done this session (autonomous, 2026-05-21 ~00:23-00:30 UTC)

### Step 0: Audit ($0, no aug code written)

Found that `data/augmentations/teams_simulation.py` (414 LOC, 3 classes) was already shipping but unused; the actual transform fired by `teams_codec_sim_p>0` is **`VideoCodecSimulation`** (`data/augmentations/transforms.py:1216`) via `pipelines.py:1562-1570`. Saved building a duplicate module per AGENT_GUIDE Rule 6 (aug-lever-rediscovery check).

- Augmentation keys propagate via full-block reapply in `train_sweep.py:207-214`; **no allowlist amendment required**.
- `anchor_aware.pool_names` is plural-capable but only 6 dor-/roee-named pools registered (`analysis/teams_pool_rescore.py:44-51`); Roy_D pool extension requires bucket-prefix + frames-upload + registry edit (deferred to user authorization).

### Step 1: 2×2 packet design

3 yamls written under `experiments/phase2_round13/`:

| Slot | YAML | FT base | anchor | codec_p |
|---|---|---|:---:|---:|
| 1 | `R13_T5C_ANCHOR_AWARE_PLUS_CODEC_2026-05-21.yaml` | Slot A v2 step3500 | ON | 0.40 |
| 2 | `R13_T5C_ANCHOR_AWARE_PLUS_CODEC_LIGHT_2026-05-21.yaml` | Slot A v2 step3500 | ON | 0.20 |
| 3 | `R13_T5C_CODEC_ONLY_NO_ANCHOR_2026-05-21.yaml` | T5C step3500 | OFF | 0.40 |

2×2 design when combined with the already-evaluated reference checkpoints:

|                     | codec_p=0.0          | codec_p=0.20      | codec_p=0.40      |
|---------------------|----------------------|-------------------|-------------------|
| anchor=ON  (Slot A v2 base) | reference (Slot A v2 step3500) | **Slot 2**        | **Slot 1**        |
| anchor=OFF (T5C base)       | reference (T5C step3500)       | (not in batch)    | **Slot 3**        |

Decomposition decision rules:
- Slot 1 ≥ Slot A v2 on lockbox_fake_recall AND Slot 3 ≈ T5C → anchor and codec compose additively
- Slot 3 ≈ Slot 1 → codec is the binding lever, anchor doesn't add
- Slot 1 ≈ Slot A v2 (no lift) → codec doesn't add on top of anchor on this base
- Slot 2 between Slot 1 and Slot A v2 → dose-response is monotone

### Step 2: Pre-launch CPU gates

`analysis/codec_restoration_gates_2026-05-21/run_gates.py` ran VideoCodecSimulation at `codec_quality=(20, 65)` (the P8A spec) on the Roy_D natural-experiment crop, scored with both FT bases:

| ckpt | baseline Roy_D | aug median (n=12) | guest ref | Gate 1 (validity ≤0.10) | Gate 3 (saturation ≤0.30) |
|---|---:|---:|---:|---|---|
| T5C step3500 | 0.7952 | **0.7239** | 0.6275 | **PASS** (gap 0.096) | PASS (swing 0.099) |
| Slot A v2 step3500 | 0.7668 | 0.7481 | 0.5987 | FAIL (gap 0.149) | PASS (swing 0.143) |

Gate 1 marginal-FAIL on Slot A v2 is in the right direction (aug pushes Roy_D score DOWN toward Guest's region) with magnitude 0.019 — single-frame test under-estimates training-time effect where the encoder sees ~110K augmented frames over 3500 steps and learns invariance. **Launch proceeded with documented caveat.**

Gate 2 (AUC preservation on a labeled panel) was deferred to the **in-flight wandb canary** that runs every 500 steps on all 3 packets — saves the labeled-panel download.

### Step 3: Image build + launch

Image rebuilt: `1.3.294 → 1.3.295` (Cloud Build `564298ac-c0ab-4ce9-aef8-08ecc5660722`, 1m 49s). Pushed to `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.295`.

Launch script: `scripts/launch_tonight_2026-05-21.sh`. All 3 submitted PENDING via `launch_experiment.sh -y phase2r13-experiments <REGION> <YAML>`:

| Slot | Region | Vertex Job ID | Image | Status |
|---|---|---|---|---|
| 1 | us-west4 | `5839421592722997248` | 1.3.295 | PENDING at submit; check current state |
| 2 | us-east1 | `1506065191836581888` | 1.3.295 | RUNNING (transitioned within ~5 min) |
| 3 (failed) | us-central1 | `5637646623317164032` | 1.3.295 | FAILED 22:30 UTC — image-currency race, yaml not in source tarball |
| 3 (relaunched) | us-central1 | `7516210617884082176` | 1.3.296 | PENDING at submit on rebuilt image |

W&B dashboard: https://wandb.ai/dtect-vision/phase2r13-experiments

## What to watch (in priority order)

### 1. W&B canary metrics on all 3 packets (every 500 steps)

Per packet, the inflight canary should report `canary/lockbox_recall_at_FPR_5pct` and `canary/chronic_mean/*`. Per `feedback_decision_points` and the 2026-05-20 verdict on the T5C_TRIPLE, watch for:

- **Slot 1 trajectory (anchor + codec p=0.40)**: should hold or improve over Slot A v2 step3500's reference `lockbox_recall_at_FPR_5pct = 0.59` (manual canary). If trajectory monotone-down across 3 trajectory points (steps 1500/2500/3500), suspect compression trap from the heavy aug stack — refer to `RESCHAIN_GRL6` retro pattern.
- **Slot 3 trajectory (no anchor + codec p=0.40)**: this is the cleanest natural-experiment validity packet. Per CPU Gate 1, codec aug reproduces ~42% of the natural Δ in single-frame test on T5C. If Slot 3's trained ckpt re-tests the natural experiment with Δ ≤ 0.10, codec aug is biting on the right axis.
- **Slot 2 (anchor + codec p=0.20)**: middle dose. Should sit between Slot 1 and Slot A v2 reference on most metrics.

### 2. Vertex job state (monitor armed)

A Monitor was armed at task `ba1ur5e2z` polling every 3 min for state transitions. Per CLAUDE.md: **if PENDING > 30 min, switch regions** (us-west4 ↔ us-east1 ↔ us-central1; don't cancel original until replacement RUNNING; **never** use asia/europe). Per `feedback_no_cancelling_vertex_jobs`: cancellation requires explicit user OK.

If a region rejects with quota, fallback order: us-west4 → us-east1 → us-central1 → (request user authorization for fallback).

### 3. Natural-experiment Δ retest on each trained ckpt

When each packet's step3500 ckpt lands, run `analysis/codec_restoration_gates_2026-05-21/run_gates.py` (point CKPTS at the new GCS paths) — this gives a direct Δ readout vs the 2026-05-19 reference (T5C Δ=−0.168, Slot A v2 Δ=−0.168). If trained Δ ≤ 0.05, the codec axis is closed for that ckpt's deployment population.

## Open questions for the user

1. **Roy_D anchor pool extension** — defer until Slot 1/3 verdict; if anchor mechanism doesn't generalize past dor (which Job D suggested with Roy_D +0.54 dev FPR), need Python edit to `analysis/teams_pool_rescore.py` to add Roy_D pool. ~60-90 min with silent-failure risk per Plan agent.
2. **Tiebreak amendment to `arena/score_teams_promotion_contract.py`** — yaml change from `lockbox_real_fpr asc` to composite k=1; promotes Slot A v2 over P8A under existing scorecards. $0 reversible; Job B evidence supports.
3. **Whether to keep the 4-ckpt 29-suite scorecard** for Slot 1/2/3 or use the inflight canary as proxy. ~$15-25 GPU per 4-ckpt batch; budgeted only if a ckpt shows interesting inflight signals.
4. **Multi-account capture sweep** — `analysis/teams_account_natural_experiment_2026-05-19/§6.1` recommendation. User-side: capture 20-50 frames per account (Roy_D, Guest, free-tier, edu-tier, enterprise-tier). If per-account distributions are separable, this is a first-class deployment failure mode for the next packet.

## What was NOT done (deferred, with reason)

- **Roy_D-specific anchor pool packet** — multi-step infra (new bucket prefix + frames upload + registry edit in `analysis/teams_pool_rescore.py:POOLS`); risk of silent fallback to dor-only pool. Per HANDOFF 2026-05-20 §5: "swapped to yaml-only ablation instead" was the right call; still right tonight under autonomous mode.
- **KLIEP discriminator measurement** of teams-v2 vs lockbox with/without each aug — useful for characterizing aug effects but not a launch gate. Defer to a follow-up CPU job after the GPU verdict lands; refer to Job D8 (2026-05-19) baseline 99.09% accuracy.
- **AUC preservation gate on labeled panel** — superseded by inflight wandb canary on all 3 packets.

## Failed approaches this session (don't repeat)

1. **Wrote tuple-unpacking bug in `run_gates.py`**: used 2-element tuples in a `for q_lo, q_hi, label in [...]` loop, crashed at iteration 1. Fixed inline (`(10, 20, "label")` not `((10, 20), "label")`). Trivial; flagged here for the unit-test missing.
2. **Attempted to draft a new `teams_transport_aug` module** at session start before reading existing code — caught by AGENT_GUIDE Rule 6 ("aug-lever-rediscovery check") during the Step 0 audit, which surfaced `teams_simulation.py` (414 LOC, 3 classes) already shipping. Course-corrected to "restore P8A's flat aug keys" instead of "build new aug module". Saved ~4 hours of duplicate work.
3. **Considered launching with 3 different levers** (codec + Roy_D pool + face_scale_jitter) before realizing user is asleep and Roy_D pool extension has silent-failure risk requiring runtime verification. Pivoted to dose-response (p=0.20, p=0.40) + control (no-anchor), giving cleaner experimental decomposition.

## Reproducibility / commit

- 3 yamls + 1 launch script + 1 CPU-gate script + 1 gates output dir = the artifacts produced this session.
- `STATE.md` updated with 2026-05-21 entry at top (entry marker: "2026-05-21 auto-mode — codec restoration triple launched").
- **Commit policy**: the 3 new yamls + launch script + gate script + gates output are clean adds; the M-files in the existing working tree (canary_probe.py, batch_inference_gcs.py, etc., from the 2026-05-20 session) are NOT committed by this session and remain for user review per `feedback_decision_points`.

## References

- 2026-05-20 7-job CPU evidence batch: `analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/RESULTS_FACTS_2026-05-20.md` and `AGENT_PROPOSAL_2026-05-20.md`.
- 2026-05-19 natural experiment: `analysis/teams_account_natural_experiment_2026-05-19/TEAMS_ACCOUNT_NATURAL_EXPERIMENT_FACTS_2026-05-19.md` and `CHEAP_FOLLOWUPS_FACTS_2026-05-19.md`.
- 2026-05-21 CPU gates: `analysis/codec_restoration_gates_2026-05-21/gates_summary.json` + `gates_per_run.csv`.
- Memory entries touched: none re-written; this session's findings can be consolidated into a new entry `project_codec_restoration_program_2026-05-21.md` once the GPU verdict lands.
