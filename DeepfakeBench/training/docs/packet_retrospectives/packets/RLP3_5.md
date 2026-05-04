# Packet RLP3.5  ·  ArcFace-margin canary + stability/smoothing probes (parallel to RLP3)

> **Template contract**: fill every section. If a section truly has no content, write `*(none — reason)*` rather than deleting the heading. Keep the status-card table intact.

## Status card

| Field | Value |
|---|---|
| Dates | 2026-04-22 → 2026-04-22 |
| Slots | 7 (RLP35_01…RLP35_07; slot 07 authored-but-not-fired) |
| Headline lever | ArcFace margin `m ∈ {0.10, 0.15, 0.20}` + stability_lambda, label_smoothing, family-rebalance probes |
| Leader slot | `R13_RLP35_02_arcface_m015` |
| Leader metric | `value_composite = 0.7442` (Δ +0.021 vs retro-scored P3 leader RLP3_05 @ 0.7232) |
| Verdict | ✅ confirmed (arcface `m=0.15` adopted as control through RLP4+) |
| Next-packet decision | RLP4 adopts `m=0.15` as control; margin scan refined to `{0.125, 0.15, 0.175}` on both backbones; stability/smoothing/family-rebalance dropped from primary levers |
| Themes touched | [promotion_contract_evolution](../threads/promotion_contract_evolution.md), [gate_alignment_story](../threads/gate_alignment_story.md), [processing_signature_shortcut](../threads/processing_signature_shortcut.md), [preprocessing_parity_bug](../threads/preprocessing_parity_bug.md) |

## Configuration

RLP3.5 was launched as a **parallel canary** to the RLP3 main line, not a successor — `us-east1` wave while RLP3's 7 slots completed in `asia-southeast1` (R13_RELAUNCH_PACKET3_5_EXPERIMENT_PLAN_2026-04-22.md:3-4). It turned on four training knobs that packets 1–3 had left off and simultaneously moved three `value_composite` gate-definition knobs.

Global deltas applied to every RLP3.5 slot (R13_RELAUNCH_PACKET3_5_EXPERIMENT_PLAN_2026-04-22.md:37-42):

- `value_composite.target_mean_fpr: 0.02 → 0.03`
- `value_composite.max_pool_fpr: 0.04 → 0.05`
- `value_composite.stability_jitter_stat: max → p95`
- `anneal_steps: 15000 → 8000` (ArcFace anneal now finishes inside 10k training steps)

Control baseline is `R13_RLP3_02_FT_proper_main` — unenhanced proper-data only, no hints, seed 737, `identity_split_mode: hash_stable`, r12g-FP32 base checkpoint (R13_RELAUNCH_PACKET3_5_EXPERIMENT_PLAN_2026-04-22.md:24-32).

Variants (each slot = exactly one training-knob delta vs RLP3_02):

- **`R13_RLP35_01_arcface_m010`** — `arcface_m: 0.0 → 0.10` (cautious margin).
- **`R13_RLP35_02_arcface_m015`** — `arcface_m: 0.0 → 0.15` (middle of cautious range). `experiments/phase2_round13/R13_RLP35_02_arcface_m015.yaml:12-14`.
- **`R13_RLP35_03_arcface_m020`** — `arcface_m: 0.0 → 0.20` (upper bound; probes backbone-collapse threshold). `experiments/phase2_round13/R13_RLP35_03_arcface_m020.yaml:13-14`.
- **`R13_RLP35_04_stability_lambda_003`** — `stability_lambda: 0.0 → 0.03` + `noise_std=0.02, crop_jitter=0.03`. Aims to unpin the stability term. `experiments/phase2_round13/R13_RLP35_04_stability_lambda_003.yaml:13-14`.
- **`R13_RLP35_05_label_smoothing_005`** — `label_smoothing: 0.0 → 0.05`. `experiments/phase2_round13/R13_RLP35_05_label_smoothing_005.yaml:12-13`.
- **`R13_RLP35_06_family_rebalance_proper_up`** — family-weight shifts to upweight deployment real surface. `experiments/phase2_round13/R13_RLP35_06_family_rebalance_proper_up.yaml:13-14`.
- **`R13_RLP35_07_stack_top3`** — contingent stack of the healthy single-lever winners; never fired (see §Conclusions). `experiments/phase2_round13/R13_RLP35_07_stack_top3.yaml:16-17`.

## Results at the time

Single-lever Δ`value_composite` vs retro-scored RLP3_02 baseline (same `(0.03, 0.05, p95)` gate):

| Slot | best `value_composite` | Δ vs retro-P3_02 |
|---|---|---|
| `RLP35_02` arcface m=0.15 | **0.7442 ★** | +0.0415 |
| `RLP35_01` arcface m=0.10 | 0.7435 | +0.0408 |
| `RLP35_04` stability_lambda=0.03 | — | +0.0067 |
| `RLP35_05` label_smoothing=0.05 | — | +0.0053 |
| `RLP35_06` family_rebalance | — | slight positive (well under +0.03) |
| `RLP35_03` arcface m=0.20 | **blocked** | N/A — `max_fpr ≈ 0.055`, never clears the 0.05 gate |

Against the RLP3 **leader** under the new gates (RLP3_05 `low_arcface + spatial` at `0.7232`), RLP35_02's margin shrinks to **+0.021** — still positive, but much smaller than it looked under legacy gates (R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md:17-19).

Slot 03 sat at `worst_pool_fpr ≈ 0.055` for 22+ hours and was eventually cancelled as "margin ceiling signal saturated" (R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md:77).

## Conclusions drawn in-session

- **m=0.15 is the preferred operating point.** m=0.10 and m=0.15 are essentially tied (Δ ≈ 0.0007); m=0.15 is the nominal leader. The margin ceiling is "somewhere between 0.15 and 0.20" — slot 03 never converged (R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md:17-19).
- **ArcFace margin is the only non-trivial single lever in this wave.** Session 87386521: *"The only large win under the new metric is arcface-margin tuning (m=0.10 and m=0.15 essentially tied at ~+0.04 vs retro-P3_02). m=0.20 hasn't converged cleanly and may not."*
- **Lead is real but ~5× smaller than it looked.** Session 1a4ce3aa: *"The P3.5 winner's lead is real but ~5× smaller than it looked under legacy gates. Against the retro'd P3 champion (RLP3_05 = 0.7232), RLP35_02's margin is +0.021, not +0.11."* This is the first clean in-session signal that gate-definition was moving as much mass as training-knob choices — an early contribution to the gate-alignment / promotion-contract story.
- **Slot 07 stack_top3: DO NOT FIRE.** The packet-3.5 §5 rule required ≥2 of {highest-healthy arcface, stability_lambda, label_smoothing} to beat retro-P3_02 by ≥+0.03. Only arcface cleared (+0.0415); stability_lambda (+0.0067) and label_smoothing (+0.0053) did not (R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md:20; session 87386521).
- **Stability / label-smoothing / family-rebalance are not primary levers.** All three landed as slight positives well under the +0.03 threshold. They do not compound enough to justify stacking and are not carried forward.
- **Session IDs**: `02873fd0`, `99732ab8`, `87386521`, `cd11b5ff`, `1a4ce3aa` (convmem-resumable).

## Retrospective (as of 2026-04-24)

- **The `m=0.15` lever held.** RLP4 adopted it as control and scanned `{0.125, 0.175}` around it on both backbones (R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md:27-34); it survived RLP5 lineage too. Scorecard numbers later shifted; the lever choice did not.
- **Stability / label-smoothing / family-rebalance stayed dropped.** RLP4 explicitly: *"Stability_lambda dropped from tonight's plan pending [user visual inspection of stress-lane p95 jitter]"* (R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md:22); none returned as primary levers in RLP5–RLP7. Settled.
- **Slot-07 pre-compute predicted the lighting-aug ceiling.** The decision rule implicitly estimated the marginal benefit of the stability+label_smoothing+family stack at ~+0.003 in isolation — foreshadowing the later RLP7 finding that lighting/perturbation robustness is **not** the deployment bottleneck. The shortcut story (see [processing_signature_shortcut](../threads/processing_signature_shortcut.md)) explains why: slot-07 would have been paying compute for the wrong problem.
- **The ~5× lead-shrink observation is a load-bearing data point for the [promotion_contract_evolution](../threads/promotion_contract_evolution.md) thread.** The in-session realization that the gate-definition move (mean_fpr 0.02→0.03, max_fpr 0.04→0.05, jitter max→p95) was doing more work than the training-knob move is the first time the team clearly separated *metric-moved-the-number* from *training-moved-the-number*. That separation is the seed of the later `selected_threshold`-aware scorecard discipline and the dev-calibrated lexicographic τ + lockbox readout.
- **m=0.20 blocker is the [gate_alignment_story](../threads/gate_alignment_story.md) in miniature.** The slot was cancelled not because it diverged in training but because one pool's FPR refused to come under the (relaxed) 0.05 ceiling. A checkpoint with genuinely strong training signal was unpromotable purely because a deployment gate declined. RLP4/RLP5 expanded this pattern; RLP6 built the calibration machinery that eventually reframed it.
- **Preprocessing-parity status: PRE-FIX.** All RLP3.5 numbers quoted in this file predate the `INTER_LINEAR` preprocessing alignment (commit `855871e`, 2026-04-24). See [preprocessing_parity_bug](../threads/preprocessing_parity_bug.md). Because every slot in RLP3.5 (and the retro-scored RLP3_02 / RLP3_05 baselines) was evaluated through the same pre-fix path, *intra-packet rankings are coherent* — m=0.15 would still beat m=0.10, the stability probes would still under-shoot +0.03, m=0.20 would still block. But *absolute* composites, and in particular per-pool FPR on Dor / webcam / other pipeline-sensitive surfaces, would shift post-fix. Any comparison of RLP3.5 numbers to post-fix baselines must be done through re-scored checkpoints, not the original W&B logs.
- **Story continues** in [RLP4](RLP4.md) (which adopts `m=0.15` and stacks it on the spatial backbone) and via the cross-cutting threads above.

## Source files

- **Handoffs**:
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_EXPERIMENT_PLAN_2026-04-22.md` — primary. Motivation :9-21, gate-defn changes :37-42, slate :60-67, decision rule :75-77, slot-07 contingency :69.
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md` — downstream adoption. Leader/tie/ceiling :17-19, slot-07 don't-fire :20, `m=0.15` adopted as control :27-34, m=0.20 cancel :77.
  - `docs/relaunch_handoffs/R13_RELAUNCH_TRAINING_SOURCE_OF_TRUTH_2026-04-21.md` — baseline / split / checkpoint conventions inherited by this wave.
- **YAMLs**: `experiments/phase2_round13/R13_RLP35_0{1..7}_*.yaml` (all seven listed in §Configuration above).
- **Scorecards / analysis**: W&B project `enhanced-aug-test`, runs `RLP35_0{1..6}`; retro-score of RLP3_02 / RLP3_05 under the new `(0.03, 0.05, p95)` block via `retro_score_value_composite.py`.
- **Memory pointers**: `project_promotion_contract.md` (dev-calibrated τ + lockbox is authoritative, not `value_composite`); `project_contract_policy_bug.md` (post-dates this packet; contextualizes why `value_composite` rankings were never deployment-grade).
