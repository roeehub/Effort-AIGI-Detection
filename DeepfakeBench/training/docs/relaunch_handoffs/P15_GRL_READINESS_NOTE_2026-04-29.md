# P15 GRL — Readiness Note

**Author:** Claude Opus 4.7 (1M context), 2026-04-29
**Branch:** `teams-relaunch-root-2026-04-17`
**Yaml:** `experiments/phase2_round13/R13_P15_GRL_FROM_P8A.yaml`
**Status:** Drafted, NOT launched. Smoke-test required before paying for a Vertex run.

---

## Why P15 (one paragraph)

P13 from-scratch + the three indirect anti-shortcut interventions (anchor-aware loss, pipeline-randomization, face scale-jitter) moved the camera-signature shortcut Δ on Axis 3 only directionally (0.66 → 0.52, gate 0.15). The interventions are input-space perturbations that pressure the model away from a specific shortcut; the model then finds an adjacent one. **P15 adds the only structurally-different anti-shortcut lever the codebase already supports: gradient reversal on a quality-domain classifier head (DANN, Ganin 2015).** Where P13's interventions perturb inputs and hope the backbone generalizes, GRL directly penalizes the backbone for producing features that predict the quality domain. P15 is the cleanest test of "does direct anti-shortcut supervision break the ceiling that input-space anti-shortcut couldn't?"

## What P15 changes from P14 (the run currently in flight)

A single block, five lines of yaml:

```yaml
use_quality_domain_head: true
quality_domain_loss_weight: 0.20    # P10 baseline 0.10 / strong 0.25 mid-bet
quality_domain_require_labels: true
quality_domain_count: 4
quality_head_hidden_dim: 128
```

Everything else (P8A init, FT learning rate 3e-5, 4000 steps, anchor_aware, pipeline_random, face_scale_jitter, periodic_saves, sampling weights) is identical to P14. Seed 1501 (P14 was 737) so wandb runs don't collide.

## Quality-domain mapping bites the right places

`detectors/effort_detector.py:248-257` + `data/sources/combined_paired.py:66-82`:

| Domain | Sources in P15 training pool |
|---|---|
| 0 clean_academic | `df40` |
| 1 webcam_codec   | `external` (VCD), **`deeplive_teams` (Teams traffic)** |
| 2 studio_capture | `deeplive`, `visomaster` |
| 3 social_media   | (none in training; `youtube` is OOD-monitor only) |

Teams traffic — the same distribution that drives modern_v2 FPR — lands in domain 1. The backbone is penalized whenever its features can identify "this is webcam_codec." That is exactly the shortcut breakdown P13_step18000 demonstrated at τ=0.5 (modern_v2 FPR exploded to 30%+ because the backbone tied "webcam look" to "fake").

## Risks / unknowns

1. **First-ever production exercise of the GRL infrastructure.** Commit `2c9778b` added the P10 GRL slate (six yamls drafted with `use_quality_domain_head: true`); none were ever launched. The detector code (`detectors/effort_detector.py:236-274`), the data labels (`combined_paired.py:65-82`), and the trainer loss aggregation (`detectors/effort_detector.py:906-958`) all exist and unit-tested (`tests/test_unpaired_reals_and_grl.py`), but never run end-to-end on Vertex.
2. **Lambda is static at 0.20.** `GradientReversalLayer.set_lambda(val)` exists but no caller in `trainer/trainer.py` invokes it; ramping λ from 0 → target over warmup steps would require a small trainer patch (~10 lines). Skipping ramp keeps this draft no-code-change. If P15 either crashes from too-strong-too-fast adversarial pressure OR is too weak to bite, P16 should add the ramp.
3. **`quality_domain_require_labels: true`** means samples without a quality_domain label silently skip the GRL loss. All P15 training sources DO emit a label (verified via grep in `combined_paired.py`), so this should be a no-op safety belt. Verify in W&B that `train/loss/quality_domain` is non-zero through training (if it's exactly 0, propagation is broken).
4. **wandb-flattening is handled.** `train_sweep.py:322-349` already has fail-fast guards for `use_quality_domain_head`, `quality_domain_count`, `quality_head_hidden_dim`, `quality_domain_loss_weight`. If any propagation breaks, the trainer raises `RuntimeError` immediately. `quality_domain_require_labels` is a top-level bool that wandb passes through cleanly; the detector default matches our yaml (True), so a silent drop would be harmless.
5. **The contract policy bug fix is local-only as of this draft.** If P15 is scored via the runner (`arena/run_target_domain_validation_sequential.py`), pass `--promotion_target_fake_recall_min 0.30` (or similar) — otherwise the contract may again pick a τ-degenerate winner. This applies to the upcoming **P14** scorecard too.

## Pre-launch checklist

- [ ] **Smoke test (no GPU).** Run `train_sweep.py` with `--config experiments/phase2_round13/R13_P15_GRL_FROM_P8A.yaml --smoke` (or local-CPU equivalent). Confirm log lines fire:
  - `Quality domain head ENABLED with 4 domains, hidden_dim=128, loss_weight=0.2, require_labels=True`
  - `Applied anchor_aware: enabled=True weight=5.0 target_mean_prob=0.1`
  - `Applied face_scale_jitter: enabled=True scale_limit=0.25`
  - `Applied periodic_saves: enabled=True step_list=[500, 1000, ..., 4000]`
- [ ] **Image rebuild — NOT required** for the yaml itself, but **required** if any in-image file must change. As of this draft, no in-image changes. Use whatever image P14 was launched on (1.3.226 or higher, post-wandb-flattening fix).
- [ ] **Promotion-contract scorer.** If you intend to score P15 via the standalone scorer or the validation runner, the local edits in `arena/score_teams_promotion_contract.py` and `arena/run_target_domain_validation_sequential.py` must either be (a) committed and image-rebuilt, or (b) the scorer must be invoked locally on downloaded reports. Pick (b) for the P14 scorecard; pick (a) once you confirm the floor behavior on a real ckpt set.
- [ ] **WANDB env vars.** Per memory `feedback_promotion_contract_launch.md`, the launcher needs `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT` exported.
- [ ] **User authorization.** Per memory `feedback_no_cancelling_vertex_jobs.md`, do not launch without explicit OK.

## Cost / expected outcome

- **Cost:** ~$60 (4000 steps, FT cadence, A100, us-east1 → us-west4 → us-central1 fallback).
- **Best-case outcome:** Axis 3 shortcut Δ moves below P14's value (which we don't know yet) AND viso/deeplive recall stays at or above P8A's level. First candidate where direct anti-shortcut supervision is acting on a working cross-domain base.
- **Modal expected outcome:** β. GRL @ static λ=0.20 will likely move Δ another 0.05–0.15, still above the 0.15 gate, but with cross-domain recall preserved by the FT init. This would license either P16 (GRL with ramped λ, larger weight) or escalation to spectral/dual-stream features.
- **Worst-case:** GRL at λ=0.20 catastrophically degrades cross-domain recall (similar to P13_FROM_SCRATCH, but for a different reason: backbone is forced to throw away too much information). Mitigation: kill switch at step 1500 if viso recall drops > 8pp below P8A.

## When to launch P15 (decision logic)

| P14 verdict | Recommended P15 action |
|---|---|
| α (passes triple-axis) | DO NOT launch P15 — ship P14, save GRL for the next round of capability extension. |
| β (Axis 3 Δ in [0.15, 0.45], cross-domain ≥ P8A) | LAUNCH P15. Goal: drive Δ further down without giving back cross-domain. |
| γ (Axis 3 Δ ≥ 0.45 OR cross-domain < P8A by > 8pp) | LAUNCH P15. The interventions need an additional structural lever; GRL is the cheapest one available. |

In all cases, the contract policy fix (local) should be applied to the P14 scorecard before a verdict is computed.
