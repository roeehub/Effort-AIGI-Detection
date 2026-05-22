# Phase 2 BACKBONE-T5C — Results FACTS (2026-05-22)

> **Status: factual-only, scope-limited.** Banned-word check applies: succeeds, fails, wins, loses, promotes, deployment-grade, ship, kill, best, worst, confirmed, refuted, gap-is-wide, gap-is-narrow.
>
> **Scope.** This document records the code-edit phase of BACKBONE-T5C (asymmetric substrate-pair-loss on T5C step3500 base). Vertex smoke + full training + scorecard are PENDING. No GPU compute has been spent.

---

## 1. Code edits committed to the working tree

Same as BACKBONE-SlotAv2 — see `analysis/phase_2_backbone_slotav2_2026-05-22/RESULTS_FACTS_2026-05-22.md` Section 1 for the full file list. The two packets share the same production code surface area (sampler, loss, trainer hook, collate stamp, allowlist).

The single yaml unique to this packet:

- `experiments/phase2_round13/R13_PAIR_LOSS_ASYM_T5C_2026-05-26.yaml`

---

## 2. Yaml configuration (BACKBONE-T5C)

`experiments/phase2_round13/R13_PAIR_LOSS_ASYM_T5C_2026-05-26.yaml`:

- `name`: R13_PAIR_LOSS_ASYM_T5C_2026-05-26
- `seed`: 9916
- `load_base_checkpoint`: true
- `gcs_base_checkpoint`: gs://training-job-outputs/best_checkpoints/jrlldtem/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth (T5C step3500)
- `total_training_steps`: 3500
- `periodic_saves.step_list`: [100, 500, 1000, 1500, 2500, 3500]
- `use_group_dro`: false (this packet does NOT use GroupDRO; SlotAv2 does)
- `substrate_pair_asymmetric_loss`: {enabled=true, lambda_pair=0.3, margin=0.0}
- `combined_paired.substrate_pair_sampling.enabled`: true (stamper active; loss reads `substrate_pair_id` + `substrate_transport`)
- `anchor_aware.enabled`: false (T5C base does not carry anchor_aware; the pair-loss is the single new lever)
- `multi_axis_grl.enabled`: true (inherited from T5C base)

The asymmetric loss formulation is one-sided:

```
L_pair = lambda_pair * mean( ReLU( prob_fake(teams).detach() − prob_fake(clean) − margin ) )
```

Per the implementation in `loss/substrate_pair_asymmetric.py`, `prob_fake(teams)` is detached so the gradient flows ONLY into `prob_fake(clean)`. This was verified by the `test_loss_gradient_flows_only_into_clean` unit test:

```
grad on clean row, prob_fake column: -1.0 (gradient descent pushes clean UP)
grad on teams row, prob_fake column:  0.0 (teams unchanged by design)
```

The asymmetric design implements the FACTS-derived rule: the lever raises `prob_fake(raw_clean)` to match `prob_fake(raw_teams)`. The reverse direction (pulling teams down to raw) is out of scope per the historical CPU-2 + E2B evidence.

---

## 3. Motivation FACTS

CPU-2 (2026-05-22) on Slot A v2 step3500 CLS-pool over 275 paired viso fakes:

| Quantity | Value |
|---|---:|
| mean(prob_fake \| raw_clean) | 0.5029 |
| mean(prob_fake \| raw_teams) | 0.5419 |
| Δ = mean(teams) − mean(clean) | +0.0390 |
| Wilcoxon p-value (paired) | 0.0020 |
| Cohort partition at τ=0.20: `teams>τ AND clean≤τ` | 62 |
| Cohort partition at τ=0.20: `clean>τ AND teams≤τ` | 1 |
| Asymmetry ratio (target:wrong_way) | 62 : 1 |

The 62:1 asymmetry indicates the model already favors teams over clean in the right tail of the score distribution. The one-sided hinge attacks this gap by raising clean to match teams.

---

## 4. Unit test results

Same 138-test sweep as BACKBONE-SlotAv2 passes (see SlotAv2 `RESULTS_FACTS_2026-05-22.md` §3).

Notable invariants verified for the asymmetric loss:

1. One-sided hinge. When `prob_clean > prob_teams`, the loss returns 0 (no penalty).
2. Detached teams. The gradient w.r.t. teams's `prob_fake` is exactly 0 (verified by autograd).
3. Multi-pair pooling. Two matched pairs of (clean=0.3, teams=0.7) → mean hinge=0.4 → lambda=0.3 → 0.12 (verified end-to-end through collate + loss in `test_end_to_end_collate_plus_asymmetric_loss`).
4. B*T expansion. Per-frame `prob` rows are correctly indexed back to per-video `substrate_pair_id` via `repeat_interleave`.

---

## 5. Inventory coverage caveat (documented gap)

Identical to BACKBONE-SlotAv2 §4. The asymmetric pair-loss operates on matched (clean, teams) pairs found in the batch via `substrate_pair_id`. Today the practical pair-fraction is bounded by the `visomaster_teams_enhanced` lane's 54-row contribution.

For BACKBONE-T5C this is MORE relevant than for SlotAv2 because the loss requires matched pairs to fire at all. The loss returns 0 when no matched pair is found in the batch — a no-op. The effective `lambda_pair` × pair_fraction is the load-bearing knob.

A follow-on packet to wire `hdtf_visomaster_teams` + `quickclips_visomaster_teams` as new data-source lanes would raise the pair-fraction substantially. This is recorded as an open loop for next-agent attention.

---

## 6. Vertex submission (PENDING)

No Vertex job has been submitted. The yaml + production code are ready for `dev.sh build-prod -y` + `arena/launch_*.sh` invocation when the user authorizes.

Expected smoke cost: ~$3 (us-central1, A100×1, 200 steps with periodic_saves at [100]).
Expected full-run cost: ~$45-65 (us-central1, A100×1, 3500 steps, ~3.5h).

Region: us-central1 per the task. Fallback per CLAUDE.md region-capacity protocol: us-east1.

---

## 7. Scorecard (PENDING)

The standard contract scorer (`arena/score_teams_promotion_contract.py`) is unmodified and can be invoked under both lex and composite λ=1.0 once Vertex produces checkpoints at `gs://training-job-outputs/best_checkpoints/<run_id>/`.

The face-pool scorer at `analysis/face_pool_scorecard_2026-05-22/score_face_pool_suites.py` is also applicable — it tests compatibility with the face-pool inference readout per the task.

---

## 8. Sentinel file

`_phase_2_backbone_t5c_complete.json` — not written. Will be created on Vertex completion + scorecard finalization.
