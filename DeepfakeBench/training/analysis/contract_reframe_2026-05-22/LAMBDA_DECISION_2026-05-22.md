# Track B λ-Decision Artifact (2026-05-22)

**Policy chosen:** `tiebreak_policy = "composite"`, `tiebreak_lambda = 1.0`.

The user has selected λ = 1.0 — the canonical default that prices one percentage point of `lockbox_real_fpr` equivalently to one percentage point of `lockbox_fake_recall`. Any λ in [0.01, 10] gives the same ordering on the published 2026-05-20 panel; λ = 1.0 sits in the middle of that range.

## Composite scores on the 2026-05-20 panel at λ = 1.0

Computed by applying `composite = lockbox_real_fpr + 1.0 × (1 − lockbox_fake_recall)` to the published per-suite numbers from `docs/packet_retrospectives/STATE.md` lines 100–103.

| rank (composite) | rank (lex) | ckpt | lockbox_real_fpr | lockbox_fake_recall | composite (λ=1.0) | tier |
|---:|---:|---|---:|---:|---:|:---:|
| 1 | 2 | SLOT_A_ANCHOR_AWARE_STEP3500 | 0.0191 | 0.688 | **0.3311** | 0 |
| 2 | 3 | T5C_PERIODIC_STEP3500 | 0.0279 | 0.660 | 0.3679 | 0 |
| 3 | 1 | P8A_REFERENCE_STEP5000 | 0.0184 | 0.387 | 0.6314 | 0 |
| 4 | 4 | SLOT_A_ANCHOR_AWARE_STEP1500 | 0.0044 | 0.640 | (n/a) | 1 |

SLOT_A_ANCHOR_AWARE_STEP1500 stays at rank 4 across all λ because its `dev_fake_macro_recall = 0.207` is below the 0.30 floor; the recall-floor tier-gate demotes it to tier 1, which is independent of the post-tier tiebreak policy.

## Rank shifts vs. lex

- **rank 1 ↔ rank 2:** lex puts P8A at #1 (lockbox_real_fpr 0.0184 < 0.0191); composite puts Slot A v2 step3500 at #1 (0.3311 < 0.3679 < 0.6314).
- **rank 3:** composite pushes P8A to #3 because its 30.1pp recall deficit (vs. Slot A v2) and 27.3pp recall deficit (vs. T5C) outweighs its 0.7pp and 9.5pp FPR advantages at λ = 1.0.
- **rank 4:** unchanged (tier-1 demotion).

## Deployment implication

Under the operator-selected composite policy at λ = 1.0, the deployment candidate is **SLOT_A_ANCHOR_AWARE_STEP3500**. The change does not require any model training, retraining, or re-evaluation — only the policy flag on the cross-checkpoint ranker. The underlying per-suite probabilities and the within-checkpoint τ-selection logic are unchanged.

## Reproducibility

Live rerun against the existing reports (when the operator has WANDB_API_KEY / WANDB_ENTITY / WANDB_PROJECT exported, per memory `feedback_promotion_contract_launch.md`):

```
python arena/score_teams_promotion_contract.py \
  --report_root gs://training-job-outputs/slot-a-v2-validation-2026-05-20/ \
  --checkpoint_map arena/checkpoint_maps/teams_target_domain.slot_a_v2_validation_2026-05-20.yaml \
  --checkpoints ALL \
  --output_dir analysis/contract_reframe_2026-05-22/composite_rerank/ \
  --tiebreak_policy composite \
  --tiebreak_lambda 1.0
```

The artifacted scorecard should reproduce the ordering above byte-identically. Unit tests in `tests/test_score_teams_promotion_contract.py` already validate the formula on these exact panel numbers (`test_composite_tiebreak_score_formula`).

## What this artifact does NOT do

- Update `arena/checkpoint_maps/teams_target_domain.slot_a_v2_validation_2026-05-20.yaml` or any production-deployment-pointing checkpoint resolver. That swap is a separate operator action.
- Append the policy change to `docs/packet_retrospectives/STATE.md`'s rolling section. Memory `wiki_ref:` updates and TIMELINE.md entries are a separate housekeeping step.
- Bind any future ckpt scorecard to composite policy. The default in code stays `lex`; only invocations that pass `--tiebreak_policy composite` use the new path.
