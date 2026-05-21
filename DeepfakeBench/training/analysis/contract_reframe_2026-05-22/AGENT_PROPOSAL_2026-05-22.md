# AGENT PROPOSAL — Track B Contract Reframe (Composite Tiebreak)

**Date:** 2026-05-22
**Plan reference:** `/Users/roeedar/.claude/plans/ok-so-we-don-t-valiant-quasar.md` (Track B)
**Companion FACTS doc:** `RESULTS_FACTS_2026-05-22.md`

## Headline recommendation

Deploy SLOT_A_ANCHOR_AWARE_STEP3500 in place of P8A_REFERENCE_STEP5000.

The crossover λ* = 0.00233 means the lex policy that currently ranks P8A first is operating in a regime where a single percentage point of `lockbox_real_fpr` is worth 430× the same percentage point of `lockbox_fake_recall`. No reasonable operational FP/FN cost ratio for a Teams deepfake detector justifies that weighting. The deployment-candidate change does not require any new model training, retraining, or evaluation — only the policy flag.

## Why this is a confident pick

1. **Direction.** Slot A v2 step3500 gives +30.1pp of `lockbox_fake_recall` (38.7% → 68.8%) at a cost of +0.07pp of `lockbox_real_fpr` (0.0184 → 0.0191). Even pricing FP cost at 100× FN cost (λ = 0.01) the composite picks Slot A v2 by 0.0023 absolute (a margin 33× larger than the FPR gap that lex is currently breaking ties on).
2. **Statistical robustness.** The 2026-05-20 paired-bootstrap 95% CI on the lockbox_real_fpr gap brackets [−0.008, +0.010] — it covers zero. The lex policy's rank-1 choice is driven by a quantity whose sign is not statistically resolved. The fake-recall gap of +30pp is far outside any plausible CI of the recall metric on a 105-fake-video lockbox.
3. **Pareto comparison to T5C.** Slot A v2 step3500 strictly dominates T5C step3500 on both metrics (lower FPR AND higher recall). No λ can promote T5C over Slot A v2. The current rank-3 position of T5C under lex matches its composite rank at any sensible λ.

## What changed in the codebase

- `arena/score_teams_promotion_contract.py`: `tiebreak_policy` and `tiebreak_lambda` added to `ContractConfig`, gated default `lex` (back-compat preserved); composite branch in `_promotion_summary_sort_key`; CLI flags `--tiebreak_policy` and `--tiebreak_lambda`.
- `arena/run_target_domain_validation_sequential.py`: parallel `--promotion_tiebreak_policy` and `--promotion_tiebreak_lambda` pass-throughs.
- `tests/test_score_teams_promotion_contract.py`: six new test functions; all 12 tests (6 existing + 6 new) pass.

The change is gated: every existing scorecard rerun without the new flags produces byte-identical output to the previous scorer. No historical FACTS doc is invalidated.

## What I am NOT claiming

- I am not naming a specific λ to deploy. The composite formula makes the cost ratio explicit but the choice itself is a USER product decision tied to the Teams customer experience (false-alarm rate tolerance vs. missed-fake tolerance). The recommendation above only depends on λ > 0.00233, which holds for any plausible operational choice.
- I am not claiming the contract reframe closes the underlying training problem. The natural-experiment Δ on Roy_D/Guest crops (0.168 on Slot A v2 step3500) remains a substrate-robustness gap that Track A is designed to attack.
- I am not claiming Slot A v2 step3500 is the best achievable model. Track A Phase 0 A0.2 (multi-layer CPU geometry probe, currently paused on disk-full) is intended to test whether a substrate-pair contrastive lever has a fulcrum at this base.

## What would change this recommendation

- A paired-bootstrap update showing that the +30pp lockbox_fake_recall gap is itself within a CI that covers zero. (Unlikely on n=105 fake videos but theoretically possible.)
- A user-policy decision that prices λ < 0.00233 explicitly — i.e. an operational stance that ≥430× FP-to-FN cost is the right ratio. The lex policy is the limit of composite as λ → 0, so this would be a coherent (if unusual) operational choice.
- A discovery that one of the per-suite probabilities in the published 2026-05-20 panel was miscomputed. The unit tests guard the rerank logic against drift but cannot guard the underlying suite-level numbers; those are inherited from the 29-suite job `146629015254335488`.

## Suggested next operational step (if user agrees)

1. Pick λ. Anything in [0.01, 10] gives the same ordering for the four panel rows; the canonical default if no other guidance applies is λ = 1.0 (one percentage point of FPR ≡ one percentage point of recall).
2. Rerun the existing scorecard with `--tiebreak_policy composite --tiebreak_lambda 1.0` against the `slot-a-v2-validation-2026-05-20/` report root to produce an artifacted composite scorecard, then file it under `analysis/contract_reframe_2026-05-22/composite_rerank/`.
3. Update `arena/checkpoint_maps/teams_target_domain.slot_a_v2_validation_2026-05-20.yaml` or its deployment-pointing successor so the production deployment artifact resolves to SLOT_A_ANCHOR_AWARE_STEP3500.
4. Append the policy change and λ choice to `docs/packet_retrospectives/STATE.md` rolling section.
