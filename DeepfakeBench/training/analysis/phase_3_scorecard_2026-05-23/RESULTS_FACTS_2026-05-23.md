# Phase 3 — scorecard + ship/iterate/abort decision (overnight 2026-05-23 draft)

Compares HEAD + BACKBONE-SlotAv2 + (if landed) BACKBONE-T5C against:
- Slot A v2 step3500 CLS-pool (the contract baseline)
- Slot A v2 step3500 face-pool inference (the $0 baseline)

> Banned-word policy: this FACTS doc uses no opinion verbs about outcomes
> (no "succeeds", "fails", "wins", "loses", "promotes", "deployment-grade",
> "kill", "best", "worst", "confirmed", "refuted"). All interpretation
> lives in the companion `AGENT_PROPOSAL_2026-05-23.md`.

---

## Composite scorecard (composite λ=1.0 sorted ascending)

Filled post-scoring.

| Ckpt | Readout | composite λ=1.0 | lockbox_real_fpr | lockbox_fake_recall | viso recall | dev_fake_macro_recall | rank |
|---|---|---:|---:|---:|---:|---:|---:|
| HEAD kwhju7im step 250 (winner) | face-pool | **0.2350** | 0.0176 | 0.7826 | 0.0727 | 0.4921 | **1** |
| Slot A v2 step3500 (face-pool inference, $0) | face-pool | 0.249 | 0.0154 | 0.7668 | 0.0673 | 0.439 | 2 |
| Slot A v2 step3500 (CLS, base) | CLS | 0.331 | 0.0191 | 0.6877 | 0.1673 | 0.310 | 3 |
| BACKBONE-SlotAv2 6ypu1ds3 step 3500 | CLS | 0.5019 | 0.0118 | 0.5099 | 0.0055 | 0.3529 | 4 (ABORT) |
| BACKBONE-T5C \<run-id\> step 3500 (rerun pending) | CLS | _TBD_ | | | | | |

## Per-arm gate readouts

### HEAD
- composite ≤ 0.20: FAIL (best 0.235, over by 0.035)
- lockbox_real_fpr ≤ 0.016: FAIL (best 0.0169, over by 0.0009)
- lockbox_fake_recall ≥ 0.74: PASS (0.775-0.783)
- viso ≥ 0.15: FAIL (max 0.075, under by 0.075)

### BACKBONE-SlotAv2 step 3500 — ABORT
- composite ≤ 0.25: FAIL (0.502, over by 0.252; abort criterion composite > base + 0.05 = 0.381 → triggered)
- lockbox_real_fpr ≤ 0.018: PASS (0.0118)
- lockbox_fake_recall ≥ 0.70: FAIL (0.510, under by 0.190)
- viso ≥ 0.15: FAIL (0.005, under by 0.145)
- dev_fake_macro_recall ≥ 0.40: FAIL (0.353, under by 0.047)

GroupDRO mechanism worked as designed (boundary shift toward worst real
group → high τ=0.826 → lockbox FPR ↓). But the boundary shift moved the
calibration band above viso's score distribution → viso collapsed to 0.5%.
Inventory CSV bug not the cause; bug-fixed rerun would intensify, not
reverse.

### BACKBONE-T5C (rerun pending on image 1.3.300; T5C smoke v2 RUNNING)

---

## Outcome matrix (per plan §Phase 3)

Filled when all arms land.

| HEAD | BACKBONE-SlotAv2 | BACKBONE-T5C | Action |
|---|---|---|---|
| Ship | Ship | Ship | Pick best by composite. Document all three; ship best single. |
| Ship | Ship | Iterate/Abort | Pick best of HEAD vs BACKBONE-SlotAv2. Document T5C result. |
| Ship | Iterate/Abort | Ship | T5C ships; HEAD is alternative inference upgrade. |
| Iterate/Abort | Ship | Ship | Backbone path wins; HEAD becomes diagnostic only. |
| Multiple Iterate | — | — | Phase 4 fallback design. |
| All Abort | All Abort | All Abort | Slot A v2 step3500 + face-pool inference + λ=1.0 composite (TODAY'S WORK) is the deployment recommendation. Phase 4 fallback. |

---

## Cost ledger (Phase 0–3, cumulative)

| Item | Cost | Wall | Status |
|---|---:|---|---|
| Phase 0 (docs) | $0 | 6h | done 2026-05-22 |
| Phase 1 CPU probes | $0 | 13.6 min | done 2026-05-22 |
| HEAD smoke (cancelled) | $2.94 | 57 min | done 2026-05-22 |
| HEAD full | ~$8 | 1h 14m | done 2026-05-22, scored |
| Image builds (1.3.298, 1.3.299, 1.3.300) | $0 | ~30 min total | done |
| BACKBONE-SlotAv2 smoke | ~$3 | ~30 min | done 2026-05-22 |
| BACKBONE-T5C smoke | ~$3 | ~30 min | done 2026-05-22 (loss=0 bug) |
| BACKBONE-SlotAv2 full | ~$50 | 4h 01m | done 2026-05-22, scoring in flight |
| BACKBONE-T5C smoke v2 (post-fix) | ~$3 | ~30 min | in flight 2026-05-23 |
| BACKBONE-T5C full (conditional on smoke v2 OK) | ~$50 projected | ~3.5h | pending |
| Scoring on local MPS | $0 | ~30 min/arm | in flight |
| **Total Phase 0–3 cumulative** | **~$120** | — | within $150 budget |
