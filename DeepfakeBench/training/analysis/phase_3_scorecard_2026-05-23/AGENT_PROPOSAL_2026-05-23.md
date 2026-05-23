# Phase 3 — scorecard + ship/iterate/abort decision (opinion doc)

Status: COMPLETE 2026-05-23 ~00:35 UTC.

FACTS in `RESULTS_FACTS_2026-05-23.md`.

---

## Phase 3 verdict — fall back to the $0 face-pool inference baseline

All three Phase 2 arms failed to clear the four-gate deploy criterion:

| Arm | Verdict | Detail |
|---|---|---|
| HEAD (face-pool head-only retrain) | ITERATE | Plateau at composite=0.235; viso stuck at 7%; Pareto-equivalent to $0 face-pool inference. Defer to Phase 4 HEAD ALT (dual-readout). |
| BACKBONE-SlotAv2 (GroupDRO substrate-balanced) | ABORT | Composite 0.502 vs base 0.331; viso 0.5%; conservative-boundary collapse. Lever-class refuted on Slot A v2 base. |
| BACKBONE-T5C (asymmetric pair-loss) | BLOCKED | Three smokes loss=0; three root causes debugged; final root cause (sampler design) requires non-overnight redesign. |

Per the plan's Phase 3 outcome matrix, when **all three fail**, the deployment recommendation is:

**Slot A v2 step3500 + face-pool inference ($0 baseline)** with composite λ=1.0 = 0.249, lockbox_real_fpr = 0.0154, lockbox_fake_recall = 0.7668, viso recall = 0.0673.

This is the same model that was already deployment-candidate before Phase 2 started. Phase 2 spent ~$120 confirming the lever set explored doesn't beat it, AND surfaced the silent-no-op data-pipeline bug chain that would have masked future similar packets.

---

## Final ranking (composite λ=1.0 ascending)

| Rank | Arm | Readout | composite | lockbox_fpr | lockbox_fake | viso | Status |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | **HEAD step 250** | face-pool (trained) | 0.235 | 0.0176 | 0.7826 | 0.073 | ITERATE → Phase 4 |
| 2 | **Slot A v2 step3500 ($0 baseline)** | face-pool inference | 0.249 | 0.0154 | 0.7668 | 0.067 | **DEPLOY** |
| 3 | Slot A v2 step3500 (training base) | CLS | 0.331 | 0.0191 | 0.6877 | 0.167 | — |
| 4 | BACKBONE-SlotAv2 step 3500 | CLS | 0.502 | 0.0118 | 0.5099 | 0.005 | ABORT |
| — | BACKBONE-T5C step 3500 | — | BLOCKED | — | — | — | sampler design |

HEAD step 250 has a marginally better composite than the $0 baseline (-0.014), with similar lockbox metrics. The trade is within noise on most metrics. **The $0 baseline is preferred for deployment because**: same ckpt, no training cost, no risk of head-fitting introducing brittleness on un-tested suites.

If you want HEAD step 250 as the deploy variant instead, it costs the same to ship (both wrap Slot A v2 step3500), but adds dependency on the kwhju7im head weights and a slightly different runtime readout. Marginal upside.

---

## Phase 4 recommendation — HEAD ALT (dual-readout)

The HEAD verdict + CPU-1 saliency mass (72.6% non-face for viso) point at the viso ceiling being a face-pool readout limitation, not a head-boundary limitation. HEAD ALT addresses this directly:

- Detector emits BOTH centered 7×7 face_pool (49 patches, what HEAD did) AND complement 147-patch non_face_pool. Each mean-pooled, projected, concatenated → 1024-dim feature. New 1024-dim head.
- Smoke + full launch: ~$33 total.
- Confidence: 35-50% to clear all four gates (depends on whether viso signature is recoverable from any per-patch mean-pool).

If HEAD ALT also misses viso ≥ 0.15, that's evidence that the viso signature lives in attention-map structure (not in any flat patch-pool), and the right next move is an attention-readout — different code path.

---

## Cost ledger (Phase 0-3 cumulative)

| Item | Cost |
|---|---:|
| Phase 0 (docs) | $0 |
| Phase 1 CPU probes | $0 |
| HEAD smoke + full + scoring | ~$11 |
| Image builds 1.3.298, 1.3.299, 1.3.300, 1.3.301 | $0 |
| BACKBONE-SlotAv2 smoke + full | ~$53 |
| BACKBONE-T5C smokes v1+v2+v3 (3× ~$3) | ~$9 |
| Scoring on local MPS | $0 |
| Auto-launcher blocked T5C fulls v2+v3 | $0 (would have been ~$100) |
| **Total spent** | **~$73** |
| **Total saved by blockers** | **~$100** |
| **Net efficiency** | +27% under the naive $173 worst-case |

Budget remaining: ~$77 of $150. Enough for HEAD ALT (smoke + full = ~$33) with $44 headroom for one more arm.

---

## Outstanding work for the morning user

1. **Decide deploy variant**: $0 face-pool inference baseline (recommended) OR HEAD step 250.
2. **Authorize HEAD ALT**: detector code change + new yaml + smoke + full. ~$33, ~4h wall.
3. **(Optional)** investigate T5C sampler redesign if the asymmetric pair-loss hypothesis is still load-bearing. ~3-4h dev + new smoke + full ~$53. Risk: structural mechanism (62:1 cohort) may have moved on after Phase 2 anyway.

The plan file at `/Users/roeedar/.claude/plans/ok-so-we-don-t-valiant-quasar.md` should be updated to reflect Phase 3 closure + Phase 4 prioritization. Handoff doc at `docs/relaunch_handoffs/HANDOFF_2026-05-23_OVERNIGHT_PHASE2_VERDICTS.md` (will be updated to include this verdict).
