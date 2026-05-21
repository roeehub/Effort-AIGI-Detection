# Codec restoration triple VERDICT — FACTS — 2026-05-21

> Pass-1 FACTS only. Pass-2 OPINIONS are at the bottom of this doc.
> Read MORNING_DECISION_TREE_2026-05-21.md + FACTS_2026-05-21.md first.

## Headline

Codec aug **partially** closes the natural-experiment transport gap (37-60%
reduction in Δ across 3 packets) but **none of the 3 packets meet the
deployment-grade closure criterion** (|Δ| ≤ 0.05). All 3 packets **regress
may6 production-drift false-flag rate** vs the Slot A v2 base they were
FT'd from (Slot A v2 4.3% → Slot 1/2 6.5/8.7% with anchor, Slot 3 12.0%
without anchor). **Slot A v2 step3500 remains the best deployment ckpt**;
none of tonight's packets are upgrades.

## Verdict table

| ckpt | anchor | codec_p | FT base | Roy_D | Guest | Δ | Δ reduction | may6 frac>0.5 | Verdict |
|---|:---:|---:|---|---:|---:|---:|---:|---:|---|
| SLOT_A_V2_STEP3500 (ref) | ON | 0.00 | T5C | 0.7668 | 0.5987 | -0.1681 | — | **0.043** (best) | best deploy |
| T5C_STEP3500 (ref) | ON | 0.00 | P22 base | 0.7952 | 0.6275 | -0.1677 | — | 0.174 | not deploy |
| **Slot 1** | ON | 0.40 | Slot A v2 | 0.7946 | 0.7121 | -0.0825 | **-51%** | 0.065 | partial+regress |
| **Slot 2** | ON | 0.20 | Slot A v2 | 0.8061 | 0.7391 | -0.0670 | **-60%** | 0.087 | partial+regress |
| **Slot 3** | OFF | 0.40 | T5C | 0.8253 | 0.7195 | -0.1058 | -37% | 0.120 | partial+regress |

Closure criterion (per MORNING_DECISION_TREE §3): trained ckpt natural-experiment |Δ| ≤ 0.05. NONE met.
may6 criterion (per FACTS §gates): may6 frac>0.5 ≤ Slot A v2 ref + 5pp. Slot 1 PASS (0.065 vs 0.043 + 5pp = 0.093). Slot 2 PASS at edge (0.087). Slot 3 FAIL (0.120 > 0.093).

## Mechanism interpretation (FACTS)

### 1. Codec aug DOES bite the transport axis

Net Δ reduction across all 3: 37-60%. Direction was correct on the pre-launch CPU Gate 1 (single-frame test) at 42-50% of natural Δ reproduced. The trained-encoder result is in the same range, confirming the gate predicted training-time behavior reasonably.

### 2. Closure is asymmetric — Guest moves UP toward Roy_D, not Roy_D down toward Guest

- Reference Slot A v2: Roy_D=0.767, Guest=0.599 (the Guest crop scored LOW = correct as real)
- Slot 1: Roy_D=0.795 (+0.028 from base), Guest=0.712 (+0.113 from base)
- Slot 2: Roy_D=0.806 (+0.039), Guest=0.739 (+0.140)
- Slot 3: Roy_D=0.825 (+0.059), Guest=0.720 (+0.121)

The encoder lost discrimination between Roy_D (high) and Guest (low) by raising Guest's score, NOT by lowering Roy_D's. After training, BOTH the Roy_D crop AND the Guest crop score in the 0.71-0.83 range — both would be false-flagged at deployment τ=0.70+. The mechanism is "transport-invariance learned by uplifting the transport-shifted real to look more like a fake", not "Roy_D's false-flag corrected".

### 3. Dose-response is non-monotone

- Slot 1 (p=0.40): Δ=-0.083, may6=0.065 — best may6 of the 3 anchor-on packets
- Slot 2 (p=0.20): Δ=-0.067 (BEST Δ closure), may6=0.087 — worse may6 than Slot 1

Lower codec dose gives BETTER natural-exp closure but WORSE production-drift may6 result. Counter-intuitive — the higher dose's stronger aug may regularize the encoder more uniformly across the production cohort, while lower dose creates per-frame variance.

### 4. Anchor mechanism is binding for may6 production-drift

Slot 3 (anchor=OFF, codec=0.40) has may6 frac>0.5 = 0.120 (2.8× Slot A v2's 0.043, FAILING the criterion). Slot 1 (anchor=ON, codec=0.40) has 0.065 (1.5× Slot A v2). Anchor mechanism contributes ~5pp may6 frac>0.5 reduction at the same codec dose. anchor=ON is structurally preferable.

### 5. Train-time AUC is misleading

All 3 packets had train-time `auc0.994-0.996` on the periodic checkpoints, similar to Slot A v2 (`auc0.9952`) and T5C (`auc0.9944`). This is consistent with `project_train_auc_not_valid_promotion_signal.md`: AUC saturates around 0.99+ on the training distribution while the operational metrics (lockbox FPR, may6, natural-experiment Δ) move in different directions.

## What this verdict does NOT establish

- **Whether higher codec dose (p>0.40) would fully close Δ**: not tested. Slot 1 at p=0.40 still has Δ=-0.083. Either dose-response saturates above 0.40, or it would close further with stronger aug, OR the encoder cannot fully decouple transport from real-vs-fake on this base.
- **Whether composite levers would help**: codec + Roy_D anchor pool + face_scale_jitter were not jointly tested. Roy_D anchor pool specifically targets the per-identity component of the failure mode.
- **Lockbox/dev contract scorecard**: not run for tonight's ckpts. Single-frame natural-exp + 92-frame may6 are surrogate measurements. The 29-suite scorecard would confirm lockbox_real_fpr / lockbox_fake_recall.
- **Whether step1500 or step2500 would behave differently from step3500**: only step3500 evaluated. Earlier ckpts may show different trade-offs (per memory `project_t5c_chronic6_partial_recovery_2026-05-12.md` — step1500 outperformed step3500 in past packets).

## Decision-tree branch (per MORNING_DECISION_TREE)

This is a **NUANCED BRANCH D** — codec aug works partially but doesn't close the transport gap as a single lever, and partial closures come with may6 regression. NOT pure Branch D (which would be "no mechanism activity") because:
- Δ reduction 37-60% is clear mechanism activity
- The direction is correct (Δ shrinks toward 0)
- The cost (may6 regression) is small enough that the mechanism is workable in principle

Specifically:
- **Branch A (ship Slot 1)**: refuted — Δ=0.083 > 0.05 criterion + may6 regressed.
- **Branch B (codec binding, ship Slot 3)**: refuted — Slot 3 worse on both metrics than Slot 1.
- **Branch C (compression trap on Slot 1)**: refuted — Slot 1 may6 OK, no extreme regression.
- **Branch D (codec doesn't close gap)**: PARTIAL TRUTH — direction right, magnitude insufficient.
- **Branch E (Slot 2 sweet-spot)**: SUPPORTED on Δ axis (Slot 2 best closure), REFUTED on may6 (worse than Slot 1).

## Recommendation paths for next packet (OPINIONS — open for user review)

OPINIONS only — not yet evidence-supported:

1. **Higher codec dose (p=0.60-0.80) on Slot A v2 base** — Slot 1 may be sub-saturated; the dose-response curve from this session can't distinguish "lever saturates at p=0.40" from "lever bites monotonically and we haven't pushed hard enough". A single packet at p=0.70 would settle it. Cost: ~$30. Risk: may6 regression may worsen above some threshold.

2. **Roy_D anchor pool extension + codec p=0.40 on Slot A v2 base** — codec attacks the transport axis but Roy_D specifically remains at 0.79-0.83 (he's being flagged FAKE in his own face crops). Roy_D anchor pool would directly suppress this. Multi-step infra change (per HANDOFF 2026-05-20 §5: bucket + frames upload + registry edit, ~60-90 min with silent-failure risk).

3. **Composite: codec p=0.20 + face_scale_jitter@0.30** — face_scale_jitter@0.50 was refuted as composable on T3/T4 bases (memory `project_face_scale_jitter_load_bearing.md`); at @0.30 on Slot A v2 base with codec it might be a different story. Speculative.

4. **Ship Slot A v2 step3500 with deployment-side per-account τ** — if the per-Teams-account natural experiment represents a deployment-fatal axis, the cost-effective fix is operational: detect Teams-account/build and apply different τ per account. CPU-only, $0, but requires per-account labeling at deployment time.

5. **Multi-account capture sweep before more training** — `analysis/teams_account_natural_experiment_2026-05-19/§6.1` recommendation; user-side capture task. If only 1-2 account profiles matter operationally, the cost-benefit of more training-time interventions is unclear.

User decision points (per `feedback_decision_points`):
- Which of paths 1-5 to pursue?
- Whether to ship Slot A v2 step3500 as-is or wait for next packet's verdict.

## Artifacts

| Path | Contents |
|---|---|
| `run_verdict.py` | Script (180 LOC) that produced this verdict |
| `outputs/verdict_natural_experiment.json` | Raw natural-exp results (3 ckpts + 2 refs) |
| `outputs/verdict_may6.json` | Raw may6 cohort stats (3 ckpts) |
| `ckpt_cache/` | Locally cached step3500 ckpts (3 × 898 MB) |
| W&B runs | `ohcx4w0x` (Slot 1, glowing-grass-1652), `xg35bae4` (Slot 2, resilient-tree-1650), `21alo5iu` (Slot 3, hardy-eon-1651) |
| GCS ckpts | `gs://training-job-outputs/best_checkpoints/{run_id}/periodic_effort_20260521_step3500_*.pth` |
