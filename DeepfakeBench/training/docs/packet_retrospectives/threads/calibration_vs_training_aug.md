# Thread: Calibration vs training-aug (lever ranking)

> **⚠ Critical-reading note (added 2026-05-04 night)**: the "70/30 training-aug / calibration" ratio in the Current stance assumed deployable per-camera calibration was on the table. The 2026-05-04 update section below qualifies this — the per-substrate calibration lift is large (24.5pp) but OFFLINE-ONLY because Teams does not surface capture mode at inference (`feedback_per_mode_tau_not_deployable.md`). Read the calibration share as referring to the narrower "per-camera at write-time / per-identity-with-prior" subset, not per-mode-at-inference. The "training-aug is dominant" conclusion strengthens under this reading; the "calibration is complementary" framing applies primarily to substrate-level filtering (modern_v2) and per-known-identity readouts, not to general per-mode policies.

## The question

Given a fixed backbone with a diagnosed camera/ISP-signature shortcut (see [processing_signature_shortcut](processing_signature_shortcut.md)), which lever closes the cross-pool FPR gap: per-camera calibration at inference time, or training-time augmentation that forces the backbone to drop the shortcut? Both look plausible on paper — the question is ratio.

## Initial belief

Going into [RLP7](../packets/RLP7.md) planning, per-camera calibration was the hopeful short-path lever. If the shortcut is "the model is well-ordered within a pool but the score distributions are shifted across pools," then a single per-pool τ plus a scalar shift would close the gap without retraining. WS-P2.b (per-identity reducer) was built partly in anticipation of this: if calibration carries the fix, you only need better per-identity readouts, not a new training run.

## What changed our mind

- **[WS-P1 probe](../packets/WS_probes.md)** (`analysis/calibration_probe_2026-04-24.summary.json`). Per-camera calibration closes **30.1%** of the cross-pool FPR gap on average across target FPRs {5, 10, 15, 25}%. Per-target breakdown: `{5%: 0.000, 10%: 0.333, 15%: 0.286, 25%: 0.583}`. Verdict field: `mixed_both_levers_needed`.
- **The plan's decision thresholds** (`summary.verdict_thresholds` in the same file) set ≥0.60 = "calibration is the primary lever" and ≤0.30 = "training-aug is the primary lever." 0.301 sits on the training-aug side of the boundary by `0.001`. Boundary result, but the right-hand boundary.
- **Per-pool means confirm the shape of the problem.** `dor-real-webcam-false-flag-no-virtual-bg` mean prob_fake 0.940; `dor-real-laptop-correct-no-virtual-bg-whiteish` 0.018. Same person, ~0.92 score separation. A 30% gap closure means the post-calibration spread is still ~0.65 — well above any operating FPR.
- **Feature-space diagnostics** (`analysis/dor_baseline_repro_2026-04-24.summary.json:947-983`) showed the pool-separating signal is concentrated in `dct_hf_ratio` (max_sep 3.46), `wb_rb_ratio` (2.32), `mean_cb` (2.23), `mean_cr` (1.97). Two codec-aligned, two chroma/lighting. A scalar per-pool shift at inference is mathematically incapable of undoing a shift that lives in a distributed feature-norm geometry.

## Current stance (2026-04-24)

Training-aug is the dominant lever; per-camera calibration is complementary, not primary. Ratio is roughly **70/30 training-aug / calibration** on the gap not yet closed. Concretely:

- [RLP7](../packets/RLP7.md) yamls embody this choice. `RLP7_04` (Teams spatial-only) and `RLP7_05` (spatial + moderate codec) are the two training-aug slots on the critical path. `RLP7_02` (codec-aggressive) is the fallback axis.
- Per-camera calibration stays on the table as an **adjunct**. If an RLP7 checkpoint lands with narrower but nonzero pool FPR spread, per-camera τ can recover the residual 30% without retraining.
- Per-identity reducer (WS-P2.b) is the readout primitive that makes calibration evaluable at launch time — it surfaces per-identity `real_fpr` / `fake_recall` so "new Dors" are cheap to catch.

**Caveat pinned.** WS-P1's input data is pre-fix (`HANDOFF.md:263`). The 30.1% is computed through the pre-INTER_LINEAR retro-score path. Re-running on post-fix numbers is the obvious next probe. The anchor-pool delta being −0.008 (from `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`) makes it unlikely the verdict flips, but possible it shifts a few percent either direction.

## Packet timeline

- [RLP6](../packets/RLP6.md) — slot-04 2-camera test makes the gap legible. `Dor webcam 0.94 vs Dor laptop 0.02` is the per-pool separation calibration would have to close.
- [WS probes](../packets/WS_probes.md) — WS-P1 runs the probe; WS-P2.b ships the per-identity reducer as the readout primitive.
- [RLP7](../packets/RLP7.md) — the training-aug response; calibration held in reserve.

## Evidence locations

- `analysis/calibration_probe_2026-04-24.summary.json` — founding datum (0.301 average gap closure, per-target breakdown, verdict thresholds)
- `analysis/dor_baseline_repro_2026-04-24.summary.json:947-983` — fingerprint ranking (`dct_hf_ratio`, `wb_rb_ratio`, `mean_cb`, `mean_cr`)
- `analysis/dor_pool_fingerprints_2026-04-24.summary.json:219-223` — `recommended_variant = V3_combined`
- `arena/postprocess_per_identity.py` — WS-P2.b CLI
- `tests/test_postprocess_per_identity.py` — guard
- [WS_probes.md](../packets/WS_probes.md) — probe details
- [RLP7.md](../packets/RLP7.md) — downstream yamls
- Handoff: `docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md:40-62`

## Open loops

- **WS-P1 re-run on post-fix inputs.** Pre-fix data is the known weakness. Re-running after a broader post-fix re-score sweep is on the implicit queue.
- **What closes the residual ~70%?** Training-aug does not automatically close all of it. If RLP7 variants land at, say, 50% gap closure via augmentation + 30% via calibration, the remaining 20% may need a different lever (representation loss, architectural change, or a third data source). No current plan past RLP7 for that 20%.
- **Per-identity reducer usage at promotion time.** `HANDOFF.md:233-240` frames "narrowest per-identity FPR spread" as the explicit RLP7 leader-pick metric. But integrating this into the promotion contract as a hard gate (rather than a sort key) is not codified.

### 2026-05-04 evening update — per-substrate τ-calibration tool quantifies the calibration ceiling at deployment-undeployable

The thread's "calibration is complementary, not primary" finding (WS-P1 30.1% gap closure) measured per-camera calibration. A new reusable per-substrate τ-calibration tool (`analysis/per_substrate_tau_calibration_2026-05-05/`, see [`webcam_fpr_dominance`](webcam_fpr_dominance.md) "2026-05-04 evening update") measures **per-capture-mode** calibration — a different but related calibration axis. The tool quantifies on P8A:

- Oracle per-mode dev-calibrated τ → lockbox `teams_fake_lockbox` recall **78.6% at 16.9% FPR** vs single global τ **54.1% at 4.3% FPR** = **24.5pp recall lift at 4× the FPR**. This is large.
- Single-τ-substrate-aware (the deployable subset of the above): at the cleanest deployable operating point (5% worst-substrate FPR ceiling on P8A), τ=0.9941 gives lockbox FPR 0.07% but viso recall **0.18%** and deeplive recall **0.0%**. Even at 20% FPR ceiling, viso recall is 21%. Deployment-honest recall under deployable single-τ stays far below the 90% target.

**The 24.5pp lift is OFFLINE-ONLY** (per user's deployment constraint, established this session): there is no way to detect `clip_capture_mode` at production. The mode classifier reaches only 52% CV accuracy on 4 IQ features (`analysis/viso_capture_mode_proxy_2026-05-05/summary.json`).

**Implication for this thread's stance**: the "70/30 training-aug / calibration" ratio measured what calibration CAN do offline at the per-camera level. The 2026-05-04 measurement strengthens the "calibration alone is insufficient" claim by showing that even the per-substrate calibration that DOES achieve large offline lifts can't be deployed. The training-aug lever is even more clearly the only deployable path; the 70/30 ratio understates it because the 30 calibration share assumes deployable per-camera calibration is available, which it largely isn't.

The `residual-70-pct-fpr-gap-no-lever-past-rlp7` open loop below remains the right framing — but it should be read as "training-aug + substrate-level data hygiene (modern_v2 filter at the eval level, NOT per-frame inference policy) is the deployable forward path." Per-substrate τ calibration is an analysis-only lever for understanding score distributions.

### Cross-thread refs

- [`processing_signature_shortcut`](processing_signature_shortcut.md) — the camera/ISP shortcut is the failure surface this thread's lever ratio is being spent against. Loop `shortcut-deployment-block` in that thread is the load-bearing close criterion the 70/30 ratio must serve.
- [`preprocessing_parity_bug`](preprocessing_parity_bug.md) — WS-P1's pre-fix inputs are the cause of the boundary-result caveat. Loop `ws-p1-rerun-on-post-fix-inputs` is in that thread; the verdict shift (if any) lands here.
- [`promotion_contract_evolution`](promotion_contract_evolution.md) — per-identity reducer integration into the contract is the codification axis. The current state (sort-key, not gate) leaves the camera-signature failure mode reachable through the same τ-tail-collapse pattern as `fpr-minimization-no-budget-tau-collapse`.
- [`webcam_fpr_dominance`](webcam_fpr_dominance.md) — 2026-05-04 evening per-substrate τ-calibration tool details + the deployability caveat that limits this thread's calibration share to offline analysis.

### Open loop: residual-70-pct-fpr-gap-no-lever-past-rlp7
status: open
severity: medium
first_seen: 2026-04-24
last_verified: 2026-04-29
close_criterion: a Packet-7+ readout reports per-pool FPR spread on the 6 Dor/Roee pools (anchor `dor-real-webcam-false-flag-no-virtual-bg` ≤ 0.30 + clean pools ≤ 0.05) — i.e., the training-aug + calibration combination demonstrably closes ≥70% of the cross-pool FPR gap; OR a documented next-lever proposal (representation loss, data-side intervention, architectural change) is filed for the residual

WS-P1's 0.301 closes only 30% of the gap; the thread's stance assumes RLP7 training-aug closes most of the rest. If RLP7 training-aug closes, say, 50% (combined with WS-P1's 30%, leaves a 20% residual), no current packet plan attacks the residual. The loop is `medium` because the project may converge before the residual matters — but it is the obvious "what's next after RLP7" question and shouldn't fade off the radar. Cross-thread ref: this loop is paired with `shortcut-deployment-block` ([`processing_signature_shortcut`](processing_signature_shortcut.md)); the close criteria are different framings of the same outcome (per-camera FPR spread vs lockbox 90/90).

### Open loop: per-identity-reducer-not-a-contract-gate
status: open
severity: low
first_seen: 2026-04-24
last_verified: 2026-04-29
close_criterion: `arena/score_teams_promotion_contract.py` (or its successor contract yaml) hard-gates on `max(per-identity real_fpr) ≤ <budget>` rather than admitting it as a sort key only — i.e., a checkpoint that has 95% mean real-pool TPR but a single-identity 50% FPR fails the contract by construction

WS-P2.b (`arena/postprocess_per_identity.py`, commit `deac44e`) ships the readout primitive. `HANDOFF.md:233-240` defines "narrowest per-identity FPR spread" as the explicit RLP7 leader-pick. But the integration into the contract is at the operator-discipline level, not the runner level — same shape as the `fpr-minimization-no-budget-tau-collapse` operator-discipline gap. If the project formally promotes the per-identity FPR ceiling into the contract, the camera-signature shortcut at any single identity becomes a contract-level fail rather than something the operator has to remember to read. `low` severity because the operator-discipline path is currently producing the right answer (Dor/Roee anchor pools are the de facto per-identity gate); the loop tracks the codification gap, not a missed-finding risk.
