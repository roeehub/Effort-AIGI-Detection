# Thread: Preprocessing parity bug (INTER_AREA → INTER_LINEAR)

## The question

Every retro-scored or arena-scored checkpoint number produced before 2026-04-24 passed through `cv2.INTER_AREA` while training resized frames with `cv2.INTER_LINEAR`. Are the pre-fix numbers absolute-comparable, rank-comparable, or neither?

## Initial belief

Retro-score numbers were treated as faithful reflections of checkpoint behavior. Dor webcam at 0.94 was read as a model-level finding ("the representation false-flags this camera"). No one had audited the inference-path interpolation against training.

## What changed our mind

- [WS-P0](../packets/WS_probes.md) (commit `855871e`, 2026-04-24). The probe traced the 0.94 Dor false-flag back to `arena/model_arena.py:472` — the retro-score code path — using `cv2.INTER_AREA`. Training at `data/sources/combined_paired.py:3455` uses `INTER_LINEAR`. Batch inference at `batch_inference_gcs.py:407` had the same bug. Fix: both inference paths now use `INTER_LINEAR`. Guarded by `tests/test_inference_train_preprocessing_parity.py` (5/5 pass).
- **[RLP6](../packets/RLP6.md) slot-04 post-fix re-score** (`analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`). Anchor pool `dor-real-webcam-false-flag-no-virtual-bg` moved `pre-fix 0.940 → post-fix 0.932` (Δ = **−0.008**). Below the 0.10 threshold that would have descoped Packet-7. Clean pools stayed <0.05; false-flag pools stayed 0.93–0.96. `decision_tree.verdict = "fully_structural__launch_full_subset"`.
- **Per-pool magnitude varies.** Webcam-style pools are the most sensitive to the interpolation kernel (INTER_AREA does box-averaging which is closer to hardware downscaling; INTER_LINEAR has a bilinear signature). Pools whose capture already matches training's interpolation kernel barely move. So absolute numbers drift inconsistently — not a uniform bias.

## Current stance (2026-04-24)

Every pre-fix retro-score or arena-score number is **suspect for absolute-value comparisons**, and especially suspect when comparing across pools with different capture pipelines. Intra-packet rankings that use consistent pre-fix preprocessing across slots are **usually preserved** — the bug is a monotonic kernel change, and all slots inside a packet saw the same drift. But cross-packet absolute-value claims (e.g. "RLP5 0.7736 vs RLP3.5 0.7442 → +0.0294") mix training numbers (post-train W&B) with retro-score numbers (pre-fix arena) and should be read as directional, not numeric.

The RLP6_04 re-score is the only post-fix baseline that currently exists. The 2-camera controlled evidence for the processing-signature shortcut **survives the fix** — that finding is not an artifact of INTER_AREA.

Every future retro-score must go through the fixed path. Any pre-fix claim that gets re-used should be labeled pre-fix explicitly.

## Which packets need re-score

Rescored post-fix so far:

- **RLP6_04** — yes. `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`. Verdict `fully_structural__launch_full_subset` on the 6 Dor/Roee pools.

Not yet rescored post-fix (listed in packet order):

- **RLP1** — `best_ood_composite` is AUC-based and internal to training; only affects any arena/lockbox revisit.
- **RLP2** — same category as RLP1. Training composites internally consistent; any retro-score readout is pre-fix.
- **RLP3** — retro-NEW VC rows `0.7027 / 0.7215 / 0.7232` plus the A2b retroactive pass over RLP1/RLP2 checkpoints are all pre-fix. The retro-score code path IS the path that changed.
- **RLP3.5** — all composites (`0.7442`, `0.7435`, etc.) pre-fix. Intra-packet rankings should survive (m=0.15 vs m=0.10 vs m=0.20 all on the same pre-fix path); absolute values do not.
- **RLP4** — no numbers to rescore (all 8 jobs failed).
- **RLP5** — `best_value_composite=0.7736` is training-side post-train, but the feature-space readout at `analysis/feature_space_2026-04-23/` and the Dor/`real_dor` scores that drove the shortcut finding are pre-fix retro-score.
- **RLP6 (non-04 slots)** — all seven other slots' scorecards are pre-fix.
- **RLP6B** — never launched; no numbers exist.

Future re-score priority is implicit: whichever pre-fix leader might shuffle materially is worth checking, but the only one known to matter for current decisions is RLP6_04 (done). RLP5_07 post-fix re-score would close whether the E3 lift is still +0.0294 in the new units; not currently on the critical path.

## Evidence locations

- Commit `855871e` — WS-P0 fix (`batch_inference_gcs.py:407`, `arena/model_arena.py:472`)
- `data/sources/combined_paired.py:3455` — training reference resize
- `tests/test_inference_train_preprocessing_parity.py` — guard
- `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json` — the one extant post-fix number
- `analysis/rlp6_04_postfix_rescore_2026-04-24.per_frame.csv` — per-frame pre/post scores
- [WS_probes.md](../packets/WS_probes.md) — probe context
- [RLP6.md](../packets/RLP6.md) — consumer of the post-fix re-score
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md:26-37` — WS-P0 writeup

## Open loops

- **Deploy-server preprocessing is still drifted.** The original 0.94 Dor false-flag came through `http://34.16.217.28:8999`; the two repo paths are fixed, production is not. Separate workstream.
- **No systematic re-score sweep of prior leaders.** Only RLP6_04 has been through the fixed path. Whether RLP5_07 at post-fix still leads RLP3.5_02 is unknown.
- **WS-P1 pre-fix inputs.** The 30.1% calibration result was computed on pre-fix data per `HANDOFF.md:263`. Re-running on post-fix would either confirm or shift the "calibration is complementary" verdict. Obvious next probe.

### Open loop: deploy-server-preprocessing-drift
status: open
severity: medium
first_seen: 2026-04-24
last_verified: 2026-04-29
close_criterion: the deployment server at `http://34.16.217.28:8999` (or its successor) is verified through a `tests/test_inference_train_preprocessing_parity`-style guard to use `cv2.INTER_LINEAR` (matching `combined_paired.py:3455`), and a smoke probe through the production endpoint reproduces the post-fix anchor-pool number within ±0.02 of `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`

WS-P0 (commit `855871e`) fixed `cv2.INTER_AREA → cv2.INTER_LINEAR` at the two in-repo inference paths (`batch_inference_gcs.py:407`, `arena/model_arena.py:472`) but not at the deployment server. The original 0.94 Dor false-flag that triggered the camera-signature investigation came through the production endpoint (`http://34.16.217.28:8999`; `HANDOFF.md:53`), so the deploy-server path carries the same silent kernel drift the repo paths used to. Cross-thread ref: `processing_signature_shortcut`'s "deployment-server preprocessing is untouched" item is the same loop. The repo guard (`tests/test_inference_train_preprocessing_parity.py`) is line-scan-based and does not see the deploy server's code. This loop is `medium` rather than `high` because (a) the shortcut survives the fix anyway (post-fix anchor delta −0.008), so the deploy-server drift is not the dominant deployment risk, and (b) once the camera-signature shortcut is broken (loop `shortcut-deployment-block` in `processing_signature_shortcut.md`), the deploy-server drift becomes the next-priority production-readiness gate.

### Open loop: prior-leader-rescore-sweep
status: open
severity: low
first_seen: 2026-04-24
last_verified: 2026-04-29
close_criterion: every pre-fix leader (RLP1_01, RLP2_02, RLP3_05, RLP3.5_02, RLP5_07, RLP6_04 — RLP6_04 is the only one already done) is re-scored through the `INTER_LINEAR` path on a matched eval suite, and the resulting post-fix ranking is documented in `analysis/preprocessing_parity_post_fix_sweep_*.summary.json` (or equivalent)

Only RLP6_04 has been re-scored through the fixed `INTER_LINEAR` path (`analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`). The thread's "Which packets need re-score" section enumerates 5 other pre-fix leaders that carry pre-fix retro-score numbers (RLP1, RLP2 are AUC-only training internals so they are mostly insulated; RLP3, RLP3.5, RLP5_07 are the ones whose retro-score values are quoted as if absolute). RLP6_04's re-score moved the anchor pool only −0.008, so a wholesale ranking flip is unlikely — but the loop stays open as a low-severity completeness gap. Closing it tightens the "directional, not numeric" caveat into "directional and numeric within ±0.01" for the cross-packet comparisons that anchor most of the project's narrative.

### Open loop: ws-p1-rerun-on-post-fix-inputs
status: open
severity: low
first_seen: 2026-04-24
last_verified: 2026-04-29
close_criterion: WS-P1 (`analysis/calibration_probe_2026-04-24.py`) is re-run on the post-INTER_LINEAR re-scored RLP6_04 frames, and the resulting `avg_gap_closure_across_target_fprs` is recorded — confirming or shifting the 0.301 verdict that placed per-camera calibration on the training-aug side of the boundary

`HANDOFF.md:263` and the `WS_probes` retro both note WS-P1's input data is pre-fix. The 0.301 number sits at the decision boundary `≤0.30 = training-aug-dominant` by `0.001`. The anchor-pool delta being −0.008 makes a full verdict flip unlikely, but the value is close enough to the boundary that the precision matters for any future "calibration is complementary" claim. Re-running is cheap (single CPU script; minutes). Loop is `low` severity because the downstream RLP7 yamls already act on the training-aug verdict; if the post-fix re-run shifts the verdict to `≥0.60 = calibration-dominant`, that would meaningfully change the launch priority — which is why it stays a tracked loop rather than a backlogged nice-to-have.
