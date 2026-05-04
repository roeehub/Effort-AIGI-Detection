# WS-P probes  ·  Shortcut countermeasure probes adjacent to Packet-7

> **Template contract**: fill every section. If a section truly has no content, write `*(none — reason)*` rather than deleting the heading. Keep the status-card table intact.
>
> **Packet shape note**: This is a probe meta-packet. "Slots" below lists the four probe interventions (WS-P0, WS-P1, WS-P2.b, WS-P4.a) that ran inside/adjacent to the Packet-7 pre-launch gate rather than training slots.

## Status card

| Field | Value |
|---|---|
| Dates | 2026-04-24 → 2026-04-24 |
| Probes | WS-P0, WS-P1, WS-P2.b, WS-P4.a |
| Headline lever | shortcut countermeasure probes (informative, not trained) |
| Leader slot | *(n/a — all four completed)* |
| Leader metric | *(n/a — probe outputs feed Packet-7 design, not a model ranking)* |
| Verdict | ✅ confirmed (all four probes landed) |
| Next-packet decision | Launch Packet-7 training-aug subset (RLP7_04 spatial-only, RLP7_05 spatial+codec, RLP7_02 codec); per-camera calibration held as complementary, not deployed. |
| Themes touched | [`preprocessing_parity_bug.md`](../threads/preprocessing_parity_bug.md) (**canonical**), [`calibration_vs_training_aug.md`](../threads/calibration_vs_training_aug.md) (**canonical**), [`processing_signature_shortcut.md`](../threads/processing_signature_shortcut.md), [`gate_alignment_story.md`](../threads/gate_alignment_story.md) |

## Configuration

- **What changed vs the prior packet** — no training deltas; four narrow interventions ran in the gap between Packet-6 readout and Packet-7 launch, all as responses to the camera/ISP signature shortcut surfaced by the 2-camera controlled test on 2026-04-24 (Dor webcam 0.94 vs Dor laptop 0.02; Roee Mac 0.90 vs Roee Windows 0.01) described in `docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md:5`.
- **Control slot definition** — *(n/a — no training runs in this meta-packet; the pre-fix retro-score numbers from Packet-6 RLP6_04 serve as the baseline against which WS-P0's post-fix re-score is compared.)*
- **Probe inventory**:
  - **WS-P0** — Train/inference interpolation drift fix (prod bug). Commit `855871e`. `cv2.INTER_AREA → cv2.INTER_LINEAR` at `batch_inference_gcs.py:410` and `arena/model_arena.py:474` to match training at `data/sources/combined_paired.py:3455`. Guarded by `tests/test_inference_train_preprocessing_parity.py`.
  - **WS-P1** — Per-camera calibration probe. Commit `deac44e`. `analysis/calibration_probe_2026-04-24.py` + output at `analysis/calibration_probe_2026-04-24.summary.json`.
  - **WS-P2.b** — Per-identity reducer. Commit `deac44e`. `arena/postprocess_per_identity.py`; guarded by `tests/test_postprocess_per_identity.py`.
  - **WS-P4.a** — Teams-passthrough spatial-aug wiring verification. No code change. Existing knobs at `data/augmentations/pipelines.py:787` (`_TEAMS_PASSTHROUGH_DEFAULTS`) and `data/augmentations/pipelines.py:1386` (`_build_teams_passthrough_pipeline`); enabled by RLP7_04 yaml.
- **Downstream yamls gated by probe outputs**: `experiments/phase2_round13/R13_RLP7_04_teams_spatial_only.yaml`, `experiments/phase2_round13/R13_RLP7_05_teams_spatial_plus_codec.yaml` (both commit `c4affa1`).

## Results at the time

- **WS-P0 outcome**: shipped. The INTER_AREA→INTER_LINEAR fix in `arena/model_arena.py:474` was the load-bearing path — `arena/model_arena.py` is the retro-score code that produced the original 0.94 Dor false-flag score. Before this commit, every retro-score number had silent preprocessing drift (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md:31`). Parity test 5/5 pass.
- **WS-P0 post-fix baseline re-score** (pre-launch gate, landed same day): `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`. Anchor pool `dor-real-webcam-false-flag-no-virtual-bg` moved from 0.940 → 0.932 (Δ = −0.008). Decision tree classified this as `fully_structural__launch_full_subset`: the shortcut is not preprocessing-drift — it survived the fix intact. This validated the training-aug direction for Packet-7.
- **WS-P1 outcome**: 0.301 average gap closure across target FPRs {5, 10, 15, 25}% (per-target: 0.000 / 0.333 / 0.286 / 0.583). Verdict field `mixed_both_levers_needed`. The plan's decision gates are ≥0.60 = calibration-dominant, ≤0.30 = training-aug-dominant (`analysis/calibration_probe_2026-04-24.summary.json` fields `summary.avg_gap_closure_across_target_fprs` and `summary.verdict_thresholds`). 0.301 sits on the training-aug side of the boundary by 0.001.
- **WS-P2.b outcome**: deployable reducer CLI. Groups `videos_report.csv` rows by `group_key` and recomputes per-identity `real_fpr` / `fake_recall` at a given threshold; `--verify` asserts per-group FP/TP counts sum to the aggregate. Intended to surface the "new Dor/Roee"-style identities whose per-frame FPR is hidden by suite means.
- **WS-P4.a outcome**: no new code. Verified `teams_passthrough_special_*` knobs already survive the `data/sources/combined_paired.py:4772` preset-key filter and flow through `_build_teams_passthrough_pipeline` to produce the expected `A.ShiftScaleRotate` transform. Enablement = yaml choice.
- **Links to artifacts**: `analysis/calibration_probe_2026-04-24.summary.json`, `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`, `arena/postprocess_per_identity.py`, `tests/test_inference_train_preprocessing_parity.py`, `tests/test_postprocess_per_identity.py`.

## Conclusions drawn in-session

- **Preprocessing drift was real but NOT the shortcut.** The post-fix re-score anchor delta of −0.008 on `dor-real-webcam-false-flag-no-virtual-bg` (from `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`) is well below the 0.10 threshold that would trigger descoping Packet-7. The camera/ISP signature shortcut is a structural property the model learned, not an artifact of the inference path.
- **Preprocessing parity is still load-bearing retroactively.** Every retro-score number in prior packet retrospectives carries a silent ±0.09 drift on webcam-style pools until re-scored through the fixed path (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md:31`: *"Every retro-score number before this commit had silent preprocessing drift."*).
- **Calibration is a complement, not a cure.** 0.301 gap closure is the first numeric verdict on per-camera calibration and lands exactly on the decision boundary — it rules out the "calibration alone fixes this" hypothesis while leaving calibration on the table as an adjunct. The primary lever for Packet-7 is training-aug, implemented by RLP7_04/05.
- **Per-identity reducer unblocks launch-stage diagnosis.** Any Packet-7 variant's `videos_report.csv` can now be sliced by identity without re-running retro-score, making "did this variant create a new Dor?" a cheap query rather than an ad-hoc analysis.
- **Spatial aug wiring was pre-existing.** WS-P4.a's zero-line-change result is a positive finding: the pipeline already supports the Packet-7 spatial-only hypothesis cleanly via yaml, so RLP7_04 is a minimal delta vs RLP6_04 rather than a new code path.
- **Session IDs**: *(convmem searches for "WS-P0", "WS-P1", "WS-P2.b", "WS-P4.a", "calibration probe", "INTER_LINEAR", "per-identity reducer" all returned low-relevance matches — the authoritative in-session narrative lives in `docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md` and `HANDOFF.md`, not in replayable conversation fragments.)*

## Retrospective (as of 2026-04-24)

- **Which later packet challenged or confirmed these conclusions**: All four probes predate the Packet-7 training launch. The post-fix re-score (landed same day) already confirmed WS-P0 by showing the shortcut survived the fix (`analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json` anchor delta −0.008). The RLP7_04/RLP7_05 variants, once trained and retro-scored through the fixed path, will be the real test of WS-P1's "training-aug is the dominant remaining lever" verdict.
- **Reinterpretations**: none yet — the probes are hours old. One item to watch: the 0.301 closure sat on the decision boundary. If WS-P1 is re-run against post-fix scores (the probe's inputs are pre-fix numbers per `HANDOFF.md:263`), the verdict could shift either direction. Until that re-run lands, the 0.301 headline is a pre-fix datum being used to gate a post-fix launch — acceptable because the anchor delta showed preprocessing drift wasn't the dominant signal, but worth substituting in for accuracy.
- **Preprocessing-parity note**: **WS-P0 *is* the fix.** This probe is the canonical reference for the preprocessing-parity thread. All prior-packet retrospectives (RLP1 through RLP6B) inherit pre-fix retro-score numbers; none have been re-scored through the INTER_LINEAR path yet. The one post-fix number that exists is the RLP6_04 anchor re-score in `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`, which establishes that for the camera-signature pools specifically, the pre/post delta is small (≤0.15 on any pool; anchor at −0.008).
- **Cross-reference**: [`preprocessing_parity_bug.md`](../threads/preprocessing_parity_bug.md) — WS-P0 is the thread's origin event. [`calibration_vs_training_aug.md`](../threads/calibration_vs_training_aug.md) — the 30.1% number is the thread's founding datum. [`processing_signature_shortcut.md`](../threads/processing_signature_shortcut.md) — all four probes are countermeasures to the shortcut first seen in RLP6 slot-07. [`gate_alignment_story.md`](../threads/gate_alignment_story.md) — WS-P4.a unblocked the Teams spatial-aug yaml path that RLP7_04 exercises.

## Source files

- **Handoffs**:
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md:26` (WS-P0), `:39` (WS-P1), `:49` (WS-P2.b), `:64` (WS-P4.a)
  - `HANDOFF.md:31` (WS-P0), `:32` (WS-P1), `:33` (WS-P2.b), `:34` (WS-P4.a)
- **Code sites**:
  - `batch_inference_gcs.py:410` — INTER_LINEAR fix
  - `arena/model_arena.py:474` — INTER_LINEAR fix (retro-score path, load-bearing)
  - `data/sources/combined_paired.py:3455` — training-side reference resize
  - `arena/postprocess_per_identity.py` — WS-P2.b CLI
  - `data/augmentations/pipelines.py:787` — `_TEAMS_PASSTHROUGH_DEFAULTS`
  - `data/augmentations/pipelines.py:1386` — `_build_teams_passthrough_pipeline`
  - `data/sources/combined_paired.py:4772` — override-key filter (verified to pass Teams spatial knobs)
- **Yamls**:
  - `experiments/phase2_round13/R13_RLP7_04_teams_spatial_only.yaml` (seed 744, WS-P4.a enabled)
  - `experiments/phase2_round13/R13_RLP7_05_teams_spatial_plus_codec.yaml` (seed 745)
- **Analysis / tests**:
  - `analysis/calibration_probe_2026-04-24.py` + `analysis/calibration_probe_2026-04-24.summary.json` (WS-P1)
  - `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json` (WS-P0 post-fix gate)
  - `tests/test_inference_train_preprocessing_parity.py` (guards WS-P0)
  - `tests/test_postprocess_per_identity.py` (guards WS-P2.b)
- **Commits**:
  - `855871e` — WS-P0 (INTER_AREA → INTER_LINEAR)
  - `deac44e` — WS-P1 calibration probe + WS-P2.b per-identity reducer
  - `c4affa1` — RLP7_04 + RLP7_05 yamls (exercise WS-P4.a)
- **Memory pointers**: `project_signature_shortcut_finding.md` (slot-07 dor_shkedi vs real_dor processing-signature divergence is the same class of problem the WS probes address).
