# Evidence log — P8A_TRAINING_DATE_CHECK_2026-05-06

Append-only. Searches and findings in temporal order of investigation.

## Search 1 — locate run_id producing P8A_REFERENCE_STEP5000

Command:
```
grep -rni "P8A_REFERENCE_STEP5000" arena/checkpoint_maps/
```

Result: 23+ checkpoint-map yamls all point P8A_REFERENCE_STEP5000 to:
```
gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth
```

Run_id: **`9lmvb5b4`** (W&B), GCS folder
`gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/`. Checkpoint
filename embeds **`20260424`** as the training date and `step5000` as
the early-stop step.

## Search 2 — P8A retro and packet documentation

File: `docs/packet_retrospectives/packets/P8A.md`

Key quotes (line numbers refer to the file as found 2026-05-06):

- Line 14: "Leader slot | `R13_RLP8_01` (P8A, run `9lmvb5b4`,
  value_composite step 5000)".
- Line 24: "**P8A** (`R13_RLP8_01_unfreeze_clip_codec.yaml`) — single-variable
  delta from RLP7_02. ... Trained 2026-04-24 21:41 UTC → 23:48 UTC,
  ~2h7min on us-east1, image 1.3.206 (commit `d5be7ce`)."
- Line 29: "**Retroactive correction (2026-04-26):** ... Every P8A run
  had `apply_svd_to_in_proj: true` but the q/k/v residuals received
  zero classification gradient until the 2026-04-26 fix (commit
  `2feea58`)."
- Line 108: "**Retroactive in_proj-SVD correction (2026-04-26, commit
  `2feea58`).** ... the q/k/v residuals received zero classification
  gradient through the entire run. The lever set was MLP-SVD +
  visual.proj + ln_post unfreeze; the in_proj-SVD piece contributed
  only via regularizer drift."

## Search 3 — overnight summary

File: `analysis/overnight_packet7_packet8_summary_2026-04-25.md`

- Line 67: "Vertex job: `1205078281779412992` (us-east1, started
  2026-04-24 21:41 UTC, ended 23:48 UTC, ~2h7min training)."
- Line 68: "W&B run: `9lmvb5b4` (smooth-haze-250)."
- Line 69: "Checkpoint scored:
  `gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth`
  (early-stopped at step 5000, same pattern as P7 runs)."

This is the cleanest single source for P8A start/end UTC timestamps
and the Vertex job ID. Confirms run timing independent of the wiki
retro.

## Search 4 — master plan log explicit pre-fix labelling

File: `april-26-training-master-plan-v2.LOG.md`

Line 214 (FT-chain inventory):
> "Verified the actual FT chain: plain CLIP → R12g (`0xxqwhxg/step14000`,
> ~14k broken-bug steps) → RLP6_04 (`h2pdu6i5/step23500`, ~23k
> broken-bug steps) → P8A (`9lmvb5b4/step5000`, ~5k broken-bug steps)
> → P10_SYM_on_P8A (`sbnzyrzq/step2500`). Total ~45k upstream
> broken-in_proj-SVD steps before C.1's 10k of fixed-code FT."

This explicitly classifies P8A's 5000 steps as "broken-bug steps" —
i.e., pre-fix.

## Search 5 — fix commit metadata

Command:
```
git show 2feea58 --format='%H %aI %ai %s' --no-patch
```

Result:
```
2feea581e9da6b925aa6a683ad6f373c14c4ad81  2026-04-26T21:13:21+02:00  2026-04-26 21:13:21 +0200  Fix silent zero-gradient bug in apply_svd_to_in_proj path
```

Author commit time in UTC: **2026-04-26T19:13:21Z**.

## Search 6 — memory cross-checks

File: `/Users/roeedar/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/project_p8a_breakthrough.md`

- "P8A (run `9lmvb5b4`, 2026-04-24/25)" — date-window confirms.
- CORRECTION block: "the in_proj-SVD piece was a no-op."

File: `/Users/roeedar/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/project_in_proj_svd_gradient_bug.md`

- "The `apply_svd_to_in_proj` path in `detectors/effort_detector.py`
  was silently broken from whenever it landed until the 2026-04-26 fix."
- "**Affected scope:** every checkpoint that set
  `backbone.apply_svd_to_in_proj: true` during training. That includes
  the entire R12/RLP/P-* lineage."

## Date arithmetic

- P8A end: 2026-04-24 23:48 UTC.
- Fix landed: 2026-04-26 19:13 UTC.
- Gap: **43h 25m** between P8A's last training step and the fix
  commit. P8A is unambiguously pre-fix.

## What I could NOT determine (and why it doesn't matter)

- I did not query Vertex (`gcloud ai custom-jobs describe
  1205078281779412992`) or W&B (`wandb.api.run("9lmvb5b4")`) for
  exact-second precision on start/end. The retros' minute-level
  precision is more than sufficient given a 43-hour gap.
- I did not check whether the local `wandb` directory contains a
  cached run sync log; it would only refine known-good information.
- I did not separately confirm whether commit `2feea58` was actually
  baked into a Docker image before any P8A-derived FT was launched.
  That question is irrelevant to the verdict (P8A finished before the
  fix; whether downstream FT chains used the fix is a separate
  per-packet question already addressed in
  `april-26-training-master-plan-v2.LOG.md` line 214).

## Verdict

**Pre-fix. Confidence: HIGH.**
