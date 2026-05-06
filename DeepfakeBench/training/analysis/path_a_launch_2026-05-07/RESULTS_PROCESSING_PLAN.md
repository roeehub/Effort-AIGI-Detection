# PATH_A_LAUNCH_2026-05-07 — Results-processing plan

**Submitted:** 2026-05-06T16:46:26Z, us-east1, job ID `3077166152858730496`
**Expected duration:** ~40 min on A100 (3 forward passes × 5,311 frames)
**Expected outputs:** `gs://training-job-outputs/analysis_outputs/frozen_pair_features_2026-05-07/`

This document is the runbook for the next agent (or the user) once the Vertex
job completes. It assumes you have read `launch_log.json` for job state and
output paths.

---

## Step 1 — Confirm job succeeded

```bash
gcloud ai custom-jobs describe 3077166152858730496 \
    --region=us-east1 --project=train-cvit2 \
    --format='value(state,startTime,endTime)'
```

Expected: `JOB_STATE_SUCCEEDED`. If `JOB_STATE_FAILED`, stream logs:

```bash
gcloud ai custom-jobs stream-logs 3077166152858730496 \
    --region=us-east1 --project=train-cvit2 2>&1 | tail -200
```

Common failure modes to watch for:
- `gsutil` not on PATH inside container — fall back to `python -c "from google.cloud import storage; ..."`.
- OOM on A100 with `batch_size=64` — drop to 32 and re-submit.
- `open_clip` missing — image 1.3.267 should have it, but if not, fall back to extracting just P8A + E2B (skip `--include_clip_raw`).

---

## Step 2 — Verify all 4 NPZ artifacts uploaded

```bash
gsutil ls -l gs://training-job-outputs/analysis_outputs/frozen_pair_features_2026-05-07/
```

Expected:
- `p8a_paired_features.npz` (~10–25 MB compressed)
- `e2b_paired_features.npz` (~10–25 MB compressed)
- `clip_b16_raw_paired_features.npz` (~10–25 MB compressed)
- `frame_manifest.csv` (~150 KB; one row per unique URI with label + pair_ids)

If any are missing, the job partially failed; check stream-logs for which extractor errored.

---

## Step 3 — Pull NPZ files locally

```bash
mkdir -p analysis/path_a_launch_2026-05-07/outputs
gsutil -m cp \
  gs://training-job-outputs/analysis_outputs/frozen_pair_features_2026-05-07/*.npz \
  gs://training-job-outputs/analysis_outputs/frozen_pair_features_2026-05-07/frame_manifest.csv \
  analysis/path_a_launch_2026-05-07/outputs/
```

Quick sanity check (per NPZ):

```python
import numpy as np
d = np.load("analysis/path_a_launch_2026-05-07/outputs/p8a_paired_features.npz", allow_pickle=True)
print({k: d[k].shape for k in d.keys()})
# expected:
#   features:   (5311, 512)
#   scores:     (5311,)
#   frame_path: (5311,)
#   label:      (5311,)
#   pair_id:    (5311,)
#   ok:         (5311,)
#   meta_json:  ()
print("ok rate:", float(d["ok"].mean()))   # should be > 0.99
print("real n:", int((d["label"] == 0).sum()))
print("fake n:", int((d["label"] == 1).sum()))
```

If `ok rate < 0.99`, some frames failed to download from GCS — investigate which URIs are dead or behind permission walls.

---

## Step 4 — Phase 0j: re-run same-source pair-gap audit

This converts 2/6 → 4/6 lane coverage with a definitive verdict on whether
pair-rank loss has signal on df40 + deeplive lanes (currently unmeasured).

The existing audit at `analysis/pair_gap_audit_2026-05-06/` was on cached
inference scores; this re-run uses fresh scores from these NPZ files at the
exact paired-frame indices.

```bash
python3 analysis/same_source_pair_gap_audit_2026-05-06/run_probe.py \
    --tight-pairs analysis/path_a_launch_2026-05-07/outputs/p8a_paired_features.npz \
                  analysis/path_a_launch_2026-05-07/outputs/e2b_paired_features.npz \
    --pair_gaps_csv analysis/pair_gap_audit_2026-05-06/outputs/pair_gaps.csv \
    --output_dir analysis/same_source_pair_gap_audit_2026-05-06/outputs_path_a/
```

> Verify the exact CLI of `same_source_pair_gap_audit_2026-05-06/run_probe.py`
> before invoking — this packet didn't run the script end-to-end. If the
> argparse signature doesn't accept `--tight-pairs` directly, build a thin
> wrapper that joins the NPZ frame_paths to `pair_gaps.csv` rows.

**Expected verdict:**
- If `P(pair_gap <= 0 | missed_fake)` > 25% on **2 of 6** lanes (df40, deeplive_v1, deeplive_v2, viso_v1, viso_enhanced, viso_teams_enhanced) → pair-rank lever GREEN; P1 launch justified.
- If <10% on all lanes → pair-rank lever RED; demote P1 in favour of P2 (PE_SBI).
- 10–25% → AMBER; P1 viable iff combined with margin tuning + GroupDRO weighting.

---

## Step 5 — Phase 0h: head-vs-encoder localisation probe

This is the load-bearing decision: does the pair-rank signal live in the head
(cheap retrain) or in the encoder (expensive full FT)?

```bash
python3 analysis/frozen_pair_head_probe_2026-05-06/run_probe.py \
    --features analysis/path_a_launch_2026-05-07/outputs/p8a_paired_features.npz \
    --pair_gaps_csv analysis/pair_gap_audit_2026-05-06/outputs/pair_gaps.csv \
    --output_dir analysis/frozen_pair_head_probe_2026-05-06/outputs_p8a/
```

Repeat for `e2b_paired_features.npz` and `clip_b16_raw_paired_features.npz`
into separate output dirs. The probe trains:

- **Head A:** linear classifier on frozen features with **CE only**.
- **Head B:** linear classifier with **CE + pair-rank loss** (margin 0.5, λ_pair 0.2).
- **Head C (optional):** Head B + group-id reweighting.

Each head trains in 30–90 sec on CPU (~5,311 pairs, 512-dim features).

**Promotion gate:**

| Head B − Head A pair_gap_lift | Verdict |
|---|---|
| ≥ 3pp | **Head-side signal.** Demote P1 in favour of head-only retrain (~$0–1 of GPU vs ~$50–200 for full FT). |
| 0–3pp | **Borderline.** P1 viable but expected ROI lower than headline; weigh against P2 cost. |
| < 0pp | **No head-side signal.** P1 only justified if encoder change is strictly necessary (lane-restricted launch acceptable per Path B in NEXT_STEPS_PLAN §8.2). |

---

## Step 6 — Decide on P1 launch path

Combine Step 4 + Step 5 verdicts:

| 0j (pair-rank lever) | 0h (head-vs-encoder) | Recommendation |
|---|---|---|
| GREEN on ≥2 lanes | Head-side ≥ 3pp | **Head-only retrain** — fastest win, ~$0–1, 1 day. |
| GREEN on ≥2 lanes | Encoder-side or borderline | **P1 full FT-from-P8A** with pair-rank — committed launch, ~$50–200, 1 week. |
| AMBER on 1–2 lanes | Head-side ≥ 3pp | Head-only retrain on the AMBER lanes only; treat as scoped patch. |
| AMBER everywhere | Anything | **P2 (PE_SBI) instead** — no-regret structural alternative. |
| RED on all lanes | Anything | **P2 (PE_SBI) instead.** P1 is dead. |

Update `docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md` §8.2 with the
selected path and authorise the relevant follow-up packet.

---

## Step 7 — Memory + status updates

When the verdict is in, write a memory file at
`/Users/roeedar/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/`
with the title `project_path_a_verdict_2026-05-07.md` covering:

- 0h pair-rank lift in pp (P8A, E2B, CLIP_B16_raw — all 3)
- 0j per-lane pair_gap stats on df40 + deeplive (which lanes flipped GREEN/AMBER/RED)
- The chosen path (head-only retrain / full P1 / P2 pivot)
- Cost-actual vs estimated ($8–13 budget)

Then bump VERSION (if any code-side changes were made), commit, and push.

---

## Cleanup

After the verdict is captured in memory + handoff, the GCS inputs at
`gs://training-job-outputs/path_a_inputs_2026-05-07/` can stay (cheap, small).
The `frozen_pair_features_2026-05-07/` outputs should ALSO stay — they're the
canonical paired-feature cache and any future head-localisation probe reuses them.
