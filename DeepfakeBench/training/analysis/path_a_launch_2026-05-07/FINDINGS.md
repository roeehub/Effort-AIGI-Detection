# Path A Results Processing — FINDINGS (2026-05-07)

> Mirrored to disk by parent agent because the sub-agent harness blocks
> direct `.md` writes. Sub-agent ran the runbook end-to-end and returned
> this content inline.

## 1. HEADLINE verdict

**Head-only retrain is NOT viable for P1.** Across all 3 ckpts (P8A, E2B,
CLIP_B16_raw), Head B (CE+pair-rank) shows **0.0pp to −0.4pp lift** vs Head
A (CE-only) on `p_fake_gt_real_on_pairs`, accuracy, and AUC — well below
the +3pp head-side-signal bar.

**Important structural caveat:** the head probe could only test on
`teams_passthrough` features (100% of the 25,327 covered pairs); 32% of
`pair_gaps.csv` frame paths are `gs://local/...` placeholders that the
extraction job could not pull, so viso/deeplive/df40 lanes were **not
testable in Phase 0h**.

## 2. Phase 0j — runbook six lanes

`P(pair_gap ≤ 0 | missed_fake)` per lane, miss_threshold=0.5. Fresh NPZ
scores used where available (teams lanes only); falls back to cached
`pair_gaps.csv` columns for non-fresh lanes.

| Lane | P8A `n_missed` | P8A p(gap≤0) | **P8A** | E2B `n_missed` | E2B p(gap≤0) | **E2B** | Source |
|---|---:|---:|---|---:|---:|---|---|
| df40 | – | – | **INSUFFICIENT_DATA** | – | – | **INSUFFICIENT_DATA** | not in csv |
| deeplive_v1 | 55 | 0.255 | **GREEN** | 0 | – | **INSUFFICIENT_DATA** | cached |
| deeplive_v2 | 13 | 0.154 | **AMBER** | 29 | 0.069 | **RED** | cached |
| viso_v1 | – | – | **INSUFFICIENT_DATA** | – | – | **INSUFFICIENT_DATA** | not in csv |
| viso_enhanced | 543 | 0.355 | **GREEN** | 2,290 | 0.040 | **RED** | cached |
| viso_teams_enhanced | – | – | **INSUFFICIENT_DATA** | – | – | **INSUFFICIENT_DATA** | not in csv |

- **P8A: 2 GREEN / 1 AMBER / 0 RED / 3 INSUFFICIENT** — meets the runbook
  ≥2-GREEN bar (deeplive_v1 + viso_enhanced).
- **E2B: 0 GREEN / 0 AMBER / 2 RED / 4 INSUFFICIENT** — DEMOTE_P1
  (consistent with prior audit; reinforces FT-base = P8A choice).

Informative additional lanes (not in runbook six):
- `teams_passthrough_dev`: P8A=RED 9.9%, E2B=GREEN 38.8% (fresh)
- `teams_passthrough_lockbox`: E2B=GREEN 98.5% (fresh)
- `extra`: P8A=AMBER 12.4%, E2B=RED 0.4%
- `live_fakes_teams_prod`: E2B=RED 1.2%

Fresh `teams_passthrough_dev` value (9.9%) reproduces the prior cached
audit's 9.86% — sanity check on the fresh-extraction pipeline.

## 3. Phase 0h — Head A vs B vs C, per ckpt

Test split: subject-out, 1 of 8 covered subjects held out. All 25,327
covered pairs are `teams_passthrough`.

| Ckpt | A AUC | A acc | A p(f>r) | B AUC | B acc | B p(f>r) | C AUC | **B−A p(f>r)** | **B−A acc** | **B−A AUC** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P8A | 0.9596 | 51.2% | 95.75% | 0.9585 | 51.2% | 95.63% | 0.9585 | **−0.13pp** | **+0.0pp** | **−0.11pp** |
| E2B | 0.6774 | 54.8% | 67.70% | 0.6749 | 53.9% | 67.33% | 0.6749 | **−0.38pp** | **−0.83pp** | **−0.25pp** |
| CLIP_B16_raw | 0.9555 | 88.5% | 95.68% | 0.9555 | 88.5% | 95.68% | 0.9555 | **0.00pp** | **0.00pp** | **0.00pp** |

**Multi-seed sensitivity (P8A only, 5 seeds, 50 epochs):**
- paired B−A AUC mean = **−0.028pp ± 0.039pp**
- paired C−A AUC mean = **−0.028pp ± 0.039pp**
- Lift is robustly zero or slightly negative.

**HP sensitivity (P8A only, 3 hp configs × 3 seeds):** {default (m=0.5,
λ=0.2), wider (m=1.0, λ=0.5), heavier (m=2.0, λ=1.0)} — all show lift
≈ 0pp. Heavier pair-rank weights do NOT extract a head-side signal.

**Important: Head C ≡ Head B numerically** because the covered pairs have
**only 1 unique method×transport group** (`teams_passthrough__teams`).
GroupDRO with one group equals ordinary mean. The full Phase 0h promise
(CE+pair-rank+GroupDRO across method-conditional groups) is **structurally
unanswerable from this extraction**.

Train-side sanity check (CE-only): P8A train AUC=0.997, E2B=0.879,
CLIP_B16_raw=1.0. Head probes converge well; the 0pp lift is not an
underfitting artifact.

## 4. Promotion gate verdict per ckpt

| Ckpt | Lift B−A pair_gap_lift | **Verdict** |
|---|---:|---|
| P8A | −0.13pp | **NO_HEAD_SIDE_SIGNAL** |
| E2B | −0.38pp | **NO_HEAD_SIDE_SIGNAL** |
| CLIP_B16_raw | 0.00pp | **BORDERLINE** (no signal in either direction) |

All 3 ckpts fail the ≥3pp head-side-signal threshold.

## 5. Recommended P1 path

The strict runbook decision matrix says: 0j GREEN on ≥2 lanes (P8A) AND
0h shows no head-side signal → **P1 full FT-from-P8A with pair-rank,
committed launch**. But this path commits ~$50–200 of GPU on a
head-vs-encoder hypothesis that was **not actually tested on the lanes
that matter** (viso_enhanced, deeplive_v1).

Three options:

- **Option 1 — P1 full FT-from-P8A with pair-rank (committed launch).**
  Justified on Phase 0j signal alone. Phase 0h was silent on the binding
  question. ~$50–200, 1 week.
- **Option 2 — P2 (PE_SBI) instead.** No-regret structural alternative.
  Same FT cost, different mechanism. Recommended if user wants to derisk
  Phase 0h's coverage gap without first re-extracting.
- **Option 3 — Re-extract paired features on a manifest filtered to
  non-`gs://local/` rows** so Phase 0h can actually test viso/deeplive/df40
  lanes; then re-run head probe before launching anything. ~$1, 30 min.
  If that probe returns ≥3pp lift, head-only retrain is viable
  (~$0–1 vs $50–200). If 0pp, P1 is dead and P2 wins.

Per the user's stated preference for explicit decision points
(`feedback_decision_points.md`), this should land back at the user.

## 6. Cost actual vs estimated

| Item | Estimated | Actual |
|---|---|---|
| Vertex extraction job runtime | ~40 min | **3.03 min** (16:47:25Z → 16:50:27Z) |
| Vertex extraction job cost (a2-highgpu-1g + 200GB pd-ssd) | $8–13 (A100 spot) | **$0.10–$0.23** (3 min) |
| CPU follow-up cost | ~$0 | **$0** (local CPU only) |
| **Total** | **$8–13** | **$0.10–$0.23** |

**95–98% under estimated budget.** The job ran much faster than expected
(3 min vs 40 min) — the per-frame forward pass is fast at batch=64 and
I/O-bound on failed-download attempts rather than GPU-bound on successful
frames.

## 7. Artifacts written

### Path A launch outputs (pulled from GCS)
- `analysis/path_a_launch_2026-05-07/outputs/p8a_paired_features.npz` (7.0 MB; 5,311 × 512-dim, ok rate 0.679)
- `analysis/path_a_launch_2026-05-07/outputs/e2b_paired_features.npz` (7.1 MB)
- `analysis/path_a_launch_2026-05-07/outputs/clip_b16_raw_paired_features.npz` (6.8 MB)
- `analysis/path_a_launch_2026-05-07/outputs/frame_manifest.csv` (901 KB)

### Phase 0j (Path-A wrapper)
- `analysis/same_source_pair_gap_audit_2026-05-06/run_probe_path_a.py` (10 KB; NEW wrapper — original `run_probe.py` did not accept `--tight-pairs` or `--pair_gaps_csv`)
- `analysis/same_source_pair_gap_audit_2026-05-06/outputs_path_a/summary.json` (8.4 KB)
- `analysis/same_source_pair_gap_audit_2026-05-06/outputs_path_a/per_lane_per_ckpt.csv` (2.0 KB)

### Phase 0h (Path-A wrapper)
- `analysis/frozen_pair_head_probe_2026-05-06/run_probe_path_a.py` (11 KB; NEW wrapper that filters NPZ by `ok==1` before training; existing `run_probe.py` would silently use zero-feature rows for failed downloads)
- `analysis/frozen_pair_head_probe_2026-05-06/multi_seed_sweep.py` (4 KB; sensitivity sweep)
- `analysis/frozen_pair_head_probe_2026-05-06/outputs_p8a/{summary.json, head_metrics.csv, run.log}`
- `analysis/frozen_pair_head_probe_2026-05-06/outputs_e2b/{summary.json, head_metrics.csv, run.log}`
- `analysis/frozen_pair_head_probe_2026-05-06/outputs_clip_b16_raw/{summary.json, head_metrics.csv, run.log}`
- `analysis/frozen_pair_head_probe_2026-05-06/p8a_hp_sweep.csv` (1.7 KB; margin/λ_pair sensitivity, 3 hp × 3 seeds)
- `analysis/frozen_pair_head_probe_2026-05-06/multi_seed_metrics.csv` (P8A complete with 15 rows)

## 8. Deviations from runbook

The runbook explicitly noted that some scripts may need wrappers. Two
were written:

1. **Phase 0j wrapper.** Existing `run_probe.py` argparse signature was
   empty — script was a cache-first auditor, not a refresh-from-NPZ runner.
   `run_probe_path_a.py` joins NPZ scores to `pair_gaps.csv` via
   `frame_path`, falls back to cached pair_gaps columns where fresh scores
   aren't available (`gs://local` rows), and emits per-lane verdicts on the
   runbook 6-lane partition + 4 informative lanes.
2. **Phase 0h wrapper.** Existing `run_probe.py` would silently use
   zero-feature rows for failed downloads (the 32% `ok==0` rows have
   features=zeros). `run_probe_path_a.py` filters `ok==1` before training;
   identical training architecture otherwise (linear head, CE / CE+pair-rank
   / CE+pair-rank+GroupDRO, margin=0.5, λ_pair=0.2, 100 epochs Adam lr=1e-3).

### Schema gotchas surfaced (parent agent should update §8.2 NEXT_STEPS_PLAN)

- **NPZ ok rate is 0.679, not the runbook's "expected >0.99".** 1,707 of
  5,311 frames failed because `gs://local/...` paths are placeholders for
  the local-only training pool. Every viso/deeplive/extra/live_prod fake or
  real has at least one `gs://local/...` path in pair_gaps.csv. The
  runbook's predicted ok rate >0.99 is structurally incorrect.
- **`pair_id` field is a semicolon-joined list of pair_ids** that include
  this frame, not a single integer. Use `frame_path` for joining.
- **All 25,327 covered pairs are teams_passthrough.** No viso, no deeplive,
  no extra in the head-probe substrate. Phase 0h is structurally restricted
  to the teams_passthrough question; cannot answer head-vs-encoder split on
  viso/deeplive lanes.
- **GroupDRO is unanswerable from this data:** only 1 unique method×transport
  group exists in the covered pairs. Head C ≡ Head B numerically.
- **Vertex job ran ~13× faster than estimated** (3 min vs 40 min). Cost was
  ~50× lower than estimated.

### Side issue worth flagging

`docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md` §8.2 should be
updated to reflect:
- Phase 0j: P8A meets ≥2-GREEN bar on the runbook six (deeplive_v1,
  viso_enhanced) but Phase 0h could not actually localize the signal.
- The Path-A extraction's `gs://local/...` coverage gap is a structural
  blocker for Phase 0h; any future paired-feature extraction needs to
  pre-filter the manifest to non-`gs://local` rows OR the parent loaders
  need to resolve the placeholders first.
- The parent agent should consider whether to authorize an Option-3
  re-extraction ($1, 30 min) to actually answer the head-vs-encoder
  question on the load-bearing lanes before committing to P1
  (~$50–200).
