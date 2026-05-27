# P1 BUNDLE_step500 — Joint tau-sweep, F1+F4+F5 simultaneous-overlap probe

**Date**: 2026-05-07
**Ckpt**: `p1_bundle_periodic_step500` (the F1-reaching ckpt — 82.6% lockbox recall at calibrated tau, best in P1).
**Type**: FACTS doc (compute + direct observations only; no scope-extending opinion).

---

## Question

Is there a single global tau in [0.95, 0.999] at which BUNDLE_step500 simultaneously satisfies all of:

- **F1**: `lockbox_fake_recall >= 0.90` (where lockbox = `teams_fake_all_lockbox`, n=425)
- **F4**: `max(HDTF real FPR over 4 suites) <= 0.05`
- **F5**: `max(per-chronic-identity FPR) <= 0.10` over the 6 chronic identities

Or are these criteria mutually exclusive at every grid point?

---

## Method

- 100 linearly spaced tau values in [0.95, 0.999].
- Per-frame `frame_prob` arrays loaded from existing Phase A / Phase C reports for `p1_bundle_periodic_step500`:
  - `teams_real_all_lockbox` (n=1418), `teams_fake_all_lockbox` (n=425) — for lockbox metrics.
  - `proper_real_clean_dev` (n=11544), `proper_real_clean_lockbox` (n=3056), `proper_real_teams_dev` (n=11552), `proper_real_teams_lockbox` (n=3056) — for F4.
  - `teams_real_all_dev` (n=4564) — for chronic-id breakdown.
- Chronic identities: `PC_Generator__s22`, `PC_Generator__s45`, `Q__s6`, `bla_bla_chow`, `bla_bla_chow__s2`, `roy_d`. Match by case-insensitive prefix on raw `video_id` (same rule as bug-fixed `phase_d/run_chronic_filter.py`; do NOT regex-strip session tokens).
- Chronic per-identity row counts (matched against `teams_real_all_dev`):
  - `PC_Generator__s22`: 227, `PC_Generator__s45`: 91, `Q__s6`: 54, `bla_bla_chow`: 491, `bla_bla_chow__s2`: 180, `roy_d`: 130.
- At each tau, FPR = (count where `frame_prob >= tau`) / n; recall same formula on the fake suite.
- Output: per-criterion qualifying tau bands, intersection, plus 100-row CSV.

Script: `run_joint_tau_sweep.py`. CSV: `tau_sweep_bundle_step500.csv`.

---

## Verdict

**Overlap region: empty.** No tau in [0.95, 0.999] satisfies F1 + F4 + F5 simultaneously.

The criteria are mutually exclusive on this ckpt because F5 (worst chronic identity ≤ 10%) and F1 (lockbox fake recall ≥ 90%) are pinned to disjoint score regions, with `roy_d` as the binding constraint that sits inside the lockbox-fake score distribution.

---

## Per-criterion qualifying tau bands

| criterion | tau-band where pass | n grid points (of 100) |
|---|---|---:|
| F1 (lockbox fake recall ≥ 0.90) | [0.9500, 0.9911] | 84 |
| F4 (max HDTF FPR ≤ 0.05) | [0.9871, 0.9990] | 25 |
| F5 (max chronic FPR ≤ 0.10) | [0.9931, 0.9990] | 13 |
| F1 ∩ F4 ∩ F5 | empty | **0** |

- F1 and F5 are disjoint: F1 ends at tau ≈ 0.991 while F5 starts at tau ≈ 0.993.
- F1 ∩ F4 = [0.9871, 0.9911] (n=9 grid points), F4 ∩ F5 = [0.9931, 0.9990] (n=13).
- F1 ∩ F4 cells: at all 9 such taus, `max_chronic_fpr ≥ 0.977` — chronic worst is essentially saturated.
- F4 ∩ F5 cells: at all 13 such taus, `lockbox_fake_recall ≤ 0.007` — recall is essentially zero.

---

## Direct observations: the score-distribution structure

### Sharpness of the F1 → F5 transition

| tau | lockbox_real_fpr | lockbox_fake_recall | max_hdtf_fpr | max_chronic_fpr | F1 | F4 | F5 |
|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 0.9871 | 0.110 | 0.979 | 0.0481 | 1.000 | T | T | F |
| 0.9891 | 0.094 | 0.979 | 0.0419 | 1.000 | T | T | F |
| 0.9901 | 0.078 | 0.955 | 0.0350 | 0.985 | T | T | F |
| 0.9911 | 0.046 | 0.908 | 0.0219 | 0.977 | T | T | F |
| 0.9916 | 0.040 | 0.885 | 0.0180 | 0.962 | F | T | F |
| 0.9921 | 0.032 | 0.776 | 0.0121 | 0.900 | F | T | F |
| 0.9926 | 0.004 | 0.414 | 0.0046 | 0.438 | F | T | F |
| 0.9931 | 0.000 | 0.007 | 0.000  | 0.015 | F | T | T |
| 0.9936 | 0.000 | 0.000 | 0.000  | 0.000 | F | T | T |

In a window of width ~0.001 (tau 0.9921 → 0.9931, ~10 grid points), `lockbox_fake_recall` collapses 78% → 0.7%, `max_chronic_fpr` collapses 90% → 1.5%, and `lockbox_real_fpr` collapses 3.2% → 0%. The model's score distribution is highly compressed near 0.99x — almost every score lands within the same narrow band.

### `roy_d` is the binding F5 constraint

Per-chronic-identity FPR at selected tau values:

| tau | PC_Generator__s22 | PC_Generator__s45 | Q__s6 | bla_bla_chow | bla_bla_chow__s2 | **roy_d** |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9500 | 0.718 | 0.264 | 0.685 | 0.678 | 0.628 | **1.000** |
| 0.9802 | 0.370 | 0.055 | 0.389 | 0.540 | 0.522 | **1.000** |
| 0.9851 | 0.172 | 0.011 | 0.333 | 0.454 | 0.472 | **1.000** |
| 0.9901 | 0.009 | 0.000 | 0.185 | 0.301 | 0.406 | **0.985** |
| 0.9921 | 0.000 | 0.000 | 0.148 | 0.136 | 0.278 | **0.900** |
| 0.9926 | 0.000 | 0.000 | 0.000 | 0.057 | 0.139 | **0.438** |
| 0.9931 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.015** |
| 0.9936 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Across the entire usable F1 band (tau ≤ 0.9911), `roy_d` FPR is ≥ 0.977. PC_Generator (s22+s45) hits ≤10% at tau ≈ 0.99, Q at ≈ 0.992, bla_bla_chow at ≈ 0.9925, but `roy_d` only relaxes below 10% at tau ≈ 0.9931 — exactly where lockbox fake recall has already collapsed to <1%.

**`roy_d` and the lockbox fakes share the same high-score region (0.99–0.993).** The model puts both classes there. Any tau low enough to keep ≥90% of lockbox fakes flagged also flags ≥90% of `roy_d` real frames.

This is consistent with the note in `RESULTS_F1_F5_FACTS_2026-05-07.md` §F5 that `roy_d` regressed from 29% under P8A to 78–93% across all P1 ckpts ("CATASTROPHICALLY WORSE for P1"). The joint sweep makes the structural cost concrete: `roy_d` lives where the lockbox fakes live.

### F4 binding suite

Per-HDTF-suite FPR confirms `proper_real_clean_lockbox` is the worst (matches the F4 calibrated-tau scorecard reading). F4 first passes at tau ≈ 0.9871; F4 is comfortably loose relative to F1 and F5.

| tau | clean_dev | clean_lockbox | teams_dev | teams_lockbox |
|---:|---:|---:|---:|---:|
| 0.9802 | 0.059 | 0.071 | 0.017 | 0.021 |
| 0.9871 | 0.038 | 0.048 | 0.009 | 0.015 |
| 0.9901 | 0.020 | 0.029 | 0.004 | 0.010 |

---

## Closest-to-feasibility tau

Defining `min_slack(tau) = min(F1_slack, F4_slack, F5_slack)` where each slack is positive when the respective criterion passes — the most "balanced" tau on the grid is `tau=0.9926`:

| tau | recall | max_hdtf | max_chronic | F1 slack | F4 slack | F5 slack |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9926 | 0.414 | 0.005 | 0.438 | −0.486 | +0.045 | −0.338 |

At this point F4 has +4.5pp headroom but F1 misses by 49pp and F5 misses by 34pp. There is no tau on the grid where F1 misses by less than ~1pp; the shape of the problem isn't "marginal joint failure", it's "joint failure by tens of pp on at least one criterion at every point".

---

## Files

- `tau_sweep_bundle_step500.csv` — 100-row sweep with per-suite and per-chronic-identity FPR columns.
- `tau_sweep_summary.csv` — per-criterion band + overlap summary.
- `run_joint_tau_sweep.py` — the script.

---

## Cross-reference

- `RESULTS_F1_F5_FACTS_2026-05-07.md` — F1 / F4 / F5 verdicts at each ckpt's individual calibrated tau (this doc covers the BUNDLE_step500 single-tau joint-overlap question).
- `phase_d/run_chronic_filter.py` — bug-fixed chronic prefix-match rule (used here).
- Memory `project_chronic_offenders_partition_per_ckpt_2026-05-04.md` — chronic-6 list.
