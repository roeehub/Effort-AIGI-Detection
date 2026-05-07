# F2 Pair-Rank Close Criterion — P1 vs P8A

## Method summary

For each Phase-A frame report, scores are joined to the cross-product
pair manifest at `analysis/pair_gap_audit_2026-05-06/outputs/pair_gaps.csv`.
Pairs are filtered to "previously-missed fakes" (P8A `frame_prob < 0.5`
on the fake side) and the metric `frac_fake_gt_real` is the fraction of
pairs in each lane where the new ckpt's `fake_score > real_score`.
Comparison is at frame-level (no video-level mean aggregation; each
unique fake_path averages 1.26 frames per video so frame ≈ video).

Lift vs P8A is reported as both absolute (`frac_p1 - frac_p8a`) and
relative (`(frac_p1 - frac_p8a) / frac_p8a * 100`). Pre-registered close
criterion (per `R13_P1_BUNDLE_FT_FROM_P8A.yaml` and
`NEXT_STEPS_PLAN_2026-05-06.md` §8.2): lift `>=` 30% relative on `>=` 2
of N paired lanes.

## Lane definitions

The yaml specifies 6 paired training lanes
(`pair_coverage_audit_2026-05-06`):

| Index | Training lane                 | Eval-substrate proxy in this F2 audit |
|------:|-------------------------------|---------------------------------------|
| 1     | df40                          | none — training-only lane |
| 2     | deeplive                      | none in Phase A (proxy needs `gs://local/...` paths) |
| 3     | visomaster_v1_base            | none in Phase A |
| 4     | visomaster_enhanced           | none in Phase A (proxy needs `gs://visomaster-enhanced-face-cropped-v2/...` paths) |
| 5     | visomaster_teams_enhanced     | none in Phase A |
| 6     | deeplive_teams (teams passthrough) | 8 canonical-subject sub-lanes via teams_real_all × teams_fake_all cross-product |

The `pair_gap_audit_2026-05-06` cross-product manifest only covers the
`deeplive_teams`-mapped substrate within Phase A (see Caveats §). The
8 canonical-subject sub-lanes within the eval-substrate teams_passthrough
cross-product are treated as the lane axis for this audit.

| Lane (canonical_subject) | Pairing convention | Total cross-product pairs | n_pairs after "previously-missed-fake" filter |
|---|---|---:|---:|
| cam_test__s32 | frame-level cross-product within subject | 4000 | 0 |
| cam_test__s38 | frame-level cross-product within subject | 4000 | 0 |
| dor_shkedi__s16 | frame-level cross-product within subject | 2418 | 31 |
| pc_generator__s15 | frame-level cross-product within subject | 2639 | 0 |
| pc_generator__s4 | frame-level cross-product within subject | 270 | 0 |
| test_cam__s73 | frame-level cross-product within subject | 4000 | 0 |
| test_cam__s76 | frame-level cross-product within subject | 4000 | 27 |
| xiang_xiang2_feng | frame-level cross-product within subject | 4000 | 2228 |

5 of 8 sub-lanes have `n_pairs=0` after filtering — P8A's fake scores on
those subjects are entirely `>=0.5` (no "previously-missed" fakes to lift).
The metric is undefined on those 5 sub-lanes.

## P8A baseline (reference)

| Lane | n_pairs | n_fake_gt_real | frac_fake_gt_real |
|---|---:|---:|---:|
| dor_shkedi__s16 | 31 | 31 | 1.000000 |
| test_cam__s76 | 27 | 27 | 1.000000 |
| xiang_xiang2_feng | 2228 | 2164 | 0.971275 |

## P1 ckpt × lane lift table

n_pairs is identical to P8A baseline within each lane (same filtered pair
set). frac_fake_gt_real is the per-ckpt rate; lift_abs and lift_rel_pct
are vs P8A baseline.

### dor_shkedi__s16 (n_pairs=31, P8A frac=1.000000)

| Ckpt | n_fake_gt_real | frac_fake_gt_real | lift_abs | lift_rel_pct | passes 30% bar |
|---|---:|---:|---:|---:|:---:|
| p1_bundle_periodic_step500    | 31 | 1.000000 | 0.000000 | 0.000000 | no |
| p1_bundle_top_n_step3750      | 30 | 0.967742 | -0.032258 | -3.225806 | no |
| p1_bundle_top_n_step4000      | 30 | 0.967742 | -0.032258 | -3.225806 | no |
| p1_pairrank_periodic_step500  | 31 | 1.000000 | 0.000000 | 0.000000 | no |
| p1_pairrank_top_n_step6000    | 30 | 0.967742 | -0.032258 | -3.225806 | no |
| p1_pairrank_top_n_step6750    | 31 | 1.000000 | 0.000000 | 0.000000 | no |

### test_cam__s76 (n_pairs=27, P8A frac=1.000000)

| Ckpt | n_fake_gt_real | frac_fake_gt_real | lift_abs | lift_rel_pct | passes 30% bar |
|---|---:|---:|---:|---:|:---:|
| p1_bundle_periodic_step500    | 27 | 1.000000 | 0.000000 | 0.000000 | no |
| p1_bundle_top_n_step3750      | 27 | 1.000000 | 0.000000 | 0.000000 | no |
| p1_bundle_top_n_step4000      | 27 | 1.000000 | 0.000000 | 0.000000 | no |
| p1_pairrank_periodic_step500  | 27 | 1.000000 | 0.000000 | 0.000000 | no |
| p1_pairrank_top_n_step6000    | 27 | 1.000000 | 0.000000 | 0.000000 | no |
| p1_pairrank_top_n_step6750    | 27 | 1.000000 | 0.000000 | 0.000000 | no |

### xiang_xiang2_feng (n_pairs=2228, P8A frac=0.971275)

| Ckpt | n_fake_gt_real | frac_fake_gt_real | lift_abs | lift_rel_pct | passes 30% bar |
|---|---:|---:|---:|---:|:---:|
| p1_bundle_periodic_step500    | 2206 | 0.990126 | 0.018851 | 1.940850 | no |
| p1_bundle_top_n_step3750      | 2178 | 0.977558 | 0.006284 | 0.646950 | no |
| p1_bundle_top_n_step4000      | 2180 | 0.978456 | 0.007181 | 0.739372 | no |
| p1_pairrank_periodic_step500  | 2170 | 0.973968 | 0.002693 | 0.277264 | no |
| p1_pairrank_top_n_step6000    | 2207 | 0.990575 | 0.019300 | 1.987061 | no |
| p1_pairrank_top_n_step6750    | 2206 | 0.990126 | 0.018851 | 1.940850 | no |

## Pass count per ckpt at 30% relative-lift bar

| Ckpt | n lanes passing 30% bar | n lanes with n_pairs > 0 |
|---|---:|---:|
| p1_bundle_periodic_step500    | 0 | 3 |
| p1_bundle_top_n_step3750      | 0 | 3 |
| p1_bundle_top_n_step4000      | 0 | 3 |
| p1_pairrank_periodic_step500  | 0 | 3 |
| p1_pairrank_top_n_step6000    | 0 | 3 |
| p1_pairrank_top_n_step6750    | 0 | 3 |

Close criterion bar: `>= 30% relative lift on >= 2 of 6 paired lanes`
(R13_P1 yaml).

Result against criterion: 0 of 6 P1 ckpts pass on `>= 2 of 6` lanes
(0 of 6 P1 ckpts pass on `>= 1 of 8` eval-substrate sub-lanes).

## Caveats

1. **Eval-substrate sub-lanes != 6 paired training lanes.** The 6 paired
   training lanes (`df40`, `deeplive`, `visomaster_v1_base`,
   `visomaster_enhanced`, `visomaster_teams_enhanced`, `deeplive_teams`)
   are training-loader constructs. Phase A reports score frames on eval
   substrates, not on training-loader output. Of the 6 paired lanes,
   only `deeplive_teams` (teams_passthrough transport) is represented in
   Phase A, and within that lane the audit subdivides by canonical_subject
   into 8 eval-substrate sub-lanes.

2. **Pair manifest is cross-product within canonical_subject, not
   `(sample_id, frame_idx)` tight pairs.** The training-loader pair-rank
   loss fires on tight `(sample_id, frame_idx)` pairs (per
   `analysis/pair_coverage_audit_2026-05-06/pairing_semantics_notes.md`),
   not on cross-product-within-subject pairs. The eval-substrate paths
   used here have no inherent `(sample_id, frame_idx)` link: the
   `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/` real and
   fake frames are drawn from different sessions (per
   `pair_gap_audit_2026-05-06/outputs/FINDINGS.md` Caveats §). The
   cross-product proxy massively over-counts the number of comparisons
   relative to what the loss sees during training (per the same
   FINDINGS document).

3. **3 of 8 sub-lanes have n_pairs=0 after the "previously-missed-fake"
   filter.** P8A's score is `>=0.5` on every fake_path in
   `cam_test__s32`, `cam_test__s38`, `pc_generator__s4`,
   `pc_generator__s15`, `test_cam__s73`. The fraction-saturated baseline
   leaves no "missed" frames for P1 to lift on these sub-lanes; the lift
   metric is undefined there (n_pairs=0).

4. **Of the 3 sub-lanes with data, P8A baseline is at 1.000 / 1.000 /
   0.971** (dor_shkedi__s16 / test_cam__s76 / xiang_xiang2_feng).
   At baseline frac=1.000 there is no headroom for positive lift; lift
   can only be `<=0`. At baseline frac=0.971 the maximum possible lift is
   `(1.000 - 0.971) / 0.971 * 100 = 2.99%`, below the 30% relative bar.
   The 30% relative-lift bar is structurally unreachable on these 3
   sub-lanes given P8A's near-saturated baseline performance.

5. **Pair-coverage proxy gap.** Per
   `same_source_pair_gap_audit_2026-05-06/outputs/FINDINGS.md`, only 2
   of the 6 paired training lanes (`visomaster_enhanced` via dor_local
   `visomaster_v2_dor`; `deeplive` via dor_local `dor_fake_local`) have
   any local cross-product proxy at all, and those use `gs://local/...`
   and `gs://visomaster-enhanced-face-cropped-v2/...` paths NOT in any
   Phase A report. `df40`, `visomaster_v1_base`,
   `visomaster_teams_enhanced` are training-only with no Phase A proxy.
   This audit therefore cannot evaluate F2 on 5 of the 6 paired training
   lanes from Phase A reports alone.

6. **"Previously-missed" filter uses P8A `frame_prob < 0.5` as the
   threshold.** No alternative threshold was tried. Wider thresholds
   (e.g. `< 0.7` or `< 0.9`) would expand the n_pairs denominator on
   the 3 active sub-lanes and might also unlock additional sub-lanes,
   but were not included per the user task spec referencing the yaml's
   wording ("previously missed").

7. **Frame-level pair count vs. unique-fake count.** Each unique
   fake_path participates in many cross-product pairs against many
   real_paths within the same canonical_subject (typically thousands
   per subject). The `n_pairs` reported is the cross-product count
   (real x fake within subject), not the unique-fake count. With each
   unique fake matching against ~30-100 reals per subject, the metric
   weights subjects with denser real coverage more heavily.

## Output files

- `per_lane_per_ckpt_lift.csv` — per (lane, ckpt) row with n_pairs,
  n_fake_gt_real, frac_fake_gt_real, lift_abs, lift_rel_pct.
- `pass_summary.csv` — per-ckpt count of lanes passing the 30%
  relative-lift bar.
- `compute_f2.py` — computation script (re-runnable; cache-free, reads
  Phase A reports and pair manifest directly).
