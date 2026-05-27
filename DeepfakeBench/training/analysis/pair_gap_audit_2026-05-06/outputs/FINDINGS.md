# PAIR_GAP_AUDIT_P8A_E2B_PA_2026-05-06 — Findings

**Generated:** 2026-05-06T14:30:01.931455+00:00Z
**Manifest:** `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv`
**Pairs constructed:** 37,327 across 11 canonical subjects
**Subjects without paired structure (label=0 only or label=1 only):** 54

## Headline verdict

**RED** — pair-rank loss is dead. Skip P1; demote to P2 (`PE_SBI`).

## Decision rule

> Per ckpt, P(pair_gap <= 0 | missed_fake):
> - **>25%** → **GREEN** (lever has signal).
> - **10-25%** → **AMBER** (marginal; depends on margin of loss + GroupDRO weighting).
> - **<10%** → **RED** (lever is dead — fakes already rank above reals on missed fakes).

## Per-ckpt summary (transport-matched primary)

| Ckpt | N pairs (TM) | N missed_fake (TM) | P(gap<=0\|missed) [pair-w, TM] | P(gap<=0\|missed) [subj-w, TM] | Mean gap (logit, all) | Verdict |
|------|------------:|------------------:|------------------------------:|-------------------------------:|---------------------:|---------|
| P8A | 37,043 | 3,644 | 0.0988 | 0.0860 | 8.0845 | RED |
| E2B | 37,043 | 3,462 | 0.0563 | 0.0294 | 7.7810 | RED |
| PA_3800 | 37,043 | 5,863 | 0.1779 | 0.1080 | 6.0266 | AMBER |

## All-pair vs transport-matched cross-check

| Ckpt | P(gap<=0\|missed) all pair-w | all subj-w | TM pair-w | TM subj-w |
|------|----------------------------:|----------:|---------:|---------:|
| P8A | 0.0987 | 0.0858 | 0.0988 | 0.0860 |
| E2B | 0.0551 | 0.0283 | 0.0563 | 0.0294 |
| PA_3800 | 0.1786 | 0.1062 | 0.1779 | 0.1080 |

## Overall pair_gap distribution (raw probability) per ckpt

| Ckpt | mean | median | std | P(gap>0) | P(gap>0.5) | P(gap>1.0) | P(gap<=0) |
|------|-----:|-------:|----:|--------:|----------:|----------:|---------:|
| P8A | 0.8065 | 0.9846 | 0.3163 | 0.9878 | 0.8194 | 0.0000 | 0.0122 |
| E2B | 0.8201 | 0.9689 | 0.2641 | 0.9935 | 0.8537 | 0.0000 | 0.0065 |
| PA_3800 | 0.7290 | 0.8905 | 0.3183 | 0.9688 | 0.7790 | 0.0000 | 0.0312 |

## Conditional: missed_fake subset

| Ckpt | N missed | mean | median | P(gap<=0) |
|------|---------:|-----:|-------:|---------:|
| P8A | 3,647 | 0.1625 | 0.1569 | 0.0987 |
| E2B | 3,573 | 0.2391 | 0.2383 | 0.0551 |
| PA_3800 | 5,958 | 0.1664 | 0.2012 | 0.1786 |

## Conditional: FP_real subset (do paired fakes still outrank?)

| Ckpt | N FP_real | mean gap | median gap | P(gap>0) |
|------|----------:|---------:|----------:|---------:|
| P8A | 3,067 | 0.1772 | 0.1576 | 0.9475 |
| E2B | 1,740 | 0.2824 | 0.3488 | 0.9552 |
| PA_3800 | 1,250 | 0.0751 | 0.1323 | 0.6960 |

## Coverage by canonical subject

| Canonical subject | N real | N fake | N pairs | Paired |
|---|---:|---:|---:|:---:|
| test_cam__s73 | 251 | 101 | 4,000 | Y |
| cam_test__s32 | 81 | 235 | 4,000 | Y |
| xiang_xiang2_feng | 301 | 135 | 4,000 | Y |
| cam_test__s38 | 85 | 85 | 4,000 | Y |
| test_cam__s76 | 214 | 138 | 4,000 | Y |
| dor_local | 568 | 2,871 | 4,000 | Y |
| extra_xiang | 224 | 73 | 4,000 | Y |
| extra_xinghe | 19 | 366 | 4,000 | Y |
| pc_generator__s15 | 29 | 91 | 2,639 | Y |
| dor_shkedi__s16 | 31 | 78 | 2,418 | Y |
| pc_generator__s4 | 9 | 30 | 270 | Y |
| pc_generator__s14 | 103 | 0 | 0 | N |
| pc_generator__s34 | 92 | 0 | 0 | N |
| pc_generator__s3 | 0 | 118 | 0 | N |
| pc_generator__s22 | 227 | 0 | 0 | N |
| bla_bla_chow | 311 | 0 | 0 | N |
| pc_generator__s13 | 212 | 0 | 0 | N |
| pc_generator__s8 | 101 | 0 | 0 | N |
| orel | 35 | 0 | 0 | N |
| md_noyn_sharker__s15 | 682 | 0 | 0 | N |
| may5_xinhe | 60 | 0 | 0 | N |
| pc_generator__s45 | 91 | 0 | 0 | N |
| roy_d | 130 | 0 | 0 | N |
| pc_generator__s9 | 0 | 46 | 0 | N |
| q__s6 | 54 | 0 | 0 | N |
| real_dor | 109 | 0 | 0 | N |
| roee_tester_real_2026-03-24 | 323 | 0 | 0 | N |
| may5_roee | 30 | 0 | 0 | N |
| royd_real_2026-03-06 | 181 | 0 | 0 | N |
| test_cam__s41 | 815 | 0 | 0 | N |
| test_cam__s53 | 0 | 165 | 0 | N |
| tester_roee_real_2026-03-06 | 173 | 0 | 0 | N |
| xiang | 159 | 0 | 0 | N |
| xiang_xiang2_feng__s23 | 102 | 0 | 0 | N |
| may5_xiang | 30 | 0 | 0 | N |
| live_prod__xinhe-fake-8-glasses | 0 | 123 | 0 | N |
| may5_noyn | 60 | 0 | 0 | N |
| live_prod__xiang-fake-5 | 0 | 61 | 0 | N |
| bla_bla_chow__s2 | 180 | 0 | 0 | N |
| cam_test__s33 | 0 | 334 | 0 | N |
| cam_test__s35 | 0 | 365 | 0 | N |
| cam_test__s46 | 0 | 124 | 0 | N |
| chikara_takahashi__s22 | 42 | 0 | 0 | N |
| dor_shkedi_teams | 1,220 | 0 | 0 | N |
| extra_roy_d | 236 | 0 | 0 | N |
| ilan | 29 | 0 | 0 | N |
| live_prod__xiang-fake-1 | 0 | 50 | 0 | N |
| live_prod__xiang-fake-2 | 0 | 84 | 0 | N |
| live_prod__xiang-fake-3 | 0 | 83 | 0 | N |
| live_prod__xiang-fake-4 | 0 | 78 | 0 | N |

## Per-subject P(gap<=0 | missed_fake) — heterogeneity check

This audit pools pairs across canonical subjects, but the result can be dominated by one heavy subject (e.g. dor_local has the most fakes in this manifest). Per-subject breakdown shows where the lever has real signal vs where it is dead.

### P8A

| Canonical subject | N missed_fake | P(gap<=0\|missed) |
|---|---:|---:|
| dor_local | 836 | 0.2763 |
| extra_xinghe | 525 | 0.1238 |
| xiang_xiang2_feng | 2,228 | 0.0287 |
| dor_shkedi__s16 | 31 | 0.0000 |
| test_cam__s76 | 27 | 0.0000 |

### E2B

| Canonical subject | N missed_fake | P(gap<=0\|missed) |
|---|---:|---:|
| dor_local | 1,471 | 0.0802 |
| xiang_xiang2_feng | 1,357 | 0.0575 |
| extra_xinghe | 250 | 0.0040 |
| dor_shkedi__s16 | 434 | 0.0000 |
| test_cam__s76 | 61 | 0.0000 |

### PA_3800

| Canonical subject | N missed_fake | P(gap<=0\|missed) |
|---|---:|---:|
| dor_local | 2,271 | 0.4249 |
| xiang_xiang2_feng | 856 | 0.0993 |
| extra_xinghe | 2,029 | 0.0069 |
| dor_shkedi__s16 | 775 | 0.0000 |
| test_cam__s76 | 27 | 0.0000 |

## Recommended FT base if pair-rank is greenlit

- **Recommended FT base:** `PA_3800` — P(pair_gap<=0 | missed_fake) [TM pair-weighted] = 0.178, highest of ['PA_3800'].
- **Rationale:** the ckpt where the greatest fraction of missed fakes are *strictly below* their paired real has the most headroom for a margin-based pair-rank loss to bite.

## Caveats

- Pairs are CROSS-PRODUCT within canonical_subject (not frame-level). The inference manifest does not carry frame-level paired structure (teams fake/real are different sessions; visomaster_v2_dor has no co-bucketed real). This audit therefore answers the question at the identity / canonical-subject pool level. If fake>real ordering is broken at this coarser level it will also be broken at the trainer's tighter same-source pair level (the latter has higher score noise). Greenlight is conservative; redlight is suggestive only.
- The inference manifest only contains 14k cached scores. The full training paired loader has access to ~5,379 DF40 + DeepLive + VisoMaster pairs that ARE frame-level paired (see `data/sources/df40_paired.py`, `combined_paired.py`). The audit here is on the **eval** substrate, NOT the training substrate. The go/no-go signal is whether the model fails the pair-rank objective on data it sees at deployment, not on data it was trained on.
- `score_PA_3800` is from the PA chain (E2B + viso enhanced data fw=4.0). PA does NOT generalise to HDTF (memory `pa_does_not_generalize_to_hdtf_2026-05-05`); its verdict here is informational only — do not seed PE from PA.
- Cross-product pairing inflates pair counts; the absolute number of pairs (millions) is not directly comparable to a per-batch training pair count. The **fraction** statistics (P(gap<=0)) are the load-bearing outputs.
- Method/enhancer/transport columns are heuristically parsed from frame_path; not all paths follow a documented schema.
