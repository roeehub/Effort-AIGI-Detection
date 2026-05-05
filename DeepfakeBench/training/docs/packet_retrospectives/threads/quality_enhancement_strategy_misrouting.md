# Thread: `quality_enhancement` strategy is misrouted to `deeplive_enhanced_fake` family — CONFIRMED BUG (2026-05-05)

> **✗ CONFIRMED BUG (2026-05-05 evening)**: visual inspection of representative frames in `/tmp/qe_check/` confirmed `quality_enhancement_*` frames are visually distinct from GFPGAN-applied frames (`edge_cases_enhanced_*`, `minimal_processing_enhanced_*`) and equivalent to base strategy frames in their lack of GFPGAN-style smoothing/super-resolution. Combined with the manifest evidence (no `enhancement: GFPGAN_sample` field), the routing is now confirmed as a bug. ~5,120 of ~18,880 deeplive-enhanced-fake training frames (~27%) were mislabeled across **every R13 packet trained since 2026-03-19**. **Fix is pending user authorization** (one line in `utils/grouping.py` + ~15 R13 yamls + image rebuild + smoke test). Until fix lands, do NOT propose new training runs that depend on correct deeplive enhanced-vs-non-enhanced family balance — they will inherit the contamination.

## The question

Are `quality_enhancement_*` deeplive fakes correctly classified as enhanced fakes at training time, or are they being silently routed into the enhanced-fake family despite carrying no enhancer post-processing?

## Evidence

### Manifest evidence (cropped bucket — what training reads)

`gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/edge_cases_enhanced_0001/manifest.json`:
```json
{
  "sample_id": "edge_cases_enhanced_0001",
  "original_sample_id": "edge_cases_0001",
  "strategy": "edge_cases",
  "enhancement": "GFPGAN_sample",
  "model": "GFPGANv1.4",
  "cropped": true,
  "frame_count_real": 16,
  "frame_count_fake": 16
}
```

`gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/minimal_processing_enhanced_0000/manifest.json` is structurally identical: `enhancement: GFPGAN_sample`, `model: GFPGANv1.4`, paired to a base via `original_sample_id`.

`gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/quality_enhancement_0002/manifest.json`:
```json
{
  "sample_id": "quality_enhancement_0002",
  "strategy": "quality_enhancement",
  "frame_count": 16,
  "frame_count_real": 16,
  "frame_count_fake": 16,
  "cropped": true
}
```

No `enhancement` field. No `model` field. No `original_sample_id`. The strategy name itself (`quality_enhancement`) is a base strategy in the deeplive pipeline; it has no paired `_enhanced` variant in the bucket.

### Code routing evidence

`utils/grouping.py:13-17`:
```python
DEFAULT_ENHANCED_STRATEGIES = (
    "quality_enhancement",
    "edge_cases_enhanced",
    "minimal_processing_enhanced",
)
```

`utils/grouping.py` `infer_group_key` (deeplive fake branch, line 224):
```python
if strategy == "quality_enhancement":
    return "deeplive_quality_enhancement_fake"
```

`utils/grouping.py` `infer_family_key` (lines 290-294):
```python
if group_key in {
    "deeplive_quality_enhancement_fake",
    "deeplive_edge_cases_enhanced_fake",
    "deeplive_minimal_processing_enhanced_fake",
}:
    return "deeplive_enhanced_fake"
```

→ `quality_enhancement` fakes end up in the same `family_key = deeplive_enhanced_fake` as confirmed-GFPGAN-applied frames at training time. The base `edge_cases` and `minimal_processing` strategies (no `_enhanced` suffix) correctly route to `deeplive_non_enhanced_fake` — only `quality_enhancement` is the disputed routing.

### Yaml propagation

Every R13 P-series yaml that overrides `enhanced_strategy_names` REPEATS the same list (RLP5_01, RLP35_07, FT3, P11, P14_FACE_SCALE_JITTER_ISOLATED, P16, RLP4_01, RLP6B_05, RLP2_02, WTB3, PA, PC, etc.):

```yaml
augmentation:
  routing:
    mode: "family_aware"
    enhanced_strategy_names:
      - "quality_enhancement"
      - "edge_cases_enhanced"
      - "minimal_processing_enhanced"
```

So the override mechanism that exists to LET callers customize `enhanced_strategy_names` has only ever been used to repeat the default. No yaml has ever excluded `quality_enhancement` from the enhanced set.

### Bucket frame counts

| Strategy | Folders (cropped) | Frames per folder | Total fake frames | Current family | If routing corrected |
|---|---:|---:|---:|---|---|
| `edge_cases` | 435 | 16 | 6,960 | `deeplive_non_enhanced_fake` | (correct, no change) |
| `edge_cases_enhanced` | 435 | 16 | 6,960 | `deeplive_enhanced_fake` | (correct, GFPGAN-applied) |
| `minimal_processing` | 425 | 16 | 6,800 | `deeplive_non_enhanced_fake` | (correct, no change) |
| `minimal_processing_enhanced` | 425 | 16 | 6,800 | `deeplive_enhanced_fake` | (correct, GFPGAN-applied) |
| **`quality_enhancement`** | **320** | **16** | **5,120** | **`deeplive_enhanced_fake` (DISPUTED)** | **`deeplive_non_enhanced_fake`?** |

If the routing is wrong, ~27% of the current `deeplive_enhanced_fake` training volume (5,120 / 18,880) is mislabeled and ~37% of `deeplive_non_enhanced_fake` volume (5,120 / 13,760) is missing.

### Family weight effect (representative — PA yaml)

`R13_PA_VISOMASTER_ENHANCED_DATA.yaml`:
```yaml
family_weights:
  deeplive_non_enhanced_fake: 2.5
  deeplive_enhanced_fake: 3.0      # quality_enhancement currently lands here
```

If the routing is wrong, `quality_enhancement` frames are sampled at weight 3.0 instead of 2.5 — a 20% over-weighting on a 5,120-frame slice.

### Git blame

Commit `38558ee5` ("refactor: modular trainer + dataset + config system", 2026-03-19, roeehub) introduced `DEFAULT_ENHANCED_STRATEGIES`. The commit was a large refactor (~965K LOC); the commit message contains no specific rationale for grouping `quality_enhancement` with the GFPGAN-applied strategies. The lines have not been touched since.

`git log --all -S "quality_enhancement" -- utils/grouping.py` returns only that one commit. No commit messages anywhere in the repo mention `quality_enhancement` directly.

### Visual evidence (queued)

Sample frames downloaded for direct comparison at `/tmp/qe_check/` (CPU job 2026-05-05):
- `01_BASE_*` and `05_BASE_*`: pre-enhancer base fakes
- `03_GFPGAN_*` and `06_GFPGAN_*`: post-GFPGAN enhanced fakes (known)
- `07_QE_*` through `10_QE_*`: disputed `quality_enhancement` frames

File-size signatures:
- BASE: 63–73 KB
- GFPGAN-enhanced: 70–81 KB (~11–12% larger than base — consistent with super-resolution adding pixel-level detail)
- `quality_enhancement`: 61–91 KB (high variance — neither cleanly base-like nor cleanly GFPGAN-like)

The size variance on `quality_enhancement` is itself informative: if it were a uniform "enhanced via mechanism X" treatment, sizes would cluster like the GFPGAN-applied folders do. The variance hints at heterogeneous treatment.

## Affected scope

**Every R13 training packet** is affected, because every yaml's `enhanced_strategy_names` override repeats the same disputed list (or falls back to `DEFAULT_ENHANCED_STRATEGIES` which includes `quality_enhancement`). This includes:
- The P-series (P8A, P9–P22)
- The S-series (S1–S3)
- The E-series (E2B, E3, etc.)
- The A/B/C-series (PA, PC-codec)
- All FT-from-* anchor candidates evaluated in the contract scorecard

The scope of affected experiments: ~all of R13 since 2026-03-19 commit landed.

## Resolution (2026-05-05 evening) — bug confirmed via visual inspection

User performed visual inspection of `/tmp/qe_check/` samples and reported: **"quality enhancement is non-GFPGAN like we suspected."** Combined with the prior manifest evidence (no `enhancement: GFPGAN_sample` field on `quality_enhancement_*` folders, vs explicit GFPGAN tagging on `*_enhanced` folders), the routing is now confirmed as a bug.

The bug was introduced in commit `38558ee5` (2026-03-19, by user) as part of a large refactor titled "modular trainer + dataset + config system" — a 965K-LOC commit where the design decision for `DEFAULT_ENHANCED_STRATEGIES` was not surfaced or documented. The line has not been touched since.

**Confirmed effect on training**:
- ~5,120 frames (27% of `deeplive_enhanced_fake` training volume) were routed to the wrong family
- Family weight 3.0 was applied instead of correct 2.5 — a 20% over-weighting on this slice
- Family-aware aug treated these frames as enhanced when they are not
- Affects every R13 P-series packet trained since 2026-03-19

**Confirmed non-effect**:
- Eval suites use `method` field directly, not the family-key inference path. Eval recall numbers are NOT affected.
- Visomaster routing is independent — viso ceiling is not explained by this bug.
- Relative comparisons between R13 ckpts (P8A vs E2B vs PA, etc.) remain valid because contamination was uniform across packets. Absolute claims about "model handles GFPGAN-enhanced fakes well" need revisiting after the fix.

## Pending fix (awaiting user authorization as of 2026-05-05 evening)

The fix is a one-line code change plus a sweep of yaml overrides plus image rebuild:

1. **`utils/grouping.py:13-17`** — remove the `"quality_enhancement",` line from `DEFAULT_ENHANCED_STRATEGIES` (one-line change)
2. **R13 yamls under `experiments/phase2_round13/`** — every yaml that has an `enhanced_strategy_names:` block needs the `"quality_enhancement"` entry removed. Confirmed yamls: `R13_RLP5_01_E3_teams2_5.yaml`, `R13_RLP35_07_stack_top3.yaml`, `R13_WTB3_with_proper_data_unenhanced_provisional.yaml`, `R13_P16_DATA_AXIS.yaml`, `R13_P11_SMOKE_TEST.yaml`, `R13_P14_FACE_SCALE_JITTER_ISOLATED.yaml`, `R13_RLP4_01_arcface_m0125_control.yaml`, `R13_RLP6B_05_label_smooth_fake.yaml`, `R13_RLP2_02_FT_WTB1_plus_proper_unenhanced_live.yaml`, `R13_FT3_enhanced_boosted_teams.yaml`, `R13_PA_VISOMASTER_ENHANCED_DATA.yaml`, `R13_PC_CODEC_PLUS_DATA.yaml` — and any others discovered via `grep -l 'enhanced_strategy_names' experiments/phase2_round13/`. Approximately 15 yamls.
3. **Test** — add or update a unit test in `tests/test_grouping.py` (or equivalent) that asserts `infer_family_key(label=1, method="quality_enhancement_0001", source="deeplive") == "deeplive_non_enhanced_fake"`. Currently the routing returns `deeplive_enhanced_fake`; post-fix it should return `deeplive_non_enhanced_fake`.
4. **Image rebuild** — `./dev.sh build-prod -y` per `image_rebuild_discipline.md`. Auto-bumps VERSION patch.
5. **Validation re-run** (optional but recommended) — a small FT-from-base packet trained with corrected routing measures the delta in deeplive enhanced-vs-non-enhanced training balance and downstream validation metrics. <$30 GPU; can be folded into the in-scoping deeplive ship experiment as the corrected-baseline.

**What changes for the deeplive ship experiment in scoping**: it should use the CORRECTED routing (post-fix codebase) so it doesn't compound the contamination into yet another packet. Otherwise we'd be measuring "correlation-penalty loss + 27%-mislabeled training" instead of the loss alone.

**Add a critical-reading banner to historical packet retros** (P8A, E2B, PA, PC, etc.): one-line note "trained on `quality_enhancement` mislabeled as enhanced family — see `quality_enhancement_strategy_misrouting.md`". Preserves all numerical content, flags the framing.

## Open loop

### Open loop: quality-enhancement-strategy-misrouted-fix-pending
status: in-progress
severity: medium
first_seen: 2026-05-05
last_verified: 2026-05-05
close_criterion: routing fix is applied — `quality_enhancement` is removed from `DEFAULT_ENHANCED_STRATEGIES` in `utils/grouping.py:13-17` AND from every R13 yaml's `enhanced_strategy_names` override (~15 yamls under `experiments/phase2_round13/`); a unit test asserts `infer_family_key` returns `deeplive_non_enhanced_fake` for a `quality_enhancement` fake input; image is rebuilt via `./dev.sh build-prod -y` (auto-bumps VERSION); and a critical-reading banner is added to the README and to historical packet retros (P8A, E2B, PA, PC) noting they trained on the contaminated routing. Validation re-run on at least one FT-from-base packet measuring the delta in deeplive enhanced-vs-non-enhanced family balance is recommended but not required for closure (could be folded into the in-scoping deeplive ship experiment instead). Verification step (visual inspection) is COMPLETE 2026-05-05 evening — user reported "quality enhancement is non-GFPGAN like we suspected." Bug confirmed; fix pending authorization.

## Implications for current/future packets

- **Deeplive ship experiment (in scoping as of 2026-05-05)**: the corrected-routing question matters here. If `quality_enhancement` is mislabeled, retraining with the corrected family-balance changes the deeplive enhanced-fake distribution by ~27%. Worth resolving the routing question BEFORE the GPU spend.
- **Visomaster experiments**: not directly affected (visomaster has its own family routing). But any agent comparing deeplive-vs-visomaster recall asymmetries should note that the deeplive enhanced-family training distribution is contaminated under the bug hypothesis.
- **Evaluation**: eval suites use `method` field (`deeplive_enhanced` for the Dor-bound suite), not the family-key inference path. So eval recall numbers are NOT affected by this routing bug. Only training-side family weights and family-aware aug are affected.
- **Historical results re-interpretation**: every prior P-series scorecard was trained on the disputed routing. If confirmed as a bug, prior conclusions about deeplive-enhanced-fake-handling are based on a contaminated training distribution. This is a uniform contamination across packets, so RELATIVE comparisons (P8A vs E2B vs PA) remain valid; ABSOLUTE conclusions about "the model can handle GFPGAN-enhanced deepfakes" need revisiting.

## Cross-references

- `utils/grouping.py:13-17` (the load-bearing definition)
- `utils/grouping.py:200-235` (`infer_group_key` deeplive routing)
- `utils/grouping.py:280-300` (`infer_family_key` mapping to family)
- `data/sources/combined_paired.py:2425-2426` (the family-aware aug strategy counts)
- `data/sources/deeplive.py:16` (the bucket connection)
- Commit `38558ee5` (the introduction)
- `experiments/phase2_round13/R13_PA_VISOMASTER_ENHANCED_DATA.yaml:131-138` (a typical override repeating the disputed list)
- Memory: `project_quality_enhancement_routing_2026-05-05.md` (auto-loaded summary)
- Sample frames: `/tmp/qe_check/` (CPU job 2026-05-05; also reproducible via `gsutil cp` from the bucket paths cited above)
