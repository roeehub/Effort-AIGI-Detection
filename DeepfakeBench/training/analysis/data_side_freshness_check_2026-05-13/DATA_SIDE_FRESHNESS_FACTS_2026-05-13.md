# DATA-SIDE FRESHNESS CHECK — FACTS

Date: 2026-05-13. Verified from primary evidence in the working tree (code, yaml, git log, cached parquets, gsutil). Memory entries used only as starting pointers — not as authoritative state.

Scope: five concerns claimed against the Teams deepfake detector training pipeline, evaluated for the four upcoming GPU slots:
- Slot 1 — continuous-axis multi-axis-GRL on T5C (`R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml`)
- Slot 2 — LoRA on P8A L10-L11 (`R13_LORA_L10_L11_2026-05-13.yaml`)
- Slot 3 — L11 anchor on T5C chronic-6
- Slot 4 — B16-scratch + Fourier-aug bands 8-13

FACTS-doc rules (per AGENT_GUIDE.md Rule 5): mechanical pass/fail only; no "succeeds/fails/wins/loses/promotes" language.

---

## §1. Concern 1 — `quality_enhancement` strategy misrouting

**Verdict: FIXED.**

Primary evidence:
- `utils/grouping.py:13-24` — `DEFAULT_ENHANCED_STRATEGIES` no longer contains `quality_enhancement`; only `edge_cases_enhanced` and `minimal_processing_enhanced` are listed. An inline comment cites the fix date and references the retrospective thread.
- `utils/grouping.py:217-218, 230-231, 290-296` — `infer_group_key` returns `deeplive_quality_enhancement_real` / `deeplive_quality_enhancement_fake`, which now fall through `infer_family_key`'s general `deeplive_*_fake` branch (line 303-306) and route to `deeplive_non_enhanced_fake`. The explicit `"quality_enhancement"` disjunct in the prior fallback was removed.
- `tests/test_grouping_quality_enhancement.py` — 8 unit tests; all pass (verified via `pytest -v` 2026-05-13):
  - `test_quality_enhancement_not_in_default_enhanced` PASSED
  - `test_quality_enhancement_routes_to_non_enhanced_family` PASSED
  - `test_quality_enhancement_routing_invariant_under_yaml_override` PASSED
  - (+5 more, all PASSED)
- Fix commit: `ce76289` ("fix: remove quality_enhancement from DEFAULT_ENHANCED_STRATEGIES + add unit test", 2026-05-05).
- Buggy commit identified in commit message: `38558ee5` (2026-03-19 refactor).
- The 3 slot yamls (P8A, T5C, LoRA) and the two T4 yamls all set `augmentation.routing.enhanced_strategy_names` to `["edge_cases_enhanced", "minimal_processing_enhanced"]` only — `quality_enhancement` is NOT in the override list, so any future change to `DEFAULT_ENHANCED_STRATEGIES` is also superseded by an explicit yaml-side allowlist.
- `R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml:235-241` (and equivalent in the other 3 slot yamls): `deeplive.include_strategies` still lists `quality_enhancement` as a pulled strategy — i.e., the data is still ingested, but is now correctly classified as `deeplive_non_enhanced_fake`.

Memory-entry status: `project_quality_enhancement_routing_2026-05-05` is STALE. It says "fix pending user authorization" — the fix landed at commit ce76289 on 2026-05-05.

Practical impact on the 4 slots: NONE — bug is closed; routing is correct for all 4 slot yamls.

---

## §2. Concern 2 — `sharpness_laplacian` computed on full image, not face crop

**Verdict: STILL A PROBLEM (eval-side); PARTIALLY-AFFECTS Slot 1 (training-side).**

Primary evidence (eval-side):
- `analysis/lockbox_tagging/layers/quality.py:81-82` (current working tree, verified 2026-05-13):
  ```
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  out["sharpness_laplacian"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
  ```
  The `img` here is the full file loaded via `cv2.imread(str(path), cv2.IMREAD_COLOR)` (line 76) — no bbox crop is applied. This is the same code path called out in `sharpness_metric_bug.md`.
- `git log -- analysis/lockbox_tagging/layers/quality.py` — empty output (no recent commits; file is unchanged since at least the audit).
- `PLAN.md:103, 170, 320` — currently marks this as an OPEN concern with documented reproduction (dor_shkedi full=760.7 / face=484.1 / bg=844.9; PC_Generator__s15 full=555.1 / face=443.6 / bg=1018.1). PLAN.md treats this as an open follow-up loop.
- `docs/packet_retrospectives/README.md:138` lists the thread `sharpness_metric_bug` as live.

Training-side picture (independent computation):
- The training-time IQ axis used by `multi_axis_grl` is computed in `loss/correlation_penalty.py:57-110` via `compute_pixel_axes(image)` (called from `detectors/effort_detector.py:1102`), NOT via `analysis/lockbox_tagging/layers/quality.py`. The two functions are independent code paths — fixing one does not fix the other.
- `loss/correlation_penalty.py:63` docstring explicitly acknowledges: "matches the lockbox-tagging parquet's column (modulo full-vs-face-crop caveat from `sharpness_metric_bug`)" — the function computes the Laplacian variance on the whole input tensor, i.e., the entire post-augmentation crop fed to the model, NOT a face bbox subset.
- Therefore Slot 1's `sharpness_laplacian_high` axis (T5C `multi_axis_grl.axes` line 147) is computed on the model's full input image. This is the SAME mode of computation as the eval-side bug — i.e., what is being adversarially predicted-then-discouraged is "full-input-image Laplacian variance," which conflates face sharpness with crop-background sharpness in the same way the eval metric does.
- This is a per-batch median split (lines 1112-1113 in effort_detector), so the GRL target is relative within batch — but the underlying signal is still confounded.

Memory-entry status: `project_sharpness_metric_bug` is CURRENT for the eval-side claim. PLAN.md correctly flags it as open.

Practical impact on the 4 slots:
- Slot 1 (multi-axis-GRL on `sharpness_laplacian_high`): the training-time IQ axis is computed on the full input crop, NOT the face bbox. The GRL will discourage the model from using full-input-image Laplacian variance. Whether this is meaningfully different from discouraging "face-area Laplacian variance" depends on the per-batch correlation between face-sharpness and full-crop-sharpness, which is non-trivial because the crops vary in tightness across substrates (see `eval_production_crop_tightness_gap` memory). Slot 1 will still run; it just isn't a clean "face-sharpness invariance" lever.
- Slot 2 (LoRA on P8A L10-L11): no IQ axis is used; not affected.
- Slot 3 (L11 anchor on chronic-6): no IQ axis is used; not affected unless the anchor-construction logic queries `sharpness_laplacian`. (Not verified in this audit — flag for Slot-3 design review.)
- Slot 4 (B16-scratch + Fourier-aug bands 8-13): no IQ axis is used; not affected.

---

## §3. Concern 3 — `visomaster_hints` / `visomaster_hints_teams` contamination

**Verdict: FIXED (yaml-level disabling).**

Primary evidence (per-yaml grep, full-block context read):
- `R13_RLP8_01_unfreeze_clip_codec.yaml` (P8A base):
  - line 156-160: `visomaster_hints:\n  enabled: false`
  - line 188-192: `visomaster_hints_teams:\n  enabled: false`
- `R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml` (T5C):
  - line 271-272: `visomaster_hints_teams:\n  enabled: false`
  - `visomaster_hints` block not present (defaults to disabled per `combined_paired.py:4020` which reads `enabled: False` from an empty dict).
- `R13_LORA_L10_L11_2026-05-13.yaml` (Slot 2 LoRA):
  - line 280-281: `visomaster_hints_teams:\n  enabled: false`
  - `visomaster_hints` block not present.
- `R13_T4_MULTI_AXIS_GRL_2026-05-10.yaml`:
  - line 269: `visomaster_hints_teams:` (block follows, `enabled: false`)

Data-loader gating verified:
- `data/sources/combined_paired.py:4020-4023`: `visomaster_hints_enabled = bool(visomaster_hints_config.get('enabled', False))` and equivalent for `_teams`. The loader only invokes `_select_visomaster_hints_samples` (line 4285-4286) and `_select_visomaster_hints_teams_samples` (line 4310) when the respective `enabled` flag is true. With both flags `false`, no hint-lane samples are pulled into the training pool for any of the 4 slot yamls.

Memory-entry status: `project_visomaster_hints_lanes_bad_data` is CURRENT in its prescription ("never train on these"), but the warning is operationally already satisfied for the 4 slot bases. The lanes still exist as code paths and could be re-enabled by a future yaml.

Practical impact on the 4 slots: NONE — all 4 slot yamls have both hint lanes disabled.

---

## §4. Concern 4 — Direct-Teams real-frame availability beyond P8A's 1,200 VCD cap

**Verdict: STILL A PROBLEM (cap is restrictive; not a code/data fix).**

Primary evidence (gsutil run without re-auth — used existing token, succeeded):
- `gsutil ls gs://effort-collected-data/real/VCD/ | wc -l` returns **158** identity-folders (one per `<md5>_<resolution>_<framerate>` directory).
- Identity folder naming pattern: `<md5>_<W>x<H>_<fps>/` — examples include `021e305217c4bb...79e6_1080x1920_30/`, `0380a333cb0f...05f_1920x1080_30/`. Resolution/framerate is encoded; the `_30` suffix indicates 30fps source video.
- Yaml caps (R13_RLP8_01:205-214 and equivalent in T5C, LoRA, T4):
  - `identity_train_fraction: 0.40` → ~63 of 158 identities used
  - `max_frames_per_identity: 15`
  - `max_total_samples: 1200`
  - Effective max ≈ min(63 × 15, 1200) = 945 frames
- Available headroom: at 30fps for ~15-30s clips per identity, the underlying frame count per identity is at minimum hundreds. Even conservatively assuming 100 frames/identity available, 158 × 100 = ~15,800 frames, so the 1,200 cap is restrictive by roughly **13×** on raw frame count.
- Other potential direct-Teams real sources visible in P8A yaml:
  - `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` (line 252-253) — OOD-monitoring only, capped at 300 videos.
  - `real-teams-dor-roee` (line 262) — readout-only, capped at 10 videos.
  - `gs://effort-collected-data/real/real_social_12_09/` — exists per `gsutil ls`, NOT referenced anywhere in the 4 slot yamls.
  - `gs://effort-collected-data/real/external_youtube_avspeech/` — used for OOD monitoring + lighting-stress, NOT training.

Memory-entry status: no memory entry directly tracks this. PLAN.md notes D9 caveat 4 ("`external_training_reals` lane … not enumerated in this audit").

Practical impact on the 4 slots:
- The 1,200-sample cap is purely a yaml change to lift. The underlying bucket has substantially more frames available without new data engineering.
- Slot 1 (T5C continuous-axis-GRL): inherits the 1,200 cap. If real-side recall on Teams-like substrate is bottlenecked by VCD-real volume, raising `max_total_samples` and/or `identity_train_fraction` is a single-yaml-line change.
- Slot 2 (LoRA): same constraint, same trivial yaml-fix path.
- Slot 3 (L11 anchor): same constraint.
- Slot 4 (B16-scratch + Fourier): same constraint.
- `real_social_12_09/` is an untapped real lane visible in the bucket — would need yaml work to add and would constitute a new data lane (not an in-place cap-lift). Not classified as "code/data-engineering required," just a new yaml `external_training_reals` entry.

---

## §5. Concern 5 — Face-pixel-area label leak still present in current training data

**Verdict: PARTIALLY-FIXED (lever exists, not used in current 4 slot yamls); UNVERIFIABLE on most-recent re-measurement.**

Primary evidence:
- `face_scale_jitter` module exists at `data/augmentations/face_scale_jitter.py` (cites `project_face_size_label_leak.md` directly in its docstring as the motivating problem).
- All 4 slot yaml face_scale_jitter blocks:
  - P8A (`R13_RLP8_01_unfreeze_clip_codec.yaml`): no `face_scale_jitter` block (i.e., NOT enabled — defaults to off).
  - T5C: `face_scale_jitter:\n  enabled: false` (line 168-169).
  - LoRA: `face_scale_jitter:\n  enabled: false` (line 165-166).
  - The two T4 yamls also explicitly disable it.
- Cohen's d reference data — `analysis/crop_shortcut_2026-04-27/training_data_face_size_by_split_label.csv`:
  - dev fake mean: 51,952 px² (n=2,444); dev real mean: 36,998 px² (n=3,832) — mean gap +14,954 px²
  - lockbox fake mean: 78,560 px² (n=425); lockbox real mean: 23,246 px² (n=414) — mean gap +55,313 px²
- Re-measurement on more-recent atlases: NOT POSSIBLE from cached state. The `iq_data_atlas_2026-05-08/_cache/*.parquet` columns are `[h, w, min_dim, max_dim, ..., lap_var, edge_mag, ..., skin_frac, bytes]` — `min_dim` / `max_dim` are full-image dimensions, NOT face-area. No face-area column is present in this atlas.
- The single existing face-area parquet, `analysis/deeplive_face_geometry_2026-05-05/face_area.parquet` (65,202 rows, columns `[frame_path, face_area_fraction, strategy, label]`), is **deeplive-only** (per its build script). It covers deeplive but does not contain the cross-method comparison needed to re-measure Cohen's d on the current training pool.
- Memory entry `project_face_size_label_leak.md` cites the 2026-04-27 finding and Cohen's d ≈ 0.37. Without a fresh cross-method face-area parquet that includes all current training sources, the "is the leak still the same size today" question is UNVERIFIABLE.

Memory-entry status: `project_face_size_label_leak.md` is OUTDATED-IN-SCOPE — it references training-pool measurements from 2026-04-27 but does not reflect that the current 4 slot yamls have `face_scale_jitter: false`, so no in-pipeline correction is active. The face-size signal in the underlying data IS likely still present at similar magnitude (no targeted re-sampling has been added between 2026-04-27 and 2026-05-13 to reduce it).

Practical impact on the 4 slots:
- Slot 1 (T5C multi-axis-GRL with axes `chronic_flag, is_dor, sharpness_laplacian_high, color_a_approx_dev_high`): the GRL does NOT target `face_area` or `face_size_high`. The face-size leak is therefore not adversarially discouraged by Slot 1.
- Slot 2 (LoRA): `face_scale_jitter: enabled: false`; face-size leak is presumed still present in inputs and inherits from P8A's base.
- Slot 3 (L11 anchor): anchor design needs review for whether `face_size` is in its conditioning set.
- Slot 4 (B16-scratch + Fourier bands 8-13): face-size leak is present in inputs. Fourier-band aug may indirectly disrupt some face-size cues but is not designed to do so.
- Recommendation: pre-launch CPU probe to re-measure face-area Cohen's d on the current training-pool sample lists (would need a one-off build of `face_area_fraction` across `train_teams_real_pool`, `train_teams_fake_pool`, and DF40 lanes from existing manifests) before assuming the 2026-04-27 magnitude still applies.

---

## §6. Recommended actions before launching the 4 slots

### Already-no-longer-concerns (skip)

- **Concern 1 (quality_enhancement misrouting)**: fixed at commit `ce76289`; tests guard against regression. Update `MEMORY.md` index entry `project_quality_enhancement_routing_2026-05-05` to status FIXED.
- **Concern 3 (visomaster_hints contamination)**: all 4 slot yamls have both hint lanes `enabled: false` and the data loader respects this gate. No yaml or code change required.

### Still-needs-attention (before-launch)

- **Concern 2 (sharpness_laplacian full-image)** — Slot 1 (multi-axis-GRL) consumes a full-input-image Laplacian variance, not a face-area Laplacian. Two options:
  1. Document explicitly in Slot 1's pre-launch FACTS that the GRL targets full-input-image sharpness (which is what the model also sees post-augmentation). This is internally consistent — the GRL discourages whatever-sharpness-signal the encoder actually receives. Cheapest.
  2. Switch `loss/correlation_penalty.compute_pixel_axes` to compute Laplacian on a face-bbox subset (would require passing landmarks or bbox through `data_dict`). Higher effort; clean fix.
  3. Out-of-scope for Slots 2, 3, 4 unless Slot 3's anchor uses an IQ axis (verify in Slot 3 design review).
- **Concern 4 (VCD cap)** — 1,200-sample cap is restrictive vs. ~13× available headroom in the same bucket. If any slot wants more direct-Teams reals, raise `max_total_samples` and `identity_train_fraction` in the yaml. NOT a blocker for slots 1-4 as currently designed (all 4 inherit the same cap), but is a cheap lever if real-side data volume becomes a binding constraint.
- **Concern 5 (face-size leak)** — UNVERIFIABLE without a one-off face-area parquet rebuild across the current training pool. Pre-launch recommendation: run a 30-60 minute CPU job to materialize `face_area_fraction` for ~5,000 frames per family from the current training manifests and re-measure Cohen's d. If still ≥ 0.30, treat as live confound for Slot 1 (multi-axis-GRL does NOT target this axis) and consider adding `face_scale_jitter: enabled: true` to one or more slot variants.

### Methodological caveats

- Concern 5's "still a problem" verdict is degraded to PARTIALLY-FIXED + UNVERIFIABLE because (a) the lever exists (face_scale_jitter), (b) no slot uses it, (c) no cached atlas can confirm the 2026-04-27 magnitude on the current pool. The combination is "the lever to fix it isn't engaged + we can't measure the current magnitude from existing files."
- Concern 2's verdict separates eval-side (STILL A PROBLEM, file unchanged) from training-side (TRAINING USES A DIFFERENT BUT EQUIVALENTLY-FLAWED CODE PATH). The eval-side fix in `quality.py` would NOT fix Slot 1's GRL axis; both code paths need attention if face-area-vs-full-image distinction is operationally important.
- gsutil ran successfully without prompting for re-auth; the existing token was valid.
- Memory entries cited and assessed: `project_quality_enhancement_routing_2026-05-05` (stale, mark FIXED), `project_sharpness_metric_bug` (current), `project_visomaster_hints_lanes_bad_data` (current prescription, operationally satisfied), `project_face_size_label_leak` (current-as-of-2026-04-27, not refreshed against 2026-05-13 pool).
