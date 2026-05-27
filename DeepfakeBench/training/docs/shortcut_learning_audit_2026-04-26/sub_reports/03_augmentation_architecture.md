# Augmentation Pipeline Architecture

Audit of `data/augmentations/pipelines.py` for the Effort deepfake detector.
Hypothesis under test: the default "asymmetric router" installs a quality
shortcut where the model learns "low quality → fake" because real and fake
samples receive systematically different augmentations.

Primary file:
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/augmentations/pipelines.py`
  (1912 lines)

Supporting files:
- `data/augmentations/transforms.py` (custom transforms incl. `VideoCodecSimulation`)
- `data/augmentations/teams_simulation.py` (Teams codec sims)
- `data/augmentations/registry.py` (registry / dispatch by version string)
- `utils/grouping.py` (`infer_family_key` resolves label/source/method → family bucket)
- `data/sources/combined_paired.py` (factory `_create_combined_transform` and meta dispatch)
- `experiments/phase2_round13/R13_P10_*.yaml` (P10 SYM/GRL launches)

---

## 1. High-level architecture of `pipelines.py`

The file is organized as a long catalog of static and configurable pipeline
factories, plus one stateful router class. Top-level public sections, in
file order:

1. **V3 / V4 / V5** static `A.Compose` factories (lines 28–151).
   Conservative → moderate → moderate+heavy hybrids. Used by older training
   recipes.
2. **V6 / V7** "portfolio" callables (lines 158–258).
   Apply randomly chosen sub-pipelines (`AUG_PIPELINE_V6_SIMULATOR`,
   `AUG_PIPELINE_V4_GENERALIST`, `AUG_PIPELINE_PURIST`,
   `AUG_PIPELINE_V3_MILD`); V6 is source-dependent, V7 is unified.
3. **Helper pipelines** (lines 265–295): `degrade_quality_pipeline`,
   `enhance_quality_pipeline`, `social_media_pipeline`.
4. **Surgical** + **General** factories (lines 302–439). Property-aware
   (e.g., `sharpness_bucket`) and config-driven generic factories.
5. **Landmark occlusion** (lines 446–576) — Task B / DeepLive only.
6. **Quality-robust pipeline** (lines 579–728) — symmetric quality
   degrade-and-enhance pipeline at three strength tiers (`light`,
   `moderate`, `strong`). Used as the fallback for the family router.
7. **Webcam / Video-call codec pipeline** (lines 731–767) — wraps
   `VideoCodecSimulation`.
8. **Family-aware Quality-Targeted Router** (lines 770–1745) — the
   centerpiece. Includes:
   - `_TEAMS_PASSTHROUGH_DEFAULTS` dict (lines 787–812).
   - `_QUALITY_TARGETED_PRESETS` dict, four strength tiers
     (`light`, `moderate`, `strong`, `vcd_targeted`) (lines 814–1002).
   - `_build_context_variation_block` (line 1011) shared lighting/exposure
     transforms.
   - `_build_teams_passthrough_special_block` (line 1086) opt-in Teams nuisance.
   - `_build_family_quality_pipeline` (line 1148) — **the asymmetric
     family-aware path** (real vs fake split lives here).
   - `_build_symmetric_quality_pipeline` (line 1421) — **the symmetric
     branch**, added in commit 2c9778b 2026-04-26.
   - `_build_teams_passthrough_pipeline` (line 1486) — for Teams families.
   - `QualityTargetedFamilyRouter` class (line 1529) — the entry point.
   - `create_quality_targeted_family_router` factory (line 1717).
9. **EVAL_STRESS_PRESETS** + `apply_eval_stress_preset` (lines 1747–1912)
   — deterministic eval-time stress (NOT training augmentation).

### Entry-point flow

The runtime path used by R13_P10_* yamls is:

`combined_paired.py::_create_combined_transform` (line 4685) reads
`config['augmentation']`, sees `version: quality_targeted_family`, calls
`create_quality_targeted_family_router(strength, routing_mode,
enhanced_strategy_names, preset_overrides, teams_codec_simulation)` at
line 4780. The returned `QualityTargetedFamilyRouter` is wrapped in a
`transform_fn(image, landmarks, meta)` that the dataset's
`_apply_transform` (line 2710 of combined_paired.py) calls per frame with
`meta = {'label': 0|1, 'source': ..., 'method': ...}`.

Inside `QualityTargetedFamilyRouter.__call__` (line 1679):
1. If `routing_mode` is neither `"family_aware"` nor `"symmetric"`, fallback
   to `create_quality_robust_pipeline("moderate")` (line 1687).
2. Otherwise, `infer_family_key(label, method, source, enhanced_strategy_names)`
   (line 1693, defined in `utils/grouping.py:250`) maps the sample to one of:
   `df40_fake`, `df40_real`, `deeplive_non_enhanced_fake`,
   `deeplive_enhanced_fake`, `visomaster_fake`, `visomaster_enhanced_fake`,
   `realpool_real`, `external_real`, `deeplive_teams_fake|real`,
   `visomaster_hints_*`, `proper_visomaster_*`, `proper_real_*`,
   `unknown_fake`, `unknown_real`, `wma_failure_fake`.
3. The router looks up `self._pipelines[family_key]` (line 1699) and applies
   it. Unknown keys fall through to `self._fallback` (line 1583 / 1550).
4. After the per-family pipeline, `_maybe_apply_teams_sim` (line 1703) may
   apply `TeamsCodecSimulation` as a post-step (only when
   `teams_codec_simulation.enabled` is True at the YAML level).

This is the same dispatch in both `family_aware` and `symmetric` modes.
The only difference is the contents of `self._pipelines` (line 1552 vs 1585).

---

## 2. The asymmetric router (`routing.mode == "family_aware"`)

In `family_aware`, each of the 13 non-teams family keys gets a hand-tuned
pipeline built by `_build_family_quality_pipeline(family_key, p)`
(line 1148). The shared building blocks are:
- `balanced_degrade` — `OneOf` over JPEG / GaussianBlur / GaussNoise /
  Downscale, each child `p=1.0` (lines 1151–1161).
- `webcam_codec_step = VideoCodecSimulation(...)` (line 1167) — applied
  to ALL families with `p = webcam_codec_p` (preset-driven).
- `color_block` — `OneOf` over `RandomBrightnessContrast` /
  `HueSaturationValue` (lines 1172–1187).
- `context_variation` — list returned by `_build_context_variation_block`
  (line 1191), only non-empty when `context_variation_enabled` is True.

For the audit, the `vcd_targeted` preset is the relevant one (every
R13_P10_* yaml uses `strength: vcd_targeted`). Preset values cited here
come from `_QUALITY_TARGETED_PRESETS["vcd_targeted"]` (lines 947–1001).

### 2.1 Quantified asymmetries (vcd_targeted preset)

#### 2.1.a OneOf degradation probability
The `balanced_degrade` group is used inside an `A.OneOf` whose outer `p`
differs by family.

| Family | Outer `OneOf` p | Source line |
|---|---|---|
| df40_fake | `p["quality_p"] = 0.60` | line 1201 |
| deeplive_non_enhanced_fake | `min(0.9, p["quality_p"] + 0.08) = 0.68` | line 1219 |
| deeplive_enhanced_fake | `min(0.95, p["quality_p"] + 0.22) = 0.82` | line 1251 |
| visomaster_fake | `min(0.92, p["quality_p"] + 0.12) = 0.72` | line 1280 |
| visomaster_enhanced_fake | `min(0.96, p["quality_p"] + 0.24) = 0.84` | line 1318 |
| df40_real | `max(0.20, real_noise_p + 0.05) = 0.30` (vcd_targeted: real_noise_p=0.25) | line 1360 |
| realpool_real, external_real | `min(0.85, p["quality_p"] + 0.04) = 0.64` | line 1394 |

This is the **largest single asymmetry**. Fake families see the degrade
group fire with p=0.60–0.84; df40_real sees it with p=0.30. Fake samples
receive heavy degradation roughly **2.0× to 2.8× more often** than df40
reals. Even the realpool/external real lane (0.64) is below all fakes.

#### 2.1.b JPEG quality floor
The `quality_lower` (lower bound on JPEG quality, lower = more
compression) varies by family:

| Family | quality_lower | Source line |
|---|---|---|
| df40_fake (balanced_degrade) | `p["jpeg_lower"] = 40` | line 1152 |
| deeplive_enhanced_fake (heavier) | `max(30, p["jpeg_lower"] - 10) = 30` (and a 2nd ImageCompression with floor 26) | lines 1237, 1242 |
| visomaster_fake | `max(36, p["jpeg_lower"] - 8) = 36` | line 1267 |
| visomaster_enhanced_fake | `max(28, p["jpeg_lower"] - 12) = 28` (and a 2nd one with floor 24) | lines 1304, 1309 |
| df40_real | `max(58 if no uplift, p["jpeg_lower"]) = 58` (vcd_targeted) | line 1352 |
| realpool/external_real | `max(52 if no uplift, p["jpeg_lower"]) = 52` | line 1381 |

So the JPEG floor for reals (52–58) is **~12–34 quality points HIGHER** (i.e.,
better quality) than for fakes (24–40). df40_real's compressed sub-arm
caps quality at floor 58 with upper 95; visomaster_enhanced_fake caps at
floor 24 with upper 68 in its second chained compression. This is a
direct, hard quality split.

#### 2.1.c Sharpen (IAASharpen) probability
| Family | sharpen p | alpha | Source line |
|---|---|---|---|
| df40_fake | `0.26` | `p["sharpen_alpha_balanced"]=(0.24,0.60)` | line 1202 |
| deeplive_non_enhanced_fake | `0.22` | balanced | line 1220 |
| deeplive_enhanced_fake | `0.08` | `(0.08, 0.22)` (very mild) | line 1255 |
| visomaster_fake | `0.14` | `(0.10, 0.30)` | line 1284 |
| visomaster_enhanced_fake | `0.06` | `(0.06, 0.18)` (essentially off) | line 1322 |
| df40_real | `real_sharpen_p = 0.60` | `p["sharpen_alpha_real"]=(0.30,0.70)` | line 1346 |
| realpool/external_real | `max(0.18, 0.60 * 0.4) = 0.24` | `(0.15, 0.38)` | line 1398 |

Reals see sharpening at p=0.60 (df40) or p=0.24 (realpool/external). Fakes
top out at p=0.26. df40_real is sharpened **~2.3× more often than
df40_fake and ~10× more than visomaster_enhanced_fake**, and uses a
stronger alpha range (0.30–0.70 vs 0.06–0.30). This pushes reals toward
sharper textures while leaving fakes unsharpened.

#### 2.1.d Other asymmetric knobs

**Downscale.** vcd_targeted preset has `downscale_min=0.50,
downscale_max=0.80` (line 952–953). Inside the family pipelines:
- df40_fake: balanced_degrade uses preset values directly (line 1155).
- deeplive_enhanced_fake: `scale_min = max(0.35, 0.50-0.10) = 0.40`,
  `scale_max = max(0.72, 0.80-0.06) = 0.74` (line 1230) — more aggressive.
- visomaster_fake: `scale_min = max(0.45, 0.50-0.08) = 0.45`,
  `scale_max = max(0.74, 0.80-0.04) = 0.76` (line 1273).
- visomaster_enhanced_fake: `scale_min=0.40`, `scale_max=0.72` (line 1297).
- df40_real: NO downscale arm. The `OneOf` for df40_real (lines 1349–1361)
  is JPEG / GaussNoise(weak) / GaussNoise(real_noise_var). No downscale.
- realpool_real / external_real: `scale_min = max(0.62, 0.50) = 0.62`,
  `scale_max = max(0.90, 0.80) = 0.90` (line 1387) — **floored higher
  than fakes**, so reals never see the bottom of the downscale range.

**Fakes see scale floors as low as 0.35–0.45; reals never go below 0.62
(realpool/external) or have zero downscale (df40_real).**

**Blur.** Blur appears only inside `balanced_degrade` (line 1153, `blur_limit
= p["blur_limit"]=(3,9)` for vcd_targeted), or in expanded form inside the
enhanced-fake pipelines (line 1246: `(3, max(7, 9+2)) = (3,11)`; line 1313:
`(3, max(9, 9+4)) = (3,13)`). df40_real has no blur in its OneOf; realpool
caps blur at `min(7, p["blur_limit"][1]) = (3,7)` (line 1385). So reals see
narrower, weaker blur ranges than enhanced fakes.

**GaussNoise (var_limit).**
- df40_fake balanced_degrade: `p["noise_var"]=(10.0,45.0)` (line 1154).
- deeplive_enhanced_fake also adds an extra `GaussNoise(var_limit*1.15, p=0.28)` (line 1252).
- visomaster_enhanced_fake adds extra `GaussNoise(var_limit*1.20, p=0.30)` (line 1319).
- df40_real OneOf includes `GaussNoise(var_limit=(3.0, max(12, var_high*0.6)), p=1.0)` (line 1356) — capped at half the fake range.
- realpool/external_real: `var_limit=(4.0, max(18, var_high*0.8))` (line 1392).
- df40_real also gets a SECOND noise sub-arm at `var_limit=real_noise_var=(5,20)` (line 1358) when vcd_targeted preset is in use.

Reals see noise variance bounded at ~12–20 max; fakes can get up to
`45 * 1.20 = 54.0`. Fakes routinely receive ~3× higher noise variance.

**Color block.** Identical p, brightness, contrast across all families
(`color_p=0.52, color_brightness=0.22, color_contrast=0.22, hue_shift=20,
sat_shift=24, val_shift=24` for vcd_targeted, lines 957–1001). This is
the only block that is symmetric in `family_aware` mode.

**Context variation.** When `context_variation_enabled: true`, the same
`_build_context_variation_block(p)` is invoked for ALL families
(lines 1204, 1222, 1254, 1283, 1321, 1348, 1397). Symmetric across labels.

**fake_extra_degrade injection.** vcd_targeted has
`fake_extra_degrade_p=0.15` (line 973). For df40_fake ONLY, an extra
heavy degradation `OneOf(Downscale(0.35–0.55), GaussianBlur(5,11), p=0.15)`
is inserted at index 2 of the steps list (line 1208–1212). This is **only
applied to df40_fake; never to any real family**. See §4 below.

#### 2.1.e Asymmetry summary table (vcd_targeted)

| Axis | Fake (worst-case enhanced) | Real (df40_real) | Ratio |
|---|---|---|---|
| OneOf-degrade outer p | 0.84 | 0.30 | 2.8× |
| JPEG quality_lower (lower=worse) | 24 | 58 | fakes -34 pts |
| Sharpen p | 0.06 | 0.60 | reals 10× |
| Sharpen alpha | (0.06, 0.18) | (0.30, 0.70) | reals ~5× stronger |
| Downscale scale_min | 0.35 | (no downscale arm) | reals never downscaled |
| GaussNoise max var | 54.0 | 12.0 | fakes 4.5× |
| fake_extra_degrade extra heavy block | p=0.15 | absent | fakes only |

This is a substantial label-conditional augmentation gap.
**The hypothesis that the asymmetric router installs a quality shortcut
is structurally consistent with the code as written.**

---

## 3. The symmetric branch (`routing.mode == "symmetric"`)

The symmetric branch was added in commit `2c9778b` on 2026-04-26 (today)
in `_build_symmetric_quality_pipeline` (line 1421) and the
`if routing_mode == "symmetric":` arm of the router constructor
(lines 1552–1583).

### 3.1 The symmetric pipeline

`_build_symmetric_quality_pipeline(p)` returns **a single `A.Compose`**
(lines 1472–1483):

```
A.HorizontalFlip(p=0.5)
A.OneOf(balanced_degrade, p=p["quality_p"])
color_block
*context_variation
A.IAASharpen(alpha=sharpen_alpha_balanced, lightness=(0.6, 1.0), p=symmetric_sharpen_p)
webcam_codec_step  (= VideoCodecSimulation, p=p["webcam_codec_p"])
```

Where `balanced_degrade` is identical to the `family_aware` version
(lines 1431–1443), `webcam_codec_step` is identical (lines 1445–1448),
`color_block` is identical (lines 1450–1465), and
`context_variation` reuses `_build_context_variation_block` (line 1467).
The sharpen block uses `symmetric_sharpen_p` (default 0.25, line 1469)
and `sharpen_alpha_balanced` (line 1470).

### 3.2 Pipeline-instance sharing

In the constructor (lines 1558–1581), **a single `sym_pipeline` object is
built once and the same reference is stored in every non-Teams family
key**. This is genuine sharing — both `_pipelines["df40_fake"]` and
`_pipelines["df40_real"]` are `is`-identical to `sym_pipeline`. `_fallback`
is also `sym_pipeline` (line 1583), so unknown_fake / unknown_real /
wma_failure_fake all use the same compose.

### 3.3 Conditional branches that touch class indicators in symmetric mode

I searched every `is_fake`, `label`, `family`, and class-derived flag in
`pipelines.py` (`grep -n "is_fake|class_indicator|label_id|family_key"`):

- Line 1198, 1215, 1227, 1260, 1289, 1343, 1375 — these
  `if family_key == "..."` branches are inside
  `_build_family_quality_pipeline`, which is **not called** in symmetric
  mode (line 1552 short-circuits to `_build_symmetric_quality_pipeline`).
- Line 1683–1701 in `__call__` — `infer_family_key` is still invoked, but
  in symmetric mode every non-teams family resolves to the same
  `sym_pipeline` reference. Teams families resolve to `teams_passthrough`.
- Line 1707, 1712 — `_maybe_apply_teams_sim` checks
  `family_key in self._teams_sim_exclude` and may dispatch
  `apply_for_family(image, family_key=family_key)` (more on this below).

**Inside `_build_symmetric_quality_pipeline` itself there is no
conditional on label, source, method, or family.** Every parameter comes
from the merged `p = preset_overrides ∪ vcd_targeted_preset` dict, which
is set once at router construction and shared across all samples.

### 3.4 So is symmetric truly symmetric?

**For all non-Teams families: YES.** Every non-Teams sample sees the same
`A.Compose` with the same probabilities. Identical augmentation
distribution for label=0 and label=1. There is no `is_fake` toggle, no
per-class JPEG floor, no real-side bonus sharpen, no fake-side extra
heavy block.

**For Teams families: SEPARATE** — see §7 below. Teams families bypass to
`teams_passthrough`, which is internally label-symmetric (it doesn't
inspect label), but **non-Teams vs Teams samples receive different
distributions** in both modes. The symmetric branch does NOT collapse
Teams into the same pipeline.

**There is one potential leak still worth flagging.** See §11.

---

## 4. `fake_extra_degrade` injection

Defined at lines 1195, 1208–1212. `fake_extra_degrade_p = p.get("fake_extra_degrade_p", 0.0)`.
For `vcd_targeted` preset this is `0.15` (line 973); for light/moderate/strong
it is `0.0` (lines 857, 898, 939).

The injection is INSIDE `_build_family_quality_pipeline`, gated on
`if family_key == "df40_fake":` (line 1198) and only fired for that one
family. It inserts an extra `OneOf([Downscale(0.35–0.55),
GaussianBlur(5,11)], p=fake_extra_degrade_p)` at position 2 of the
df40_fake steps list. Real families never reach this code path.

**In symmetric mode this code is dead.** `_build_family_quality_pipeline`
is not invoked when `routing_mode == "symmetric"` (router constructor
short-circuits at line 1552). `fake_extra_degrade` is therefore not
applied in symmetric mode at all — neither to fakes nor to reals. This
confirms one of the asymmetry sources is truly removed in the new branch.

---

## 5. `webcam_codec_step` (label symmetry?)

Defined twice — once for `_build_family_quality_pipeline` (line 1167) and
once for `_build_symmetric_quality_pipeline` (line 1445). Both are
identical:

```python
webcam_codec_step = VideoCodecSimulation(
    codec_quality=p.get("webcam_codec_quality", (30, 80)),
    p=p.get("webcam_codec_p", 0.0),
)
```

The transform itself is `data/augmentations/transforms.py:1216
class VideoCodecSimulation(ImageOnlyTransform)`. It samples a global
codec_quality per image, then chains: resolution-reduction → bilateral
deblock → block quantization noise → frequency-shaped codec noise → JPEG
re-encode. **It does not inspect label, family, or any class indicator.**
All randomness comes from `random.uniform/randint` and `np.random.normal`
on per-image samples.

**`webcam_codec_step` is label-symmetric in both `family_aware` and
`symmetric` modes.** Within each mode it is appended uniformly to every
non-teams family pipeline (lines 1205, 1223, 1256, 1285, 1323, 1362, 1399
in family_aware; line 1481 in symmetric).

For vcd_targeted preset, `webcam_codec_p = 0.12` (line 962); R13_P10_*
yamls override this to `webcam_codec_p: 0.35` (e.g., `R13_P10_SYM_baseline.yaml:85`).

---

## 6. `teams_codec_sim` (label symmetry?)

Two distinct things share this name family. Be careful.

### 6.1 `teams_codec_sim_p` inside `_build_teams_passthrough_pipeline`

Defined in `_TEAMS_PASSTHROUGH_DEFAULTS` (line 810: `"teams_codec_sim_p": 0.0`,
line 811: `"teams_codec_sim_quality": (30, 75)`).

The actual injection is at `_build_teams_passthrough_pipeline` (line 1486),
specifically lines 1507–1515:

```python
teams_codec_sim_p = float(p.get("teams_codec_sim_p", 0.0) or 0.0)
extra: list = []
if teams_codec_sim_p > 0:
    extra.append(
        VideoCodecSimulation(
            codec_quality=p.get("teams_codec_sim_quality", (30, 75)),
            p=teams_codec_sim_p,
        )
    )

return A.Compose([
    A.HorizontalFlip(p=p.get("teams_passthrough_flip_p", 0.5)),
    *extra,
    A.RandomBrightnessContrast(...),
    *_build_teams_passthrough_special_block(p),
])
```

The transform inspected is `VideoCodecSimulation` — the same one used in
the non-Teams routes. It does NOT inspect label / family / class. Inside
the Teams passthrough branch, the same `A.Compose` instance is shared
across `deeplive_teams_fake`, `deeplive_teams_real`,
`visomaster_hints_teams_fake|real`, `proper_visomaster_teams_fake`,
`proper_visomaster_enhanced_teams_fake`, `proper_real_teams` (lines 1574–1580
in symmetric, 1603–1609 in family_aware). So:

**`teams_codec_sim_p` is label-symmetric within Teams families.**

R13_P10_* yamls set `teams_codec_sim_p: 0.40, teams_codec_sim_quality: [20, 65]`
(e.g., `R13_P10_SYM_baseline.yaml:91-92`). With p=0.40, ~40% of Teams-passthrough
frames (real and fake alike) get an extra `VideoCodecSimulation` pass on top of
their existing Teams codec fingerprint. Motivation: per the docstring at
line 1500–1503, this targets the dor_shkedi vs real_dor pipeline-signature
shortcut found in `analysis/dor_pool_fingerprint_diff_2026-04-24`.

### 6.2 `teams_codec_simulation` (post-pipeline TeamsCodecSimulation)

Separate, opt-in mechanism at lines 1612–1714. Driven by a top-level
`augmentation.teams_codec_simulation` YAML block (NOT by preset overrides).
When enabled, after the per-family pipeline runs, the router rolls a
Bernoulli(p) and may apply one of `TeamsCodecSimulation`,
`TeamsAdaptiveCodecSimulation`, or `TeamsHybridCodecSimulation` (chosen by
`policy`).

`exclude_families` (line 1707) lets the caller skip Teams-passthrough
families to avoid double-applying. **No R13_P10_* yaml currently sets this
block** — they all use the cheaper inline `teams_codec_sim_p` knob in §6.1.

**Within each family, `teams_codec_simulation` is label-symmetric** — the
roll is per-image, family-keyed, and label is not inspected directly.
However, `TeamsAdaptiveCodecSimulation` and `TeamsHybridCodecSimulation`
have an `enhanced_families` parameter (lines 1638, 1657) that splits
behavior by family bucket (e.g., applies a different mode probability for
`visomaster_enhanced_fake` vs others). Since "enhanced" buckets are
**fake-only** (no `*_enhanced_real` family exists in the router), this
constitutes a per-family — and effectively per-class — branch when the
adaptive/hybrid policy is selected. Not active in any current P10 yaml,
but worth noting.

---

## 7. Teams-passthrough handling

### 7.1 What augmentations Teams families receive

`_build_teams_passthrough_pipeline(p)` (line 1486) composes:
- `A.HorizontalFlip(p=teams_passthrough_flip_p, default 0.5)` (line 1518).
- Optional `VideoCodecSimulation(p=teams_codec_sim_p)` if
  `teams_codec_sim_p > 0` (lines 1507–1515).
- `A.RandomBrightnessContrast(brightness_limit=teams_passthrough_brightness_limit
  default 0.08, contrast_limit=0.08, p=teams_passthrough_brightness_contrast_p
  default 0.3)` (lines 1520–1524).
- Optional special block via `_build_teams_passthrough_special_block(p)`
  (line 1525) — opt-in only, controlled by
  `teams_passthrough_special_aug_enabled` (default False, line 794).

### 7.2 Teams passthrough symmetry

The Teams passthrough pipeline does **not** inspect label, source, or
method. It is built once per router instance and the same `A.Compose`
reference is stored under all 7 Teams family keys (lines 1574–1580 in
symmetric, 1603–1609 in family_aware). So **Teams passthrough is
internally label-symmetric.**

### 7.3 Teams vs non-Teams asymmetry

Even in `routing_mode: symmetric`, **Teams samples receive a much lighter
augmentation** (essentially flip + ±8% brightness/contrast +
optional codec sim) than non-Teams samples (flip + balanced_degrade OneOf
+ color block + context variation + sharpen + codec sim). This is by
design — the docstring at lines 1487–1503 argues that Teams data has
already been through the real codec pipeline and shouldn't be
re-degraded. **But it does mean the train-time augmentation distribution
splits along a Teams/non-Teams axis even in symmetric mode.**
(Whether this is a concern depends on whether Teams sample mix is
balanced across labels — see WT-B, ID-balanced sampling configuration.
The R13_P10_* sampling weights at lines 217–229 of the SYM yamls have
`deeplive_teams_fake: 5.0, deeplive_teams_real: 4.0, df40_real: 0.4` etc.,
so Teams is roughly balanced label-wise.)

---

## 8. Strategy-driven routing for DeepLive

`include_strategies` is a YAML knob under `combined_paired.deeplive`
(R13_P10_SYM_baseline.yaml lines 124–129). The values cited in the audit
prompt — `edge_cases_enhanced`, `minimal_processing_enhanced` — control
which DeepLive method buckets are loaded. They land in
`enhanced_strategy_names` (router constructor line 1536, default
`_DEFAULT_ENHANCED_STRATEGIES = ("quality_enhancement",
"edge_cases_enhanced", "minimal_processing_enhanced")` at lines 1004–1008).

The dispatch is at `infer_family_key` (`utils/grouping.py:250`), which
calls `infer_group_key` (line 111). DeepLive samples have method strings
like `deeplive_edge_cases`, `deeplive_edge_cases_enhanced`,
`deeplive_minimal_processing_enhanced`, etc. (set in
`combined_paired.py:244` as `f"deeplive_{effective_strategy}"`).

In `infer_group_key`, the strategy is extracted at line 203
(`_extract_deeplive_strategy`). For label=1 (fake):
- `edge_cases` → `deeplive_edge_cases_fake` → family `deeplive_non_enhanced_fake` (grouping.py:281).
- `minimal_processing` → `deeplive_minimal_processing_fake` → family `deeplive_non_enhanced_fake`.
- `edge_cases_enhanced` → `deeplive_edge_cases_enhanced_fake` → family `deeplive_enhanced_fake` (grouping.py:286).
- `minimal_processing_enhanced` → `deeplive_minimal_processing_enhanced_fake` → family `deeplive_enhanced_fake`.
- `quality_enhancement` → `deeplive_quality_enhancement_fake` → family `deeplive_enhanced_fake`.

For label=0 (real), all strategies (including `_enhanced` ones) collapse
to `realpool_real` (grouping.py:336–337). **Real samples loaded from
DeepLive — even from `_enhanced` strategies — go to `realpool_real`,
not to a `deeplive_enhanced_real` bucket** (which doesn't exist).

### Augmentation behavior consequences

- **In `family_aware` mode:** `deeplive_enhanced_fake` (e.g., from
  edge_cases_enhanced strategy) gets the heaviest degradation profile in
  the entire router (OneOf p=0.82, JPEG floor 30 with a chained 26 floor,
  blur up to 11, downscale floor 0.40, near-zero sharpen) (lines 1227–1258).
  But the matched real frame from the same DeepLive sample goes to
  `realpool_real`, which gets a much lighter pipeline (OneOf p=0.64,
  JPEG floor 52, downscale floor 0.62, sharpen p=0.24).

  This is the strategy-driven asymmetry: a sample's `include_strategies`
  membership controls how heavily its **fake** frames are degraded, while
  its **real** frames always get the realpool treatment.

- **In `symmetric` mode:** strategy is irrelevant for augmentation choice
  (every non-teams family resolves to `sym_pipeline`). `enhanced_strategy_names`
  is still threaded in (router line 1536) — only because `infer_family_key`
  consumes it to pick a `family_key`, but in symmetric mode all those
  family_keys map to the same pipeline. The strategy field is essentially a
  no-op for augmentation in symmetric mode. **POTENTIAL LEAK clarification:**
  it remains a no-op as long as `_pipelines.get(family_key)` returns
  `sym_pipeline` for every key, which is true given lines 1561–1573.

---

## 9. Color jitter, geometric, lighting stress (training vs eval)

The labels mentioned in the prompt — `vcd_targeted_stress`,
`backlight_dim_stress`, `warm_harsh_stress`, `crop_shift`, `scale`,
`rotation` — are **NOT used in training augmentation**. They are
deterministic eval-time stress presets defined at lines 1747–1912:

```python
EVAL_STRESS_PRESETS = {
    "vcd_targeted_stress",
    "backlight_dim_stress",
    "warm_harsh_stress",
    "crop_shift",
    "scale",
    "rotation",
}
```
(lines 1765–1772)

`apply_eval_stress_preset(img_np, preset_name, seed=42)` (line 1885) seeds
random/numpy, builds an `A.Compose` with collapsed `(min, max)` ranges
(every aug fires deterministically with fixed parameters at p=1.0), and
returns the stressed image. The presets are referenced in
`combined_paired.py:2044, 2055` and consumed via the `eval_augmentation`
field on OOD validation sources (e.g., R13_P10_SYM_baseline.yaml lines
283, 291, 299, 309, 317, 325).

**These presets only apply to `lighting_stress_sources` and
`spatial_stress_sources` under `ood_monitoring.*` in the YAML. Training
loaders (combined_paired) never invoke `apply_eval_stress_preset`.**

For training-time color/geometric/lighting nuisance:
- Color jitter (`RandomBrightnessContrast`, `HueSaturationValue`) lives
  inside `color_block` and is applied to ALL non-Teams families
  symmetrically (line 1172 family_aware, line 1450 symmetric).
- Geometric jitter (`ShiftScaleRotate`) only enters via
  `_build_context_variation_block` (line 1037) — which fires only when
  `context_variation_enabled: true`. R13_P10_* yamls set this true (e.g.,
  SYM_baseline.yaml line 93).
- Lighting (gamma, CCT, directional shadow, gamma-up) all live in
  `_build_context_variation_block` (lines 1027, 1052, 1064, 1077). All
  applied uniformly across families.

---

## 10. Git history of routing changes

`git log --all --pretty=format:"%h %ai %s" -- DeepfakeBench/training/data/augmentations/pipelines.py` shows 7 commits (file is in this path; pre-2026-03-19 history is from
the upstream import and irrelevant to this audit):

| Hash | Date | Subject |
|---|---|---|
| 2c9778b | 2026-04-26 | **Add P10 anti-shortcut packet: symmetric router + GRL slate** |
| ebce585 | 2026-04-25 | Add real_codec_uplift flag to family-aware aug router |
| c6f1034 | 2026-04-22 | Land A1-A10 OOD instrumentation and R13 Packet 3 yamls |
| 77facfc | 2026-04-19 | Add proper-data runtime and relaunch review packet |
| f9303eb | 2026-04-19 | Land WT-B runtime and launcher smoke readiness |
| 47cea27 | 2026-04-17 | WT-C add opt-in Teams special augmentation |
| 6661bbb | 2026-04-17 | WT-C truthful gamma-up sidecars |

Pickaxe searches:
- `git log -S "QualityTargetedFamilyRouter"` → first appears in
  `ba36067 2026-03-19` (the initial repo consolidation that re-imported
  the trainer / augmentation tree). The `family_aware`-style router was
  effectively "always there" once the repo was consolidated.
- `git log -S "routing_mode"` → introduced in `c6f1034 2026-04-22 Land
  A1-A10 OOD instrumentation`. Before that commit there was no
  `routing_mode` parameter at all; `family_aware` behavior was
  unconditional.
- `git log -S "symmetric"` → the literal string `"symmetric"` first
  appears in the routing-mode sense in `2c9778b 2026-04-26` (today).
  The earlier match (`6661bbb 2026-04-17 WT-C truthful gamma-up sidecars`)
  is the bare word "symmetric" in unrelated comments
  ("symmetrically to x and y", "asymmetric brightness biased upward").

**Concretely: asymmetric (family-aware) routing has been the de-facto
behavior since the repo was consolidated on 2026-03-19; the
`routing_mode` parameter became the explicit switch on 2026-04-22 (still
defaulted to `family_aware`); the `symmetric` branch was added today
(2026-04-26) as part of the P10 anti-shortcut packet.**

Diff for the symmetric add (commit 2c9778b) introduces only:
- `_build_symmetric_quality_pipeline(p)` (a +65-line block above
  `_build_teams_passthrough_pipeline`).
- A new `if routing_mode == "symmetric":` arm at line 1552 inside the
  router constructor.
- `if self.routing_mode != "family_aware":` was widened to
  `if self.routing_mode not in ("family_aware", "symmetric"):` at line 1686.

The `family_aware` arm and `_build_family_quality_pipeline` were not
modified. The symmetric branch is purely additive.

---

## 11. CRITICAL — silent leaks

Exhaustive audit of every conditional that touches a class indicator
within `routing_mode == "symmetric"`:

### 11.1 Confirmed clean

- `_build_symmetric_quality_pipeline` body (lines 1421–1483): no `if`
  on label/family/method/source. Every parameter is from `p` (the
  preset+overrides dict).
- `_build_context_variation_block` (lines 1011–1083): branches only on
  `p.get("context_variation_enabled")`, `cct_p`, `shadow_p`,
  `gamma_up_p`. None depend on label.
- `webcam_codec_step` (`VideoCodecSimulation` at transforms.py:1216):
  per-image RNG only, no class inspection.
- `_build_teams_passthrough_pipeline` (lines 1486–1526): no class
  inspection; same compose used for `*_teams_real` and `*_teams_fake`.
- `_build_teams_passthrough_special_block` (lines 1086–1145): branches on
  `*_aug_enabled`, `shift_p`, `cct_p`, `shadow_p`, `gamma_up_p` — all
  preset-level.
- The constructor's `_pipelines` dict (lines 1560–1581): all 13 non-Teams
  family keys map to the same `sym_pipeline` reference (`is`-identical
  object). All 7 Teams family keys map to the same `teams_passthrough`
  reference.

### 11.2 POTENTIAL LEAK 1 — `_maybe_apply_teams_sim` family-keyed dispatch

Lines 1703–1714. When `teams_codec_simulation.enabled = true` AND the
selected policy is `adaptive_mixture` or `family_split`, the call
delegates to `apply_for_family(image, family_key=family_key)` (line 1712),
which inspects `family_key` against `enhanced_families` (default tuple at
lines 1639–1643). Since "enhanced" families are fake-only and there is no
matching `*_enhanced_real` family, **the adaptive/hybrid policy
effectively conditions augmentation on a fake-side bucket**, which leaks
class information.

This is **inactive in all current R13_P10_* yamls** — none of the SYM/GRL
yamls define a `teams_codec_simulation` block. So in practice it does not
fire. But the leak channel exists and must not be enabled together with
`routing_mode: symmetric` without re-auditing.

### 11.3 POTENTIAL LEAK 2 — Teams vs non-Teams pipeline split

Even in symmetric mode, Teams families (`*_teams_*`) get
`teams_passthrough` while non-Teams families get `sym_pipeline`
(lines 1574–1580 vs 1561–1573). The two pipelines differ substantially
(passthrough has flip + brightness/contrast + optional codec sim only;
sym has the full balanced_degrade OneOf + sharpen + webcam codec at
p=0.35).

Within each Teams family the pipeline is label-symmetric. But the
training distribution conditional on `(label=fake, source=teams)`
versus `(label=fake, source=non-teams)` differs. **This is not a label
leak per se, but a source-conditional branch.** If the model can detect
"is this sample a Teams sample" from the image content (and Teams is a
specific codec fingerprint, so it almost certainly can), and if the Teams
training subset has a different real/fake ratio than the non-Teams
training subset, then a quality-shortcut can re-emerge through this
path. The R13_P10_* sampling weights (e.g., SYM_baseline.yaml lines
217–229) have `deeplive_teams_fake: 5.0, deeplive_teams_real: 4.0` —
slightly fake-heavy on the Teams branch. This is unlikely to be the
primary shortcut source but **is a residual data-distribution coupling
between source and label that the symmetric pipeline does not address.**

### 11.4 POTENTIAL LEAK 3 — `infer_family_key` continues to consume `enhanced_strategy_names`

In symmetric mode, `infer_family_key` is still called per-sample
(line 1693) with `enhanced_strategy_names`, even though every non-Teams
family_key resolves to the same `sym_pipeline`. There is no behavior
difference, but if a future change makes the symmetric `_pipelines` dict
contain a non-shared object for some key (e.g., a typo, or an attempt at
a sub-experiment that mutates the dict), the per-class divergence would
silently re-appear. No leak today. **Risk: structural fragility, not
active leak.**

### 11.5 POTENTIAL LEAK 4 — RNG seeding shared across calls

`A.Compose` does not pin per-image RNG state, and `VideoCodecSimulation`
uses module-level `random` and `np.random` (transforms.py:1287–1396).
This is normal per-image randomness; no class-conditional behavior.
**No leak.**

### 11.6 Outside the symmetric path (just for completeness)

In `routing_mode == "family_aware"`, EVERY axis enumerated in §2 is a
class-conditional branch. They are all in `_build_family_quality_pipeline`
(lines 1198, 1215, 1227, 1260, 1289, 1343, 1375), which is **not invoked
in symmetric mode**.

---

## 12. Config-side augmentation knobs

`_create_combined_transform` in `combined_paired.py:4685` reads
`config['augmentation']` and dispatches by `augmentation.version`. For
`version: quality_targeted_family`, the recognised YAML keys are:

### Top-level (NOT preset overrides)
| Key | Effect | Code |
|---|---|---|
| `augmentation.version` | Pipeline factory selector. `"quality_targeted_family"` → router. Other values: `quality_robust(_light/_moderate/_strong)`, integer 3-7, `surgical`, `general`, `webcam_codec`, etc. | combined_paired.py:4716 |
| `augmentation.strength` | Preset key — `light`, `moderate`, `strong`, `vcd_targeted` | combined_paired.py:4759, pipelines.py:1540 |
| `augmentation.routing.mode` | `family_aware` (default) or `symmetric`. Anything else falls back to `create_quality_robust_pipeline(strength)` | combined_paired.py:4760, pipelines.py:1546, 1686 |
| `augmentation.routing.enhanced_strategy_names` | Tuple of DeepLive strategy strings → `enhanced` family bucket. Default `("quality_enhancement", "edge_cases_enhanced", "minimal_processing_enhanced")` | combined_paired.py:4761, pipelines.py:1004 |
| `augmentation.teams_codec_simulation.enabled` | Activates post-pipeline `TeamsCodecSimulation` step | pipelines.py:1618 |
| `augmentation.teams_codec_simulation.probability` | Bernoulli p for the post-pipeline step (default 0.15) | pipelines.py:1619 |
| `augmentation.teams_codec_simulation.policy` | `legacy_single` / `adaptive_mixture` / `family_split` | pipelines.py:1621–1677 |
| `augmentation.teams_codec_simulation.exclude_families` | List of family keys to skip | pipelines.py:1620 |
| `augmentation.teams_codec_simulation.enhanced_families` | Override the enhanced-family tuple for adaptive/hybrid | pipelines.py:1638, 1657 |
| `augmentation.teams_codec_simulation.ordinary_mode_probability_non_enhanced` | Mode-mix probability | pipelines.py:1645, 1664 |
| `augmentation.teams_codec_simulation.ordinary_mode_probability_enhanced` | Mode-mix probability for enhanced | pipelines.py:1648, 1667 |

### Preset-level overrides (anything that is a key in the chosen preset)
The factory at combined_paired.py:4769–4775 collects all `aug_config`
keys other than `version`, `strength`, `routing` that match a key in the
first preset dict (`_QUALITY_TARGETED_PRESETS["light"].keys()`). Effective
preset is `_QUALITY_TARGETED_PRESETS[strength] ∪ overrides`.

The full set of accepted preset keys (mirroring the `light` preset at
lines 815–859 — this is the schema that `_valid_preset_keys` is built
from):

| Key | Effect |
|---|---|
| `jpeg_lower`, `jpeg_upper` | Bounds for `A.ImageCompression` quality. Lower = more compression. |
| `blur_limit` | Tuple kernel range for `A.GaussianBlur`. |
| `noise_var` | Tuple var_limit for `A.GaussNoise`. |
| `downscale_min`, `downscale_max` | Bounds for `A.Downscale`. |
| `quality_p` | Outer `A.OneOf` probability for the balanced_degrade group (REAL effect: gates how often any quality augmentation fires; family_aware ADDS per-family offsets to this; symmetric uses it as-is). |
| `color_p` | `color_block` `A.OneOf` p. |
| `color_brightness`, `color_contrast` | Bounds for `A.RandomBrightnessContrast`. |
| `hue_shift`, `sat_shift`, `val_shift` | `A.HueSaturationValue` shift limits. |
| `webcam_codec_p`, `webcam_codec_quality` | Probability and quality for `VideoCodecSimulation` (always added at the end). |
| `sharpen_alpha_balanced` | `IAASharpen.alpha` for the symmetric pipeline AND for fake-family pipelines in family_aware. |
| `sharpen_alpha_real` | `IAASharpen.alpha` used by `df40_real` only in family_aware. (Has no effect in symmetric mode — symmetric uses `sharpen_alpha_balanced`.) |
| `real_noise_p`, `real_noise_var` | df40_real / external_real extra noise injection in family_aware. (No effect in symmetric.) |
| `real_sharpen_p` | Sharpen probability for real-family routes in family_aware. (No effect in symmetric — symmetric uses `symmetric_sharpen_p`.) |
| `fake_extra_degrade_p` | df40_fake heavy degradation injection in family_aware. (No effect in symmetric.) |
| `context_variation_enabled` | Master switch for `_build_context_variation_block`. |
| `context_variation_gamma_limit` | `A.RandomGamma` gamma_limit. |
| `context_variation_brightness`, `context_variation_contrast` | `A.RandomBrightnessContrast` inside context block. |
| `context_variation_shift`, `context_variation_scale`, `context_variation_rotate` | `A.ShiftScaleRotate` parameters. |
| `context_variation_oneof_p` | Kept for back-compat; **ignored** since R12 (line 1019). |
| `context_variation_individual_p` | Per-transform probability inside context block (post-R12). |
| `context_variation_cct_p`, `context_variation_cct_range` | `ColorTemperatureShift`. |
| `context_variation_shadow_p`, `context_variation_shadow_intensity`, `context_variation_shadow_softness` | `DirectionalShadow`. |
| `context_variation_gamma_up_p`, `context_variation_gamma_up_range` | `GammaUp` (always-brighten). |
| `teams_passthrough_flip_p` | Flip probability inside Teams passthrough. |
| `teams_passthrough_brightness_contrast_p`, `teams_passthrough_brightness_limit`, `teams_passthrough_contrast_limit` | Light Teams passthrough color jitter. |
| `teams_passthrough_special_aug_enabled` (default False) | Master switch for Teams passthrough special block. |
| `teams_passthrough_special_shift_p`, `teams_passthrough_special_shift`, `teams_passthrough_special_scale`, `teams_passthrough_special_rotate` | ShiftScaleRotate inside Teams special block. |
| `teams_passthrough_special_cct_p`, `teams_passthrough_special_cct_range` | CCT inside Teams special. |
| `teams_passthrough_special_shadow_p`, `teams_passthrough_special_shadow_intensity`, `teams_passthrough_special_shadow_softness` | DirectionalShadow inside Teams special. |
| `teams_passthrough_special_gamma_up_p`, `teams_passthrough_special_gamma_up_range` | GammaUp inside Teams special. |
| `teams_codec_sim_p`, `teams_codec_sim_quality` | INLINE `VideoCodecSimulation` inside Teams passthrough (NOT the same as `teams_codec_simulation` post-pipeline knob in §6.2). |
| `real_codec_uplift`, `real_codec_uplift_floor`, `real_codec_uplift_chain_p` | df40_real / external_real codec uplift in family_aware (added in commit ebce585 2026-04-25). No effect in symmetric. |

### Note on key validation

The factory at combined_paired.py:4773 accepts only keys that are in
`_valid_preset_keys`. Unknown keys are silently dropped (no warning).
Any typo in a preset override is therefore a silent no-op. There is
also no warning when the user sets an override that is real-side-only
or fake-side-only and is consequently inert in symmetric mode (e.g.,
`real_sharpen_p`, `fake_extra_degrade_p`). Authors of P10 yamls should
be aware that those keys are dead in symmetric mode.

### Observed P10 overrides (R13_P10_SYM_baseline.yaml lines 76–93)

```yaml
augmentation:
  version: "quality_targeted_family"
  strength: "vcd_targeted"
  routing:
    mode: "symmetric"
    enhanced_strategy_names: [quality_enhancement, edge_cases_enhanced, minimal_processing_enhanced]
  webcam_codec_p: 0.35           # vcd_targeted default 0.12 → 0.35
  webcam_codec_quality: [20, 65] # vcd_targeted default (35, 80) → (20, 65)
  quality_p: 0.60                # vcd_targeted default 0.60 (no-op)
  jpeg_lower: 40                 # vcd_targeted default 40 (no-op)
  downscale_min: 0.50            # vcd_targeted default 0.50 (no-op)
  downscale_max: 0.80            # vcd_targeted default 0.80 (no-op)
  teams_codec_sim_p: 0.40        # default 0.0 → 0.40 (Teams passthrough VideoCodecSimulation)
  teams_codec_sim_quality: [20, 65] # default (30, 75) → (20, 65)
  context_variation_enabled: true # default true for vcd_targeted (no-op for vcd_targeted)
```

P10_GRL_baseline yaml is identical apart from `routing.mode: family_aware`,
`quality_p: 0.72`, `jpeg_lower: 30`, `downscale_min: 0.35`. So GRL_baseline
keeps the asymmetric router and pushes the fake-side knobs HARDER (lower
JPEG floor 30 vs 40, more aggressive downscale floor 0.35 vs 0.50,
higher OneOf p 0.72 vs 0.60).

---

## Bottom-line summary for the receiving agent

1. **The asymmetric router is real and substantial.** In `family_aware`
   + `vcd_targeted`: fake families see degrade-OneOf p 0.60–0.84 vs reals
   0.30–0.64; JPEG floor 24–40 (fake) vs 52–58 (real); sharpen p 0.06–0.26
   (fake) vs 0.24–0.60 (real); fakes are downscaled to 0.35–0.45 floor,
   reals are floored at 0.62 or skip downscale entirely; fakes can get a
   `fake_extra_degrade` heavy block at p=0.15. The structural conditions
   for "low quality = fake" being learnable as a free shortcut are
   present.

2. **The symmetric branch (commit 2c9778b, 2026-04-26) is a clean fix
   for the in-pipeline asymmetry.** All 13 non-Teams family_keys point
   to the same `A.Compose` instance; `_build_symmetric_quality_pipeline`
   contains zero label-conditional logic; `fake_extra_degrade`,
   `sharpen_alpha_real`, `real_sharpen_p`, `real_noise_p`,
   `real_codec_uplift` are all dead code in symmetric mode.

3. **Teams families still split out to a separate, lighter
   `teams_passthrough` pipeline in BOTH modes.** Teams data has its real
   codec fingerprint and is intentionally not re-degraded. Within the
   Teams branch, augmentation is label-symmetric. But Teams vs non-Teams
   is a substantial pipeline split that survives `routing.mode: symmetric`.

4. **`teams_codec_sim_p=0.40`** in the P10 yamls puts a coherent
   `VideoCodecSimulation` pass on top of ~40% of Teams-passthrough frames
   (label-symmetric within Teams). Aimed at the dor_shkedi vs real_dor
   pipeline-fingerprint shortcut from analysis/dor_pool_fingerprint_diff_2026-04-24.

5. **EVAL_STRESS_PRESETS (vcd_targeted_stress, backlight_dim_stress, ...)
   are eval-time only.** They never apply during training. They are
   referenced under `ood_monitoring.lighting_stress_sources` and
   `spatial_stress_sources` only.

6. **Residual coupling worth tracking even with symmetric routing:**
   (a) Teams vs non-Teams pipeline split combined with an unbalanced
   per-source label distribution; (b) `teams_codec_simulation.policy`
   adaptive/hybrid family-keyed dispatch (currently inactive but enabled
   would re-introduce a class signal via the enhanced_families tuple);
   (c) silent no-op of fake-side / real-side preset overrides in
   symmetric mode (they don't error; they just do nothing).
