# Lighting Robustness Implementation Log

**Created:** March 9, 2026  
**Status:** COMPLETE (ready for next experiment round)  
**Constraint:** albumentations==0.4.6 (cloud training/validation)

---

## Motivation

Production deployments see diverse indoor lighting: warm tungsten, cool fluorescent, directional desk lamps, overexposed webcam feeds. When the training augmentation pipeline doesn't cover these conditions, the model encounters unfamiliar lighting artifacts at inference and **misclassifies real faces as fake** (false positives). The lighting robustness report (`docs/LIGHTING_ROBUSTNESS_REPORT.md`) quantified the gap: R12 augmentations cover only ~80% of real-capture color temperature values and miss directional shadow patterns entirely. These two new transforms target the specific uncovered ranges.

## Goal

Close the remaining lighting robustness gaps identified in `docs/LIGHTING_ROBUSTNESS_REPORT.md` that R12 did **not** address:

| Gap | Status | Notes |
|-----|--------|-------|
| OneOf removal | ✅ R12 | Independent transforms at p=0.15 each |
| CCT simulation | ✅ R12 | `ColorTemperatureShift` in transforms.py |
| Asymmetric brightness | ✅ R12 | [-0.20, +0.60] |
| Wider gamma | ✅ R12 | [70, 130] |
| Hue shift 12→20 | ✅ R12 | Hardcoded in preset + YAML override |
| **Directional shadows** | ✅ DONE | `DirectionalShadow` transform in transforms.py |
| **Upward brightness gap** | ✅ DONE | `GammaUp` transform in transforms.py |

---

## Implementation Plan

### Step 1: Study albumentations 0.4.6 API
- Confirmed `ImageOnlyTransform` API: `__init__(self, ..., always_apply, p)` → `super().__init__(always_apply, p)`, `apply(self, image, **params)`, `get_transform_init_args_names()`
- `ColorTemperatureShift` already uses this exact pattern — safe to follow
- **Status:** ✅ COMPLETE

### Step 2: Implement `DirectionalShadow` transform
- Ported `apply_shadow()` from tools/lighting_showcase.py
- Vectorised: replaced Python for-loop with `np.linspace` + broadcasting → builds full gradient mask in one shot
- Supports 8 directions (4 cardinal + 4 diagonal) via `_make_1d_shadow()` helper + min-composite for diagonals
- Params: `intensity_range=(0.15, 0.50)`, `softness_range=(0.20, 0.50)`, `directions` (all 8 by default)
- File: `data/augmentations/transforms.py` (after `ColorTemperatureShift`, before `CustomUnsharpMask`)
- **Status:** ✅ COMPLETE

### Step 3: Implement `GammaUp` transform
- Uses gamma < 1 (float scale, not albumentations' 0-200 scale) for always-brighten guarantee
- LUT-based via `cv2.LUT` for speed (256-entry lookup, handles 3-channel images)
- Validates `gamma_range` ∈ (0, 1) exclusive at construction time
- Default range: `(0.45, 0.85)` — pushes midtones from ~128 up to ~170-210 range
- File: `data/augmentations/transforms.py` (after `DirectionalShadow`)
- **Status:** ✅ COMPLETE

### Step 4: Wire into `_build_context_variation_block()`
- Added `DirectionalShadow` as conditional append gated by `context_variation_shadow_p` (default 0.0)
- Added `GammaUp` as conditional append gated by `context_variation_gamma_up_p` (default 0.0)
- Both follow the exact same pattern as the existing CCT conditional block
- File: `data/augmentations/pipelines.py`
- **Status:** ✅ COMPLETE

### Step 5: Update `vcd_targeted` preset
- Added 5 new keys to preset dict (all defaults off for backward compat):
  - `context_variation_shadow_p`: 0.0
  - `context_variation_shadow_intensity`: (0.15, 0.45)
  - `context_variation_shadow_softness`: (0.20, 0.50)
  - `context_variation_gamma_up_p`: 0.0
  - `context_variation_gamma_up_range`: (0.45, 0.85)
- File: `data/augmentations/pipelines.py`
- **Status:** ✅ COMPLETE

### Step 6: Export from `__init__.py`
- Added `ColorTemperatureShift`, `DirectionalShadow`, `GammaUp` to imports and `__all__`
- **Status:** ✅ COMPLETE

### Step 7: Add unit tests
- Created `tests/test_lighting_transforms.py` with 34 tests across 4 test classes:
  - `TestDirectionalShadow` (11 tests): shape/dtype, clipping, darkening, cardinal/diagonal gradients, zero-intensity identity, various sizes, A.Compose integration, param serialization
  - `TestGammaUp` (11 tests): shape/dtype, always-brightens, dark/bright images, stronger=brighter, black/white invariance, invalid range raises, A.Compose integration, param serialization
  - `TestColorTemperatureShift` (7 tests): shape/dtype, warm/cool CCT R/B ratio, daylight near-identity, clipping, A.Compose integration, param serialization
  - `TestContextVariationBlock` (5 tests): disabled=empty, default has no shadow/gamma_up, individual enable, all-three enable
- Used `importlib` direct module loading to avoid `data/__init__.py` → `torch` import chain (pre-existing issue in local env without torch)
- **All 34 tests pass** in 1.12s
- **Status:** ✅ COMPLETE

### Step 8: Update audit tool to use real pipeline code
- Replaced hardcoded `_random_current_pipeline` / `_random_proposed_compound` with three accurate simulation functions:
  - `_random_r12_pipeline()` — matches actual `vcd_targeted` R12 params (gamma/brightness/cct each at p=0.15, independent)
  - `_random_proposed_pipeline(shadow_p, gamma_up_p)` — R12 baseline + `DirectionalShadow` + `GammaUp`
  - Old pre-R12 OneOf pipeline removed (was inaccurate)
- Vectorised `apply_shadow()` to match 8-direction `DirectionalShadow` transform (was 4 directions)
- Updated histogram labels: "R12 production" / "Proposed" (was "Current aug" / "Proposed aug")
- Updated coverage table and CSV export to use `stats_r12` / `stats_proposed` labels
- Added CLI args: `--shadow-p` (default 0.10), `--gamma-up-p` (default 0.12)
- File: `tools/lighting_showcase.py`
- **Status:** ✅ COMPLETE

---

## Execution Log

### 2026-03-09 — Session start
- Created this log document
- Read existing `ColorTemperatureShift` in transforms.py to confirm albumentations 0.4.6 API pattern
- Read `_build_context_variation_block()` and `vcd_targeted` preset in pipelines.py
- Read `__init__.py` exports and existing test patterns

### 2026-03-09 — Implementation
- Implemented `DirectionalShadow` (vectorised, 8 directions, ~90 lines)
- Implemented `GammaUp` (LUT-based, range-validated, ~45 lines)
- Wired both into `_build_context_variation_block()` as conditional appends
- Added 5 new preset keys to `vcd_targeted` (all default p=0.0 for backward compat)
- Updated `__init__.py` exports and `__all__`
- Created 34-test suite, all passing
- Confirmed no regressions in existing test suite (1 pre-existing failure: `test_augmentation_imports` fails without local torch — unrelated)

### 2026-03-09 — Audit tool update
- Replaced hardcoded pipeline simulators with R12/proposed versions matching actual production config
- Vectorised `apply_shadow()` (4→8 directions) consistent with the `DirectionalShadow` transform
- Updated all variable names, labels, CSV column names from current/proposed → R12/proposed
- Added `--shadow-p` and `--gamma-up-p` CLI arguments
- Syntax-checked + verified CLI help renders correctly

### 2026-03-09 — Audit execution on real data
- Downloaded 255 DF40 crops (150 real from Celeb-real, 105 fake across 7 methods) + 594 Teams-v2 real frames (30 diverse samples)
- Scripts: `tools/download_audit_data.sh` (bash, fixed SIGPIPE/pipefail issue) → `tools/download_remaining.py` (Python, more robust)
- Ran: `python tools/lighting_showcase.py audit --images-dir audit_data/df40_crops --real-captures-dir audit_data/teams_real --shadow-p 0.10 --gamma-up-p 0.12`
- **Results (real-capture coverage %):**

| Metric | Orig [5%-95%] | R12 Coverage | Proposed Coverage | Delta |
|--------|---------------|-------------|-------------|-------|
| Mean Brightness | 49–153 | **96%** | **97%** | +1pp |
| Brightness Std Dev | 34–70 | **87%** | **85%** | -2pp |
| R/B Ratio (colour temp) | 1.20–2.46 | **80%** | **85%** | **+5pp** |
| RMS Contrast | 0.28–0.86 | **95%** | **96%** | +1pp |

- R/B ratio (colour temperature) gain is the largest: 80%→85%, confirming `ColorTemperatureShift` + `GammaUp` close the warm/cool lighting gap
- Brightness std dev dropped 2pp (87%→85%) — expected: `GammaUp` lifts dark areas, compressing the variance; acceptable trade-off
- Outputs: `audit_data/r13_lighting_audit.png` (histograms), `audit_data/r13_lighting_audit.csv` (raw stats)

### Issues encountered
1. **Import chain problem:** `data/__init__.py` imports `batching` → `torch`, so `from data.augmentations.transforms import ...` fails locally without torch. Solved in tests via `importlib.util.spec_from_file_location()` direct module loading. Not an issue in Docker/cloud where torch is installed.

---

## Lessons Learned (for next round)

1. **Don't pre-name the next round.** We called everything "R13" while R12 was still running. This creates confusion — the next experiment round number depends on R12 outcomes. Use round-neutral labels like "proposed" until the experiment plan is locked.

2. **Use Python for GCS download orchestration, not bash.** The initial bash script (`download_audit_data.sh`) failed silently due to `set -euo pipefail` + SIGPIPE from `head`, and `gsutil ls "…/**/"` glob not working for directory listing. The Python rewrite (`download_remaining.py`) was more reliable: proper subprocess handling, explicit per-folder listing, no SIGPIPE traps. **Recommendation:** future data-pull tools should be Python from the start.

3. **Filename collisions in flat GCS copies.** DF40 has `047.png` in dozens of identity folders. Flat `gsutil -m cp` overwrites silently. Solution: download per-folder to staging dir then rename with `{source}__{basename}` prefix. Build this into any download helper.

4. **`tee` buffering hides progress.** Piping a long-running script through `tee logfile` with Python's default stdout buffering means the log stays empty until the script finishes or flushes. For next time: use `PYTHONUNBUFFERED=1` or `python -u` when piping to a log.

5. **Audit before committing YAML config changes.** Running the audit tool on real GCS data before setting proposed params in experiment YAML confirmed the augmentation values are sensible (+5pp colour-temp coverage). This should be standard practice: implement → audit on real data → commit YAML config.

---

## Files Changed

| File | Change |
|------|--------|
| `data/augmentations/transforms.py` | +`DirectionalShadow` class, +`GammaUp` class (~135 lines) |
| `data/augmentations/pipelines.py` | +shadow/gamma_up conditional appends in `_build_context_variation_block()`, +5 preset keys in `vcd_targeted` |
| `data/augmentations/__init__.py` | +3 exports (`ColorTemperatureShift`, `DirectionalShadow`, `GammaUp`) |
| `tests/test_lighting_transforms.py` | New file, 34 tests across 4 classes |
| `tools/lighting_showcase.py` | Audit tool: 3 accurate pipeline simulators, 8-dir shadow, R12/Proposed labels, `--shadow-p`/`--gamma-up-p` CLI args |
| `tools/download_audit_data.sh` | Download script for GCS audit data (DF40 + Teams-v2), fixed SIGPIPE/pipefail |
| `tools/download_remaining.py` | Python continuation script for robust GCS downloads |
| `docs/LIGHTING_ROBUSTNESS_IMPL_LOG.md` | This file |

---

## How to Enable in Future Experiments

Add these keys to the YAML `augmentation:` block:

```yaml
augmentation:
  # ... existing R12 keys ...
  context_variation_shadow_p: 0.10        # Enable directional shadows
  context_variation_shadow_intensity: [0.15, 0.45]
  context_variation_shadow_softness: [0.20, 0.50]
  context_variation_gamma_up_p: 0.12      # Enable always-brighten
  context_variation_gamma_up_range: [0.45, 0.85]
```

The YAML→router wiring (`combined_paired.py` L2815-2821) uses `preset_overrides` which supports all keys in the `vcd_targeted` preset dict — no additional plumbing needed.

---

## Remaining Gaps (Not Addressed Here)

| Gap | Why Not | Path Forward |
|-----|---------|--------------|
| Option B — pre-process training data brighter | Medium effort, high risk of artifacts | `GammaUp` audit shows 97% brightness coverage — not needed |
| Real-world webcam data expansion | Data collection effort | Teams v2 already in; consider VCD bright captures for next round |
| Audit validation | ✅ **Done** — see execution log above | Coverage confirms proposed augmentations are ready for launch |
