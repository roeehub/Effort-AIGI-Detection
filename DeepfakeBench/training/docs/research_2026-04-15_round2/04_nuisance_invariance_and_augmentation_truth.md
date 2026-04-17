# Nuisance Invariance And Augmentation Truth

## 1. The question for this round

The repo narrative has blamed lighting, gamma, compression, sharpening, and crop sensitivity. Round 2 checked which of those augmentations were actually active in runtime, rather than trusting YAML intent.

## 2. What is definitely active today

For the non-Teams families under `quality_targeted_family` with `strength: vcd_targeted`, the base preset does provide:

- JPEG / blur / noise / downscale variation
- brightness / contrast / hue / saturation variation
- random gamma (`context_variation_gamma_limit`)
- shift / scale / rotate
- colour-temperature simulation (`context_variation_cct_*`)
- sharpening / real-side sharpening differences

This means the non-Teams families are not completely under-augmented.

## 3. What is definitely not active as the YAML currently suggests

### 3.1 `gamma_up_*` is a silent no-op in the current R13 configs

Current configs use:

- `gamma_up_p`
- `gamma_up_range`

But the augmentation block that actually consumes those settings expects the `context_variation_gamma_up_*` names, and the combined-source augmentation wrapper only forwards a restricted preset-key allowlist.

That creates a two-layer failure:

1. current YAML uses the wrong key names
2. even correctly renamed gamma-up keys would still be filtered out by the current allowlist logic, because that allowlist is built from the first preset dict rather than the union of all preset keys

Net effect:

- `GammaUp` is off in the current R13 runtime
- the repo is not currently testing the upward-brightness augmentation it appears to be requesting

### 3.2 Some override keys are cosmetic because the base preset already has the same values

The current `vcd_targeted` preset already includes:

- `context_variation_enabled: True`
- `context_variation_cct_p: 0.15`
- `context_variation_cct_range: (2700, 8000)`

So the YAML lines repeating those values do not change behavior, even though some of them are also filtered by the current allowlist.

### 3.3 `context_variation_oneof_p` is not the control knob anymore

`_build_context_variation_block()` uses independent transforms. `context_variation_oneof_p` is kept for compatibility but ignored in this path. The live control knob is `context_variation_individual_p`.

## 4. The biggest branch-level mismatch

Direct Teams rows do **not** receive the same nuisance coverage as the rest of the train pool.

`_build_teams_passthrough_pipeline()` only applies:

- horizontal flip
- light `RandomBrightnessContrast`

It does **not** apply:

- blur
- downscale
- codec simulation
- stronger colour-temperature variation
- spatial jitter

That matters because the project's deployment target is exactly the Teams branch.

## 5. Nuisance hypothesis audit

### Lighting / gamma / white balance

- Established active on non-Teams families:
  - random gamma
  - brightness / contrast
  - colour-temperature shift
- Established inactive or weak where it matters most:
  - `GammaUp` is off
  - Teams passthrough gets only a light brightness/contrast perturbation
- Interpretation:
  - the repo does have some lighting robustness machinery
  - the exact upward-brightness and direct-Teams lighting protections are weaker than the YAML suggests

### Compression / blur / sharpening

- Established active on non-Teams families:
  - JPEG, blur, noise, downscale
  - sharpen / real-side sharpen asymmetry
- Established inactive on direct Teams rows:
  - no extra codec or blur/downscale transforms in the Teams passthrough branch
- Interpretation:
  - non-Teams quality robustness is present
  - actual Teams rows are being preserved rather than diversified

### Crop jitter / spatial sensitivity

- Established active on non-Teams families:
  - `ShiftScaleRotate` from the context-variation block
- Established inactive on direct Teams rows:
  - no spatial perturbation in `_build_teams_passthrough_pipeline()`
- Interpretation:
  - crop sensitivity is still structurally plausible on the exact deployment branch

## 6. Silent-drop patterns worth calling out directly

### Harmful silent drop

- `gamma_up_p`
- `gamma_up_range`

These are intended to change behavior and currently do not.

### Cosmetic silent drop in current configs

- `context_variation_cct_p`
- `context_variation_cct_range`

These are filtered by the current allowlist, but current `vcd_targeted` preset defaults already match the configured values.

### Future silent drop risk

If someone adds YAML-only shadow overrides such as:

- `context_variation_shadow_p`
- `context_variation_shadow_intensity`
- `context_variation_shadow_softness`

they will not be trustworthy until the preset-key allowlist is fixed.

## 7. Smallest trustworthy ablation set worth running

This cannot be a pure YAML ablation. One tiny plumbing fix is a prerequisite:

- change the preset-key allowlist in `combined_paired.py` from "keys from the first preset dict" to "union of valid preset keys"
- then use the actual runtime key names, including `context_variation_gamma_up_*`

After that, the smallest trustworthy ablation set is:

### Cell A: plumbing-control

- apply the allowlist fix
- rename gamma-up keys correctly
- keep `context_variation_gamma_up_p = 0.0`

Purpose:

- prove the plumbing fix itself did not silently change unrelated behavior

### Cell B: gamma-up only

- same as Cell A
- set `context_variation_gamma_up_p = 0.15`
- set `context_variation_gamma_up_range = [0.45, 0.85]`

Purpose:

- test the exact upward-brightness hypothesis that current R13 YAML intended but did not actually run

### Cell C: Teams-spatial only

- same as Cell A
- add a **light** spatial perturbation to `_build_teams_passthrough_pipeline()`

Suggested envelope:

- shift about `0.02`
- scale about `0.05`
- rotate about `3`
- probability about `0.2`

Purpose:

- test the crop/spatial sensitivity hypothesis on the actual deployment branch instead of only on proxy families

What not to do in the first ablation pass:

- do not mix shadow simulation into the same run
- do not simultaneously change codec simulation
- do not simultaneously change sampler/curriculum

## 8. Status

- Established:
  - current Track A family augmentation is not fully equal to YAML intent
  - `GammaUp` is not active
  - direct Teams rows still get only a minimal augmentation path
- Plausible:
  - a real gamma-up ablation and a light Teams-spatial ablation are higher-value than more generic augmentation churn
- Still unknown:
  - whether the main observed Teams real-FP pain is more sensitive to lighting robustness or spatial robustness once the plumbing is fixed
