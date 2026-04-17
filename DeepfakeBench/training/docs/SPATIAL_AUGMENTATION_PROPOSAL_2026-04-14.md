# Spatial Augmentation Proposal — Crop Jitter for Effort SVD Training

**Date:** April 14, 2026  
**Status:** Proposal — ready for implementation  
**Risk level:** Low  
**Priority:** Should ship with the next training round (R13 follow-on or R14)

---

## 1. Problem Statement

The deployed Effort detector is highly sensitive to small changes in face crop boundaries. When the same source frame is cropped with slight spatial variation (e.g., different YOLO letterbox input size producing a few-pixel bbox shift), the model produces materially different predictions. This should not happen — deepfake artifacts are not crop-alignment-dependent.

### Root Cause

Two factors combine:

1. **Ultra-narrow trainable subspace.** The Effort method trains only `k=32` residual singular directions (out of 768) per attention layer, constrained to be orthogonal to the frozen CLIP backbone. The decision surface is razor-thin.

2. **Near-zero spatial augmentation during training.** The current `quality_targeted_family` pipeline applies `A.ShiftScaleRotate` at `p=0.15` with `shift_limit=0.04` (±9 pixels on 224×224). That means **85% of training images see zero spatial perturbation**. There is no `RandomResizedCrop`, no `RandomCrop`, no crop jitter, and no test-time augmentation.

### Mechanism

ViT-B/16 divides the 224×224 input into a 14×14 grid of 16×16 patches. Shifting the crop boundary by a few pixels changes which content falls into which patch. The 32 trainable residual directions have learned artifact patterns at *specific patch positions* because training data overwhelmingly arrives with one fixed crop geometry per face. Content shifting across patch boundaries alters the residual activation pattern enough to flip predictions.

---

## 2. Proposed Change

**Increase the probability and range of spatial perturbation in the `_build_context_variation_block` function, driven by experiment YAML config.**

No architecture changes. No new transforms. No changes to the data loading pipeline or collate function. Just config-driven parameter changes to an existing transform.

### Concrete Parameter Changes

| Parameter | Current Value | Proposed Value | Rationale |
|-----------|:---:|:---:|---|
| `context_variation_shift` | `0.04` (~9px) | `0.08` (~18px) | Covers realistic YOLO bbox jitter and tracking noise. One full ViT patch = 16px, so ±18px ensures the model sees content shifting across at least one patch boundary. |
| `context_variation_scale` | `0.12` | `0.12` | **No change.** Scale variation is already adequate. |
| `context_variation_rotate` | `8` | `8` | **No change.** Rotation is already adequate. |
| `context_variation_individual_p` | `0.15` | `0.40` | Ensures ~40% of images see spatial perturbation instead of 15%. This forces the residual directions to learn position-invariant features. |

### Why These Specific Values

- **Shift `0.08`:** ±8% of 224 = ±18 pixels. This is slightly more than one ViT patch width (16px). It matches realistic production variation: YOLO bbox jitter from resolution/confidence changes, SORT tracker noise, frame-to-frame face movement. It does NOT approach the face boundary — even a worst-case 18px shift on a centered 224×224 face crop leaves >90% content overlap. The remaining face region is filled by `BORDER_REFLECT_101`, which is already the current border mode.

- **Probability `0.40`:** This is the minimum needed to ensure the model regularly sees spatial variation. At `p=0.15`, the expected number of spatially-augmented samples per epoch for a given identity is ~0.15×frames — often zero. At `p=0.40`, most identities will see at least one spatially-shifted version per epoch. This is still well below `p=1.0`, preserving the model's ability to learn fine-grained spatial features from the majority of samples.

### What NOT to change

- **Do NOT add `RandomResizedCrop`.** That transform aggressively varies the crop scale and can destroy fine spatial artifact structure (blending boundaries, GAN grid patterns). The existing `ShiftScaleRotate` with modest shift is the right tool.
- **Do NOT add test-time augmentation (multi-crop averaging).** That would mask the symptom without fixing the fragility, and doubles inference cost.
- **Do NOT change the architecture or rank.** The k=32 sweet spot is well-established through multiple ablations (R2.5, R12_B, R12_D).

---

## 3. Where to Change — Code Locations

### 3.1 The Only Code That Needs to Change: Nothing

**No code changes are needed.** The `_build_context_variation_block` function in `DeepfakeBench/training/data/augmentations/pipelines.py` (lines 944–1025) already reads all spatial parameters from the preset dict:

```python
A.ShiftScaleRotate(
    shift_limit=p.get("context_variation_shift", 0.03),
    scale_limit=p.get("context_variation_scale", 0.10),
    rotate_limit=p.get("context_variation_rotate", 6),
    border_mode=cv2.BORDER_REFLECT_101,
    p=ind_p,  # from context_variation_individual_p
)
```

These values are overridden by experiment YAML → preset_overrides chain (see §3.2). The change is YAML-only.

### 3.2 Config → Code Path

The full chain from YAML to transform object:

1. Experiment YAML sets `context_variation_shift`, `context_variation_individual_p`, etc. under `augmentation:`.
2. `_create_combined_transform()` in `combined_paired.py` (line ~3597) filters YAML keys against `_valid_preset_keys` and builds `preset_overrides`.
3. `create_quality_targeted_family_router(preset_overrides=...)` merges them on top of `_QUALITY_TARGETED_PRESETS["vcd_targeted"]`.
4. `_build_context_variation_block(merged_preset)` reads the overridden values and constructs the `ShiftScaleRotate` transform.

**Verified:** `context_variation_shift` and `context_variation_individual_p` are both in `_valid_preset_keys` (they exist in the `vcd_targeted` preset dict at `pipelines.py` lines 884–935). YAML overrides will flow through correctly.

### 3.3 What to Create: New Experiment YAML

Create a new experiment config derived from the current best R13 config. Suggested name:

```
DeepfakeBench/training/experiments/phase2_round13/R13_SA_spatial_aug_ablation.yaml
```

This should be an exact copy of the current best R13 config (likely `R13_A_trackA_teams_enhanced.yaml`) with only the spatial augmentation parameters changed:

```yaml
# --- Spatial augmentation ablation ---
# Only these two lines differ from R13_A:
context_variation_shift: 0.08        # was 0.04
context_variation_individual_p: 0.40  # was 0.15
```

Everything else — data sources, sampling weights, backbone, rank, learning rate, schedule — must remain identical to enable a fair A/B comparison.

### 3.4 Matched Control

The matched control is the existing `R13_A_trackA_teams_enhanced.yaml` run (W&B run `f04l917o`, step-15500 checkpoint). No new control run is needed if the data sources and other config remain identical.

---

## 4. Expected Outcomes

### If the change works (likely)

| Metric | Expected Direction | Why |
|--------|:---:|---|
| Holdout AUC | Slight decrease (0.1–0.3%) | Lost some overfit spatial signal |
| OOD AUC | Flat or slight increase | Signal that survives is more transferable |
| Target-domain FPR (Teams real) | Decrease (better) | Less sensitivity to webcam/resolution variation |
| Frame-to-frame prediction stability | Much more stable | Core improvement target |
| Per-method recall | Flat or slight redistribution | Some method signals are spatial, most are not |

### If the change fails

If both holdout AND OOD metrics drop, the shift range is too aggressive. Fall back to:
- `context_variation_shift: 0.06` (±13px, still >1 patch width in most directions)
- `context_variation_individual_p: 0.30`

### Red flags to watch for

- Training loss increasing significantly faster than the control → the augmentation is too aggressive, reduce probability.
- OOD AUC dropping below 0.95 → the model lost genuine detection capacity, not just spatial overfit.
- Holdout AUC dropping more than 0.5% → reduce shift magnitude first, then probability.

---

## 5. Validation Plan

### 5.1 Before Training: Sanity Check

Run a quick visual check to confirm the augmentation produces reasonable crops:

```python
from data.augmentations.pipelines import _build_context_variation_block

p = {
    "context_variation_enabled": True,
    "context_variation_shift": 0.08,
    "context_variation_scale": 0.12,
    "context_variation_rotate": 8,
    "context_variation_individual_p": 1.0,  # force it for visual check
}
transforms = _build_context_variation_block(p)
# Apply to a sample face crop and visually inspect
```

Confirm that shifted images still clearly contain the full face region.

### 5.2 During Training: In-Training Monitoring

Watch these W&B metrics relative to the R13_A control:

- `train/loss/overall` — should track within 10% of control
- `val_holdout/overall/auc` — tolerate up to 0.3% regression
- `ood/overall/auc` — must not regress
- `val_primary/ood_composite` — must not regress

### 5.3 After Training: Target-Domain Scorecard

Run the frozen Track C target-domain scorecard on the best checkpoint from this run, comparing against the R13_A step-15500 candidate:

- `teams_real_all_dev` FPR
- `teams_real_poor_quality_dev` FPR
- `teams_fake_all_dev` recall
- `visomaster_enhanced_macro_dev` recall

### 5.4 Stability Test (New)

Run inference on 5–10 Teams videos using two different YOLO operating points (e.g., `imgsz=640` vs `imgsz=480`) and compare prediction variance. The spatially-augmented model should show significantly less prediction drift than the R13_A baseline.

---

## 6. Implementation Checklist

- [ ] Create `R13_SA_spatial_aug_ablation.yaml` by copying `R13_A_trackA_teams_enhanced.yaml`
- [ ] Change only: `context_variation_shift: 0.08` and `context_variation_individual_p: 0.40`
- [ ] Verify YAML parses correctly (dry-run: `./dev.sh train-dry-run`)
- [ ] Build image and launch on Vertex
- [ ] Monitor training metrics against R13_A baseline
- [ ] Run Track C scorecard on best checkpoint
- [ ] Run stability test across YOLO operating points
- [ ] Document result in the upgrade plan

---

## 7. Incidental Finding: Possible `gamma_up_p` Config Bug

While researching this proposal, I noticed that `R13_A_trackA_teams_enhanced.yaml` sets:

```yaml
gamma_up_p: 0.15
gamma_up_range: [0.45, 0.85]
```

But the preset key names are `context_variation_gamma_up_p` and `context_variation_gamma_up_range`. The YAML override mechanism filters by `_valid_preset_keys`, which contains the `context_variation_*` prefixed names. The unprefixed `gamma_up_p` key would **not** match and would be silently dropped — meaning `GammaUp` stays disabled (p=0.0) in R13_A despite the YAML appearing to enable it.

**Recommendation:** If GammaUp was intended to be active, fix the YAML key names in the new experiment config:

```yaml
context_variation_gamma_up_p: 0.15        # was: gamma_up_p (silently ignored)
context_variation_gamma_up_range: [0.45, 0.85]  # was: gamma_up_range (silently ignored)
```

This is separate from the spatial augmentation proposal and should be verified independently.

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|:---:|:---:|---|
| Holdout AUC regresses >0.5% | Low | Low | Reduce parameters; holdout isn't the deployment metric |
| OOD AUC regresses | Very low | Medium | Abort and revert to R13_A config |
| Training becomes unstable | Very low | Low | The augmentation is mild; `BORDER_REFLECT_101` prevents artifacts |
| Change has no effect | Low | None | Proves the model is already robust (good news) |
| Change helps OOD but hurts a specific method | Low | Low | Per-method scorecard will catch this |

**Overall risk: Low.** This is a single-parameter change to an existing transform that is already active in the pipeline. The change makes the training distribution closer to production conditions. The worst realistic outcome is "no measurable effect," not "model breaks."

---

## 9. Relationship to Other Active Work

- **Track A (Teams-enhanced data source):** Independent. Spatial augmentation applies to all families uniformly. Can run in parallel.
- **Track B (Teams codec simulation):** Independent. The Teams simulation is a separate post-pipeline step. Spatial augmentation does not interact with it.
- **Track C (Target-domain scorecard):** This proposal adds one more checkpoint to score on the existing Track C suite. No Track C code changes needed.
- **Rank parameter (k):** Independent. If a future round tests k=40 or k=48, spatial augmentation should still be applied. The two changes are complementary — more residual capacity + more spatial robustness.
