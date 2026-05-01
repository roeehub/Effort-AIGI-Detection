# Phase 2D — Pre-launch code-change checklist for R13_P18 yamls

**Date**: 2026-05-01
**Yamls**: `experiments/phase2_round13/R13_P18_METHOD_DOMAIN_GRL.yaml` (treatment) + `experiments/phase2_round13/R13_P18_NO_GRL_CONTROL.yaml` (control), both DRAFT.

The yamls reference a 12-bucket method-domain map that does not yet exist in the trainer's data + detector code. Before any P18 launch, the following code changes must land **and the existing GRL test suite must go green**.

## Required code changes

### 1. `data/sources/combined_paired.py` — replace QUALITY_DOMAIN_MAP with method-aware lookup

**Current** (lines 66–87):
```python
QUALITY_DOMAIN_MAP = {
    "df40": 0,
    "external": 1,
    ...
    "youtube": 3,
}

def _quality_domain_for_source(source: str) -> int:
    return QUALITY_DOMAIN_MAP.get(source, 0)
```

**New**: import from `analysis/method_class_audit_2026-05-01/proposed_method_domain_map.py` and use `lookup_method_domain(method, source)`. Recommended: copy the proposed map into `data/sources/method_domain_map.py` (canonical location), and update `combined_paired.py` to import it.

```python
from data.sources.method_domain_map import lookup_method_domain, METHOD_DOMAIN_NAMES

# Backward-compat alias (so legacy callers don't break):
QUALITY_DOMAIN_MAP = METHOD_DOMAIN_NAMES  # name → ID is wrong direction; rewrite call sites

def _quality_domain_for_sample(method: str, source: str) -> int:
    return lookup_method_domain(method, source)
```

### 2. Update 21+ call sites in `combined_paired.py`

Every line currently calling `_quality_domain_for_source(<source_str>)` needs to pass `method` AND `source`:

```python
# Before:
'quality_domain': _quality_domain_for_source('df40'),

# After:
'quality_domain': _quality_domain_for_sample(method=sample.method, source='df40'),
```

Locations (per `grep -n _quality_domain_for_source`): 2762, 2783, 2841, 2862, 2914, 2935, 2986, 3007, 3049, 3051, 3056, 3165, 3186, 3246, 3322, 3343, 3399. Each iterator constructs a `quality_domain` field per sample.

### 3. `detectors/effort_detector.py:280` — update QualityDomainHead.DOMAIN_MAP (or rename)

The detector's domain head likely has its own DOMAIN_MAP for label validation. Update it to mirror the 12-bucket `METHOD_DOMAIN_NAMES`. Consider renaming `QualityDomainHead` → `MethodDomainHead` for clarity (or just expand the existing class's domain count).

The dynamic `quality_domain_count` reading at `effort_detector.py:398` already supports arbitrary counts — the head architecture itself doesn't need rewriting, just the labels-mapping table that validates incoming labels.

### 4. Fix `tests/test_unpaired_reals_and_grl.py` import bug

Per PLAN.md §10.7 (Codex round 1 verification 2026-04-29): the test currently fails at `TestVCDTargetedPreset.test_vcd_targeted_in_presets` with `ImportError: attempted relative import with no known parent package` originating in `data/augmentations/pipelines.py:21`. The first 9 unit tests pass; downstream presets/yaml-instantiation tests do not. Fix the relative import (likely needs package init or test-side `sys.path` shim) before any P18 launch.

After fix: full test file must pass green.

### 5. Add new tests

- `tests/test_method_domain_map.py` — verify the 14 lookup tests in `proposed_method_domain_map.py` still pass after copying it to its canonical location.
- Extend `tests/test_unpaired_reals_and_grl.py` with: 12-class trainer instantiation smoke test (yaml → model → forward pass → loss is finite).

## Pre-launch CPU smoke gates (run after code changes)

Per Phase 1 synthesis recommendations + PLAN.md §7.4 round-1 escalation:

### a) GRL domain-population gate

```bash
python scripts/launch/check_grl_domain_populations.py \
  --config experiments/phase2_round13/R13_P18_METHOD_DOMAIN_GRL.yaml
```

Expected: prints per-(domain × label) sample count over the first ~256 batches. Fails if any active domain (0–6, 10–11) has zero training samples or any single domain has >90% concentration in a single source family.

### b) Frozen-feature method-LR baseline (does P8A's [CLS] linearly support 12-class method discrimination?)

```bash
python analysis/domain_confusion_probe_2026-04-30/domain_probe.py \
  --domains 0,1,2,3,4,5,6,10,11 \
  --domain-labels-from method \
  --ckpts P8A=analysis/_features_cache_2026-04-30/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth \
  --output_dir analysis/domain_confusion_probe_2026-04-30/outputs/probe_p8a_method_12class/
```

Expected: macro-OVR AUC ≥ 0.85 — i.e. the 12 method classes ARE linearly separable in P8A's [CLS] features. If macro-OVR AUC < 0.70, the GRL classifier has nothing to attack and the packet is pre-doomed.

### c) Vertex smoke run (200 steps, no checkpoint upload)

Same yaml + `--total_training_steps 200` override + `keep_last_n=0`. Confirm in W&B:
- "Quality domain head ENABLED with 12 domains, hidden_dim=256, ..." log line at trainer init.
- `train/loss/quality_domain` non-zero throughout.
- `train/grad_norm/quality_domain_head` non-zero.

Cost: ~$5 / 30 min. Catches integration bugs at ~5% the cost of a full launch.

## Optional but recommended for v2

- **Ramped λ**: 10-line trainer patch wiring `GradientReversalLayer.set_lambda(t)` to a linear ramp 0 → λ_target over `lr_scheduler_warmup_steps`. The `set_lambda` method exists in `effort_detector.py`; no caller in `trainer/trainer.py` invokes it. P18 v1 ships static λ=0.20.
- **In-trainer kill gate**: hook into `evaluate_every_steps` to compute lockbox-substrate-direction probe AUC; abort training if AUC < 0.5 by step 1000 (the P17 trajectory-flip step). New code; not in v1.
- **Production-tight training crops**: requires either (a) preprocessing all training buckets through `recrop_dataset.py` to RFA=0.85, OR (b) a new training-time bbox-aware deterministic crop transform similar to `face_scale_jitter.py` but driven by manifest face_bbox. Heavy preprocessing or new transform; not in v1.

## Launch checklist (after all of the above)

1. Code changes 1–5 land in working tree; tests green.
2. Smoke gates a/b/c pass.
3. ./dev.sh build-prod -y for image rebuild (auto-bumps VERSION).
4. Submit treatment + control to Vertex (us-east1 or us-west4) via
   `scripts/launch/launch_experiment.sh` (or whichever launcher is canonical).
5. Monitor with the `by2cnqbc7`-style state poll (per `gcloud ai custom-jobs describe`).
6. After both jobs land: run promotion-contract scorecard on each (re-using the
   `arena/launch_teams_promotion_contract.sh` flow that succeeded for the v3
   verify run).

## Estimated total engineering effort

| Step | Effort |
|---|---|
| 1+2 (combined_paired.py changes) | 2–3 hours |
| 3 (effort_detector.py update) | 1 hour |
| 4 (test import fix) | 30 min – 2 hours |
| 5 (new tests) | 1–2 hours |
| Smoke gates a/b/c | 1 hour wall + ~$5 |
| Vertex launch + monitor | 8 hours wall, ~$120 |

**Total before launch**: roughly half-day to one day of careful engineering. The yaml drafts are ready to consume once the code lands.
