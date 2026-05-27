# Handoff — wandb-flattening config bug, anchor_aware / face_scale_jitter / periodic_saves silently disabled

**Author:** Claude Opus 4.7 (1M context), 2026-04-28 ~18:10 CEST
**Branch:** `teams-relaunch-root-2026-04-17`
**Working dir:** `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training`
**Status:** Bug diagnosed, fix designed and not applied. The Vertex job that surfaced the bug has been cancelled (job id `8345130870696312832`, us-east1). About **$0.70** burned. **The next agent must apply the fix, smoke-test it, rebuild the image, and relaunch P13_FROM_SCRATCH.**

---

## 0. TL;DR

`train_sweep.py` calls `wandb.init(config=single_cfg)` which silently flattens nested dicts into dot-notation keys (`anchor_aware.enabled`, etc.). The file then has a manual "bypass W&B flattening" block (lines 172–282) that re-applies an **allowlist** of nested dict blocks back onto `config`. Three keys present in the P13 yaml are missing from that allowlist: **`anchor_aware`**, **`face_scale_jitter`**, **`periodic_saves`**. As a result, when the trainer reads `self.config.get('anchor_aware')` it gets `None`, falls back to defaults, and logs `AnchorAwarePenalty DISABLED (enabled=False weight=5.0)` — even though the yaml said `enabled: true`. Same for face_scale_jitter (silently disabled) and periodic_saves (silently no checkpoints saved). PipelineRandomization works because it lives **under** `augmentation:`, which IS in the allowlist.

**Fix:** Add three `if 'anchor_aware' in single_cfg: config['anchor_aware'] = single_cfg['anchor_aware']`-style blocks to `train_sweep.py:255-282`. Mechanical, mirrors the existing pattern. ~30 lines.

---

## 1. Symptom (what was observed in Vertex logs)

Job `8345130870696312832` was launched at 17:50 CEST on 2026-04-28 with image `effort-detector:1.3.225` running `R13_P13_FROM_SCRATCH.yaml`. At trainer init the watch-item monitor caught these lines:

```
2026-04-28 16:01:28,069 - INFO - PipelineRandomization ENABLED p_real=0.55 p_fake=0.45 jpeg_q=(40, 95) downscale=(0.85, 1.0) gamma=(0.92, 1.08) sub_p={'jpeg': 0.5, 'downscale': 0.5, 'chroma_blur': 0.5, 'yuv_roundtrip': 0.5, 'gamma': 0.5}
2026-04-28 16:01:53,826 - INFO - AnchorAwarePenalty DISABLED (enabled=False weight=5.0)
2026-04-28 16:01:53,829 - INFO - FaceScaleJitter DISABLED
```

PipelineRandomization is correctly **ENABLED**. AnchorAwarePenalty and FaceScaleJitter are **DISABLED** despite the yaml asserting otherwise:

```yaml
# experiments/phase2_round13/R13_P13_FROM_SCRATCH.yaml lines 78-91
anchor_aware:
  enabled: true
  weight: 5.0
  target_mean_prob: 0.10
  samples_per_step: 16

face_scale_jitter:
  enabled: true
  scale_limit: 0.25
```

Same job experienced this on its **previous** launch as well (job id `99039952980934656`, also cancelled). Two attempts, same symptom.

---

## 2. Root cause

### 2.1 Why wandb breaks nested dicts

`train_sweep.py:144-147`:

```python
wandb_run = wandb.init(
    mode="online",
    config=single_cfg  # None -> sweep agent supplies config; dict -> single run
)
```

`single_cfg` is the parsed yaml — a Python dict with nested dict values (e.g. `single_cfg['anchor_aware']` is `{'enabled': True, 'weight': 5.0, ...}`).

`wandb.init` flattens nested dicts into dot-notation top-level keys. After this call, `wandb.config` does **not** have a key `'anchor_aware'`; instead it has `'anchor_aware.enabled'`, `'anchor_aware.weight'`, etc., as flat keys. `wandb.config.get('anchor_aware')` returns `None`.

Evidence from the codebase itself — the comment at `train_sweep.py:172-174` is explicit:

```python
# CRITICAL FIX: Apply single_cfg directly to data_config BEFORE wandb overrides
# W&B flattens nested dicts, so wandb.config.get('dataset_methods') returns None
# even if single_cfg has it. We must apply these directly.
```

### 2.2 The existing manual re-apply (and what it covers)

Lines 176-282 of `train_sweep.py` work around the flattening by copying nested blocks straight from `single_cfg` back into the `config` object (which is what gets passed to the Trainer). The block is essentially:

```python
if 'dataset_methods' in single_cfg:
    data_config['dataset_methods'] = single_cfg['dataset_methods']
if 'lesson_data_control' in single_cfg:
    config['lesson_data_control'] = single_cfg['lesson_data_control']
if 'lesson_gate' in single_cfg:
    config['lesson_gate'] = single_cfg['lesson_gate']
if 'augmentation' in single_cfg:
    config['augmentation'] = single_cfg['augmentation']
    data_config['augmentation'] = single_cfg['augmentation']
if 'combined_paired' in single_cfg:
    data_config['combined_paired'] = single_cfg['combined_paired']
if 'deeplive' in single_cfg:
    data_config['deeplive'] = single_cfg['deeplive']
if 'visomaster' in single_cfg:
    data_config['visomaster'] = single_cfg['visomaster']
if 'backbone' in single_cfg:
    config['backbone'] = single_cfg['backbone']
if 'checkpointing' in single_cfg:
    config['checkpointing'] = single_cfg['checkpointing']
# ... a few flat-key blocks ...
if 'use_group_dro' in single_cfg: config['use_group_dro'] = single_cfg['use_group_dro']
if 'group_dro_params' in single_cfg: config['group_dro_params'] = single_cfg['group_dro_params']
if 'value_composite' in single_cfg: config['value_composite'] = single_cfg['value_composite']
```

11 nested-dict allowlist entries. Some go into `config`, some into `data_config`, some into both — depending on which downstream consumer reads which dict.

### 2.3 The gap

The P13_FROM_SCRATCH yaml has **9 top-level nested-dict blocks**:

| Yaml key (line)       | In allowlist? | Consumer                           |
|---|---|---|
| `wandb` (28)          | n/a (managed by wandb itself) | wandb |
| `backbone` (40)       | ✅ yes | model init |
| **`anchor_aware` (78)** | **❌ NO**  | `trainer.py:496` → AnchorAwarePenalty |
| **`face_scale_jitter` (88)** | **❌ NO** | `trainer.py:505` → set_face_scale_jitter_config |
| `augmentation` (92)   | ✅ yes (PipelineRandomization lives here as `augmentation.pipeline_random_*`) | augmentation pipeline |
| `combined_paired` (124) | ✅ yes | data sources |
| **`periodic_saves` (269)** | **❌ NO** | `trainer.py:2375` → periodic ckpt saves |
| `gcs_assets` (280)    | ✅ yes (post-wandb section, line 322) | model loader |
| `checkpointing` (288) | ✅ yes | ckpt save path |

The three missing keys are exactly the three trainer features that logged DISABLED (or, in periodic_saves' case, will silently fail to save ckpts at step_list).

### 2.4 Why PipelineRandomization works

PipelineRandomization is **not** a top-level yaml key. It's nested under `augmentation:` as `augmentation.pipeline_random_p_real`, `augmentation.pipeline_random_p_fake`, etc. When `train_sweep.py` re-applies `augmentation` (line 207-214), the entire nested block is copied as-is, so the augmentation pipeline sees the correct flags. This confirms the fix pattern works — it just needs to be extended to the three missing keys.

---

## 3. Why the prior fix attempt did not work

Earlier today (before this handoff was written) I diagnosed the same DISABLED symptom under a **wrong** hypothesis: I assumed wandb wraps nested dicts as `wandb.sdk.wandb_config.Config` sub-objects (dict-like but not dict subclasses), so `isinstance(raw, dict)` checks at the trainer layer fail. I applied a defensive helper at the trainer layer (`trainer.py:56-73 _to_plain_dict`) and rebuilt + relaunched. **The DISABLED log lines reappeared in the new run.**

The reason: my hypothesis was wrong. wandb does **not** wrap as a Config sub-object — it **flattens to dot-notation**. So `self.config.get('anchor_aware')` returns `None`, and `_to_plain_dict(None)` correctly returns `{}` — but that empty dict still triggers the defaults-fallback path. The trainer-layer fix could not work; the bug is upstream of the trainer, in `train_sweep.py`.

### 3.1 What is currently in the working tree (from that prior attempt)

```
M trainer/trainer.py
?? tests/test_to_plain_dict.py
```

`trainer/trainer.py` modifications (uncommitted):
- Lines 56-73: new `_to_plain_dict(raw)` helper. Coerces None / dict / dict-like-with-`.keys()` to a plain dict; `{}` for anything else. **Harmless defensive code; can stay or be reverted — does NOT fix the bug.**
- Line 495-496: `anchor_cfg = _to_plain_dict(self.config.get('anchor_aware'))` (was `(_aa_raw or {}) if isinstance(_aa_raw, dict) else {}`)
- Line 505: `fsj_cfg = _to_plain_dict(self.config.get('face_scale_jitter'))` (same shape)
- Line 2374-2375: `_ps_raw = self.config.get('periodic_saves'); periodic_cfg = _to_plain_dict(_ps_raw)` (same shape)

`tests/test_to_plain_dict.py` (new, 7 tests, all PASS): exercises the `_to_plain_dict` helper using a `_FakeWandbConfig` mock that mimics dict-like-but-not-dict. **The test premise is wrong** (real wandb flattens; it does not preserve as a Config sub-object). The tests still pass because they test the helper in isolation, but they do NOT reproduce the actual production bug. The next agent should decide whether to keep this file (as defensive coverage), revise it to also exercise the wandb-flattening case, or revert it.

### 3.2 Recommendation on the trainer.py / test_to_plain_dict.py changes

- **Keep `_to_plain_dict`** — it's harmless, costs nothing at runtime, and shields against any future case where someone passes a `wandb.Config` sub-object to the trainer (hypothetical but cheap to defend against).
- **Update or replace `tests/test_to_plain_dict.py`** with a test that documents the actual bug class — i.e., test the train_sweep.py re-apply path. See §6.
- The trainer-layer changes are **not load-bearing** for fixing this bug. The fix lives in train_sweep.py.

---

## 4. The fix

Add three blocks to `train_sweep.py` between line ~282 and the closing `print("=" * 70)` at line 284, mirroring the existing pattern. Place them logically (e.g., after the `value_composite` block at line 279-282, since these are similar trainer-config nested blocks).

### 4.1 Patch text

```python
# Apply anchor_aware config directly (nested dict — W&B flattens; must copy).
# Trainer reads self.config.get('anchor_aware') at trainer.py:496 and constructs
# loss/anchor_aware_penalty.AnchorAwarePenalty. Without this, AnchorAwarePenalty
# silently logs DISABLED and the anti-shortcut anchor-aware loss term is no-op.
if 'anchor_aware' in single_cfg:
    config['anchor_aware'] = single_cfg['anchor_aware']
    aa = single_cfg['anchor_aware']
    print(f"  ✅ Applied anchor_aware: enabled={aa.get('enabled')} weight={aa.get('weight')} target_mean_prob={aa.get('target_mean_prob')}")
    logger.info(f"  Applied anchor_aware: enabled={aa.get('enabled')} weight={aa.get('weight')} target_mean_prob={aa.get('target_mean_prob')}")

# Apply face_scale_jitter config directly (nested dict — W&B flattens; must copy).
# Trainer reads self.config.get('face_scale_jitter') at trainer.py:505 and calls
# data.augmentations.face_scale_jitter.set_face_scale_jitter_config. Without this,
# FaceScaleJitter silently logs DISABLED and face-size canonicalization is no-op.
if 'face_scale_jitter' in single_cfg:
    config['face_scale_jitter'] = single_cfg['face_scale_jitter']
    fsj = single_cfg['face_scale_jitter']
    print(f"  ✅ Applied face_scale_jitter: enabled={fsj.get('enabled')} scale_limit={fsj.get('scale_limit')}")
    logger.info(f"  Applied face_scale_jitter: enabled={fsj.get('enabled')} scale_limit={fsj.get('scale_limit')}")

# Apply periodic_saves config directly (nested dict — W&B flattens; must copy).
# Trainer reads self.config.get('periodic_saves') at trainer.py:2374 to decide
# whether to save checkpoints at fixed step_list values. Without this, no
# periodic checkpoints are saved (only metric-gated saves), and Day-4 evaluation
# loses access to mid-training trajectory ckpts.
if 'periodic_saves' in single_cfg:
    config['periodic_saves'] = single_cfg['periodic_saves']
    ps = single_cfg['periodic_saves']
    print(f"  ✅ Applied periodic_saves: enabled={ps.get('enabled')} step_list={ps.get('step_list')}")
    logger.info(f"  Applied periodic_saves: enabled={ps.get('enabled')} step_list={ps.get('step_list')}")
```

### 4.2 Where to insert

Open `train_sweep.py`, locate the `value_composite` block at lines 276-282:

```python
# Apply value_composite config directly (nested dict — W&B flattens; must copy).
# Keys: target_mean_fpr, max_pool_fpr, stability_jitter_stat. Trainer falls
# back to legacy (0.02 / 0.04 / "max") when absent, so this is legacy-safe.
if 'value_composite' in single_cfg:
    config['value_composite'] = single_cfg['value_composite']
    print(f"  ✅ Applied value_composite: {single_cfg['value_composite']}")
    logger.info(f"  Applied value_composite: {single_cfg['value_composite']}")
```

Paste the three new blocks immediately after the closing `logger.info` of the value_composite block, BEFORE the `print("=" * 70)` at line 284.

### 4.3 Why this is the right fix layer

Three layers were considered:

1. **Generic refactor** — auto-re-apply every nested dict in `single_cfg` in a loop (instead of an allowlist). **Rejected** because it widens the surface of changed behavior; some keys (e.g. `wandb`, `data_source`) intentionally stay flat and may have downstream consumers that rely on `config[key]` being a string. This is a 30-line mechanical patch vs. a behavioral refactor.
2. **Trainer-layer defense** — already attempted, didn't work (see §3). The trainer can't recover information that's already been flattened away by wandb.
3. **Mechanical patch (this fix)** — three new blocks, mirroring 11 existing ones. Zero behavior change for any code path other than the three intended config blocks.

The mechanical patch is the smallest viable change.

---

## 5. Verification plan

### 5.1 Pre-rebuild static checks

After applying the fix, before rebuilding the image:

```bash
# 1. The three new keys appear in train_sweep.py
grep -nE "'anchor_aware' in single_cfg|'face_scale_jitter' in single_cfg|'periodic_saves' in single_cfg" train_sweep.py
# Expect 3 matching lines (one per key) somewhere in the 280-310 range.

# 2. The yaml still has the three blocks (sanity check no accidental removal)
grep -nE "^anchor_aware:|^face_scale_jitter:|^periodic_saves:" experiments/phase2_round13/R13_P13_FROM_SCRATCH.yaml
# Expect 3 matching lines.

# 3. Run the existing trainer-layer tests (already pass; should still pass)
python tests/test_to_plain_dict.py
# Expect: 7 passed, 0 failed.
```

### 5.2 Image rebuild

```bash
./dev.sh build-prod -y
```

Monitor `/tmp/build_prod_*.log` for `✅ Build successful!` and capture the new VERSION (will auto-bump to `1.3.226`).

### 5.3 Relaunch

```bash
./scripts/launch/launch_experiment.sh -y phase2r13-experiments us-east1 \
  experiments/phase2_round13/R13_P13_FROM_SCRATCH.yaml
```

(If us-east1 stays PENDING > 30 min, switch to us-west4 or us-central1 per CLAUDE.md region-capacity rule.)

### 5.4 Watch-item verification

In Vertex stream-logs, **all three** of these lines must appear within the first ~3 minutes of `Trainer Is Using device: cuda`. They should appear in the `train_sweep.py` re-apply block output BEFORE the trainer init logs them as ENABLED.

**In `train_sweep.py` re-apply output** (the `--- Applying single_cfg directly (bypassing W&B flattening) ---` section):
```
  ✅ Applied anchor_aware: enabled=True weight=5.0 target_mean_prob=0.1
  ✅ Applied face_scale_jitter: enabled=True scale_limit=0.25
  ✅ Applied periodic_saves: enabled=True step_list=[2000, 4000, 6000, 9000, 12000, 15000, 18000]
```

**In trainer init**:
```
PipelineRandomization ENABLED p_real=0.55 p_fake=0.45 ...
FaceScaleJitter ENABLED: scale_limit=0.250
AnchorAwarePenalty ENABLED: weight=5.000 target=0.100 samples_per_step=16
```

All three "ENABLED" forms must be present. If **any** says DISABLED, **cancel immediately** and dig deeper — there may be additional keys with the same problem class.

For periodic_saves, the diagnostic at `trainer.py:2378-2386` will fire on every validation call and print:
```
periodic_saves diagnostic: step_cnt=... raw_type=dict resolved_keys=['enabled', 'step_list'] enabled=True save_ckpt_ok=True step_list=[2000, ...] step_in_list=...
```
`raw_type=dict` (not `NoneType`) is the success indicator.

### 5.5 Vertex state monitor (per CLAUDE.md)

Job must reach `JOB_STATE_RUNNING` within 30 min. If `JOB_STATE_PENDING` stays > 30 min, relaunch in us-west4 then cancel original after replacement is RUNNING (per CLAUDE.md).

---

## 6. Suggested test update (optional but recommended)

`tests/test_to_plain_dict.py` was based on the wrong premise (wandb-as-Config-sub-object). Suggest replacing or augmenting with a test that documents the **actual** bug class:

```python
"""Test that the train_sweep.py re-apply allowlist covers all nested dicts that
the trainer reads via self.config.get('<nested_block>'). This is the bug class
that bit anchor_aware / face_scale_jitter / periodic_saves on 2026-04-28.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# All nested-dict yaml keys the trainer expects to read at runtime.
TRAINER_NESTED_KEYS = [
    'anchor_aware',
    'face_scale_jitter',
    'periodic_saves',
    'value_composite',
    'augmentation',
    'combined_paired',
    'group_dro_params',
    'backbone',
    'checkpointing',
    'gcs_assets',
    # add new keys here as new nested config blocks are introduced
]

def test_train_sweep_reapplies_all_trainer_nested_keys():
    """train_sweep.py must include each trainer-consumed nested dict in its
    'bypassing W&B flattening' allowlist. Otherwise wandb flattens and the
    trainer reads None at runtime."""
    src = (ROOT / 'train_sweep.py').read_text()
    missing = []
    for k in TRAINER_NESTED_KEYS:
        # Look for either `'k' in single_cfg` (the dict-membership test) OR
        # the post-wandb override section for gcs_assets which uses a different
        # surface ('gcs_assets' in single_cfg passes too).
        if f"'{k}' in single_cfg" not in src:
            missing.append(k)
    assert not missing, (
        f"train_sweep.py is missing wandb-flattening re-apply for: {missing}. "
        "Add a block like `if '<key>' in single_cfg: config['<key>'] = single_cfg['<key>']` "
        "in the bypass-flattening section (around lines 176-282)."
    )
```

This test will catch the same bug class for any future nested-config addition (anyone who adds `foo_block: {...}` to a yaml AND reads `self.config.get('foo_block')` in the trainer must add `foo_block` to both the trainer key list above and the train_sweep.py allowlist).

---

## 7. Files & line numbers (cheat-sheet)

| Path | Lines | Purpose |
|---|---|---|
| `train_sweep.py` | 144-147 | `wandb.init(config=single_cfg)` — the flattening source |
| `train_sweep.py` | 172-175 | The "CRITICAL FIX: bypassing W&B flattening" comment |
| `train_sweep.py` | 176-282 | Manual re-apply block — **the fix goes here** (after line 282) |
| `train_sweep.py` | 284 | `print("=" * 70)` — close of the re-apply block |
| `trainer/trainer.py` | 56-73 | `_to_plain_dict` defensive helper (uncommitted, harmless, can stay) |
| `trainer/trainer.py` | 495-501 | anchor_aware load + AnchorAwarePenalty construction |
| `trainer/trainer.py` | 503-510 | face_scale_jitter load + set_face_scale_jitter_config call |
| `trainer/trainer.py` | 2365-2390 | periodic_saves load + diagnostic logging |
| `experiments/phase2_round13/R13_P13_FROM_SCRATCH.yaml` | 78-91 | anchor_aware + face_scale_jitter blocks |
| `experiments/phase2_round13/R13_P13_FROM_SCRATCH.yaml` | 269-278 | periodic_saves block |
| `loss/anchor_aware_penalty.py` | 56 | `self.weight = float(cfg.get('weight', 5.0))` — the source of the misleading `weight=5.0` in the DISABLED log line |
| `data/augmentations/face_scale_jitter.py` | (search `set_face_scale_jitter_config`) | Where FaceScaleJitter DISABLED log line originates |
| `tests/test_to_plain_dict.py` | (full file) | Wrong-premise test, see §6 for replacement |

---

## 8. Recent git history (relevant commits)

```
e748d78 Bump VERSION to 1.3.224 (P13 anti-shortcut image) + handoff for launch
cab2909 Add P13 anti-shortcut interventions: anchor-aware loss, pipeline-random aug, face scale-jitter
c7dc828 Add periodic step-based checkpoint saves with isinstance(dict) defense
ed6f53b Fix promotion contract crash on missing readout-only suite reports
ad76cd8 Add pre-launch image-currency guard and close out PCPs (plan-v2 LOG)
```

`cab2909` is the commit that added `anchor_aware` and `face_scale_jitter` to the yaml + trainer code paths but **omitted** updating `train_sweep.py`'s re-apply allowlist. `c7dc828` did the same for `periodic_saves`.

---

## 9. Vertex evidence

Two cancelled jobs surfaced the same symptom:

| Job ID | Region | Image | Cancel reason |
|---|---|---|---|
| `99039952980934656` | us-east1 | `1.3.224` | DISABLED markers seen at trainer init (this is the original P13 launch) |
| `8345130870696312832` | us-east1 | `1.3.225` | DISABLED markers reappeared after the wrong-layer fix attempt |

Cumulative spend on the broken attempts: **~$3.50** (rebuild $1 + $0.70 + $1 + $0.70 ≈ $3.50). Of $150 budget per full P13_FROM_SCRATCH run, this is < 3%.

The W&B project is `phase2r13-experiments` under entity `dtect-vision`. Both runs registered there; they can be inspected at `https://wandb.ai/dtect-vision/phase2r13-experiments` if needed for further forensics.

---

## 10. What success looks like (after the fix is applied and a new run launched)

1. New job state reaches `JOB_STATE_RUNNING` within 30 min.
2. In the trainer log:
   - `--- Applying single_cfg directly (bypassing W&B flattening) ---` prints the three new `✅ Applied anchor_aware: ...`, `✅ Applied face_scale_jitter: ...`, `✅ Applied periodic_saves: ...` lines.
   - Subsequently, `PipelineRandomization ENABLED ...`, `FaceScaleJitter ENABLED: scale_limit=0.250`, `AnchorAwarePenalty ENABLED: weight=5.000 target=0.100 samples_per_step=16` all appear.
3. Training proceeds past step 1000 without crashing (confirms the augmentations don't NaN — see V6 plan §10.2 micro-smoke notes).
4. First periodic checkpoint save fires at step 2000 (look for `periodic_save triggered at step=2000` in the log).
5. The job runs for ~18 hours (18,000 steps) and produces 7 periodic checkpoints in `gs://training-job-outputs/phase2r13_experiments/<wandb_run_id>/`.

---

## 11. Out of scope for this handoff

- The Day-4 scoring infrastructure (clean_eval_v1 / shortcut_probe_v1 / triple-axis verdict / per-pipeline FPR breakdown / substrate comparison). These are documented in `april-28-training-master-plan-v5.md` §3.6 + §4 and the V6 plan at `~/.claude/plans/ultrathink-read-all-the-generic-flute.md` §3.6 + §4. Build these only after confirming the new run is healthy.
- The two pre-existing dormant `isinstance(dict)` patterns at `trainer.py:413-418` (in `_nested_ood_knob`). Same bug class, but inactive for P13 because the flat fallbacks `ood_monitoring_start_step` and `ood_monitoring_every_steps` are set in the yaml. Not blocking; document and defer.

---

## 12. References (for the next agent's reading list)

1. This file (you are here).
2. `april-28-training-master-plan-v5.md` (in this dir) — the broader plan that introduced anchor_aware / face_scale_jitter / pipeline_randomization as anti-shortcut interventions.
3. `~/.claude/plans/ultrathink-read-all-the-generic-flute.md` (V6 of the plan) — the from-scratch pivot; current authority for the overall sprint.
4. `MEMORY.md` and especially: `feedback_no_cancelling_vertex_jobs.md`, `feedback_decision_points.md`, `reference_image_rebuild.md`.
5. `train_sweep.py` lines 1-330 — the config-loading flow.
6. `trainer/trainer.py` lines 355-510 — Trainer.__init__ where the load-bearing config reads happen.

---

**End of handoff. Apply the fix, verify per §5, then proceed with the relaunch.**
