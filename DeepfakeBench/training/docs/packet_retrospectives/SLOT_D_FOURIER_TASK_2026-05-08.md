# SLOT D — Fourier-aug task spec for the next agent (2026-05-08)

> **This is a one-off operational task spec, not a context document.** The wiki under `docs/packet_retrospectives/` carries the full project context. Your job is **operational**: implement → smoke-test → build → launch → monitor → document. **No new packets, no proposals, no synthesis.** When all 4 P2 slots land, the wiki will reflect a complete state for a fresh session to pick up from.

---

## 1. What you are (and what you are not)

**You are**: an operational agent. You write a small amount of new code (band-limited Fourier amplitude augmentation), wire it into the existing training stack, smoke-test it locally, build the production image, launch Slot D as a Vertex training job, then monitor all 4 P2 slots (A/B/C already in flight + D you launch).

**You are not**: a strategy agent. You do not propose new packets. You do not draw conclusions about what the canary signals "mean." You do not propose to relax/tighten any close criterion. You do not change the F1-F5 framework. You do not modify any existing yaml. You do not modify the canary probe or the correlation_penalty loss.

When results land, you only **record** them: pull artifacts, populate the eval folder skeletons per `eval_folder_template.md`, append TIMELINE entries, update STATE.md "In flight" → "Just landed", regenerate OPEN_LOOPS.md. The user starts a fresh session afterward to interpret.

---

## 2. Read this much, no more

Read in this exact order. Do **not** read `AGENT_PROPOSAL_*.md` or any opinion docs — you do not need them for this operational task.

1. `docs/packet_retrospectives/AGENTS.md` (full) — the read/update protocol you must follow.
2. `docs/packet_retrospectives/STATE.md` (full) — current rolling snapshot; tells you what's in flight.
3. `docs/packet_retrospectives/AGENT_GUIDE.md` Rules 0-4 (skip Rule 5 — opinion-vs-fact discipline you don't need).
4. `docs/packet_retrospectives/packets/P2.md` (full) — the packet you're extending with Slot D.
5. `docs/packet_retrospectives/threads/in_training_canary_signal.md` (full) — the canary infrastructure you need to enable on Slot D.
6. `docs/packet_retrospectives/threads/processing_signature_shortcut.md` §"Probe 6 (closes loop)" subsection (lines ~189-212) — the empirical foundation for the Fourier-aug recipe. Memory anchor: `project_fourier_band_overlap_2026-05-06.md`.
7. `analysis/fourier_band_overlap_2026-05-06/run_probe.py` — the empirical probe code. Mirror its `RES=224`, `N_RADIAL=16`, and `radial_bins()` function exactly so band indices match the GREEN-verdict bands.
8. `experiments/phase2_round13/R13_P2_SCRATCH_CORR_ONLY.yaml` — Slot B yaml. **Use this as the template for Slot D** (drop `correlation_penalty` and add `fourier_aug` block; keep everything else identical).

That should take you under 30 minutes. Stop reading once you have these in head.

---

## 3. The task — bands 8-13 Fourier amplitude randomization

### 3.1 What to implement

A new augmentation primitive: **band-limited Fourier amplitude randomization**. Per-frame, per-channel multiplicative noise on FFT amplitude in radial bands 8-13 of a 16-band partition at 224×224 resolution. Phase preserved universally. Bands 5-6 explicitly preserved (they carry manipulation signal per Probe 6 GREEN verdict). With probability `p_apply` (default 0.5) the aug fires; otherwise the frame passes through untouched. Apply per-frame independently — no cross-frame mixing — which preserves labels by construction ("same-label" in the memory's recipe sketch).

### 3.2 Concrete spec

Pin the implementation to these parameter defaults — the user has not authorized you to tune them. They come from Probe 6's GREEN-verdict bands plus conservative noise magnitude:

| Parameter | Value | Source |
|---|---|---|
| `p_apply` | 0.5 | matches `pipeline_randomization` cadence |
| `radial_resolution` | 16 | mirrors `analysis/fourier_band_overlap_2026-05-06/run_probe.py` `N_RADIAL` |
| `image_resolution` | 224 | mirrors probe `RES`; matches CLIP-B16 input |
| `bands_to_randomize` | `[8, 9, 10, 11, 12, 13]` | bands 12-13 are the cleanest cell (shortcut AUC 0.97, signal AUC 0.46-0.52); 8-10 are the secondary safe zone |
| `bands_to_preserve` | `[5, 6]` | explicitly signal-carrying per probe |
| `noise_distribution` | log-uniform multiplicative, sampled per (band × channel) once per frame | per-frame independent → "same-label" |
| `noise_log_range` | `[-0.3, +0.3]` | factor ∈ [exp(-0.3), exp(0.3)] ≈ [0.74, 1.35] — conservative, prevents catastrophic distortion |
| `apply_to` | per-channel RGB | preserves color information; matches existing aug's tensor shape contract |
| `phase_handling` | preserved | per recipe sketch in memory |

### 3.3 Where to put the code

Single new file: `data/augmentations/fourier_band_aug.py` (~120 LOC). Mirror the existing aug primitive style in `data/augmentations/face_scale_jitter.py` (a small albumentations-compatible callable + a config-based `set_*_config` global setter, since the dataloader composition uses module-level config state).

The aug should expose:
- `class FourierBandAmpAug` with `__call__(image: np.ndarray) -> np.ndarray` — operates on a single HWC uint8 or float32 image at 224×224. Returns same shape/dtype.
- `set_fourier_aug_config(cfg: dict)` — global setter, called once at trainer init from the config block.
- `is_fourier_aug_enabled() -> bool` — for trainer logging.

The aug must be **idempotent and pure**: same input + same RNG state → same output. Use a numpy `np.random.Generator` instance per call (seeded by torch's per-worker RNG) so DataLoader workers get independent streams.

### 3.4 How to wire it in

Add the config block to the yaml (see §3.6 below).

Wire it into the existing aug pipeline. Find where `face_scale_jitter` is invoked in the augmentation chain (likely `data/combined_paired.py` or `data/augmentations/pipelines.py`); insert `fourier_band_aug` at the same level (a sibling, not a replacement). It should run on the 224×224-resized frame, AFTER face crop + resize but BEFORE channel normalization (since it operates on pixel-domain images). If unsure where exactly: call it from the `pipeline_randomization` aug chain at the end (since pipeline_randomization is also pixel-domain on the post-resize frame).

Add to `train_sweep.py` re-apply allowlist (memory `project_wandb_flattens_nested_dicts.md`): mirror the `pair_rank_loss`/`canary_probe`/`correlation_penalty` blocks at lines ~290-345. Without this, the yaml block silently drops to `enabled: false`.

### 3.5 Reference implementation outline

```python
# data/augmentations/fourier_band_aug.py
import numpy as np

_GLOBAL_CFG = {'enabled': False, 'p_apply': 0.5, 'bands_randomize': [8,9,10,11,12,13],
               'bands_preserve': [5,6], 'noise_log_range': [-0.3, 0.3],
               'radial_resolution': 16, 'image_resolution': 224}
_RBINS = None  # cached radial-bin grid

def set_fourier_aug_config(cfg):
    global _GLOBAL_CFG, _RBINS
    if cfg is None or not isinstance(cfg, dict):
        return
    _GLOBAL_CFG.update({k: cfg[k] for k in cfg if k in _GLOBAL_CFG})
    _RBINS = None  # force recompute on next use

def is_fourier_aug_enabled():
    return bool(_GLOBAL_CFG.get('enabled', False))

def _radial_bins():
    """Mirror analysis/fourier_band_overlap_2026-05-06/run_probe.py."""
    res = int(_GLOBAL_CFG['image_resolution'])
    n_radial = int(_GLOBAL_CFG['radial_resolution'])
    yy, xx = np.mgrid[:res, :res]
    cx = cy = (res - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = r.max()
    return np.clip((r / r_max * n_radial).astype(np.int32), 0, n_radial - 1)

class FourierBandAmpAug:
    def __init__(self):
        pass

    def __call__(self, image):
        global _RBINS
        if not is_fourier_aug_enabled():
            return image
        rng = np.random.default_rng()
        if rng.random() > float(_GLOBAL_CFG['p_apply']):
            return image
        # image is HWC uint8 or float32 at the configured resolution
        if _RBINS is None:
            _RBINS = _radial_bins()
        bands_rand = set(int(b) for b in _GLOBAL_CFG['bands_randomize'])
        bands_pres = set(int(b) for b in _GLOBAL_CFG['bands_preserve'])
        log_lo, log_hi = float(_GLOBAL_CFG['noise_log_range'][0]), float(_GLOBAL_CFG['noise_log_range'][1])
        was_uint8 = (image.dtype == np.uint8)
        x = image.astype(np.float32)
        # per-channel FFT
        out = np.empty_like(x)
        for c in range(x.shape[2]):
            F = np.fft.fft2(x[:, :, c])
            F = np.fft.fftshift(F)
            amp = np.abs(F)
            phase = np.angle(F)
            # per-band scalar in target bands; preserve bands_pres (no-op there)
            new_amp = amp.copy()
            for b in bands_rand:
                if b in bands_pres:
                    continue
                mask = (_RBINS == b)
                if not mask.any():
                    continue
                scale = float(np.exp(rng.uniform(log_lo, log_hi)))
                new_amp[mask] = amp[mask] * scale
            F2 = new_amp * np.exp(1j * phase)
            F2 = np.fft.ifftshift(F2)
            out[:, :, c] = np.real(np.fft.ifft2(F2))
        out = np.clip(out, 0.0, 255.0)
        if was_uint8:
            out = out.astype(np.uint8)
        return out
```

You may adapt formatting/naming to match the existing codebase's style. The behavior must match the spec.

### 3.6 The Slot D yaml — `R13_P2_SCRATCH_FOURIER.yaml`

Copy `R13_P2_SCRATCH_CORR_ONLY.yaml` verbatim, then make these changes:

1. Update `name`, `description`, `seed: 5304`, and the `wandb.tags` list (replace `p2-scratch-corr-only` → `p2-scratch-fourier`, drop `correlation-penalty-4axis`, add `fourier-band-amp-aug`, `bands-8-13-randomize`, `bands-5-6-preserve`).
2. Set `correlation_penalty.enabled: false` (drop the `lambda` and `axes` keys; mirror Slot C's pattern).
3. **Keep** `pair_rank_loss.lambda: 0.0`, `face_scale_jitter.enabled: true scale_limit: 0.50`, `canary_probe.enabled: true ...`. The aug-only nature is the lever.
4. Add a new yaml block:
   ```yaml
   fourier_aug:
     enabled: true
     p_apply: 0.5
     bands_randomize: [8, 9, 10, 11, 12, 13]
     bands_preserve: [5, 6]
     noise_log_range: [-0.3, 0.3]
     radial_resolution: 16
     image_resolution: 224
   ```
5. All other fields identical to Slot B/C (data, schedule, periodic_saves, ood_monitoring, etc.).

Add the file at `experiments/phase2_round13/R13_P2_SCRATCH_FOURIER.yaml`.

### 3.7 Smoke test (mandatory before image build)

Three-tier smoke test, all CPU, all under 10 minutes total. **All three must pass before you submit Cloud Build.**

**Tier 1 — unit test on synthetic image** (~30 sec):
```python
import numpy as np
from data.augmentations.fourier_band_aug import FourierBandAmpAug, set_fourier_aug_config
set_fourier_aug_config({'enabled': True, 'p_apply': 1.0,
                        'bands_randomize': [8,9,10,11,12,13], 'bands_preserve': [5,6],
                        'noise_log_range': [-0.3, 0.3],
                        'radial_resolution': 16, 'image_resolution': 224})
aug = FourierBandAmpAug()
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
out = aug(img)
assert out.shape == img.shape and out.dtype == img.dtype
diff = (out.astype(np.float32) - img.astype(np.float32))
assert np.abs(diff).mean() > 0.5  # aug had measurable effect
assert np.abs(diff).mean() < 30   # but not catastrophic
print('Tier 1 PASS')
```

**Tier 2 — disabled path no-op** (~5 sec):
```python
set_fourier_aug_config({'enabled': False, 'p_apply': 1.0,
                        'bands_randomize': [8,9,10,11,12,13], 'bands_preserve': [5,6],
                        'noise_log_range': [-0.3, 0.3],
                        'radial_resolution': 16, 'image_resolution': 224})
aug = FourierBandAmpAug()
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
out = aug(img)
assert np.array_equal(img, out), 'disabled aug must be exact pass-through'
print('Tier 2 PASS')
```

**Tier 3 — visual sanity on a real face crop** (~1 min):
Load any 224×224 face frame from `arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet` (download one row's `frame_path`). Apply the aug 5 times with different RNG. Confirm:
- Aug output is recognizable as the same face (not catastrophically distorted).
- p99 absolute pixel diff between original and any output is < 80 (out of 255).
- The aug is not a no-op (mean abs diff > 1.0).

If Tier 3 fails (output is garbled or unchanged), the aug parameters need adjustment **but you do not change them unilaterally**. Stop, write what you saw to `analysis/p2_eval_2026-05-08/SLOT_D_SMOKE_FAIL.md`, and surface the issue to the user. Do not launch.

---

## 4. Build + launch

After all 3 smoke tests pass:

1. **Commit the new code + yaml** as a single commit:
   ```
   git add DeepfakeBench/training/data/augmentations/fourier_band_aug.py
   git add DeepfakeBench/training/experiments/phase2_round13/R13_P2_SCRATCH_FOURIER.yaml
   git add DeepfakeBench/training/data/augmentations/__init__.py  # if you exported the new class
   git add DeepfakeBench/training/data/combined_paired.py  # or wherever you wired it
   git add DeepfakeBench/training/train_sweep.py  # the allowlist entry
   git commit -m "P2 Slot D: band-limited Fourier amp aug + R13_P2_SCRATCH_FOURIER yaml"
   ```
   Use the standard commit-message footer per CLAUDE.md.

2. **Build the prod image**:
   ```
   ./dev.sh build-prod -y
   ```
   Expected to bump VERSION 1.3.271 → 1.3.272. Wait for `Build complete!` / `STATUS: SUCCESS` in the log. Use a Monitor task to avoid polling. Cloud Build typically takes 12-20 min on this project.

3. **Launch Slot D**. The other 3 slots are already on us-east1, us-west4, us-central1. Per CLAUDE.md: prefer US regions. Pick whichever US region has the lowest current load — `us-east1` is a safe default since GPU class A100 is most readily available there. Multiple jobs per region is fine.
   ```
   ./scripts/launch/launch_experiment.sh -y phase2-round13 us-east1 \
     experiments/phase2_round13/R13_P2_SCRATCH_FOURIER.yaml
   ```
   Capture the Vertex job ID + display name. Per CLAUDE.md, if the job stays `JOB_STATE_PENDING` past 30 min, relaunch in another US region; do **not** cancel the original until the replacement is `RUNNING`.

4. **After launch** — within 5 minutes of `Job submitted:`:
   - Update `docs/packet_retrospectives/STATE.md` "In flight" section: add Slot D row to the table with the Vertex job ID + state.
   - Append a TIMELINE entry: `2026-05-08 HH:MM local — Slot D launched on image 1.3.272 ...`
   - Update `docs/packet_retrospectives/packets/P2.md`:
     - Status card: Slots field `3 → 4`.
     - Configuration: add a Slot D row mirroring Slot A/B/C structure.
   - Commit these doc changes as a single follow-up commit.

---

## 5. Monitor playbook (all 4 slots)

Set up monitoring once Slot D is launched. Goal: be ready to act on three classes of event without missing them.

### 5.1 Job state transitions

For each of the 4 slots, track the transition `PENDING → RUNNING → SUCCEEDED / FAILED`. Use a polling Monitor (60-300 sec interval is fine):
```bash
for slot in A B C D; do
  case $slot in
    A) JOB="projects/700371397073/locations/us-east1/customJobs/8395459915946131456" ;;
    B) JOB="projects/700371397073/locations/us-west4/customJobs/5580112770428305408" ;;
    C) JOB="projects/700371397073/locations/us-central1/customJobs/3126673777023254528" ;;
    D) JOB="<TBD — your launched job's full path>" ;;
  esac
  STATE=$(gcloud ai custom-jobs describe $JOB --format="value(state)" 2>/dev/null)
  echo "$slot: $STATE"
done
```

Reactions:
- **PENDING > 30 min** for any slot → relaunch in another US region; do not cancel original until replacement is RUNNING.
- **RUNNING** → record the transition time in TIMELINE.md; that's when you start expecting canary readouts at step 1000.
- **SUCCEEDED** → pull artifacts (see §5.3).
- **FAILED** → pull the failure log via `gcloud ai custom-jobs stream-logs $JOB --region=$REGION` (last ~200 lines), inspect, decide if it's a retryable infra issue (rerun the same yaml in a different region) or a bug in your new code (Slot D only — surface to user with the log excerpt; do not blindly retry).

### 5.2 Canary-probe firing

Once a slot is RUNNING, the canary probe fires at every multiple of 1000 steps. Watch the W&B run for keys starting with `canary/`. Critical signals to flag (do **not** interpret — just flag the observation in the eval folder if extreme):
- `canary/score_p95_on_reals` rising past 0.85 mid-training → log as observation.
- `canary/lockbox_recall_at_FPR_10pct` falling between two consecutive probes → log.
- `canary/max_per_identity_mean_score` rising past 0.50 → log.
- `canary/n_frames_evaluated < 700` → indicates the canary loader dropped frames; capture the trainer.log warning that explains why; flag it as a data-pipeline issue. Probably benign (some GCS frames decoded badly) but worth noting.

You record these in `analysis/p2_eval_2026-05-08/CANARY_TRAJECTORY_FACTS_2026-05-08.md` (which you will create — see §5.3). You do **not** decide whether they "mean" the run is failing. The user decides.

### 5.3 When a slot SUCCEEDED — pull and document

For each slot that completes:

1. Identify the W&B run ID from the W&B dashboard or the slot's stream-logs.
2. Identify the GCS output path: `gs://training-job-outputs/best_checkpoints/<wandb_run_id>/`.
3. Pull the per-step canary readouts to local CSV (W&B export):
   ```
   python -c "
   import wandb
   api = wandb.Api()
   run = api.run('dtect-vision/phase2-round13/<wandb_run_id>')
   import pandas as pd
   df = run.history(keys=[k for k in run.summary.keys() if k.startswith('canary/')], samples=1000)
   df.to_csv('analysis/p2_eval_2026-05-08/<slot>_canary_history.csv', index=False)
   "
   ```
4. List the saved checkpoints under `gs://training-job-outputs/best_checkpoints/<wandb_run_id>/` and record the file names + sizes (do not download — they're 5-7 GB each).
5. Populate the eval folder skeleton:
   ```
   analysis/p2_eval_2026-05-08/
     RESULTS_FACTS_2026-05-08.md    # 1 row per slot × column = (run_id, ckpt_count, train_outcome, canary_first_fire_step, canary_last_fire_step)
     CANARY_TRAJECTORY_FACTS_2026-05-08.md  # per-slot canary trajectory, observations only
     <slot>_canary_history.csv
   ```
   Use the FACTS-doc discipline from `eval_folder_template.md`. **Forbidden words: succeeds, fails, wins, promotes, deployment-grade.** Numbers and direct observations only.

### 5.4 When all 4 slots have terminated

Run `python tools/regenerate_open_loops.py` from `docs/packet_retrospectives/` to refresh OPEN_LOOPS.md. Append a TIMELINE entry covering all 4 outcomes. Update STATE.md "In flight" → "Just landed".

Do **not** propose a Phase A scorecard run for the P2 ckpts. The user will authorize that next session after reviewing the canary trajectories.

---

## 6. Documentation playbook

Every event you act on must produce a TIMELINE entry. Format mirrors the existing TIMELINE entries — one line per event, owning thread(s), outcome.

| Event | Wiki updates |
|---|---|
| Slot D launched | STATE.md "In flight" Slot D row; TIMELINE entry; packets/P2.md Status card slots `3 → 4` + Configuration row + Source files row |
| Slot transitions PENDING → RUNNING | TIMELINE entry only |
| Slot fails (FAILED state) | TIMELINE entry; STATE.md flag the failure; if it's Slot D and the cause is the new aug code, write a `analysis/p2_eval_2026-05-08/SLOT_D_FAILURE_<HHMM>.md` FACTS-doc with the log excerpt and surface to user — do **not** auto-retry without user authorization |
| Canary observation crossing the §5.2 thresholds | record in `analysis/p2_eval_2026-05-08/CANARY_TRAJECTORY_FACTS_2026-05-08.md`; no TIMELINE entry needed unless the run also fails |
| Slot SUCCEEDED | TIMELINE entry; pull artifacts + canary CSVs (§5.3); populate eval folder skeletons; update P2.md "Results at the time" with the run IDs + ckpt step list |
| All 4 slots terminated | regenerate OPEN_LOOPS.md; STATE.md "In flight" → "Just landed"; final TIMELINE summary entry; commit |

Commit cadence: one commit per logical batch. After Slot D launch — one commit. After each individual slot lands — one commit. After all 4 land — one final summary commit. Use the standard commit-message footer per CLAUDE.md.

---

## 7. Hard constraints

- Do **not** modify any existing yaml (Slot A/B/C).
- Do **not** modify the canary probe or the corr_penalty loss code.
- Do **not** modify any thread other than to **append** dated update subsections — never rewrite earlier subsections.
- Do **not** propose new packets or next experiments.
- Do **not** draw conclusions about which lever class "won" or "failed."
- Do **not** authorize any Vertex job retries beyond the one CLAUDE.md region-failover rule.
- Do **not** cancel any Vertex job without explicit user authorization (memory `feedback_no_cancelling_vertex_jobs.md`).
- Do **not** skip the smoke test before image build.
- Do **not** push to remote without explicit user authorization.

If you encounter a situation not covered by this spec, **stop and surface to the user** with a one-line description + the relevant artifact path. Do not improvise.

---

## 8. Success criteria — when can you stop?

You can mark this task complete when **all** of:

1. `R13_P2_SCRATCH_FOURIER.yaml` exists, committed, and was the source of a launched Vertex job.
2. The Slot D Vertex job is in either `JOB_STATE_RUNNING` or a terminal state (`SUCCEEDED` or `FAILED`).
3. STATE.md has a Slot D row in the in-flight table.
4. TIMELINE.md has the Slot D launch entry.
5. packets/P2.md reflects 4 slots in the Status card + Configuration sections.
6. For each terminated slot (any of A/B/C/D), the eval folder skeletons are populated and committed.
7. After all 4 terminate: OPEN_LOOPS.md is regenerated, STATE.md is updated, final TIMELINE entry committed.

When all 7 are met, **stop**. Do not start a new task. Hand control back to the user. The next session — fresh, no context inherited — will read the wiki and decide next steps.

---

## 9. Quick-reference paths

| Path | Purpose |
|---|---|
| `data/augmentations/fourier_band_aug.py` | NEW file you create (~120 LOC) |
| `experiments/phase2_round13/R13_P2_SCRATCH_FOURIER.yaml` | NEW file you create (copy of Slot B with deltas in §3.6) |
| `data/augmentations/face_scale_jitter.py` | template for the aug primitive style |
| `train_sweep.py` lines ~290-345 | where to add the wandb-allowlist entry |
| `analysis/fourier_band_overlap_2026-05-06/run_probe.py` | reference for `radial_bins()` and `RES`/`N_RADIAL` constants |
| `arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet` | source of one real frame for Tier 3 smoke |
| `analysis/p2_eval_2026-05-08/` | NEW directory you populate as results land |
| `experiments/phase2_round13/R13_P2_SCRATCH_CORR_ONLY.yaml` | the verbatim template you copy + edit per §3.6 |

---

## 10. Slot ID mapping (for monitoring)

| Slot | Vertex region | Vertex job ID | yaml |
|---|---|---|---|
| A — BUNDLE | us-east1 | `8395459915946131456` | `R13_P2_SCRATCH_BUNDLE.yaml` |
| B — CORR_ONLY | us-west4 | `5580112770428305408` | `R13_P2_SCRATCH_CORR_ONLY.yaml` |
| C — PAIRRANK_ONLY | us-central1 | `3126673777023254528` | `R13_P2_SCRATCH_PAIRRANK_ONLY.yaml` |
| **D — FOURIER** | TBD (your launch) | TBD | `R13_P2_SCRATCH_FOURIER.yaml` (you create) |

W&B project (all 4): `dtect-vision/phase2-round13`.
GCP project: `train-cvit2`.

---

## 11. Self-check before submitting Cloud Build

Tick all five before running `./dev.sh build-prod -y`:

- [ ] All 3 smoke-test tiers passed (§3.7).
- [ ] `R13_P2_SCRATCH_FOURIER.yaml` parses with `python -c "import yaml; yaml.safe_load(open('experiments/phase2_round13/R13_P2_SCRATCH_FOURIER.yaml'))"`.
- [ ] `train_sweep.py` allowlist contains the `fourier_aug` re-apply block.
- [ ] `git status` shows no unrelated changes staged.
- [ ] You have read AGENTS.md, STATE.md, AGENT_GUIDE.md, packets/P2.md, threads/in_training_canary_signal.md, threads/processing_signature_shortcut.md §"Probe 6" subsection.

If any box is unchecked, fix it before continuing.
