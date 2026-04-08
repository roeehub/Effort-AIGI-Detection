# Training Speedup Plan — R13+ Experiments

**Date:** 2026-04-08
**Target:** Reduce 10K-step wall-clock from ~28h to ~16-18h on `a2-highgpu-1g` (1× A100 40GB, 12 vCPUs, 85 GB RAM)
**Constraint:** Zero change to training dynamics — same gradients, same convergence, same results.

---

## Problem Statement

Training has gotten progressively slower from R3 → R13 due to:

1. **7 data sources** (was 3 in R3) — more GCS I/O per step
2. **Per-frame sequential GCS downloads** — no parallelism within each DataLoader worker
3. **Uncached `storage.Client()` in visomaster loaders** — new HTTP+credential setup per sample
4. **~50-100 CUDA syncs per step** from the gradient norm `.item()` loop
5. **Heavy OOD monitoring** (~3,200 videos × 8 frames) starting at step 500
6. **Only 4/12 vCPUs utilized** for data loading

The GPU (A100) is data-starved: ViT-B-16 forward on 32 fp16 frames takes ~10ms, but fetching those frames from GCS takes 80-240ms per sample.

---

## Approved Changes

### 1. Delay OOD Monitoring Start to Step 5000

**Config change only.**

```yaml
ood_monitoring_start_step: 5000   # was: 500
ood_monitoring_every_steps: 1000  # unchanged
```

**What it does:** Skips ~4-5 OOD monitoring events in the first 5K steps. Each OOD event processes ~3,200 videos × 8 frames = ~25,600 GCS downloads + full model inference, blocking training for 5-15 minutes.

**Estimated saving:** ~30-60 minutes over a 10K-step run.

**Risk:** None. Early OOD metrics are noisy and not actionable — the model hasn't converged enough for OOD signal to be meaningful. OOD-composite checkpointing only matters in the later stages anyway.

---

### 2. Reduce OOD VCD Real Count

**Config change only.**

```yaml
# In ood_monitoring.external_real_sources, the VCD entry:
- bucket: "effort-collected-data"
  prefix: "real/VCD"
  method: "zoom_vcd_real"
  grouping: "per_image"
  max_videos: 300          # was: 1200
  deterministic: true
```

Also update `expected_counts` to match:

```yaml
expected_counts:
  zoom_vcd_real: 300         # was: 1200
```

**What it does:** Reduces VCD real evaluation from 1,200 to 300 videos per OOD pass. The total OOD set drops from ~3,200 to ~2,300 videos — a ~28% reduction per OOD event.

**Estimated saving:** ~2-4 minutes per OOD pass. With OOD running from step 5000 at every 1000 steps, that's 5 passes in a 10K run → ~10-20 minutes saved.

**Risk:** Very low. 300 VCD reals is sufficient for a reliable trend signal (AUC/EER on VCD real direction). The 1,200 count was over-sampled. The other OOD sources (YouTube AVSpeech 200, WMA failure fake 1,202, Teams OOD real/fake 300 each) are unchanged and already smaller.

---

### 3. Increase `num_workers` from 4 to 8

**Config change only.**

```yaml
num_workers: 8     # was: 4
```

**What it does:** Doubles the number of DataLoader worker processes fetching frames from GCS in parallel. Each worker gets ~12.5% of identities (was 25%) and downloads sequentially within its shard, but more workers = more concurrent GCS connections = better pipeline overlap with GPU compute.

**Estimated saving:** ~10-20% step throughput improvement. The bottleneck is GCS I/O latency, not CPU. More workers keep the GPU fed.

**Risk:** Very low. The `a2-highgpu-1g` node has **12 vCPUs**. With 8 workers + 1 main process, that's 9/12 vCPUs utilized — well within capacity. System RAM (85 GB) is more than sufficient for 8 workers' buffers. The main thread still has 3 vCPUs for collation, model, and optimizer.

**Machine spec verification:**
- `a2-highgpu-1g`: 12 vCPUs, 85 GB RAM, 1× A100 40GB
- Current usage: 4 workers + 1 main = 5 vCPUs (~42% utilization)
- Proposed usage: 8 workers + 1 main = 9 vCPUs (~75% utilization)

---

### 4. Increase `prefetch_factor` from 2 to 4

**Config change only.**

```yaml
prefetch_factor: 4   # was: 2
```

**What it does:** Each worker pre-fetches 4 batches ahead instead of 2. With 8 workers × 4 prefetch = 32 batches buffered, the GPU has a deeper runway of data before it stalls.

**Estimated saving:** Reduces GPU idle time waiting for data. Most impactful when combined with `num_workers=8`. Hard to quantify independently — contributes to the overall I/O throughput gain.

**Risk:** Very low. Extra RAM per worker for buffering — each batch is ~32 frames × 224×224×3 × 4 bytes ≈ 19 MB. With 8 workers × 4 prefetch × 19 MB ≈ 600 MB total buffer. Negligible relative to 85 GB system RAM.

---

### 5. Fix Gradient Norm `.item()` Loop

**Code change in `trainer/trainer.py`.**

**Current code** (lines 1266-1276):
```python
if hasattr(self, 'gradient_clip_val') and self.gradient_clip_val:
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

# Compute gradient health metrics before they're cleared
model_ref = self.model.module if is_ddp else self.model
_grad_norm = 0.0
_num_params_with_grad = 0
for p in model_ref.parameters():
    if p.grad is not None:
        _grad_norm += p.grad.data.norm(2).item() ** 2   # ← CUDA sync per param!
        _num_params_with_grad += 1
```

**Problem:** The loop iterates all model parameters (~200+) and calls `.norm(2).item()` on each one with a gradient. Each `.item()` forces a CUDA synchronize — the CPU blocks waiting for the GPU to finish all pending work. With ~50-100 parameters having gradients, this creates 50-100 GPU pipeline stalls **every single training step**.

Meanwhile, `torch.nn.utils.clip_grad_norm_()` — called 3 lines above — already computes the exact same total gradient norm internally but **its return value is discarded**.

**Proposed fix:** Capture the return value of `clip_grad_norm_` (which is the total norm) and use a single `.item()` call on it:

```python
if hasattr(self, 'gradient_clip_val') and self.gradient_clip_val:
    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
else:
    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))

_grad_norm = total_norm.item()  # single CUDA sync
_num_params_with_grad = sum(1 for p in model_ref.parameters() if p.grad is not None)
```

**Estimated saving:** Eliminates 50-100 CUDA syncs per step → ~5-15% wall-clock speedup. Over 10K steps, that's 500K-1M unnecessary sync points removed.

**Risk:** Very low.
- `clip_grad_norm_` with `max_norm=inf` computes the total norm without clipping — pure read operation.
- The logged `_grad_norm` value will be **mathematically identical** (it's the same L2 norm computation).
- The `_num_params_with_grad` count is still computed but via a simple Python generator (no CUDA sync needed — `p.grad is not None` is a CPU pointer check).
- No gradient values are changed. No optimizer behavior change. Only the *observation* of the norm avoids redundant GPU syncs.

---

### 6. Add `persistent_workers=True` to DataLoader

**Code change in `data/sources/combined_paired.py`.**

**Current code** (DataLoader construction):
```python
train_loader = DataLoader(
    train_iterable,
    batch_size=batching_config.batch_size,
    num_workers=batching_config.num_workers,
    prefetch_factor=batching_config.prefetch_factor if num_workers > 0 else None,
    collate_fn=combined_paired_collate_fn,
    pin_memory=True,
)
```

**Proposed fix:**
```python
train_loader = DataLoader(
    train_iterable,
    batch_size=batching_config.batch_size,
    num_workers=batching_config.num_workers,
    prefetch_factor=batching_config.prefetch_factor if num_workers > 0 else None,
    collate_fn=combined_paired_collate_fn,
    pin_memory=True,
    persistent_workers=True if batching_config.num_workers > 0 else False,
)
```

**What it does:** Workers stay alive between epochs instead of being destroyed and recreated. This preserves:
- Python process state (no fork/spawn overhead)
- Any cached `storage.Client()` instances within the worker (especially relevant for visomaster client caching in change #7)
- Warm TCP connections to GCS

**Estimated saving:** Eliminates per-epoch worker respawn delay. With `IterableDataset`, each epoch boundary currently kills and recreates 8 Python processes (including credential resolution). This saves ~5-15 seconds per epoch × 5-10 epochs in a 10K-step run.

**Risk:** Very low. Standard PyTorch feature (`persistent_workers` was introduced in PyTorch 1.8). The only consideration is that workers hold their memory between epochs, but worker state is minimal (sample list + GCS client) and 85 GB RAM is ample. Compatible with `IterableDataset`.

---

### 7. Cache `storage.Client()` Per Worker in VisoMaster Loaders

**Code change in `data/sources/visomaster.py`.**

**Problem:** Three functions create a **new `storage.Client()`** on every call:

| Line | Function |
|------|----------|
| L467 | `load_visomaster_frames()` |
| L1401 | `load_visomaster_enhanced_frames()` |
| L1471 | `load_visomaster_teams_enhanced_frames()` |

Each `storage.Client()` involves credential resolution from the GCE metadata server + HTTP session setup. In R13, visomaster-family sources are heavily weighted (`visomaster_fake: 2.5`, `visomaster_enhanced_fake: 3.5`), so this overhead compounds significantly.

**Proven pattern already in codebase:** The teams sub-source caches its client at `combined_paired.py` L2234-2237:
```python
if not hasattr(self, '_teams_gcs_client'):
    self._teams_gcs_client = storage.Client()
client = self._teams_gcs_client
```

**Proposed fix:** Apply the same pattern. Since these are free functions (not methods on the IterableDataset), use a module-level `_thread_local` or pass a client parameter:

Option A — Add an optional `client` parameter and cache in the caller:
```python
def load_visomaster_frames(sample, anchor_indices, as_array=True, client=None):
    if client is None:
        client = storage.Client()
    bucket = client.bucket(sample.bucket_name)
    ...
```

Then in `_iterate_visomaster_sample()`, cache the client on `self` (same as teams):
```python
if not hasattr(self, '_visomaster_gcs_client'):
    self._visomaster_gcs_client = storage.Client()
real_frames, fake_frames = load_visomaster_frames(
    sample, anchor_indices, client=self._visomaster_gcs_client
)
```

Same for `_iterate_visomaster_enhanced_sample()` and `_iterate_visomaster_teams_enhanced_sample()`.

**Estimated saving:** ~5-15ms saved per visomaster-family sample. With these sources comprising ~40-50% of training samples (due to high weights), that's significant cumulative savings — roughly 5-10% of per-step data loading time.

**Risk:** Very low.
- Proven pattern already working in production for the teams source.
- `storage.Client` is documented as thread-safe and reusable.
- With `persistent_workers=True` (change #6), the cached client survives across epochs, making this even more effective.
- No change to what data is loaded or how it's processed.

---

### 8. Parallel Frame Downloads Within Each Worker

**Code change in `data/sources/visomaster.py` (and similar for other sources).**

**Current behavior:** Each worker downloads 16 frames per sample (8 real + 8 fake) **sequentially**:
```python
for idx in anchor_indices:       # 8 iterations
    blob = bucket.blob(real_path)
    data = blob.download_as_bytes()   # ~10ms blocking
    ...
    blob = bucket.blob(fake_path)
    data = blob.download_as_bytes()   # ~10ms blocking
```

Total: ~160ms per sample (16 × ~10ms).

**Proposed fix:** Use `concurrent.futures.ThreadPoolExecutor` within each worker to download frames in parallel:
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def _download_one(bucket, blob_path):
    blob = bucket.blob(blob_path)
    return blob.download_as_bytes()

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {}
    for idx in anchor_indices:
        futures[pool.submit(_download_one, bucket, real_path)] = ('real', idx)
        futures[pool.submit(_download_one, bucket, fake_path)] = ('fake', idx)
    for future in as_completed(futures):
        label, idx = futures[future]
        data = future.result()
        ...
```

Total: ~40ms per sample (16 downloads in ~4 concurrent batches).

**Estimated saving:** ~75% reduction in per-sample GCS latency. This is likely the **single biggest speedup** — data loading is the primary bottleneck. Could cut per-step time nearly in half.

**Risk:** Medium.
- `google.cloud.storage.Client` is thread-safe — concurrent downloads from the same bucket are supported.
- `ThreadPoolExecutor` within a DataLoader worker is standard Python.
- **Must validate** that frame ordering is preserved (the futures dict tracks index + label), and that PIL `Image.open(io.BytesIO(data))` is safe across threads (it is — each thread gets its own BytesIO).
- **Must validate** training results are bitwise-identical before/after by running a short comparison (100 steps with fixed seed, compare loss curves).
- Adds ~4 threads per worker × 8 workers = 32 threads total. The GCE network stack and GCS API handle this easily.

---

## Not Implementing (Would Change Training)

| Idea | Why not |
|------|---------|
| **Increase batch size (32 → 64/128)** | Changes effective gradient noise, learning rate dynamics, and convergence behavior. Would require full LR re-tuning and re-validation. Not a "free" speedup. |
| **`torch.compile`** | PyTorch 2.1.1 has known edge cases with custom autograd functions and dynamic shapes. `SVDResidualLinear` and `ArcMarginProduct` are non-standard modules. Risk of silent numerical differences or compilation failures. Worth exploring as a separate R&D track, not a training speedup patch. |
| **Frame-level disk caching** | 200 GB boot SSD is shared with OS, model weights (~350 MB), CLIP backbone (~600 MB). A full frame cache for 7 data sources would be 10-50 GB, adding disk pressure and cache invalidation complexity. The parallel download approach (change #8) addresses the same latency bottleneck more directly. |

---

## Implementation Order

Recommended sequencing (easiest/safest first):

1. **Config changes** (#1, #2, #3, #4) — zero risk, apply to next experiment YAML
2. **`persistent_workers`** (#6) — one-line addition, standard PyTorch
3. **Gradient norm fix** (#5) — small, self-contained, big impact
4. **VisoMaster client caching** (#7) — proven pattern from teams source
5. **Parallel frame downloads** (#8) — biggest impact but needs validation run

Changes 1-4 can ship together. Change 5 requires a short validation run (100 steps, fixed seed, compare loss curves) before trusting it in production.

---

## Expected Outcome

| Scenario | 10K step time |
|----------|--------------|
| Current (R13 baseline) | ~28 hours |
| Config changes only (#1-4) | ~22-24 hours |
| + Code changes #5-7 | ~18-20 hours |
| + Parallel downloads #8 | ~15-18 hours |
