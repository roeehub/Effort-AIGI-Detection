# Rerun Instructions — Session 20260227_010941

## Problem

Session 3 captured **38,038 faces** over 11.1 hours, but the WMA face detector
only produced usable crops for **278 / 1,179** sample pairs (23.6%).
The remaining **901 pairs** (76.4%) have at least one side with fewer than 3 good
face crops.

### Failure Breakdown

| Category | Count | Description |
|---|---|---|
| Both sides complete | 278 | No rerun needed |
| Needs fake only | 105 | Real is fine, fake got 0-2 faces |
| Needs real only | 88 | Fake is fine, real got 0-2 faces |
| Needs both | 708 | Neither side has ≥3 faces |
| **Total to rerun** | **901** | |

### Per-Strategy

| Strategy | Total | Complete | Needs Rerun |
|---|---|---|---|
| edge_cases | 434 | 164 | 270 |
| minimal_processing | 425 | 70 | 355 |
| quality_enhancement | 320 | 44 | 276 |

### Time Estimate

- 901 pairs × 2 videos × 12s = 21,624s video time
- 901 × 3s (A sep) + 901 × 3s (B sep) + 3 × 7s (C sep) = 5,427s separator time
- **Total: ~27,051s ≈ 7.5 hours**

---

## Files

- **`rerun_manifest.json`** — Machine-readable list of all 901 sample pairs that
  need re-streaming. Each entry contains:
  - `sample_id`, `strategy`, `real_index`, `fake_index` (from original playlist)
  - `real_faces`, `fake_faces` (count of good crops from session 3)
  - `real_status`, `fake_status` ("complete", "partial", or "empty")
  - `needs_real`, `needs_fake` (boolean — which side failed)

- **`obs_full_pipeline.py`** — The original pipeline script (on broadcast machine)

---

## Required Changes to `obs_full_pipeline.py`

### Approach: `--rerun-manifest` Mode

Add a new CLI flag `--rerun-manifest <path>` that replaces the GCS discovery +
playlist building with the curated list from `rerun_manifest.json`. The rest of
the pipeline (separator logic, download, playback, logging) stays identical.

### Suggested Code Changes

#### 1. New CLI argument (in `main()`, argparse block)

```python
parser.add_argument("--rerun-manifest", type=str, default=None,
                    help="Path to rerun_manifest.json — only stream listed samples")
```

#### 2. Replace playlist building logic (in `main()`, after GCS client setup)

Replace the block that does `discover_samples()` → `build_playlist()` with a
conditional:

```python
if args.rerun_manifest:
    # ── Rerun mode: load manifest instead of full discovery ──
    print(f"\n🔄 RERUN MODE: loading {args.rerun_manifest}")
    with open(args.rerun_manifest) as f:
        manifest = json.load(f)

    rerun_list = manifest["rerun_samples"]
    print(f"   {len(rerun_list)} sample pairs to re-stream")

    # Build strategy_samples from manifest (preserves strategy grouping)
    strategy_samples = {s: [] for s in STRATEGY_ORDER}
    for entry in rerun_list:
        strategy_samples[entry["strategy"]].append(entry["sample_id"])

    # They're already sorted in the manifest, but sort again to be safe
    for strat in STRATEGY_ORDER:
        strategy_samples[strat].sort()

    playlist = build_playlist(strategy_samples)
else:
    # ── Normal mode: discover all samples from GCS ──
    strategy_samples = discover_samples(bucket)

    if args.test:
        print("\n⚡ TEST MODE: limiting to 2 samples per strategy")
        for strat in STRATEGY_ORDER:
            strategy_samples[strat] = strategy_samples[strat][:2]

    playlist = build_playlist(strategy_samples)
```

#### 3. Logger — tag the session as a rerun

In the `session_start` log entry, add the rerun flag so the reconstruction
pipeline knows this is a continuation:

```python
logger.log("session_start", playlist_entries=len(playlist),
           play_duration=args.play_duration,
           test_mode=args.test,
           resume_from=args.resume_from,
           rerun_mode=bool(args.rerun_manifest),              # ← ADD
           rerun_source=manifest["metadata"]["source_session"] # ← ADD
               if args.rerun_manifest else None)
```

#### 4. No other changes needed

The `build_playlist()` function already assigns fresh sequential indices starting
from 0. The separator logic (A/B/C) is driven by the playlist structure, so it
will automatically produce the correct pattern:

```
[C 7s]  ← edge_cases header
  [A 3s] → edge_cases_0122/real.mp4 → [B 3s] → edge_cases_0122/fake.mp4
  [A 3s] → edge_cases_0154/real.mp4 → [B 3s] → edge_cases_0154/fake.mp4
  ...
[C 7s]  ← minimal_processing header
  ...
[C 7s]  ← quality_enhancement header
  ...
```

The reconstruction script on the WMA machine will see the same separator-based
segmentation and assign faces to segments as before.

---

## Important Notes

### Why re-stream BOTH sides even if only one failed?

The WMA face capture pipeline uses separator images (A/B/C) to segment the
continuous face stream into individual video clips. The structure is always:

```
A_sep → real video → B_sep → fake video → A_sep → ...
```

If we only streamed the `fake` side of a sample, there would be no `A_sep` or
`B_sep` around it, and the reconstruction script wouldn't know where the segment
boundaries are. So we must always stream the full pair.

### New session directory

The rerun will produce a **new session** on the WMA machine (e.g.
`session_20260228_XXXXXX`). Run the reconstruction pipeline on it separately,
then merge the results with the session 3 `teams_dataset/`:

```
# After reconstruction of the rerun session:
# For each sample in rerun, if the rerun has more faces, use those instead
```

### Merge Strategy (post-rerun)

After reconstruction of the rerun session, for each sample pair in the manifest:

1. Compare face counts: `session_3/{sample_id}/{type}/` vs `rerun/{sample_id}/{type}/`
2. Keep whichever has more good faces (or the rerun if both are equal)
3. Copy the winner into a unified `teams_dataset_merged/` directory

A simple merge script can automate this using the manifest.

### `--resume-from` still works

If the rerun itself crashes partway through, `--resume-from <index>` can pick up
where it left off, just like in the original run.

---

## Quick Start

On the **broadcast machine**:

```bash
# 1. Copy rerun_manifest.json to the broadcast machine
# 2. Run the modified pipeline:
python obs_full_pipeline.py \
    --password myPass \
    --rerun-manifest rerun_manifest.json \
    --output-dir ./rerun_session

# Or test with a small batch first:
python obs_full_pipeline.py \
    --password myPass \
    --rerun-manifest rerun_manifest.json \
    --test \
    --output-dir ./rerun_test
```

On the **WMA machine**: make sure face capture is running and pointed at the
Teams call before starting the broadcast.
