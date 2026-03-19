#!/usr/bin/env python3
"""Deep-dive into live-deepfake-methods grouping and mixed labels."""
import csv
from collections import defaultdict, Counter

fpath = "inference_results/r9a_run1__live-deepfake-methods-real-and-fake-frames-cropped-teams.csv"
with open(fpath) as f:
    rows = list(csv.DictReader(f))

# Group by video_id
by_video = defaultdict(list)
for r in rows:
    by_video[r['video_id']].append(r)

# For mixed-label groups, show subgroup sizes
print("=== Mixed-label groups: subgroup analysis ===")
subgroup_sizes = {'fake': [], 'real': []}
all_subgroups = []  # (label, size)
for vid, frames in by_video.items():
    labels = set(r['label_str'] for r in frames)
    if len(labels) > 1:
        by_label = defaultdict(list)
        for r in frames:
            by_label[r['label_str']].append(r)
        for lbl, sub in by_label.items():
            subgroup_sizes[lbl].append(len(sub))
            all_subgroups.append((lbl, len(sub), vid))

for lbl in ['fake', 'real']:
    sizes = sorted(subgroup_sizes[lbl])
    if sizes:
        print(f"\n{lbl} subgroups in mixed groups: count={len(sizes)}, min={sizes[0]}, max={sizes[-1]}, median={sizes[len(sizes)//2]}")
        hist = Counter()
        for s in sizes:
            if s < 16: hist['<16'] += 1
            elif s < 24: hist['16-23'] += 1
            elif s < 32: hist['24-31'] += 1
            else: hist['32+'] += 1
        print(f"  Size distribution: {dict(sorted(hist.items()))}")

# Pure-label groups
print("\n=== Pure-label groups ===")
for vid, frames in by_video.items():
    labels = set(r['label_str'] for r in frames)
    if len(labels) == 1:
        lbl = list(labels)[0]
        n = len(frames)
        if n >= 16:  # Only show decent-sized ones
            pass  # just count them
pure_fake = [(vid, len(f)) for vid, f in by_video.items() 
             if len(set(r['label_str'] for r in f)) == 1 and f[0]['label_str'] == 'fake']
pure_real = [(vid, len(f)) for vid, f in by_video.items() 
             if len(set(r['label_str'] for r in f)) == 1 and f[0]['label_str'] == 'real']
print(f"Pure fake groups: {len(pure_fake)}, sizes: {sorted([s for _,s in pure_fake], reverse=True)[:20]}")
print(f"Pure real groups: {len(pure_real)}, sizes: {sorted([s for _,s in pure_real], reverse=True)[:20]}")

# Now compute: how many homogeneous windows can we make?
print("\n=== Achievable homogeneous windows per W ===")
for W in [16, 24, 32]:
    windows = {'fake': 0, 'real': 0}
    for vid, frames in by_video.items():
        by_label = defaultdict(list)
        for r in frames:
            by_label[r['label_str']].append(r)
        for lbl, sub in by_label.items():
            windows[lbl] += len(sub) // W
    total = windows['fake'] + windows['real']
    print(f"W={W}: fake_windows={windows['fake']}, real_windows={windows['real']}, total={total}")

# Also for the teams-flat bucket
print("\n\n=== teams-faces-data-test: alternative grouping ===")
fpath2 = "inference_results/r9a_run1__teams-faces-data-test-2914-fake-4420-real-feb-28.csv"
with open(fpath2) as f:
    rows2 = list(csv.DictReader(f))

# Check if video_id is meaningful
by_vid2 = defaultdict(list)
for r in rows2:
    by_vid2[r['video_id']].append(r)
    
biggest = sorted(by_vid2.items(), key=lambda x: len(x[1]), reverse=True)[:5]
print(f"Biggest groups by video_id:")
for vid, frames in biggest:
    labels = Counter(r['label_str'] for r in frames)
    print(f"  '{vid}': {len(frames)} frames, labels={dict(labels)}")

# The real group - what are the filenames like?
for vid, frames in by_vid2.items():
    if len(frames) > 100:
        print(f"\n  Sample filenames from big group '{vid}':")
        for r in frames[:5]:
            print(f"    {r['frame_name']}, label={r['label_str']}")
        break

# Check if we can make a better grouping from filenames
print("\nFilename patterns (first 10):")
for r in rows2[:10]:
    print(f"  {r['frame_name']} → vid={r['video_id']}, label={r['label_str']}")
