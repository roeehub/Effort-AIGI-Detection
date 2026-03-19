#!/usr/bin/env python3
"""Explore the grouping structure of inference CSVs to plan window-based strategies."""
import csv
from collections import defaultdict

RESULTS_DIR = "inference_results"
FILES = [
    f"{RESULTS_DIR}/r9a_run1__teams-faces-data-test-2914-fake-4420-real-feb-28.csv",
    f"{RESULTS_DIR}/r9a_run1__live-deepfake-methods-real-and-fake-frames-cropped-teams.csv",
    f"{RESULTS_DIR}/r9a_run1__poc-phase-1-test.csv",
]

for fpath in FILES:
    print(f"\n{'='*70}")
    print(f"FILE: {fpath.split('/')[-1]}")
    with open(fpath) as f:
        rows = list(csv.DictReader(f))
    
    # Show first 3 rows
    print(f"\nFirst 3 rows (key fields):")
    for r in rows[:3]:
        print(f"  label_str={r['label_str']}, method={r['method']}, video_id={r['video_id']}, "
              f"frame_name={r['frame_name']}, strategy={r['strategy']}, prob_fake={r['prob_fake']}")
    
    # Group by video_id
    by_video = defaultdict(list)
    for r in rows:
        vid = r['video_id'] or r['strategy'] or r['frame_name']  # fallback grouping
        by_video[vid].append(r)
    
    # Distribution of frames per video
    lens = sorted([len(v) for v in by_video.values()])
    print(f"\nTotal frames: {len(rows)}")
    print(f"Unique groups (video_id/strategy): {len(by_video)}")
    print(f"Frames per group: min={lens[0]}, max={lens[-1]}, median={lens[len(lens)//2]}")
    
    # Histogram of group sizes
    from collections import Counter
    size_hist = Counter()
    for l in lens:
        if l == 1: size_hist["1"] += 1
        elif l <= 8: size_hist["2-8"] += 1
        elif l <= 16: size_hist["9-16"] += 1
        elif l <= 24: size_hist["17-24"] += 1
        elif l <= 32: size_hist["25-32"] += 1
        elif l <= 64: size_hist["33-64"] += 1
        elif l <= 128: size_hist["65-128"] += 1
        else: size_hist["129+"] += 1
    print(f"Group size distribution:")
    for bucket in ["1", "2-8", "9-16", "17-24", "25-32", "33-64", "65-128", "129+"]:
        if bucket in size_hist:
            print(f"  {bucket:>8}: {size_hist[bucket]} groups")
    
    # Label distribution within groups
    label_by_group = {}
    for vid, frames in by_video.items():
        labels = set(r['label_str'] for r in frames)
        label_by_group[vid] = labels
    
    mixed = sum(1 for v in label_by_group.values() if len(v) > 1)
    print(f"Groups with mixed labels: {mixed}")
    
    # Method distribution by group
    methods = defaultdict(int)
    for vid, frames in by_video.items():
        m = frames[0]['method']
        methods[m] += 1
    print(f"Groups per method: {dict(sorted(methods.items()))}")
    
    # For groups >= 16, show label breakdown
    groups_ge16 = {k: v for k, v in by_video.items() if len(v) >= 16}
    groups_ge24 = {k: v for k, v in by_video.items() if len(v) >= 24}
    groups_ge32 = {k: v for k, v in by_video.items() if len(v) >= 32}
    total_frames_ge16 = sum(len(v) for v in groups_ge16.values())
    total_frames_ge24 = sum(len(v) for v in groups_ge24.values())
    total_frames_ge32 = sum(len(v) for v in groups_ge32.values())
    
    print(f"\nGroups with >= 16 frames: {len(groups_ge16)} ({total_frames_ge16} frames)")
    print(f"Groups with >= 24 frames: {len(groups_ge24)} ({total_frames_ge24} frames)")
    print(f"Groups with >= 32 frames: {len(groups_ge32)} ({total_frames_ge32} frames)")
    
    # Show a few sample group IDs with their sizes
    print(f"\nSample groups (first 5):")
    for vid in list(by_video.keys())[:5]:
        frames = by_video[vid]
        print(f"  '{vid}': {len(frames)} frames, label={frames[0]['label_str']}, method={frames[0]['method']}")
