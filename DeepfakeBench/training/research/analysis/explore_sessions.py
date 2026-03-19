#!/usr/bin/env python3
"""Try to re-group teams-flat by session ID from filenames."""
import csv, re
from collections import defaultdict, Counter

fpath = "inference_results/r9a_run1__teams-faces-data-test-2914-fake-4420-real-feb-28.csv"
with open(fpath) as f:
    rows = list(csv.DictReader(f))

# Parse session from filename: Cam_Test__s{session}_{segment}_frame_{num}_crop_{num}__{hash}.jpg
session_groups = defaultdict(list)
no_match = []
for r in rows:
    m = re.match(r'Cam_Test__s(\d+)_', r['frame_name'])
    if m:
        session_groups[f"s{m.group(1)}"].append(r)
    else:
        no_match.append(r)

print(f"Sessions found: {len(session_groups)}")
print(f"Frames without session match: {len(no_match)}")

for sess in sorted(session_groups.keys(), key=lambda x: int(x[1:])):
    frames = session_groups[sess]
    labels = Counter(r['label_str'] for r in frames)
    print(f"  {sess}: {len(frames)} frames → {dict(labels)}")
    
# For frames that don't match, check patterns
if no_match:
    print(f"\nNon-matching filenames (first 10):")
    for r in no_match[:10]:
        print(f"  {r['frame_name']} label={r['label_str']}")
    # Try other patterns
    patterns = Counter()
    for r in no_match:
        fn = r['frame_name']
        if '__s' in fn:
            patterns['has_s'] += 1
        elif fn.startswith('real_'):
            patterns['real_prefix'] += 1
        else:
            patterns['other'] += 1
    print(f"  Pattern breakdown: {dict(patterns)}")
