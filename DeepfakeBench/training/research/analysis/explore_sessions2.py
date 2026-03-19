#!/usr/bin/env python3
"""Re-group ALL teams-flat frames by extracting session from any pattern."""
import csv, re
from collections import defaultdict, Counter

fpath = "inference_results/r9a_run1__teams-faces-data-test-2914-fake-4420-real-feb-28.csv"
with open(fpath) as f:
    rows = list(csv.DictReader(f))

# Try broader pattern: any prefix__s{N}_
session_groups = defaultdict(list)
no_match = []
for r in rows:
    m = re.search(r'__s(\d+)_', r['frame_name'])
    if m:
        session_groups[f"s{m.group(1)}"].append(r)
    else:
        no_match.append(r)

print(f"Sessions found: {len(session_groups)}")
print(f"Frames matched: {sum(len(v) for v in session_groups.values())}")
print(f"Frames without match: {len(no_match)}")

for sess in sorted(session_groups.keys(), key=lambda x: int(x[1:])):
    frames = session_groups[sess]
    labels = Counter(r['label_str'] for r in frames)
    n_fake = labels.get('fake', 0)
    n_real = labels.get('real', 0)
    print(f"  {sess}: {len(frames)} frames (fake={n_fake}, real={n_real})")

if no_match:
    print(f"\nNon-matching filenames (sample 10):")
    for r in no_match[:10]:
        print(f"  {r['frame_name']} label={r['label_str']}")
    # Check misc patterns
    pfx = Counter()
    for r in no_match:
        fn = r['frame_name']
        bits = fn.split('_')
        pfx[bits[0]] += 1
    print(f"  Name prefixes: {dict(pfx.most_common(10))}")
