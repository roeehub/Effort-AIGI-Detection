#!/usr/bin/env python3
"""Quick stats check for batch inference CSVs."""
import csv, sys, glob
from collections import Counter, defaultdict

pattern = sys.argv[1] if len(sys.argv) > 1 else "inference_results/*.csv"
files = sorted(glob.glob(pattern))

for fpath in files:
    print(f"\n{'='*60}")
    print(f"FILE: {fpath}")
    with open(fpath) as f:
        rows = list(csv.DictReader(f))
    print(f"Total rows: {len(rows)}")
    print(f"Status: {dict(Counter(r['status'] for r in rows))}")
    print(f"Labels: {dict(Counter(r['label_str'] for r in rows))}")

    for label_str in ['fake', 'real']:
        subset = [r for r in rows if r['label_str'] == label_str]
        if not subset:
            continue
        probs = [float(r['prob_fake']) for r in subset]
        mean_p = sum(probs) / len(probs)
        if label_str == 'fake':
            correct = sum(1 for p in probs if p >= 0.5)
        else:
            correct = sum(1 for p in probs if p < 0.5)
        acc = correct / len(subset)
        print(f"  {label_str}: n={len(subset)}, mean_prob_fake={mean_p:.4f}, acc@0.5={acc:.4f}")

    by_method = defaultdict(list)
    for r in rows:
        by_method[r['method']].append((float(r['prob_fake']), int(r['label'])))
    if any(m for m in by_method if m):
        print("\nPer-method stats:")
        for m in sorted(by_method):
            if not m:
                continue
            items = by_method[m]
            probs = [p for p, _ in items]
            labels = [l for _, l in items]
            mean_p = sum(probs) / len(probs)
            # acc: fake (label=1) should have prob>=0.5, real (label=0) should have prob<0.5
            correct = sum(1 for p, l in items if (l == 1 and p >= 0.5) or (l == 0 and p < 0.5))
            acc = correct / len(items)
            print(f"  {m}: n={len(items)}, mean_prob={mean_p:.4f}, acc@0.5={acc:.4f}")
