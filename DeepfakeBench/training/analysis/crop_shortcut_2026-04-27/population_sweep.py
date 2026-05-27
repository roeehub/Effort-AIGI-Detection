"""Run crop-sweep across the full production-honest pool.

Samples N frames from each of 6 tag folders, runs the sweep at a focused set
of tightness levels, aggregates results into a regime × tightness table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

from PIL import Image

import importlib
_crop = importlib.import_module("analysis.crop_shortcut_2026-04-27.crop_sweep")
make_variant = _crop.make_variant
run_check_frame = _crop.run_check_frame
OUT = _crop.OUT
RESULTS = _crop.RESULTS

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
PROD_FRAMES = REPO / "analysis/deployment_honest_eval_2026-04-27/_prod_cache/frames"

TAGS_OK = [
    "dor-real-laptop-correct-no-virtual-bg-whiteish",
    "dor-real-laptop-correct-no-virtual-bg-yellowish",
    "roee-real-windows-laptop-correct",
]
TAGS_FAIL = [
    "dor-real-webcam-false-flag",
    "dor-real-webcam-false-flag-no-virtual-bg",
    "roee-mac-laptop-false-flag-virtual-bg",
]


def regime_for(tag: str) -> str:
    return "FAIL" if "false-flag" in tag else "OK"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-tag", type=int, default=5)
    ap.add_argument(
        "--tightness", default="0.55,0.70,0.85,1.00,1.20",
        help="Focused tightness levels",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tightnesses = [float(x) for x in args.tightness.split(",")]
    rng = Random(args.seed)

    rows = []
    for tag in TAGS_OK + TAGS_FAIL:
        folder = PROD_FRAMES / tag
        if not folder.exists():
            print(f"[skip] missing {folder}")
            continue
        files = sorted(folder.glob("*.png"))
        rng.shuffle(files)
        sample = files[: args.n_per_tag]
        regime = regime_for(tag)
        print(f"\n=== {tag} ({regime}) — {len(sample)} frames ===")
        for fp in sample:
            img = Image.open(fp).convert("RGB")
            for t in tightnesses:
                v = make_variant(img, t)
                out_path = OUT / f"pop_{tag}_{fp.stem}__t{t:.2f}.png"
                v.save(out_path)
                prob, pred, _ = run_check_frame(out_path)
                rows.append({
                    "tag": tag, "regime": regime, "frame": fp.name,
                    "tightness": t, "prob_fake": prob, "pred_label": pred,
                })
                print(f"  {fp.name}  t={t:.2f}  prob_fake={prob:.3f}  pred={pred}")

    with open(RESULTS, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Aggregate
    print("\n\n=== AGGREGATE: mean prob_fake by (regime, tightness) ===")
    agg: dict[tuple[str, float], list[float]] = {}
    for r in rows:
        if r.get("prob_fake") is None:
            continue
        agg.setdefault((r["regime"], r["tightness"]), []).append(float(r["prob_fake"]))
    print(f"{'regime':<6} {'tight':<6} {'n':<4} {'mean':<7} {'p10':<7} {'p50':<7} {'p90':<7} {'frac>=0.5':<10}")
    for (regime, t), vs in sorted(agg.items()):
        vs.sort()
        n = len(vs)
        mean = sum(vs) / n
        p10 = vs[int(0.1 * n)]
        p50 = vs[n // 2]
        p90 = vs[min(n - 1, int(0.9 * n))]
        frac_fake = sum(1 for v in vs if v >= 0.5) / n
        print(f"{regime:<6} {t:<6.2f} {n:<4} {mean:<7.3f} {p10:<7.3f} {p50:<7.3f} {p90:<7.3f} {frac_fake:<10.2%}")

    print("\n\n=== AGGREGATE: per-tag mean prob_fake ===")
    pertag: dict[tuple[str, float], list[float]] = {}
    for r in rows:
        if r.get("prob_fake") is None:
            continue
        pertag.setdefault((r["tag"], r["tightness"]), []).append(float(r["prob_fake"]))
    last_tag = None
    for (tag, t), vs in sorted(pertag.items()):
        if tag != last_tag:
            print(f"\n{tag}:")
            last_tag = tag
        mean = sum(vs) / len(vs)
        frac_fake = sum(1 for v in vs if v >= 0.5) / len(vs)
        print(f"  t={t:<5.2f}  mean={mean:<7.3f}  fpr={frac_fake:<6.2%} n={len(vs)}")


if __name__ == "__main__":
    main()
