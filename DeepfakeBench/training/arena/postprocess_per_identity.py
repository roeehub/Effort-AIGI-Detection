"""Per-identity reducer for videos_report.csv (WS-P2.b).

Groups video-level rows by `group_key` (identity) and recomputes per-identity
metrics at a given threshold. Aggregate counts across identities must equal
the suite aggregate — this is checked via --verify.

Typical use:

    python arena/postprocess_per_identity.py \
        --reports <suite>_<ckpt>_videos_report.csv [more...] \
        --threshold 0.5 \
        --out per_identity_<ckpt>.csv

With multiple --reports, the output contains one row per (report, group_key)
so cross-suite comparisons are easy.

Columns emitted:
    report, suite (inferred from filename), checkpoint (inferred), group_key,
    family_key, n_videos, n_real, n_fake, real_fpr, fake_recall,
    mean_prob, p50_prob, p90_prob
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class VideoRow:
    video_id: str
    label: int
    prob: float
    method: str
    group_key: str
    family_key: str


def _load_rows(path: str) -> List[VideoRow]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows: List[VideoRow] = []
        for raw in reader:
            rows.append(
                VideoRow(
                    video_id=str(raw.get("video_id", "")).strip(),
                    label=int(float(str(raw.get("label", "0")).strip() or "0")),
                    prob=float(str(raw.get("avg_video_prob", "0")).strip() or "0"),
                    method=str(raw.get("method", "")).strip(),
                    group_key=str(raw.get("group_key", "")).strip(),
                    family_key=str(raw.get("family_key", "")).strip(),
                )
            )
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def _infer_suite_and_ckpt(filename: str) -> Tuple[str, str]:
    """Filename convention: <suite>_<ckpt>_videos_report.csv"""
    stem = os.path.basename(filename)
    if stem.endswith("_videos_report.csv"):
        stem = stem[: -len("_videos_report.csv")]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, ""


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    frac = position - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _reduce_per_group(
    rows: Iterable[VideoRow],
    threshold: float,
) -> List[Dict[str, float | int | str]]:
    by_group: Dict[str, List[VideoRow]] = defaultdict(list)
    for r in rows:
        by_group[r.group_key].append(r)

    result: List[Dict[str, float | int | str]] = []
    for group_key in sorted(by_group):
        grp = by_group[group_key]
        probs = [r.prob for r in grp]
        reals = [r for r in grp if r.label == 0]
        fakes = [r for r in grp if r.label == 1]
        fp = sum(1 for r in reals if r.prob >= threshold)
        tp = sum(1 for r in fakes if r.prob >= threshold)
        # Multiple family_keys within one identity can occur across fake methods;
        # pick the most common, fall back to a sorted join if tied by count.
        fk_counts: Dict[str, int] = defaultdict(int)
        for r in grp:
            fk_counts[r.family_key] += 1
        family_key = max(fk_counts, key=lambda k: (fk_counts[k], k))

        result.append(
            {
                "group_key": group_key,
                "family_key": family_key,
                "n_videos": len(grp),
                "n_real": len(reals),
                "n_fake": len(fakes),
                "real_fpr": (fp / len(reals)) if reals else "",
                "fake_recall": (tp / len(fakes)) if fakes else "",
                "mean_prob": mean(probs) if probs else 0.0,
                "p50_prob": _quantile(probs, 0.50),
                "p90_prob": _quantile(probs, 0.90),
            }
        )
    return result


def _aggregate_check(
    rows: Iterable[VideoRow], per_group: Sequence[Dict[str, float | int | str]], threshold: float
) -> Dict[str, float]:
    rows_list = list(rows)
    probs = [r.prob for r in rows_list]
    n_real = sum(1 for r in rows_list if r.label == 0)
    n_fake = sum(1 for r in rows_list if r.label == 1)
    fp = sum(1 for r in rows_list if r.label == 0 and r.prob >= threshold)
    tp = sum(1 for r in rows_list if r.label == 1 and r.prob >= threshold)

    agg_real_fpr = (fp / n_real) if n_real else None
    agg_fake_recall = (tp / n_fake) if n_fake else None

    # Sum group fp-count and tp-count from per_group
    # (real_fpr * n_real_in_group) over groups should equal fp
    sum_fp = 0.0
    sum_tp = 0.0
    for g in per_group:
        if g["n_real"] and g["real_fpr"] != "":
            sum_fp += float(g["real_fpr"]) * float(g["n_real"])
        if g["n_fake"] and g["fake_recall"] != "":
            sum_tp += float(g["fake_recall"]) * float(g["n_fake"])

    return {
        "aggregate_real_fpr": agg_real_fpr if agg_real_fpr is not None else float("nan"),
        "aggregate_fake_recall": agg_fake_recall if agg_fake_recall is not None else float("nan"),
        "n_videos": len(rows_list),
        "n_real": n_real,
        "n_fake": n_fake,
        "fp_aggregate": fp,
        "tp_aggregate": tp,
        "fp_from_group_sum": sum_fp,
        "tp_from_group_sum": sum_tp,
        "mean_prob": mean(probs) if probs else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="One or more <suite>_<ckpt>_videos_report.csv files.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Decision threshold tau. prob >= tau => predicted fake.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output CSV: one row per (report, group_key).",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Check per-group sums equal aggregate counts within rounding. "
             "Fail with non-zero exit if they don't.",
    )
    args = ap.parse_args()

    all_out_rows: List[Dict[str, float | int | str]] = []
    any_verify_failure = False
    for report in args.reports:
        suite, ckpt = _infer_suite_and_ckpt(report)
        rows = _load_rows(report)
        per_group = _reduce_per_group(rows, threshold=args.threshold)
        for g in per_group:
            enriched = {
                "report": os.path.basename(report),
                "suite": suite,
                "checkpoint": ckpt,
                **g,
            }
            all_out_rows.append(enriched)

        if args.verify:
            agg = _aggregate_check(rows, per_group, threshold=args.threshold)
            fp_delta = abs(agg["fp_aggregate"] - agg["fp_from_group_sum"])
            tp_delta = abs(agg["tp_aggregate"] - agg["tp_from_group_sum"])
            tol = 1e-6
            status = "OK" if fp_delta <= tol and tp_delta <= tol else "MISMATCH"
            print(
                f"[verify] {os.path.basename(report)} tau={args.threshold:.4f} "
                f"fp_agg={agg['fp_aggregate']:.0f} fp_sum={agg['fp_from_group_sum']:.6f} "
                f"tp_agg={agg['tp_aggregate']:.0f} tp_sum={agg['tp_from_group_sum']:.6f} "
                f"real_fpr={agg['aggregate_real_fpr']:.4f} "
                f"fake_recall={agg['aggregate_fake_recall']:.4f} -> {status}",
                file=sys.stderr,
            )
            if status != "OK":
                any_verify_failure = True

    fieldnames = [
        "report",
        "suite",
        "checkpoint",
        "group_key",
        "family_key",
        "n_videos",
        "n_real",
        "n_fake",
        "real_fpr",
        "fake_recall",
        "mean_prob",
        "p50_prob",
        "p90_prob",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_out_rows:
            writer.writerow(row)

    print(f"Wrote {len(all_out_rows)} rows -> {args.out}")
    if any_verify_failure:
        print("FAIL: per-group sums did not match aggregates (see stderr).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
