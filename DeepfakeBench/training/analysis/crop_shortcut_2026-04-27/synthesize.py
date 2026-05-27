"""Aggregate crop_sweep_results.jsonl into a synthesis report.

For each frame, produce: native prob_fake, swing-magnitude across crops,
best-alternative crop, and a summary of TTA aggregation strategies (mean,
median, min, max) over canonical neighborhoods.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
RESULTS = REPO / "analysis/crop_shortcut_2026-04-27/crop_sweep_results.jsonl"
OUT = REPO / "analysis/crop_shortcut_2026-04-27/synthesis.md"

# Frames are keyed by (tag, source-or-frame-name).
# Each row: {tag, regime?, source/frame, tightness, prob_fake, pred_label}


def read_rows() -> list[dict]:
    rows = []
    with open(RESULTS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("prob_fake") is None:
                continue
            rows.append(r)
    return rows


def regime_for_tag(tag: str) -> str:
    if tag.startswith("dor_real_OK_") or "windows-laptop-correct" in tag or "laptop-correct-no-virtual-bg" in tag:
        return "OK_real"
    if "false-flag" in tag or tag.endswith("_FAIL_webcam") or tag.endswith("_FAIL_mac"):
        return "FAIL_real"
    if "deeplive" in tag.lower() or "fake" in tag.lower() or "diverse" in tag.lower():
        return "FAKE"
    return "?"


def frame_key(r: dict) -> tuple[str, str]:
    src = r.get("frame") or Path(r.get("source", "?")).name
    return (r["tag"], src)


def main() -> None:
    rows = read_rows()
    by_frame: dict[tuple[str, str], dict[float, float]] = defaultdict(dict)
    for r in rows:
        k = frame_key(r)
        by_frame[k][float(r["tightness"])] = float(r["prob_fake"])

    lines: list[str] = []

    lines.append("# Crop-tightness shortcut: synthesis (2026-04-27)\n")
    lines.append(f"Total frames sweept: {len(by_frame)}\n")
    lines.append(f"Total predictions: {len(rows)}\n\n")

    # Per-frame summary table.
    lines.append("## Per-frame swing summary\n\n")
    lines.append("| regime | tag | frame | t=1.00 | min | max | best_t | worst_t | flips? |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")

    flip_count = 0
    swing_buckets: dict[str, list[float]] = defaultdict(list)
    best_minus_native: dict[str, list[float]] = defaultdict(list)

    for (tag, frame), tps in sorted(by_frame.items()):
        regime = regime_for_tag(tag)
        if 1.00 not in tps:
            continue
        native = tps[1.00]
        vals = list(tps.values())
        mn, mx = min(vals), max(vals)
        best_t = min(tps.items(), key=lambda kv: kv[1])[0]
        worst_t = max(tps.items(), key=lambda kv: kv[1])[0]
        flips = (mn < 0.5) != (mx < 0.5)
        if flips:
            flip_count += 1
        swing = mx - mn
        swing_buckets[regime].append(swing)
        # For real frames: best_minus_native is "improvement" (lower is better, so native - best)
        # For fake frames: best is the highest prob_fake. So native - max.
        if regime in ("OK_real", "FAIL_real"):
            best_minus_native[regime].append(native - mn)  # >0 means perturb helped
        else:
            best_minus_native[regime].append(mx - native)
        flips_s = "YES" if flips else ""
        lines.append(
            f"| {regime} | {tag[:35]} | {frame[:30]} | {native:.3f} | {mn:.3f} | "
            f"{mx:.3f} | {best_t:.2f} | {worst_t:.2f} | {flips_s} |\n"
        )

    # Aggregate by regime and tightness
    by_regime_t: dict[tuple[str, float], list[float]] = defaultdict(list)
    for (tag, frame), tps in by_frame.items():
        regime = regime_for_tag(tag)
        for t, p in tps.items():
            by_regime_t[(regime, t)].append(p)

    lines.append(f"\n**Frames whose crop-perturbation flipped the pred ≥0.5 boundary**: {flip_count} / {len(by_frame)}\n\n")

    lines.append("## Aggregate by (regime, tightness)\n\n")
    lines.append("| regime | tightness | n | mean | median | p10 | p90 | frac_FAKE |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for (regime, t), vs in sorted(by_regime_t.items()):
        vs = sorted(vs)
        n = len(vs)
        m = mean(vs)
        md = median(vs)
        p10 = vs[max(0, int(0.1 * n))]
        p90 = vs[min(n - 1, int(0.9 * n))]
        frac = sum(1 for v in vs if v >= 0.5) / n
        lines.append(
            f"| {regime} | {t:.2f} | {n} | {m:.3f} | {md:.3f} | {p10:.3f} | {p90:.3f} | {frac:.0%} |\n"
        )

    # TTA aggregation evaluation: for each frame, compute mean/median/min/max
    # across (a) [0.85, 1.0, 1.2] (close-neighborhood TTA) and report which is
    # best for each regime.
    NB = (0.85, 1.00, 1.20)
    lines.append(f"\n## TTA aggregation eval (neighborhood = {NB})\n\n")
    lines.append("| regime | strategy | mean prob_fake | frac_FAKE@0.5 | n |\n")
    lines.append("|---|---|---|---|---|\n")
    for regime in ("OK_real", "FAIL_real", "FAKE"):
        for strat_name, strat in (("native_only", lambda vs: vs[1.0] if 1.0 in vs else None),
                                  ("mean", lambda vs: mean([vs[t] for t in NB if t in vs]) if all(t in vs for t in NB) else None),
                                  ("median", lambda vs: median([vs[t] for t in NB if t in vs]) if all(t in vs for t in NB) else None),
                                  ("min", lambda vs: min(vs[t] for t in NB if t in vs) if all(t in vs for t in NB) else None),
                                  ("max", lambda vs: max(vs[t] for t in NB if t in vs) if all(t in vs for t in NB) else None)):
            scores = []
            for (tag, frame), tps in by_frame.items():
                if regime_for_tag(tag) != regime:
                    continue
                v = strat(tps)
                if v is not None:
                    scores.append(v)
            if not scores:
                continue
            m = mean(scores)
            frac = sum(1 for s in scores if s >= 0.5) / len(scores)
            lines.append(f"| {regime} | {strat_name} | {m:.3f} | {frac:.0%} | {len(scores)} |\n")

    # Per-tag native FPR (informational for FAIL diagnosis)
    lines.append("\n## Per-tag native FPR/recall and best-alternative crop\n\n")
    lines.append("| regime | tag | n | native_mean | native_FPR_or_recall | best_t_mean | improvement |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    by_tag: dict[str, list[dict[float, float]]] = defaultdict(list)
    for (tag, frame), tps in by_frame.items():
        by_tag[tag].append(tps)
    for tag, lst in sorted(by_tag.items()):
        regime = regime_for_tag(tag)
        natives = [tps[1.0] for tps in lst if 1.0 in tps]
        if not natives:
            continue
        nm = mean(natives)
        if regime in ("OK_real", "FAIL_real"):
            n_metric_label = "FPR"
            n_metric = sum(1 for v in natives if v >= 0.5) / len(natives)
        else:
            n_metric_label = "recall"
            n_metric = sum(1 for v in natives if v >= 0.5) / len(natives)
        # best per-frame alternative
        if regime in ("OK_real", "FAIL_real"):
            best_alts = [min(tps.values()) for tps in lst]
        else:
            best_alts = [max(tps.values()) for tps in lst]
        bm = mean(best_alts)
        if regime in ("OK_real", "FAIL_real"):
            improvement = nm - bm  # positive means perturb-helped
        else:
            improvement = bm - nm
        lines.append(
            f"| {regime} | {tag[:42]} | {len(lst)} | {nm:.3f} | {n_metric:.0%} ({n_metric_label}) "
            f"| {bm:.3f} | {improvement:+.3f} |\n"
        )

    OUT.write_text("".join(lines))
    print(f"[synthesize] wrote {OUT}")
    print("".join(lines))


if __name__ == "__main__":
    main()
