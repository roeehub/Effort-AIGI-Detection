"""Day-4 triple-axis verdict for R13_P13_FROM_SCRATCH (Plan v6 §4.5).

Inputs:
- promotion_contract/checkpoint_summary.csv (selected_threshold + per-suite recalls)
- promotion_contract/selected_threshold_scorecard.csv (per-suite at selected τ)
- reports/<suite>_<candidate>_frames_report.csv (per-frame frame_prob)
- analysis/clean_eval_2026-04-29/clean_eval_v1_fake.yaml (26 frames, Axis 2)
- analysis/clean_eval_2026-04-29/shortcut_probe_v1_pairs.yaml (11 pairs, Axis 3)
- analysis/modern_lockbox_v2_2026-04-27/modern_lockbox_real_v2_frames.yaml (Axis 1 modern_v2)

Outputs:
- p13_day4_verdict.csv  (per-candidate triple-axis numbers)
- p13_day4_verdict.json (verdict + winner candidate)
- p13_day4_verdict_REPORT.md (human-readable for user wake-up)

Usage:
  python -m analysis.p13_day4_verdict_2026-04-29.compute_verdict \\
    [--reports-prefix gs://.../p13-from-scratch-scorecard-20260428-uswest4] \\
    [--fallback-tau none|fixed5pct]
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = REPO / "analysis/p13_day4_verdict_2026-04-29"
CLEAN_EVAL_DIR = REPO / "analysis/clean_eval_2026-04-29"
MODERN_V2_DIR = REPO / "analysis/modern_lockbox_v2_2026-04-27"

DEFAULT_PREFIX = (
    "gs://training-job-outputs/test_results/teams_promotion_contract/"
    "p13-from-scratch-scorecard-20260428-uswest4"
)

# 8 suites in the contract manifest. We need ALL frame reports for Axis 2/3
# joins — clean_eval frames live in real or fake suites; shortcut probe frames
# are reals across capture modes.
ALL_SUITES = [
    "teams_real_all_dev",
    "teams_real_poor_quality_dev",
    "teams_real_lighting_extreme_dev",
    "teams_fake_all_dev",
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
    "teams_real_all_lockbox",
    "teams_fake_all_lockbox",
]

# Axis 1 thresholds (Plan v6 §4.5)
AXIS1_VISO_MIN = 0.90
AXIS1_DEEPLIVE_MIN = 0.90
AXIS1_TEAMS_FAKE_MIN = 0.90
AXIS1_MODERN_V2_FPR_MAX = 0.05
AXIS1_TEAMS_REAL_DEV_FPR_MAX = 0.07

# Axis 2/3 thresholds
AXIS2_RECALL_MIN = 0.80
AXIS3_DELTA_MAX = 0.15

# β/γ widths
BETA_PP_WINDOW = 0.15  # 5-15pp short = β
GAMMA_PP_WINDOW = 0.15  # ≥15pp short = γ
SHORTCUT_GAMMA_THRESHOLD = 0.30  # shortcut Δ > 0.30 → γ regardless


def gsutil_cat(uri: str) -> bytes:
    res = subprocess.run(["gsutil", "cat", uri], capture_output=True, check=False)
    if res.returncode != 0:
        return b""
    return res.stdout


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def load_clean_eval_fake_uris() -> set[str]:
    doc = load_yaml(CLEAN_EVAL_DIR / "clean_eval_v1_fake.yaml")
    return {f["gcs_uri"] for f in doc["frames"]}


def load_shortcut_pairs() -> list[tuple[str, str, str]]:
    doc = load_yaml(CLEAN_EVAL_DIR / "shortcut_probe_v1_pairs.yaml")
    return [(p["pair_id"], p["a"]["gcs_uri"], p["b"]["gcs_uri"]) for p in doc["pairs"]]


def load_modern_v2_real_uris() -> set[str]:
    doc = load_yaml(MODERN_V2_DIR / "modern_lockbox_real_v2_frames.yaml")
    # The yaml is a flat list under "frames"
    if isinstance(doc, dict) and "frames" in doc:
        return set(doc["frames"])
    if isinstance(doc, list):
        return set(doc)
    raise ValueError(f"Unexpected modern_v2 yaml shape: {type(doc)}")


def list_report_keys(reports_prefix: str) -> list[tuple[str, str]]:
    """Return (suite, candidate_suffix) tuples found under reports_prefix.
    A frame report file is named '<suite>_<candidate>_frames_report.csv'.
    Candidate suffix is what comes between '<suite>_' and '_frames_report.csv'.
    """
    res = subprocess.run(
        ["gsutil", "ls", f"{reports_prefix.rstrip('/')}/reports/"],
        capture_output=True, check=False, text=True
    )
    if res.returncode != 0:
        return []
    items = []
    for line in res.stdout.splitlines():
        if not line.endswith("_frames_report.csv"):
            continue
        name = line.rsplit("/", 1)[-1].removesuffix("_frames_report.csv")
        # Match longest suite prefix
        for suite in sorted(ALL_SUITES, key=len, reverse=True):
            if name.startswith(suite + "_"):
                items.append((suite, name[len(suite) + 1:]))
                break
    return items


def load_report(reports_prefix: str, suite: str, candidate: str) -> pd.DataFrame | None:
    uri = f"{reports_prefix.rstrip('/')}/reports/{suite}_{candidate}_frames_report.csv"
    raw = gsutil_cat(uri)
    if not raw:
        return None
    return pd.read_csv(io.BytesIO(raw))


def load_concat_all_reports(reports_prefix: str, candidate: str) -> pd.DataFrame:
    """Concat per-frame reports across ALL 8 suites for one candidate."""
    frames = []
    for suite in ALL_SUITES:
        df = load_report(reports_prefix, suite, candidate)
        if df is not None:
            df = df.copy()
            df["__suite"] = suite
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Dedupe by frame_path: prefer fake suites over real for fake-labelled frames
    # (a frame might appear in both teams_fake_all_dev and lockbox if there's overlap).
    out = out.drop_duplicates(subset=["frame_path"], keep="first")
    return out


def axis2_clean_eval_recall(all_frames: pd.DataFrame, fake_uris: set[str], tau: float) -> dict:
    """Axis 2: fake recall on clean_eval_v1 fake frames at τ."""
    matched = all_frames[all_frames["frame_path"].isin(fake_uris)]
    n_matched = len(matched)
    n_total = len(fake_uris)
    if n_matched == 0:
        return {"recall": float("nan"), "n_matched": 0, "n_total": n_total, "n_hits": 0}
    n_hits = int((matched["frame_prob"] >= tau).sum())
    return {
        "recall": n_hits / n_matched,
        "n_matched": n_matched,
        "n_total": n_total,
        "n_hits": n_hits,
    }


def axis3_shortcut_delta(all_frames: pd.DataFrame, pairs: list[tuple[str, str, str]]) -> dict:
    """Axis 3: max |p_fake_a - p_fake_b| across shortcut probe pairs."""
    by_uri = dict(zip(all_frames["frame_path"], all_frames["frame_prob"]))
    deltas = []
    missing = []
    for pid, a_uri, b_uri in pairs:
        if a_uri not in by_uri or b_uri not in by_uri:
            missing.append(pid)
            continue
        deltas.append((pid, abs(float(by_uri[a_uri]) - float(by_uri[b_uri]))))
    if not deltas:
        return {
            "max_delta": float("nan"),
            "max_pair": None,
            "n_resolved": 0,
            "n_total": len(pairs),
            "missing": missing,
        }
    deltas.sort(key=lambda x: -x[1])
    return {
        "max_delta": deltas[0][1],
        "max_pair": deltas[0][0],
        "n_resolved": len(deltas),
        "n_total": len(pairs),
        "missing": missing,
        "all_deltas": [(p, round(d, 4)) for p, d in deltas],
    }


def axis1_modern_v2_fpr(reports_prefix: str, candidate: str, modern_v2_real_uris: set[str], tau: float) -> dict:
    """Axis 1: modern_v2 FPR at τ from teams_real_all_lockbox filtered by modern_v2."""
    df = load_report(reports_prefix, "teams_real_all_lockbox", candidate)
    if df is None:
        return {"fpr": float("nan"), "n_matched": 0}
    matched = df[df["frame_path"].isin(modern_v2_real_uris)]
    if matched.empty:
        return {"fpr": float("nan"), "n_matched": 0}
    n_fp = int((matched["frame_prob"] >= tau).sum())
    return {"fpr": n_fp / len(matched), "n_matched": len(matched), "n_fp": n_fp}


def classify_axis(value: float, threshold: float, direction: str) -> tuple[str, float]:
    """Returns (status, gap_pp) where status ∈ {pass, beta, gamma}."""
    if value != value:  # NaN
        return ("nan", float("nan"))
    if direction == "ge":
        gap = threshold - value
    else:  # le
        gap = value - threshold
    if gap <= 0:
        return ("pass", 0.0)
    if gap < BETA_PP_WINDOW:
        return ("beta", gap)
    return ("gamma", gap)


def overall_verdict(axis1_status: list[tuple[str, float]], axis2_status: tuple[str, float], axis3_status: tuple[str, float], axis3_delta: float) -> str:
    """Apply Plan v6 α/β/γ gate."""
    statuses = [s for s, _ in axis1_status] + [axis2_status[0], axis3_status[0]]
    if all(s == "pass" for s in statuses):
        return "α"
    # γ trigger: all axes ≥ 15pp short OR shortcut Δ > 0.30
    if axis3_delta > SHORTCUT_GAMMA_THRESHOLD:
        return "γ"
    if all(s == "gamma" for s in statuses if s != "pass"):
        return "γ"
    return "β"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--fallback-tau",
        choices=["none", "fixed5pct"],
        default="none",
        help="If contract τ looks degenerate (>0.99 + near-zero recall), fall back.",
    )
    parser.add_argument(
        "--fixed-tau",
        type=float,
        default=None,
        help="Override: use this τ for all candidates (e.g. 0.5 or a per-pool 5pct number)",
    )
    parser.add_argument(
        "--candidates-only",
        nargs="*",
        default=None,
        help="Restrict to specific candidate suffixes (default: all in checkpoint_summary)",
    )
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: load contract artifacts
    cs_uri = f"{args.reports_prefix.rstrip('/')}/promotion_contract/checkpoint_summary.csv"
    cs_raw = gsutil_cat(cs_uri)
    if not cs_raw:
        print(f"ERROR: checkpoint_summary.csv not found at {cs_uri}", file=sys.stderr)
        return 2
    summary = pd.read_csv(io.BytesIO(cs_raw))
    print(f"loaded {len(summary)} candidates from checkpoint_summary.csv")

    sel_uri = f"{args.reports_prefix.rstrip('/')}/promotion_contract/selected_threshold_scorecard.csv"
    sel_raw = gsutil_cat(sel_uri)
    sel_df = pd.read_csv(io.BytesIO(sel_raw)) if sel_raw else pd.DataFrame()

    # Step 2: load Axis 2/3 inputs
    fake_uris = load_clean_eval_fake_uris()
    pairs = load_shortcut_pairs()
    modern_v2_real_uris = load_modern_v2_real_uris()
    print(f"clean_eval fake: {len(fake_uris)} URIs | "
          f"shortcut pairs: {len(pairs)} | "
          f"modern_v2 real: {len(modern_v2_real_uris)} URIs")

    # Step 3: derive candidate suffix per row.
    # The selected_threshold_scorecard 'report_path' column has the
    # exact suite + suffix used in filenames; parse from there.
    suffix_by_key = {}
    if not sel_df.empty:
        for _, r in sel_df.iterrows():
            key = r["checkpoint_key"]
            if key in suffix_by_key:
                continue
            rp = str(r.get("report_path", ""))
            if "_videos_report.csv" not in rp:
                continue
            fname = rp.rsplit("/", 1)[-1].removesuffix("_videos_report.csv")
            for suite in sorted(ALL_SUITES, key=len, reverse=True):
                if fname.startswith(suite + "_"):
                    suffix_by_key[key] = fname[len(suite) + 1:]
                    break

    if not suffix_by_key:
        # Fallback: derive from checkpoint_key by lowercasing
        for _, r in summary.iterrows():
            key = r["checkpoint_key"]
            suffix_by_key[key] = key.lower()
        print("WARN: derived suffixes by lowercasing checkpoint_key (verify against actual reports)")

    # Step 4: per-candidate verdict
    verdict_rows = []
    print()
    for _, srow in summary.iterrows():
        key = srow["checkpoint_key"]
        suffix = suffix_by_key.get(key)
        if suffix is None:
            print(f"  {key}: cannot resolve filename suffix; skipping")
            continue
        if args.candidates_only and suffix not in args.candidates_only:
            continue

        tau = args.fixed_tau if args.fixed_tau is not None else float(srow["selected_threshold"])
        is_degenerate = (tau >= 0.99 and float(srow.get("teams_fake_all_dev__fake_recall", 0)) < 0.05)

        # Concat all reports for this candidate
        all_frames = load_concat_all_reports(args.reports_prefix, suffix)
        if all_frames.empty:
            print(f"  {key} ({suffix}): no reports loaded; skipping")
            continue

        # Axis 1 components from summary
        viso = float(srow["visomaster_enhanced_macro_dev__fake_recall"])
        deeplive = float(srow["deeplive_enhanced_dev__fake_recall"])
        teams_fake = float(srow["teams_fake_all_dev__fake_recall"])
        teams_real_dev_fpr = float(srow["dev_primary_real_fpr"])

        # Axis 1 modern_v2 FPR
        mv2 = axis1_modern_v2_fpr(args.reports_prefix, suffix, modern_v2_real_uris, tau)

        # Axis 2
        a2 = axis2_clean_eval_recall(all_frames, fake_uris, tau)

        # Axis 3
        a3 = axis3_shortcut_delta(all_frames, pairs)

        # Classify each axis
        axis1_components = [
            ("viso_dev", viso, AXIS1_VISO_MIN, "ge"),
            ("deeplive_dev", deeplive, AXIS1_DEEPLIVE_MIN, "ge"),
            ("teams_fake_dev", teams_fake, AXIS1_TEAMS_FAKE_MIN, "ge"),
            ("modern_v2_fpr", mv2["fpr"], AXIS1_MODERN_V2_FPR_MAX, "le"),
            ("teams_real_dev_fpr", teams_real_dev_fpr, AXIS1_TEAMS_REAL_DEV_FPR_MAX, "le"),
        ]
        axis1_status = [classify_axis(v, t, d) for _, v, t, d in axis1_components]
        axis2_st = classify_axis(a2["recall"], AXIS2_RECALL_MIN, "ge")
        axis3_st = classify_axis(a3["max_delta"], AXIS3_DELTA_MAX, "le")

        verdict = overall_verdict(axis1_status, axis2_st, axis3_st, a3["max_delta"] if a3["max_delta"] == a3["max_delta"] else 0.0)

        row = {
            "checkpoint_key": key,
            "checkpoint_path": srow["checkpoint_path"],
            "suffix": suffix,
            "selected_threshold": tau,
            "tau_degenerate": is_degenerate,
            # Axis 1
            "viso_dev_recall": viso,
            "deeplive_dev_recall": deeplive,
            "teams_fake_dev_recall": teams_fake,
            "modern_v2_fpr": mv2["fpr"],
            "modern_v2_n_real": mv2.get("n_matched", 0),
            "teams_real_dev_fpr": teams_real_dev_fpr,
            # Axis 2
            "axis2_recall": a2["recall"],
            "axis2_n_matched": a2["n_matched"],
            "axis2_n_total": a2["n_total"],
            # Axis 3
            "axis3_max_delta": a3["max_delta"],
            "axis3_max_pair": a3["max_pair"],
            "axis3_n_resolved": a3["n_resolved"],
            "axis3_n_total": a3["n_total"],
            "axis3_missing_pairs": a3.get("missing", []),
            # Status
            "axis1_status": [s for s, _ in axis1_status],
            "axis2_status": axis2_st[0],
            "axis3_status": axis3_st[0],
            "verdict": verdict,
        }
        verdict_rows.append(row)
        print(f"  {key} (τ={tau:.4f}{' DEGEN' if is_degenerate else ''}): "
              f"viso={viso:.3f} deeplive={deeplive:.3f} teams_fake={teams_fake:.3f} "
              f"mv2_fpr={mv2['fpr']:.4f} treal_fpr={teams_real_dev_fpr:.4f} | "
              f"a2={a2['recall']:.3f} ({a2['n_matched']}/{a2['n_total']}) | "
              f"a3={a3['max_delta']:.4f} ({a3['n_resolved']}/{a3['n_total']}) | "
              f"verdict={verdict}")

    # Step 5: emit outputs
    if not verdict_rows:
        print("ERROR: no candidates produced rows", file=sys.stderr)
        return 3

    df = pd.DataFrame(verdict_rows)
    df_csv = df.copy()
    for col in ("axis1_status", "axis3_missing_pairs"):
        df_csv[col] = df_csv[col].apply(json.dumps)
    df_csv.to_csv(out_dir / "p13_day4_verdict.csv", index=False)
    print(f"\nwrote {out_dir / 'p13_day4_verdict.csv'}")

    # Pick winner = first row with verdict α; else first β with smallest max gap; else first γ
    p13_rows = [r for r in verdict_rows if r["checkpoint_key"].startswith("P13_FROM_SCRATCH")]
    pool = p13_rows if p13_rows else verdict_rows
    alpha = [r for r in pool if r["verdict"] == "α"]
    beta = [r for r in pool if r["verdict"] == "β"]
    if alpha:
        winner = alpha[0]
        verdict = "α"
    elif beta:
        # Pick smallest collective gap
        def gap_score(r):
            comps = []
            for v, t in [
                (r["viso_dev_recall"], AXIS1_VISO_MIN),
                (r["deeplive_dev_recall"], AXIS1_DEEPLIVE_MIN),
                (r["teams_fake_dev_recall"], AXIS1_TEAMS_FAKE_MIN),
                (r["axis2_recall"], AXIS2_RECALL_MIN),
            ]:
                if v == v:
                    comps.append(max(0.0, t - v))
            for v, t in [
                (r["modern_v2_fpr"], AXIS1_MODERN_V2_FPR_MAX),
                (r["teams_real_dev_fpr"], AXIS1_TEAMS_REAL_DEV_FPR_MAX),
                (r["axis3_max_delta"], AXIS3_DELTA_MAX),
            ]:
                if v == v:
                    comps.append(max(0.0, v - t))
            return sum(comps)
        winner = min(beta, key=gap_score)
        verdict = "β"
    else:
        winner = pool[0]
        verdict = "γ"

    summary_out = {
        "verdict": verdict,
        "winner": {k: winner[k] for k in ("checkpoint_key", "checkpoint_path", "suffix", "selected_threshold", "verdict")},
        "winner_metrics": {
            "axis1": {
                "viso_dev_recall": winner["viso_dev_recall"],
                "deeplive_dev_recall": winner["deeplive_dev_recall"],
                "teams_fake_dev_recall": winner["teams_fake_dev_recall"],
                "modern_v2_fpr": winner["modern_v2_fpr"],
                "teams_real_dev_fpr": winner["teams_real_dev_fpr"],
            },
            "axis2": {"recall": winner["axis2_recall"], "n_matched": winner["axis2_n_matched"], "n_total": winner["axis2_n_total"]},
            "axis3": {"max_delta": winner["axis3_max_delta"], "max_pair": winner["axis3_max_pair"], "n_resolved": winner["axis3_n_resolved"]},
        },
        "candidates_evaluated": len(verdict_rows),
        "p13_candidates_evaluated": len(p13_rows),
    }
    (out_dir / "p13_day4_verdict.json").write_text(json.dumps(summary_out, indent=2, default=str))
    print(f"wrote {out_dir / 'p13_day4_verdict.json'}")
    print(f"\nVerdict: {verdict}  candidate: {winner['checkpoint_key']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
