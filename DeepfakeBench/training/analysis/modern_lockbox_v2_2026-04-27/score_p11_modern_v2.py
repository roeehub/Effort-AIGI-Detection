"""Post-hoc modern_lockbox_v2 FPR + recall computation for P11 overnight slate.

Inputs:
- frame-level reports under gs://.../p11-overnight-2026-04-28/reports/
- modern_lockbox_real_v2_frames.yaml + modern_lockbox_fake_v2_frames.yaml (281 + 367)

Outputs:
- per-candidate FPR @ τ=0.5 and τ=0.9741 on:
  (a) full lockbox real (414 frames), (b) modern_v2 real subset (281 frames)
- per-candidate recall @ both τ on:
  (a) full lockbox fake (425 frames), (b) modern_v2 fake subset (367 frames)
- decision-grade comparison vs P8A reference

Usage:
  python -m analysis.modern_lockbox_v2_2026-04-27.score_p11_modern_v2 \\
    [--reports-prefix gs://.../p11-overnight-2026-04-28/reports/]
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = REPO / "analysis/modern_lockbox_v2_2026-04-27"

DEFAULT_REPORTS_PREFIX = "gs://training-job-outputs/test_results/teams_promotion_contract/p11-overnight-2026-04-28/reports"

CANDIDATES = [
    "p11_mild_step1000",
    "p11_heavy_step1000",
    "p11_heavy_deeplive_step1000",
    "p11_webcam_harden_step1000",
    "p8a_reference_step5000",
]

TAU_DEFAULT = 0.5
TAU_PROD_5PCT = 0.9741  # P8A's calibrated 5%-prod τ on full lockbox; we re-fit per candidate too


def gsutil_cat(uri: str) -> bytes:
    res = subprocess.run(["gsutil", "cat", uri], capture_output=True, check=False)
    if res.returncode != 0:
        return b""
    return res.stdout


def load_frame_uri_set(yaml_path: Path) -> set[str]:
    doc = yaml.safe_load(yaml_path.read_text())
    return set(doc.get("frames", []))


def load_report_csv(reports_prefix: str, suite: str, candidate: str) -> pd.DataFrame | None:
    uri = f"{reports_prefix.rstrip('/')}/{suite}_{candidate}_frames_report.csv"
    raw = gsutil_cat(uri)
    if not raw:
        print(f"  WARN: missing {uri}")
        return None
    return pd.read_csv(io.BytesIO(raw))


def fpr_at(df_real: pd.DataFrame, tau: float) -> float:
    """Real frames: label==0; FPR = fraction with score >= tau."""
    if df_real.empty:
        return float("nan")
    return float((df_real["frame_prob"] >= tau).mean())


def recall_at(df_fake: pd.DataFrame, tau: float) -> float:
    """Fake frames: label==1; recall = fraction with score >= tau."""
    if df_fake.empty:
        return float("nan")
    return float((df_fake["frame_prob"] >= tau).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-prefix", default=DEFAULT_REPORTS_PREFIX)
    args = parser.parse_args()

    real_v2_uris = load_frame_uri_set(OUT_DIR / "modern_lockbox_real_v2_frames.yaml")
    fake_v2_uris = load_frame_uri_set(OUT_DIR / "modern_lockbox_fake_v2_frames.yaml")
    print(f"modern_v2: {len(real_v2_uris)} real / {len(fake_v2_uris)} fake frames")

    rows = []
    for cand in CANDIDATES:
        # Lockbox real
        real_df = load_report_csv(args.reports_prefix, "teams_real_all_lockbox", cand)
        # Lockbox fake
        fake_df = load_report_csv(args.reports_prefix, "teams_fake_all_lockbox", cand)
        # Dev fake suites for recall
        viso_df = load_report_csv(args.reports_prefix, "visomaster_enhanced_macro_dev", cand)
        deeplive_df = load_report_csv(args.reports_prefix, "deeplive_enhanced_dev", cand)
        teams_fake_dev_df = load_report_csv(args.reports_prefix, "teams_fake_all_dev", cand)
        # Dev real for FPR cap
        teams_real_dev_df = load_report_csv(args.reports_prefix, "teams_real_all_dev", cand)

        if real_df is None or fake_df is None:
            rows.append({"candidate": cand, "status": "missing_reports"})
            continue

        # Subset reals/fakes by modern_v2 frame URIs
        real_v2 = real_df[real_df["frame_path"].isin(real_v2_uris)]
        fake_v2 = fake_df[fake_df["frame_path"].isin(fake_v2_uris)]

        row = {
            "candidate": cand,
            "n_real_lockbox": len(real_df),
            "n_real_v2": len(real_v2),
            "n_fake_lockbox": len(fake_df),
            "n_fake_v2": len(fake_v2),
            # FPR — full lockbox
            "fpr_lockbox_tau0.5": fpr_at(real_df, TAU_DEFAULT),
            "fpr_lockbox_tau0.9741": fpr_at(real_df, TAU_PROD_5PCT),
            # FPR — modern_v2
            "fpr_v2_tau0.5": fpr_at(real_v2, TAU_DEFAULT),
            "fpr_v2_tau0.9741": fpr_at(real_v2, TAU_PROD_5PCT),
            # recall — full lockbox fake
            "recall_lockbox_tau0.5": recall_at(fake_df, TAU_DEFAULT),
            "recall_lockbox_tau0.9741": recall_at(fake_df, TAU_PROD_5PCT),
            # recall — modern_v2 fake
            "recall_v2_tau0.5": recall_at(fake_v2, TAU_DEFAULT),
            "recall_v2_tau0.9741": recall_at(fake_v2, TAU_PROD_5PCT),
        }

        # Dev fake recalls (Plan v3 §6 gates)
        if viso_df is not None:
            row["recall_viso_dev_tau0.5"] = recall_at(viso_df, TAU_DEFAULT)
            row["recall_viso_dev_tau0.9741"] = recall_at(viso_df, TAU_PROD_5PCT)
        if deeplive_df is not None:
            row["recall_deeplive_dev_tau0.5"] = recall_at(deeplive_df, TAU_DEFAULT)
            row["recall_deeplive_dev_tau0.9741"] = recall_at(deeplive_df, TAU_PROD_5PCT)
        if teams_fake_dev_df is not None:
            row["recall_teams_fake_dev_tau0.5"] = recall_at(teams_fake_dev_df, TAU_DEFAULT)
            row["recall_teams_fake_dev_tau0.9741"] = recall_at(teams_fake_dev_df, TAU_PROD_5PCT)
        if teams_real_dev_df is not None:
            row["fpr_teams_real_dev_tau0.5"] = fpr_at(teams_real_dev_df, TAU_DEFAULT)
            row["fpr_teams_real_dev_tau0.9741"] = fpr_at(teams_real_dev_df, TAU_PROD_5PCT)

        rows.append(row)
        print(f"  {cand}: lockbox FPR @ τ=0.5={row['fpr_lockbox_tau0.5']:.4f}, "
              f"v2 FPR @ τ=0.5={row['fpr_v2_tau0.5']:.4f}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "p11_overnight_modern_v2_scorecard.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    # Decision-grade verdict per Plan v3 §6
    print("\n=== Plan v3 §6 Day-2 Gate ===")
    for _, r in df.iterrows():
        if r.get("status") == "missing_reports":
            print(f"  {r['candidate']}: MISSING REPORTS")
            continue
        viso = r.get("recall_viso_dev_tau0.5", float("nan"))
        deeplive = r.get("recall_deeplive_dev_tau0.5", float("nan"))
        teams = r.get("recall_teams_fake_dev_tau0.5", float("nan"))
        v2_fpr = r.get("fpr_v2_tau0.9741", float("nan"))
        teams_real = r.get("fpr_teams_real_dev_tau0.5", float("nan"))

        gate_pass = (
            viso >= 0.90 and deeplive >= 0.90 and teams >= 0.90
            and v2_fpr <= 0.05 and teams_real <= 0.07
        )
        verdict = "α SHIP" if gate_pass else "β/γ ASSESS"
        print(f"  {r['candidate']}: {verdict}")
        print(f"    viso_dev={viso:.3f} | deeplive_dev={deeplive:.3f} | teams_fake_dev={teams:.3f}")
        print(f"    modern_v2_fpr@τ=0.9741={v2_fpr:.4f} | teams_real_dev@τ=0.5={teams_real:.4f}")

    out_json = OUT_DIR / "p11_overnight_modern_v2_summary.json"
    out_json.write_text(json.dumps(rows, indent=2, default=str))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
