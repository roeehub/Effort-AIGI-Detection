"""F4-style substrate filtering on cached Stage 2 scores (Dor cohort + Roy_D).

Tests whether the Stage 2 step4500 regression numbers (15-28× P8A's dor
real-FPR baseline; 96-100% Roy_D real-FPR @ τ=0.5) soften when known
substrate-pollution categories are dropped.

F4 filter operations (per memory `project_job14_substrate_clean_2026-05-04`):
  1. Drop chronic-6 identities (Roy_D, PC_Generator, bla_bla_chow,
     dor_shkedi-as-real, healthy_dor_shkedi-as-real, Md_noyn_Sharker)
  2. Drop is_no_face frames
  3. Drop frames with min(W,H) < 200

Sources:
  - analysis/stage2_cpu_2026-05-09/outputs/stage2_dor_cohort_scores.csv
    (scores for 388-frame Dor cohort, 12 ckpts)
  - analysis/stage2_cpu_2026-05-09/outputs/roy_d_stage2_per_frame.csv
    (130-frame Roy_D real set, 9 Stage 2 ckpts + reference)
  - analysis/dor_encoder_axis_2026-05-08/_cache/cohort_manifest.csv
    (frame-level metadata: width, height, identity, is_no_face)

Outputs:
  outputs/f4_filter_dor_cohort.csv
  outputs/f4_filter_roy_d.csv
  outputs/f4_filter_summary.csv
  F4_FILTER_FACTS_2026-05-09.md (factual-only)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
OUTPUTS = THIS_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

DOR_SCORES_CSV = (
    REPO_ROOT / "analysis" / "stage2_cpu_2026-05-09" / "outputs"
    / "stage2_dor_cohort_scores.csv"
)
ROY_D_SCORES_CSV = (
    REPO_ROOT / "analysis" / "stage2_cpu_2026-05-09" / "outputs"
    / "roy_d_stage2_per_frame.csv"
)
DOR_MANIFEST = (
    REPO_ROOT / "analysis" / "dor_encoder_axis_2026-05-08" / "_cache"
    / "cohort_manifest.csv"
)

# Per memory project_chronic_offenders_partition_per_ckpt_2026-05-04 +
# project_eval_substrate_reframe_2026-05-04. The set may need refinement;
# substring-matched against identity_key.
CHRONIC_6_PATTERNS = [
    "Roy_D",
    "PC_Generator",
    "bla_bla_chow",
    "Md_noyn_Sharker",
    "dor_shkedi",
    "healthy_dor",  # `healthy_dor_shkedi`
]

REF_CKPTS = ["P8A", "E2B", "P2D"]
STAGE2_CKPTS = [
    "S1_step500", "S1_step2500", "S1_step4500",
    "S2_step500", "S2_step2500", "S2_step4500",
    "S3_step500", "S3_step2500", "S3_step4500",
]
ALL_CKPTS = REF_CKPTS + STAGE2_CKPTS

logger = logging.getLogger("f4-filter")


def chronic_match(identity_str: str) -> bool:
    """Return True if identity matches any chronic-6 pattern."""
    if not isinstance(identity_str, str):
        return False
    s = identity_str.lower()
    return any(p.lower() in s for p in CHRONIC_6_PATTERNS)


def load_dor_with_metadata() -> pd.DataFrame:
    """Join the 388-frame Dor scores with manifest metadata for filtering."""
    if not DOR_SCORES_CSV.exists():
        logger.error("missing %s", DOR_SCORES_CSV)
        return pd.DataFrame()
    if not DOR_MANIFEST.exists():
        logger.error("missing %s", DOR_MANIFEST)
        return pd.DataFrame()
    scores = pd.read_csv(DOR_SCORES_CSV)
    manifest = pd.read_csv(DOR_MANIFEST)
    # Manifest columns include frame_path; join on that.
    if "frame_path" not in scores.columns:
        logger.error("scores file missing frame_path")
        return pd.DataFrame()
    keep_meta_cols = ["frame_path"]
    for c in ["identity_key", "session_id", "video_id", "method",
              "width", "height", "is_no_face"]:
        if c in manifest.columns:
            keep_meta_cols.append(c)
    merged = scores.merge(manifest[keep_meta_cols], on="frame_path", how="left")
    return merged


def fpr_recall_summary(df: pd.DataFrame, ckpt: str, tau: float = 0.5) -> dict:
    """Compute real-FPR (label==0) and fake-recall (label==1) at τ."""
    if ckpt not in df.columns:
        return {}
    n_total = len(df)
    if n_total == 0:
        return {}
    above = (df[ckpt] >= tau)
    real_n = int((df["label"] == 0).sum())
    fake_n = int((df["label"] == 1).sum())
    real_fp = int(((df["label"] == 0) & above).sum())
    fake_tp = int(((df["label"] == 1) & above).sum())
    return {
        "ckpt": ckpt,
        "n_total": n_total,
        "n_real": real_n,
        "n_fake": fake_n,
        "real_fpr_at_0p5": real_fp / max(real_n, 1),
        "fake_recall_at_0p5": fake_tp / max(fake_n, 1),
        "real_p50_score": float(df[df["label"] == 0][ckpt].median()) if real_n > 0 else float("nan"),
        "fake_p50_score": float(df[df["label"] == 1][ckpt].median()) if fake_n > 0 else float("nan"),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    rows = []

    # 1) Dor cohort recompute (388 frames, 5 sub-cohorts, 12 ckpts).
    df_dor = load_dor_with_metadata()
    if len(df_dor) == 0:
        logger.warning("no Dor cohort data; skipping")
    else:
        # Filter version 1: drop chronic-6 identities (substring match).
        dor_no_chronic = df_dor[~df_dor["identity_key"].apply(chronic_match)].reset_index(drop=True)
        # Filter version 2: drop chronic + is_no_face.
        if "is_no_face" in df_dor.columns:
            dor_no_chronic_no_noface = dor_no_chronic[~dor_no_chronic["is_no_face"].astype(bool, errors="ignore").fillna(False)].reset_index(drop=True)
        else:
            dor_no_chronic_no_noface = dor_no_chronic
        # Filter version 3: + min(W,H) >= 200.
        if "width" in dor_no_chronic_no_noface.columns and "height" in dor_no_chronic_no_noface.columns:
            min_dim = dor_no_chronic_no_noface[["width", "height"]].min(axis=1)
            dor_full_f4 = dor_no_chronic_no_noface[min_dim >= 200].reset_index(drop=True)
        else:
            dor_full_f4 = dor_no_chronic_no_noface

        for tag, sub in [("F0_full", df_dor),
                         ("F1_no_chronic", dor_no_chronic),
                         ("F2_no_chronic_no_noface", dor_no_chronic_no_noface),
                         ("F4_no_chronic_no_noface_min_dim_ge200", dor_full_f4)]:
            for ckpt in ALL_CKPTS:
                if ckpt not in sub.columns:
                    continue
                summary = fpr_recall_summary(sub, ckpt)
                if summary:
                    summary["filter"] = tag
                    summary["scope"] = "DOR_COHORT_388"
                    rows.append(summary)
            logger.info("[%s] DOR_COHORT n=%d after filter", tag, len(sub))

        # Per-cohort breakdown (DOR_REAL_LOCKBOX is the load-bearing one).
        for cohort_name in df_dor["cohort"].unique():
            sub_full = df_dor[df_dor["cohort"] == cohort_name]
            sub_no_chronic = sub_full[~sub_full["identity_key"].apply(chronic_match)].reset_index(drop=True)
            for tag, sub in [(f"F0_{cohort_name}", sub_full),
                             (f"F1_{cohort_name}_no_chronic", sub_no_chronic)]:
                for ckpt in ALL_CKPTS:
                    if ckpt not in sub.columns:
                        continue
                    summary = fpr_recall_summary(sub, ckpt)
                    if summary:
                        summary["filter"] = tag
                        summary["scope"] = cohort_name
                        rows.append(summary)

    # 2) Roy_D: 130 real frames, all from Roy_D identity → "drop chronic"
    # would empty the set. Instead, look at min_dim filter only.
    if ROY_D_SCORES_CSV.exists():
        df_roy = pd.read_csv(ROY_D_SCORES_CSV)
        # All frames are label=0 (real); add label column.
        df_roy["label"] = 0
        # Roy_D doesn't have width/height in this CSV; will skip min_dim filter.
        # Just record F0.
        # Identify the available ckpt columns dynamically.
        roy_ckpt_cols = [c for c in df_roy.columns if c in ALL_CKPTS or
                         c.startswith("P1_BUNDLE") or c.startswith("P1_PAIRRANK")]
        for ckpt in roy_ckpt_cols:
            summary = fpr_recall_summary(df_roy, ckpt)
            if summary:
                summary["filter"] = "F0_full"
                summary["scope"] = "ROY_D_130"
                rows.append(summary)
        logger.info("Roy_D: %d frames, %d ckpts", len(df_roy), len(roy_ckpt_cols))
    else:
        logger.warning("Roy_D scores file missing: %s", ROY_D_SCORES_CSV)

    # Save.
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUTS / "f4_filter_summary.csv", index=False)
    logger.info("wrote %s (n=%d rows)", OUTPUTS / "f4_filter_summary.csv", len(summary_df))

    # Console headlines.
    print("\n" + "=" * 80)
    print("DOR_REAL_LOCKBOX real-FPR @ τ=0.5 — F0 (full) vs F1 (drop chronic-6)")
    print("=" * 80)
    pivot = summary_df[summary_df["scope"].isin(["DOR_REAL_LOCKBOX",
                                                  "DOR_COHORT_388"])].copy()
    pivot = pivot[pivot["filter"].str.startswith(("F0", "F1"))]
    p = pivot.pivot_table(index="ckpt", columns="filter",
                          values="real_fpr_at_0p5")
    pd.options.display.float_format = "{:.4f}".format
    if len(p) > 0:
        print(p.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
