"""
GROUP_ID_DESIGN_AUDIT_2026-05-06

CPU-only audit to choose the multi-axis GroupDRO `group_id` definition for
PE_PAIR_RANK_DRO (and PE_SBI). Asymmetric fake/real keys per advisor + Agent 3.

Inputs:
- analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv (14,626 frames)
- analysis/checkpoint_cohort_diagnosis_2026-05-06/outputs/paired_frames_with_outcomes.csv
  (10,289 paired frames; already has method_family, enhancer_family, transport,
   quality_band, person_cluster, outcome_<ckpt>_t0.5)

Outputs to analysis/group_id_design_audit_2026-05-06/outputs/:
- candidate_groups.csv         per single-side scheme: group counts, failure-rate spread
- pairwise_evaluation.csv      every (F-x, R-y) pair: joint stats, recommendation rank
- chronic_flag_definition.json explicit chronic identity list, provenance
- quality_band_thresholds.json chosen quality_band cutoffs
- recommendation.json          top-3 (F, R) pairs + rationale
- group_id_python_snippet.py   ready-to-paste construction code
- FINDINGS.md                  1-2 page synthesis (parent agent writes the .md)

Constraints: CPU only; n_jobs=1; no model loading.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np


# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
MANIFEST_PATH = ROOT / "analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv"
PAIRED_PATH = ROOT / "analysis/checkpoint_cohort_diagnosis_2026-05-06/outputs/paired_frames_with_outcomes.csv"
OUT_DIR = ROOT / "analysis/group_id_design_audit_2026-05-06/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

# Chronic-FP identities — see project_chronic_offenders_partition_per_ckpt_2026-05-04.md
# These are the 6 identities Job 11 audited at tau@FPR=10% on teams_real_all_dev.
CHRONIC_BASE_IDENTITIES = [
    "bla_bla_chow",
    "bla_bla_chow__s2",
    "PC_Generator__s22",
    "PC_Generator__s45",
    "roy_d",
    "Q__s6",
]

# Map a base_identity (or person_cluster) to a chronic flag.
# We check both raw match and case-insensitive variants.
def _is_chronic(base_identity: str | float, person_cluster: str | float = "") -> bool:
    if not isinstance(base_identity, str):
        base_identity = ""
    if not isinstance(person_cluster, str):
        person_cluster = ""
    s = base_identity
    pc = person_cluster
    # Direct match.
    if s in CHRONIC_BASE_IDENTITIES:
        return True
    # Look for chronic identity tokens inside the base_identity (handles
    # identifiers like "PC_Generator__s22" vs "pc_generator__s22").
    s_lower = s.lower()
    for cid in CHRONIC_BASE_IDENTITIES:
        if cid.lower() in s_lower:
            return True
    # Some Job 11 chronic identities are PC_Generator subjects covered by the
    # `pc_generator` person_cluster label.
    if pc.lower() == "pc_generator" and any(
        cid.lower().startswith("pc_generator") for cid in CHRONIC_BASE_IDENTITIES
    ):
        # Conservative: only flag if the specific subject ID is in CHRONIC list.
        return s in CHRONIC_BASE_IDENTITIES
    return False


# DRO design rules:
MIN_GROUP_SIZE = 50              # rule of thumb
MIN_FAILURE_SPREAD = 0.15        # 15pp between best/worst
MAX_GROUP_DOMINANCE = 0.30       # no group >30% of total batch share

CKPT_LIST = ["P8A", "E2B", "PA_3800"]


# --------------------------------------------------------------------------- #
# Manifest enrichment                                                         #
# --------------------------------------------------------------------------- #

def _suite_to_source(suite: str) -> str:
    """Map suite name to a coarse 'source' lane. Used for real-side R-A grouping."""
    if not isinstance(suite, str):
        return "unknown"
    s = suite.lower()
    if "teams_real_all_lockbox" in s:
        return "teams_real_lockbox"
    if "teams_real_all_dev" in s or s == "teams_real_dor_dev":
        return "teams_real_dev"
    if s in {"dor_evening", "dor_morning", "team_sanity_may5", "xinhe_may6_falseflag"}:
        return "live_reals_local"
    if s == "live_reals_teams_prod":
        return "live_reals_teams_prod"
    if s == "live_fakes_teams_prod":
        return "live_fakes_teams_prod"
    if s == "visomaster_v2_dor":
        return "visomaster_v2"
    if s == "extra":
        return "extra"
    if "teams_fake" in s:
        return "teams_fake"
    if s == "dor_fake_local":
        return "dor_fake_local"
    return s


def _suite_to_transport(suite: str) -> str:
    """Coarse transport when paired_frames doesn't already have it."""
    s = suite.lower() if isinstance(suite, str) else ""
    if "teams_real" in s or "teams_fake" in s or "live_fakes_teams" in s or "live_reals_teams" in s:
        return "teams_capture"
    if "live_fakes_teams_prod" in s or "live_reals_teams_prod" in s:
        return "teams_live"
    if "visomaster" in s:
        return "visomaster"
    return "raw_capture"


def _quality_band_from_quality(quality: Any) -> str:
    """Three-way binning. Already given in manifest as 'hi-q'/'lo-q'/'unknown'."""
    if not isinstance(quality, str):
        return "unknown"
    q = quality.strip().lower()
    if q in {"hi-q", "lo-q", "unknown"}:
        return q
    return "unknown"


def _quality_band_collapsed_hl(quality_band: str) -> str:
    """Collapse to {hi, lo} only — F-E variant. 'unknown' rolls to 'lo' (conservative)."""
    if quality_band == "hi-q":
        return "hi"
    return "lo"


def _ckpt_failure_at_t05(row: pd.Series, ckpt: str, label: int) -> int:
    """Return 1 if the ckpt fails at this row, else 0.
    Failure semantics:
      - real (label=0): outcome == 'real_FP'  → 1
      - fake (label=1): outcome == 'fake_missed' → 1
    """
    col = f"outcome_{ckpt}_t0.5"
    out = row.get(col, "")
    if not isinstance(out, str):
        return 0
    if label == 0:
        return 1 if "real_FP" == out else 0
    return 1 if "fake_missed" == out else 0


# --------------------------------------------------------------------------- #
# Load & build the audit substrate                                            #
# --------------------------------------------------------------------------- #

def load_audit_frame() -> pd.DataFrame:
    """Build the master frame for the audit.

    Strategy:
      Use paired_frames_with_outcomes.csv as the failure-rate substrate (10,289
      paired frames with per-ckpt outcomes already computed). Also load the full
      14,626-row manifest to compute group sizes that include unpaired frames
      (real-only) — important because real-side groups must include unpaired
      reals to reflect real training/eval batch composition.
    """
    paired = pd.read_csv(PAIRED_PATH)
    full = pd.read_csv(MANIFEST_PATH)

    # Enrich the FULL manifest with the same coarse axes we use for grouping.
    # method_family/enhancer_family aren't in the full manifest by default —
    # they were added by Agent 3's probe. For the full manifest we synthesise
    # them where possible by joining on (frame_path) for paired rows, and
    # taking 'real_or_unknown' / 'none' for the rest.
    paired_keyed = paired.set_index("frame_path", drop=False)

    def _from_paired(col: str, default: Any) -> pd.Series:
        idx = full["frame_path"]
        return idx.map(lambda fp: paired_keyed.at[fp, col] if fp in paired_keyed.index else default)

    # Add columns that exist in paired but not in full.
    for col, default in [
        ("method_family", "real_or_unknown"),
        ("enhancer_family", "none"),
        ("transport", None),         # we'll fill below from suite
        ("quality_band", None),
        ("face_size_band", None),
        ("person_cluster", None),
    ]:
        if col not in full.columns:
            full[col] = _from_paired(col, default)

    # Fill in transport / quality_band / face_size_band / person_cluster
    # from suite/quality where paired didn't cover the row.
    full["transport"] = full.apply(
        lambda r: r["transport"] if isinstance(r.get("transport"), str)
        else _suite_to_transport(r.get("suite", "")),
        axis=1,
    )
    full["quality_band"] = full.apply(
        lambda r: r["quality_band"] if isinstance(r.get("quality_band"), str) and r["quality_band"] != ""
        else _quality_band_from_quality(r.get("quality", "")),
        axis=1,
    )
    if "face_size_band" not in full.columns:
        full["face_size_band"] = full.get("face_size", pd.Series(["unknown"] * len(full)))
    full["face_size_band"] = full["face_size_band"].fillna("unknown")
    full["person_cluster"] = full["person_cluster"].fillna(full["base_identity"]).fillna("unknown")

    # Coarse 'source' for real-side R-A.
    full["source"] = full["suite"].apply(_suite_to_source)

    # Chronic flag.
    full["chronic_flag"] = full.apply(
        lambda r: bool(_is_chronic(r.get("base_identity", ""), r.get("person_cluster", ""))),
        axis=1,
    )
    paired["source"] = paired["suite"].apply(_suite_to_source)
    paired["chronic_flag"] = paired.apply(
        lambda r: bool(_is_chronic(r.get("base_identity", ""), r.get("person_cluster", ""))),
        axis=1,
    )

    # Per-ckpt failure flag (real_FP for reals, fake_missed for fakes).
    for ckpt in CKPT_LIST:
        paired[f"fail_{ckpt}"] = paired.apply(
            lambda r: _ckpt_failure_at_t05(r, ckpt, int(r["label"])),
            axis=1,
        )

    return full, paired


# --------------------------------------------------------------------------- #
# Group construction                                                          #
# --------------------------------------------------------------------------- #

def _build_group(df: pd.DataFrame, label: int, scheme: str) -> pd.Series:
    """Return a per-row group_id string for the given scheme.

    Schemes (label-prefixed):
      F-A: fake × method_family
      F-B: fake × method_family × enhancer_family
      F-C: fake × method_family × enhancer_family × transport
      F-D: fake × method_family × enhancer_family × transport × quality_band
      F-E: F-D with quality_band collapsed to {hi, lo}

      R-A: real × source
      R-B: real × source × transport
      R-C: real × source × transport × chronic_flag
      R-D: real × source × transport × quality_band × chronic_flag
    """
    side = "fake" if label == 1 else "real"
    sub = df[df["label"] == label].copy()
    if scheme == "F-A":
        return side + "|" + sub["method_family"].astype(str)
    if scheme == "F-B":
        return side + "|" + sub["method_family"].astype(str) + "|" + sub["enhancer_family"].astype(str)
    if scheme == "F-C":
        return (
            side + "|" + sub["method_family"].astype(str)
            + "|" + sub["enhancer_family"].astype(str)
            + "|" + sub["transport"].astype(str)
        )
    if scheme == "F-D":
        return (
            side + "|" + sub["method_family"].astype(str)
            + "|" + sub["enhancer_family"].astype(str)
            + "|" + sub["transport"].astype(str)
            + "|" + sub["quality_band"].astype(str)
        )
    if scheme == "F-E":
        qb_hl = sub["quality_band"].apply(_quality_band_collapsed_hl)
        return (
            side + "|" + sub["method_family"].astype(str)
            + "|" + sub["enhancer_family"].astype(str)
            + "|" + sub["transport"].astype(str)
            + "|" + qb_hl
        )
    if scheme == "R-A":
        return side + "|" + sub["source"].astype(str)
    if scheme == "R-B":
        return side + "|" + sub["source"].astype(str) + "|" + sub["transport"].astype(str)
    if scheme == "R-C":
        return (
            side + "|" + sub["source"].astype(str)
            + "|" + sub["transport"].astype(str)
            + "|" + np.where(sub["chronic_flag"], "chronic", "regular")
        )
    if scheme == "R-D":
        return (
            side + "|" + sub["source"].astype(str)
            + "|" + sub["transport"].astype(str)
            + "|" + sub["quality_band"].astype(str)
            + "|" + np.where(sub["chronic_flag"], "chronic", "regular")
        )
    raise ValueError(scheme)


def _group_size_stats(sizes: List[int]) -> Dict[str, float]:
    if not sizes:
        return {}
    arr = np.array(sizes)
    return {
        "n_groups": int(len(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "n_below_min": int((arr < MIN_GROUP_SIZE).sum()),
        "share_below_min": float((arr < MIN_GROUP_SIZE).sum() / len(arr)),
        "max_group_share": float(arr.max() / arr.sum()) if arr.sum() > 0 else 0.0,
    }


def _failure_spread_per_ckpt(
    paired: pd.DataFrame, scheme: str, label: int
) -> Dict[str, Dict[str, float]]:
    """Per-ckpt: max - min failure-rate across groups (only groups with n>=MIN_GROUP_SIZE)."""
    sub = paired[paired["label"] == label].copy()
    sub["g"] = _build_group(paired, label, scheme).values
    out = {}
    for ckpt in CKPT_LIST:
        agg = sub.groupby("g").agg(n=("g", "size"), fr=(f"fail_{ckpt}", "mean")).reset_index()
        agg = agg[agg["n"] >= MIN_GROUP_SIZE]
        if len(agg) == 0:
            out[ckpt] = {"spread_pp": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
            continue
        out[ckpt] = {
            "spread_pp": float(agg["fr"].max() - agg["fr"].min()),
            "min": float(agg["fr"].min()),
            "max": float(agg["fr"].max()),
            "median": float(agg["fr"].median()),
        }
    return out


# --------------------------------------------------------------------------- #
# Per-scheme audit                                                            #
# --------------------------------------------------------------------------- #

FAKE_SCHEMES = ["F-A", "F-B", "F-C", "F-D", "F-E"]
REAL_SCHEMES = ["R-A", "R-B", "R-C", "R-D"]


def audit_single_side(full: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Fake schemes.
    for sc in FAKE_SCHEMES:
        # Group sizes: from the full manifest fake population.
        ids = _build_group(full, 1, sc)
        sizes = ids.value_counts().tolist()
        size_stats = _group_size_stats(sizes)
        # Failure spread: from paired (since outcomes only exist there).
        spread = _failure_spread_per_ckpt(paired, sc, 1)
        rows.append(
            {
                "scheme": sc,
                "side": "fake",
                **size_stats,
                **{f"spread_{ck}": spread[ck]["spread_pp"] for ck in CKPT_LIST},
                **{f"min_{ck}": spread[ck]["min"] for ck in CKPT_LIST},
                **{f"max_{ck}": spread[ck]["max"] for ck in CKPT_LIST},
            }
        )
    # Real schemes.
    for sc in REAL_SCHEMES:
        ids = _build_group(full, 0, sc)
        sizes = ids.value_counts().tolist()
        size_stats = _group_size_stats(sizes)
        spread = _failure_spread_per_ckpt(paired, sc, 0)
        rows.append(
            {
                "scheme": sc,
                "side": "real",
                **size_stats,
                **{f"spread_{ck}": spread[ck]["spread_pp"] for ck in CKPT_LIST},
                **{f"min_{ck}": spread[ck]["min"] for ck in CKPT_LIST},
                **{f"max_{ck}": spread[ck]["max"] for ck in CKPT_LIST},
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Train/eval stability proxy                                                  #
# --------------------------------------------------------------------------- #

def _stability_chi2(full: pd.DataFrame, scheme: str, label: int) -> float:
    """Cosine similarity between train (is_lockbox=False) and eval (is_lockbox=True)
    distribution over groups. 1.0 = identical shape, 0.0 = orthogonal.
    """
    sub = full[full["label"] == label].copy()
    sub["g"] = _build_group(full, label, scheme).values
    counts_train = sub[sub["is_lockbox"] == False]["g"].value_counts()  # noqa: E712
    counts_eval = sub[sub["is_lockbox"] == True]["g"].value_counts()    # noqa: E712
    keys = sorted(set(counts_train.index) | set(counts_eval.index))
    if not keys:
        return 0.0
    a = np.array([counts_train.get(k, 0) for k in keys], dtype=float)
    b = np.array([counts_eval.get(k, 0) for k in keys], dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))


# --------------------------------------------------------------------------- #
# Pairwise (F, R) evaluation                                                  #
# --------------------------------------------------------------------------- #

def evaluate_pair(
    full: pd.DataFrame,
    paired: pd.DataFrame,
    f_scheme: str,
    r_scheme: str,
) -> Dict[str, Any]:
    """Joint statistics for a (fake-side, real-side) pair."""
    # Build group IDs side by side, then merge into one column.
    full = full.copy()
    full["g"] = ""
    full.loc[full["label"] == 1, "g"] = _build_group(full, 1, f_scheme).values
    full.loc[full["label"] == 0, "g"] = _build_group(full, 0, r_scheme).values
    sizes = full["g"].value_counts().tolist()
    size_stats = _group_size_stats(sizes)

    # Failure spread on paired.
    paired = paired.copy()
    paired["g"] = ""
    paired.loc[paired["label"] == 1, "g"] = _build_group(paired, 1, f_scheme).values
    paired.loc[paired["label"] == 0, "g"] = _build_group(paired, 0, r_scheme).values

    fail_stats = {}
    for ckpt in CKPT_LIST:
        agg = paired.groupby("g").agg(n=("g", "size"), fr=(f"fail_{ckpt}", "mean")).reset_index()
        agg = agg[agg["n"] >= MIN_GROUP_SIZE]
        if len(agg) == 0:
            fail_stats[f"spread_{ckpt}"] = 0.0
            fail_stats[f"max_fail_{ckpt}"] = 0.0
            continue
        fail_stats[f"spread_{ckpt}"] = float(agg["fr"].max() - agg["fr"].min())
        fail_stats[f"max_fail_{ckpt}"] = float(agg["fr"].max())

    # Train/eval distribution stability (cosine over group counts).
    counts_train = full[full["is_lockbox"] == False]["g"].value_counts()  # noqa: E712
    counts_eval = full[full["is_lockbox"] == True]["g"].value_counts()    # noqa: E712
    keys = sorted(set(counts_train.index) | set(counts_eval.index))
    a = np.array([counts_train.get(k, 0) for k in keys], dtype=float)
    b = np.array([counts_eval.get(k, 0) for k in keys], dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos_train_eval = float((a @ b) / (na * nb)) if (na and nb) else 0.0

    # 2nd stability proxy: fraction of groups with BOTH train and eval representation.
    train_keys = set(counts_train[counts_train > 0].index)
    eval_keys = set(counts_eval[counts_eval > 0].index)
    if keys:
        coverage_both = len(train_keys & eval_keys) / len(keys)
    else:
        coverage_both = 0.0
    # 3rd stability proxy: among groups with eval samples, fraction also in train.
    if eval_keys:
        eval_train_recall = len(train_keys & eval_keys) / len(eval_keys)
    else:
        eval_train_recall = 0.0

    return {
        "f_scheme": f_scheme,
        "r_scheme": r_scheme,
        **size_stats,
        **fail_stats,
        "cos_train_eval": cos_train_eval,
        "coverage_both": coverage_both,
        "eval_train_recall": eval_train_recall,
    }


def rank_pairs(pairs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Rank by: pass DRO-stability rules, then maximise (avg_ckpt_spread + cos_train_eval - max_group_share*0.5)."""
    df = pd.DataFrame(pairs)
    df["avg_ckpt_spread"] = df[[f"spread_{ck}" for ck in CKPT_LIST]].mean(axis=1)
    df["passes_min_size"] = df["min"] >= MIN_GROUP_SIZE
    df["passes_max_share"] = df["max_group_share"] <= MAX_GROUP_DOMINANCE
    df["passes_spread"] = df["avg_ckpt_spread"] >= MIN_FAILURE_SPREAD
    df["passes_all"] = df["passes_min_size"] & df["passes_max_share"] & df["passes_spread"]
    # Composite score: spread + cos - 0.5*share. Higher is better.
    df["composite"] = (
        df["avg_ckpt_spread"]
        + 0.5 * df["cos_train_eval"]
        - 0.5 * df["max_group_share"]
    )
    df = df.sort_values(
        by=["passes_all", "composite"], ascending=[False, False]
    ).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    print("[group_id_design_audit] loading data…")
    full, paired = load_audit_frame()
    print(f"  full   = {len(full)} rows  ({(full['label']==0).sum()} real, {(full['label']==1).sum()} fake)")
    print(f"  paired = {len(paired)} rows ({(paired['label']==0).sum()} real, {(paired['label']==1).sum()} fake)")

    # ---------------- chronic_flag definition ---------------- #
    chronic_def = {
        "definition": (
            "binary flag set when base_identity exactly matches one of the chronic-6 "
            "subjects from Job 11 (analysis/job_11_identity_audit_2026-05-04/). The "
            "Job 11 cohort is at tau@FPR=10% on teams_real_all_dev; an identity is "
            "'chronic' if it appears in the worst-FPR table across at least one ckpt."
        ),
        "provenance": "memory/project_chronic_offenders_partition_per_ckpt_2026-05-04.md",
        "identities": CHRONIC_BASE_IDENTITIES,
        "match_rule": (
            "row.base_identity in chronic_list, OR (case-insensitive substring match of "
            "any chronic-6 identity inside base_identity) — captures suffixed variants "
            "like 'PC_Generator__s22__seg_5.0__real'."
        ),
        "n_real_chronic_in_full": int(((full["label"] == 0) & (full["chronic_flag"])).sum()),
        "n_fake_chronic_in_full": int(((full["label"] == 1) & (full["chronic_flag"])).sum()),
        "n_real_chronic_in_paired": int(((paired["label"] == 0) & (paired["chronic_flag"])).sum()),
        "n_fake_chronic_in_paired": int(((paired["label"] == 1) & (paired["chronic_flag"])).sum()),
    }
    with open(OUT_DIR / "chronic_flag_definition.json", "w") as f:
        json.dump(chronic_def, f, indent=2)
    print(
        f"  chronic_flag: {chronic_def['n_real_chronic_in_full']} real, "
        f"{chronic_def['n_fake_chronic_in_full']} fake in full manifest"
    )

    # ---------------- quality_band definition ---------------- #
    qb_def = {
        "definition": (
            "ternary {hi-q, lo-q, unknown} taken directly from manifest column "
            "'quality'. The manifest's binning is 'hi-q' for sharp/well-lit faces "
            "(approx top-third Laplacian variance) and 'lo-q' for low-resolution / "
            "low-contrast / webcam-style faces. 'unknown' covers training-set rows "
            "for which no parquet quality score is available."
        ),
        "provenance": "analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv (column 'quality')",
        "ternary": ["hi-q", "lo-q", "unknown"],
        "binary_collapse_for_F-E": {
            "hi": ["hi-q"],
            "lo": ["lo-q", "unknown"],
        },
        "counts_full": {
            band: int((full["quality_band"] == band).sum())
            for band in sorted(full["quality_band"].dropna().unique())
        },
    }
    with open(OUT_DIR / "quality_band_thresholds.json", "w") as f:
        json.dump(qb_def, f, indent=2)
    print(f"  quality_band counts: {qb_def['counts_full']}")

    # ---------------- single-side audit ---------------- #
    single = audit_single_side(full, paired)
    # Add stability cosine per scheme.
    side_label = {"fake": 1, "real": 0}
    single["cos_train_eval"] = single.apply(
        lambda r: _stability_chi2(full, r["scheme"], side_label[r["side"]]), axis=1
    )
    single.to_csv(OUT_DIR / "candidate_groups.csv", index=False)
    print(f"  wrote candidate_groups.csv ({len(single)} schemes)")

    # ---------------- pairwise (F, R) audit ---------------- #
    pairs = []
    for fs in FAKE_SCHEMES:
        for rs in REAL_SCHEMES:
            pairs.append(evaluate_pair(full, paired, fs, rs))
    pair_df = rank_pairs(pairs)
    pair_df.to_csv(OUT_DIR / "pairwise_evaluation.csv", index=False)
    print(f"  wrote pairwise_evaluation.csv ({len(pair_df)} pairs)")
    print("  top-3 pairs by composite:")
    for _, row in pair_df.head(3).iterrows():
        print(
            f"    rank={row['rank']:.0f}  ({row['f_scheme']}, {row['r_scheme']})  "
            f"min={row['min']:.0f}  med={row['median']:.0f}  "
            f"max_share={row['max_group_share']:.3f}  "
            f"avg_spread={row['avg_ckpt_spread']:.3f}  "
            f"cos_te={row['cos_train_eval']:.3f}  passes_all={row['passes_all']}"
        )

    # ---------------- recommendation.json ---------------- #
    top3 = pair_df.head(3).to_dict(orient="records")
    rationale_map = {}
    for rec in top3:
        rationale_map[f"{rec['f_scheme']}__{rec['r_scheme']}"] = {
            "rank": int(rec["rank"]),
            "f_scheme": rec["f_scheme"],
            "r_scheme": rec["r_scheme"],
            "n_groups": int(rec["n_groups"]),
            "min_group_size": int(rec["min"]),
            "median_group_size": float(rec["median"]),
            "max_group_share": float(rec["max_group_share"]),
            "avg_ckpt_spread_pp": float(rec["avg_ckpt_spread"]),
            "spread_per_ckpt": {ck: float(rec[f"spread_{ck}"]) for ck in CKPT_LIST},
            "max_fail_per_ckpt": {ck: float(rec[f"max_fail_{ck}"]) for ck in CKPT_LIST},
            "cos_train_eval": float(rec["cos_train_eval"]),
            "passes_all_thresholds": bool(rec["passes_all"]),
            "composite_score": float(rec["composite"]),
        }
    headline = top3[0]
    recommendation = {
        "headline_pair": {
            "f_scheme": headline["f_scheme"],
            "r_scheme": headline["r_scheme"],
            "rationale": (
                f"Fake-side {headline['f_scheme']} and real-side {headline['r_scheme']} "
                f"yields min group size {int(headline['min'])} (rule {MIN_GROUP_SIZE}), "
                f"median {float(headline['median']):.0f}, max-group share "
                f"{float(headline['max_group_share']):.3f} (rule ≤{MAX_GROUP_DOMINANCE}), "
                f"avg ckpt failure-rate spread {float(headline['avg_ckpt_spread']):.3f} "
                f"(rule ≥{MIN_FAILURE_SPREAD}), train/eval cosine "
                f"{float(headline['cos_train_eval']):.3f}. Passes all thresholds: "
                f"{bool(headline['passes_all'])}."
            ),
        },
        "thresholds": {
            "min_group_size": MIN_GROUP_SIZE,
            "min_failure_spread_pp": MIN_FAILURE_SPREAD,
            "max_group_dominance": MAX_GROUP_DOMINANCE,
        },
        "top_3": rationale_map,
        "ckpts_evaluated": CKPT_LIST,
        "key_observations": {
            "external_vcd_real_skip_rule": (
                "external_vcd_real lane has 0 paired fakes (1200 unpaired reals, ~80 ids). "
                "Pair-rank cannot fire. The collate function must skip this lane for the "
                "pair-rank loss term but it MUST contribute to the real-side GroupDRO "
                "term (it is a real-side group of its own under R-B/R-C/R-D)."
            ),
            "F-E_vs_F-D_tradeoff": (
                "F-E (binary quality_band {hi,lo}) reduces fake-side cardinality and "
                "raises min group size vs F-D (ternary). Useful when F-D's smallest "
                "groups fall below the 50-frame threshold."
            ),
            "chronic_flag_payoff": (
                "Adding chronic_flag (R-C, R-D) creates a tiny but high-failure-rate "
                "group. This is the lever that gives DRO something to do — without it, "
                "real-side spreads are dominated by the source axis alone."
            ),
        },
    }
    with open(OUT_DIR / "recommendation.json", "w") as f:
        json.dump(recommendation, f, indent=2)
    print("  wrote recommendation.json")

    # ---------------- code snippet ---------------- #
    f_scheme_top = headline["f_scheme"]
    r_scheme_top = headline["r_scheme"]
    snippet = build_snippet(f_scheme_top, r_scheme_top)
    with open(OUT_DIR / "group_id_python_snippet.py", "w") as f:
        f.write(snippet)
    print("  wrote group_id_python_snippet.py")

    print("[group_id_design_audit] done.")


def build_snippet(f_scheme: str, r_scheme: str) -> str:
    return f'''"""
Multi-axis GroupDRO group_id construction — recommended by
analysis/group_id_design_audit_2026-05-06.

Fake-side scheme: {f_scheme}
Real-side scheme: {r_scheme}

Drop this into the data loader / collate function. Returns a single string
group_id per row that the existing trainer/mixins/group_dro.py can use after
extending to accept arbitrary str keys (the current code uses int method_id
via data_params.method_mapping; you must build a dict {{group_str -> int}} at
config-load time and pass it as `data_params.group_id_mapping`).

Chronic identity list comes from
memory/project_chronic_offenders_partition_per_ckpt_2026-05-04.md (Job 11,
2026-05-04).
"""

CHRONIC_IDENTITIES = [
    "bla_bla_chow",
    "bla_bla_chow__s2",
    "PC_Generator__s22",
    "PC_Generator__s45",
    "roy_d",
    "Q__s6",
]


def is_chronic(base_identity: str) -> bool:
    if not isinstance(base_identity, str):
        return False
    if base_identity in CHRONIC_IDENTITIES:
        return True
    s = base_identity.lower()
    return any(cid.lower() in s for cid in CHRONIC_IDENTITIES)


def quality_band(quality: str | None) -> str:
    if isinstance(quality, str) and quality.lower() in {{"hi-q", "lo-q"}}:
        return quality.lower()
    return "unknown"


def make_group_id(row: dict) -> str | None:
    """Build the asymmetric group_id for a manifest row.

    Returns None when the row should be skipped from the GroupDRO term entirely
    (currently: never — even external_vcd_real is grouped as a real lane). The
    PAIR-RANK loss separately skips external_vcd_real via is_unpaired_real.

    Required row keys:
      label                 0 or 1
      method_family         e.g. 'deeplive', 'inswapper', 'real_or_unknown'
      enhancer_family       e.g. 'gpen', 'gfpgan', 'none'
      transport             e.g. 'raw_capture', 'teams_capture', 'visomaster'
      quality               'hi-q' / 'lo-q' / None
      source                coarse lane id (e.g. 'teams_real_dev', 'visomaster_v2')
      base_identity         exact identity string

    Skip rule for pair-rank: an upstream collate flag `is_unpaired_real` (set on
    rows from external_vcd_real, where there is no paired fake) must suppress
    the pair-rank loss contribution. The GroupDRO term still applies.
    """
    label = int(row["label"])
    qb = quality_band(row.get("quality"))
    transport = row.get("transport", "raw_capture")
    if label == 1:
        # Fake-side: {f_scheme}
        return (
            f"fake|{{row.get('method_family', 'unknown')}}"
            f"|{{row.get('enhancer_family', 'none')}}"
            f"|{{transport}}"
            f"|{{qb}}"
        )
    # Real-side: {r_scheme}
    chronic_tag = "chronic" if is_chronic(row.get("base_identity", "")) else "regular"
    return (
        f"real|{{row.get('source', 'unknown')}}"
        f"|{{transport}}"
        f"|{{qb}}"
        f"|{{chronic_tag}}"
    )


def build_group_id_mapping(manifest_rows) -> dict:
    """Pre-pass at config-load time. Walk all training rows, collect distinct
    group_id strings, return dict[str -> int] for trainer/mixins/group_dro.py.
    Pass this as data_params.group_id_mapping (replaces method_mapping)."""
    seen = sorted({{make_group_id(r) for r in manifest_rows if make_group_id(r) is not None}})
    return {{g: i for i, g in enumerate(seen)}}
'''


if __name__ == "__main__":
    main()
