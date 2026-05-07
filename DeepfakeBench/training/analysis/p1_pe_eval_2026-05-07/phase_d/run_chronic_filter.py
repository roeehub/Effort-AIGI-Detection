"""
Phase D — F5 chronic-FP cohort filter for P1 (PE_PAIR_RANK_DRO).
Reads Phase A per-frame teams_real_all_dev reports, filters to chronic-6
identities, computes 3-tier close criterion read.

BUGFIX 2026-05-07: original used `extract_base_identity` (regex strips
`__s\\d+`/`__seq\\d+`) BEFORE the chronic-check. That collapsed
`PC_Generator__s22` and `PC_Generator__s45` to `PC_Generator`, so the
chronic-id list (which includes the session suffix) no longer matched.
PC_Generator + Q__s6 (the actual F5 targets) were silently dropped from
the chronic aggregate. Net effect: F5 read was wrong by ~30+pp.

Fix: match each chronic id as a prefix of the raw video_id
(case-insensitive). `extract_base_identity` is retained only for display
grouping in the per-identity breakdown.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------ paths / config

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
PHASE_DIR = ROOT / "analysis/p1_pe_eval_2026-05-07/phase_d"
RAW_REPORTS = ROOT / "analysis/p1_pe_eval_2026-05-07/raw_reports/phase_a"
SCORECARD_DIR = ROOT / "analysis/p1_pe_eval_2026-05-07/scorecard"
CKPT_MAP_YAML = ROOT / "arena/checkpoint_maps/teams_target_domain.p1_pe_pair_rank_2026-05-07.yaml"
CHRONIC_JSON = ROOT / "analysis/group_id_design_audit_2026-05-06/outputs/chronic_flag_definition.json"

PHASE_DIR.mkdir(parents=True, exist_ok=True)

SUITE = "teams_real_all_dev"
TAU_FIXED = 0.5
P8A_KEY = "p8a_reference_step5000"
PC_GENERATOR_IDS = ("PC_Generator__s22", "PC_Generator__s45")

# match the base-identity strip rule used by job_11_identity_audit_2026-05-04
# (collapses session/frame/crop/seq tokens to the person id).
_IDENT_STRIP_BASE = re.compile(
    r"(__seq\d+|__seg_[\d.]+|__s\d+|_s\d+|_frame_\d+|_seq\d+|__real|__fake|_real|_fake|_crop_\d+|__[0-9a-f]{6,})",
    flags=re.IGNORECASE,
)

# ------------------------------------------------------------ logging

LOG_PATH = PHASE_DIR / "run.log"
if LOG_PATH.exists():
    LOG_PATH.unlink()

logger = logging.getLogger("phase_d")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
logger.addHandler(sh)


# ------------------------------------------------------------ helpers

def load_chronic() -> tuple[list[str], dict]:
    with open(CHRONIC_JSON) as f:
        d = json.load(f)
    ids = list(d["identities"])
    return ids, d


def load_ckpt_keys() -> list[str]:
    keys: list[str] = []
    pattern = re.compile(r"^([A-Z][A-Z0-9_]*):\s*\"gs://")
    with open(CKPT_MAP_YAML) as f:
        for line in f:
            m = pattern.match(line)
            if m:
                keys.append(m.group(1).lower())
    return keys


def extract_base_identity(vid: str) -> str:
    if not isinstance(vid, str):
        return "UNK"
    s = vid
    prev = None
    while prev != s:
        prev = s
        s = _IDENT_STRIP_BASE.sub("", s)
    return s.strip("_") or "UNK"


def video_matches_cid(video_id: str, cid: str) -> bool:
    """Prefix-match cid against raw video_id (case-insensitive).

    Examples (cid -> video_id):
      'PC_Generator__s22' matches 'PC_Generator__s22__seg_488.2__real'
      'bla_bla_chow'      matches 'bla_bla_chow__s1__seg_...' AND
                                  'bla_bla_chow__s2__seg_...'
      'roy_d'             matches 'Roy_D__seq1001__real' (case-insensitive)
    """
    if not isinstance(video_id, str):
        return False
    vl = video_id.lower()
    cl = cid.lower()
    return vl == cl or vl.startswith(cl + "_") or vl.startswith(cl + "__")


def is_chronic_row(video_id: str, chronic_ids: list[str]) -> bool:
    """Returns True if video_id is part of ANY chronic identity (prefix match)."""
    return any(video_matches_cid(video_id, cid) for cid in chronic_ids)


def load_selected_thresholds() -> dict[str, float]:
    sc_path = SCORECARD_DIR / "selected_threshold_scorecard.csv"
    if not sc_path.exists():
        logger.warning("selected_threshold_scorecard.csv missing at %s — selected-tau column will be NaN", sc_path)
        return {}
    sc = pd.read_csv(sc_path)
    if "checkpoint_key" not in sc.columns or "threshold" not in sc.columns:
        logger.warning("selected_threshold_scorecard.csv schema mismatch (cols=%s)", list(sc.columns))
        return {}
    out: dict[str, float] = {}
    for _, row in sc.drop_duplicates(subset=["checkpoint_key"]).iterrows():
        out[str(row["checkpoint_key"]).lower()] = float(row["threshold"])
    return out


def report_path(ckpt_tag: str) -> Path:
    return RAW_REPORTS / f"{SUITE}_{ckpt_tag}_frames_report.csv"


def aggregate_fpr(scores: pd.Series, tau: float) -> tuple[int, int, float]:
    n_total = int(len(scores))
    if n_total == 0:
        return 0, 0, float("nan")
    n_fp = int((scores >= tau).sum())
    return n_total, n_fp, n_fp / n_total


# ------------------------------------------------------------ main

def main() -> int:
    chronic_ids, chronic_def = load_chronic()
    logger.info("chronic-6: %s", chronic_ids)
    logger.info("chronic_flag_definition.json provenance: %s", chronic_def.get("provenance", "?"))

    ckpt_keys = load_ckpt_keys()
    logger.info("ckpts from yaml (%d): %s", len(ckpt_keys), ckpt_keys)

    if not RAW_REPORTS.exists() or not any(RAW_REPORTS.iterdir()):
        logger.warning("Phase A reports not yet present at %s", RAW_REPORTS)
        print(f"Phase A reports not yet present at {RAW_REPORTS}")
        return 0

    missing = [k for k in ckpt_keys if not report_path(k).exists()]
    present = [k for k in ckpt_keys if report_path(k).exists()]
    if missing:
        logger.warning("missing per-frame reports for ckpts: %s", missing)
    if not present:
        logger.warning("no teams_real_all_dev frame reports found in %s for ckpts %s", RAW_REPORTS, ckpt_keys)
        print(f"Phase A reports not yet present at {RAW_REPORTS}")
        return 0

    selected_taus = load_selected_thresholds()

    agg_rows = []
    per_id_rows = []
    pc_rows = []

    for ckpt in present:
        rp = report_path(ckpt)
        df = pd.read_csv(rp)
        if "video_id" not in df.columns or "frame_prob" not in df.columns:
            logger.error("schema mismatch in %s (cols=%s) — skipping", rp, list(df.columns))
            continue
        df = df.copy()
        # `base_identity` is for grouping/display only — DO NOT use for chronic match
        # (regex strips `__s22/__s45` and breaks the chronic-id alignment).
        df["base_identity"] = df["video_id"].apply(extract_base_identity)
        df["is_chronic"] = df["video_id"].apply(lambda v: is_chronic_row(v, chronic_ids))

        chronic_df = df[df["is_chronic"]].copy()
        n_real_total = int(len(df))
        n_chronic = int(len(chronic_df))
        logger.info("%s: n_total=%d n_chronic=%d (%.1f%%)",
                    ckpt, n_real_total, n_chronic, 100 * n_chronic / max(1, n_real_total))

        tau_selected = selected_taus.get(ckpt, float("nan"))
        for tau_label, tau_value in [("tau_0.5", TAU_FIXED), ("tau_selected", tau_selected)]:
            if pd.isna(tau_value):
                agg_rows.append({
                    "ckpt": ckpt, "tau_label": tau_label, "tau_value": float("nan"),
                    "n_chronic_total": n_chronic, "n_chronic_fp": 0, "chronic_fpr": float("nan"),
                })
                continue
            n_total, n_fp, fpr = aggregate_fpr(chronic_df["frame_prob"], float(tau_value))
            agg_rows.append({
                "ckpt": ckpt, "tau_label": tau_label, "tau_value": float(tau_value),
                "n_chronic_total": n_total, "n_chronic_fp": n_fp, "chronic_fpr": fpr,
            })

        for cid in chronic_ids:
            # Prefix-match raw video_id (case-insensitive) — see docstring of video_matches_cid.
            sub = chronic_df[chronic_df["video_id"].apply(lambda v: video_matches_cid(v, cid))]
            for tau_label, tau_value in [("tau_0.5", TAU_FIXED), ("tau_selected", tau_selected)]:
                if pd.isna(tau_value):
                    per_id_rows.append({
                        "ckpt": ckpt, "identity": cid, "tau_label": tau_label,
                        "tau_value": float("nan"), "n_total": int(len(sub)),
                        "n_fp": 0, "fpr": float("nan"),
                    })
                    continue
                n_total, n_fp, fpr = aggregate_fpr(sub["frame_prob"], float(tau_value))
                per_id_rows.append({
                    "ckpt": ckpt, "identity": cid, "tau_label": tau_label,
                    "tau_value": float(tau_value), "n_total": n_total,
                    "n_fp": n_fp, "fpr": fpr,
                })

        # Match PC_Generator__s22/s45 via raw video_id prefix (consistent with chronic match above).
        pc_mask = chronic_df["video_id"].apply(
            lambda v: any(video_matches_cid(v, p) for p in PC_GENERATOR_IDS)
        )
        pc_sub = chronic_df[pc_mask]
        for tau_label, tau_value in [("tau_0.5", TAU_FIXED), ("tau_selected", tau_selected)]:
            if pd.isna(tau_value):
                pc_rows.append({
                    "ckpt": ckpt, "tau_label": tau_label, "tau_value": float("nan"),
                    "n_total": int(len(pc_sub)), "n_fp": 0, "fpr": float("nan"),
                })
                continue
            n_total, n_fp, fpr = aggregate_fpr(pc_sub["frame_prob"], float(tau_value))
            pc_rows.append({
                "ckpt": ckpt, "tau_label": tau_label, "tau_value": float(tau_value),
                "n_total": n_total, "n_fp": n_fp, "fpr": fpr,
            })

    agg_df = pd.DataFrame(agg_rows)
    per_id_df = pd.DataFrame(per_id_rows)
    pc_df = pd.DataFrame(pc_rows)

    agg_df.to_csv(PHASE_DIR / "chronic6_aggregate_fpr.csv", index=False)
    per_id_df.to_csv(PHASE_DIR / "per_identity_fpr.csv", index=False)
    pc_df.to_csv(PHASE_DIR / "pc_generator_cluster_fpr.csv", index=False)
    logger.info("wrote chronic6_aggregate_fpr.csv (%d rows)", len(agg_df))
    logger.info("wrote per_identity_fpr.csv (%d rows)", len(per_id_df))
    logger.info("wrote pc_generator_cluster_fpr.csv (%d rows)", len(pc_df))

    # ------------------------------------------------------------ 3-tier close criterion
    p8a_fpr_by_tau: dict[str, float] = {}
    if not agg_df.empty:
        p8a = agg_df[agg_df["ckpt"] == P8A_KEY]
        for _, r in p8a.iterrows():
            p8a_fpr_by_tau[r["tau_label"]] = float(r["chronic_fpr"])

    bundle_keys = [k for k in ckpt_keys if k.startswith("p1_bundle_")]
    pairrank_keys = [k for k in ckpt_keys if k.startswith("p1_pairrank_")]

    print("")
    print("=" * 72)
    print("Phase D — 3-tier close criterion read (chronic-6 FPR)")
    print("=" * 72)
    print(f"Slot 1 = BUNDLE (multi-axis GroupDRO + pair-rank); Slot 2 = PAIRRANK_ONLY")
    print(f"P8A baseline FPR by tau: {p8a_fpr_by_tau}")
    print("")
    fmt_hdr = f"{'slot1_ckpt':<32}{'slot2_ckpt':<32}{'tau':<14}{'s1_fpr':<10}{'s2_fpr':<10}{'p8a':<10}{'(1)<=p8a':<10}{'(2)<=8%':<10}{'(3)>=10pp':<12}"
    print(fmt_hdr)
    print("-" * len(fmt_hdr))

    for s1 in bundle_keys:
        for s2 in pairrank_keys:
            for tau_label in ["tau_0.5", "tau_selected"]:
                s1_row = agg_df[(agg_df["ckpt"] == s1) & (agg_df["tau_label"] == tau_label)]
                s2_row = agg_df[(agg_df["ckpt"] == s2) & (agg_df["tau_label"] == tau_label)]
                if s1_row.empty or s2_row.empty:
                    continue
                s1_fpr = float(s1_row["chronic_fpr"].iloc[0])
                s2_fpr = float(s2_row["chronic_fpr"].iloc[0])
                p8a_fpr = p8a_fpr_by_tau.get(tau_label, float("nan"))
                tier1 = (not pd.isna(s1_fpr)) and (not pd.isna(p8a_fpr)) and (s1_fpr <= p8a_fpr)
                tier2 = (not pd.isna(s1_fpr)) and (s1_fpr <= 0.08)
                tier3 = (not pd.isna(s1_fpr)) and (not pd.isna(s2_fpr)) and ((s2_fpr - s1_fpr) >= 0.10)
                print(f"{s1:<32}{s2:<32}{tau_label:<14}"
                      f"{s1_fpr:<10.4f}{s2_fpr:<10.4f}{p8a_fpr:<10.4f}"
                      f"{str(tier1):<10}{str(tier2):<10}{str(tier3):<12}")
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
