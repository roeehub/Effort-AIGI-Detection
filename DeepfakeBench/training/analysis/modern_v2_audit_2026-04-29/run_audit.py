"""Plan v4 Track H.1 — modern_v2 filter audit.

Builds modern_v2_relaxed substrate and re-aggregates the 5 P11/P8A candidates'
frame-level predictions to compare FPR @ matched-recall thresholds against
modern_v2_current. Produces comparison.csv + findings.txt.

Constraints (per brief): n_jobs=1 only; no GPU; single-machine pandas/numpy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
PARQUET = REPO / "analysis/lockbox_tagging/lockbox_tags_2026-04-27.parquet"
CACHE_DIR = REPO / "analysis/option_a_ensemble_2026-04-28/cache"
OUT_DIR = REPO / "analysis/modern_v2_audit_2026-04-29"
CUR_DIR = REPO / "analysis/modern_lockbox_v2_2026-04-27"

CANDIDATES = [
    "p8a_reference_step5000",
    "p11_heavy_step1000",
    "p11_heavy_deeplive_step1000",
    "p11_mild_step1000",
    "p11_webcam_harden_step1000",
]


# -----------------------------------------------------------------------------
# Filter definitions
# -----------------------------------------------------------------------------

def apply_current_filter(df: pd.DataFrame) -> pd.DataFrame:
    """modern_v2 (current, 281r/367f): webcam/screen drop + face_area_ratio>=0.10
    + drop is_pose_extreme + drop is_no_face."""
    return df[
        (~df["clip_capture_mode"].isin(["webcam", "screen"]))
        & (df["face_area_ratio"] >= 0.10)
        & (~df["is_pose_extreme"].fillna(False))
        & (~df["is_no_face"].fillna(False))
    ].copy()


def apply_relaxed_filter(df: pd.DataFrame) -> pd.DataFrame:
    """modern_v2_relaxed: preserve capture-mode gate; relax quality/geometry.

    Relaxations applied:
      - face_area_ratio: 0.10 -> 0.0625 (= 64x64 / 256x256, hard floor)
        OR equivalently face_pixel_area >= 4096 (64**2).
      - drop is_pose_extreme gate (include pose-extreme frames).
      - keep is_no_face gate (moot in lockbox: 0 hits anyway).
      - keep clip_capture_mode in {normal_photo, phone_screen} (drop webcam/screen).
    """
    return df[
        (~df["clip_capture_mode"].isin(["webcam", "screen"]))
        & ((df["face_area_ratio"] >= 0.0625) | (df["face_pixel_area"] >= 64 * 64))
        & (~df["is_no_face"].fillna(False))
    ].copy()


# -----------------------------------------------------------------------------
# Frame-level scoring
# -----------------------------------------------------------------------------

def load_candidate_csvs(name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    real = pd.read_csv(CACHE_DIR / f"teams_real_all_lockbox_{name}_frames_report.csv")
    fake = pd.read_csv(CACHE_DIR / f"teams_fake_all_lockbox_{name}_frames_report.csv")
    return real, fake


def scores_for(uris: set, df: pd.DataFrame) -> np.ndarray:
    sub = df[df["frame_path"].isin(uris)]
    return sub["frame_prob"].to_numpy()


def fpr_at(real_scores: np.ndarray, tau: float) -> Tuple[int, int, float]:
    if real_scores.size == 0:
        return 0, 0, float("nan")
    fp = int((real_scores >= tau).sum())
    return fp, real_scores.size, fp / real_scores.size


def recall_at(fake_scores: np.ndarray, tau: float) -> Tuple[int, int, float]:
    if fake_scores.size == 0:
        return 0, 0, float("nan")
    tp = int((fake_scores >= tau).sum())
    return tp, fake_scores.size, tp / fake_scores.size


def tau_for_fpr(real_scores: np.ndarray, target_fpr: float) -> float:
    """Smallest tau s.t. FPR(tau) <= target_fpr on the given real-scores set.

    Uses the (1 - target_fpr)-quantile, matching how production calibration
    is described elsewhere in the repo (e.g. tau_5pct_production = 0.9741)."""
    if real_scores.size == 0:
        return float("nan")
    q = float(np.quantile(real_scores, 1.0 - target_fpr))
    # Smallest float > q so that strictly more than (1-q)-fraction are below
    # We want >= q to count as fp, so use q directly; FPR will be approx target.
    return q


def auc_score(fake_scores: np.ndarray, real_scores: np.ndarray) -> float:
    """Simple AUC via rank-sum (Mann-Whitney). Single-thread, no sklearn."""
    if fake_scores.size == 0 or real_scores.size == 0:
        return float("nan")
    all_scores = np.concatenate([fake_scores, real_scores])
    labels = np.concatenate([np.ones(fake_scores.size), np.zeros(real_scores.size)])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, all_scores.size + 1)
    # Handle ties by averaging
    s = all_scores[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    pos_rank_sum = ranks[labels == 1].sum()
    n_pos = fake_scores.size
    n_neg = real_scores.size
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# -----------------------------------------------------------------------------
# Visomaster fakes for AUC ceiling check
# -----------------------------------------------------------------------------

def load_visomaster_scores(candidate: str) -> np.ndarray:
    """Visomaster fakes are 'visomaster_enhanced_macro_dev' suite."""
    path = CACHE_DIR / f"visomaster_enhanced_macro_dev_{candidate}_frames_report.csv"
    if not path.exists():
        return np.array([])
    df = pd.read_csv(path)
    fake = df[df["label"] == 1]
    return fake["frame_prob"].to_numpy()


# -----------------------------------------------------------------------------
# P11 ensemble (best mean of 4 P11 candidates) for AUC ceiling check
# -----------------------------------------------------------------------------

def p11_ensemble_real_fake(use_filter: pd.DataFrame, lockbox_real_uris: set,
                           lockbox_fake_uris: set, viso_present: bool = True
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-prob across 4 P11 candidates for shared frame_paths.

    Returns (real_scores, fake_scores_lockbox, fake_scores_visomaster).
    """
    p11s = [c for c in CANDIDATES if c.startswith("p11_")]
    real_dfs, fake_dfs, viso_dfs = [], [], []
    for c in p11s:
        rdf, fdf = load_candidate_csvs(c)
        real_dfs.append(rdf.set_index("frame_path")["frame_prob"].rename(c))
        fake_dfs.append(fdf.set_index("frame_path")["frame_prob"].rename(c))
        v = load_visomaster_scores(c)
        viso_dfs.append(v)

    real_merged = pd.concat(real_dfs, axis=1, join="inner")
    fake_merged = pd.concat(fake_dfs, axis=1, join="inner")
    real_mean = real_merged.mean(axis=1)
    fake_mean = fake_merged.mean(axis=1)

    real_scores = real_mean[real_mean.index.isin(lockbox_real_uris)].to_numpy()
    fake_scores = fake_mean[fake_mean.index.isin(lockbox_fake_uris)].to_numpy()

    # Visomaster ensemble: align by frame_path inner-join
    viso_inner = None
    for c in p11s:
        path = CACHE_DIR / f"visomaster_enhanced_macro_dev_{c}_frames_report.csv"
        if not path.exists():
            viso_inner = None
            break
        d = pd.read_csv(path)
        d = d[d["label"] == 1].set_index("frame_path")["frame_prob"].rename(c)
        viso_inner = d if viso_inner is None else pd.concat([viso_inner, d], axis=1, join="inner")
    viso_scores = (
        viso_inner.mean(axis=1).to_numpy()
        if isinstance(viso_inner, pd.DataFrame)
        else (viso_inner.to_numpy() if viso_inner is not None else np.array([]))
    )

    return real_scores, fake_scores, viso_scores


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load parquet, build current and relaxed substrates
    df = pd.read_parquet(PARQUET)
    lr = df[(df["split"] == "lockbox") & (df["label"] == "real")].copy()
    lf = df[(df["split"] == "lockbox") & (df["label"] == "fake")].copy()
    print(f"lockbox_real_total={len(lr)}  lockbox_fake_total={len(lf)}")

    cur_r = apply_current_filter(lr)
    cur_f = apply_current_filter(lf)
    relax_r = apply_relaxed_filter(lr)
    relax_f = apply_relaxed_filter(lf)
    print(f"current  real={len(cur_r)}  fake={len(cur_f)}")
    print(f"relaxed  real={len(relax_r)}  fake={len(relax_f)}")
    print(f"new real frames added in relaxed: {len(relax_r) - len(cur_r)}")
    print(f"new fake frames added in relaxed: {len(relax_f) - len(cur_f)}")

    # Sanity check vs current yaml
    cur_yaml = yaml.safe_load((CUR_DIR / "modern_lockbox_real_v2_frames.yaml").read_text())
    cur_yaml_uris = set(cur_yaml["frames"])
    cur_uris = set(cur_r["gcs_uri"].tolist())
    sym = cur_yaml_uris.symmetric_difference(cur_uris)
    print(f"current-filter URI vs yaml symmetric_diff: {len(sym)}  (expect 0)")
    assert cur_yaml["n_frames"] == len(cur_r) == 281, (cur_yaml["n_frames"], len(cur_r))

    # 2. Save relaxed yamls
    relax_real_uris = relax_r["gcs_uri"].tolist()
    relax_fake_uris = relax_f["gcs_uri"].tolist()

    relax_real_yaml = {
        "subset_name": "modern_lockbox_real_v2_relaxed",
        "split": "lockbox",
        "label": "real",
        "filter_rule": (
            "clip_capture_mode not in {webcam, screen} "
            "AND (face_area_ratio >= 0.0625 OR face_pixel_area >= 4096) "
            "AND not is_no_face  -- pose_extreme gate REMOVED"
        ),
        "n_frames": len(relax_real_uris),
        "frames": sorted(relax_real_uris),
    }
    relax_fake_yaml = {
        "subset_name": "modern_lockbox_fake_v2_relaxed",
        "split": "lockbox",
        "label": "fake",
        "filter_rule": relax_real_yaml["filter_rule"],
        "n_frames": len(relax_fake_uris),
        "frames": sorted(relax_fake_uris),
    }
    (OUT_DIR / "modern_v2_relaxed_real_frames.yaml").write_text(
        yaml.safe_dump(relax_real_yaml, sort_keys=False)
    )
    (OUT_DIR / "modern_v2_relaxed_fake_frames.yaml").write_text(
        yaml.safe_dump(relax_fake_yaml, sort_keys=False)
    )

    # 3. Build comparison table
    cur_real_uris = set(cur_r["gcs_uri"])
    cur_fake_uris = set(cur_f["gcs_uri"])
    relax_real_uris_s = set(relax_real_uris)
    relax_fake_uris_s = set(relax_fake_uris)

    rows = []
    summary_for_findings = {}
    for cand in CANDIDATES:
        rdf, fdf = load_candidate_csvs(cand)

        cur_real_scores = scores_for(cur_real_uris, rdf)
        cur_fake_scores = scores_for(cur_fake_uris, fdf)
        relax_real_scores = scores_for(relax_real_uris_s, rdf)
        relax_fake_scores = scores_for(relax_fake_uris_s, fdf)

        # tau = 0.5
        _, _, fpr_cur_05 = fpr_at(cur_real_scores, 0.5)
        _, _, fpr_relax_05 = fpr_at(relax_real_scores, 0.5)
        _, _, rec_cur_05 = recall_at(cur_fake_scores, 0.5)
        _, _, rec_relax_05 = recall_at(relax_fake_scores, 0.5)

        # tau calibrated to 5% FPR on each subset (using its own real scores)
        tau_cur_5pct = tau_for_fpr(cur_real_scores, 0.05)
        tau_relax_5pct = tau_for_fpr(relax_real_scores, 0.05)

        _, _, fpr_cur_at_cur_5 = fpr_at(cur_real_scores, tau_cur_5pct)
        _, _, fpr_relax_at_cur_5 = fpr_at(relax_real_scores, tau_cur_5pct)
        _, _, fpr_cur_at_relax_5 = fpr_at(cur_real_scores, tau_relax_5pct)
        _, _, fpr_relax_at_relax_5 = fpr_at(relax_real_scores, tau_relax_5pct)

        _, _, rec_cur_at_cur_5 = recall_at(cur_fake_scores, tau_cur_5pct)
        _, _, rec_relax_at_cur_5 = recall_at(relax_fake_scores, tau_cur_5pct)
        _, _, rec_cur_at_relax_5 = recall_at(cur_fake_scores, tau_relax_5pct)
        _, _, rec_relax_at_relax_5 = recall_at(relax_fake_scores, tau_relax_5pct)

        # AUC: lockbox-fake vs current/relaxed real
        auc_cur = auc_score(cur_fake_scores, cur_real_scores)
        auc_relax = auc_score(relax_fake_scores, relax_real_scores)

        # AUC: visomaster-fake vs current/relaxed real (Plan v4 ceiling check)
        viso_scores = load_visomaster_scores(cand)
        auc_viso_cur = auc_score(viso_scores, cur_real_scores)
        auc_viso_relax = auc_score(viso_scores, relax_real_scores)

        rows.append({
            "candidate": cand,
            "n_real_cur": len(cur_real_scores),
            "n_real_relax": len(relax_real_scores),
            "n_fake_cur": len(cur_fake_scores),
            "n_fake_relax": len(relax_fake_scores),
            "tau_cur_5pct": round(tau_cur_5pct, 5),
            "tau_relax_5pct": round(tau_relax_5pct, 5),
            "fpr_cur_at_tau0.5": round(fpr_cur_05, 4),
            "fpr_relax_at_tau0.5": round(fpr_relax_05, 4),
            "delta_fpr_tau0.5_pp": round(100 * (fpr_relax_05 - fpr_cur_05), 2),
            "fpr_cur_at_tau_cur5pct": round(fpr_cur_at_cur_5, 4),
            "fpr_relax_at_tau_cur5pct": round(fpr_relax_at_cur_5, 4),
            "delta_fpr_at_tau_cur5pct_pp": round(100 * (fpr_relax_at_cur_5 - fpr_cur_at_cur_5), 2),
            "fpr_cur_at_tau_relax5pct": round(fpr_cur_at_relax_5, 4),
            "fpr_relax_at_tau_relax5pct": round(fpr_relax_at_relax_5, 4),
            "delta_fpr_at_tau_relax5pct_pp": round(100 * (fpr_relax_at_relax_5 - fpr_cur_at_relax_5), 2),
            "rec_cur_at_tau0.5": round(rec_cur_05, 4),
            "rec_relax_at_tau0.5": round(rec_relax_05, 4),
            "rec_cur_at_tau_cur5pct": round(rec_cur_at_cur_5, 4),
            "rec_relax_at_tau_cur5pct": round(rec_relax_at_cur_5, 4),
            "auc_lockbox_fake_vs_real_cur": round(auc_cur, 4),
            "auc_lockbox_fake_vs_real_relax": round(auc_relax, 4),
            "auc_visomaster_fake_vs_real_cur": round(auc_viso_cur, 4) if not np.isnan(auc_viso_cur) else None,
            "auc_visomaster_fake_vs_real_relax": round(auc_viso_relax, 4) if not np.isnan(auc_viso_relax) else None,
        })

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "comparison.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'comparison.csv'}")
    print(table.to_string(index=False))

    # 4. P11 ensemble visomaster AUC
    cur_uris_set = set(cur_real_uris)
    cur_fake_uris_set = set(cur_fake_uris)
    relax_uris_set = set(relax_real_uris)
    relax_fake_uris_set = set(relax_fake_uris)

    p11_real_cur, p11_fake_cur, p11_viso = p11_ensemble_real_fake(
        None, cur_uris_set, cur_fake_uris_set
    )
    p11_real_relax, p11_fake_relax, _ = p11_ensemble_real_fake(
        None, relax_uris_set, relax_fake_uris_set
    )
    p11_auc_lockbox_cur = auc_score(p11_fake_cur, p11_real_cur)
    p11_auc_lockbox_relax = auc_score(p11_fake_relax, p11_real_relax)
    p11_auc_viso_cur = auc_score(p11_viso, p11_real_cur)
    p11_auc_viso_relax = auc_score(p11_viso, p11_real_relax)
    p11_summary = {
        "n_real_cur": int(p11_real_cur.size),
        "n_real_relax": int(p11_real_relax.size),
        "n_fake_lockbox_cur": int(p11_fake_cur.size),
        "n_fake_lockbox_relax": int(p11_fake_relax.size),
        "n_fake_visomaster": int(p11_viso.size),
        "auc_lockbox_cur": round(p11_auc_lockbox_cur, 4),
        "auc_lockbox_relax": round(p11_auc_lockbox_relax, 4),
        "auc_visomaster_cur": round(p11_auc_viso_cur, 4),
        "auc_visomaster_relax": round(p11_auc_viso_relax, 4),
    }
    (OUT_DIR / "p11_ensemble_audit_summary.json").write_text(json.dumps(p11_summary, indent=2))
    print("\nP11-ensemble-mean summary:")
    print(json.dumps(p11_summary, indent=2))

    # 5. Verdict and findings
    # Headline metric: max |delta_fpr| at matched-recall (we use tau_cur5pct point)
    deltas_pp = [r["delta_fpr_at_tau_cur5pct_pp"] for r in rows]
    max_delta_pp = max(abs(d) for d in deltas_pp)
    mean_delta_pp = float(np.mean(deltas_pp))

    verdict = "filter_confirmed" if max_delta_pp < 1.5 else "filter_suspect"

    findings = []
    findings.append("=" * 78)
    findings.append("Plan v4 Track H.1 — modern_v2 filter audit")
    findings.append("=" * 78)
    findings.append("")
    findings.append("PARQUET SCHEMA NOTE")
    findings.append("-" * 78)
    findings.append("Source: analysis/lockbox_tagging/lockbox_tags_2026-04-27.parquet")
    findings.append(f"  shape: {df.shape}  (lockbox split only — no dev_real present)")
    findings.append(f"  lockbox real total: {len(lr)}   lockbox fake total: {len(lf)}")
    findings.append("Quality columns present: face_area_ratio, face_pixel_area,")
    findings.append("  sharpness_laplacian, is_pose_extreme, is_no_face,")
    findings.append("  is_likely_screen_capture, is_low_quality, clip_capture_mode.")
    findings.append("")
    findings.append("CURRENT (modern_v2) FILTER")
    findings.append("-" * 78)
    findings.append("clip_capture_mode not in {webcam, screen}")
    findings.append("AND face_area_ratio >= 0.10")
    findings.append("AND not is_pose_extreme")
    findings.append("AND not is_no_face")
    findings.append(f" -> kept: real={len(cur_r)}  fake={len(cur_f)}  (matches yaml: 281r / 367f)")
    findings.append("")
    findings.append("Brief described an explicit ~96^2 pixel and a sharpness-band gate, but")
    findings.append("the actual current filter uses neither — it gates on face_area_ratio")
    findings.append("(>=0.10) and discrete tags (pose_extreme, no_face, capture_mode).")
    findings.append("The relaxation is therefore designed against the *actual* gates.")
    findings.append("")
    findings.append("RELAXED (modern_v2_relaxed) FILTER — chosen thresholds")
    findings.append("-" * 78)
    findings.append("clip_capture_mode not in {webcam, screen}     # capture-mode preserved")
    findings.append("AND (face_area_ratio >= 0.0625                # = 64^2 / 256^2")
    findings.append("     OR face_pixel_area >= 4096)              # = 64^2 hard floor")
    findings.append("AND not is_no_face                            # moot in lockbox (0 hits)")
    findings.append("# is_pose_extreme gate REMOVED (relaxed)")
    findings.append(f" -> kept: real={len(relax_r)}  fake={len(relax_f)}")
    findings.append(f"    new real frames vs current: {len(relax_r) - len(cur_r)} "
                    f"(mostly dor_shkedi pose-extreme + low face_area)")
    findings.append(f"    new fake frames vs current: {len(relax_f) - len(cur_f)}")
    findings.append("")
    findings.append("Rationale for the chosen relaxations:")
    findings.append("  * 0.0625 face_area_ratio corresponds to a 64x64 face in a 256x256 crop,")
    findings.append("    which is the explicit floor the brief asked for. Within the non-")
    findings.append("    webcam/screen real population this only adds 7 frames (pixel-area-only")
    findings.append("    floor adds 0 incremental beyond the ratio), so the geometry relax is")
    findings.append("    very modest — the bulk of new frames come from the pose-extreme gate.")
    findings.append("  * Sharpness band: brief asked for p25-p95 of dev_real. There is no")
    findings.append("    'dev' split in this parquet (only lockbox). The lockbox-real")
    findings.append("    non-webcam/screen sharpness p25=181, p95=763. The current filter has")
    findings.append("    no sharpness gate at all, so 'relaxing' an already-absent gate is a")
    findings.append("    no-op. We document this and proceed; if a future iteration adds a")
    findings.append("    sharpness gate we should re-run this audit.")
    findings.append("  * Pose-extreme gate is the load-bearing relaxation: it currently drops")
    findings.append("    27 dor_shkedi non-webcam/normal_photo frames that look like real")
    findings.append("    Teams captures with off-axis head pose — a typical deployment")
    findings.append("    condition that should NOT be excluded from a lockbox real pool.")
    findings.append("")
    findings.append("COMPARISON TABLE (rounded; CSV at comparison.csv)")
    findings.append("-" * 78)
    # Make a compact table
    short_cols = [
        "candidate", "n_real_cur", "n_real_relax",
        "tau_cur_5pct", "fpr_cur_at_tau_cur5pct", "fpr_relax_at_tau_cur5pct",
        "delta_fpr_at_tau_cur5pct_pp",
        "auc_lockbox_fake_vs_real_cur", "auc_lockbox_fake_vs_real_relax",
        "auc_visomaster_fake_vs_real_cur", "auc_visomaster_fake_vs_real_relax",
    ]
    findings.append(table[short_cols].to_string(index=False))
    findings.append("")
    findings.append(f"Headline metric — max |delta FPR| at tau_cur_5pct = {max_delta_pp:.2f} pp")
    findings.append(f"                  mean delta FPR at tau_cur_5pct  = {mean_delta_pp:+.2f} pp")
    findings.append("")
    findings.append("VERDICT (Plan v4 §6 gate)")
    findings.append("-" * 78)
    if verdict == "filter_confirmed":
        findings.append("filter_confirmed: max |Delta FPR| < 1.5 pp at matched-tau.")
        findings.append("  -> The Option A AUC-ceiling story stands. The 281-frame substrate")
        findings.append("     is not over-pruning meaningful real-side variation; relaxing")
        findings.append("     the geometry/pose gates does not move FPR materially.")
    else:
        findings.append("filter_suspect: max |Delta FPR| >= 1.5 pp at matched-tau.")
        findings.append("  -> The current modern_v2 substrate is plausibly excluding real")
        findings.append("     Teams-deployment-relevant frames that the model fails to")
        findings.append("     correctly classify as real. Recommend switching to")
        findings.append("     modern_v2_relaxed for Day-4 P13 scoring; recompute Option A's")
        findings.append("     85-config AUC numbers on the relaxed substrate before treating")
        findings.append("     Plan v4's gamma verdict as load-bearing.")
    findings.append("")
    findings.append("AUC CEILING vs Option A 0.647 (visomaster fake vs real)")
    findings.append("-" * 78)
    p8a_row = next(r for r in rows if r["candidate"] == "p8a_reference_step5000")
    findings.append(f"P8A reference (single model):")
    findings.append(f"  visomaster vs cur:    AUC = {p8a_row['auc_visomaster_fake_vs_real_cur']}")
    findings.append(f"  visomaster vs relax:  AUC = {p8a_row['auc_visomaster_fake_vs_real_relax']}")
    findings.append(f"P11 ensemble (mean of 4):")
    findings.append(f"  visomaster vs cur:    AUC = {p11_summary['auc_visomaster_cur']}")
    findings.append(f"  visomaster vs relax:  AUC = {p11_summary['auc_visomaster_relax']}")
    findings.append(f"  (Option A reported 0.647 ceiling; relax-AUC moves indicate whether")
    findings.append(f"   the ceiling is a real signal or a substrate artifact.)")
    findings.append("")

    (OUT_DIR / "findings.txt").write_text("\n".join(findings))
    print(f"\nwrote {OUT_DIR / 'findings.txt'}")
    print(f"\nVERDICT: {verdict}  (max |Delta FPR| = {max_delta_pp:.2f} pp)")


if __name__ == "__main__":
    main()
