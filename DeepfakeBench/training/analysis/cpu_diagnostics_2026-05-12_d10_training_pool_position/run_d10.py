"""D10 — Training-pool position on the dev-vs-lockbox KLIEP discriminator axis.

Refits the KLIEP-style LR discriminator (D8 estimator C) on (dev-real vs
lockbox-real) CLIP-frozen L11 features, then projects three CLIP-frozen real
pools onto that 1-D axis and characterizes the overlap geometry:

  Pool A — dev-real (n=2000)  — from D8 cache
  Pool B — lockbox-real (n=414) — from D8 cache
  Pool C — training-real (n=371) — from D7 cache (train_teams_real_pool.parquet
          subset that was locally cached)

Numerical outputs (CSV + FACTS md). No interpretation in this script.

Operational constraints (per AGENT_GUIDE + memory):
  - n_jobs=1 in every sklearn call.
  - CPU only (no GPU). Caches are pre-extracted (D7 + D8 already paid the
    extraction cost).
  - Forbidden words excluded in the FACTS document; numerical only.

Inputs:
  - analysis/cpu_diagnostics_2026-05-12_d7_fpr_decomposition/_clip_l11_features.npz
    (eval_feats 3416×768, train_feats 371×768)
  - analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced/
    outputs/clip_frozen_l11__n4839.npz (features 4839×768, ordered
    2000 dev_real, 2000 dev_fake, 414 lockbox_real, 425 lockbox_fake)
  - analysis/lockbox_tagging/full_tags_2026-04-27.parquet (for label/path
    ordering, replayed with seed=42 stratified sampling matching D8)
  - analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet (IQ axes)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
)
THIS_DIR = REPO_ROOT / "analysis" / "cpu_diagnostics_2026-05-12_d10_training_pool_position"
OUTPUTS = THIS_DIR / "outputs"
LOG_PATH = THIS_DIR / "_run.log"

D7_CACHE = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-12_d7_fpr_decomposition/_clip_l11_features.npz"
D8_CACHE = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced/outputs/clip_frozen_l11__n4839.npz"
LOCKBOX_PARQUET = REPO_ROOT / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
TRAIN_REAL_POOL_PARQUET = REPO_ROOT / "analysis/iq_data_atlas_2026-05-08/_cache/train_teams_real_pool.parquet"
IQ_ATLAS_PARQUET = REPO_ROOT / "analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet"

IQ_AXES = ["lap_var", "min_dim", "color_a_dev", "color_b_dev", "saturation_mean", "luma_mean"]

SEED = 42
N_BOOTSTRAP = 20

# ---------------------------------------------------------------------------
# Logging.
# ---------------------------------------------------------------------------
logger = logging.getLogger("d10")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="w"),
            logging.StreamHandler(),
        ],
    )


# ---------------------------------------------------------------------------
# Load & order frames matching the D8 dev/lockbox sampling.
# ---------------------------------------------------------------------------
def replay_dev_lockbox_paths() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Same as D8.replay_dev_lockbox_paths — seed=42 stratified 2000/2000 dev,
    full lockbox."""
    df = pd.read_parquet(LOCKBOX_PARQUET)
    df = df[df["local_path"].astype(str).str.len() > 0].reset_index(drop=True)
    dev_df = df[df["split"] == "dev"].copy()
    dev_real = dev_df[dev_df["label"] == "real"]
    dev_fake = dev_df[dev_df["label"] == "fake"]
    rng = np.random.default_rng(seed=42)
    dev_real_idx = rng.choice(len(dev_real), size=2000, replace=False)
    dev_fake_idx = rng.choice(len(dev_fake), size=2000, replace=False)
    dev_sample = pd.concat(
        [dev_real.iloc[dev_real_idx], dev_fake.iloc[dev_fake_idx]]
    ).reset_index(drop=True)
    lb_df = df[df["split"] == "lockbox"].copy().reset_index(drop=True)
    return dev_sample, lb_df


def load_real_pools() -> Dict[str, np.ndarray]:
    """Return CLIP-frozen real features for {dev, lockbox, train} pools."""
    # D8 cache: [dev (4000)] + [lockbox (839)] in path order.
    d8 = np.load(D8_CACHE, allow_pickle=True)
    feats = d8["features"].astype(np.float32)
    assert feats.shape[0] == 4839, f"D8 cache shape mismatch: {feats.shape}"

    dev_sample, lb_df = replay_dev_lockbox_paths()
    dev_labels = (dev_sample["label"] == "fake").astype(np.int64).to_numpy()
    lb_labels = (lb_df["label"] == "fake").astype(np.int64).to_numpy()

    clip_dev = feats[:4000]
    clip_lb = feats[4000:]
    assert clip_dev.shape[0] == len(dev_sample), "dev/feat count mismatch"
    assert clip_lb.shape[0] == len(lb_df), "lb/feat count mismatch"

    dev_real_mask = (dev_labels == 0)
    lb_real_mask = (lb_labels == 0)

    dev_real_feats = clip_dev[dev_real_mask]
    lb_real_feats = clip_lb[lb_real_mask]
    dev_real_paths = dev_sample.loc[dev_real_mask, "gcs_uri"].tolist()
    lb_real_paths = lb_df.loc[lb_real_mask, "gcs_uri"].tolist()

    # D7 cache: train_feats 371×768 + train_basenames
    d7 = np.load(D7_CACHE, allow_pickle=True)
    train_feats = d7["train_feats"].astype(np.float32)
    train_basenames = list(d7["train_basenames"])

    logger.info(
        "loaded pools: dev_real=%d lockbox_real=%d train_real=%d",
        dev_real_feats.shape[0], lb_real_feats.shape[0], train_feats.shape[0],
    )
    return {
        "dev_real_feats": dev_real_feats,
        "lockbox_real_feats": lb_real_feats,
        "train_real_feats": train_feats,
        "dev_real_paths": dev_real_paths,
        "lockbox_real_paths": lb_real_paths,
        "train_real_basenames": train_basenames,
    }


# ---------------------------------------------------------------------------
# Fit the KLIEP-style discriminator (LR on l2-normalized features) and project.
# Mirrors D8 estimator C; the only departure from D8 is max_iter=2000 (D8 used
# 5000) per the task brief.
# ---------------------------------------------------------------------------
def fit_kliep_discriminator(
    dev_real_feats: np.ndarray, lb_real_feats: np.ndarray
) -> Dict:
    """LogisticRegression on l2-normalized CLIP-frozen features.

    Returns dict with w (768,), b (scalar), accuracy, coef_norm.
    """
    from sklearn.linear_model import LogisticRegression

    X = np.concatenate([dev_real_feats, lb_real_feats], axis=0).astype(np.float64)
    y = np.concatenate(
        [np.zeros(len(dev_real_feats)), np.ones(len(lb_real_feats))]
    ).astype(np.int64)
    # l2-normalize each row (cosine geometry of CLIP).
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    clf = LogisticRegression(
        C=1.0, max_iter=2000, solver="lbfgs", n_jobs=1, class_weight="balanced",
    )
    clf.fit(X, y)
    w = clf.coef_[0].astype(np.float64)
    b = float(clf.intercept_[0])
    acc = float(clf.score(X, y))
    coef_norm = float(np.linalg.norm(w))
    return {
        "w": w,
        "b": b,
        "accuracy": acc,
        "coef_norm": coef_norm,
        "clf": clf,
    }


def project_onto_axis(feats: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Compute scalar w · x_normalized + b for each row x of feats."""
    Xn = feats.astype(np.float64)
    Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12)
    return Xn @ w + b


# ---------------------------------------------------------------------------
# Distribution stats and overlap statistics.
# ---------------------------------------------------------------------------
def projection_stats(proj: np.ndarray, name: str) -> Dict:
    return {
        "pool": name,
        "n": int(len(proj)),
        "mean": float(np.mean(proj)),
        "std": float(np.std(proj, ddof=1)) if len(proj) > 1 else float("nan"),
        "min": float(np.min(proj)),
        "p5": float(np.percentile(proj, 5)),
        "p25": float(np.percentile(proj, 25)),
        "p50": float(np.percentile(proj, 50)),
        "p75": float(np.percentile(proj, 75)),
        "p95": float(np.percentile(proj, 95)),
        "max": float(np.max(proj)),
    }


def overlap_fraction(query: np.ndarray, ref: np.ndarray) -> float:
    """Fraction of `query` whose values fall within the [p5, p95] range of `ref`."""
    lo = float(np.percentile(ref, 5))
    hi = float(np.percentile(ref, 95))
    return float(((query >= lo) & (query <= hi)).mean())


# ---------------------------------------------------------------------------
# IQ-PC1 vector and angle to KLIEP axis.
# ---------------------------------------------------------------------------
def fit_axis_direction(X: np.ndarray, axis_values: np.ndarray) -> np.ndarray:
    """LinearRegression(features -> z-scored axis). Return unit vector. Matches
    `analysis/cpu_diagnostics_2026-05-12_d2_encoder_separation/run_d2.py:369`
    and run_d6.py:209.
    """
    from sklearn.linear_model import LinearRegression

    mu = float(axis_values.mean())
    sd = float(axis_values.std())
    if sd <= 0:
        return np.zeros(X.shape[1], dtype=np.float64)
    z = (axis_values - mu) / sd
    reg = LinearRegression(n_jobs=1)
    reg.fit(X, z)
    w = reg.coef_.astype(np.float64)
    n = np.linalg.norm(w)
    if n <= 0:
        return w
    return w / n


def compute_iq_pc1(
    feats: np.ndarray,
    iq_panel: pd.DataFrame,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """For each of the 6 IQ axes, regress feats -> z-scored axis to get a unit
    direction. Stack 6 × 768, SVD, return Vt[0] as IQ-PC1.
    """
    X = feats[valid_mask].astype(np.float64)
    w_axes: List[np.ndarray] = []
    axis_counts: Dict[str, int] = {}
    for ax in IQ_AXES:
        ax_vals = iq_panel[ax].values[valid_mask].astype(np.float64)
        # exclude any nan in the axis column
        finite = np.isfinite(ax_vals)
        if finite.sum() < 10:
            w_ax = np.zeros(X.shape[1], dtype=np.float64)
            axis_counts[ax] = int(finite.sum())
            w_axes.append(w_ax)
            continue
        w_ax = fit_axis_direction(X[finite], ax_vals[finite])
        axis_counts[ax] = int(finite.sum())
        w_axes.append(w_ax)
    W = np.stack(w_axes, axis=0)
    norms = np.linalg.norm(W, axis=1)
    valid_rows = W[norms > 1e-12]
    if valid_rows.shape[0] < 2:
        return np.zeros(X.shape[1], dtype=np.float64), axis_counts
    _, _, Vt = np.linalg.svd(valid_rows, full_matrices=False)
    return Vt[0], axis_counts


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Unsigned acute angle in degrees, via |cos|."""
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu <= 0 or nv <= 0:
        return float("nan")
    c = float(np.dot(u, v) / (nu * nv))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(abs(c))))


def signed_angle(u: np.ndarray, v: np.ndarray) -> float:
    """Signed angle [0, 180] via arccos of raw cos."""
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu <= 0 or nv <= 0:
        return float("nan")
    c = float(np.dot(u, v) / (nu * nv))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


# ---------------------------------------------------------------------------
# Bootstrap to assess train-projection mean stability.
# ---------------------------------------------------------------------------
def bootstrap_train_proj_mean(
    dev_real_feats: np.ndarray,
    lb_real_feats: np.ndarray,
    train_real_feats: np.ndarray,
    B: int = N_BOOTSTRAP,
    base_seed: int = SEED,
) -> Dict:
    """For each of B bootstrap resamples of (dev, lockbox), refit KLIEP and
    record the mean train projection.
    """
    rng = np.random.default_rng(base_seed)
    train_means: List[float] = []
    dev_means: List[float] = []
    lb_means: List[float] = []
    accs: List[float] = []
    for b in range(B):
        dev_idx = rng.choice(len(dev_real_feats), size=len(dev_real_feats), replace=True)
        lb_idx = rng.choice(len(lb_real_feats), size=len(lb_real_feats), replace=True)
        boot = fit_kliep_discriminator(dev_real_feats[dev_idx], lb_real_feats[lb_idx])
        proj_train = project_onto_axis(train_real_feats, boot["w"], boot["b"])
        proj_dev = project_onto_axis(dev_real_feats, boot["w"], boot["b"])
        proj_lb = project_onto_axis(lb_real_feats, boot["w"], boot["b"])
        train_means.append(float(np.mean(proj_train)))
        dev_means.append(float(np.mean(proj_dev)))
        lb_means.append(float(np.mean(proj_lb)))
        accs.append(boot["accuracy"])
    return {
        "B": B,
        "train_mean_mean": float(np.mean(train_means)),
        "train_mean_std": float(np.std(train_means, ddof=1)),
        "dev_mean_mean": float(np.mean(dev_means)),
        "dev_mean_std": float(np.std(dev_means, ddof=1)),
        "lockbox_mean_mean": float(np.mean(lb_means)),
        "lockbox_mean_std": float(np.std(lb_means, ddof=1)),
        "boot_accuracy_mean": float(np.mean(accs)),
        "boot_accuracy_std": float(np.std(accs, ddof=1)),
        "train_means_all": train_means,
        "dev_means_all": dev_means,
        "lockbox_means_all": lb_means,
    }


# ---------------------------------------------------------------------------
# Plot helper (optional).
# ---------------------------------------------------------------------------
def plot_histogram(
    dev_proj: np.ndarray, lb_proj: np.ndarray, train_proj: np.ndarray, path: Path
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        bins = np.linspace(
            min(dev_proj.min(), lb_proj.min(), train_proj.min()),
            max(dev_proj.max(), lb_proj.max(), train_proj.max()),
            60,
        )
        ax.hist(
            dev_proj, bins=bins, alpha=0.5, label=f"dev_real (n={len(dev_proj)})",
            density=True, color="steelblue",
        )
        ax.hist(
            lb_proj, bins=bins, alpha=0.5, label=f"lockbox_real (n={len(lb_proj)})",
            density=True, color="firebrick",
        )
        ax.hist(
            train_proj, bins=bins, alpha=0.5,
            label=f"train_real (n={len(train_proj)})",
            density=True, color="seagreen", linewidth=1.0,
        )
        ax.set_xlabel("KLIEP-axis projection (w · x_norm + b)")
        ax.set_ylabel("density")
        ax.set_title(
            "Training-pool position on D8 KLIEP axis "
            "(LR on l2-normalized CLIP-frozen L11)"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("plot saved: %s", path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("plot failed: %s", exc)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    setup_logging()
    t0 = time.time()
    logger.info("D10 — training-pool position on dev/lockbox KLIEP axis — start")

    # 1. Load all three real pools.
    pools = load_real_pools()
    dev_real_feats = pools["dev_real_feats"]
    lb_real_feats = pools["lockbox_real_feats"]
    train_real_feats = pools["train_real_feats"]
    dev_real_paths = pools["dev_real_paths"]
    lb_real_paths = pools["lockbox_real_paths"]
    train_basenames = pools["train_real_basenames"]
    logger.info(
        "shapes: dev=%s lockbox=%s train=%s",
        dev_real_feats.shape, lb_real_feats.shape, train_real_feats.shape,
    )

    # 2. Fit KLIEP discriminator.
    logger.info("[step 2] fit KLIEP discriminator (LR on l2-normalized CLIP-frozen)")
    kliep = fit_kliep_discriminator(dev_real_feats, lb_real_feats)
    w = kliep["w"]
    b = kliep["b"]
    logger.info(
        "KLIEP: balanced-acc=%.4f, |w|=%.4f, b=%.4f",
        kliep["accuracy"], kliep["coef_norm"], b,
    )

    # 3. Project all three pools.
    logger.info("[step 3] projecting pools onto KLIEP axis")
    dev_proj = project_onto_axis(dev_real_feats, w, b)
    lb_proj = project_onto_axis(lb_real_feats, w, b)
    train_proj = project_onto_axis(train_real_feats, w, b)

    # 4. Distribution stats.
    logger.info("[step 4] distribution statistics")
    stats_rows = [
        projection_stats(dev_proj, "dev_real"),
        projection_stats(lb_proj, "lockbox_real"),
        projection_stats(train_proj, "train_real"),
    ]
    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(OUTPUTS / "projection_distribution_stats.csv", index=False)
    logger.info("dev mean=%.4f, lockbox mean=%.4f, train mean=%.4f",
                stats_rows[0]["mean"], stats_rows[1]["mean"], stats_rows[2]["mean"])

    # 4b. Sign / decision-boundary breakdown. The KLIEP discriminator boundary is
    # at projection == 0 (since proj = w · x_norm + b).
    sign_rows = []
    for nm, p in [("dev_real", dev_proj), ("lockbox_real", lb_proj), ("train_real", train_proj)]:
        n = len(p)
        n_pos = int((p >= 0).sum())
        n_neg = int((p < 0).sum())
        sign_rows.append({
            "pool": nm,
            "n": n,
            "n_projection_geq_0": n_pos,
            "frac_projection_geq_0": float(n_pos / n) if n > 0 else float("nan"),
            "n_projection_lt_0": n_neg,
            "frac_projection_lt_0": float(n_neg / n) if n > 0 else float("nan"),
        })
    pd.DataFrame(sign_rows).to_csv(OUTPUTS / "decision_boundary_split.csv", index=False)
    for r in sign_rows:
        logger.info("  decision-boundary %s: %d/%d (%.2f%%) at proj>=0",
                    r["pool"], r["n_projection_geq_0"], r["n"], r["frac_projection_geq_0"]*100)

    # 5. Per-frame projections (for downstream analysis / viewer).
    df_perframe = pd.concat([
        pd.DataFrame({
            "pool": "dev_real",
            "frame_id": dev_real_paths,
            "projection": dev_proj,
        }),
        pd.DataFrame({
            "pool": "lockbox_real",
            "frame_id": lb_real_paths,
            "projection": lb_proj,
        }),
        pd.DataFrame({
            "pool": "train_real",
            "frame_id": train_basenames,
            "projection": train_proj,
        }),
    ], ignore_index=True)
    df_perframe.to_csv(OUTPUTS / "projection_per_frame.csv", index=False)

    # 6. Overlap fractions.
    logger.info("[step 6] overlap fractions / Wasserstein / KS")
    from scipy.stats import wasserstein_distance, ks_2samp
    overlap_rows = []
    for q_name, q in [("train_real", train_proj), ("dev_real", dev_proj), ("lockbox_real", lb_proj)]:
        for r_name, r in [("dev_real", dev_proj), ("lockbox_real", lb_proj), ("train_real", train_proj)]:
            if q_name == r_name:
                continue
            overlap_rows.append({
                "query_pool": q_name,
                "ref_pool": r_name,
                "overlap_fraction_in_p5_p95": overlap_fraction(q, r),
            })
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(OUTPUTS / "overlap_fractions.csv", index=False)

    # Wasserstein + KS pairwise.
    distance_rows = []
    pairs = [
        ("train_real", train_proj, "dev_real", dev_proj),
        ("train_real", train_proj, "lockbox_real", lb_proj),
        ("dev_real", dev_proj, "lockbox_real", lb_proj),
    ]
    for n1, p1, n2, p2 in pairs:
        wd = float(wasserstein_distance(p1, p2))
        ks_stat, ks_pval = ks_2samp(p1, p2)
        distance_rows.append({
            "pool_a": n1,
            "pool_b": n2,
            "wasserstein_1": wd,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "mean_diff": float(np.mean(p1) - np.mean(p2)),
        })
    distance_df = pd.DataFrame(distance_rows)
    distance_df.to_csv(OUTPUTS / "pairwise_distances.csv", index=False)

    # 7. IQ-PC1 angle.
    logger.info("[step 7] computing IQ-PC1 and angle to KLIEP axis")
    # Build IQ panel for dev+lockbox real frames using the atlas.
    atlas = pd.read_parquet(IQ_ATLAS_PARQUET)[["frame_path"] + IQ_AXES].copy()
    atlas = atlas.drop_duplicates(subset=["frame_path"], keep="first")

    all_paths = dev_real_paths + lb_real_paths
    all_feats = np.concatenate([dev_real_feats, lb_real_feats], axis=0)
    panel = pd.DataFrame({"frame_path": all_paths})
    merged = panel.merge(atlas, on="frame_path", how="left")
    iq_mat = merged[IQ_AXES].to_numpy(dtype=np.float64)
    valid = np.all(np.isfinite(iq_mat), axis=1)
    logger.info("IQ atlas join coverage: %d / %d frames have all 6 axes",
                int(valid.sum()), len(merged))

    iq_pc1, axis_counts = compute_iq_pc1(all_feats, merged, valid)
    angle_acute = angle_between(w, iq_pc1)
    angle_raw = signed_angle(w, iq_pc1)
    logger.info(
        "angle(KLIEP, IQ-PC1): acute=%.2f deg, raw=%.2f deg; |iq_pc1|=%.4f",
        angle_acute, angle_raw, float(np.linalg.norm(iq_pc1)),
    )

    # 7b. Per-IQ-axis angle (each of the 6 axes' fitted direction vs KLIEP).
    per_axis_rows = []
    Xv = all_feats[valid].astype(np.float64)
    for ax in IQ_AXES:
        ax_vals = merged[ax].values[valid].astype(np.float64)
        finite = np.isfinite(ax_vals)
        if finite.sum() < 10:
            per_axis_rows.append({
                "axis": ax, "n_used": int(finite.sum()),
                "angle_acute_deg_to_KLIEP_w": float("nan"),
            })
            continue
        w_ax = fit_axis_direction(Xv[finite], ax_vals[finite])
        per_axis_rows.append({
            "axis": ax,
            "n_used": int(finite.sum()),
            "angle_acute_deg_to_KLIEP_w": angle_between(w, w_ax),
        })
    pd.DataFrame(per_axis_rows).to_csv(OUTPUTS / "per_axis_angles.csv", index=False)
    for row in per_axis_rows:
        logger.info("  per-axis %s: %.2f deg (n=%d)",
                    row["axis"], row["angle_acute_deg_to_KLIEP_w"], row["n_used"])

    iq_angle_rows = [
        {
            "vector_a": "KLIEP_w",
            "vector_b": "IQ_PC1",
            "n_frames_used_for_iq_pc1": int(valid.sum()),
            "angle_acute_deg": angle_acute,
            "angle_raw_deg": angle_raw,
            "cos_raw": float(
                np.dot(w, iq_pc1)
                / (np.linalg.norm(w) * max(np.linalg.norm(iq_pc1), 1e-12))
            ),
            "kliep_w_norm": float(np.linalg.norm(w)),
            "iq_pc1_norm": float(np.linalg.norm(iq_pc1)),
        }
    ]
    pd.DataFrame(iq_angle_rows).to_csv(OUTPUTS / "iq_axis_angle.csv", index=False)
    pd.DataFrame(
        [{"axis": ax, "n_finite_used": ct} for ax, ct in axis_counts.items()]
    ).to_csv(OUTPUTS / "iq_axis_counts.csv", index=False)

    # 8. Bootstrap (B=20).
    logger.info("[step 8] bootstrap KLIEP (B=%d) on (dev, lockbox) resamples", N_BOOTSTRAP)
    boot = bootstrap_train_proj_mean(
        dev_real_feats, lb_real_feats, train_real_feats,
        B=N_BOOTSTRAP, base_seed=SEED,
    )
    pd.DataFrame([{
        "B": boot["B"],
        "train_mean_mean": boot["train_mean_mean"],
        "train_mean_std": boot["train_mean_std"],
        "dev_mean_mean": boot["dev_mean_mean"],
        "dev_mean_std": boot["dev_mean_std"],
        "lockbox_mean_mean": boot["lockbox_mean_mean"],
        "lockbox_mean_std": boot["lockbox_mean_std"],
        "boot_accuracy_mean": boot["boot_accuracy_mean"],
        "boot_accuracy_std": boot["boot_accuracy_std"],
    }]).to_csv(OUTPUTS / "bootstrap_summary.csv", index=False)
    # Per-bootstrap row.
    pd.DataFrame({
        "bootstrap_ix": list(range(boot["B"])),
        "train_mean": boot["train_means_all"],
        "dev_mean": boot["dev_means_all"],
        "lockbox_mean": boot["lockbox_means_all"],
    }).to_csv(OUTPUTS / "bootstrap_per_iter.csv", index=False)

    # 9. Plot histogram.
    logger.info("[step 9] plot histogram")
    plot_histogram(dev_proj, lb_proj, train_proj, OUTPUTS / "projection_histogram.png")

    # 10. Summary JSON.
    summary = {
        "kliep_balanced_accuracy": kliep["accuracy"],
        "kliep_coef_norm": kliep["coef_norm"],
        "kliep_intercept": b,
        "n_dev_real": int(dev_real_feats.shape[0]),
        "n_lockbox_real": int(lb_real_feats.shape[0]),
        "n_train_real": int(train_real_feats.shape[0]),
        "stats": stats_rows,
        "overlap_rows": overlap_rows,
        "distance_rows": distance_rows,
        "iq_angle_rows": iq_angle_rows,
        "per_axis_angles": per_axis_rows,
        "decision_boundary_split": sign_rows,
        "bootstrap_summary": {k: boot[k] for k in [
            "B", "train_mean_mean", "train_mean_std",
            "dev_mean_mean", "dev_mean_std",
            "lockbox_mean_mean", "lockbox_mean_std",
            "boot_accuracy_mean", "boot_accuracy_std",
        ]},
        "elapsed_sec": float(time.time() - t0),
    }
    with open(OUTPUTS / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("D10 done in %.1f sec", time.time() - t0)
    logger.info("summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
