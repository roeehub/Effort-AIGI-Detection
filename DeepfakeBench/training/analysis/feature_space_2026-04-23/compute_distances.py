#!/usr/bin/env python3
"""Post-process feature-space extraction: pairwise Fréchet + MMD + centroid
distances between all sources. Runs locally after the Vertex job finishes.

Reads per-source .npz files from a GCS prefix (or local dir) and writes:
  - distance_matrix_frechet.csv
  - distance_matrix_mmd_rbf.csv
  - distance_matrix_centroid.csv
  - per_source_summary.csv       (n, prob_mean, prob_fake_rate, feat_norm)
  - REPORT.md                    (human-readable findings)

Usage:
    python3 compute_distances.py \
        --input gs://training-job-outputs/feature_space_analysis/<run_id>/ \
        --output_dir analysis/feature_space_2026-04-23/
"""
from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("feature-space-distances")


def download_from_gcs(gcs_prefix: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    if not gcs_prefix.endswith("/"):
        gcs_prefix = gcs_prefix + "/"
    logger.info("gsutil -m cp %s* %s/", gcs_prefix, local_dir)
    subprocess.run(
        ["gsutil", "-m", "cp", f"{gcs_prefix}*", str(local_dir) + "/"],
        check=True,
    )


def load_sources(local_dir: Path) -> dict:
    out = {}
    for p in sorted(local_dir.glob("*.npz")):
        data = np.load(p, allow_pickle=True)
        name = str(data["name"])
        out[name] = {
            "features": data["features"],
            "probs": data["probs"],
            "provenance": str(data["provenance"]),
            "label": str(data["label"]),
            "uris": data["uris"].tolist() if "uris" in data.files else [],
        }
        logger.info("  loaded %s n=%d features_shape=%s prob_mean=%.3f",
                    name, data["features"].shape[0], data["features"].shape,
                    float(data["probs"].mean()) if data["probs"].size else float("nan"))
    return out


def frechet_distance(mu1, S1, mu2, S2, eps=1e-6):
    diff = mu1 - mu2
    result = sqrtm(S1.dot(S2))
    # Older SciPy returns (arr, err); newer returns arr directly.
    S_product = result[0] if isinstance(result, tuple) else result
    if not np.isfinite(S_product).all():
        S1 = S1 + np.eye(S1.shape[0]) * eps
        S2 = S2 + np.eye(S2.shape[0]) * eps
        result = sqrtm(S1.dot(S2))
        S_product = result[0] if isinstance(result, tuple) else result
    if np.iscomplexobj(S_product):
        S_product = S_product.real
    return float(diff @ diff + np.trace(S1 + S2 - 2 * S_product))


def mmd_rbf(X, Y, sigma=None):
    """MMD² with RBF kernel, unbiased estimator."""
    m, n = X.shape[0], Y.shape[0]
    if sigma is None:
        Z = np.vstack([X, Y])
        dists_sq = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
        sigma = float(np.sqrt(np.median(dists_sq[dists_sq > 0])))
        if sigma == 0:
            sigma = 1.0

    def _k(A, B):
        d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-d2 / (2 * sigma ** 2))

    kxx = _k(X, X); np.fill_diagonal(kxx, 0.0)
    kyy = _k(Y, Y); np.fill_diagonal(kyy, 0.0)
    kxy = _k(X, Y)
    mmd2 = (kxx.sum() / (m * (m - 1)) +
            kyy.sum() / (n * (n - 1)) -
            2 * kxy.sum() / (m * n))
    return float(max(mmd2, 0.0))


def write_matrix_csv(matrix, names, path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, row in enumerate(matrix):
            w.writerow([names[i]] + [f"{v:.6g}" for v in row])
    logger.info("wrote %s", path)


def top_pairs(matrix, names, top_k=15, descending=True):
    n = len(names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((names[i], names[j], matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=descending)
    return pairs[:top_k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="gs:// prefix or local dir containing per-source .npz")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--mmd_subsample", type=int, default=100,
                    help="subsample N per source for MMD to keep O(n²) tractable")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input.startswith("gs://"):
        staging = Path(tempfile.mkdtemp(prefix="fs_distances_"))
        download_from_gcs(args.input, staging)
        data_dir = staging
    else:
        data_dir = Path(args.input).resolve()

    logger.info("Loading sources from %s", data_dir)
    sources = load_sources(data_dir)
    names = sorted(sources.keys())
    n = len(names)

    logger.info("Computing per-source moments …")
    mus, sigmas, samples_for_mmd, probs = {}, {}, {}, {}
    rng = np.random.default_rng(737)
    for name in names:
        F = sources[name]["features"]
        if F.shape[0] < 2:
            logger.warning("%s has <2 samples — skipping", name)
            continue
        mus[name] = F.mean(axis=0)
        sigmas[name] = np.cov(F, rowvar=False)
        take = min(args.mmd_subsample, F.shape[0])
        idx = rng.choice(F.shape[0], size=take, replace=False)
        samples_for_mmd[name] = F[idx]
        probs[name] = sources[name]["probs"]

    valid_names = [n for n in names if n in mus]
    V = len(valid_names)

    # Pairwise matrices
    frechet = np.zeros((V, V))
    mmd = np.zeros((V, V))
    centroid = np.zeros((V, V))
    logger.info("Computing pairwise distances over %d valid sources …", V)
    for i, ni in enumerate(valid_names):
        for j, nj in enumerate(valid_names):
            if j < i:
                frechet[i, j] = frechet[j, i]
                mmd[i, j] = mmd[j, i]
                centroid[i, j] = centroid[j, i]
                continue
            if i == j:
                continue
            frechet[i, j] = frechet_distance(mus[ni], sigmas[ni], mus[nj], sigmas[nj])
            mmd[i, j] = mmd_rbf(samples_for_mmd[ni], samples_for_mmd[nj])
            centroid[i, j] = float(np.linalg.norm(mus[ni] - mus[nj]))
        logger.info("  row %d/%d (%s) done", i + 1, V, ni)

    write_matrix_csv(frechet, valid_names, out_dir / "distance_matrix_frechet.csv")
    write_matrix_csv(mmd, valid_names, out_dir / "distance_matrix_mmd_rbf.csv")
    write_matrix_csv(centroid, valid_names, out_dir / "distance_matrix_centroid.csv")

    # Per-source summary
    with open(out_dir / "per_source_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "n", "label", "provenance",
                    "prob_mean", "prob_med", "prob_std",
                    "prob_fake_rate_tau05", "feat_norm_mean"])
        for name in valid_names:
            F = sources[name]["features"]
            P = probs[name]
            w.writerow([
                name,
                F.shape[0],
                sources[name]["label"],
                sources[name]["provenance"],
                f"{P.mean():.4f}",
                f"{np.median(P):.4f}",
                f"{P.std():.4f}",
                f"{(P > 0.5).mean():.4f}",
                f"{np.linalg.norm(F, axis=1).mean():.3f}",
            ])
    logger.info("wrote %s", out_dir / "per_source_summary.csv")

    # REPORT.md
    feat_dim = next(iter(mus.values())).shape[0] if mus else 0
    report = []
    report.append("# Feature-space distance analysis — detector perspective")
    report.append("")
    report.append(f"Sources: {V}. Pairwise distances over {feat_dim}-dim backbone features.")
    report.append("")
    report.append("## Per-source prob_fake (what the detector thinks)")
    report.append("")
    report.append("| source | n | label | prob mean | prob med | prob>0.5 rate |")
    report.append("|---|---:|---|---:|---:|---:|")
    for name in valid_names:
        P = probs[name]
        report.append(
            f"| `{name}` | {sources[name]['features'].shape[0]} | {sources[name]['label']} | "
            f"{P.mean():.3f} | {np.median(P):.3f} | {(P > 0.5).mean():.3f} |"
        )
    report.append("")
    report.append("## Top-15 largest Fréchet gaps (bigger = more distributionally distant in feature space)")
    report.append("")
    report.append("| pair | Fréchet |")
    report.append("|---|---:|")
    for a, b, v in top_pairs(frechet, valid_names, 15):
        report.append(f"| `{a}` ↔ `{b}` | {v:.2f} |")
    report.append("")
    report.append("## Top-15 largest MMD² (RBF) gaps")
    report.append("")
    report.append("| pair | MMD² |")
    report.append("|---|---:|")
    for a, b, v in top_pairs(mmd, valid_names, 15):
        report.append(f"| `{a}` ↔ `{b}` | {v:.4f} |")
    report.append("")
    report.append("## Top-15 largest centroid Euclidean gaps")
    report.append("")
    report.append("| pair | ‖μ_a − μ_b‖ |")
    report.append("|---|---:|")
    for a, b, v in top_pairs(centroid, valid_names, 15):
        report.append(f"| `{a}` ↔ `{b}` | {v:.3f} |")
    report.append("")
    report.append("## Training-vs-OOD gate drivers")
    report.append("")
    report.append("The OOD gate grades `worst_pool_fpr` against the fake pools listed under "
                  "`ood_monitoring.external_fake_sources` (currently `wma_failure_fake` + `teams_ood_fake`) "
                  "and the real pools under `external_real_sources` (incl. `external_vcd_real`, "
                  "`external_youtube_avspeech_real`, `teams_ood_real`). The relevant comparisons "
                  "for generalization are therefore between the **training-side** fake/real sources and these gate pools.")
    report.append("")
    report.append("### Gate-fake pools vs training-fake sources (Fréchet)")
    report.append("")
    training_fakes = [n for n in valid_names if "fake" in n and n not in (
        "wma_failure_fake", "visomaster_enhanced_v2_fake",
    ) and "ood" not in n and "enhanced_clean" not in n]
    gate_fakes = [n for n in valid_names if n in ("wma_failure_fake",)]
    if training_fakes and gate_fakes:
        report.append("| gate pool | closest training fake | Fréchet | farthest training fake | Fréchet |")
        report.append("|---|---|---:|---|---:|")
        for gp in gate_fakes:
            gi = valid_names.index(gp)
            dists = [(tf, float(frechet[gi, valid_names.index(tf)])) for tf in training_fakes]
            dists.sort(key=lambda x: x[1])
            closest = dists[0]
            farthest = dists[-1]
            report.append(f"| `{gp}` | `{closest[0]}` | {closest[1]:.2f} | "
                          f"`{farthest[0]}` | {farthest[1]:.2f} |")
    report.append("")
    report.append("## Files")
    report.append("- `distance_matrix_frechet.csv`")
    report.append("- `distance_matrix_mmd_rbf.csv`")
    report.append("- `distance_matrix_centroid.csv`")
    report.append("- `per_source_summary.csv`")
    (out_dir / "REPORT.md").write_text("\n".join(report))
    logger.info("wrote %s", out_dir / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
