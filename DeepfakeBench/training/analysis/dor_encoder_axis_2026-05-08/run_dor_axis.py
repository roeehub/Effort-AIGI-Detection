"""Dor encoder-axis characterization (check (c)).

Purpose: understand WHY P2-D-step3000 lost P8A's signature dor invariance
(J4: real_FPR 8% → 46%, fake_recall 86% → 62% on Dor cohorts).

Disambiguates between:
  - INTERPRETATION 1 (CONFUSED): Dor reals + Dor fakes in a confused region;
    centroids close together, margin small.
  - INTERPRETATION 2 (SHIFTED): Dor reals shifted toward fake centroid; an
    axis-specific failure mode, possibly correlated with non-IQ feature.

Method:
  1. For each ckpt, normalize CLS features per row (cosine geometry).
  2. Compute centroids:
       C_dor_real      = mean(features over DOR_REAL_DEV ∪ DOR_REAL_LOCKBOX)
       C_dor_fake      = mean(features over DOR_FAKE_DEV)
       C_non_dor_real  = mean(features over NON_DOR_REAL_DEV)
       C_non_dor_fake  = mean(features over NON_DOR_FAKE_DEV)
  3. Per-pair centroid distances (cosine + L2):
       dist(C_dor_real, C_non_dor_real)  -- "Dor real position vs non-Dor real"
       dist(C_dor_fake, C_non_dor_fake)  -- "Dor fake position vs non-Dor fake"
       dist(C_dor_real, C_dor_fake)      -- "Dor real-fake separation"
       dist(C_non_dor_real, C_non_dor_fake) -- "Non-Dor real-fake separation"
  4. Per-frame margin: cos-distance to own-class centroid minus cos-distance
     to opposite-class centroid (positive ⇒ closer to own class).
       margin_dor_real_frame = d(f, C_real) - d(f, C_fake)
       on Dor real cohort: how confidently positioned vs fake centroid?
  5. Per-frame distance from each Dor real frame to fake centroid: smaller
     distance ⇒ closer-to-fake-centroid ⇒ INTERPRETATION-2 signature.
  6. Per-class score statistics: per-cohort, per-ckpt p10/p50/p90.

Distinguishing signatures (heuristic):
  - INTERPRETATION 1 is supported if BOTH:
      dist(C_dor_real, C_dor_fake) << dist(C_non_dor_real, C_non_dor_fake)
      AND margin variance shrinks on Dor.
  - INTERPRETATION 2 is supported if:
      C_dor_real has shifted toward C_fake without symmetric C_dor_fake shift,
      i.e. dist(C_dor_real, C_non_dor_fake) < dist(C_non_dor_real, C_non_dor_fake)
      while dist(C_dor_fake, C_non_dor_fake) ~= dist(C_non_dor_fake, ...).

  Both together (1 ∧ 2) are also possible.

  Per-frame scoring concentration: if Dor reals' p90 score increases on
  P2D vs P8A (more frames pushed toward fake decision boundary) while
  Dor fakes' p10 score decreases (fakes pulled away from fake decision),
  INTERPRETATION 2 is bidirectional — encoder uses "Dor identity" as a
  fake-cue rather than maintaining identity-invariant separation.

Outputs:
  outputs/dor_axis_FACTS.csv — per-ckpt centroid & margin tables
  outputs/dor_axis_summary.json — summary numbers
  outputs/dor_axis_per_frame.csv — per-frame margins/distances
  outputs/iq_correlation_dor.csv — per-feature correlation with score on Dor
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

THIS_DIR = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = THIS_DIR / "_cache"
OUTPUTS = THIS_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

ATLAS_PARQUET = (
    REPO_ROOT / "analysis" / "iq_data_atlas_2026-05-08" / "outputs" / "per_frame.parquet"
)

CKPT_LABELS = ["P8A", "E2B", "P2D"]
SCORE_COL = {
    "P8A": "p8a_reference_step5000",
    "E2B": "e2b_top_n_step3200",
    "P2D": "p2_d_fourier_periodic_step3000",
}

# Cohort-class assignments (label-axis-agnostic).
# Real cohorts: DOR_REAL_DEV, DOR_REAL_LOCKBOX, NON_DOR_REAL_DEV (label 0)
# Fake cohorts: DOR_FAKE_DEV, NON_DOR_FAKE_DEV (label 1)
PRIMARY_6 = ["lap_var", "min_dim", "luma_mean", "color_b_dev", "edge_mag", "skin_frac"]

logger = logging.getLogger("dor-axis")


def cosine_dist(A: np.ndarray, B: np.ndarray) -> float:
    """Cosine distance (1 - cos similarity) between two vectors."""
    a = A / (np.linalg.norm(A) + 1e-12)
    b = B / (np.linalg.norm(B) + 1e-12)
    return float(1.0 - np.dot(a, b))


def l2_dist(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B))


def per_frame_distance_to_centroid(F: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Per-frame cosine distance to centroid C."""
    F_n = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    c_n = C / (np.linalg.norm(C) + 1e-12)
    cs = F_n @ c_n
    return 1.0 - cs


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

    cohort = pd.read_csv(LOCAL_CACHE_DIR / "cohort_manifest.csv")
    cohort["row_ix"] = np.arange(len(cohort))
    # Reconstruct local_path from frame_path basename (matches build_cohort_features.py mapping).
    if "local_path" not in cohort.columns:
        cohort["local_path"] = cohort["frame_path"].map(
            lambda u: str(LOCAL_CACHE_DIR / "frames" / u.split("/")[-1])
        )
    logger.info("cohort:\n%s", cohort.groupby(["cohort", "label"]).size().to_string())

    feats: dict[str, dict[str, np.ndarray]] = {}
    for label in CKPT_LABELS:
        blob = np.load(LOCAL_CACHE_DIR / f"cohort_features__{label}.npz")
        # Some frames may have been invalid; valid_mask is True for valid input frames
        # but the saved finals_arr only contains valid ones; we re-align.
        valid_mask = blob["valid_mask"].astype(bool)
        finals = blob["final_cls"]
        layer11 = blob["layer11_cls"]
        # Validate length
        if finals.shape[0] != int(valid_mask.sum()):
            logger.warning("[%s] finals shape %d != valid count %d",
                           label, finals.shape[0], int(valid_mask.sum()))
        feats[label] = {
            "valid_mask": valid_mask,
            "final": finals,
            "layer11": layer11,
        }

    # Use the valid_mask from the first ckpt; assume identical across ckpts
    # since extraction is deterministic given the same image-decode pipeline.
    primary_mask = feats[CKPT_LABELS[0]]["valid_mask"]
    # Assert that all ckpts have the same mask
    for label in CKPT_LABELS[1:]:
        if not np.array_equal(feats[label]["valid_mask"], primary_mask):
            logger.warning("[%s] valid_mask differs from %s", label, CKPT_LABELS[0])
    cohort_valid = cohort[primary_mask].reset_index(drop=True)
    logger.info("kept %d valid frames after image-decode mask", len(cohort_valid))

    # Compute centroids per (ckpt, cohort, feature_layer).
    centroid_rows = []
    pair_rows = []
    per_frame_rows = []

    for label in CKPT_LABELS:
        for layer_name in ["layer11", "final"]:
            F = feats[label][layer_name]  # (n_valid, dim)
            # Normalize per-row for cosine geometry
            F_n = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)

            cohorts_unique = cohort_valid["cohort"].unique()
            centroids = {}
            for c in cohorts_unique:
                mask = (cohort_valid["cohort"] == c).values
                if mask.sum() == 0:
                    continue
                centroids[c] = F[mask].mean(axis=0)
                centroid_rows.append({
                    "ckpt": label,
                    "feature_layer": layer_name,
                    "cohort": c,
                    "n_frames": int(mask.sum()),
                    "centroid_norm": float(np.linalg.norm(centroids[c])),
                    "centroid_dim": F.shape[1],
                })

            # Combined Dor-real (DOR_REAL_DEV + DOR_REAL_LOCKBOX)
            dor_real_mask = cohort_valid["cohort"].isin(["DOR_REAL_DEV", "DOR_REAL_LOCKBOX"]).values
            non_dor_real_mask = (cohort_valid["cohort"] == "NON_DOR_REAL_DEV").values
            dor_fake_mask = (cohort_valid["cohort"] == "DOR_FAKE_DEV").values
            non_dor_fake_mask = (cohort_valid["cohort"] == "NON_DOR_FAKE_DEV").values

            C_dor_real = F[dor_real_mask].mean(axis=0)
            C_non_dor_real = F[non_dor_real_mask].mean(axis=0)
            C_dor_fake = F[dor_fake_mask].mean(axis=0)
            C_non_dor_fake = F[non_dor_fake_mask].mean(axis=0)

            pairs = [
                ("dor_real_vs_non_dor_real", C_dor_real, C_non_dor_real),
                ("dor_fake_vs_non_dor_fake", C_dor_fake, C_non_dor_fake),
                ("dor_real_vs_dor_fake", C_dor_real, C_dor_fake),
                ("non_dor_real_vs_non_dor_fake", C_non_dor_real, C_non_dor_fake),
                ("dor_real_vs_non_dor_fake", C_dor_real, C_non_dor_fake),
                ("non_dor_real_vs_dor_fake", C_non_dor_real, C_dor_fake),
            ]
            for name, A, B in pairs:
                pair_rows.append({
                    "ckpt": label,
                    "feature_layer": layer_name,
                    "pair": name,
                    "cosine_dist": cosine_dist(A, B),
                    "l2_dist": l2_dist(A, B),
                })

            # Per-frame cohort-conditional margin: for each Dor real frame,
            # cos-distance to non-Dor real centroid, to Dor fake centroid,
            # to non-Dor fake centroid; margin = d(non_dor_fake) - d(non_dor_real)
            # (high margin = far from fake centroid relative to real centroid)
            for c in cohorts_unique:
                mask = (cohort_valid["cohort"] == c).values
                if mask.sum() == 0:
                    continue
                F_c = F[mask]
                d_to_non_dor_real = per_frame_distance_to_centroid(F_c, C_non_dor_real)
                d_to_dor_real = per_frame_distance_to_centroid(F_c, C_dor_real)
                d_to_non_dor_fake = per_frame_distance_to_centroid(F_c, C_non_dor_fake)
                d_to_dor_fake = per_frame_distance_to_centroid(F_c, C_dor_fake)
                cohort_rows = cohort_valid[mask]
                for i, (_, row) in enumerate(cohort_rows.iterrows()):
                    per_frame_rows.append({
                        "ckpt": label,
                        "feature_layer": layer_name,
                        "cohort": c,
                        "frame_path": row["frame_path"],
                        "label": row["label"],
                        "score": row[SCORE_COL[label]],
                        "d_to_non_dor_real": d_to_non_dor_real[i],
                        "d_to_dor_real": d_to_dor_real[i],
                        "d_to_non_dor_fake": d_to_non_dor_fake[i],
                        "d_to_dor_fake": d_to_dor_fake[i],
                    })

    # Save core tables
    centroid_df = pd.DataFrame(centroid_rows)
    pair_df = pd.DataFrame(pair_rows)
    per_frame_df = pd.DataFrame(per_frame_rows)

    centroid_df.to_csv(OUTPUTS / "dor_axis_centroids.csv", index=False)
    pair_df.to_csv(OUTPUTS / "dor_axis_pair_distances.csv", index=False)
    per_frame_df.to_csv(OUTPUTS / "dor_axis_per_frame.csv", index=False)

    # Per-cohort score statistics
    score_stats = []
    for label in CKPT_LABELS:
        for c in cohort_valid["cohort"].unique():
            sub = cohort_valid[cohort_valid["cohort"] == c]
            scores = sub[SCORE_COL[label]].values
            if len(scores) == 0:
                continue
            score_stats.append({
                "ckpt": label,
                "cohort": c,
                "n": len(scores),
                "score_p10": float(np.percentile(scores, 10)),
                "score_p50": float(np.percentile(scores, 50)),
                "score_p90": float(np.percentile(scores, 90)),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
            })
    score_df = pd.DataFrame(score_stats)
    score_df.to_csv(OUTPUTS / "dor_axis_score_stats.csv", index=False)

    # IQ correlation on Dor cohort
    # Use atlas IQ panel if available; fall back to inline compute.
    atlas = pd.read_parquet(ATLAS_PARQUET)[["frame_path"] + PRIMARY_6].copy()
    # De-dup atlas: same frame_path may appear in multiple pools
    atlas = atlas.drop_duplicates(subset=["frame_path"]).reset_index(drop=True)
    iq_matched = cohort_valid.merge(atlas, on="frame_path", how="left").reset_index(drop=True)
    n_in_atlas = int(iq_matched[PRIMARY_6[0]].notna().sum())
    logger.info("Dor cohort frames with atlas IQ: %d/%d", n_in_atlas, len(iq_matched))

    # Compute inline for missing
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "iq_perlayer_compute",
        REPO_ROOT / "analysis" / "iq_perlayer_probe_2026-05-08" / "run_probe.py",
    )
    iqmod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(iqmod)  # type: ignore

    # compute inline for missing — use positional indices into iq_matched.
    miss_pos = iq_matched.index[iq_matched[PRIMARY_6[0]].isna()].tolist()
    if len(miss_pos) > 0:
        local_paths = iq_matched.loc[miss_pos, "local_path"].tolist()
        logger.info("computing IQ inline for %d Dor cohort frames", len(local_paths))
        inline = iqmod.compute_iq_panel_inline(local_paths)
        for col in PRIMARY_6:
            iq_matched.loc[miss_pos, col] = inline[col].values

    iq_corr_rows = []
    for label in CKPT_LABELS:
        for c in iq_matched["cohort"].unique():
            sub = iq_matched[iq_matched["cohort"] == c].dropna(subset=PRIMARY_6 + [SCORE_COL[label]])
            if len(sub) < 10:
                continue
            for feat in PRIMARY_6:
                # Pearson r between feat and score
                x = sub[feat].values.astype(float)
                y = sub[SCORE_COL[label]].values.astype(float)
                if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                    r = 0.0
                else:
                    r = float(np.corrcoef(x, y)[0, 1])
                iq_corr_rows.append({
                    "ckpt": label,
                    "cohort": c,
                    "iq_feature": feat,
                    "n": len(sub),
                    "pearson_r_score_iq": r,
                })
    iq_corr_df = pd.DataFrame(iq_corr_rows)
    iq_corr_df.to_csv(OUTPUTS / "iq_correlation_dor.csv", index=False)

    # Summary JSON: highlights for the FACTS doc.
    # Build summary dict with str keys only
    layer11_pivot = pair_df[pair_df["feature_layer"] == "layer11"].pivot_table(
        index="pair", columns="ckpt", values="cosine_dist"
    )
    final_pivot = pair_df[pair_df["feature_layer"] == "final"].pivot_table(
        index="pair", columns="ckpt", values="cosine_dist"
    )

    def df_to_jsonable(df: pd.DataFrame) -> dict:
        out = {}
        for idx in df.index:
            out[str(idx)] = {str(c): float(df.loc[idx, c]) for c in df.columns}
        return out

    summary = {
        "n_frames_per_cohort": {str(k): int(v) for k, v in cohort_valid["cohort"].value_counts().items()},
        "centroid_distances_layer11_cosine": df_to_jsonable(layer11_pivot),
        "centroid_distances_final_cosine": df_to_jsonable(final_pivot),
        "per_cohort_scores": (
            score_df.set_index(["ckpt", "cohort"])[
                ["n", "score_p10", "score_p50", "score_p90", "score_mean"]
            ]
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    with open(OUTPUTS / "dor_axis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Console readout
    print("\n" + "=" * 80)
    print("DOR ENCODER-AXIS  (final-CLS centroid distances; cosine)")
    print("=" * 80)
    pivot = pair_df[pair_df["feature_layer"] == "final"].pivot_table(
        index="pair", columns="ckpt", values="cosine_dist"
    )
    pd.options.display.float_format = "{:.4f}".format
    print(pivot.to_string())
    print()
    print("=" * 80)
    print("PER-COHORT SCORE STATS (p50, p90)")
    print("=" * 80)
    pivot_s = score_df.pivot_table(index="cohort", columns="ckpt", values="score_p50")
    print("p50:")
    print(pivot_s.to_string())
    print()
    pivot_p = score_df.pivot_table(index="cohort", columns="ckpt", values="score_p90")
    print("p90:")
    print(pivot_p.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
