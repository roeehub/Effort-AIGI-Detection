"""Pose / face-detection audit on the 550 cached viso frames.

Uses MediaPipe FaceMesh to extract 478 landmarks per frame, then computes:
  - Detection success rate (binary: did MediaPipe find a face?)
  - Bounding box area in pixels (relative to image)
  - Eye-distance proxy in pixels
  - Yaw proxy: x-distance of nose tip from eye-line midpoint
  - Pitch proxy: y-distance of nose tip from eye-line center
  - Roll proxy: angle of eye line (atan2)
  - Mouth open ratio (mouth aspect ratio)

For each metric, compares 144 both-missed pairs vs the 131 sometimes-caught
pairs (P8A categorisation at tau=0.5).

Output:
  outputs/pose_attributes.csv
  outputs/figures/pose_attr_dist.png
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FRAMES = OUT / "viso_full_paired"

PATTERN = re.compile(r"visomaster_enhanced_(raw|teams)__frame_(\d+)_(seq\d+)\.png")

# MediaPipe FaceMesh canonical landmark indices
LM_NOSE_TIP = 1
LM_LEFT_EYE_INNER = 133
LM_LEFT_EYE_OUTER = 33
LM_RIGHT_EYE_INNER = 362
LM_RIGHT_EYE_OUTER = 263
LM_LEFT_EYE_CENTER = 468  # iris (refine_landmarks)
LM_RIGHT_EYE_CENTER = 473
LM_MOUTH_TOP = 13
LM_MOUTH_BOTTOM = 14
LM_MOUTH_LEFT = 78
LM_MOUTH_RIGHT = 308


def detect_landmarks(arr: np.ndarray, mesh):
    res = mesh.process(arr)
    if not res.multi_face_landmarks:
        return None
    lms = res.multi_face_landmarks[0].landmark
    h, w = arr.shape[:2]
    coords = np.array([[lm.x * w, lm.y * h, lm.z] for lm in lms])
    return coords


def pose_metrics(arr: np.ndarray, coords) -> Dict[str, float]:
    if coords is None:
        return {"detected": 0}
    h, w = arr.shape[:2]
    le = coords[LM_LEFT_EYE_OUTER]
    re_ = coords[LM_RIGHT_EYE_OUTER]
    eye_mid = (le + re_) / 2
    nose = coords[LM_NOSE_TIP]
    eye_dist = float(np.linalg.norm(le[:2] - re_[:2]))

    # Roll: angle of eye line
    delta = re_ - le
    roll_deg = float(np.degrees(np.arctan2(delta[1], delta[0])))

    # Yaw proxy: nose-x relative to eye midpoint, normalised by eye dist
    yaw_proxy = float((nose[0] - eye_mid[0]) / max(eye_dist, 1e-6))

    # Pitch proxy: nose-y relative to eye midpoint
    pitch_proxy = float((nose[1] - eye_mid[1]) / max(eye_dist, 1e-6))

    # Face bbox from all landmarks
    xmin, ymin = coords[:, 0].min(), coords[:, 1].min()
    xmax, ymax = coords[:, 0].max(), coords[:, 1].max()
    bbox_w = xmax - xmin
    bbox_h = ymax - ymin
    bbox_area = bbox_w * bbox_h
    img_area = w * h

    # Mouth opening
    mt = coords[LM_MOUTH_TOP]
    mb = coords[LM_MOUTH_BOTTOM]
    ml = coords[LM_MOUTH_LEFT]
    mr = coords[LM_MOUTH_RIGHT]
    mouth_open = float(np.linalg.norm(mt[:2] - mb[:2]))
    mouth_width = float(np.linalg.norm(ml[:2] - mr[:2]))
    mouth_ar = mouth_open / max(mouth_width, 1e-6)

    return {
        "detected": 1,
        "img_h": h, "img_w": w,
        "eye_dist_px": eye_dist,
        "eye_dist_rel": eye_dist / max(w, 1),
        "yaw_proxy": yaw_proxy,
        "pitch_proxy": pitch_proxy,
        "roll_deg": roll_deg,
        "bbox_w": float(bbox_w),
        "bbox_h": float(bbox_h),
        "bbox_area_px": float(bbox_area),
        "bbox_area_frac": float(bbox_area / img_area),
        "mouth_aspect_ratio": mouth_ar,
    }


def main():
    import mediapipe as mp
    mesh_solution = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.3,
    )

    files = sorted(FRAMES.glob("visomaster_enhanced_*.png"))
    print(f"[load] {len(files)} cached viso frames")
    rows = []
    n_detected = 0
    for i, p in enumerate(files):
        m = PATTERN.search(p.name)
        if not m:
            continue
        subtype, frame_num, seq_id = m.groups()
        arr = np.asarray(Image.open(p).convert("RGB"))
        coords = detect_landmarks(arr, mesh_solution)
        metrics = pose_metrics(arr, coords)
        metrics.update({"seq_id": seq_id, "subtype": subtype, "filename": p.name})
        rows.append(metrics)
        n_detected += metrics["detected"]
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(files)}] detection rate so far: {n_detected/(i+1):.3f}")
    print(f"[done] face detection: {n_detected}/{len(rows)} = {n_detected/len(rows):.3f}")
    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(OUT / "pose_attributes.csv", index=False)

    # Pivot by seq_id (raw vs teams)
    detected_only = df[df["detected"] == 1].copy()
    pose_pivot = detected_only.pivot_table(
        index="seq_id", columns="subtype",
        values=["eye_dist_rel", "yaw_proxy", "pitch_proxy", "roll_deg", "bbox_area_frac", "mouth_aspect_ratio"],
        aggfunc="first"
    ).reset_index()
    pose_pivot.columns = [f"{a}_{b}" if b else a for a, b in pose_pivot.columns]

    # Per-pair detection rate (both detected? only one?)
    both_det = detected_only.groupby("seq_id").size().reset_index(name="n_subtypes_detected")
    both_det["both_detected"] = both_det["n_subtypes_detected"] == 2
    print(f"[pairs] both substrates detected: {both_det['both_detected'].sum()}/{len(both_det)} pairs")

    # Categorise by P8A pair categorisation
    pairs = pd.read_csv(OUT / "viso_pairs.csv")
    p8a_pairs = pairs[pairs["model"] == "P8A"].copy()
    rh = p8a_pairs["frame_prob_raw"] >= 0.5
    th = p8a_pairs["frame_prob_teams"] >= 0.5
    p8a_pairs["category"] = np.where(rh & th, "both_caught",
                            np.where(rh & ~th, "raw_only",
                            np.where(~rh & th, "teams_only", "both_missed")))

    merged = p8a_pairs[["seq_id", "category"]].merge(pose_pivot, on="seq_id", how="left")
    merged = merged.merge(both_det[["seq_id", "both_detected"]], on="seq_id", how="left")

    # Cross-model never-caught (108 pairs from the prior analysis)
    pivot = pairs.pivot_table(index="seq_id", columns="model",
                              values=["frame_prob_raw", "frame_prob_teams"], aggfunc="first")
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot["never_caught"] = (pivot < 0.5).all(axis=1)
    pivot = pivot.reset_index()
    merged = merged.merge(pivot[["seq_id", "never_caught"]], on="seq_id", how="left")

    # ── Comparison: never_caught vs ever_caught
    print("\n=== Pose attribute comparison: never_caught vs ever_caught ===")
    pose_cols = [c for c in merged.columns if c.endswith("_raw") or c.endswith("_teams")]
    pose_cols = [c for c in pose_cols if c not in ("category_raw",)]
    summary_rows = []
    for col in pose_cols:
        nc = merged[merged["never_caught"] == True][col].dropna()
        ec = merged[merged["never_caught"] == False][col].dropna()
        if len(nc) < 5 or len(ec) < 5:
            continue
        try:
            u, p = mannwhitneyu(nc, ec, alternative="two-sided")
        except Exception:
            u, p = float("nan"), float("nan")
        summary_rows.append({
            "feature": col,
            "never_mean": float(nc.mean()),
            "ever_mean": float(ec.mean()),
            "delta_mean": float(nc.mean() - ec.mean()),
            "never_n": int(len(nc)),
            "ever_n": int(len(ec)),
            "MWU_p": float(p),
            "significant_5pct": bool(p < 0.05),
        })
    pose_sig = pd.DataFrame(summary_rows).sort_values("MWU_p")
    pose_sig.to_csv(OUT / "pose_significance.csv", index=False)
    print(pose_sig.to_string(index=False, float_format="%.4f"))

    # Detection rate by category
    print("\n=== Face detection rate per pair category ===")
    det_summary = merged.groupby("category")["both_detected"].agg(["count", "sum", "mean"])
    print(det_summary)

    # ── Plot distributions
    plot_features = [c for c in pose_cols if "raw" in c][:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, col in zip(axes.flat, plot_features):
        nc = merged[merged["never_caught"] == True][col].dropna()
        ec = merged[merged["never_caught"] == False][col].dropna()
        if nc.empty or ec.empty:
            ax.axis("off"); continue
        all_v = pd.concat([nc, ec])
        bins = np.linspace(all_v.quantile(0.02), all_v.quantile(0.98), 25)
        ax.hist(ec, bins=bins, alpha=0.55, color="green", label=f"ever_caught (n={len(ec)})", edgecolor="black")
        ax.hist(nc, bins=bins, alpha=0.55, color="red", label=f"never_caught (n={len(nc)})", edgecolor="black")
        try:
            u, p = mannwhitneyu(nc, ec, alternative="two-sided")
            sig = " *" if p < 0.05 else ""
            ax.set_title(f"{col}\np={p:.4f}{sig}", fontsize=10)
        except Exception:
            ax.set_title(col, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Pose attribute distribution — never_caught vs ever_caught", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "pose_attr_dist.png", dpi=110)
    plt.close(fig)

    print(f"\n[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
