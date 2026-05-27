"""Audit PC_Generator__s13 (dev split, all-REAL) — the user's reported case.

User showed two visibly different people both labeled `PC_Generator__s13`.
We download up to N frames, run ArcFace, and cluster. If there are >=2
non-trivial clusters, we have hard evidence of within-real-label corruption,
which propagates to dev-split FPR claims and possibly to training data.

Run from training/:
  python3 -m analysis.identity_corruption_audit_2026-04-27.audit_s13_dev [--n 30]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = REPO / "analysis/identity_corruption_audit_2026-04-27/s13_dev"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from analysis.lockbox_tagging.io_utils import load_manifest_index, load_predictions, fetch_frame, FrameRow
from analysis.lockbox_tagging.layers.identity import compute_identity
from analysis.lockbox_tagging.layers.face_geometry import compute_face_geometry


def gather_s13_frames(n: int = 30, seed: int = 0) -> list[FrameRow]:
    idx = load_manifest_index()
    preds = load_predictions()
    rows: list[FrameRow] = []
    for u, m in idx.items():
        if m.get("identity_key") == "PC_Generator__s13":
            p = preds.get(u, {})
            bucket, blob = u[len("gs://"):].split("/", 1)
            rows.append(FrameRow(
                gcs_uri=u,
                blob_path=p.get("blob_path") or blob,
                bucket=p.get("bucket") or bucket,
                label=m.get("label") or "real",
                split=m.get("split", "dev"),
                identity_key="PC_Generator__s13",
                session_id=m.get("session_id"),
                video_id=m.get("video_id") or "",
                method=m.get("method") or "real",
                prob_fake=p.get("prob_fake"),
            ))
    import random
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--sim-threshold", type=float, default=0.45)
    args = ap.parse_args()

    frames = gather_s13_frames(n=args.n)
    print(f"[s13-audit] sampled {len(frames)} PC_Generator__s13 dev frames")

    embeds: list[np.ndarray] = []
    used: list[FrameRow] = []
    for fr in frames:
        try:
            local = fetch_frame(fr)
        except Exception as e:
            print(f"[s13-audit] fetch failed {fr.blob_path}: {e}")
            continue
        # Use face geometry for bbox (avoids feeding the whole image, which often
        # contains background and harms ArcFace embedding stability).
        geom = compute_face_geometry(local)
        bbox = None
        if geom.get("face_bbox_x") is not None:
            bbox = (geom["face_bbox_x"], geom["face_bbox_y"], geom["face_bbox_w"], geom["face_bbox_h"])
        out = compute_identity(local, face_bbox=bbox)
        emb = out.get("arcface_embed")
        if emb is None or len(emb) != 512:
            continue
        e = np.asarray(emb, dtype=np.float32)
        nrm = float(np.linalg.norm(e))
        if nrm <= 0:
            continue
        embeds.append(e / nrm)
        used.append(fr)
    print(f"[s13-audit] embedded {len(embeds)} / {len(frames)} frames")

    if len(embeds) < 2:
        print("[s13-audit] insufficient embeddings — abort")
        return

    embs = np.stack(embeds, axis=0)
    sims = embs @ embs.T

    # Greedy single-link clustering identical to the main audit.
    n = len(embs)
    cluster_of = -np.ones(n, dtype=np.int64)
    next_cluster = 0
    for i in range(n):
        best = -1
        for j in range(i):
            if cluster_of[j] >= 0 and sims[i, j] >= args.sim_threshold:
                if best < 0 or cluster_of[j] < best:
                    best = cluster_of[j]
        if best < 0:
            cluster_of[i] = next_cluster
            next_cluster += 1
        else:
            cluster_of[i] = best

    sizes_per_cluster: dict[int, int] = {}
    for c in cluster_of:
        sizes_per_cluster[int(c)] = sizes_per_cluster.get(int(c), 0) + 1
    sizes = sorted(sizes_per_cluster.values(), reverse=True)
    print(f"[s13-audit] cluster sizes: {sizes}")

    # Save evidence per cluster.
    rows = []
    for fr, c in zip(used, cluster_of):
        rows.append({
            "gcs_uri": fr.gcs_uri,
            "blob_path": fr.blob_path,
            "video_id": fr.video_id,
            "label": fr.label,
            "prob_fake": fr.prob_fake,
            "arcface_cluster_id": int(c),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "s13_clusters.csv", index=False)

    # Also: pairwise stats — distinguishes "noise" splits from real multi-person splits.
    intra = []  # within-cluster sims
    inter = []  # between-cluster sims
    for i in range(n):
        for j in range(i + 1, n):
            if cluster_of[i] == cluster_of[j]:
                intra.append(float(sims[i, j]))
            else:
                inter.append(float(sims[i, j]))
    summary = {
        "n_frames_embedded": int(n),
        "sim_threshold": args.sim_threshold,
        "n_clusters": len(sizes),
        "cluster_sizes": sizes,
        "intra_cluster_sim_mean": float(np.mean(intra)) if intra else None,
        "intra_cluster_sim_p10": float(np.quantile(intra, 0.1)) if intra else None,
        "inter_cluster_sim_mean": float(np.mean(inter)) if inter else None,
        "inter_cluster_sim_p90": float(np.quantile(inter, 0.9)) if inter else None,
        "inter_cluster_sim_max": float(np.max(inter)) if inter else None,
        "verdict_multi_identity": len(sizes) >= 2 and sizes[1] >= max(3, int(0.10 * n)),
    }
    with open(OUT_DIR / "s13_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
