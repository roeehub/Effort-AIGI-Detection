"""Identity-corruption audit (2026-04-27 evening).

Runs ArcFace on cached lockbox frames + a downloaded PC_Generator__s13 sample,
clusters within each labeled identity_key, and reports how many *labeled*
identities actually contain multiple distinct people.

This is the load-bearing diagnostic for the Plan-v3 Day-1 decision gate:
- If lockbox identity labels are corrupted, the parallel agent's per-identity
  findings (§1.5) are unreliable and the modern-subset definition needs to be
  cluster-based, not label-based.
- If training-set construction shares the same provenance, identity-balanced
  sampling and contrastive loss in P11_TARGETED are at risk.

Outputs:
  analysis/identity_corruption_audit_2026-04-27/identity_audit.parquet
  analysis/identity_corruption_audit_2026-04-27/per_label_summary.json
  analysis/identity_corruption_audit_2026-04-27/report.txt

Run from training/:
  python3 -m analysis.identity_corruption_audit_2026-04-27 [--include-s13-dev]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
PARQUET_IN = REPO / "analysis/lockbox_tagging/lockbox_tags_2026-04-27.parquet"
OUT_DIR = REPO / "analysis/identity_corruption_audit_2026-04-27"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Match the parallel agent's compute_identity API.
from analysis.lockbox_tagging.layers.identity import compute_identity


def embed_frame(local_path: Path, bbox_xywh: tuple[float, float, float, float] | None) -> np.ndarray | None:
    out = compute_identity(local_path, face_bbox=bbox_xywh)
    emb = out.get("arcface_embed")
    if emb is None or len(emb) != 512:
        return None
    return np.asarray(emb, dtype=np.float32)


def cluster_within_label(embs: np.ndarray, sim_threshold: float = 0.45) -> tuple[np.ndarray, list[list[int]]]:
    """Greedy single-link clustering by cosine similarity.

    sim_threshold=0.45 is a deliberately *permissive* setting: ArcFace tends
    to give same-identity pairs >0.6 in good lighting, and different-identity
    pairs <0.3 in most cases. 0.45 keeps the audit conservative — a label is
    only flagged as multi-identity if pair sims dip clearly below same-person
    range.
    """
    n = len(embs)
    sims = embs @ embs.T
    # Greedy: assign each frame to the lowest-index existing cluster it's similar to;
    # otherwise open a new cluster.
    cluster_of = -np.ones(n, dtype=np.int64)
    next_cluster = 0
    for i in range(n):
        best = -1
        for j in range(i):
            if cluster_of[j] >= 0 and sims[i, j] >= sim_threshold:
                if best < 0 or cluster_of[j] < best:
                    best = cluster_of[j]
        if best < 0:
            cluster_of[i] = next_cluster
            next_cluster += 1
        else:
            cluster_of[i] = best
    # Collect cluster members.
    members: list[list[int]] = [[] for _ in range(next_cluster)]
    for i, c in enumerate(cluster_of):
        members[int(c)].append(i)
    return cluster_of, members


def audit_label(label_key: str, df_label: pd.DataFrame, sim_threshold: float = 0.45) -> dict:
    """Embed all frames for one label, cluster, return per-label statistics."""
    embeds: list[np.ndarray] = []
    rows_kept: list[int] = []
    for idx, r in df_label.iterrows():
        local = Path(r["local_path"]) if r.get("local_path") else None
        if local is None or not local.exists():
            continue
        bbox = None
        if pd.notna(r.get("face_bbox_x")):
            bbox = (
                float(r["face_bbox_x"]),
                float(r["face_bbox_y"]),
                float(r["face_bbox_w"]),
                float(r["face_bbox_h"]),
            )
        emb = embed_frame(local, bbox)
        if emb is None:
            continue
        n = float(np.linalg.norm(emb))
        if n <= 0:
            continue
        embeds.append(emb / n)  # unit-normalize for cosine
        rows_kept.append(int(idx))
    if len(embeds) < 2:
        return {
            "identity_key": label_key,
            "n_frames_total": int(len(df_label)),
            "n_frames_embedded": len(embeds),
            "n_clusters": len(embeds),
            "largest_cluster_share": 1.0 if embeds else 0.0,
            "cluster_sizes": [len(embeds)] if embeds else [],
            "is_multi_identity": False,
        }
    embs = np.stack(embeds, axis=0)
    cluster_of, members = cluster_within_label(embs, sim_threshold=sim_threshold)
    sizes = sorted([len(m) for m in members], reverse=True)
    largest_share = sizes[0] / sum(sizes) if sizes else 0.0
    is_multi = len(sizes) >= 2 and sizes[1] >= max(3, int(0.10 * sum(sizes)))  # second cluster of meaningful size
    return {
        "identity_key": label_key,
        "n_frames_total": int(len(df_label)),
        "n_frames_embedded": int(len(embeds)),
        "n_clusters": int(len(sizes)),
        "largest_cluster_share": float(largest_share),
        "cluster_sizes": sizes,
        "is_multi_identity": bool(is_multi),
        "_rows_kept": rows_kept,
        "_cluster_of": [int(c) for c in cluster_of],
    }, embs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-threshold", type=float, default=0.45)
    ap.add_argument("--include-s13-dev", action="store_true",
                    help="Also download a sample of PC_Generator__s13 dev frames and audit")
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET_IN)
    print(f"[audit] loaded {len(df)} rows; {df['identity_key'].nunique()} distinct identity_key values")
    print(f"[audit] sim_threshold = {args.sim_threshold}")

    summaries = []
    enriched_rows = []
    for ident, sub in df.groupby("identity_key"):
        print(f"[audit] processing {ident} (n={len(sub)}) ...")
        out = audit_label(ident, sub, sim_threshold=args.sim_threshold)
        if isinstance(out, tuple):
            summary, embs = out
            # store summary minus internal keys
            rows_kept = summary.pop("_rows_kept")
            cluster_of = summary.pop("_cluster_of")
            for ridx, c in zip(rows_kept, cluster_of):
                enriched_rows.append({
                    "row_index": ridx,
                    "identity_key": ident,
                    "arcface_cluster_id": c,
                })
        else:
            summary = out
        summaries.append(summary)
        print(f"        n_clusters={summary['n_clusters']}, "
              f"largest_share={summary['largest_cluster_share']:.2f}, "
              f"sizes_top5={summary['cluster_sizes'][:5]}, "
              f"multi_identity={summary['is_multi_identity']}")

    # Save outputs.
    summaries_df = pd.DataFrame(summaries)
    enriched_df = pd.DataFrame(enriched_rows)
    summaries_df.to_csv(OUT_DIR / "per_label_summary.csv", index=False)
    summaries_df.to_json(OUT_DIR / "per_label_summary.json", orient="records", indent=2)
    enriched_df.to_parquet(OUT_DIR / "frame_clusters.parquet", index=False)

    # Compose verdict.
    n_multi = sum(1 for s in summaries if s.get("is_multi_identity"))
    total_frames = int(summaries_df["n_frames_embedded"].sum())
    misgrouped_frames = 0
    for s in summaries:
        if s.get("is_multi_identity"):
            sizes = s["cluster_sizes"]
            misgrouped_frames += sum(sizes[1:])
    verdict = {
        "n_labels_total": int(len(summaries)),
        "n_labels_multi_identity": int(n_multi),
        "frames_embedded_total": total_frames,
        "frames_in_non_dominant_clusters": int(misgrouped_frames),
        "frame_corruption_share": float(misgrouped_frames / max(total_frames, 1)),
        "sim_threshold": args.sim_threshold,
    }
    with open(OUT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    report_lines = [
        "=" * 72,
        f"Identity-Corruption Audit  ({PARQUET_IN.name})",
        f"sim_threshold={args.sim_threshold}",
        "=" * 72,
        "",
        "Per-label summary:",
        summaries_df[["identity_key", "n_frames_embedded", "n_clusters",
                      "largest_cluster_share", "is_multi_identity"]].to_string(index=False),
        "",
        "Verdict:",
        json.dumps(verdict, indent=2),
    ]
    report = "\n".join(report_lines)
    (OUT_DIR / "report.txt").write_text(report)
    print()
    print(report)


if __name__ == "__main__":
    main()
