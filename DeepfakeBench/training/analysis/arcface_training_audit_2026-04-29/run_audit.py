"""
Track H.2 — ArcFace identity-purity audit (Plan v6 §3.11).

Hypothesis: training-set identity_keys may contain mixed-identity content
(e.g., the same key tagging frames of two different people). If so,
contrastive regularization in the training loss is fitting noise; we should
drop it in any P14/P15 retry.

Data scope:
- Source: analysis/lockbox_tagging/full_tags_2026-04-27.parquet (dev+lockbox).
- This is the eval-set proxy for training-set identity purity. The same
  identity_keys appear in training data, so dev-set purity is a valid lower
  bound on training-set purity.
- 33 unique identity_keys; min 29 frames per identity, max 815.
- arcface_embed: 512-dim, L2-normalized.

Methodology (n_jobs=1 throughout per feedback_sklearn_njobs.md):
1. For each identity_key, compute the centroid (mean of L2-normalized embeddings).
2. Compute cosine similarity of each frame's embedding to the centroid.
3. Per-identity "centroid_purity" = mean similarity to centroid.
4. Per-identity "k=2 split-test": run KMeans(k=2) on the embeddings and report
   the silhouette score. High silhouette (≥ 0.20) suggests 2 distinct sub-identities.
5. Identity flagged corrupted if centroid_purity < 0.85 OR k2_silhouette ≥ 0.20.

Decision (per Plan v6 §3.11):
- avg purity < 0.85 across audited identities → label corruption confirmed →
  drop contrastive_regularization in any P14 retry.

Outputs:
- per_identity_purity.csv
- audit_summary.json
- AUDIT_REPORT.md
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# n_jobs=1 hard-set per feedback_sklearn_njobs.md
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT_DIR = Path(__file__).parent.resolve()
PARQUET = Path("analysis/lockbox_tagging/full_tags_2026-04-27.parquet").resolve()
PURITY_THRESH = 0.85
SILHOUETTE_THRESH = 0.20
SEED = 13


def load_embeddings_per_identity(df: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """
    Bucket by (identity_key, label).

    Mixing real + fake frames of the same identity under one bucket is
    GUARANTEED to give low purity — a deepfake swap alters the ArcFace
    embedding by design. The honest audit asks: within "real frames of
    person X" alone, is the embedding consistent? And separately for
    "fake frames tagged with person X". That isolates label-corruption
    from deepfake-induced embedding drift.
    """
    bucket: dict[tuple[str, str], list[np.ndarray]] = {}
    for _, row in df.iterrows():
        emb = np.asarray(row["arcface_embed"], dtype=np.float32)
        n = np.linalg.norm(emb)
        if n < 1e-6:
            continue
        key = (row["identity_key"], row["label"])
        bucket.setdefault(key, []).append(emb / n)
    return {k: np.vstack(v) for k, v in bucket.items()}


def per_identity_metrics(embeddings: np.ndarray) -> dict[str, float]:
    """embeddings: (N, 512) L2-normalized."""
    centroid = embeddings.mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-6)

    sims_to_centroid = embeddings @ centroid  # (N,)
    centroid_purity = float(sims_to_centroid.mean())
    centroid_purity_p10 = float(np.percentile(sims_to_centroid, 10))

    n = len(embeddings)
    metrics: dict[str, float] = {
        "n_frames": n,
        "centroid_purity_mean": centroid_purity,
        "centroid_purity_p10": centroid_purity_p10,
    }

    if n >= 10:
        try:
            km = KMeans(n_clusters=2, n_init=10, random_state=SEED)
            labels = km.fit_predict(embeddings)
            sil = silhouette_score(embeddings, labels, metric="cosine") if len(set(labels)) > 1 else float("nan")
            counts = np.bincount(labels)
            cluster_imbalance = float(counts.min() / counts.max())
            metrics["k2_silhouette"] = float(sil) if not np.isnan(sil) else float("nan")
            metrics["k2_min_max_ratio"] = cluster_imbalance
        except Exception as e:
            metrics["k2_silhouette"] = float("nan")
            metrics["k2_min_max_ratio"] = float("nan")
            metrics["k2_error"] = str(e)
    else:
        metrics["k2_silhouette"] = float("nan")
        metrics["k2_min_max_ratio"] = float("nan")

    return metrics


def main() -> int:
    df = pq.read_table(str(PARQUET)).to_pandas()
    print(f"loaded {len(df)} rows; {df['identity_key'].nunique()} identities")

    embs_by_bucket = load_embeddings_per_identity(df)
    print(f"(identity, label) buckets with valid embeddings: {len(embs_by_bucket)}")

    rows = []
    for (ident, label), embs in embs_by_bucket.items():
        m = per_identity_metrics(embs)
        m["identity_key"] = ident
        m["label"] = label
        m["centroid_purity_flag"] = bool(m["centroid_purity_mean"] < PURITY_THRESH)
        m["k2_split_flag"] = bool(
            not np.isnan(m["k2_silhouette"]) and m["k2_silhouette"] >= SILHOUETTE_THRESH
        )
        m["corrupted_flag"] = bool(m["centroid_purity_flag"] or m["k2_split_flag"])
        rows.append(m)

    out = pd.DataFrame(rows).sort_values("centroid_purity_mean")
    out.to_csv(OUT_DIR / "per_identity_purity.csv", index=False)

    real_only = out[out["label"] == "real"]
    fake_only = out[out["label"] == "fake"]

    avg_purity_real = float(real_only["centroid_purity_mean"].mean())
    avg_purity_fake = float(fake_only["centroid_purity_mean"].mean())
    avg_purity_overall = float(out["centroid_purity_mean"].mean())

    flagged_real = real_only[real_only["centroid_purity_flag"]]
    flagged_fake = fake_only[fake_only["centroid_purity_flag"]]
    flagged_split = out[out["k2_split_flag"]]

    # Decision rule operates on REAL-only purity. Fake purity is naturally
    # lower because of deepfake-induced embedding drift; that's not corruption.
    summary = {
        "n_buckets_audited": int(len(out)),
        "n_buckets_real": int(len(real_only)),
        "n_buckets_fake": int(len(fake_only)),
        "avg_centroid_purity_real": avg_purity_real,
        "avg_centroid_purity_fake": avg_purity_fake,
        "avg_centroid_purity_overall": avg_purity_overall,
        "purity_thresh": PURITY_THRESH,
        "silhouette_thresh": SILHOUETTE_THRESH,
        "n_real_flagged_low_purity": int(len(flagged_real)),
        "n_fake_flagged_low_purity": int(len(flagged_fake)),
        "n_flagged_k2_split": int(len(flagged_split)),
        "verdict_real_purity_below_threshold": bool(avg_purity_real < PURITY_THRESH),
        "decision_rule": (
            "Per V6 §3.11: if avg purity (REAL identities) < 0.85 -> label corruption "
            "confirmed -> drop contrastive_regularization in P14. Fake purity is "
            "naturally lower because deepfake morphs ArcFace embedding by design."
        ),
    }
    (OUT_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Markdown report
    verdict_str = (
        "REAL identity corruption confirmed"
        if summary["verdict_real_purity_below_threshold"]
        else "REAL identities sufficiently pure"
    )
    md = [
        "# Track H.2 — ArcFace identity-purity audit",
        "",
        "Generated 2026-04-28 by `analysis/arcface_training_audit_2026-04-29/run_audit.py`.",
        "Source: `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` (dev+lockbox; proxy for training-set identity purity).",
        "All compute: n_jobs=1 (per `feedback_sklearn_njobs.md`).",
        "",
        "## Methodology",
        "",
        "- Bucket by (identity_key, label). Mixing real+fake under one identity_key is",
        "  guaranteed to depress purity because deepfake morphs ArcFace embedding by",
        "  design — that's not corruption, it's the swap working. Within-label purity",
        "  isolates the actual H.2 hypothesis (mixed-identity content under one key).",
        "- 512-dim L2-normalized ArcFace embeddings.",
        "- Per bucket: centroid = mean(embeddings); purity = mean cosine-sim to centroid.",
        f"- k=2 split-test: KMeans(k=2) silhouette score (≥ {SILHOUETTE_THRESH} suggests two sub-identities).",
        f"- Flagged corrupted if mean purity < {PURITY_THRESH} OR silhouette ≥ {SILHOUETTE_THRESH}.",
        "- **Decision rule operates on REAL-bucket purity only.**",
        "",
        "## Headline",
        "",
        f"- Buckets audited: **{summary['n_buckets_audited']}** (real={summary['n_buckets_real']}, fake={summary['n_buckets_fake']})",
        f"- **Avg purity (REAL buckets): {avg_purity_real:.3f}** (threshold {PURITY_THRESH})",
        f"- Avg purity (fake buckets): {avg_purity_fake:.3f} *(lower-by-design, see Methodology)*",
        f"- REAL buckets flagged purity < 0.85: **{summary['n_real_flagged_low_purity']}** / {summary['n_buckets_real']}",
        f"- Fake buckets flagged purity < 0.85: **{summary['n_fake_flagged_low_purity']}** / {summary['n_buckets_fake']}",
        f"- Buckets flagged by k=2 split (≥ {SILHOUETTE_THRESH}): **{summary['n_flagged_k2_split']}**",
        "",
        f"**Verdict: {verdict_str}** "
        f"(REAL avg purity {'<' if summary['verdict_real_purity_below_threshold'] else '>='} threshold).",
        "",
        "## Implication for P14 / P15",
        "",
        "- **If corruption confirmed (REAL avg < 0.85)**: drop `contrastive_regularization` in any retry (per Plan v6 §3.11).",
        "- **If pure**: contrastive_reg can stay. Audit refutes the H9 corruption hypothesis from Plan v3 §1.6.",
        "- **Fake-bucket purity is informational** — it indicates how distinct the deepfake morphs are from the source identity. Lower fake-bucket purity = more aggressive face-swap = more identity-loss in the training signal.",
        "",
        "## Buckets flagged (worst purity first)",
        "",
        "| identity_key | label | n_frames | centroid_purity | k2_silhouette |",
        "|---|---|---|---|---|",
    ]
    flagged_any = out[out["corrupted_flag"]]
    flagged_sorted = flagged_any.sort_values("centroid_purity_mean")
    for _, r in flagged_sorted.iterrows():
        md.append(
            f"| {r['identity_key']} | {r['label']} | {int(r['n_frames'])} | "
            f"{r['centroid_purity_mean']:.3f} | {r['k2_silhouette']:.3f} |"
        )
    if not len(flagged_sorted):
        md.append("| (none) | | | | |")

    md += [
        "",
        "## All buckets (sorted by purity)",
        "",
        "| identity_key | label | n_frames | centroid_purity | k2_silhouette |",
        "|---|---|---|---|---|",
    ]
    for _, r in out.iterrows():
        md.append(
            f"| {r['identity_key']} | {r['label']} | {int(r['n_frames'])} | "
            f"{r['centroid_purity_mean']:.3f} | {r['k2_silhouette']:.3f} |"
        )
    md.append("")
    (OUT_DIR / "AUDIT_REPORT.md").write_text("\n".join(md) + "\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
