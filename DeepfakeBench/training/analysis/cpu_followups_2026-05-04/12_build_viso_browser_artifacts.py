"""Build viewer artifacts to browse the 'uncaught viso fakes' question.

Outputs:
  outputs/viso_per_frame_with_all_ckpts.csv
      One row per viso fake frame. Columns: frame_path, video_id, P8A_score,
      E2B_3200_score, E3_6600_score, P8A_caught (1/0 at FPR=10%), E2B_caught,
      E3_caught, caught_by_subset, n_ckpts_caught.

  outputs/viso_score_space_tsne.csv
      Per-frame 2D embedding in score-space (each frame is (P8A, E2B, E3)
      scores; t-SNE/PCA to 2D). Viewer-compatible columns: checkpoint, label,
      tsne_x, tsne_y, frame_path, score, video_id, method, caught_by.

  outputs/viso_uncaught_at_fpr10.csv
      Just the 364 frames caught by NO ckpt at FPR=10%. Same columns as the
      first table. This is the "uncaught" set for direct browsing.

  Pseudo-run wired into viewer/model_dashboard_runs.yaml as 'viso_uncaught'.
"""
import csv
from pathlib import Path
import numpy as np

ANALYSIS_DIR = Path(__file__).parent
RAW_DIR = ANALYSIS_DIR / "raw_reports"
OUT_DIR = ANALYSIS_DIR / "outputs"

CKPTS = {
    "P8A": "p8a_reference_step5000",
    "E2B_3200": "e2b_top_n_step3200",
    "E3_6600": "e3_top_n_step6600",
}


def load(suite, ckpt_token):
    f = RAW_DIR / f"{suite}_{ckpt_token}_frames_report.csv"
    if not f.exists():
        return []
    with f.open() as fh:
        return list(csv.DictReader(fh))


def calib(real_scores, target_fpr):
    s = np.sort(real_scores)
    idx = max(0, int(np.ceil(len(s) * (1 - target_fpr))) - 1)
    return float(s[idx]) if idx < len(s) else 1.01


def main():
    # 1. Calibrate τ at FPR=10% per ckpt
    taus = {}
    for ckpt_name, ckpt_token in CKPTS.items():
        real = load("teams_real_all_dev", ckpt_token)
        rs = np.array([float(f["frame_prob"]) for f in real if int(f["label"]) == 0])
        taus[ckpt_name] = calib(rs, 0.10)

    # 2. Load viso fake frames per ckpt and join
    by_path = {}
    for ckpt_name, ckpt_token in CKPTS.items():
        for row in load("visomaster_enhanced_macro_dev", ckpt_token):
            if int(row["label"]) != 1:
                continue
            fp = row["frame_path"]
            if fp not in by_path:
                by_path[fp] = {
                    "frame_path": fp,
                    "video_id": row["video_id"],
                    "method": row["method"],
                    "label": 1,
                }
            by_path[fp][f"{ckpt_name}_score"] = float(row["frame_prob"])

    # 3. Build joined table
    rows = []
    for fp, d in by_path.items():
        if not all(f"{c}_score" in d for c in CKPTS):
            continue
        caught = []
        for c in CKPTS:
            score = d[f"{c}_score"]
            is_caught = int(score >= taus[c])
            d[f"{c}_caught_at_fpr10"] = is_caught
            if is_caught:
                caught.append(c)
        d["caught_by_subset"] = "+".join(sorted(caught)) if caught else "uncaught"
        d["n_ckpts_caught"] = len(caught)
        rows.append(d)

    # Sort by total catch (uncaught first — most interesting)
    rows.sort(key=lambda r: (r["n_ckpts_caught"], -r["P8A_score"]))

    # Write full joined table
    full_csv = OUT_DIR / "viso_per_frame_with_all_ckpts.csv"
    fieldnames = ["frame_path", "video_id", "method", "label",
                  "P8A_score", "E2B_3200_score", "E3_6600_score",
                  "P8A_caught_at_fpr10", "E2B_3200_caught_at_fpr10", "E3_6600_caught_at_fpr10",
                  "caught_by_subset", "n_ckpts_caught"]
    with full_csv.open("w", newline="") as fh:
        wrt = csv.DictWriter(fh, fieldnames=fieldnames)
        wrt.writeheader()
        wrt.writerows(rows)
    print(f"  → {full_csv} ({len(rows)} viso fake frames)")

    # Write uncaught-only table
    uncaught_csv = OUT_DIR / "viso_uncaught_at_fpr10.csv"
    uncaught_rows = [r for r in rows if r["n_ckpts_caught"] == 0]
    with uncaught_csv.open("w", newline="") as fh:
        wrt = csv.DictWriter(fh, fieldnames=fieldnames)
        wrt.writeheader()
        wrt.writerows(uncaught_rows)
    print(f"  → {uncaught_csv} ({len(uncaught_rows)} uncaught viso fakes)")

    # 4. Build viewer-compatible per_frame_scores CSVs
    # Format: method,label,video_id,frame_path,frame_prob,group_key,family_key
    # We make 3 such CSVs (one per ckpt) but with ONLY the uncaught frames
    # so the viewer's frame browser surfaces them via False Negatives view.
    viewer_artifact_dir = ANALYSIS_DIR / "viewer_artifacts" / "viso_uncaught_browse"
    viewer_artifact_dir.mkdir(parents=True, exist_ok=True)

    # Add real frames too so the threshold view makes sense (else only fakes show up)
    # Pull the same teams_real_all_dev real frames so the viewer can compute FPR
    # ALL frames (caught and uncaught) so the user can also see the catchable ones for contrast
    for ckpt_name, ckpt_token in CKPTS.items():
        # Combine viso fakes (all of them) + dev real (subset to keep size reasonable)
        viso_fakes = load("visomaster_enhanced_macro_dev", ckpt_token)
        real_frames = load("teams_real_all_dev", ckpt_token)
        # Take 500 real frames for the FPR baseline
        real_subset = [r for r in real_frames if int(r["label"]) == 0][:500]
        out_csv = viewer_artifact_dir / f"viso_browse_{ckpt_token}_frames_report.csv"
        with out_csv.open("w", newline="") as fh:
            wrt = csv.DictWriter(fh, fieldnames=["method", "label", "video_id", "frame_path", "frame_prob", "group_key", "family_key"])
            wrt.writeheader()
            wrt.writerows(viso_fakes + real_subset)
        print(f"  → {out_csv} ({len(viso_fakes)} viso + {len(real_subset)} real)")

    # 5. Score-space pseudo-manifold (no GPU needed — just project 3-score vectors to 2D)
    print("\nComputing score-space embedding (PCA + t-SNE on (P8A, E2B, E3) score vectors)...")
    coords = np.array([[r["P8A_score"], r["E2B_3200_score"], r["E3_6600_score"]] for r in rows])

    # Try sklearn — fall back to PCA if t-SNE not available
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        scaled = StandardScaler().fit_transform(coords)
        # NOTE: per memory `feedback_sklearn_njobs.md`, n_jobs=1 to avoid swap exhaustion
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=1)
        emb = tsne.fit_transform(scaled)
        method_used = "t-SNE"
    except Exception as e:
        print(f"  t-SNE failed ({e}); using PCA fallback")
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2).fit_transform(coords)
        method_used = "PCA"

    # Write viewer-compatible manifold CSV
    # Existing manifold loader expects: checkpoint, tsne_x, tsne_y, label, frame_path, score
    manifold_csv = OUT_DIR / "viso_score_space_tsne.csv"
    with manifold_csv.open("w", newline="") as fh:
        wrt = csv.writer(fh)
        wrt.writerow(["checkpoint", "label", "tsne_x", "tsne_y", "frame_path", "gcs_uri",
                      "video_id", "method",
                      "P8A_score", "E2B_3200_score", "E3_6600_score", "caught_by_subset"])
        for r, (x, y) in zip(rows, emb):
            # gcs_uri = same as frame_path (which is a gs:// URI in our data)
            wrt.writerow(["VISO_UNCAUGHT", "fake", x, y, r["frame_path"], r["frame_path"],
                          r["video_id"], r["method"],
                          r["P8A_score"], r["E2B_3200_score"], r["E3_6600_score"], r["caught_by_subset"]])
    print(f"  → {manifold_csv} ({len(rows)} points, method={method_used})")

    # 6. Print summary tables
    print("\n=== Summary ===")
    print(f"Total viso fake frames: {len(rows)}")
    by_subset = {}
    for r in rows:
        by_subset.setdefault(r["caught_by_subset"], 0)
        by_subset[r["caught_by_subset"]] += 1
    print("Caught by subset:")
    for k, v in sorted(by_subset.items(), key=lambda x: -x[1]):
        print(f"  {k:30s} {v:4d} ({v/len(rows)*100:5.1f}%)")

    print(f"\nThresholds at FPR=10% on teams_real_all_dev:")
    for c, t in taus.items():
        print(f"  {c}: τ = {t:.4f}")


if __name__ == "__main__":
    main()
