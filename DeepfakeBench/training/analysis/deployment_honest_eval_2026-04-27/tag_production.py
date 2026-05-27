"""Download the 180 production-tagged frames from gs://real-teams-dor-roee/...
and run quality + face_geometry layers on them. Then compare the working
regime (GREEN, scores ~0.02-0.04) vs the false-flag regime (RED, scores
near 1.0) on every property axis to identify which properties separate them.

This is the deployment-honest equivalent of the parallel agent's per-frame
property analysis on the lockbox parquet — but on the actual production
real-world distribution, with model scores already attached (no inference
needed).

Run from training/:
  python3 -m analysis.deployment_honest_eval_2026-04-27.tag_production
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = REPO / "analysis/deployment_honest_eval_2026-04-27"
CACHE_DIR = OUT_DIR / "_prod_cache"
TAGS_JSON = CACHE_DIR / "combined_frame_tags.json"

from analysis.lockbox_tagging.layers.quality import compute_quality
from analysis.lockbox_tagging.layers.face_geometry import compute_face_geometry


def gcs_path_for_frame(tag_name: str, source_filename: str) -> str:
    bucket = "real-teams-dor-roee"
    parent = "session_20260424_combined_tags_121458_121007"
    return f"gs://{bucket}/{parent}/{tag_name}/session_20260424_121458__dor shkedi__{source_filename}"


def fetch_frame_local(args: tuple[str, str]) -> str | None:
    """Download a frame to local cache; return local path or None on failure.

    args: (gcs_uri, target_local_path)
    """
    uri, target = args
    target_p = Path(target)
    if target_p.exists() and target_p.stat().st_size > 0:
        return str(target_p)
    target_p.parent.mkdir(parents=True, exist_ok=True)
    rc = os.system(f"gsutil -q cp '{uri}' '{target}' 2>/dev/null")
    return str(target_p) if rc == 0 and target_p.exists() else None


def main() -> None:
    with open(TAGS_JSON) as f:
        d = json.load(f)
    parent = d["session_name"]

    # Build the work list across all 6 tags.
    rows = []
    fetch_args = []
    for tag in d.get("tags", []):
        tag_name = tag.get("name")
        for it in tag.get("items", []):
            participant = it.get("participant", "")
            source_fn = it.get("source_filename")
            if not source_fn:
                continue
            # The actual filename in the upload uses `__<participant>__<source_filename>` form,
            # but the README says one parent prefix is `session_20260424_121458__dor shkedi__`
            # plus the source filename. However for "roee-real-windows-laptop-correct" the
            # participant is "roee" -> prefix differs. We need to GS-list to know exactly.
            local_target = CACHE_DIR / "frames" / tag_name / source_fn
            rows.append({
                "tag": tag_name,
                "participant": participant,
                "source_filename": source_fn,
                "score": it.get("score"),
                "verdict": it.get("verdict"),
                "confidence": it.get("confidence"),
                "local_path": str(local_target),
            })

    # Build candidate URIs robustly: list each tag folder once and match by source_filename.
    print(f"[tag-prod] resolving GCS paths via gsutil ls per tag ...")
    gs_index: dict[tuple[str, str], str] = {}
    for tag in d.get("tags", []):
        tag_name = tag.get("name")
        prefix = f"gs://real-teams-dor-roee/{parent}/{tag_name}/"
        # `gsutil ls` returns one URI per line, ending with the filename.
        out = os.popen(f"gsutil ls '{prefix}' 2>/dev/null").read()
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            fname = line.rsplit("/", 1)[-1]
            # Filename in storage is e.g. `session_20260424_121458__dor shkedi__frame_001030_seq1796.png`.
            # Match by trailing source_filename portion (after the last `__`).
            for it in tag.get("items", []):
                src_fn = it.get("source_filename")
                if src_fn and line.endswith(src_fn):
                    gs_index[(tag_name, src_fn)] = line
                    break

    print(f"[tag-prod] resolved {len(gs_index)} URIs")

    # Download in parallel, ~16 workers.
    fetch_args = []
    for r in rows:
        key = (r["tag"], r["source_filename"])
        uri = gs_index.get(key)
        if uri is None:
            r["gcs_uri"] = None
            continue
        r["gcs_uri"] = uri
        fetch_args.append((uri, r["local_path"]))

    print(f"[tag-prod] downloading {len(fetch_args)} frames ...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(fetch_frame_local, fetch_args))

    # Tag each frame.
    print(f"[tag-prod] tagging frames ...")
    enriched = []
    for r in rows:
        rec = dict(r)
        local = Path(r["local_path"])
        if not local.exists():
            enriched.append(rec)
            continue
        rec.update(compute_quality(local))
        rec.update(compute_face_geometry(local))
        # face_pixel_area should be face_bbox_w * face_bbox_h (mediapipe default).
        if rec.get("face_bbox_w") and rec.get("face_bbox_h"):
            rec.setdefault("face_pixel_area",
                           float(rec["face_bbox_w"]) * float(rec["face_bbox_h"]))
        enriched.append(rec)

    df = pd.DataFrame(enriched)
    df.to_parquet(OUT_DIR / "production_180_tags.parquet", index=False)
    print(f"[tag-prod] wrote {len(df)} rows to production_180_tags.parquet")

    # ---- Side-by-side comparison: working vs failing regimes ----
    df["regime"] = df["tag"].map(lambda t: "FAIL" if "false-flag" in (t or "").lower() else "OK")
    print(f"\nRegime sizes: {df['regime'].value_counts().to_dict()}")

    # Property comparison
    cols_compare = [
        "face_pixel_area", "sharpness_laplacian", "brightness_v_mean",
        "brightness_v_std", "contrast_rms", "yaw_deg", "pitch_deg",
        "face_count", "score",
    ]
    print(f"\n=== Production property comparison: working (OK) vs failing (FAIL) regimes ===")
    summary_rows = []
    for col in cols_compare:
        if col not in df.columns:
            continue
        for regime in ["OK", "FAIL"]:
            sub = df[(df["regime"] == regime) & df[col].notna()]
            if len(sub) == 0:
                continue
            s = sub[col]
            summary_rows.append({
                "column": col, "regime": regime, "n": len(sub),
                "mean": float(s.mean()),
                "p10": float(s.quantile(0.1)),
                "p50": float(s.quantile(0.5)),
                "p90": float(s.quantile(0.9)),
                "min": float(s.min()),
                "max": float(s.max()),
            })
    sd = pd.DataFrame(summary_rows)
    print(sd.to_string(index=False))
    sd.to_csv(OUT_DIR / "production_180_regime_comparison.csv", index=False)

    # ---- Diagnostic: which property has the BIGGEST OK-vs-FAIL separation ----
    print(f"\n=== Per-tag breakdown ===")
    for col in ["face_pixel_area", "sharpness_laplacian", "brightness_v_mean", "yaw_deg", "pitch_deg"]:
        if col not in df.columns:
            continue
        print(f"\n--- {col} ---")
        for tag, sub in df.groupby("tag"):
            if sub[col].isna().all():
                print(f"  {tag}: all NaN")
                continue
            print(f"  {tag}: n={len(sub)}, mean={sub[col].mean():.2f}, p10={sub[col].quantile(0.1):.2f}, p90={sub[col].quantile(0.9):.2f}")

    # Save final
    out_summary = {
        "n_total": int(len(df)),
        "n_with_face": int((df["face_count"] >= 1).sum()) if "face_count" in df.columns else None,
        "regime_counts": df["regime"].value_counts().to_dict(),
        "score_by_regime": {
            "OK_mean": float(df.loc[df["regime"] == "OK", "score"].mean()),
            "FAIL_mean": float(df.loc[df["regime"] == "FAIL", "score"].mean()),
        },
        "outputs": ["production_180_tags.parquet", "production_180_regime_comparison.csv"],
    }
    (OUT_DIR / "production_180_summary.json").write_text(json.dumps(out_summary, indent=2))
    print(f"\n[tag-prod] summary: {json.dumps(out_summary, indent=2)}")


if __name__ == "__main__":
    main()
