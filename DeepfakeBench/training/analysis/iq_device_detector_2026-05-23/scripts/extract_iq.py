"""Extract per-frame image-quality features for team-identity + Mac-Roee.

Uses analysis/lockbox_tagging/layers/quality.py compute_quality() + extends
with per-channel color features (R/G/B means, std, color cast indicators).

Resolves frames via local cache (the team-identity readout's local_frame_resolver).
Falls back to GCS if needed.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis/team_identity_deploy_readout_expanded_2026-05-23/scripts"))
sys.path.insert(0, str(REPO_ROOT / "analysis/lockbox_tagging/layers"))

from quality import compute_quality  # noqa: E402

try:
    from local_frame_resolver import resolve_local  # noqa: E402
except ImportError:
    resolve_local = None

PER_FRAME = REPO_ROOT / "analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_frame_full.csv"
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def compute_color_features(local_path: str) -> dict:
    """Per-channel color features beyond compute_quality."""
    img = cv2.imread(local_path, cv2.IMREAD_COLOR)
    if img is None:
        return {}
    # cv2 is BGR
    b, g, r = cv2.split(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_lab, a_lab, b_lab = cv2.split(lab)
    return {
        "r_mean": float(r.mean()), "g_mean": float(g.mean()), "b_mean": float(b.mean()),
        "r_std": float(r.std()), "g_std": float(g.std()), "b_std": float(b.std()),
        "color_cast_rg": float(r.mean()) - float(g.mean()),  # warm-cool axis
        "color_cast_rb": float(r.mean()) - float(b.mean()),  # warm-cool axis
        "color_cast_gb": float(g.mean()) - float(b.mean()),
        "lab_l_mean": float(l_lab.mean()), "lab_a_mean": float(a_lab.mean()), "lab_b_mean": float(b_lab.mean()),
        "lab_l_std": float(l_lab.std()), "lab_a_std": float(a_lab.std()), "lab_b_std": float(b_lab.std()),
    }


def main() -> None:
    df = pd.read_csv(PER_FRAME)
    print(f"loaded {len(df)} frames from per_frame_full")
    # Filter to REAL frames (team-id + Mac-Roee) — we only need real for the device detector
    real = df[df["role"] == "real"].copy()
    print(f"real frames: {len(real)} (deploy={int(real['deploy_relevant'].sum())}, mac_oos={int((~real['deploy_relevant'].astype(bool)).sum())})")

    out_path = OUT / "per_frame_iq.parquet"
    if out_path.exists():
        print(f"CACHE HIT: {out_path}")
        return

    rows = []
    n_fail_resolve = 0
    n_fail_decode = 0
    t0 = time.time()
    for i, row in enumerate(real.itertuples(index=False)):
        # Resolve to local path
        gcs_path = row.frame_path
        local_path = None
        if resolve_local is not None:
            try:
                local_path = resolve_local(gcs_path)
            except Exception:
                local_path = None
        if local_path is None or not Path(local_path).exists():
            n_fail_resolve += 1
            if i < 5:
                print(f"  resolve fail: {gcs_path}")
            continue

        try:
            q = compute_quality(Path(local_path))
            if not q.get("decode_ok"):
                n_fail_decode += 1
                continue
            c = compute_color_features(local_path)
            r_out = {
                "frame_path": gcs_path,
                "human": row.human,
                "base_identity": row.base_identity,
                "deploy_relevant": row.deploy_relevant,
                "prob_P8A": row.prob_P8A,
                **q,
                **c,
            }
            rows.append(r_out)
        except Exception as e:
            n_fail_decode += 1
            if i < 5:
                print(f"  decode/iq fail on {gcs_path}: {e}")

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(real)}] {rate:.1f} fps, fail_resolve={n_fail_resolve}, fail_decode={n_fail_decode}")

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(out_df)} rows to {out_path}")
    print(f"  fail_resolve={n_fail_resolve}, fail_decode={n_fail_decode}")
    print(f"  columns: {list(out_df.columns)}")


if __name__ == "__main__":
    main()
