"""IQ extraction for FAKE frames — mirrors extract_iq_v2.py but filters on role != real.

Reuses compute_iq_from_img() + fetch_gcs() infrastructure from extract_iq_v2.
Output: outputs/per_frame_iq_fakes_v2.parquet
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis/iq_device_detector_2026-05-23/scripts"))
sys.path.insert(0, str(REPO_ROOT / "analysis/team_identity_deploy_readout_expanded_2026-05-23/scripts"))
sys.path.insert(0, str(REPO_ROOT / "analysis/frozen_clip_team_identity_baseline_2026-05-23/scripts"))

from extract_iq_v2 import compute_iq_from_img  # noqa: E402
from extract_clip_features import resolve_frame_path  # noqa: E402

PER_FRAME = REPO_ROOT / "analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_frame_full.csv"
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(PER_FRAME)
    fakes = df[df["role"] != "real"].copy()
    print(f"fake frames: {len(fakes)}")
    print(f"  by role: {fakes['role'].value_counts().to_dict()}")
    print(f"  by deploy_relevant: {fakes['deploy_relevant'].value_counts().to_dict()}")

    out_path = OUT / "per_frame_iq_fakes_v2.parquet"
    if out_path.exists():
        print(f"CACHE HIT: {out_path}")
        return

    rows = []
    n_local = 0
    n_gcs = 0
    n_failed = 0
    client_cache: dict = {}
    t0 = time.time()

    for i, row in enumerate(fakes.itertuples(index=False)):
        fp = row.frame_path
        img_bgr = None
        raw_bytes = None

        local_path = resolve_frame_path(fp)
        if local_path is not None:
            try:
                with open(local_path, "rb") as f:
                    raw_bytes = f.read()
                img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    n_local += 1
            except Exception:
                img_bgr = None
                raw_bytes = None

        if img_bgr is None and fp.startswith("gs://") and not fp.startswith("gs://local/"):
            try:
                if "storage_client" not in client_cache:
                    from google.cloud import storage
                    client_cache["storage_client"] = storage.Client()
                client = client_cache["storage_client"]
                no_scheme = fp[len("gs://"):]
                bucket, blob_path = no_scheme.split("/", 1)
                b_obj = client.bucket(bucket).blob(blob_path)
                raw_bytes = b_obj.download_as_bytes()
                img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    n_gcs += 1
            except Exception:
                img_bgr = None
                raw_bytes = None

        if img_bgr is None:
            n_failed += 1
            continue

        try:
            iq = compute_iq_from_img(img_bgr, raw_bytes)
            r_out = {
                "frame_path": fp,
                "role": row.role,
                "human": row.human,
                "base_identity": row.base_identity,
                "deploy_relevant": row.deploy_relevant,
                "prob_P8A": row.prob_P8A,
                **iq,
            }
            rows.append(r_out)
        except Exception as e:
            n_failed += 1
            if i < 5:
                print(f"  IQ compute fail on {fp}: {e}")

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(fakes)}] {rate:.1f} fps  local={n_local} gcs={n_gcs} fail={n_failed}")

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(out_df)} rows to {out_path}")
    print(f"  local={n_local}, gcs={n_gcs}, failed={n_failed}")
    print(f"  per-role: {out_df.groupby('role').size().to_dict()}")
    print(f"  per-base_identity: {out_df.groupby('base_identity').size().sort_values(ascending=False).head(10).to_dict()}")


if __name__ == "__main__":
    main()
