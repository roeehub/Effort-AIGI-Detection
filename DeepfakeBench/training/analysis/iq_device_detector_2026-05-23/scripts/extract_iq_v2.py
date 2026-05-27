"""IQ extraction v2 — uses GCS download fallback for frames not available locally.

Computes IQ features directly from in-memory image bytes (no temp files).
"""
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis/team_identity_deploy_readout_expanded_2026-05-23/scripts"))
sys.path.insert(0, str(REPO_ROOT / "analysis/frozen_clip_team_identity_baseline_2026-05-23/scripts"))

from extract_clip_features import fetch_gcs, resolve_frame_path  # noqa: E402

PER_FRAME = REPO_ROOT / "analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_frame_full.csv"
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def _jpeg_qf_from_bytes(b: bytes) -> float | None:
    try:
        with Image.open(io.BytesIO(b)) as im:
            qt = getattr(im, "quantization", None)
        if not qt:
            return None
        luma = None
        if isinstance(qt, dict):
            for k in (0, "0"):
                if k in qt:
                    luma = qt[k]
                    break
            if luma is None and qt:
                luma = next(iter(qt.values()))
        if luma is None or len(luma) == 0:
            return None
        arr = np.asarray(luma, dtype=np.float32)
        s = float(arr.sum())
        if s <= 0:
            return None
        if s >= 1117 * 2:
            qf = 5000.0 / s
        else:
            qf = 100.0 - s / (1117.0 * 2.0) * 50.0
        return max(1.0, min(100.0, qf))
    except Exception:
        return None


def compute_iq_from_img(img_bgr: np.ndarray, raw_bytes: bytes | None = None) -> dict:
    """Compute IQ features from BGR ndarray. Returns dict; no None values for numerics."""
    h, w = img_bgr.shape[:2]
    out: dict = {
        "width": w, "height": h,
        "aspect_ratio": round(w / h, 4) if h else None,
    }
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    out["sharpness_laplacian"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    out["contrast_rms"] = float(gray.std())

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    s = hsv[..., 1]
    out["brightness_v_mean"] = float(v.mean())
    out["brightness_v_std"] = float(v.std())
    out["saturation_s_mean"] = float(s.mean())
    out["clipped_highlights_frac"] = float((v == 255).mean())

    b, g, r = cv2.split(img_bgr)
    out["r_mean"] = float(r.mean()); out["g_mean"] = float(g.mean()); out["b_mean"] = float(b.mean())
    out["r_std"] = float(r.std()); out["g_std"] = float(g.std()); out["b_std"] = float(b.std())
    out["color_cast_rg"] = out["r_mean"] - out["g_mean"]
    out["color_cast_rb"] = out["r_mean"] - out["b_mean"]
    out["color_cast_gb"] = out["g_mean"] - out["b_mean"]

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_lab, a_lab, b_lab = cv2.split(lab)
    out["lab_l_mean"] = float(l_lab.mean()); out["lab_a_mean"] = float(a_lab.mean()); out["lab_b_mean"] = float(b_lab.mean())
    out["lab_l_std"] = float(l_lab.std()); out["lab_a_std"] = float(a_lab.std()); out["lab_b_std"] = float(b_lab.std())

    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    out["edge_density"] = float(edges.mean()) / 255.0

    out["jpeg_qf"] = _jpeg_qf_from_bytes(raw_bytes) if raw_bytes else None
    return out


def main() -> None:
    df = pd.read_csv(PER_FRAME)
    real = df[df["role"] == "real"].copy()
    print(f"real frames: {len(real)} (deploy={int(real['deploy_relevant'].sum())}, mac_oos={int((~real['deploy_relevant'].astype(bool)).sum())})")

    out_path = OUT / "per_frame_iq_v2.parquet"
    if out_path.exists():
        print(f"CACHE HIT: {out_path}")
        return

    rows = []
    n_local = 0
    n_gcs = 0
    n_failed = 0
    client_cache: dict = {}
    t0 = time.time()

    for i, row in enumerate(real.itertuples(index=False)):
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
            # GCS download — re-implements fetch_gcs but also returns raw bytes for JPEG QF
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
            print(f"  [{i+1}/{len(real)}] {rate:.1f} fps  local={n_local} gcs={n_gcs} fail={n_failed}")

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(out_df)} rows to {out_path}")
    print(f"  local={n_local}, gcs={n_gcs}, failed={n_failed}")
    print(f"  per-human: {out_df.groupby('human').size().to_dict()}")


if __name__ == "__main__":
    main()
