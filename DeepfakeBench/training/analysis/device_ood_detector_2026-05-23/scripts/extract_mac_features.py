"""Extract frozen-CLIP L11 features for Mac-Roee frames (out-of-scope per deploy spec but needed for device-OOD detector).

Reuses the existing extract_all() from the frozen-CLIP baseline pipeline.
"""
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FROZEN_BASELINE = REPO_ROOT / "analysis/frozen_clip_team_identity_baseline_2026-05-23"
sys.path.insert(0, str(FROZEN_BASELINE / "scripts"))

import pandas as pd
from extract_clip_features import extract_all  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

PER_FRAME_CSV = REPO_ROOT / "analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_frame_full.csv"


def main() -> int:
    df = pd.read_csv(PER_FRAME_CSV, low_memory=False)
    print(f"loaded {len(df)} rows")
    df_mac = df[(df['human'] == 'Roee_Mac') & (df['role'] == 'real')].reset_index(drop=True)
    print(f"Mac-Roee real frames: {len(df_mac)}")

    out_npz = OUT / f"clip_frozen_l11__mac_roee_n{len(df_mac)}.npz"
    if out_npz.exists():
        print(f"CACHE HIT: {out_npz}")
        return 0

    extract_all(df_mac, out_npz)
    print(f"Saved: {out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
