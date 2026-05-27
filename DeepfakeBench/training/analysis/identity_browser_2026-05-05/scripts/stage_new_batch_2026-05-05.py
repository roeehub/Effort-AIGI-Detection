"""Stage the 3-source new-data batch into the identity browser.

Source dirs under analysis/new_data_batch_2026-05-05/raw/all/ are split into
three suites by prefix:
  dor_fake_*            (excluding deeplive_enhanced_*) → suite=dor_fake_local
  dor_fake_deeplive_*                                    → suite=live_fakes_teams_prod (extending)
  *_real_*                                               → suite=live_reals_teams_prod
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
SRC_ROOT = ROOT / "analysis/new_data_batch_2026-05-05"
SRC_FRAMES = SRC_ROOT / "raw/all"
SRC_SCORES = SRC_ROOT / "scores"
DST = ROOT / "analysis/identity_browser_2026-05-05"

CKPT_MAP = {
    "P8A_REFERENCE_STEP5000": "P8A",
    "E2B_TOP_N_STEP3200": "E2B",
    "PA_TOP_N_STEP3800": "PA_3800",
}

THUMB_MAX_EDGE = 256


def classify(dir_name: str) -> tuple[str, str, int]:
    """Return (suite, uri_prefix, label) for a directory name."""
    if dir_name.startswith("dor_fake_deeplive"):
        return ("live_fakes_teams_prod",
                f"gs://live-fakes-teams-prod/fake/session_20260324_174822/{dir_name.replace('dor_fake_deeplive_enhanced_', 'dor-fake-deeplive-enhanced-')}/",
                1)
    if dir_name.startswith("dor_fake_"):
        return ("dor_fake_local",
                f"gs://local/dor_deep_live_cam/{dir_name}/",
                1)
    # *_real_<DATE>
    return ("live_reals_teams_prod",
            f"gs://live-fakes-teams-prod/real/{dir_name}/",
            0)


def main() -> int:
    (DST / "data").mkdir(parents=True, exist_ok=True)
    manifests: dict[str, list[dict]] = {}      # suite -> rows
    score_rows: dict[str, list[dict]] = {n: [] for n in CKPT_MAP.values()}

    # Load score dicts: src_basename -> prob (per ckpt)
    score_lookups: dict[str, dict[str, float]] = {}
    for src_name, dst_name in CKPT_MAP.items():
        src_csv = SRC_SCORES / f"{src_name}.csv"
        df = pd.read_csv(src_csv)
        # Inference frame_path: "raw/all/<dir>/<file>"
        score_lookups[dst_name] = {}
        for _, row in df.iterrows():
            key = Path(row["frame_path"]).parent.name + "/" + Path(row["frame_path"]).name
            score_lookups[dst_name][key] = float(row["frame_prob"])

    for ident_dir in sorted(SRC_FRAMES.iterdir()):
        if not ident_dir.is_dir():
            continue
        suite, uri_prefix, label = classify(ident_dir.name)
        manifests.setdefault(suite, [])
        for img in sorted(ident_dir.iterdir()):
            if img.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            uri = f"{uri_prefix}{img.name}"
            manifests[suite].append({
                "suite": suite,
                "video_id": ident_dir.name,
                "frame_path": uri,
                "identity": ident_dir.name,
                "label": label,
                "is_lockbox": False,
                "bucket": uri.split("/")[2] if uri.startswith("gs://") else "local",
                "_local_src": img,
            })
            # Score rows for this URI
            score_key = f"{ident_dir.name}/{img.name}"
            for ckpt_name in CKPT_MAP.values():
                p = score_lookups[ckpt_name].get(score_key)
                if p is not None:
                    score_rows[ckpt_name].append({"frame_path": uri, "frame_prob": p})

    # Write manifests + scores
    for suite, rows in manifests.items():
        out = pd.DataFrame([{k: v for k, v in r.items() if k != "_local_src"} for r in rows])
        path = DST / "data" / f"{suite}_new_2026-05-05_manifest.csv"
        out.to_csv(path, index=False)
        print(f"  manifest [{suite}]: {len(out)} rows -> {path}")

    for ckpt_name, rows in score_rows.items():
        out = pd.DataFrame(rows).drop_duplicates(subset=["frame_path"])
        path = DST / "data" / f"new_batch_2026-05-05_scores_{ckpt_name}.csv"
        out.to_csv(path, index=False)
        print(f"  scores [{ckpt_name}]: {len(out)} rows -> {path}")

    # Stage frames + thumbs
    n_frames = 0
    n_thumbs = 0
    for rows in manifests.values():
        for row in rows:
            suite = row["suite"]
            ident = row["video_id"]
            src_name = Path(row["frame_path"]).name
            target_name = f"{suite}__{src_name}"
            src_img = row["_local_src"]
            dst_img = DST / "frames" / ident / target_name
            dst_jpg = (DST / "thumbs" / ident /
                       Path(target_name).with_suffix(".jpg").name)
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_jpg.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
                n_frames += 1
            if not dst_jpg.exists():
                with Image.open(src_img) as im:
                    im = im.convert("RGB")
                    im.thumbnail((THUMB_MAX_EDGE, THUMB_MAX_EDGE), Image.LANCZOS)
                    im.save(dst_jpg, "JPEG", quality=85)
                n_thumbs += 1

    print(f"  staged: {n_frames} frames copied, {n_thumbs} thumbs generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
