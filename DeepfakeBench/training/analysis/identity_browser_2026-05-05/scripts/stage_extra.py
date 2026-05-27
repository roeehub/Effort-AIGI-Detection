"""Stage extra_<name> entities (xiang, xinghe, roy_d) into the browser.

Real/fake variants of the same name collapse into ONE base_identity via the
__real / __fake suffix convention that build_browser's base_identity() strips.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
SRC_ROOT = ROOT / "analysis/extra_2026-05-05"
SRC_FRAMES = SRC_ROOT / "raw/all"
SRC_SCORES = SRC_ROOT / "scores"
DST = ROOT / "analysis/identity_browser_2026-05-05"

SUITE = "extra"
BUCKET = "local"

CKPT_MAP = {
    "P8A_REFERENCE_STEP5000": "P8A",
    "E2B_TOP_N_STEP3200": "E2B",
    "PA_TOP_N_STEP3800": "PA_3800",
}

THUMB_MAX_EDGE = 256


def classify(dir_name: str) -> tuple[str, int]:
    """Return (video_id, label) for an inference dir name.

    `extra_xiang_real` → ('extra_xiang__real', 0)
    `extra_xiang_fake` → ('extra_xiang__fake', 1)
    `extra_roy_d`      → ('extra_roy_d', 0)         # default real
    """
    if dir_name.endswith("_real"):
        return (dir_name[: -len("_real")] + "__real", 0)
    if dir_name.endswith("_fake"):
        return (dir_name[: -len("_fake")] + "__fake", 1)
    return (dir_name, 0)


def main() -> int:
    (DST / "data").mkdir(parents=True, exist_ok=True)

    # Score lookup: {ident_dir/file_name -> prob} per ckpt
    score_lookups: dict[str, dict[str, float]] = {}
    for src_name, dst_name in CKPT_MAP.items():
        df = pd.read_csv(SRC_SCORES / f"{src_name}.csv")
        score_lookups[dst_name] = {}
        for _, row in df.iterrows():
            p = Path(row["frame_path"])
            score_lookups[dst_name][f"{p.parent.name}/{p.name}"] = float(row["frame_prob"])

    manifest_rows = []
    score_rows: dict[str, list[dict]] = {n: [] for n in CKPT_MAP.values()}
    for ident_dir in sorted(SRC_FRAMES.iterdir()):
        if not ident_dir.is_dir():
            continue
        video_id, label = classify(ident_dir.name)
        for img in sorted(ident_dir.iterdir()):
            if img.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            uri = f"gs://local/extra/{ident_dir.name}/{img.name}"
            manifest_rows.append({
                "suite": SUITE,
                "video_id": video_id,
                "frame_path": uri,
                "identity": video_id,
                "label": label,
                "is_lockbox": False,
                "bucket": BUCKET,
                "_local_src": img,
                "_ident_dir": ident_dir.name,
            })
            key = f"{ident_dir.name}/{img.name}"
            for ckpt in CKPT_MAP.values():
                p = score_lookups[ckpt].get(key)
                if p is not None:
                    score_rows[ckpt].append({"frame_path": uri, "frame_prob": p})

    out_manifest = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in manifest_rows]
    )
    manifest_path = DST / "data" / "extra_manifest.csv"
    out_manifest.to_csv(manifest_path, index=False)
    print(f"  manifest: {len(out_manifest)} rows -> {manifest_path}")

    for ckpt, rows in score_rows.items():
        out = pd.DataFrame(rows)
        path = DST / "data" / f"extra_scores_{ckpt}.csv"
        out.to_csv(path, index=False)
        print(f"  scores [{ckpt}]: {len(out)} rows -> {path}")

    n_frames = 0
    n_thumbs = 0
    for row in manifest_rows:
        ident = row["video_id"]
        src_name = Path(row["frame_path"]).name
        target_name = f"{SUITE}__{src_name}"
        src_img = row["_local_src"]
        dst_img = DST / "frames" / ident / target_name
        dst_jpg = DST / "thumbs" / ident / Path(target_name).with_suffix(".jpg").name
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
