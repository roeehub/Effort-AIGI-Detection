"""
Build clean_eval_v1 + shortcut_probe_v1 from the n=7334 lockbox-tagging parquet.

Per Plan v6 §3.6:
- clean_eval_v1: ~50 frames, ~12-15 identities, balanced across modern Teams
  capture conditions, NOT dor_shkedi-skewed. Source for the Day-4 Axis-2
  (recall ≥ 80) deployment-honest gate.
- shortcut_probe_v1: ~10-20 same-face-different-pipeline pairs. Source for
  the Day-4 Axis-3 (max-min Δprob ≤ 0.15) shortcut gate.

Quality gates applied to both:
- face_area_ratio >= 0.10 (matches modern_v2 floor)
- NOT is_pose_extreme
- NOT is_no_face
- decode_ok

Capture-mode policy:
- clean_eval_v1 uses {normal_photo, phone_screen} only (deployment-realistic).
  Excludes webcam (FPR-mode driver per project_lockbox_fpr_dominated_by_webcam_mode)
  and screen/screen_recording (rare, OOD).
- shortcut_probe_v1 INCLUDES webcam pairs because the test is precisely whether
  the model reads pipeline (mode) signal as label signal. dor_shkedi has only
  normal_photo+webcam coverage so the canonical (dor_shkedi normal_photo,
  dor_shkedi webcam) pair lives there.

Outputs:
- clean_eval_v1_real.yaml      (frame URI list, ~25 frames)
- clean_eval_v1_fake.yaml      (frame URI list, ~25 frames)
- shortcut_probe_v1_pairs.yaml (paired URIs, ~15 pairs)
- BUILD_REPORT.md              (composition, identities, methods, modes)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SEED = 13
OUT_DIR = Path(__file__).parent.resolve()
PARQUET = Path("analysis/lockbox_tagging/full_tags_2026-04-27.parquet").resolve()


def quality_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["face_area_ratio"] >= 0.10)
        & (~df["is_pose_extreme"].astype(bool))
        & (~df["is_no_face"].astype(bool))
        & (df["decode_ok"].astype(bool))
    )


def build_clean_eval_real(df: pd.DataFrame) -> pd.DataFrame:
    # Deployment-realistic capture modes only.
    pool = df[
        (df["label"] == "real")
        & (df["clip_capture_mode"].isin(["normal_photo", "phone_screen"]))
        & quality_mask(df)
    ].copy()

    # Identity diversity: pick up to 2 frames per (identity, mode), max 4 per identity.
    # Skip dor_shkedi to avoid skew (it's the canonical anchor pool, evaluated separately).
    pool = pool[pool["identity_key"] != "dor_shkedi"]

    # Rank identities by frame count desc; take top 13 with usable coverage.
    top_ids = pool["identity_key"].value_counts().head(13).index.tolist()
    pool = pool[pool["identity_key"].isin(top_ids)]

    rows = []
    for ident in top_ids:
        sub = pool[pool["identity_key"] == ident]
        # Up to 2 per mode, max 4 per identity.
        per_mode = (
            sub.groupby("clip_capture_mode", group_keys=False)
            .apply(lambda g: g.sample(n=min(2, len(g)), random_state=SEED))
        )
        rows.append(per_mode.head(4))
    out = pd.concat(rows, ignore_index=True)
    return out.head(28)  # cap ~25-28 frames


def build_clean_eval_fake(df: pd.DataFrame) -> pd.DataFrame:
    pool = df[
        (df["label"] == "fake")
        & (df["clip_capture_mode"].isin(["normal_photo", "phone_screen"]))
        & quality_mask(df)
    ].copy()

    # Method diversity: 1-3 frames per method, prioritize methods with ≥30 frames.
    method_counts = pool["method"].value_counts()
    eligible = method_counts[method_counts >= 30].index.tolist()
    pool = pool[pool["method"].isin(eligible)]

    rows = []
    for method in eligible[:13]:  # top 13 methods
        sub = pool[pool["method"] == method]
        rows.append(sub.sample(n=min(2, len(sub)), random_state=SEED))
    out = pd.concat(rows, ignore_index=True)
    return out.head(26)


def build_shortcut_probe(df: pd.DataFrame) -> list[dict]:
    """
    Pairs of same-identity, different-capture-mode REAL frames.
    Model should give similar prob_fake on both members of each pair.

    Pair selection:
    - Identity must have ≥10 real frames in 2 distinct capture modes.
    - One frame randomly drawn per mode (deterministic seed).
    - Up to 2 pairs per identity (different mode-combos), capped at 15 total.
    """
    pool = df[(df["label"] == "real") & quality_mask(df)].copy()

    pairs = []
    rng_seed = SEED
    for ident, sub in pool.groupby("identity_key"):
        mode_counts = sub["clip_capture_mode"].value_counts()
        eligible_modes = mode_counts[mode_counts >= 10].index.tolist()
        if len(eligible_modes) < 2:
            continue

        # Form mode pairs (canonical ordering)
        mode_pairs = [
            (eligible_modes[i], eligible_modes[j])
            for i in range(len(eligible_modes))
            for j in range(i + 1, len(eligible_modes))
        ]
        for ma, mb in mode_pairs[:2]:
            a = sub[sub["clip_capture_mode"] == ma].sample(1, random_state=rng_seed).iloc[0]
            b = sub[sub["clip_capture_mode"] == mb].sample(1, random_state=rng_seed).iloc[0]
            pairs.append({
                "identity_key": ident,
                "a_uri": a["gcs_uri"],
                "a_mode": ma,
                "a_local_path": a["local_path"],
                "b_uri": b["gcs_uri"],
                "b_mode": mb,
                "b_local_path": b["local_path"],
                "label": "real",
                "expected": "Delta prob_fake should be small (model not pipeline-keying)",
            })
            rng_seed += 1
        if len(pairs) >= 15:
            break
    return pairs[:15]


def write_yaml_frame_list(frames: pd.DataFrame, path: Path, label: str) -> None:
    lines = [
        f"# {path.name}",
        f"# Generated by analysis/clean_eval_2026-04-29/build_clean_eval_and_probe.py",
        f"# Source: analysis/lockbox_tagging/full_tags_2026-04-27.parquet",
        f"# label={label}  count={len(frames)}",
        "",
        "frames:",
    ]
    for _, row in frames.iterrows():
        lines.append(f"  - gcs_uri: {row['gcs_uri']}")
        lines.append(f"    local_path: {row['local_path']}")
        lines.append(f"    identity_key: {row['identity_key']}")
        lines.append(f"    label: {row['label']}")
        lines.append(f"    method: {row.get('method', 'real')}")
        lines.append(f"    clip_capture_mode: {row['clip_capture_mode']}")
        lines.append(f"    face_pixel_area: {int(row['face_pixel_area'])}")
        lines.append(f"    face_area_ratio: {float(row['face_area_ratio']):.4f}")
        lines.append(f"    sharpness_laplacian: {float(row['sharpness_laplacian']):.1f}")
        lines.append(f"    split: {row['split']}")
    path.write_text("\n".join(lines) + "\n")


def write_pairs_yaml(pairs: list[dict], path: Path) -> None:
    lines = [
        f"# {path.name}",
        f"# Generated by analysis/clean_eval_2026-04-29/build_clean_eval_and_probe.py",
        f"# Source: analysis/lockbox_tagging/full_tags_2026-04-27.parquet",
        f"# Same-face-different-pipeline pairs (REAL label, varying capture mode)",
        f"# count={len(pairs)}",
        "",
        "pairs:",
    ]
    for i, p in enumerate(pairs):
        lines.append(f"  - pair_id: {i:02d}")
        lines.append(f"    identity_key: {p['identity_key']}")
        lines.append(f"    label: {p['label']}")
        lines.append(f"    a:")
        lines.append(f"      gcs_uri: {p['a_uri']}")
        lines.append(f"      local_path: {p['a_local_path']}")
        lines.append(f"      clip_capture_mode: {p['a_mode']}")
        lines.append(f"    b:")
        lines.append(f"      gcs_uri: {p['b_uri']}")
        lines.append(f"      local_path: {p['b_local_path']}")
        lines.append(f"      clip_capture_mode: {p['b_mode']}")
        lines.append(f"    expected: {p['expected']!r}")
    path.write_text("\n".join(lines) + "\n")


def write_report(real: pd.DataFrame, fake: pd.DataFrame, pairs: list[dict], path: Path) -> None:
    real_modes = real["clip_capture_mode"].value_counts().to_dict()
    real_ids = real["identity_key"].value_counts().to_dict()
    fake_modes = fake["clip_capture_mode"].value_counts().to_dict()
    fake_methods = fake["method"].value_counts().to_dict()
    pair_modes = defaultdict(int)
    for p in pairs:
        key = " <-> ".join(sorted([p["a_mode"], p["b_mode"]]))
        pair_modes[key] += 1
    pair_ids = defaultdict(int)
    for p in pairs:
        pair_ids[p["identity_key"]] += 1

    md = [
        "# clean_eval_v1 + shortcut_probe_v1 BUILD REPORT",
        "",
        "Generated 2026-04-28 by `analysis/clean_eval_2026-04-29/build_clean_eval_and_probe.py`.",
        "",
        f"Source: `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` (n=7334)",
        "",
        "## Quality gates",
        "",
        "- face_area_ratio >= 0.10",
        "- NOT is_pose_extreme",
        "- NOT is_no_face",
        "- decode_ok",
        "",
        "## clean_eval_v1 — deployment-honest substrate",
        "",
        f"- **Total frames**: {len(real) + len(fake)}",
        f"- Real: {len(real)} frames across {real['identity_key'].nunique()} identities",
        f"- Fake: {len(fake)} frames across {fake['method'].nunique()} methods",
        "- Capture modes restricted to {normal_photo, phone_screen} (deployment-realistic).",
        "- dor_shkedi excluded from real (anchor-pool overlap).",
        "",
        "### Real composition",
        "",
        f"- Capture modes: {real_modes}",
        f"- Identities: {real_ids}",
        "",
        "### Fake composition",
        "",
        f"- Capture modes: {fake_modes}",
        f"- Methods: {fake_methods}",
        "",
        "## shortcut_probe_v1 — same-face-different-pipeline pairs",
        "",
        f"- **Total pairs**: {len(pairs)}",
        "- Each pair: same identity_key, label=real, two distinct clip_capture_mode values.",
        "- Includes webcam (the test is precisely whether the model reads mode as label).",
        "",
        f"- Identity coverage: {dict(pair_ids)}",
        f"- Mode-combo coverage: {dict(pair_modes)}",
        "",
        "## Day-4 usage",
        "",
        "- **Axis 2**: score model on clean_eval_v1; require fake recall ≥ 80% at the same operating threshold used for the 90/5 axis.",
        "- **Axis 3**: score on shortcut_probe_v1 pair members; compute |prob_fake(a) - prob_fake(b)| per pair; take max across pairs. Require max Δ ≤ 0.15.",
        "",
    ]
    path.write_text("\n".join(md) + "\n")


def main() -> int:
    if not PARQUET.exists():
        print(f"ERROR: parquet not found at {PARQUET}", file=sys.stderr)
        return 1

    df = pq.read_table(str(PARQUET)).to_pandas()

    real = build_clean_eval_real(df)
    fake = build_clean_eval_fake(df)
    pairs = build_shortcut_probe(df)

    write_yaml_frame_list(real, OUT_DIR / "clean_eval_v1_real.yaml", "real")
    write_yaml_frame_list(fake, OUT_DIR / "clean_eval_v1_fake.yaml", "fake")
    write_pairs_yaml(pairs, OUT_DIR / "shortcut_probe_v1_pairs.yaml")
    write_report(real, fake, pairs, OUT_DIR / "BUILD_REPORT.md")

    summary = {
        "clean_eval_v1_real_count": len(real),
        "clean_eval_v1_fake_count": len(fake),
        "clean_eval_v1_real_identities": real["identity_key"].nunique(),
        "clean_eval_v1_fake_methods": fake["method"].nunique(),
        "shortcut_probe_v1_pair_count": len(pairs),
        "shortcut_probe_v1_identity_count": len({p["identity_key"] for p in pairs}),
    }
    (OUT_DIR / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
