"""P17 OQ1 — Substrate-classifier direction probe.

Question (from P17_FINAL_VERDICT.md, Open Question 1): the trained heads
picked a direction in feature space that is orthogonal (cos +0.03..+0.09) to
the substrate-INVARIANT direction (fresh-LR fake-vs-real on dev). We don't yet
know what substrate axis they DID align with.

This script trains LRs predicting various substrate labels from the same L3
features that the trained heads see (cached from P8A backbone, layer-3 [CLS]),
then computes cosine similarity between each trained head's decision direction
and each substrate-classifier direction.

Substrate axes probed:
  1. is_lockbox       (binary, 87 lb vs 713 dv) — the P17 transfer axis itself
  2. is_webcam        (clip_capture_mode == 'webcam', 238 vs 562) — webcam-mode
                      drives lockbox FPR per project_lockbox_fpr_dominated_by_webcam_mode
  3. is_phone_screen  (255 vs 545)
  4. is_screen_any    (screen + screen_recording, 43 vs 757)
  5. is_normal_photo  (264 vs 536)

Steers Phase 3 packet design:
  - High cos(trained_head_dir, substrate_dir) on any axis => trained head learned
    that substrate. Ramped GRL on that axis is structurally tight to the problem.
  - Orthogonal to ALL substrate axes probed => trained head learned something else
    entirely; need a different intervention or a different probe.

Run:
    python3 analysis/intermediate_layer_probe_2026-04-30/substrate_classifier_direction_2026-05-01.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
SAMPLED_CSV = (
    REPO_ROOT
    / "analysis"
    / "embedding_triptych_2026-04-30"
    / "outputs"
    / "triptych_p8a_slot2_slot3"
    / "sampled_frames.csv"
)
OUTPUT_DIR = REPO_ROOT / "analysis" / "intermediate_layer_probe_2026-04-30" / "outputs"
CKPT_DIR = Path("/tmp/p17_ckpts")
N_SAMPLES = 800

ARMS = [
    ("L3_ARCFACE_ep1",    CKPT_DIR / "first_best_effort_20260501_ep1_auc0.5006_eer0.4926.pth", "ARCFACE"),
    ("L3_ARCFACE_s1687",  CKPT_DIR / "top_n_effort_20260501_step1687_auc0.6804_eer0.3775.pth", "ARCFACE"),
    ("L3_ARCFACE_s1928",  CKPT_DIR / "top_n_effort_20260501_step1928_auc0.7216_eer0.3440.pth", "ARCFACE"),
    ("L3_ARCFACE_s2088",  CKPT_DIR / "melp4mol_step2088.pth", "ARCFACE"),
    ("L3_LINEAR_ep1",     CKPT_DIR / "first_best_effort_20260501_ep1_auc0.4994_eer0.4926.pth", "LINEAR"),
    ("L3_LINEAR_s1044",   CKPT_DIR / "top_n_effort_20260501_step1044_auc0.6713_eer0.3762.pth", "LINEAR"),
    ("L3_LINEAR_s1205",   CKPT_DIR / "top_n_effort_20260501_step1205_auc0.7090_eer0.3481.pth", "LINEAR"),
    ("L3_LINEAR_s1285",   CKPT_DIR / "nqvfz44v_step1285.pth", "LINEAR"),
]


def l2norm(x, axis):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def fit_substrate_lr(feats_norm, y_binary, axis_name):
    """Train an LR on normalized L3 features predicting y_binary.
    Returns (direction, in_sample_auc, n_pos, n_neg)."""
    if y_binary.sum() < 5 or (~y_binary).sum() < 5:
        return None, None, int(y_binary.sum()), int((~y_binary).sum())
    lr = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
    lr.fit(feats_norm, y_binary.astype(int))
    scores = lr.predict_proba(feats_norm)[:, 1]
    auc = float(roc_auc_score(y_binary.astype(int), scores))
    direction = lr.coef_.flatten().astype(np.float64)
    return direction, auc, int(y_binary.sum()), int((~y_binary).sum())


def cos(a, b):
    if a is None or b is None:
        return None
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(np.dot(a, b) / (na * nb))


def head_dir(ck, recipe):
    """Decision direction in normalized feature space."""
    head_w = ck["state_dict"]["head.weight"].cpu().numpy().astype(np.float64)
    if recipe == "ARCFACE":
        return l2norm(head_w[1:2], axis=1)[0] - l2norm(head_w[0:1], axis=1)[0]
    else:
        return head_w[1] - head_w[0]


def fresh_fake_real_direction(feats_norm_dev, lbl_dev):
    """Fresh-LR fake-vs-real direction on normalized dev features."""
    lr = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
    lr.fit(feats_norm_dev, lbl_dev)
    return lr.coef_.flatten().astype(np.float64)


def main():
    df = pd.read_csv(SAMPLED_CSV).iloc[:N_SAMPLES].reset_index(drop=True)
    label_int = (df["label"].astype(str) == "fake").astype(int).to_numpy()
    is_lockbox_all = (df["split"].astype(str) == "lockbox").to_numpy()
    is_dev_all = (df["split"].astype(str) == "dev").to_numpy()
    capture_mode = df["clip_capture_mode"].astype(str).to_numpy()
    method = df["method"].astype(str).to_numpy()
    identity = df["identity_key"].astype(str).to_numpy()
    face_pixel_area = df["face_pixel_area"].astype(float).to_numpy()
    sharpness = df["sharpness_laplacian"].astype(float).to_numpy()
    brightness = df["brightness_v_mean"].astype(float).to_numpy()

    cache_l3 = CACHE_DIR / f"intermediate__P8A__layer03__n{N_SAMPLES}.npz"
    blob = np.load(cache_l3)
    feats = blob["features"].astype(np.float32)
    valid_idx = blob["valid_idx"].astype(np.int64)

    lbl = label_int[valid_idx]
    is_lb = is_lockbox_all[valid_idx]
    is_dv = is_dev_all[valid_idx]
    cap = capture_mode[valid_idx]
    mth = method[valid_idx]
    ident = identity[valid_idx]
    fpa = face_pixel_area[valid_idx]
    shrp = sharpness[valid_idx]
    brt = brightness[valid_idx]

    feats_n = l2norm(feats.astype(np.float64), axis=1)
    print(f"sample: n={len(feats_n)}  dev={int(is_dv.sum())}  lockbox={int(is_lb.sum())}")
    print(f"        fakes={int(lbl.sum())}  reals={int((1-lbl).sum())}")
    print()

    # --- Define substrate axes (boolean per-frame) ---
    # Compute medians on the dev subset (avoids label leak from substrate split)
    fpa_med = float(np.nanmedian(fpa[is_dv]))
    shrp_med = float(np.nanmedian(shrp[is_dv]))
    brt_med = float(np.nanmedian(brt[is_dv]))
    axes = {
        "is_lockbox":          is_lb,
        "is_webcam":           cap == "webcam",
        "is_phone_screen":     cap == "phone_screen",
        "is_screen_any":       (cap == "screen") | (cap == "screen_recording"),
        "is_normal_photo":     cap == "normal_photo",
        "is_teams_real":       mth == "teams_real",
        "is_teams_capture":    np.array(["teams_capture" in m for m in mth]),
        "is_deeplive_enh":     mth == "deeplive_enhanced",
        "is_dor_shkedi":       np.array(["dor_shkedi" in i for i in ident]),
        "is_pc_generator":     np.array(["pc_generator" in i.lower() or "PC_Generator" in i for i in ident]),
        "is_chikara":          np.array(["chikara" in i.lower() for i in ident]),
        "face_size_above_med": fpa > fpa_med,
        "sharpness_above_med": shrp > shrp_med,
        "brightness_above_med": brt > brt_med,
    }
    print("Substrate axes (counts):")
    for name, y in axes.items():
        print(f"  {name:<20} pos={int(y.sum()):>4}  neg={int((~y).sum()):>4}")
    print()

    # --- Fit substrate-classifier LRs (each on full normalized feature set) ---
    substrate_dirs = {}
    print("Substrate-classifier in-sample AUCs (using ALL 800 frames; in-sample, not OOF):")
    for name, y in axes.items():
        d, auc, npos, nneg = fit_substrate_lr(feats_n, y, name)
        substrate_dirs[name] = d
        if d is None:
            print(f"  {name:<20} SKIP (insufficient class balance)")
        else:
            print(f"  {name:<20} AUC={auc:.4f}  ||w||={np.linalg.norm(d):.3f}  pos={npos}  neg={nneg}")
    print()

    # --- Fresh-LR fake-vs-real direction (substrate-INVARIANT reference) ---
    print("Fresh-LR fake-vs-real direction (dev only):")
    fresh_dir = fresh_fake_real_direction(feats_n[is_dv], lbl[is_dv])
    print(f"  ||fresh_dir||={np.linalg.norm(fresh_dir):.3f}")
    print()

    # --- Sanity: cosines BETWEEN substrate-classifier directions + fresh ---
    print("Inter-direction cosine (sanity matrix):")
    keys = list(substrate_dirs.keys()) + ["fresh_LR_fake_vs_real"]
    all_dirs = {**substrate_dirs, "fresh_LR_fake_vs_real": fresh_dir}
    print(f"{'':<24}", " ".join(f"{k[:14]:>14}" for k in keys))
    inter = {}
    for k1 in keys:
        row = []
        for k2 in keys:
            c = cos(all_dirs[k1], all_dirs[k2])
            row.append(c)
            inter[f"{k1}::{k2}"] = c
        print(f"{k1:<24}", " ".join(f"{c:+14.3f}" if c is not None else f"{'NA':>14}" for c in row))
    print()

    # --- For each trained head: cosine with every direction ---
    print("Trained-head decision direction alignment with substrate axes + fresh-LR:")
    print(f"{'arm':<22} {'recipe':<8}", " ".join(f"{k[:14]:>14}" for k in keys))
    rows = []
    for arm_label, ckpt_path, recipe in ARMS:
        if not ckpt_path.exists():
            print(f"  SKIP {arm_label}: ckpt not found at {ckpt_path}")
            continue
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        hd = head_dir(ck, recipe)

        cosines = {}
        for k in keys:
            cosines[k] = cos(hd, all_dirs[k])

        print(f"  {arm_label:<22} {recipe:<8}", " ".join(
            f"{cosines[k]:+14.3f}" if cosines[k] is not None else f"{'NA':>14}" for k in keys
        ))
        rows.append({
            "arm": arm_label,
            "recipe": recipe,
            "ckpt": ckpt_path.name,
            **{f"cos_{k}": cosines[k] for k in keys},
        })
    print()

    # --- Verdict ---
    print("=" * 110)
    print("VERDICT")
    print("=" * 110)
    print("Reading guide:")
    print("  cos > 0.30  : trained head decision direction is meaningfully aligned with that axis")
    print("  cos in [-0.10, 0.10] : essentially orthogonal — head learned something else along that axis")
    print("  Compare to cos(fresh_LR, axis) row above as the per-axis 'invariant baseline'")
    print()
    if rows:
        for r in rows:
            best_axis = max(
                substrate_dirs.keys(),
                key=lambda k: abs(r.get(f"cos_{k}", 0.0) or 0.0),
            )
            best_cos = r.get(f"cos_{best_axis}", 0.0)
            print(f"  {r['arm']:<22} most-aligned substrate axis: {best_axis:<20} cos={best_cos:+.3f}")
    print("=" * 110)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUTPUT_DIR / "substrate_classifier_direction_2026-05-01.json"
    out_csv = OUTPUT_DIR / "substrate_classifier_direction_2026-05-01.csv"
    with open(out_json, "w") as f:
        json.dump({
            "axes_counts": {k: {"pos": int(y.sum()), "neg": int((~y).sum())} for k, y in axes.items()},
            "inter_direction_cosines": inter,
            "trained_heads": rows,
        }, f, indent=2)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  outputs: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
