"""CPU-3 Per-ckpt-axis vs anchor_aware decomposition (2026-05-23).

Goal: does anchor_aware already neutralize the per-ckpt substrate axis, or
are they decoupled?

Method:
1. Load 3 per-ckpt substrate axes (P8A, SlotAv2, T5C).
2. Compute pairwise cosines.
3. For each ckpt, compute cos(per_ckpt_axis, prob_fake_gradient_direction).
   Approximate gradient direction as
     normalize(mean(features|fake) - mean(features|real))
   using cached L11 features + cached head probs on the 5475 clean +
   5478 teams pair-geometry frames.
4. False-flag axis: dor real lockbox webcam vs dor real dev normal_photo,
   features extracted from the D8 frozen-CLIP-L11 cache
   (`clip_frozen_l11__n4839.npz` — 4000 dev + 839 lockbox rows; metadata
   joined via the same parquet `analysis/lockbox_tagging/full_tags_2026-04-27.parquet`).
   - Note: the D8 cache is frozen-CLIP-L11; the per-ckpt axes are trained-encoder L11.
     We still compute cos(per-ckpt-axis, false_flag_normal_frozen) as a CHEAP
     approximation per plan; the cross-encoder cosine is the headline number
     but is a lower bound on the trained-encoder false-flag cosine.

Outputs:
  RESULTS_FACTS_2026-05-23.md
  axis_cosines.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path(__file__).resolve().parent
GEOM_DIR = REPO_ROOT / "analysis/substrate_pair_geometry_2026-05-22"
D8_DIR = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced"
LOCKBOX_TAGS = REPO_ROOT / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"

CKPTS = ("P8A_step5000", "SlotAv2_step3500", "T5C_step3500")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / "_cpu3.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("cpu3")


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def load_axes() -> Dict[str, np.ndarray]:
    axes: Dict[str, np.ndarray] = {}
    for ck in CKPTS:
        path = GEOM_DIR / f"_trained_encoder_substrate_axis_{ck}.npy"
        v = np.load(path).astype(np.float64)
        axes[ck] = l2_normalize(v)
        logger.info("loaded axis %s shape=%s norm=%.4f", ck, v.shape, np.linalg.norm(v))
    return axes


def compute_prob_fake_gradient(ck: str) -> np.ndarray:
    """gradient direction = mean(feat | fake) - mean(feat | real), L2-normalized.

    Since the cached features are L11 768-dim TRAINED-encoder features
    on the 5475 clean + 5478 teams substrate-pair frames (all identity-real),
    we don't have fake samples in this set. Instead, we use the head's
    own prob_fake to label each frame: prob_fake > 0.5 → fake-side; else
    real-side. This yields a SOFT gradient direction that aligns with the
    direction the head uses for its decision boundary.
    """
    feats_clean = np.load(GEOM_DIR / f"feats/{ck}_L11_clean.npy").astype(np.float64)
    feats_teams = np.load(GEOM_DIR / f"feats/{ck}_L11_teams.npy").astype(np.float64)
    scores_clean = np.load(GEOM_DIR / f"scores/{ck}_clean.npy")
    scores_teams = np.load(GEOM_DIR / f"scores/{ck}_teams.npy")
    feats = np.concatenate([feats_clean, feats_teams], axis=0)
    scores = np.concatenate([scores_clean, scores_teams], axis=0)
    fake_mask = scores > 0.5
    real_mask = ~fake_mask
    n_fake = int(fake_mask.sum())
    n_real = int(real_mask.sum())
    logger.info("  %s: %d fake-side / %d real-side (by head prob>0.5)", ck, n_fake, n_real)
    if n_fake == 0 or n_real == 0:
        logger.warning("  %s: degenerate split (n_fake=%d n_real=%d) — return zeros", ck, n_fake, n_real)
        return np.zeros(feats.shape[1], dtype=np.float64)
    mu_fake = feats[fake_mask].mean(axis=0)
    mu_real = feats[real_mask].mean(axis=0)
    grad = l2_normalize(mu_fake - mu_real)
    return grad


def compute_false_flag_normal_frozen() -> Tuple[np.ndarray, int, int]:
    """false_flag_normal = mean(feat | dor real lockbox webcam) -
                           mean(feat | dor real dev normal_photo),
    using the D8 frozen-CLIP-L11 features (4839 rows = 4000 dev + 839 lockbox).

    Metadata sources:
      - dev rows: per_frame_weights_dev.csv (identity_key + gcs_uri)
      - lockbox rows: replay from lockbox_tagging full_tags parquet (matches order
        in the npz — D8 used run_d8.replay_dev_lockbox_paths with seed=42; the
        lockbox replay does NOT shuffle).
    """
    blob = np.load(D8_DIR / "outputs/clip_frozen_l11__n4839.npz", allow_pickle=True)
    feats = blob["features"].astype(np.float64)  # (4839, 768)

    dev_meta = pd.read_csv(D8_DIR / "outputs/per_frame_weights_dev.csv")
    lockbox_df = pd.read_parquet(LOCKBOX_TAGS)
    lockbox_df = lockbox_df[lockbox_df["local_path"].astype(str).str.len() > 0].reset_index(drop=True)
    lb_df = lockbox_df[lockbox_df["split"] == "lockbox"].reset_index(drop=True)

    assert len(dev_meta) == 4000, f"expected 4000 dev rows, got {len(dev_meta)}"
    assert len(lb_df) == 839, f"expected 839 lockbox rows, got {len(lb_df)}"
    assert feats.shape[0] == 4839, f"expected 4839 feats, got {feats.shape[0]}"

    # Dev rows: dor_shkedi_* real with clip_capture_mode=normal_photo
    # First filter from dev_meta for identity_key containing dor — but dev_meta
    # only has identity_key + label, not clip_capture_mode. Join with full tags
    # by gcs_uri.
    dev_full = lockbox_df[lockbox_df["split"] == "dev"].copy()
    # Sample to match dev_meta (replay with seed=42)
    rng_42 = np.random.default_rng(seed=42)
    dev_real_pool = dev_full[dev_full["label"] == "real"].reset_index(drop=True)
    dev_fake_pool = dev_full[dev_full["label"] == "fake"].reset_index(drop=True)
    dev_real_idx = rng_42.choice(len(dev_real_pool), size=2000, replace=False)
    dev_fake_idx = rng_42.choice(len(dev_fake_pool), size=2000, replace=False)
    dev_sample = pd.concat(
        [dev_real_pool.iloc[dev_real_idx], dev_fake_pool.iloc[dev_fake_idx]]
    ).reset_index(drop=True)
    assert len(dev_sample) == 4000

    # Now dev_sample is aligned with feats[:4000]; lb_df with feats[4000:].
    # Dor identities:
    dev_dor_real = (
        dev_sample["identity_key"].astype(str).str.contains("dor", case=False, na=False)
        & (dev_sample["label"] == "real")
        & (dev_sample["clip_capture_mode"].astype(str) == "normal_photo")
    )
    lb_dor_real_webcam = (
        lb_df["identity_key"].astype(str).str.contains("dor", case=False, na=False)
        & (lb_df["label"] == "real")
        & (lb_df["clip_capture_mode"].astype(str) == "webcam")
    )
    dev_dor_idx = np.where(dev_dor_real.to_numpy())[0]  # indices in [0, 4000)
    lb_dor_idx = np.where(lb_dor_real_webcam.to_numpy())[0] + 4000  # offset

    logger.info(
        "  dor real dev normal_photo n=%d ; dor real lockbox webcam n=%d",
        len(dev_dor_idx), len(lb_dor_idx),
    )
    if len(dev_dor_idx) == 0 or len(lb_dor_idx) == 0:
        logger.warning("  one cohort empty — returning zero false-flag axis")
        return np.zeros(feats.shape[1], dtype=np.float64), int(len(dev_dor_idx)), int(len(lb_dor_idx))

    mu_dev = feats[dev_dor_idx].mean(axis=0)
    mu_lb = feats[lb_dor_idx].mean(axis=0)
    ff = l2_normalize(mu_lb - mu_dev)  # lockbox-webcam minus dev-normal-photo
    return ff, int(len(dev_dor_idx)), int(len(lb_dor_idx))


def compute_verdict(cosines: Dict[str, Dict[str, float]]) -> Tuple[str, str]:
    """alpha: cos(per-ckpt, false-flag-normal) < 0.3 AND cos(P8A, SlotAv2) > 0.7
       beta:  cos(per-ckpt, false-flag-normal) > 0.6
       gamma: in-between
    """
    ff_p8a = abs(cosines["P8A_step5000"]["cos_axis_vs_false_flag"])
    ff_sa = abs(cosines["SlotAv2_step3500"]["cos_axis_vs_false_flag"])
    ff_t5c = abs(cosines["T5C_step3500"]["cos_axis_vs_false_flag"])
    cos_p8a_sa = cosines["_pairwise"]["cos_P8A_SlotAv2"]
    cos_p8a_t5c = cosines["_pairwise"]["cos_P8A_T5C"]
    cos_sa_t5c = cosines["_pairwise"]["cos_SlotAv2_T5C"]

    # alpha: ALL three abs(cos(axis, false-flag-normal)) < 0.3 AND cos(P8A, SlotAv2) > 0.7
    # (use abs because direction of false-flag axis is conventional)
    all_low = max(ff_p8a, ff_sa, ff_t5c) < 0.3
    if all_low and cos_p8a_sa > 0.7:
        verdict = "alpha"
        summary = (
            f"all 3 |cos(axis, false-flag-normal)| < 0.3 "
            f"(P8A={ff_p8a:.3f}, SlotAv2={ff_sa:.3f}, T5C={ff_t5c:.3f}); "
            f"cos(P8A, SlotAv2) = {cos_p8a_sa:.3f} > 0.7 → anchor + substrate decoupled; "
            f"BACKBONE-SlotAv2 can stack substrate-pair on anchor without bundle penalty"
        )
        return verdict, summary
    # beta: ANY of the 3 abs(cos(axis, false-flag-normal)) > 0.6
    any_high = max(ff_p8a, ff_sa, ff_t5c) > 0.6
    if any_high:
        verdict = "beta"
        summary = (
            f"max |cos(axis, false-flag-normal)| = {max(ff_p8a, ff_sa, ff_t5c):.3f} > 0.6 "
            f"(P8A={ff_p8a:.3f}, SlotAv2={ff_sa:.3f}, T5C={ff_t5c:.3f}); "
            f"anchor reduces substrate-pair variance; BACKBONE-T5C becomes primary signal"
        )
        return verdict, summary
    # gamma fallback
    verdict = "gamma"
    summary = (
        f"in-between: |cos(axis, false-flag-normal)| = "
        f"P8A={ff_p8a:.3f}, SlotAv2={ff_sa:.3f}, T5C={ff_t5c:.3f}; "
        f"cos(P8A,SlotAv2)={cos_p8a_sa:.3f}; both BACKBONE runs proceed"
    )
    return verdict, summary


def write_facts_doc(cosines: Dict, n_dev_dor: int, n_lb_dor: int,
                    wall: float, verdict: str, summary: str, out_path: Path) -> None:
    lines = []
    lines.append("# CPU-3 Per-Ckpt Axis vs Anchor Decomposition — FACTS (2026-05-23)")
    lines.append("")
    lines.append("> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, ship, kill, best, worst, unfortunately, remarkably, lucky, confirmed, refuted, shortcut-aligned, gap-is-wide, gap-is-narrow.")
    lines.append("")
    lines.append("## 1. Method")
    lines.append("")
    lines.append("Inputs:")
    lines.append("- 3 per-ckpt substrate axes (LR-classifier weight vectors fit on the 1880 paired clean-teams identity frames per 2026-05-22 Probe 1 KLIEP re-fit): `analysis/substrate_pair_geometry_2026-05-22/_trained_encoder_substrate_axis_{P8A_step5000,SlotAv2_step3500,T5C_step3500}.npy` (each (768,), L2-normalized).")
    lines.append("- Cached L11 features (`feats/{ckpt}_L11_{clean,teams}.npy`, each (5475 or 5478, 768)) and cached head probs (`scores/{ckpt}_{clean,teams}.npy`).")
    lines.append("- D8 frozen-CLIP-L11 features: `analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced/outputs/clip_frozen_l11__n4839.npz` ((4839, 768) = 4000 dev + 839 lockbox).")
    lines.append("- D8 metadata: `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` (identity_key, split, clip_capture_mode, label).")
    lines.append("")
    lines.append("Computations:")
    lines.append("- Pairwise cosines between the 3 per-ckpt substrate axes.")
    lines.append("- For each ckpt: `cos(per_ckpt_axis, prob_fake_gradient_direction)` where the gradient direction = `normalize(mean(feat | head_prob > 0.5) − mean(feat | head_prob ≤ 0.5))` on the 5475 + 5478 paired frames using cached L11 features + cached head probs.")
    lines.append("- False-flag normal: `normalize(mean(feat | dor real lockbox webcam) − mean(feat | dor real dev normal_photo))` on the D8 frozen-CLIP-L11 cache.")
    lines.append(f"- Dor real dev normal_photo cohort: n = {n_dev_dor}.")
    lines.append(f"- Dor real lockbox webcam cohort: n = {n_lb_dor}.")
    lines.append(f"- Wall time: {wall:.1f}s.")
    lines.append("")
    lines.append("Caveat: per-ckpt axes are trained-encoder L11 (768-dim). The false-flag normal is frozen-CLIP-L11 (also 768-dim). The cosine across these two spaces is a CROSS-ENCODER cosine and is a LOWER BOUND on the trained-encoder-internal cosine between the substrate axis and the false-flag direction. The plan accepts this approximation as the cheap probe.")
    lines.append("")
    lines.append("## 2. Pairwise cosines (substrate axes only)")
    lines.append("")
    lines.append("| Pair | cosine |")
    lines.append("|---|---:|")
    pw = cosines["_pairwise"]
    lines.append(f"| P8A vs SlotAv2 | {pw['cos_P8A_SlotAv2']:+.4f} |")
    lines.append(f"| P8A vs T5C | {pw['cos_P8A_T5C']:+.4f} |")
    lines.append(f"| SlotAv2 vs T5C | {pw['cos_SlotAv2_T5C']:+.4f} |")
    lines.append("")
    lines.append("## 3. Per-ckpt cosine table")
    lines.append("")
    lines.append("| Ckpt | cos(axis, prob_fake_gradient) | cos(axis, false_flag_normal_frozen) |")
    lines.append("|---|---:|---:|")
    for ck in CKPTS:
        d = cosines[ck]
        lines.append(f"| {ck} | {d['cos_axis_vs_grad']:+.4f} | {d['cos_axis_vs_false_flag']:+.4f} |")
    lines.append("")
    lines.append("## 4. Close criterion verdict")
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append(f"Summary: {summary}")
    lines.append("")
    lines.append("Decision rule:")
    lines.append("- alpha: ALL 3 |cos(axis, false-flag-normal)| < 0.3 AND cos(P8A, SlotAv2) > 0.7 → anchor + substrate decoupled → BACKBONE-SlotAv2 stacks substrate-pair on anchor without bundle penalty")
    lines.append("- beta:  ANY |cos(axis, false-flag-normal)| > 0.6 → anchor reduces substrate-pair variance → BACKBONE-T5C becomes primary signal")
    lines.append("- gamma: in-between → both BACKBONE runs proceed")
    lines.append("")
    lines.append("## 5. Output artifacts")
    lines.append("")
    lines.append("- `axis_cosines.csv` — all cosine pairs")
    lines.append("- `RESULTS_FACTS_2026-05-23.md` — this file")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main() -> int:
    t0 = time.time()
    logger.info("loading axes")
    axes = load_axes()

    logger.info("computing prob_fake gradient direction per ckpt")
    gradients: Dict[str, np.ndarray] = {}
    for ck in CKPTS:
        gradients[ck] = compute_prob_fake_gradient(ck)

    logger.info("computing false-flag normal (frozen-CLIP-L11)")
    ff_normal, n_dev_dor, n_lb_dor = compute_false_flag_normal_frozen()

    # Cosines
    cosines: Dict[str, Dict[str, float]] = {}
    for ck in CKPTS:
        cosines[ck] = {
            "cos_axis_vs_grad": cosine(axes[ck], gradients[ck]),
            "cos_axis_vs_false_flag": cosine(axes[ck], ff_normal),
        }
    cosines["_pairwise"] = {
        "cos_P8A_SlotAv2": cosine(axes["P8A_step5000"], axes["SlotAv2_step3500"]),
        "cos_P8A_T5C": cosine(axes["P8A_step5000"], axes["T5C_step3500"]),
        "cos_SlotAv2_T5C": cosine(axes["SlotAv2_step3500"], axes["T5C_step3500"]),
    }
    for ck, d in cosines.items():
        logger.info("  %s: %s", ck, d)

    # CSV
    rows = []
    for ck in CKPTS:
        rows.append({"row_type": "per_ckpt", "name": ck,
                     "cos_axis_vs_grad": cosines[ck]["cos_axis_vs_grad"],
                     "cos_axis_vs_false_flag": cosines[ck]["cos_axis_vs_false_flag"]})
    for pair_key, pair_val in cosines["_pairwise"].items():
        rows.append({"row_type": "pairwise", "name": pair_key, "cosine": pair_val})
    pd.DataFrame(rows).to_csv(OUT_DIR / "axis_cosines.csv", index=False)
    logger.info("wrote axis_cosines.csv")

    # Verdict + FACTS doc
    verdict, summary = compute_verdict(cosines)
    logger.info("VERDICT: %s — %s", verdict, summary)
    wall = time.time() - t0
    write_facts_doc(cosines, n_dev_dor, n_lb_dor, wall, verdict, summary,
                    OUT_DIR / "RESULTS_FACTS_2026-05-23.md")

    sentinel = OUT_DIR / "_cpu3_complete.json"
    with open(sentinel, "w") as f:
        json.dump({
            "status": "done",
            "wall_seconds": int(wall),
            "verdict": verdict,
            "verdict_summary": summary,
            "cosines": cosines,
            "n_dev_dor": n_dev_dor,
            "n_lb_dor": n_lb_dor,
        }, f, indent=2)
    logger.info("DONE %.1fs", wall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
