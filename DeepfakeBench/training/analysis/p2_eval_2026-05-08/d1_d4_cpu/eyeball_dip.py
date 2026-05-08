#!/usr/bin/env python3
"""Eyeball — chronic-real frames where Slot D step3000 (low-saturation) and
Slot D step19000 (high-saturation) disagree the most.

The canary fire at `_step=18000` showed a non-monotonic dip: Roy_D mean dropped
0.94 -> 0.68 and bla_bla_chow 0.91 -> 0.41 between fires. There is no saved
ckpt at exactly that state, but D5 found `slotD_periodic_step3000` is the
closest savable approximation (chronic-real activations on the LOW side of
the trajectory). `top_n_step19000` is the late-saturated peer.

For each of the 6 chronic-real identities, render the 6 frames with the
largest |score_step19000 - score_step3000| (i.e. frames where the model's
behavior CHANGED most between low-saturation and saturated states).
Annotate each frame with all 4 scores (step3000, step19000, P8A_reference,
delta). One PNG contact sheet per identity, plus a summary table CSV.

Output:
  outputs/eyeball_dip_summary.csv  — per-frame table for the surfaced frames
  figs/eyeball_<identity>.png      — contact sheet per chronic identity
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")
log = logging.getLogger("eyeball_dip")

THIS = Path(__file__).resolve().parent
SCORES = THIS / "scores"
FRAME_CACHE = THIS / "_frame_cache"
OUT = THIS / "outputs"
FIGS = THIS / "figs"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

CHRONIC_IDS = [
    "PC_Generator__s22", "PC_Generator__s45", "Q__s6", "Roy_D",
    "bla_bla_chow", "bla_bla_chow__s2",
]

# Map identity -> canary cohort name
COHORT_FOR = {
    "PC_Generator__s22": "chronic_PCGen_s22",
    "PC_Generator__s45": "chronic_PCGen_s45",
    "Q__s6": "chronic_Q_s6",
    "Roy_D": "chronic_Roy_D",
    "bla_bla_chow": "chronic_bla_bla_chow",
    "bla_bla_chow__s2": "chronic_bla_bla_chow_s2",
}

CKPTS_TO_SHOW = [
    ("step3000", "slotD_periodic_step3000.csv"),
    ("step19000", "slotD_top_n_step19000.csv"),
]
N_FRAMES_PER_ID = 6


def load_meta() -> pd.DataFrame:
    return pd.read_csv(SCORES / "_canary_meta.csv")


def load_scores() -> pd.DataFrame:
    df = load_meta().copy()
    for col, csv in CKPTS_TO_SHOW:
        s = pd.read_csv(SCORES / csv)[["frame_idx", "frame_path", "prob_fake"]]
        s = s.rename(columns={"prob_fake": col})
        df = df.merge(s, on=["frame_idx", "frame_path"], how="left")
    df["delta_19k_minus_3k"] = df["step19000"] - df["step3000"]
    return df


def render_contact_sheet(identity: str, sub: pd.DataFrame):
    cols = 3
    rows = (len(sub) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 4.0))
    axes = np.atleast_2d(axes).reshape(rows, cols)
    fig.suptitle(f"{identity} — top |Δ(step19k−step3k)| frames "
                 f"(n={len(sub)} of 50)\n"
                 f"step3000 ≈ low-saturation state (canary _step=3000 fire match)\n"
                 f"step19000 ≈ late-saturated state (canary _step=24000 fire match)",
                 fontsize=10)
    for i, (_, row) in enumerate(sub.iterrows()):
        ax = axes[i // cols, i % cols]
        bn = row["frame_path"].split("/")[-1]
        local = FRAME_CACHE / bn
        if local.exists():
            img = cv2.cvtColor(cv2.imread(str(local), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        ax.imshow(img)
        ax.axis("off")
        title = (f"frame {int(row['frame_idx'])}  Δ={row['delta_19k_minus_3k']:+.3f}\n"
                 f"step3k={row['step3000']:.3f}  step19k={row['step19000']:.3f}\n"
                 f"P8A={row['p8a_reference_score']:.3f}")
        ax.set_title(title, fontsize=8)
    # Hide extra panels
    for j in range(len(sub), rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = FIGS / f"eyeball_{COHORT_FOR[identity]}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    df = load_scores()
    log.info("merged frame table: %s rows × %s cols", *df.shape)
    summary_rows = []
    for ident in CHRONIC_IDS:
        sub = df[df["base_identity"] == ident].copy()
        sub["abs_delta"] = sub["delta_19k_minus_3k"].abs()
        sub = sub.sort_values("abs_delta", ascending=False).head(N_FRAMES_PER_ID)
        log.info("[%s] surfaced %d frames; mean(step3k)=%.3f mean(step19k)=%.3f",
                 ident, len(sub), sub["step3000"].mean(), sub["step19000"].mean())
        out = render_contact_sheet(ident, sub)
        log.info("  -> %s", out)
        for _, row in sub.iterrows():
            summary_rows.append({
                "identity": ident,
                "cohort": row["cohort"],
                "frame_idx": int(row["frame_idx"]),
                "frame_path": row["frame_path"],
                "label": int(row["label"]),
                "step3000": row["step3000"],
                "step19000": row["step19000"],
                "delta_19k_minus_3k": row["delta_19k_minus_3k"],
                "p8a_reference_score": row["p8a_reference_score"],
            })
    summary = pd.DataFrame(summary_rows)
    summary_csv = OUT / "eyeball_dip_summary.csv"
    summary.to_csv(summary_csv, index=False)
    log.info("wrote %s", summary_csv)
    print()
    print("=== Per-identity headline (top |delta| frames) ===")
    grp = summary.groupby("identity").agg(
        n=("frame_idx", "count"),
        delta_mean=("delta_19k_minus_3k", "mean"),
        delta_max=("delta_19k_minus_3k", "max"),
        delta_min=("delta_19k_minus_3k", "min"),
        step3k_mean=("step3000", "mean"),
        step19k_mean=("step19000", "mean"),
        p8a_mean=("p8a_reference_score", "mean"),
    ).reset_index()
    print(grp.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
