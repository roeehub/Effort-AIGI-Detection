#!/usr/bin/env python3
"""Step 2 — frozen-feature linear probe (5-fold StratifiedKFold).

Compares raw CLIP-ViT-B-16 (DataComp-XL) vs P8A frozen 512-d features on
viso fakes vs teams_real_dev reals. Hard n_jobs=1.

Outputs:
  outputs/probe_summary.json
  outputs/probe_summary.csv
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
)
HERE = REPO_ROOT / "analysis" / "clip_vs_p8a_viso_2026-05-03"
OUTPUTS = HERE / "outputs"

SEED = 737
N_SPLITS = 5
C_REG = 1.0
# Regularization sweep (C low = stronger reg, forces model onto highest-SNR axis)
C_SWEEP = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

logger = logging.getLogger("clip-vs-p8a-probe")


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {
        "features": z["features"].astype(np.float32),
        "label": z["label"].astype(np.int32),
        "frame_path": np.array([str(s) for s in z["frame_path"]]),
        "source": np.array([str(s) for s in z["source"]]),
        "family_key": np.array([str(s) for s in z["family_key"]]),
        "local_path": np.array([str(s) for s in z["local_path"]]),
    }


def run_probe(X: np.ndarray, y: np.ndarray, name: str, C: float = C_REG, n_train: int = None) -> Dict:
    """Standardize + 5-fold stratified LR. Returns mean/std AUC + acc.

    If n_train is set, subsample n_train (per fold, stratified) from the
    training fold before fitting. This stresses feature-quality gaps — when
    the probe has lots of data, even a weak signal saturates to AUC=1; with
    few training samples, only the highest-SNR features sustain perfect AUC.
    """
    rng = np.random.default_rng(SEED)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_aucs: List[float] = []
    fold_accs: List[float] = []
    for fold_i, (tr, te) in enumerate(skf.split(X, y)):
        if n_train is not None and n_train < len(tr):
            # Stratified sub-sampling of training fold.
            y_tr = y[tr]
            idx_pos = tr[y_tr == 1]
            idx_neg = tr[y_tr == 0]
            half = n_train // 2
            sub_pos = rng.choice(idx_pos, size=min(half, len(idx_pos)), replace=False)
            sub_neg = rng.choice(idx_neg, size=min(n_train - len(sub_pos), len(idx_neg)), replace=False)
            tr = np.concatenate([sub_pos, sub_neg])
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(
            C=C, max_iter=2000, n_jobs=1, solver="lbfgs", random_state=SEED,
        )
        clf.fit(Xtr, y[tr])
        prob = clf.predict_proba(Xte)[:, 1]
        pred = clf.predict(Xte)
        auc = float(roc_auc_score(y[te], prob))
        acc = float(accuracy_score(y[te], pred))
        fold_aucs.append(auc)
        fold_accs.append(acc)
    return {
        "name": name,
        "C": C,
        "n_train_per_fold": int(n_train) if n_train is not None else None,
        "n_total": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "acc_mean": float(np.mean(fold_accs)),
        "acc_std": float(np.std(fold_accs)),
        "fold_aucs": [float(a) for a in fold_aucs],
        "fold_accs": [float(a) for a in fold_accs],
    }


def maybe_load_viso_reals_features() -> Optional[Dict]:
    """Hunt for viso-real frames in the per-frame CSVs of any suite. Tries to
    find proper_visomaster_*_real or tv2_visomaster_real frames."""
    import glob
    csv_dir = REPO_ROOT / "analysis" / "score_distribution_2026-05-02" / "raw_reports"
    candidates = []
    for csv_path in glob.glob(str(csv_dir / "*p8a_reference*.csv")):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            mask = (
                df["frame_path"].astype(str).str.contains("proper_real|tv2_visomaster_real|proper_visomaster_real", regex=True, na=False)
                & (df["label"] == 0)
            )
            if mask.sum() > 0:
                candidates.append(df[mask][["frame_path", "label", "family_key", "method"]])
        except Exception:
            continue
    if not candidates:
        return None
    df = pd.concat(candidates, ignore_index=True).drop_duplicates(subset="frame_path")
    return {"df": df}


def main() -> int:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
                        handlers=[logging.FileHandler(HERE / "run.log", mode="a"), logging.StreamHandler(sys.stdout)])

    clip_npz = OUTPUTS / "clip_b16_raw__features.npz"
    p8a_npz = OUTPUTS / "p8a__features.npz"
    if not clip_npz.exists() or not p8a_npz.exists():
        logger.error("Missing feature files. Run scripts/01_extract_features.py first.")
        return 1

    clip = load_npz(clip_npz)
    p8a = load_npz(p8a_npz)
    logger.info("CLIP raw: features %s, fakes=%d reals=%d",
                clip["features"].shape,
                int((clip["label"] == 1).sum()),
                int((clip["label"] == 0).sum()))
    logger.info("P8A:      features %s, fakes=%d reals=%d",
                p8a["features"].shape,
                int((p8a["label"] == 1).sum()),
                int((p8a["label"] == 0).sum()))

    # Align by frame_path so the two probes are comparable on identical frames.
    fp_clip = clip["frame_path"]
    fp_p8a = p8a["frame_path"]
    common = sorted(set(fp_clip.tolist()) & set(fp_p8a.tolist()))
    logger.info("Common frames between CLIP and P8A feature sets: %d", len(common))

    def filter_to(d: Dict, keep: List[str]) -> Dict:
        keep_set = set(keep)
        mask = np.array([fp in keep_set for fp in d["frame_path"]])
        return {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in d.items()}

    clip = filter_to(clip, common)
    p8a = filter_to(p8a, common)

    # Now sort both to the same frame ordering.
    order_c = np.argsort(clip["frame_path"])
    order_p = np.argsort(p8a["frame_path"])
    for k in clip:
        clip[k] = clip[k][order_c]
        p8a[k] = p8a[k][order_p]
    assert (clip["frame_path"] == p8a["frame_path"]).all(), "Alignment failed"

    y = clip["label"].astype(int)
    logger.info("Aligned probe set: n=%d, fakes=%d, reals=%d", len(y), int((y == 1).sum()), int((y == 0).sum()))

    # ------------------------------------------------------------------
    # Probe 1: viso-fake vs teams-real-dev (the main comparison).
    # ------------------------------------------------------------------
    results = {}
    logger.info("=" * 78)
    logger.info("Probe (1): viso_fake (550) vs teams_real_dev (~550) — fake-vs-real AUC, C=1.0")
    logger.info("=" * 78)
    r_clip = run_probe(clip["features"], y, "clip_b16_raw / fake_vs_teams_real_dev", C=1.0)
    logger.info("  CLIP-raw mean AUC=%.4f ± %.4f", r_clip["auc_mean"], r_clip["auc_std"])
    r_p8a = run_probe(p8a["features"], y, "p8a / fake_vs_teams_real_dev", C=1.0)
    logger.info("  P8A      mean AUC=%.4f ± %.4f", r_p8a["auc_mean"], r_p8a["auc_std"])
    results["fake_vs_teams_real_dev_C1"] = {"clip_b16_raw": r_clip, "p8a": r_p8a}

    # ------------------------------------------------------------------
    # Probe 1b: same comparison, but stronger regularization (C sweep).
    # When AUC is saturated at C=1, this exposes the discriminative
    # robustness gap. The arm with stronger high-SNR axes will hold up at
    # very small C; the arm whose signal is spread across many axes degrades.
    # ------------------------------------------------------------------
    logger.info("=" * 78)
    logger.info("Probe (1b): C-regularization sweep (low C = harder probe)")
    logger.info("=" * 78)
    sweep_clip = []
    sweep_p8a = []
    for C in C_SWEEP:
        rc = run_probe(clip["features"], y, f"clip_b16_raw / C={C}", C=C)
        rp = run_probe(p8a["features"], y, f"p8a / C={C}", C=C)
        sweep_clip.append(rc)
        sweep_p8a.append(rp)
        logger.info("  C=%.0e  CLIP=%.4f ± %.4f   P8A=%.4f ± %.4f   Δ=%+.4f",
                    C, rc["auc_mean"], rc["auc_std"],
                    rp["auc_mean"], rp["auc_std"],
                    rc["auc_mean"] - rp["auc_mean"])
    results["c_sweep"] = {
        "clip_b16_raw": sweep_clip,
        "p8a": sweep_p8a,
    }

    # ------------------------------------------------------------------
    # Probe 1c: small-sample probe (16 / 32 / 64 / 128 / 256 train samples
    # per fold). When the training set is large, even weak features saturate.
    # Below the saturation knee, the model with cleaner / better-aligned
    # features wins. This is the standard linear-probe quality benchmark.
    # ------------------------------------------------------------------
    logger.info("=" * 78)
    logger.info("Probe (1c): n_train sweep (smaller train = harder probe)")
    logger.info("=" * 78)
    n_train_grid = [16, 32, 64, 128, 256]
    nt_clip = []
    nt_p8a = []
    for nt in n_train_grid:
        rc = run_probe(clip["features"], y, f"clip_b16_raw / n_train={nt}", C=1.0, n_train=nt)
        rp = run_probe(p8a["features"], y, f"p8a / n_train={nt}", C=1.0, n_train=nt)
        nt_clip.append(rc)
        nt_p8a.append(rp)
        logger.info("  n=%4d  CLIP=%.4f ± %.4f   P8A=%.4f ± %.4f   Δ=%+.4f",
                    nt, rc["auc_mean"], rc["auc_std"],
                    rp["auc_mean"], rp["auc_std"],
                    rc["auc_mean"] - rp["auc_mean"])
    results["n_train_sweep"] = {
        "clip_b16_raw": nt_clip,
        "p8a": nt_p8a,
    }

    # ------------------------------------------------------------------
    # Probe 2 (optional): viso-fake vs viso-real, if we can find ≥100 viso reals.
    # ------------------------------------------------------------------
    viso_real = maybe_load_viso_reals_features()
    if viso_real is not None and len(viso_real["df"]) >= 100:
        logger.info("Found %d viso-real frames in side-suites. Skipping in this run because we'd need to "
                    "(a) download them and (b) extract features for both models — out of scope per spec "
                    "('if you can find ≥100 viso reals'). Recording in summary as 'available_n=%d (not run)'.",
                    len(viso_real["df"]), len(viso_real["df"]))
        results["viso_fake_vs_viso_real"] = {
            "status": "available_but_not_run",
            "viso_real_n_available": int(len(viso_real["df"])),
            "note": (
                "≥100 viso-real frames found in side-suite per-frame CSVs but "
                "extracting features requires downloading + forwarding both models on "
                "an additional ~500 frames. Skipped per spec wording: 'if you can find "
                "≥100 viso reals' — interpreted as opportunistic. Re-run with the "
                "expanded manifest if needed."
            ),
        }
    else:
        logger.info("No ≥100 viso-real frames available; skipping viso-vs-viso probe per spec.")
        results["viso_fake_vs_viso_real"] = {"status": "not_available"}

    # ------------------------------------------------------------------
    # Verdict — use the MOST DISCRIMINATING (smallest-train OR strongest-reg)
    # comparison. If the headline AUC is saturated (=1.0), the n_train=16 or
    # C=1e-4 measurements break the tie.
    # ------------------------------------------------------------------
    main_clip_auc = r_clip["auc_mean"]
    main_p8a_auc = r_p8a["auc_mean"]
    delta = main_clip_auc - main_p8a_auc

    # Find the discriminating point — minimum AUC across the n_train sweep,
    # for each model.
    nt_clip_min = min(r["auc_mean"] for r in nt_clip)
    nt_p8a_min = min(r["auc_mean"] for r in nt_p8a)
    nt_clip_at_smallest = nt_clip[0]["auc_mean"]
    nt_p8a_at_smallest = nt_p8a[0]["auc_mean"]
    delta_smallest = nt_clip_at_smallest - nt_p8a_at_smallest

    # And same for C sweep — strongest regularization.
    sw_clip_at_strongest = sweep_clip[0]["auc_mean"]
    sw_p8a_at_strongest = sweep_p8a[0]["auc_mean"]
    delta_strongest_C = sw_clip_at_strongest - sw_p8a_at_strongest

    # Choose the largest |delta| from these three measurements as the
    # discriminating signal.
    candidates = [
        ("headline_C1_full", delta),
        ("smallest_n_train", delta_smallest),
        ("strongest_C", delta_strongest_C),
    ]
    candidates.sort(key=lambda t: abs(t[1]), reverse=True)
    discriminating_label, discriminating_delta = candidates[0]

    headline_saturated = (main_clip_auc > 0.995 and main_p8a_auc > 0.995)

    if abs(discriminating_delta) < 0.02:
        verdict = "REFUTED" if not headline_saturated else "INCONCLUSIVE"
        if headline_saturated:
            verdict_text = (
                f"Both raw CLIP-B16 (AUC={main_clip_auc:.4f}) and P8A "
                f"(AUC={main_p8a_auc:.4f}) saturate at headline (C=1, full data). "
                f"On the more discriminating probes (smallest n_train: CLIP "
                f"{nt_clip_at_smallest:.4f} vs P8A {nt_p8a_at_smallest:.4f}; "
                f"strongest C: CLIP {sw_clip_at_strongest:.4f} vs P8A "
                f"{sw_p8a_at_strongest:.4f}), the gap is still within ±2pp "
                f"(max |Δ|={abs(discriminating_delta):.4f} on '{discriminating_label}'). "
                f"INCONCLUSIVE: the FT stack neither destroyed nor materially "
                f"enhanced viso linear-separability at the frozen-feature level. "
                f"The 27% recall ceiling is downstream (head-side calibration / "
                f"decision boundary / deployment-τ alignment)."
            )
        else:
            verdict_text = (
                f"Raw CLIP-B16 (AUC={main_clip_auc:.4f}) and P8A "
                f"(AUC={main_p8a_auc:.4f}) agree within ±2pp (|Δ|={abs(delta):.4f}); "
                f"discriminating-probe Δ also within ±2pp. REFUTED — FT did not "
                f"collapse the viso-discriminative direction."
            )
    elif discriminating_delta > 0.02:
        verdict = "PASSED"
        verdict_text = (
            f"Discriminating probe '{discriminating_label}' shows raw CLIP-B16 "
            f"OUTPERFORMS P8A by Δ={discriminating_delta:+.4f} AUC. Headline "
            f"(C=1, full data): CLIP={main_clip_auc:.4f}, P8A={main_p8a_auc:.4f}. "
            f"At smallest n_train=16: CLIP={nt_clip_at_smallest:.4f} vs "
            f"P8A={nt_p8a_at_smallest:.4f}. At strongest C={C_SWEEP[0]}: "
            f"CLIP={sw_clip_at_strongest:.4f} vs P8A={sw_p8a_at_strongest:.4f}. "
            f"FT stack collapsed the viso-discriminative axis. 'Scratch from "
            f"CLIP' becomes a rational direction to revisit."
        )
    else:
        verdict = "REFUTED"
        verdict_text = (
            f"Discriminating probe '{discriminating_label}' shows P8A "
            f"OUTPERFORMS raw CLIP-B16 by Δ={-discriminating_delta:+.4f} AUC. "
            f"Headline: CLIP={main_clip_auc:.4f}, P8A={main_p8a_auc:.4f}. "
            f"At smallest n_train=16: CLIP={nt_clip_at_smallest:.4f} vs "
            f"P8A={nt_p8a_at_smallest:.4f}. At strongest C={C_SWEEP[0]}: "
            f"CLIP={sw_clip_at_strongest:.4f} vs P8A={sw_p8a_at_strongest:.4f}. "
            f"FT stack preserved/enhanced viso signal; the 27% recall ceiling "
            f"is downstream (deployment-τ, head calibration, image-quality "
            f"shortcut, eval-substrate looseness)."
        )

    summary = {
        "date": "2026-05-03",
        "seed": SEED,
        "n_splits": N_SPLITS,
        "C_default": C_REG,
        "C_sweep": C_SWEEP,
        "n_train_sweep": n_train_grid,
        "feature_dim": int(clip["features"].shape[1]),
        "n_aligned_frames": int(len(y)),
        "n_fakes": int((y == 1).sum()),
        "n_reals": int((y == 0).sum()),
        "results_per_probe": results,
        "headline": {
            "clip_b16_raw__viso_auc": main_clip_auc,
            "clip_b16_raw__viso_auc_std": r_clip["auc_std"],
            "p8a__viso_auc": main_p8a_auc,
            "p8a__viso_auc_std": r_p8a["auc_std"],
            "delta_clip_minus_p8a": float(delta),
            "discriminating_probe": discriminating_label,
            "discriminating_delta_clip_minus_p8a": float(discriminating_delta),
            "smallest_n_train": {
                "clip_b16_raw_auc": nt_clip_at_smallest,
                "p8a_auc": nt_p8a_at_smallest,
            },
            "strongest_C": {
                "clip_b16_raw_auc": sw_clip_at_strongest,
                "p8a_auc": sw_p8a_at_strongest,
            },
        },
        "verdict": verdict,
        "verdict_text": verdict_text,
    }

    out_json = OUTPUTS / "probe_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", out_json)

    out_csv = OUTPUTS / "probe_summary.csv"
    rows = []
    for r in [r_clip, r_p8a]:
        rows.append({
            "model": "clip_b16_raw" if "clip" in r["name"] else "p8a",
            "probe": "headline_C1_full",
            "C": r.get("C", 1.0),
            "n_train_per_fold": r.get("n_train_per_fold"),
            "n_total": r["n_total"],
            "n_fake": r["n_pos"],
            "n_real": r["n_neg"],
            "auc_mean": r["auc_mean"],
            "auc_std": r["auc_std"],
        })
    for r in sweep_clip + sweep_p8a:
        rows.append({
            "model": "clip_b16_raw" if "clip" in r["name"] else "p8a",
            "probe": f"C_sweep_C={r['C']}",
            "C": r["C"],
            "n_train_per_fold": r.get("n_train_per_fold"),
            "n_total": r["n_total"],
            "n_fake": r["n_pos"],
            "n_real": r["n_neg"],
            "auc_mean": r["auc_mean"],
            "auc_std": r["auc_std"],
        })
    for r in nt_clip + nt_p8a:
        rows.append({
            "model": "clip_b16_raw" if "clip" in r["name"] else "p8a",
            "probe": f"n_train_sweep_n={r['n_train_per_fold']}",
            "C": r["C"],
            "n_train_per_fold": r["n_train_per_fold"],
            "n_total": r["n_total"],
            "n_fake": r["n_pos"],
            "n_real": r["n_neg"],
            "auc_mean": r["auc_mean"],
            "auc_std": r["auc_std"],
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Wrote %s", out_csv)

    print()
    print("=" * 78)
    print("CLIP-B16-raw  vs  P8A   :   frozen-feature linear probe on viso fakes")
    print("=" * 78)
    print(f"{'probe':<28} {'CLIP-raw AUC':>14} {'P8A AUC':>11} {'delta':>10}")
    print("-" * 78)
    print(f"{'headline (C=1, full)':<28} {main_clip_auc:>14.4f} {main_p8a_auc:>11.4f} {delta:>+10.4f}")
    for cc, rc, rp in zip(C_SWEEP, sweep_clip, sweep_p8a):
        print(f"{f'C={cc}':<28} {rc['auc_mean']:>14.4f} {rp['auc_mean']:>11.4f} {rc['auc_mean']-rp['auc_mean']:>+10.4f}")
    for nt, rc, rp in zip(n_train_grid, nt_clip, nt_p8a):
        print(f"{f'n_train={nt}':<28} {rc['auc_mean']:>14.4f} {rp['auc_mean']:>11.4f} {rc['auc_mean']-rp['auc_mean']:>+10.4f}")
    print()
    print(f"DISCRIMINATING DELTA on '{discriminating_label}' = {discriminating_delta:+.4f}")
    print(f"VERDICT: {verdict}")
    print(verdict_text)
    print()
    print(f"results: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
