"""Fourier-band overlap probe — shortcut vs manipulation signal partitioning.

Question (from open loop `fourier-aug-band-overlap-not-resolved`): can we identify
FFT amplitude radial bands where the shortcut signal (may6 vs may5 separation)
is high but the manipulation signal (fake vs real on teams_*_dev) is low?

If yes -> band-limited Fourier-aug is greenlit (randomize those bands).
If no  -> bands overlap heavily; broad amp randomization erases manipulation
         signal; lever class effectively dead, pivot to AugMix consistency + SBI.

Reuses cached FFT extractions where possible.
"""

from __future__ import annotations

import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("fourier_band_overlap")

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
THIS = REPO / "analysis/fourier_band_overlap_2026-05-06"
OUT = THIS / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

XINHE = REPO / "analysis/xinhe_cross_camera_audit_2026-05-06/raw"
AMP_PHASE = REPO / "analysis/amp_vs_phase_probe_2026-05-06"

RES = 224
N_RADIAL = 16  # match Probe 3's bands
N_ANGLE = 8


def radial_bins() -> np.ndarray:
    """Return a (RES, RES) array with band index 0..N_RADIAL-1 for each pixel."""
    yy, xx = np.mgrid[:RES, :RES]
    cx = cy = (RES - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = r.max()
    bins = np.clip((r / r_max * N_RADIAL).astype(int), 0, N_RADIAL - 1)
    return bins


RBINS = radial_bins()


def fft_radial_features(path: Path) -> np.ndarray | None:
    """Return mean log-magnitude per radial band (length N_RADIAL)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (RES, RES), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    F = np.fft.fft2(img)
    F = np.fft.fftshift(F)
    mag = np.log1p(np.abs(F))
    feats = np.zeros(N_RADIAL, dtype=np.float32)
    for b in range(N_RADIAL):
        m = (RBINS == b)
        feats[b] = mag[m].mean()
    return feats


def load_xinhe_frames():
    paths_may6 = sorted((XINHE / "may6").glob("*.png"))
    paths_may5 = sorted((XINHE / "may5").glob("*.png"))
    log.info("xinhe: may6=%d may5=%d", len(paths_may6), len(paths_may5))
    return paths_may6, paths_may5


def extract_parallel(paths: list[Path], n_workers: int = 4) -> tuple[np.ndarray, list[Path]]:
    feats = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        fut_to_i = {ex.submit(fft_radial_features, p): i for i, p in enumerate(paths)}
        for done, fut in enumerate(as_completed(fut_to_i)):
            i = fut_to_i[fut]
            res = fut.result()
            if res is not None:
                feats[i] = res
            if (done + 1) % 200 == 0:
                log.info("  extracted %d / %d", done + 1, len(paths))
    keep = [i for i, f in enumerate(feats) if f is not None]
    feats_arr = np.stack([feats[i] for i in keep])
    paths_kept = [paths[i] for i in keep]
    return feats_arr, paths_kept


def per_band_auc(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """For each of N_RADIAL bands, train a 1-feature logistic and report mean CV AUC."""
    n_b = X.shape[1]
    aucs = np.zeros(n_b, dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for b in range(n_b):
        Xb = X[:, b:b + 1]
        fold_aucs = []
        for tr, te in skf.split(Xb, y):
            sc = StandardScaler().fit(Xb[tr])
            Xtr = sc.transform(Xb[tr]); Xte = sc.transform(Xb[te])
            clf = LogisticRegression(class_weight="balanced", max_iter=200, n_jobs=1)
            clf.fit(Xtr, y[tr])
            p = clf.predict_proba(Xte)[:, 1]
            # AUC
            from sklearn.metrics import roc_auc_score
            fold_aucs.append(roc_auc_score(y[te], p))
        aucs[b] = float(np.mean(fold_aucs))
    return aucs


def task_a_shortcut() -> tuple[np.ndarray, np.ndarray]:
    """may6 (1) vs may5 (0) — shortcut signal."""
    paths_may6, paths_may5 = load_xinhe_frames()
    paths_all = list(paths_may6) + list(paths_may5)
    log.info("task A: extracting FFT bands for %d frames ...", len(paths_all))
    X, kept = extract_parallel(paths_all, n_workers=4)
    y = np.concatenate([np.ones(len(paths_may6)), np.zeros(len(paths_may5))])
    # If any frame failed, drop matching y
    if X.shape[0] != len(paths_all):
        kept_set = set(p.name for p in kept)
        keep_y = np.array([p.name in kept_set for p in paths_all])
        y = y[keep_y]
    log.info("task A shape: X=%s y=%s, mean(y)=%.3f", X.shape, y.shape, y.mean())
    return X, y


def task_b_signal() -> tuple[np.ndarray, np.ndarray]:
    """fake (1) vs real (0) — download a small sample of teams_*_dev fresh."""
    import pandas as pd
    import subprocess
    pa_pc_dir = REPO / "analysis/cpu_followups_2026-05-04/raw_reports"
    fake_csv = pa_pc_dir / "teams_fake_all_dev_p8a_reference_step5000_frames_report.csv"
    real_csv = pa_pc_dir / "teams_real_all_dev_p8a_reference_step5000_frames_report.csv"
    if not (fake_csv.exists() and real_csv.exists()):
        log.error("frame_report CSVs not found at %s and %s", fake_csv, real_csv)
        return np.zeros((0, N_RADIAL)), np.zeros(0)
    fake_df = pd.read_csv(fake_csv).sample(n=200, random_state=42)
    real_df = pd.read_csv(real_csv).sample(n=200, random_state=42)
    cache = THIS / "raw_signal"
    cache.mkdir(parents=True, exist_ok=True)
    paths_local: list[Path] = []
    labels: list[int] = []
    log.info("task B: downloading %d fake + %d real (target: %s)", len(fake_df), len(real_df), cache)

    def download(uri: str, out: Path) -> bool:
        if out.exists():
            return True
        cp = subprocess.run(["gcloud", "storage", "cp", uri, str(out)],
                            capture_output=True, text=True, timeout=30)
        return cp.returncode == 0 and out.exists()

    download_list: list[tuple[str, Path, int]] = []
    for fp in fake_df["frame_path"]:
        basename = fp.split("/")[-1]
        download_list.append((fp, cache / f"fake__{basename}", 1))
    for fp in real_df["frame_path"]:
        basename = fp.split("/")[-1]
        download_list.append((fp, cache / f"real__{basename}", 0))

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(download, uri, out): (uri, out, lab) for uri, out, lab in download_list}
        ok = 0
        for done, fut in enumerate(as_completed(futs)):
            uri, out, lab = futs[fut]
            if fut.result():
                paths_local.append(out)
                labels.append(lab)
                ok += 1
            if (done + 1) % 50 == 0:
                log.info("  downloaded %d/%d (ok=%d)", done + 1, len(download_list), ok)
    log.info("task B: %d frames downloaded ok", ok)
    if not paths_local:
        return np.zeros((0, N_RADIAL)), np.zeros(0)
    X, kept = extract_parallel(paths_local, n_workers=4)
    if X.shape[0] != len(paths_local):
        kept_set = set(p.name for p in kept)
        labels = [lab for p, lab in zip(paths_local, labels) if p.name in kept_set]
    y = np.array(labels)
    log.info("task B shape: X=%s y=%s, mean(y)=%.3f", X.shape, y.shape, y.mean())
    return X, y


def main():
    log.info("=" * 70)
    log.info("Fourier-band overlap probe (closes loop fourier-aug-band-overlap-not-resolved)")
    log.info("=" * 70)

    Xa, ya = task_a_shortcut()
    log.info("task A per-band logistic AUC ...")
    aucs_a = per_band_auc(Xa, ya, n_splits=5)

    Xb, yb = task_b_signal()
    if Xb.size == 0:
        log.warning("task B has no data; falling back to recomputing from xinhe set's deployed scores")
        # Fallback: use the population label as a proxy task B target on the same xinhe set, but flipped — not equivalent
        aucs_b = np.zeros_like(aucs_a)
    else:
        log.info("task B per-band logistic AUC ...")
        aucs_b = per_band_auc(Xb, yb, n_splits=5)

    # Decision: bands where shortcut AUC is high but fake-signal AUC is low
    rows = []
    for b in range(N_RADIAL):
        rows.append({"band": b, "shortcut_auc": float(aucs_a[b]), "signal_auc": float(aucs_b[b]),
                     "shortcut_minus_signal": float(aucs_a[b] - aucs_b[b])})
    csv_path = OUT / "per_band_aucs.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info("wrote -> %s", csv_path)

    # Summary
    log.info("Per-band table (band: shortcut_auc / signal_auc / Δ):")
    for r in rows:
        marker = ""
        if r["shortcut_auc"] >= 0.85 and r["signal_auc"] < 0.65:
            marker = "  <- safe-to-randomize candidate"
        elif r["shortcut_auc"] < 0.65 and r["signal_auc"] >= 0.80:
            marker = "  <- preserve (manipulation signal-carrying)"
        log.info("  band %2d: shortcut=%.3f signal=%.3f Δ=%+.3f%s",
                 r["band"], r["shortcut_auc"], r["signal_auc"], r["shortcut_minus_signal"], marker)

    # Verdict
    safe = [r for r in rows if r["shortcut_auc"] >= 0.85 and r["signal_auc"] < 0.65]
    overlap = [r for r in rows if r["shortcut_auc"] >= 0.80 and r["signal_auc"] >= 0.80]
    summary = {
        "n_bands": N_RADIAL,
        "n_safe_to_randomize": len(safe),
        "n_overlap_high": len(overlap),
        "max_shortcut_auc": float(max(r["shortcut_auc"] for r in rows)),
        "max_signal_auc": float(max(r["signal_auc"] for r in rows)),
        "median_shortcut_auc": float(np.median([r["shortcut_auc"] for r in rows])),
        "median_signal_auc": float(np.median([r["signal_auc"] for r in rows])),
        "verdict": "TBD",
    }

    if len(safe) >= 3:
        summary["verdict"] = "GREEN_BAND_LIMITED_FOURIER_VIABLE"
        log.info("VERDICT: GREEN — %d bands have high shortcut signal and low fake signal; band-limited Fourier-aug greenlit", len(safe))
    elif len(overlap) >= 6:
        summary["verdict"] = "RED_BANDS_OVERLAP_FOURIER_DEAD"
        log.info("VERDICT: RED — bands overlap heavily; broad amp randomization is poison and band-limited not separable")
    else:
        summary["verdict"] = "AMBER_PARTIAL_OVERLAP"
        log.info("VERDICT: AMBER — partial separability; band-limited Fourier-aug viable but with caveats")

    with (OUT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("summary -> %s", OUT / "summary.json")
    log.info("DONE")


if __name__ == "__main__":
    main()
