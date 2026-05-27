"""Chroma Fourier-band overlap probe — replicates Probe 6 per-channel.

Probe 6 ran the radial-band shortcut/signal partition on grayscale FFT amplitude,
identified bands 12-13 as the cleanest cell (high may6/may5 shortcut AUC, low
fake/real signal AUC). The PE_FOURIER_BAND recipe randomises bands 12-13 + 8-9.

But the may6 vs may5 shortcut is **chroma-loaded** (Probe 1: sat_std d=-5.95,
b_std d=-2.61, r_std d=-2.79, g_std d=-2.87 — all chroma statistics). Grayscale
FFT collapses RGB into luminance and may miss chroma-specific shortcut bands or
mis-locate cleanest cells. This probe re-runs Probe 6 per-channel for R, G, B,
L, a, b — to gate whether the Fourier-aug recipe must extend to chroma.

Reuses already-cached frames from Probe 6:
  - may5/may6: analysis/xinhe_cross_camera_audit_2026-05-06/raw/{may5,may6}
  - signal: analysis/fourier_band_overlap_2026-05-06/raw_signal/{fake__,real__}
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("chroma_band_fourier")

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
THIS = REPO / "analysis/chroma_band_fourier_2026-05-06"
OUT = THIS / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

XINHE = REPO / "analysis/xinhe_cross_camera_audit_2026-05-06/raw"
RAW_SIGNAL = REPO / "analysis/fourier_band_overlap_2026-05-06/raw_signal"

RES = 224
N_RADIAL = 16
N_ANGLE = 8

# Grayscale Probe 6 reference (for cross-channel comparison)
GRAYSCALE_REF_CSV = REPO / "analysis/fourier_band_overlap_2026-05-06/outputs/per_band_aucs.csv"

# Cleanest-cell criteria (carried from Probe 6 verdict logic)
SHORTCUT_HIGH = 0.85   # band carries shortcut signal
SIGNAL_LOW = 0.65      # band does NOT carry manipulation signal  -> safe to randomise
SIGNAL_HIGH = 0.65     # band DOES carry manipulation signal       -> preserve
SHORTCUT_LOW_FOR_INVERSE = 0.65  # for "inverse" preserve-bands

CHANNELS = ["R", "G", "B", "L", "lab_a", "lab_b"]


def radial_bins() -> np.ndarray:
    """(RES, RES) array with radial band index 0..N_RADIAL-1."""
    yy, xx = np.mgrid[:RES, :RES]
    cx = cy = (RES - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = r.max()
    bins = np.clip((r / r_max * N_RADIAL).astype(int), 0, N_RADIAL - 1)
    return bins


RBINS = radial_bins()


def fft_radial_per_channel(path: Path) -> dict[str, np.ndarray] | None:
    """Return {channel_name: np.ndarray of length N_RADIAL}.

    Computes FFT log-magnitude per channel, then mean per radial band.
    Channels: R, G, B (BGR-source split), L, a, b (LAB after BGR2LAB).
    """
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_bgr = cv2.resize(img_bgr, (RES, RES), interpolation=cv2.INTER_LINEAR)

    # RGB split (cv2 reads BGR; we want R, G, B in semantic order)
    B = img_bgr[:, :, 0].astype(np.float32)
    G = img_bgr[:, :, 1].astype(np.float32)
    R = img_bgr[:, :, 2].astype(np.float32)

    # LAB conversion. cv2 BGR2LAB returns L in [0,255], a/b in [0,255] (centered ~128).
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = img_lab[:, :, 0].astype(np.float32)
    a = img_lab[:, :, 1].astype(np.float32)
    b = img_lab[:, :, 2].astype(np.float32)

    out: dict[str, np.ndarray] = {}
    for name, ch in zip(CHANNELS, [R, G, B, L, a, b]):
        F = np.fft.fft2(ch)
        F = np.fft.fftshift(F)
        mag = np.log1p(np.abs(F))
        feats = np.zeros(N_RADIAL, dtype=np.float32)
        for k in range(N_RADIAL):
            m = (RBINS == k)
            feats[k] = mag[m].mean()
        out[name] = feats
    return out


def extract_parallel(paths: list[Path], n_workers: int = 4) -> tuple[dict[str, np.ndarray], list[Path]]:
    """Returns ({channel: (N, N_RADIAL)}, kept_paths)."""
    feats: list[dict[str, np.ndarray] | None] = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        fut_to_i = {ex.submit(fft_radial_per_channel, p): i for i, p in enumerate(paths)}
        for done, fut in enumerate(as_completed(fut_to_i)):
            i = fut_to_i[fut]
            res = fut.result()
            if res is not None:
                feats[i] = res
            if (done + 1) % 100 == 0:
                log.info("  extracted %d / %d", done + 1, len(paths))
    keep_idx = [i for i, f in enumerate(feats) if f is not None]
    paths_kept = [paths[i] for i in keep_idx]
    out: dict[str, np.ndarray] = {}
    for ch in CHANNELS:
        out[ch] = np.stack([feats[i][ch] for i in keep_idx])
    return out, paths_kept


def per_band_auc(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """For each band, train a 1-feature logistic and report mean CV AUC."""
    n_b = X.shape[1]
    aucs = np.zeros(n_b, dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for k in range(n_b):
        Xb = X[:, k:k + 1]
        fold_aucs = []
        for tr, te in skf.split(Xb, y):
            sc = StandardScaler().fit(Xb[tr])
            Xtr = sc.transform(Xb[tr]); Xte = sc.transform(Xb[te])
            clf = LogisticRegression(class_weight="balanced", max_iter=200, n_jobs=1)
            clf.fit(Xtr, y[tr])
            p = clf.predict_proba(Xte)[:, 1]
            fold_aucs.append(roc_auc_score(y[te], p))
        aucs[k] = float(np.mean(fold_aucs))
    return aucs


def load_xinhe_paths() -> tuple[list[Path], list[Path]]:
    paths_may6 = sorted((XINHE / "may6").glob("*.png"))
    paths_may5 = sorted((XINHE / "may5").glob("*.png"))
    log.info("xinhe: may6=%d may5=%d", len(paths_may6), len(paths_may5))
    return paths_may6, paths_may5


def load_signal_paths() -> tuple[list[Path], list[Path]]:
    fake_paths = sorted(RAW_SIGNAL.glob("fake__*"))
    real_paths = sorted(RAW_SIGNAL.glob("real__*"))
    log.info("signal: fake=%d real=%d", len(fake_paths), len(real_paths))
    return fake_paths, real_paths


def task_a_shortcut() -> tuple[dict[str, np.ndarray], np.ndarray]:
    paths_may6, paths_may5 = load_xinhe_paths()
    paths_all = list(paths_may6) + list(paths_may5)
    log.info("task A: extracting per-channel FFT bands for %d frames ...", len(paths_all))
    Xch, kept = extract_parallel(paths_all, n_workers=4)
    y = np.concatenate([np.ones(len(paths_may6)), np.zeros(len(paths_may5))])
    n_kept = next(iter(Xch.values())).shape[0]
    if n_kept != len(paths_all):
        kept_set = set(p.name for p in kept)
        keep_y = np.array([p.name in kept_set for p in paths_all])
        y = y[keep_y]
    log.info("task A: kept=%d, mean(y)=%.3f", n_kept, y.mean())
    return Xch, y


def task_b_signal() -> tuple[dict[str, np.ndarray], np.ndarray]:
    fake_paths, real_paths = load_signal_paths()
    paths_all = list(fake_paths) + list(real_paths)
    if not paths_all:
        log.error("no signal frames cached at %s", RAW_SIGNAL)
        return {ch: np.zeros((0, N_RADIAL)) for ch in CHANNELS}, np.zeros(0)
    log.info("task B: extracting per-channel FFT bands for %d frames ...", len(paths_all))
    Xch, kept = extract_parallel(paths_all, n_workers=4)
    y = np.concatenate([np.ones(len(fake_paths)), np.zeros(len(real_paths))])
    n_kept = next(iter(Xch.values())).shape[0]
    if n_kept != len(paths_all):
        kept_set = set(p.name for p in kept)
        keep_y = np.array([p.name in kept_set for p in paths_all])
        y = y[keep_y]
    log.info("task B: kept=%d, mean(y)=%.3f", n_kept, y.mean())
    return Xch, y


def write_per_channel_csv(channel: str, aucs_a: np.ndarray, aucs_b: np.ndarray) -> Path:
    rows = []
    for k in range(N_RADIAL):
        rows.append({
            "band": k,
            "shortcut_auc": float(aucs_a[k]),
            "signal_auc": float(aucs_b[k]),
            "shortcut_minus_signal": float(aucs_a[k] - aucs_b[k]),
            "cleanest_cell": int(aucs_a[k] >= SHORTCUT_HIGH and aucs_b[k] < SIGNAL_LOW),
            "signal_carrying": int(aucs_b[k] >= SIGNAL_HIGH and aucs_a[k] < SHORTCUT_LOW_FOR_INVERSE),
        })
    csv_path = OUT / f"per_band_aucs_{channel}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return csv_path


def load_grayscale_ref() -> dict[int, dict[str, float]]:
    """Return {band: {shortcut_auc, signal_auc, ...}} from Probe 6."""
    ref: dict[int, dict[str, float]] = {}
    if not GRAYSCALE_REF_CSV.exists():
        log.warning("grayscale ref %s not found; cross-comparison degraded", GRAYSCALE_REF_CSV)
        return ref
    with GRAYSCALE_REF_CSV.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            ref[int(r["band"])] = {
                "shortcut_auc": float(r["shortcut_auc"]),
                "signal_auc": float(r["signal_auc"]),
                "shortcut_minus_signal": float(r["shortcut_minus_signal"]),
            }
    return ref


def main():
    log.info("=" * 70)
    log.info("Chroma Fourier-band overlap probe (CHROMA_BAND_FOURIER_2026-05-06)")
    log.info("=" * 70)

    Xa_ch, ya = task_a_shortcut()
    Xb_ch, yb = task_b_signal()

    if ya.size == 0 or yb.size == 0:
        log.error("missing data; aborting")
        return

    grayscale_ref = load_grayscale_ref()

    # Per-channel results
    per_channel: dict[str, dict] = {}
    cleanest_table_rows: list[dict] = []

    for ch in CHANNELS:
        log.info("--- channel %s ---", ch)
        aucs_a = per_band_auc(Xa_ch[ch], ya, n_splits=5)
        aucs_b = per_band_auc(Xb_ch[ch], yb, n_splits=5)
        csv_path = write_per_channel_csv(ch, aucs_a, aucs_b)
        log.info("wrote -> %s", csv_path)

        cleanest = []
        signal_carrying = []
        for k in range(N_RADIAL):
            sa = float(aucs_a[k]); si = float(aucs_b[k])
            is_clean = (sa >= SHORTCUT_HIGH and si < SIGNAL_LOW)
            is_sig = (si >= SIGNAL_HIGH and sa < SHORTCUT_LOW_FOR_INVERSE)
            if is_clean:
                cleanest.append(k)
            if is_sig:
                signal_carrying.append(k)
            cleanest_table_rows.append({
                "channel": ch,
                "band": k,
                "shortcut_auc": round(sa, 4),
                "signal_auc": round(si, 4),
                "delta": round(sa - si, 4),
                "cleanest_cell_flag": int(is_clean),
                "signal_carrying_flag": int(is_sig),
            })
            marker = ""
            if is_clean:
                marker = "  <- safe-to-randomize candidate"
            elif is_sig:
                marker = "  <- preserve (signal-carrying)"
            log.info("  band %2d: shortcut=%.3f signal=%.3f Δ=%+.3f%s", k, sa, si, sa - si, marker)

        per_channel[ch] = {
            "aucs_a": aucs_a.tolist(),
            "aucs_b": aucs_b.tolist(),
            "cleanest_cell_bands": cleanest,
            "signal_carrying_bands": signal_carrying,
            "max_shortcut_auc": float(aucs_a.max()),
            "max_signal_auc": float(aucs_b.max()),
            "median_shortcut_auc": float(np.median(aucs_a)),
            "median_signal_auc": float(np.median(aucs_b)),
            "argmax_shortcut_band": int(aucs_a.argmax()),
            "argmax_signal_band": int(aucs_b.argmax()),
        }

    # Cross-channel comparison vs grayscale (Probe 6)
    grayscale_summary = None
    if grayscale_ref:
        gs_clean = []
        gs_sig = []
        for k in range(N_RADIAL):
            gs_sa = grayscale_ref[k]["shortcut_auc"]
            gs_si = grayscale_ref[k]["signal_auc"]
            if gs_sa >= SHORTCUT_HIGH and gs_si < SIGNAL_LOW:
                gs_clean.append(k)
            if gs_si >= SIGNAL_HIGH and gs_sa < SHORTCUT_LOW_FOR_INVERSE:
                gs_sig.append(k)
        grayscale_summary = {
            "cleanest_cell_bands": gs_clean,
            "signal_carrying_bands": gs_sig,
        }

    # Cross-channel: union/intersection of cleanest cells across channels
    sets_clean = [set(per_channel[ch]["cleanest_cell_bands"]) for ch in CHANNELS]
    sets_sig = [set(per_channel[ch]["signal_carrying_bands"]) for ch in CHANNELS]
    chroma_only_channels = ["R", "G", "B", "lab_a", "lab_b"]  # no L
    sets_chroma_clean = [set(per_channel[ch]["cleanest_cell_bands"]) for ch in chroma_only_channels]
    sets_chroma_sig = [set(per_channel[ch]["signal_carrying_bands"]) for ch in chroma_only_channels]

    union_clean = set().union(*sets_clean)
    inter_clean = set.intersection(*sets_clean) if sets_clean else set()
    union_sig = set().union(*sets_sig)

    union_chroma_clean = set().union(*sets_chroma_clean)
    inter_chroma_clean = set.intersection(*sets_chroma_clean) if sets_chroma_clean else set()

    # Bands cleanest in chroma but NOT in grayscale (potential "chroma extension")
    chroma_extra_clean = set()
    if grayscale_summary:
        gs_clean_set = set(grayscale_summary["cleanest_cell_bands"])
        chroma_extra_clean = union_chroma_clean - gs_clean_set

    # Write cleanest_cells_table.csv
    table_csv = OUT / "cleanest_cells_table.csv"
    with table_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cleanest_table_rows[0].keys()))
        w.writeheader()
        for r in cleanest_table_rows:
            w.writerow(r)
    log.info("wrote -> %s", table_csv)

    # Aggregate summary
    summary = {
        "n_bands": N_RADIAL,
        "n_xinhe_may6": int((ya == 1).sum()),
        "n_xinhe_may5": int((ya == 0).sum()),
        "n_signal_fake": int((yb == 1).sum()),
        "n_signal_real": int((yb == 0).sum()),
        "criteria": {
            "shortcut_high": SHORTCUT_HIGH,
            "signal_low": SIGNAL_LOW,
            "signal_high_for_inverse": SIGNAL_HIGH,
            "shortcut_low_for_inverse": SHORTCUT_LOW_FOR_INVERSE,
        },
        "per_channel": per_channel,
        "grayscale_reference_probe6": grayscale_summary,
        "cross_channel": {
            "union_cleanest_cell_bands_all": sorted(union_clean),
            "intersection_cleanest_cell_bands_all": sorted(inter_clean),
            "union_signal_carrying_bands_all": sorted(union_sig),
            "union_cleanest_cell_bands_chroma_only": sorted(union_chroma_clean),
            "intersection_cleanest_cell_bands_chroma_only": sorted(inter_chroma_clean),
            "chroma_extra_clean_vs_grayscale": sorted(chroma_extra_clean),
        },
    }

    # Verdict logic
    n_intersect_clean = len(inter_clean)
    if grayscale_summary:
        gs_clean_set = set(grayscale_summary["cleanest_cell_bands"])
        agreement = inter_clean & gs_clean_set
        summary["cross_channel"]["intersection_with_grayscale"] = sorted(agreement)
        if agreement and not chroma_extra_clean:
            summary["verdict"] = "GRAYSCALE_BANDS_GENERALIZE"
        elif chroma_extra_clean and agreement:
            summary["verdict"] = "GRAYSCALE_PLUS_CHROMA_EXTENSION_NEEDED"
        elif chroma_extra_clean and not agreement:
            summary["verdict"] = "CHROMA_DOMINATES_DIFFERENT_BANDS"
        else:
            summary["verdict"] = "NO_CLEAN_BANDS_ON_CHROMA"
    else:
        summary["verdict"] = (
            "CHROMA_HAS_CLEAN_INTERSECTION" if n_intersect_clean > 0 else "CHROMA_NO_CLEAN_INTERSECTION"
        )

    summary_path = OUT / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("wrote -> %s", summary_path)
    log.info("VERDICT: %s", summary["verdict"])
    log.info("union cleanest (all 6 channels): %s", sorted(union_clean))
    log.info("intersection cleanest (all 6 channels): %s", sorted(inter_clean))
    log.info("union cleanest (chroma RGBab): %s", sorted(union_chroma_clean))
    log.info("chroma_extra_clean_vs_grayscale: %s", sorted(chroma_extra_clean))
    log.info("DONE")


if __name__ == "__main__":
    main()
