"""Probe 7 — extensive per-layer P8A vs E2B vs PA_3800 divergence audit.

Asks: at which transformer layer do P8A and E2B (and PA_3800, FT-from-E2B)
diverge? Where does the shortcut signal (may6 vs may5 capture-pipeline drift)
become readable? Where does the manipulation signal (teams fake vs real)
become readable? Does the 2026-04-30 finding that layer 6 has peak
discriminability survive when measured against the deployed model (E2B)?

Reuses the hook pattern from
analysis/intermediate_layer_probe_2026-04-30/intermediate_layer_probe.py.

Outputs everything to analysis/per_layer_p8a_e2b_pa_2026-05-06/outputs/.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
sys.path.insert(0, str(REPO))

from arena.model_arena import _download_checkpoint  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("per_layer_probe")

THIS = REPO / "analysis/per_layer_p8a_e2b_pa_2026-05-06"
OUT = THIS / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Same configs as run_local_inference.py
DETECTOR_CONFIG = REPO / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO / "config/defaults.yaml"

CKPTS = {
    "P8A": "gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
    "E2B": "gs://training-job-outputs/best_checkpoints/rmat8lwx/top_n_effort_20260503_step3200_auc0.9863_eer0.0310.pth",
    "PA_3800": "gs://training-job-outputs/best_checkpoints/26u8bn1t/top_n_effort_20260504_step3800_auc0.9892_eer0.0402.pth",
}

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
RES = 224
LAYERS = list(range(12))  # all 12 resblocks (CLIP-B16)

# Substrates ----------------------------------------------------------------
XINHE = REPO / "analysis/xinhe_cross_camera_audit_2026-05-06/raw"
FAKE_REAL = REPO / "analysis/fourier_band_overlap_2026-05-06/raw_signal"


def collect_substrates() -> dict[str, list[tuple[str, Path, int]]]:
    """Build the substrate frame lists. Each entry: (population_label, path, binary_label).

    Returns:
        {
          "shortcut": list — may6 (1) vs may5 (0)  [n=152]
          "fake_signal": list — teams_fake (1) vs teams_real (0)  [n=400]
        }
    """
    out: dict[str, list[tuple[str, Path, int]]] = {}

    may6 = sorted((XINHE / "may6").glob("*.png"))
    may5 = sorted((XINHE / "may5").glob("*.png"))
    s = [("may6", p, 1) for p in may6] + [("may5", p, 0) for p in may5]
    out["shortcut"] = s
    log.info("substrate shortcut: %d frames (may6=%d, may5=%d)", len(s), len(may6), len(may5))

    fake = sorted(FAKE_REAL.glob("fake__*"))
    real = sorted(FAKE_REAL.glob("real__*"))
    s2 = [("teams_fake", p, 1) for p in fake] + [("teams_real", p, 0) for p in real]
    out["fake_signal"] = s2
    log.info("substrate fake_signal: %d frames (fake=%d, real=%d)", len(s2), len(fake), len(real))

    return out


# Model loading -------------------------------------------------------------
def load_model(ckpt_path: str | Path, device: torch.device) -> torch.nn.Module:
    import yaml
    from detectors import DETECTOR

    with open(DETECTOR_CONFIG) as f:
        cfg = yaml.safe_load(f)
    with open(TRAIN_CONFIG) as f:
        cfg.update(yaml.safe_load(f))

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        model_config = ckpt.get("model_config", {})
        for k, v in model_config.items():
            if k != "current_arcface_s":
                cfg[k] = v
    else:
        state_dict = ckpt
        model_config = {}

    model = DETECTOR[cfg["model_name"]](cfg).to(device)
    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])
    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model


def get_resblocks(model: torch.nn.Module) -> torch.nn.ModuleList:
    visual = model.backbone.visual if hasattr(model.backbone, "visual") else model.backbone
    return visual.transformer.resblocks


# Image loader --------------------------------------------------------------
def preprocess(path: Path) -> torch.Tensor | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (RES, RES), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - CLIP_MEAN) / CLIP_STD
    return torch.from_numpy(img.transpose(2, 0, 1))


# Per-layer feature extraction ---------------------------------------------
def extract_per_layer(model: torch.nn.Module, paths: list[Path], device: torch.device,
                      batch_size: int = 16) -> tuple[dict[int, np.ndarray], list[int]]:
    captured: dict[int, list[np.ndarray]] = {ix: [] for ix in LAYERS}
    resblocks = get_resblocks(model)

    def make_hook(ix: int):
        def hook(_m, _i, output):
            if output.dim() == 3:
                if output.shape[0] >= output.shape[1]:
                    cls = output[0]  # seq-first
                else:
                    cls = output[:, 0]  # batch-first
            elif output.dim() == 2:
                cls = output
            else:
                raise RuntimeError(f"unexpected resblock output shape: {tuple(output.shape)}")
            captured[ix].append(cls.detach().cpu().to(torch.float32).numpy())
        return hook

    handles = [resblocks[ix].register_forward_hook(make_hook(ix)) for ix in LAYERS]
    try:
        valid_idx: list[int] = []
        pending: list[tuple[int, torch.Tensor]] = []
        for i, p in enumerate(paths):
            t = preprocess(p)
            if t is None:
                continue
            pending.append((i, t))

        n = len(pending)
        t0 = time.time()
        for j in range(0, n, batch_size):
            chunk = pending[j:j + batch_size]
            batch_idx = [c[0] for c in chunk]
            batch = torch.stack([c[1] for c in chunk]).to(device)
            with torch.inference_mode():
                _ = model.backbone(batch)
            valid_idx.extend(batch_idx)
            done = j + len(chunk)
            if (j // batch_size) % 5 == 0 or done == n:
                elapsed = time.time() - t0
                fps = done / max(elapsed, 1e-3)
                log.info("    extract %d/%d (%.1f fps, elapsed=%.1fs)", done, n, fps, elapsed)
        for ix in LAYERS:
            captured[ix] = np.concatenate(captured[ix], axis=0) if captured[ix] else np.zeros((0, 0))
    finally:
        for h in handles:
            h.remove()
    return captured, valid_idx


# Analyses ------------------------------------------------------------------
def cos_distribution(A: np.ndarray, B: np.ndarray) -> dict[str, float]:
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    cs = np.einsum("ij,ij->i", An, Bn)
    return {
        "n": int(cs.size),
        "mean": float(cs.mean()),
        "p50": float(np.median(cs)),
        "p10": float(np.percentile(cs, 10)),
        "p90": float(np.percentile(cs, 90)),
        "min": float(cs.min()),
        "max": float(cs.max()),
        "frac_below_0_95": float((cs < 0.95).mean()),
        "frac_below_0_90": float((cs < 0.90).mean()),
        "frac_below_0_80": float((cs < 0.80).mean()),
    }


def lr_auc(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 0) -> tuple[float, float]:
    """Return (mean AUC, std) across folds. Uses balanced LR with default C."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(Xn, y):
        sc = StandardScaler().fit(Xn[tr])
        Xtr = sc.transform(Xn[tr]); Xte = sc.transform(Xn[te])
        clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1, class_weight="balanced", solver="lbfgs")
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


# Main ----------------------------------------------------------------------
def main():
    log.info("=" * 78)
    log.info("Per-layer P8A vs E2B vs PA_3800 divergence probe (12 resblocks × 3 ckpts × 2 substrates)")
    log.info("=" * 78)

    device = torch.device("cpu")

    # Build paths and labels first
    substrates = collect_substrates()
    all_paths: list[Path] = []
    substrate_meta: list[tuple[str, str, int]] = []  # (substrate_name, population, label)
    for sname, items in substrates.items():
        for pop, p, lab in items:
            all_paths.append(p)
            substrate_meta.append((sname, pop, lab))
    n_total = len(all_paths)
    log.info("total frames across substrates: %d", n_total)

    # Extract features for each ckpt
    feats_by_ckpt: dict[str, dict[int, np.ndarray]] = {}
    valid_by_ckpt: dict[str, list[int]] = {}
    for name, uri in CKPTS.items():
        log.info("[%s] downloading + loading ckpt ...", name)
        local = _download_checkpoint(uri)
        t0 = time.time()
        model = load_model(local, device)
        log.info("[%s] model loaded in %.1fs", name, time.time() - t0)
        log.info("[%s] extracting per-layer [CLS] for %d frames ...", name, n_total)
        feats, valid = extract_per_layer(model, all_paths, device, batch_size=16)
        feats_by_ckpt[name] = feats
        valid_by_ckpt[name] = valid
        log.info("[%s] valid frames: %d/%d, layer 0 shape=%s, layer 11 shape=%s",
                 name, len(valid), n_total, feats[0].shape, feats[11].shape)
        del model

    # Align: only keep frame indices that succeeded across all 3 ckpts
    common_idx = sorted(set(valid_by_ckpt["P8A"]) & set(valid_by_ckpt["E2B"]) & set(valid_by_ckpt["PA_3800"]))
    log.info("common valid indices across all ckpts: %d/%d", len(common_idx), n_total)

    def slice_by_ckpt(name: str, layer: int, target_idx: list[int]) -> np.ndarray:
        valid = valid_by_ckpt[name]
        idx_map = {orig: row for row, orig in enumerate(valid)}
        rows = [idx_map[i] for i in target_idx if i in idx_map]
        return feats_by_ckpt[name][layer][rows]

    # Substrate-specific subsets
    shortcut_idx = [i for i in common_idx if substrate_meta[i][0] == "shortcut"]
    fake_idx = [i for i in common_idx if substrate_meta[i][0] == "fake_signal"]
    shortcut_y = np.array([substrate_meta[i][2] for i in shortcut_idx])
    fake_y = np.array([substrate_meta[i][2] for i in fake_idx])
    log.info("shortcut substrate: n=%d (mean(y)=%.3f)", len(shortcut_idx), shortcut_y.mean())
    log.info("fake_signal substrate: n=%d (mean(y)=%.3f)", len(fake_idx), fake_y.mean())

    # Per-layer analyses ----------------------------------------------------
    rows_cos: list[dict] = []
    rows_auc: list[dict] = []
    pairs = [("P8A", "E2B"), ("P8A", "PA_3800"), ("E2B", "PA_3800")]

    log.info("computing per-layer cosines and LR AUCs ...")
    for layer in LAYERS:
        log.info(" layer %d:", layer)
        # Cosine distance per pair, on shortcut substrate
        for a, b in pairs:
            for substrate_name, idx in [("shortcut", shortcut_idx), ("fake_signal", fake_idx)]:
                Fa = slice_by_ckpt(a, layer, idx)
                Fb = slice_by_ckpt(b, layer, idx)
                d = cos_distribution(Fa, Fb)
                d.update({"layer": layer, "ckpt_a": a, "ckpt_b": b, "substrate": substrate_name})
                rows_cos.append(d)

        # LR AUC for each ckpt at this layer, on each substrate task
        for name in CKPTS:
            # Shortcut task: may6 vs may5
            X_short = slice_by_ckpt(name, layer, shortcut_idx)
            auc_s, std_s = lr_auc(X_short, shortcut_y)
            # Fake-signal task
            X_fake = slice_by_ckpt(name, layer, fake_idx)
            auc_f, std_f = lr_auc(X_fake, fake_y)
            rows_auc.append({"layer": layer, "ckpt": name,
                             "shortcut_auc": auc_s, "shortcut_auc_std": std_s,
                             "fake_signal_auc": auc_f, "fake_signal_auc_std": std_f})
            log.info("   [%s] shortcut AUC=%.3f±%.3f, fake_signal AUC=%.3f±%.3f",
                     name, auc_s, std_s, auc_f, std_f)

    # Write CSVs ------------------------------------------------------------
    cos_csv = OUT / "per_layer_cosine.csv"
    with cos_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_cos[0].keys()))
        w.writeheader()
        w.writerows(rows_cos)
    log.info("wrote %s (%d rows)", cos_csv, len(rows_cos))

    auc_csv = OUT / "per_layer_aucs.csv"
    with auc_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_auc[0].keys()))
        w.writeheader()
        w.writerows(rows_auc)
    log.info("wrote %s (%d rows)", auc_csv, len(rows_auc))

    # Summary ---------------------------------------------------------------
    summary = {
        "probe": "per_layer_p8a_e2b_pa",
        "date": "2026-05-06",
        "substrates": {
            "shortcut": {"n": int(len(shortcut_idx)), "task": "may6 (1) vs may5 (0)"},
            "fake_signal": {"n": int(len(fake_idx)), "task": "teams_fake_all_dev (1) vs teams_real_all_dev (0)"},
        },
        "ckpts": list(CKPTS.keys()),
        "layers": LAYERS,
        "key_findings": {},
    }

    # Per-layer cosine table for P8A↔E2B on each substrate
    cos_p8a_e2b_short = [r for r in rows_cos if r["ckpt_a"] == "P8A" and r["ckpt_b"] == "E2B" and r["substrate"] == "shortcut"]
    cos_p8a_e2b_fake = [r for r in rows_cos if r["ckpt_a"] == "P8A" and r["ckpt_b"] == "E2B" and r["substrate"] == "fake_signal"]
    summary["key_findings"]["P8A_E2B_cosine_p50_by_layer_shortcut"] = {r["layer"]: r["p50"] for r in cos_p8a_e2b_short}
    summary["key_findings"]["P8A_E2B_cosine_p50_by_layer_fake"] = {r["layer"]: r["p50"] for r in cos_p8a_e2b_fake}

    # Layer where P8A and E2B first diverge meaningfully (frac<0.90 ≥ 5%)
    div_layer = next((r["layer"] for r in cos_p8a_e2b_short if r["frac_below_0_90"] >= 0.05), None)
    summary["key_findings"]["P8A_E2B_first_divergence_layer_shortcut"] = div_layer

    # Best layer per ckpt per task
    for name in CKPTS:
        rs_short = [r for r in rows_auc if r["ckpt"] == name]
        best_short_layer = max(rs_short, key=lambda r: r["shortcut_auc"])
        best_fake_layer = max(rs_short, key=lambda r: r["fake_signal_auc"])
        summary["key_findings"][f"{name}_best_shortcut_layer"] = {
            "layer": best_short_layer["layer"],
            "auc": best_short_layer["shortcut_auc"],
        }
        summary["key_findings"][f"{name}_best_fake_signal_layer"] = {
            "layer": best_fake_layer["layer"],
            "auc": best_fake_layer["fake_signal_auc"],
        }

    with (OUT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("wrote %s", OUT / "summary.json")
    log.info("DONE")


if __name__ == "__main__":
    main()
