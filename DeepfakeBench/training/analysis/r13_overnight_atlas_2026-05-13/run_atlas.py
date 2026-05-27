"""L11 atlas inv_mean recompute for the 4 R13 overnight candidate ckpts (2026-05-13).

Methodology mirrors A3 (2026-05-11) exactly. See:
  analysis/cpu_diagnostics_2026-05-11_a3_atlas_composition/_run_a3_atlas_composition.py

Candidate ckpts (one representative per slot, highest train-AUC per slot):
  Slot 1: LoRA-P8A         top_n_step2000   (gf6l06rf)
  Slot 2: LoRA-T5C         periodic_step1500 (912kd88q)
  Slot 3: T5C+jitter030    top_n_step4500   (502dcznh)
  Slot 4: B16-scratch+Fourier top_n_step10000 (qrpf5dtr)
+ Reference P8A_REFERENCE_STEP5000 (recompute baseline parity).

Two phases:
  Phase 1 (extract): build EffortDetector, optionally wrap LoRA, load state-dict,
                     hook resblocks[11], run 800-frame triptych, cache CLS L11.
  Phase 2 (probe):   re-fit 5-fold CV LR probes for (forgery + 5 shortcuts) per
                     substrate slice; compute inv_mean and Δ vs P8A baseline.

CPU/MPS detection per A3 convention. n_jobs=1 per memory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "analysis" / "r13_overnight_atlas_2026-05-13"
CKPT_DIR = OUT / "_cache"
FEAT_DIR = OUT / "_features"
OUT.mkdir(parents=True, exist_ok=True)
FEAT_DIR.mkdir(parents=True, exist_ok=True)

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Same triptych as A3 (first 800 rows of sampled_frames.csv).
SAMPLED_CSV = (
    REPO / "analysis" / "embedding_triptych_2026-04-30"
    / "outputs" / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"
)
ATLAS_PARQUET = REPO / "analysis" / "iq_data_atlas_2026-05-08" / "outputs" / "per_frame.parquet"
DEFAULT_DETECTOR_CONFIG = REPO / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO / "config" / "train_config.yaml"
P8A_PRIOR_CACHE = REPO / "analysis" / "iq_perlayer_probe_2026-05-08" / "_cache" / "intermediate__P8A__layer11__n800.npz"

# --- Candidate definitions ---
# Each entry: { label, ckpt_path, lora_config (or None) }
# LoRA hyperparams from experiments/phase2_round13/R13_LORA_*.yaml (rank=16, alpha=32, layers [10,11])
LORA_CONFIG = dict(target_layers=[10, 11], rank=16, alpha=32.0,
                   target_modules=("attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj"))

CANDIDATES = {
    "SLOT1_LORA_P8A_top_n_step2000":      dict(
        ckpt=CKPT_DIR / "top_n_effort_20260513_step2000_auc0.9929_eer0.0229.pth",
        lora=LORA_CONFIG,
    ),
    "SLOT2_LORA_T5C_periodic_step1500":   dict(
        ckpt=CKPT_DIR / "periodic_effort_20260513_step1500_auc0.9941_eer0.0198.pth",
        lora=LORA_CONFIG,
    ),
    "SLOT3_T5C_JITTER030_top_n_step4500": dict(
        ckpt=CKPT_DIR / "top_n_effort_20260513_step4500_auc0.9923_eer0.0304.pth",
        lora=None,
    ),
    "SLOT4_B16SC_FOURIER_top_n_step10000": dict(
        ckpt=CKPT_DIR / "top_n_effort_20260513_step10000_auc0.9902_eer0.0259.pth",
        lora=None,
    ),
}

P8A_REF_CKPT = REPO / "analysis" / "_features_cache_2026-04-30" / "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"

# --- Atlas substrate config (matches A3) ---
CHRONIC_6_PATTERNS = [
    "Roy_D", "PC_Generator", "bla_bla_chow",
    "Md_noyn_Sharker", "dor_shkedi", "healthy_dor",
]
PRIMARY_6_IQ = ["lap_var", "min_dim", "luma_mean", "color_b_dev", "edge_mag", "skin_frac"]

logger = logging.getLogger("r13-atlas")


# =============================================================================
# Phase 1 — Feature extraction
# =============================================================================

def load_effort_model(ckpt_path: Path, device: torch.device,
                      lora_cfg: Optional[Dict] = None) -> torch.nn.Module:
    """Build EffortDetector with cfg from saved model_config; wrap LoRA if needed;
    load state dict. LoRA wrap must happen BEFORE load so module-name changes
    (`out_proj.base_layer.weight_main`) match the state dict.
    """
    import yaml
    from detectors import DETECTOR

    with open(DEFAULT_DETECTOR_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    with open(DEFAULT_TRAIN_CONFIG, "r") as f:
        train_cfg = yaml.safe_load(f)
    cfg.update(train_cfg)

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

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)

    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])

    # Wrap LoRA AFTER the model is built but BEFORE state load.
    if lora_cfg is not None:
        from detectors.lora_adapter import (
            apply_lora_to_openclip_visual, freeze_base_clip_encoder, count_lora_parameters,
        )
        visual = model.backbone.visual
        n_wrapped = apply_lora_to_openclip_visual(
            visual,
            target_layers=list(lora_cfg["target_layers"]),
            rank=int(lora_cfg["rank"]),
            alpha=float(lora_cfg["alpha"]),
            target_modules=tuple(lora_cfg["target_modules"]),
        )
        # Don't freeze for inference; just wrap. Move freshly-allocated LoRA params to device.
        model.to(device)
        lora_params, total_params = count_lora_parameters(visual)
        logger.info("  [LoRA] wrapped %d layers (%d lora params of %d total)",
                    n_wrapped, lora_params, total_params)

    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        # Filter out spurious "missing" keys that are LoRA delta paths the
        # forward path constructs at runtime (we don't fail).
        miss_lora = [k for k in missing if "lora_A" in k or "lora_B" in k]
        miss_other = [k for k in missing if "lora_A" not in k and "lora_B" not in k]
        if miss_other:
            logger.info("  missing %d non-LoRA keys (e.g. %s)", len(miss_other), miss_other[:3])
        if miss_lora:
            logger.info("  missing %d LoRA keys (e.g. %s)", len(miss_lora), miss_lora[:3])
    if unexpected:
        logger.info("  dropped %d unexpected keys (e.g. %s)", len(unexpected), unexpected[:3])

    model.eval()
    return model


def get_resblocks(model: torch.nn.Module) -> torch.nn.ModuleList:
    visual = model.backbone.visual if hasattr(model.backbone, "visual") else model.backbone
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        return visual.transformer.resblocks
    raise RuntimeError("Could not find transformer.resblocks")


def load_and_preprocess(local_path: Path, resolution: int = 224) -> Optional[torch.Tensor]:
    import cv2
    img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1))


def extract_l11(model, paths: List[str], device: torch.device, batch_size: int = 16
                ) -> Tuple[np.ndarray, np.ndarray]:
    resblocks = get_resblocks(model)
    captured: List[np.ndarray] = []

    def hook(_m, _i, output):
        if output.dim() == 3:
            cls = output[0] if output.shape[0] >= output.shape[1] else output[:, 0]
        elif output.dim() == 2:
            cls = output
        else:
            raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
        captured.append(cls.detach().cpu().to(torch.float32).numpy())

    handle = resblocks[11].register_forward_hook(hook)
    try:
        valid: List[int] = []
        pending: List[Tuple[int, torch.Tensor]] = []
        for i, p in enumerate(paths):
            t = load_and_preprocess(Path(p))
            if t is not None:
                pending.append((i, t))
        logger.info("  loaded %d/%d valid frames", len(pending), len(paths))
        for j in range(0, len(pending), batch_size):
            chunk = pending[j: j + batch_size]
            batch_idx = [c[0] for c in chunk]
            batch = torch.stack([c[1] for c in chunk]).to(device, non_blocking=True)
            with torch.inference_mode():
                _ = model.backbone(batch)
            valid.extend(batch_idx)
            if (j // batch_size) % 5 == 0:
                logger.info("  processed batch %d/%d", j // batch_size + 1,
                            (len(pending) + batch_size - 1) // batch_size)
        feats = np.concatenate(captured, axis=0) if captured else np.zeros((0, 0), dtype=np.float32)
    finally:
        handle.remove()
    return feats, np.array(valid, dtype=np.int64)


def extract_phase(device: torch.device) -> None:
    """Extract L11 CLS features for each candidate; skip if already cached."""
    sampled = pd.read_csv(SAMPLED_CSV).iloc[:800].reset_index(drop=True)
    paths = sampled["local_path"].tolist()

    # P8A reference: reuse prior cache if present, else extract.
    p8a_cache = FEAT_DIR / "intermediate__P8A_REF__layer11__n800.npz"
    if not p8a_cache.exists():
        if P8A_PRIOR_CACHE.exists():
            import shutil
            shutil.copy(P8A_PRIOR_CACHE, p8a_cache)
            logger.info("[P8A_REF] copied prior cache from %s", P8A_PRIOR_CACHE)
        elif P8A_REF_CKPT.exists():
            logger.info("[P8A_REF] extracting from %s", P8A_REF_CKPT)
            model = load_effort_model(P8A_REF_CKPT, device, lora_cfg=None)
            feats, valid = extract_l11(model, paths, device)
            np.savez_compressed(p8a_cache, features=feats.astype(np.float32), valid_idx=valid)
            del model
        else:
            raise FileNotFoundError("P8A_REF ckpt not available; cannot establish baseline.")

    for label, info in CANDIDATES.items():
        cache = FEAT_DIR / f"intermediate__{label}__layer11__n800.npz"
        if cache.exists():
            logger.info("[%s] cached, skipping", label)
            continue
        ckpt = info["ckpt"]
        if not ckpt.exists():
            logger.warning("[%s] ckpt missing: %s", label, ckpt)
            continue
        if ckpt.stat().st_size < 100_000_000:
            logger.warning("[%s] ckpt too small (%d bytes), skipping", label, ckpt.stat().st_size)
            continue
        logger.info("[%s] loading model from %s", label, ckpt)
        try:
            model = load_effort_model(ckpt, device, lora_cfg=info["lora"])
        except Exception as exc:
            logger.error("[%s] load failed: %s", label, exc, exc_info=True)
            continue

        # Smoke: forward one batch, check finite.
        smoke_t = load_and_preprocess(Path(paths[0]))
        if smoke_t is not None:
            with torch.inference_mode():
                out = model.backbone(smoke_t.unsqueeze(0).to(device))
            if isinstance(out, dict) and "pooler_output" in out:
                finite = bool(torch.isfinite(out["pooler_output"]).all())
                logger.info("  [%s] smoke finite=%s pool_shape=%s",
                            label, finite, tuple(out["pooler_output"].shape))
            else:
                logger.info("  [%s] smoke ran; output type=%s", label, type(out).__name__)

        feats, valid = extract_l11(model, paths, device)
        if feats.size == 0:
            logger.warning("[%s] empty features after extract", label)
            continue
        nonfinite = (~np.isfinite(feats)).sum()
        if nonfinite > 0:
            logger.warning("[%s] %d non-finite feature values!", label, nonfinite)
        np.savez_compressed(cache, features=feats.astype(np.float32), valid_idx=valid)
        logger.info("[%s] saved (shape=%s, nonfinite=%d, n_valid=%d)",
                    label, feats.shape, int(nonfinite), len(valid))
        del model


# =============================================================================
# Phase 2 — Per-substrate inv_mean recomputation
# =============================================================================

def chronic_match(s) -> bool:
    if not isinstance(s, str):
        return False
    sl = s.lower()
    return any(p.lower() in sl for p in CHRONIC_6_PATTERNS)


def is_dor_fn(identity_key) -> bool:
    if not isinstance(identity_key, str):
        return False
    return "dor" in identity_key.lower()


def build_panel() -> pd.DataFrame:
    cols = ["gcs_uri", "label", "split", "identity_key",
            "session_id", "video_id", "method",
            "local_path", "width", "height", "is_no_face",
            "face_pixel_area"]
    df = pd.read_csv(SAMPLED_CSV, usecols=cols).iloc[:800].reset_index(drop=True)
    df["row_ix"] = np.arange(len(df))
    df["is_dor"] = df["identity_key"].apply(is_dor_fn).astype(int)
    df["is_chronic_6"] = df["identity_key"].apply(chronic_match).astype(int)
    df["is_lockbox"] = (df["split"] == "lockbox").astype(int)
    df["is_real_vs_fake"] = (df["label"] == "fake").astype(int)

    # Attach IQ features from atlas parquet.
    if ATLAS_PARQUET.exists():
        atlas = pd.read_parquet(ATLAS_PARQUET)[["frame_path"] + PRIMARY_6_IQ].copy()
        atlas = atlas.drop_duplicates(subset=["frame_path"], keep="first")
        df = df.merge(atlas, left_on="gcs_uri", right_on="frame_path", how="left")
        assert len(df) == 800, f"merge inflated panel to {len(df)}"
    else:
        for c in PRIMARY_6_IQ:
            df[c] = np.nan

    # Inline IQ for missing rows (matches A3).
    missing = df["lap_var"].isna()
    if missing.any():
        import cv2
        for ix in df.index[missing]:
            try:
                img = cv2.imread(str(df.loc[ix, "local_path"]), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                luma_mean = float(hsv[..., 2].astype(np.float32).mean())
                sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edge_mag = float(np.sqrt(sx**2 + sy**2).mean())
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
                color_b_dev = float(np.abs(lab[..., 2] - 128.0).mean())
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                skin = ((ycrcb[..., 0] > 80) & (ycrcb[..., 1] >= 133)
                        & (ycrcb[..., 1] <= 173) & (ycrcb[..., 2] >= 77)
                        & (ycrcb[..., 2] <= 127))
                df.loc[ix, "lap_var"] = lap_var
                df.loc[ix, "min_dim"] = float(min(h, w))
                df.loc[ix, "luma_mean"] = luma_mean
                df.loc[ix, "edge_mag"] = edge_mag
                df.loc[ix, "color_b_dev"] = color_b_dev
                df.loc[ix, "skin_frac"] = float(skin.mean())
            except Exception:
                pass

    nan_min = df["min_dim"].isna() if "min_dim" in df.columns else None
    if nan_min is not None and nan_min.any():
        df.loc[nan_min, "min_dim"] = df.loc[nan_min, ["width", "height"]].min(axis=1)

    return df


def fit_probe_auc(X, y, n_splits=5, seed=0) -> Tuple[float, int, int]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y, dtype=int)
    if len(y) < 20:
        return float("nan"), len(y), int(y.sum())
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return float("nan"), len(y), int(y.sum())

    actual_splits = min(n_splits, int(y.sum()), int(len(y) - y.sum()))
    if actual_splits < 3:
        return float("nan"), len(y), int(y.sum())

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=3000, n_jobs=1, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof)), len(y), int(y.sum())


def per_substrate_inv_mean(panel: pd.DataFrame, ckpt_to_npz: Dict[str, Path]) -> pd.DataFrame:
    """Recompute inv_mean per ckpt per substrate slice. Matches A3 exactly."""
    panel = panel.copy()
    panel["lap_var_high"] = (panel["lap_var"] > panel["lap_var"].median()).astype(int)
    panel["min_dim_high"] = (panel["min_dim"] > panel["min_dim"].median()).astype(int)
    panel["face_size_high"] = (panel["face_pixel_area"] > panel["face_pixel_area"].median()).astype(int)

    SIGNALS = [
        ("is_real_vs_fake", "is_real_vs_fake"),
        ("is_dor", "is_dor"),
        ("is_chronic_6", "is_chronic_6"),
        ("lap_var_high", "lap_var_high"),
        ("min_dim_high", "min_dim_high"),
        ("face_size_high", "face_size_high"),
    ]
    SHORTCUTS = ["is_dor", "is_chronic_6", "lap_var_high", "min_dim_high", "face_size_high"]

    SLICES = {
        "full_n800":        pd.Series(True, index=panel.index),
        "dev_only":         (panel["split"] == "dev"),
        "lockbox_only":     (panel["split"] == "lockbox"),
        "chronic_6_only":   (panel["is_chronic_6"] == 1),
        "non_chronic_only": (panel["is_chronic_6"] == 0),
    }

    rows = []
    for ckpt_name, npz_path in ckpt_to_npz.items():
        if not Path(npz_path).exists():
            logger.warning("[%s] feature cache missing: %s", ckpt_name, npz_path)
            continue
        blob = np.load(npz_path)
        feats = blob["features"].astype(np.float32)
        valid_idx = blob["valid_idx"].astype(np.int64)
        valid_to_pos = {int(v): r for r, v in enumerate(valid_idx)}
        valid_set = set(valid_to_pos.keys())

        # Audit feature finiteness once per ckpt.
        n_total = int(feats.size)
        n_nonfinite = int((~np.isfinite(feats)).sum())
        if n_nonfinite > 0:
            logger.warning("[%s] non-finite features %d/%d", ckpt_name, n_nonfinite, n_total)

        for slice_name, slice_mask in SLICES.items():
            sub = panel[slice_mask].copy()
            sub = sub[sub["row_ix"].isin(valid_set)].reset_index(drop=True)
            if len(sub) < 30:
                logger.info("  [%s] slice=%s too small (n=%d), skipping",
                            ckpt_name, slice_name, len(sub))
                continue
            sel_pos = [valid_to_pos[int(rx)] for rx in sub["row_ix"].values]
            X = feats[sel_pos]
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

            sig_aucs = {}
            for sig_label, sig_col in SIGNALS:
                if sig_col not in sub.columns:
                    sig_aucs[sig_label] = float("nan")
                    continue
                y = sub[sig_col].values.astype(int)
                auc, _, _ = fit_probe_auc(X, y)
                sig_aucs[sig_label] = auc

            forgery_auc = sig_aucs.get("is_real_vs_fake", float("nan"))
            shortcut_vals = [sig_aucs.get(s, float("nan")) for s in SHORTCUTS]
            shortcut_vals_clean = [v for v in shortcut_vals if not (v is None or np.isnan(v))]
            mean_shortcut = (float(np.mean(shortcut_vals_clean))
                             if shortcut_vals_clean else float("nan"))
            inv_mean = (forgery_auc - mean_shortcut
                        if (not np.isnan(forgery_auc) and not np.isnan(mean_shortcut))
                        else float("nan"))

            row = {
                "ckpt": ckpt_name,
                "slice": slice_name,
                "n": len(sub),
                "n_fake": int(sub["is_real_vs_fake"].sum()),
                "n_real": int(len(sub) - sub["is_real_vs_fake"].sum()),
                "forgery_auc": forgery_auc,
                "mean_shortcut_auc": mean_shortcut,
                "inv_mean": inv_mean,
            }
            for s in SHORTCUTS:
                row[f"auc_{s}"] = sig_aucs.get(s, float("nan"))
            rows.append(row)
            logger.info("  [%s] slice=%-18s n=%3d forgery=%.4f inv_mean=%+.4f",
                        ckpt_name, slice_name, len(sub), forgery_auc, inv_mean)

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["extract", "probe", "all"], default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s")

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("device=%s", device)

    if args.phase in ("extract", "all"):
        logger.info("=== Phase 1: feature extraction ===")
        extract_phase(device)

    if args.phase in ("probe", "all"):
        logger.info("=== Phase 2: per-substrate inv_mean probes ===")
        panel = build_panel()
        logger.info("panel: %d rows (split=dev %d / split=lockbox %d; chronic_6 %d / non %d)",
                    len(panel), int((panel["split"] == "dev").sum()),
                    int((panel["split"] == "lockbox").sum()),
                    int((panel["is_chronic_6"] == 1).sum()),
                    int((panel["is_chronic_6"] == 0).sum()))

        # Include P8A reference + all 4 candidates.
        ckpt_to_npz: Dict[str, Path] = OrderedDict()
        ckpt_to_npz["P8A_REF"] = FEAT_DIR / "intermediate__P8A_REF__layer11__n800.npz"
        for label in CANDIDATES.keys():
            ckpt_to_npz[label] = FEAT_DIR / f"intermediate__{label}__layer11__n800.npz"

        inv = per_substrate_inv_mean(panel, ckpt_to_npz)
        inv_csv = OUT / "per_substrate_inv_mean.csv"
        inv.to_csv(inv_csv, index=False)
        logger.info("wrote %s", inv_csv)

        # Δ-vs-P8A pivot
        wide_inv = inv.pivot(index="slice", columns="ckpt", values="inv_mean")
        wide_forg = inv.pivot(index="slice", columns="ckpt", values="forgery_auc")
        wide_shortcut = inv.pivot(index="slice", columns="ckpt", values="mean_shortcut_auc")

        # Reorder columns for readability
        ordered_ckpts = ["P8A_REF"] + list(CANDIDATES.keys())
        col_order = [c for c in ordered_ckpts if c in wide_inv.columns]
        wide_inv = wide_inv[col_order]
        wide_forg = wide_forg[col_order]
        wide_shortcut = wide_shortcut[col_order]

        # Δ vs P8A
        delta_rows = []
        slice_order = ["full_n800", "dev_only", "lockbox_only", "non_chronic_only", "chronic_6_only"]
        for sl in slice_order:
            if sl not in wide_inv.index:
                continue
            r = {"slice": sl}
            p8a_v = wide_inv.loc[sl, "P8A_REF"]
            for c in col_order:
                if c == "P8A_REF":
                    r[c] = 0.0
                else:
                    r[c] = float(wide_inv.loc[sl, c] - p8a_v)
            delta_rows.append(r)
        delta_inv = pd.DataFrame(delta_rows).set_index("slice")
        delta_inv.to_csv(OUT / "delta_inv_mean_vs_p8a.csv")
        logger.info("wrote %s", OUT / "delta_inv_mean_vs_p8a.csv")

        # Tables for FACTS doc
        print("\n=== Absolute inv_mean per ckpt per slice (P8A_REF + 4 candidates) ===")
        print(wide_inv.to_string(float_format="%+.4f"))
        print("\n=== Δ inv_mean vs P8A_REF baseline ===")
        print(delta_inv.to_string(float_format="%+.4f"))
        print("\n=== forgery_AUC per ckpt per slice ===")
        print(wide_forg.to_string(float_format="%.4f"))
        print("\n=== mean_shortcut_AUC per ckpt per slice ===")
        print(wide_shortcut.to_string(float_format="%.4f"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
