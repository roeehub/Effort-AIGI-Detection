"""Score T6/T7/T5C ckpts on Dor lockbox + Roy_D + may5 + may6 cohorts.

Cheap CPU/MPS probe for pruning the GPU scorecard candidate list.

Mirrors analysis/cpu_diagnostics_2026-05-09/run_t3_score_probe.py for the
Dor + Roy_D cohorts, and adds may5/may6 cohorts (xinhe production-drift
signature) from analysis/xinhe_cross_camera_audit_2026-05-06/raw/.

Outputs:
  - per_ckpt_cohort_scores.csv      — per-frame scores stacked
  - per_ckpt_cohort_stats.csv       — mean / p25 / p50 / p75 / p90 stats
  - per_ckpt_deployment_fpr.csv     — FPR at deployment τs (0.5, 0.7, 0.9, 0.92)
"""
from __future__ import annotations

import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

THIS_DIR = Path(__file__).resolve().parent
OUT = THIS_DIR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# === Cohort data sources ====================================================
DOR_CACHE_DIR = REPO_ROOT / "analysis" / "dor_encoder_axis_2026-05-08" / "_cache"
DOR_MANIFEST = DOR_CACHE_DIR / "cohort_manifest.csv"
FRAMES_DIR = DOR_CACHE_DIR / "frames"

ROY_D_DIR = REPO_ROOT / "analysis" / "stage2_cpu_2026-05-09" / "_roy_d_frames"

XINHE_DIR = REPO_ROOT / "analysis" / "xinhe_cross_camera_audit_2026-05-06" / "raw"

TRIPTYCH_CSV = (
    REPO_ROOT / "analysis" / "embedding_triptych_2026-04-30"
    / "outputs" / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"
)
CHRONIC_6_PATTERNS = [
    "Roy_D", "PC_Generator", "bla_bla_chow",
    "Md_noyn_Sharker", "dor_shkedi", "healthy_dor",
]

# === Model loader ============================================================
DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# === Candidate ckpts ========================================================
CKPT_BASE = THIS_DIR / "_ckpts"

CANDIDATE_CKPTS: Dict[str, Path] = {
    # T6 — 8 ckpts (smmcn6tj — T3 SLOT1 + face_scale_jitter@0.50, no GRL)
    "T6_periodic_step500":  CKPT_BASE / "t6" / "periodic_effort_20260511_step500_auc0.9780_eer0.0520.pth",
    "T6_periodic_step1500": CKPT_BASE / "t6" / "periodic_effort_20260511_step1500_auc0.9903_eer0.0340.pth",
    "T6_periodic_step2500": CKPT_BASE / "t6" / "periodic_effort_20260511_step2500_auc0.9926_eer0.0240.pth",
    "T6_periodic_step3500": CKPT_BASE / "t6" / "periodic_effort_20260511_step3500_auc0.9918_eer0.0220.pth",
    "T6_periodic_step4500": CKPT_BASE / "t6" / "periodic_effort_20260511_step4500_auc0.9940_eer0.0200.pth",
    "T6_top_n_step3250":    CKPT_BASE / "t6" / "top_n_effort_20260511_step3250_auc0.9940_eer0.0240.pth",
    "T6_top_n_step6000":    CKPT_BASE / "t6" / "top_n_effort_20260511_step6000_auc0.9943_eer0.0200.pth",
    "T6_top_n_step10250":   CKPT_BASE / "t6" / "top_n_effort_20260511_step10250_auc0.9972_eer0.0180.pth",
    # T7 — 7 ckpts (c2ju9fbn — T4 multi-axis-L11-GRL + face_scale_jitter@0.50)
    "T7_periodic_step500":  CKPT_BASE / "t7" / "periodic_effort_20260511_step500_auc0.9782_eer0.0787.pth",
    "T7_periodic_step1500": CKPT_BASE / "t7" / "periodic_effort_20260511_step1500_auc0.9857_eer0.0476.pth",
    "T7_periodic_step2500": CKPT_BASE / "t7" / "periodic_effort_20260511_step2500_auc0.9929_eer0.0311.pth",
    "T7_periodic_step3500": CKPT_BASE / "t7" / "periodic_effort_20260511_step3500_auc0.9874_eer0.0497.pth",
    "T7_periodic_step5000": CKPT_BASE / "t7" / "periodic_effort_20260511_step5000_auc0.9946_eer0.0228.pth",
    "T7_top_n_step4250":    CKPT_BASE / "t7" / "top_n_effort_20260511_step4250_auc0.9944_eer0.0248.pth",
    "T7_top_n_step4750":    CKPT_BASE / "t7" / "top_n_effort_20260511_step4750_auc0.9956_eer0.0207.pth",
    # T5C — 7 ckpts (jrlldtem — T4 multi-axis-L11-GRL + hidden_dim 1024)
    "T5C_periodic_step500":  CKPT_BASE / "t5c" / "periodic_effort_20260511_step500_auc0.9894_eer0.0285.pth",
    "T5C_periodic_step1500": CKPT_BASE / "t5c" / "periodic_effort_20260511_step1500_auc0.9874_eer0.0570.pth",
    "T5C_periodic_step2500": CKPT_BASE / "t5c" / "periodic_effort_20260511_step2500_auc0.9926_eer0.0373.pth",
    "T5C_periodic_step3500": CKPT_BASE / "t5c" / "periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth",
    "T5C_periodic_step5000": CKPT_BASE / "t5c" / "periodic_effort_20260511_step5000_auc0.9937_eer0.0285.pth",
    "T5C_top_n_step2750":    CKPT_BASE / "t5c" / "top_n_effort_20260511_step2750_auc0.9940_eer0.0219.pth",
    "T5C_top_n_step3750":    CKPT_BASE / "t5c" / "top_n_effort_20260511_step3750_auc0.9948_eer0.0154.pth",
}

# === Anchor ckpts ===========================================================
ANCHOR_CKPTS: Dict[str, Path] = {
    "P8A":           REPO_ROOT / "analysis" / "p2_eval_2026-05-08" / "d1_d4_cpu" / "ckpts" / "p8a_step5000.pth",
    "E2B":           REPO_ROOT / "analysis" / "iq_perlayer_probe_2026-05-08" / "_cache" / "e2b_top_n_step3200.pth",
    "T3_S1_step1500": REPO_ROOT / "analysis" / "cpu_diagnostics_2026-05-10" / "_ckpts_t3" / "T3_SLOT1_step1500.pth",
}

logger = logging.getLogger("t67-t5c-probe")


def load_effort_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
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
    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if unexpected:
        logger.info("  dropped %d unexpected keys (e.g. %s)", len(unexpected), unexpected[:3])
    model.eval()
    return model


def load_and_preprocess(local_path: Path, resolution: int = 224) -> torch.Tensor | None:
    import cv2
    img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1))


def score_frames(
    model: torch.nn.Module,
    paths: List[Path],
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    out_probs = np.full(len(paths), np.nan, dtype=np.float64)
    pending: List[Tuple[int, torch.Tensor]] = []

    def flush(batch):
        if not batch:
            return
        idx = [b[0] for b in batch]
        x = torch.stack([b[1] for b in batch]).to(device)
        with torch.no_grad():
            data = {"image": x}
            try:
                pred = model(data, inference=True)
            except TypeError:
                pred = model(data)
            if isinstance(pred, dict):
                logits = pred.get("cls", pred.get("logits", pred.get("score")))
            else:
                logits = pred
            if logits is None:
                raise RuntimeError("model returned no logits")
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        for ii, pp in zip(idx, probs):
            out_probs[ii] = float(pp)

    for i, p in enumerate(paths):
        t = load_and_preprocess(Path(p))
        if t is not None:
            pending.append((i, t))
        if len(pending) >= batch_size:
            flush(pending)
            pending = []
    flush(pending)
    return out_probs


def build_cohort_table() -> pd.DataFrame:
    """Construct one DataFrame with all frame_path × cohort × label rows."""
    rows = []

    # Dor cohort (388 frames in 5 sub-cohorts)
    if DOR_MANIFEST.exists():
        cohort = pd.read_csv(DOR_MANIFEST)
        cohort["local_path"] = cohort["frame_path"].apply(
            lambda u: FRAMES_DIR / u.split("/")[-1]
        )
        cohort = cohort[cohort["local_path"].apply(lambda p: p.exists() and p.stat().st_size > 0)].reset_index(drop=True)
        for _, r in cohort.iterrows():
            rows.append({
                "frame_path": str(r["local_path"]),
                "label": int(r["label"]),
                "cohort": str(r["cohort"]),  # e.g. DOR_REAL_LOCKBOX
                "cohort_group": "DOR",
            })

    # Roy_D — 130-ish frames, all REAL chronic_6
    if ROY_D_DIR.exists():
        roy_paths = sorted(ROY_D_DIR.glob("*.png")) + sorted(ROY_D_DIR.glob("*.jpg")) + sorted(ROY_D_DIR.glob("*.jpeg"))
        for p in roy_paths:
            rows.append({
                "frame_path": str(p),
                "label": 0,
                "cohort": "ROY_D",
                "cohort_group": "ROY_D",
            })

    # may5 + may6 — production-drift cohort, all REAL
    for sub, group in [("may5", "MAY5"), ("may6", "MAY6")]:
        d = XINHE_DIR / sub
        if d.exists():
            ps = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg"))
            for p in ps:
                rows.append({
                    "frame_path": str(p),
                    "label": 0,
                    "cohort": group,
                    "cohort_group": group,
                })

    # chronic_6 (REAL only) from the triptych — 207 dev + 34 lockbox + 41 fake
    # We want REAL chronic_6 reals; chronic_6_real_dev (207) and chronic_6_real_lockbox (34)
    if TRIPTYCH_CSV.exists():
        trip = pd.read_csv(TRIPTYCH_CSV,
                           usecols=["identity_key", "label", "local_path", "split"]).iloc[:800]
        trip["is_chronic"] = trip["identity_key"].astype(str).str.lower().apply(
            lambda s: any(p.lower() in s for p in CHRONIC_6_PATTERNS)
        )
        for _, r in trip[trip["is_chronic"]].iterrows():
            lp = str(r["local_path"])
            if Path(lp).exists():
                lab_int = 1 if r["label"] == "fake" else 0
                cohort_name = f"CHRONIC6_{r['label']}_{r['split']}".upper()
                rows.append({
                    "frame_path": lp,
                    "label": lab_int,
                    "cohort": cohort_name,
                    "cohort_group": "CHRONIC6",
                })

    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("device=%s", device)

    cohort_df = build_cohort_table()
    logger.info("cohort table: %d frames", len(cohort_df))
    logger.info("counts by cohort: %s", cohort_df["cohort"].value_counts().to_dict())

    paths = [Path(p) for p in cohort_df["frame_path"].tolist()]

    # Build ckpts to score (candidates + anchors)
    all_ckpts: Dict[str, Path] = {}
    for label, path in ANCHOR_CKPTS.items():
        if path.exists():
            all_ckpts[label] = path
        else:
            logger.warning("anchor %s missing at %s", label, path)
    for label, path in CANDIDATE_CKPTS.items():
        if path.exists() and path.stat().st_size > 1_000_000_00:  # >100MB sanity check
            all_ckpts[label] = path
        else:
            logger.warning("candidate %s missing or partial at %s (size=%s)", label, path,
                           path.stat().st_size if path.exists() else "N/A")

    logger.info("scoring %d ckpts on %d frames", len(all_ckpts), len(paths))

    # Score per ckpt — append cohort_df column for each
    scored = cohort_df.copy()
    for label, ckpt_path in all_ckpts.items():
        logger.info("[%s] loading", label)
        try:
            model = load_effort_model(ckpt_path, device)
        except Exception as exc:
            logger.error("[%s] load failed: %s", label, exc)
            scored[label] = np.nan
            continue
        logger.info("[%s] scoring %d frames", label, len(paths))
        probs = score_frames(model, paths, device, batch_size=16)
        scored[label] = probs
        del model

    scored.to_csv(OUT / "per_ckpt_cohort_scores.csv", index=False)
    logger.info("wrote %s (%d rows × %d cols)", OUT / "per_ckpt_cohort_scores.csv",
                len(scored), len(scored.columns))

    # ===== Summary stats =====
    score_cols = [c for c in scored.columns if c in all_ckpts]
    stat_rows = []
    fpr_rows = []
    DEPLOYMENT_TAUS = [0.5, 0.7, 0.9, 0.92]
    for ckpt in score_cols:
        for cohort_name, sub in scored.groupby("cohort"):
            s = sub[ckpt].dropna().values.astype(float)
            if len(s) == 0:
                continue
            stat_rows.append({
                "ckpt": ckpt, "cohort": cohort_name, "n": len(s),
                "mean": float(s.mean()),
                "p25": float(np.percentile(s, 25)),
                "p50": float(np.percentile(s, 50)),
                "p75": float(np.percentile(s, 75)),
                "p90": float(np.percentile(s, 90)),
                "std": float(s.std()),
            })
            # For REAL cohorts, compute FPR @ τ. For FAKE cohorts, compute recall @ τ.
            label_count = sub["label"].iloc[0]
            for tau in DEPLOYMENT_TAUS:
                fpr_rows.append({
                    "ckpt": ckpt, "cohort": cohort_name, "tau": tau,
                    "n": len(s), "label": int(label_count),
                    "frac_above_tau": float((s >= tau).sum() / len(s)),
                    "n_above_tau": int((s >= tau).sum()),
                })
    pd.DataFrame(stat_rows).to_csv(OUT / "per_ckpt_cohort_stats.csv", index=False)
    pd.DataFrame(fpr_rows).to_csv(OUT / "per_ckpt_deployment_fpr.csv", index=False)
    logger.info("wrote per_ckpt_cohort_stats.csv + per_ckpt_deployment_fpr.csv")


if __name__ == "__main__":
    raise SystemExit(main())
