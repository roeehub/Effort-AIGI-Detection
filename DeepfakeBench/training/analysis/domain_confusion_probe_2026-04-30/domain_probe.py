"""Domain-confusion linear probe (Slot 2 GRL validation).

Frozen-feature 3-way logistic regression: predict quality_domain from
[CLS] features. If the gradient-reversal head trained in Slot 2 successfully
removed quality-domain information from the backbone representation, the
linear probe's macro-OVR AUC should drop from ~1.0 (P8A baseline, fully
separable) toward 0.33 (3-balanced-class chance) on Slot 2's features.

How to read the output:
- `macro_ovr_auc[P8A]` >> `macro_ovr_auc[SLOT2]`: GRL succeeded at flattening
  the domain manifold. The closer Slot 2 is to ~0.33-0.50, the more
  domain-invariant the features are.
- If both are ~1.0: GRL did not bite. The reversed-gradient head failed
  to push the backbone away from the shortcut feature. Cross-reference
  with `quality_domain_loss_raw` from training W&B to corroborate.
- Confusion matrices show WHICH pair of domains collapsed. Off-diagonal
  mass between domain 1 (webcam/teams-passthrough) and domain 2
  (studio/visomaster/deeplive) is the load-bearing direction; this is the
  shortcut documented in project_signature_shortcut_finding.md.

Source data: by default reads `analysis/lockbox_tagging/full_tags_2026-04-27.parquet`,
which contains `teams_*` (domain 1) and `deeplive_enhanced` (domain 2). Domain
0 (df40) is generally absent from lockbox; users can either accept a 2-way
probe or supply a wider parquet via --parquet. The script warns if a domain
has fewer than --sample_per_domain rows available.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import yaml
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectors import DETECTOR  # noqa: E402

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

DEFAULT_PARQUET = REPO_ROOT / "analysis" / "lockbox_tagging" / "full_tags_2026-04-27.parquet"
DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"

# Map lockbox parquet method -> quality domain (matches QUALITY_DOMAIN_MAP in
# data/sources/combined_paired.py:65-82).
DEFAULT_METHOD_TO_DOMAIN: Dict[str, int] = {
    # df40 reals / fakes (only present if user supplies an extended parquet)
    "df40_real": 0,
    "df40_fake": 0,
    # external = webcam-codec
    "external_vcd_real": 1,
    "zoom_vcd_real": 1,
    # teams-passthrough = webcam-codec domain
    # (every method whose name starts with "teams_" lands in domain 1)
    "teams_real": 1,
    # deeplive / visomaster studio_capture
    "deeplive_enhanced": 2,
    "deeplive_baseline": 2,
    "visomaster": 2,
    "visomaster_enhanced": 2,
}

logger = logging.getLogger("domain-probe")


# -----------------------------------------------------------------------------
# Checkpoint loader (shared with face_size_invariance).
# -----------------------------------------------------------------------------
def _maybe_download_gcs_checkpoint(ckpt_uri: str, cache_dir: Path) -> Path:
    if not ckpt_uri.startswith("gs://"):
        return Path(ckpt_uri)
    cache_dir.mkdir(parents=True, exist_ok=True)
    blob_name = ckpt_uri[5:].split("/", 1)[1]
    local_path = cache_dir / Path(blob_name).name
    if local_path.exists():
        logger.info("Checkpoint cached at %s", local_path)
        return local_path
    from google.cloud import storage

    bucket_name = ckpt_uri[5:].split("/", 1)[0]
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    if not blob.exists(client=client):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_uri}")
    logger.info("Downloading %s -> %s", ckpt_uri, local_path)
    blob.download_to_filename(str(local_path))
    return local_path


def load_effort_model(
    checkpoint_path: Path,
    detector_config: Path,
    train_config: Path,
    device: torch.device,
) -> torch.nn.Module:
    with open(detector_config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(train_config, "r") as f:
        train_cfg = yaml.safe_load(f)
    cfg.update(train_cfg)

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
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
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Feature extraction (matches batch_inference_gcs.py preprocessing).
# -----------------------------------------------------------------------------
def load_and_preprocess(local_path: Path, resolution: int) -> Optional[torch.Tensor]:
    img_bgr = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_bgr = cv2.resize(img_bgr, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    return transform(img_rgb)


def extract_cls_features(
    model: torch.nn.Module,
    paths: List[Path],
    device: torch.device,
    batch_size: int = 64,
    resolution: int = 224,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (features [N,D], valid_mask [N])."""
    feats: List[np.ndarray] = []
    valid: List[int] = []
    pending: List[Tuple[int, torch.Tensor]] = []
    for i, p in enumerate(paths):
        t = load_and_preprocess(p, resolution)
        if t is None:
            continue
        pending.append((i, t))

    for i in range(0, len(pending), batch_size):
        chunk = pending[i:i + batch_size]
        batch_idx = [c[0] for c in chunk]
        batch = torch.stack([c[1] for c in chunk]).to(device, non_blocking=True)
        with torch.inference_mode():
            outputs = model({"image": batch}, inference=True)
            f = outputs["feat"].detach().cpu().numpy()
        feats.append(f)
        valid.extend(batch_idx)

    feats_arr = np.concatenate(feats, axis=0) if feats else np.zeros((0, 0), dtype=np.float32)
    valid_arr = np.array(valid, dtype=np.int64)
    return feats_arr, valid_arr


def cache_path_for(ckpt_label: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"domain_probe__{ckpt_label}.npz"


# -----------------------------------------------------------------------------
# Sampling.
# -----------------------------------------------------------------------------
def assign_domain_default(method: str) -> Optional[int]:
    """Map a lockbox method string to a quality domain.

    Falls back to:
      - methods starting with `teams_`  -> 1
      - methods containing `deeplive`   -> 2
      - methods containing `visomaster` -> 2
      - methods starting with `df40`    -> 0
    """
    if method in DEFAULT_METHOD_TO_DOMAIN:
        return DEFAULT_METHOD_TO_DOMAIN[method]
    if method.startswith("teams_"):
        return 1
    if "deeplive" in method:
        return 2
    if "visomaster" in method:
        return 2
    if method.startswith("df40"):
        return 0
    return None


def sample_frames(
    parquet_path: Path,
    domains: List[int],
    sample_per_domain: int,
    seed: int = 0,
) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "decode_ok" in df.columns:
        df = df[df["decode_ok"].astype(bool)]
    df["quality_domain"] = df["method"].map(assign_domain_default)
    rng = np.random.default_rng(seed)
    picks = []
    for d in domains:
        sub = df[df["quality_domain"] == d]
        n_avail = len(sub)
        if n_avail == 0:
            logger.warning("Domain %d has 0 rows in parquet — skipping", d)
            continue
        n = min(n_avail, sample_per_domain)
        if n_avail < sample_per_domain:
            logger.warning("Domain %d has only %d rows (< %d requested)", d, n_avail, sample_per_domain)
        idx = rng.choice(sub.index.values, size=n, replace=False)
        picks.append(df.loc[idx])
    sampled = pd.concat(picks, axis=0).reset_index(drop=True)
    return sampled


# -----------------------------------------------------------------------------
# Linear probe.
# -----------------------------------------------------------------------------
def train_and_eval_probe(
    features: np.ndarray,
    domain_labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        confusion_matrix,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    classes = sorted(np.unique(domain_labels).tolist())
    if len(classes) < 2:
        raise ValueError(f"Need at least 2 unique domains, got: {classes}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics: List[dict] = []
    all_proba = np.zeros((len(domain_labels), len(classes)))
    all_pred = np.zeros(len(domain_labels), dtype=int)

    for fold_id, (tr_idx, te_idx) in enumerate(skf.split(features, domain_labels)):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(features[tr_idx])
        x_te = scaler.transform(features[te_idx])
        # n_jobs=1 explicitly (sklearn n_jobs=-1 caused 3 reboots; see
        # feedback_sklearn_njobs.md).
        clf = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=2000,
            n_jobs=1,
            random_state=seed,
        )
        clf.fit(x_tr, domain_labels[tr_idx])
        pred_proba = clf.predict_proba(x_te)
        # Align proba columns to global class order
        col_idx_for_global = [list(clf.classes_).index(c) if c in clf.classes_ else None for c in classes]
        proba_aligned = np.zeros((len(te_idx), len(classes)))
        for j, src_col in enumerate(col_idx_for_global):
            if src_col is not None:
                proba_aligned[:, j] = pred_proba[:, src_col]
        all_proba[te_idx, :] = proba_aligned
        all_pred[te_idx] = clf.predict(x_te)

        # Per-class one-vs-rest AUC for this fold
        per_class_auc = {}
        for j, c in enumerate(classes):
            y_bin = (domain_labels[te_idx] == c).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                per_class_auc[int(c)] = float("nan")
            else:
                per_class_auc[int(c)] = float(roc_auc_score(y_bin, proba_aligned[:, j]))
        fold_metrics.append({"fold": fold_id, "per_class_auc": per_class_auc})

    # Aggregate metrics on out-of-fold predictions
    cm = confusion_matrix(domain_labels, all_pred, labels=classes)
    macro_ovr = []
    per_class_auc_overall = {}
    for j, c in enumerate(classes):
        y_bin = (domain_labels == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            per_class_auc_overall[int(c)] = float("nan")
        else:
            auc_c = float(roc_auc_score(y_bin, all_proba[:, j]))
            per_class_auc_overall[int(c)] = auc_c
            macro_ovr.append(auc_c)
    macro_auc = float(np.mean(macro_ovr)) if macro_ovr else float("nan")

    return {
        "classes": classes,
        "macro_ovr_auc": macro_auc,
        "per_class_auc": per_class_auc_overall,
        "confusion_matrix": cm.tolist(),
        "fold_metrics": fold_metrics,
        "n_samples_per_class": {int(c): int((domain_labels == c).sum()) for c in classes},
    }


# -----------------------------------------------------------------------------
# Plotting.
# -----------------------------------------------------------------------------
def plot_results(
    results_per_ckpt: Dict[str, dict],
    output_dir: Path,
) -> None:
    """Two outputs: (1) confusion-matrix grid (one per ckpt), (2) macro-AUC bar chart."""
    n_models = len(results_per_ckpt)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]
    for ax, (label, res) in zip(axes, results_per_ckpt.items()):
        cm = np.array(res["confusion_matrix"])
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=8,
                )
        ax.set_xticks(range(len(res["classes"])), [f"d={c}" for c in res["classes"]])
        ax.set_yticks(range(len(res["classes"])), [f"d={c}" for c in res["classes"]])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{label}\nmacro-OVR AUC = {res['macro_ovr_auc']:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=120)
    plt.close(fig)

    # Macro-AUC bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(results_per_ckpt.keys())
    aucs = [results_per_ckpt[l]["macro_ovr_auc"] for l in labels]
    bars = ax.bar(labels, aucs, color=["#4477aa", "#cc6677", "#117733", "#ddcc77"][: len(labels)])
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=11)
    ax.axhline(1.0 / len(labels), color="grey", linestyle="--", alpha=0.5,
               label=f"chance ({1.0 / len(results_per_ckpt[labels[0]]['classes']):.2f})")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("3-way macro-OVR AUC (lower = more domain-invariant)")
    ax.set_title("Linear probe domain-confusion AUC per checkpoint")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "macro_auc_bar.png", dpi=120)
    plt.close(fig)


def write_csv(results_per_ckpt: Dict[str, dict], output_dir: Path) -> None:
    rows = []
    for label, res in results_per_ckpt.items():
        for fm in res["fold_metrics"]:
            for c, auc in fm["per_class_auc"].items():
                rows.append({"checkpoint": label, "fold": fm["fold"], "domain": c, "auc_ovr": auc})
        # Also include the OOF macro AUC summary as a row with fold=-1
        for c, auc in res["per_class_auc"].items():
            rows.append({"checkpoint": label, "fold": -1, "domain": c, "auc_ovr": auc})
    pd.DataFrame(rows).to_csv(output_dir / "per_fold_per_class_auc.csv", index=False)


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------
def parse_ckpt_args(raw_list: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in raw_list:
        if "=" not in raw:
            raise ValueError(f"--ckpts entries must be LABEL=URI; got: {raw}")
        label, uri = raw.split("=", 1)
        out[label.strip()] = uri.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="3-way domain-confusion linear probe on frozen [CLS] features."
    )
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="Space-separated LABEL=URI entries, e.g. P8A=gs://... SLOT2=gs://...")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET,
                    help="Parquet with at least columns: method, local_path, decode_ok")
    ap.add_argument("--sample_per_domain", type=int, default=300)
    ap.add_argument("--domains", nargs="+", type=int, default=[0, 1, 2],
                    help="Domain IDs to probe over (default: 0,1,2)")
    ap.add_argument("--n_splits", type=int, default=5,
                    help="Number of stratified k-fold splits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--cache_dir", type=Path,
                    default=REPO_ROOT / "analysis" / "_features_cache_2026-04-30")
    ap.add_argument("--detector_config", type=Path, default=DEFAULT_DETECTOR_CONFIG)
    ap.add_argument("--train_config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    ckpts = parse_ckpt_args(args.ckpts)

    # Sample frames once and reuse across all checkpoints.
    sampled_df = sample_frames(
        parquet_path=args.parquet,
        domains=args.domains,
        sample_per_domain=args.sample_per_domain,
        seed=args.seed,
    )
    logger.info("Sampled %d frames across domains=%s", len(sampled_df), sorted(sampled_df["quality_domain"].unique()))
    if "local_path" not in sampled_df.columns:
        raise RuntimeError("Parquet missing `local_path` column.")

    # Filter for files that actually exist on disk.
    sampled_df["exists"] = sampled_df["local_path"].apply(lambda p: os.path.exists(str(p)))
    n_missing = (~sampled_df["exists"]).sum()
    if n_missing:
        logger.warning("Dropping %d rows whose local_path does not exist", n_missing)
        sampled_df = sampled_df[sampled_df["exists"]].reset_index(drop=True)

    paths = [Path(p) for p in sampled_df["local_path"].tolist()]
    domain_labels_full = sampled_df["quality_domain"].astype(int).to_numpy()

    results_per_ckpt: Dict[str, dict] = {}
    for label, uri in ckpts.items():
        cache_p = cache_path_for(label, args.cache_dir)
        if cache_p.exists():
            logger.info("[%s] features already cached at %s", label, cache_p)
            cached = np.load(cache_p, allow_pickle=False)
            features = cached["features"]
            valid_idx = cached["valid_idx"]
        else:
            logger.info("[%s] loading checkpoint and extracting features", label)
            ckpt_path = _maybe_download_gcs_checkpoint(uri, args.cache_dir)
            model = load_effort_model(ckpt_path, args.detector_config, args.train_config, device)
            features, valid_idx = extract_cls_features(
                model=model,
                paths=paths,
                device=device,
                batch_size=args.batch_size,
                resolution=args.resolution,
            )
            np.savez_compressed(cache_p, features=features, valid_idx=valid_idx)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        domain_labels = domain_labels_full[valid_idx]
        logger.info("[%s] features shape=%s, n=%d", label, features.shape, len(domain_labels))

        result = train_and_eval_probe(
            features=features,
            domain_labels=domain_labels,
            n_splits=args.n_splits,
            seed=args.seed,
        )
        results_per_ckpt[label] = result

    plot_results(results_per_ckpt, args.output_dir)
    write_csv(results_per_ckpt, args.output_dir)

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "ckpts": ckpts,
                "domains": args.domains,
                "sample_per_domain": args.sample_per_domain,
                "n_splits": args.n_splits,
                "results": results_per_ckpt,
            },
            f, indent=2,
        )

    print()
    print("=" * 70)
    print("DOMAIN-CONFUSION LINEAR PROBE")
    print("=" * 70)
    for label, res in results_per_ckpt.items():
        print(f"  {label:<10} macro-OVR AUC = {res['macro_ovr_auc']:.4f}   "
              f"(per-class: " + ", ".join(
                  f"d{c}={res['per_class_auc'][c]:.3f}" for c in res["classes"]
              ) + ")")
    print()
    print("Headline: lower macro-AUC on Slot 2 (vs P8A) means GRL succeeded at")
    print("removing quality-domain information from the backbone representation.")
    print()
    print(f"CSV/PNG/JSON written under: {args.output_dir}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
