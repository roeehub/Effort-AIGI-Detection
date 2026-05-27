"""D3 cross-substrate probe: encoder substrate-agnosticism test (2026-05-12).

Question
--------
Is the encoder's separation direction substrate-agnostic (transfers cleanly
between dev and lockbox) or substrate-specific?

Method
------
For each ckpt in {P8A, E2B, T5C_periodic_step3500, T3_S1_step1500}:

  Tier 1 — uses cached 800-frame triptych L11 features (713 dev + 87 lockbox):

    1. Direction 1: train LR on DEV features, evaluate AUC on LOCKBOX features.
    2. Direction 2: train LR on LOCKBOX features, evaluate AUC on DEV features.
    3. In-sample baseline: 5-fold CV AUC on each split alone.
    4. 95% bootstrap CIs on transfer AUCs (100 resamples of the test set).

  Tier 2 — only if Tier 1 produces AUC in [0.85, 0.95] (ambiguous band) for at
  least one ckpt × direction. Per task brief, Tier 2 extends features via the
  existing extraction script.

Constraints
-----------
- n_jobs=1 per memory `feedback_sklearn_njobs.md`.
- Deterministic via fixed seed (42).
- Inputs: cached features at
    analysis/iq_perlayer_probe_2026-05-08/_cache/intermediate__{LABEL}__layer11__n800.npz
  joined with split column from
    analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/sampled_frames.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "analysis" / "iq_perlayer_probe_2026-05-08" / "_cache"
TRIPTYCH_CSV = (
    REPO_ROOT / "analysis" / "embedding_triptych_2026-04-30"
    / "outputs" / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"
)
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPTS = [
    ("P8A", CACHE_DIR / "intermediate__P8A__layer11__n800.npz"),
    ("E2B", CACHE_DIR / "intermediate__E2B__layer11__n800.npz"),
    ("T5C_periodic_step3500",
     CACHE_DIR / "intermediate__T5C_periodic_step3500__layer11__n800.npz"),
    ("T3_S1_step1500",
     CACHE_DIR / "intermediate__T3_S1_step1500__layer11__n800.npz"),
]

SEED = 42
N_BOOTSTRAP = 100
LR_KW = dict(C=1.0, max_iter=2000, n_jobs=1, solver="lbfgs")


def load_features(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (features, valid_idx) — valid_idx maps each feature row to the
    triptych CSV row index (0..799).
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing cached features: {npz_path}")
    arr = np.load(str(npz_path), allow_pickle=True)
    feats = arr["features"].astype(np.float32)
    valid_idx = arr["valid_idx"].astype(np.int64)
    if feats.shape[0] != valid_idx.shape[0]:
        raise ValueError(
            f"features.shape[0]={feats.shape[0]} != valid_idx.shape[0]={valid_idx.shape[0]}"
        )
    return feats, valid_idx


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Returns (auc, ci_low, ci_high) at 95% via resampling of the test set."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    auc_full = roc_auc_score(y_true, y_score)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if yt.min() == yt.max():
            # Degenerate bootstrap sample (single class) — skip
            continue
        boots.append(roc_auc_score(yt, ys))
    if len(boots) < 2:
        return auc_full, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(auc_full), float(lo), float(hi)


def cv_auc(
    X: np.ndarray, y: np.ndarray, seed: int = 42
) -> Tuple[float, float]:
    """5-fold stratified CV AUC. Returns (mean, std)."""
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    n_per_class = int(min(np.bincount(y)))
    # Cap to as many folds as the minority class size, capped at 5
    k = int(min(5, n_per_class))
    if k < 2:
        return float("nan"), float("nan")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    lr = LogisticRegression(**LR_KW)
    scores = cross_val_score(lr, X, y, scoring="roc_auc", cv=skf, n_jobs=1)
    return float(scores.mean()), float(scores.std())


def main() -> int:
    # 1. Load triptych metadata.
    if not TRIPTYCH_CSV.exists():
        print(f"FATAL: missing triptych CSV {TRIPTYCH_CSV}", file=sys.stderr)
        return 1
    meta = pd.read_csv(TRIPTYCH_CSV)
    print(f"[meta] loaded {len(meta)} rows; split counts: "
          f"{meta['split'].value_counts().to_dict()}; label counts: "
          f"{meta['label'].value_counts().to_dict()}")

    if len(meta) != 800:
        print(f"WARN: triptych meta has {len(meta)} rows, expected 800",
              file=sys.stderr)

    # Binary label: real=0, fake=1
    label_map = {"real": 0, "fake": 1}
    meta["y"] = meta["label"].map(label_map).astype(np.int64)
    if meta["y"].isna().any():
        bad = meta[meta["y"].isna()]["label"].unique()
        print(f"FATAL: unmapped labels {bad}", file=sys.stderr)
        return 1

    # split mask (after we know valid_idx per ckpt)
    is_dev = (meta["split"] == "dev").to_numpy()
    is_lockbox = (meta["split"] == "lockbox").to_numpy()
    y_all = meta["y"].to_numpy()

    transfer_rows = []
    insample_rows = []

    for label, path in CKPTS:
        print(f"\n[ckpt={label}] loading {path.name}")
        try:
            feats, vi = load_features(path)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        # vi indexes into triptych metadata rows
        dev_mask = is_dev[vi]
        lb_mask = is_lockbox[vi]
        y_v = y_all[vi]

        X_dev = feats[dev_mask]
        y_dev = y_v[dev_mask]
        X_lb = feats[lb_mask]
        y_lb = y_v[lb_mask]

        print(f"  n_dev={X_dev.shape[0]} (real={int((y_dev==0).sum())}, "
              f"fake={int((y_dev==1).sum())}), "
              f"n_lockbox={X_lb.shape[0]} (real={int((y_lb==0).sum())}, "
              f"fake={int((y_lb==1).sum())})")

        # Direction 1: train DEV → test LOCKBOX
        lr_dev = LogisticRegression(**LR_KW)
        lr_dev.fit(X_dev, y_dev)
        scores_d2l = lr_dev.decision_function(X_lb)
        auc_d2l, lo_d2l, hi_d2l = bootstrap_auc_ci(
            y_lb, scores_d2l, n_boot=N_BOOTSTRAP, seed=SEED
        )
        print(f"  DEV->LOCKBOX  AUC={auc_d2l:.4f}  95% CI [{lo_d2l:.4f},{hi_d2l:.4f}]")
        transfer_rows.append({
            "ckpt": label,
            "direction": "DEV_to_LOCKBOX",
            "n_train": int(X_dev.shape[0]),
            "n_test": int(X_lb.shape[0]),
            "auc": auc_d2l,
            "ci_low": lo_d2l,
            "ci_high": hi_d2l,
        })

        # Direction 2: train LOCKBOX → test DEV
        lr_lb = LogisticRegression(**LR_KW)
        lr_lb.fit(X_lb, y_lb)
        scores_l2d = lr_lb.decision_function(X_dev)
        auc_l2d, lo_l2d, hi_l2d = bootstrap_auc_ci(
            y_dev, scores_l2d, n_boot=N_BOOTSTRAP, seed=SEED
        )
        print(f"  LOCKBOX->DEV  AUC={auc_l2d:.4f}  95% CI [{lo_l2d:.4f},{hi_l2d:.4f}]")
        transfer_rows.append({
            "ckpt": label,
            "direction": "LOCKBOX_to_DEV",
            "n_train": int(X_lb.shape[0]),
            "n_test": int(X_dev.shape[0]),
            "auc": auc_l2d,
            "ci_low": lo_l2d,
            "ci_high": hi_l2d,
        })

        # In-sample CV
        cv_dev_mean, cv_dev_std = cv_auc(X_dev, y_dev, seed=SEED)
        cv_lb_mean, cv_lb_std = cv_auc(X_lb, y_lb, seed=SEED)
        print(f"  CV (dev)      AUC={cv_dev_mean:.4f} ± {cv_dev_std:.4f}")
        print(f"  CV (lockbox)  AUC={cv_lb_mean:.4f} ± {cv_lb_std:.4f}")
        insample_rows.append({
            "ckpt": label, "split": "dev",
            "n": int(X_dev.shape[0]),
            "cv_auc_mean": cv_dev_mean, "cv_auc_std": cv_dev_std,
        })
        insample_rows.append({
            "ckpt": label, "split": "lockbox",
            "n": int(X_lb.shape[0]),
            "cv_auc_mean": cv_lb_mean, "cv_auc_std": cv_lb_std,
        })

    # Write CSVs
    df_t = pd.DataFrame(transfer_rows)
    df_b = pd.DataFrame(insample_rows)
    df_t.to_csv(OUT_DIR / "probe_transfer_tier1.csv", index=False)
    df_b.to_csv(OUT_DIR / "in_sample_baselines.csv", index=False)
    print(f"\n[write] {OUT_DIR/'probe_transfer_tier1.csv'} ({len(df_t)} rows)")
    print(f"[write] {OUT_DIR/'in_sample_baselines.csv'} ({len(df_b)} rows)")

    # Tier-2 decision logic: trigger if any transfer AUC ∈ [0.85, 0.95]
    ambiguous = df_t[(df_t["auc"] >= 0.85) & (df_t["auc"] <= 0.95)]
    decision = {
        "tier2_triggered": bool(len(ambiguous) > 0),
        "ambiguous_rows": ambiguous.to_dict(orient="records"),
        "rule": "AUC in [0.85, 0.95] band",
    }
    with open(OUT_DIR / "tier2_decision.json", "w") as f:
        json.dump(decision, f, indent=2)
    print(f"[write] {OUT_DIR/'tier2_decision.json'}: "
          f"trigger={decision['tier2_triggered']}, "
          f"n_ambiguous={len(ambiguous)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
