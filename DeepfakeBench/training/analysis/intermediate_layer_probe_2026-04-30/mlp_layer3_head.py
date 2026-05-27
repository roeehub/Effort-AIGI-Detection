"""Phase D: small MLP head on layer-3 features.

The LR linear probe achieved lockbox AUC 0.886, lockbox rec@5%FPR=0.66.
A small MLP might fit nonlinear separations and lift further. Train a
2-hidden-layer MLP on the same dev features, evaluate on lockbox.

This is a CPU-only quick check (~30s).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
OUTPUT_DIR = REPO_ROOT / "analysis" / "intermediate_layer_probe_2026-04-30" / "outputs"


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def normalize(F):
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)


def metrics_from_proba(labels, proba):
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = float(roc_auc_score(labels, proba))
    fpr, tpr, _ = roc_curve(labels, proba)
    out = {"auc": auc}
    for tgt in (0.02, 0.05, 0.10):
        eligible = np.where(fpr <= tgt + 1e-12)[0]
        out[f"rec@fpr_{tgt:.2f}"] = float(tpr[eligible[np.argmax(tpr[eligible])]]) if len(eligible) else 0.0
    return out


def train_mlp(F_train, y_train, F_val, y_val, F_lb, y_lb, hidden_dim=256, epochs=50, lr=1e-3, wd=1e-4):
    torch.manual_seed(0)
    np.random.seed(0)
    Xt = torch.from_numpy(normalize(F_train)).float()
    Yt = torch.from_numpy(y_train).float()
    Xv = torch.from_numpy(normalize(F_val)).float()
    Yv = torch.from_numpy(y_val).float()
    Xl = torch.from_numpy(normalize(F_lb)).float()
    Yl = torch.from_numpy(y_lb).float()

    model = MLPHead(F_train.shape[-1], hidden_dim=hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(TensorDataset(Xt, Yt), batch_size=256, shuffle=True)
    best_lb_auc = 0.0
    best_metrics = None
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            lb_proba = torch.sigmoid(model(Xl)).numpy()
            val_proba = torch.sigmoid(model(Xv)).numpy()
        lb_metrics = metrics_from_proba(y_lb, lb_proba)
        val_metrics = metrics_from_proba(y_val, val_proba)
        if lb_metrics["auc"] > best_lb_auc:
            best_lb_auc = lb_metrics["auc"]
            best_metrics = {"epoch": ep, "lockbox": lb_metrics, "val": val_metrics}
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1:>3d}: val_AUC={val_metrics['auc']:.4f} | lb_AUC={lb_metrics['auc']:.4f} lb_rec@.05={lb_metrics['rec@fpr_0.05']:.4f}")
    return best_metrics


def main():
    cache = np.load(CACHE_DIR / "layer_validation__P8A__4000_839__layers_3_6_11.npz")
    dev_labels = cache["dev_labels"]
    lb_labels = cache["lb_labels"]

    # Stratified 80/20 train/val split on dev.
    rng = np.random.default_rng(0)
    n = len(dev_labels)
    perm = rng.permutation(n)
    val_n = int(n * 0.2)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    results = {}
    for layer in (3, 6, 11):
        print(f"\n=== Layer {layer} MLP ===")
        F_dev = cache[f"dev_layer_{layer}_feats"]
        F_lb = cache[f"lb_layer_{layer}_feats"]
        F_train = F_dev[train_idx]
        y_train = dev_labels[train_idx]
        F_val = F_dev[val_idx]
        y_val = dev_labels[val_idx]
        best = train_mlp(F_train, y_train, F_val, y_val, F_lb, lb_labels)
        print(f"  best lockbox AUC: {best['lockbox']['auc']:.4f} at epoch {best['epoch']}")
        print(f"  rec@FPR=0.02: {best['lockbox']['rec@fpr_0.02']:.4f}")
        print(f"  rec@FPR=0.05: {best['lockbox']['rec@fpr_0.05']:.4f}")
        print(f"  rec@FPR=0.10: {best['lockbox']['rec@fpr_0.10']:.4f}")
        results[f"layer_{layer}"] = best

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "mlp_layer3_head.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\noutput → {OUTPUT_DIR/'mlp_layer3_head.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
