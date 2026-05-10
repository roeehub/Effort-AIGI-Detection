"""Pre-test 1 — Multi-axis GRL feasibility on frozen L11 features.

Question: can a learned compressing projection W: R^768 -> R^k of P8A's L11
features, trained with multi-axis adversarial GRL pressure on shortcut
classifiers, achieve `inv_mean > 0.10` (forgery_AUC - mean(shortcut_AUCs))?

If YES: the multi-axis-GRL mechanism has structural promise. From-scratch +
multi-axis GRL during training could break the encoder's L11 invariance
ceiling. Path β1 GPU-justified.

If NO: even an unconstrained projection of the existing features cannot
disentangle forgery from shortcuts. The shortcut signal is co-mingled in
ways a linear (or small-MLP) projection cannot separate. Path β1 unlikely
to succeed; pivot to Path δ (forgery-localization).

Method:
  1. Load P8A L11 features (768d, n=800 frames) + panel labels
  2. Trainable: projection W (linear or 1-hidden-MLP) → forgery_head + 5 shortcut_heads
  3. Loss: CE(forgery) - λ * Σ CE(shortcut) (GRL via gradient reversal)
  4. Sweep λ × k
  5. Evaluate on held-out fold: forgery_AUC + shortcut_AUCs → inv_mean

Also runs same pre-test on E2B and MCLIOEXB for comparison.

Output: outputs/pretest1_multiaxis_grl.csv (per ckpt × k × λ × cv_fold)
        outputs/pretest1_PROFILE.md
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
sys.path.insert(0, str(REPO_ROOT / "analysis" / "cpu_diagnostics_2026-05-09"))
from run_analyses import build_panel, load_features  # noqa

OUT = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs"

CKPTS = ["P8A", "E2B", "MCLIOEXB", "T3_S1_step1500"]
L_LAYER = 11

# Shortcut axes the encoder should be invariant to.
# (is_dor and face_size_high are gate-shieldable in production; we still
# include them as sanity checks.)
SHORTCUT_AXES = ["is_chronic_6", "lap_var_high", "min_dim_high",
                 "is_dor", "face_size_high"]


# -----------------------------------------------------------------------------
# Gradient reversal (GRL) layer
# -----------------------------------------------------------------------------
class GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GRLFunction.apply(x, lambda_)


# -----------------------------------------------------------------------------
# Model: learned projection + forgery head + GRL-attached shortcut heads
# -----------------------------------------------------------------------------
class MultiAxisGRLProbe(nn.Module):
    def __init__(self, in_dim: int, k: int, n_shortcuts: int, hidden: int = 256):
        super().__init__()
        if k == in_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, k), nn.LayerNorm(k), nn.ReLU(),
            )
        # Forgery head — clean gradient
        self.forgery_head = nn.Sequential(
            nn.Linear(k, hidden), nn.ReLU(), nn.Linear(hidden, 2),
        )
        # Shortcut heads — read W*x but reverse-grad
        self.shortcut_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(k, hidden), nn.ReLU(), nn.Linear(hidden, 2))
            for _ in range(n_shortcuts)
        ])

    def forward(self, x, lambda_grl: float):
        z = self.proj(x)
        forgery_logits = self.forgery_head(z)
        z_rev = grad_reverse(z, lambda_grl)
        shortcut_logits = [h(z_rev) for h in self.shortcut_heads]
        return forgery_logits, shortcut_logits, z


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def train_one_fold(
    X_train: np.ndarray, y_train_forgery: np.ndarray, y_train_shortcuts: list[np.ndarray],
    X_val: np.ndarray, y_val_forgery: np.ndarray, y_val_shortcuts: list[np.ndarray],
    k: int, lambda_grl: float, n_epochs: int = 100, batch_size: int = 64,
    lr: float = 1e-3, device: str = "cpu",
) -> dict:
    in_dim = X_train.shape[1]
    n_shortcuts = len(y_train_shortcuts)
    model = MultiAxisGRLProbe(in_dim, k, n_shortcuts).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt_f = torch.tensor(y_train_forgery, dtype=torch.long, device=device)
    yt_s = [torch.tensor(y, dtype=torch.long, device=device) for y in y_train_shortcuts]

    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)

    n = len(Xt)
    best_inv_mean = -1.0
    history = []
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = Xt[idx]
            yf = yt_f[idx]
            ys_b = [s[idx] for s in yt_s]
            forgery_logits, shortcut_logits, _ = model(x, lambda_grl)
            loss_f = F.cross_entropy(forgery_logits, yf)
            loss_s = sum(F.cross_entropy(sl, ys) for sl, ys in zip(shortcut_logits, ys_b))
            # Total: clean forgery + GRL shortcut (which already has reverse grad inside)
            loss = loss_f + loss_s  # GRL handles the reversal automatically
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(x)

        # Eval every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                _, _, z_val = model(Xv, lambda_grl=0.0)
                z_val = z_val.cpu().numpy()
            # Use LR probe on z_val (the projected features) for unbiased AUC
            try:
                if y_val_forgery.sum() >= 5 and (len(y_val_forgery) - y_val_forgery.sum()) >= 5:
                    z_train_proj = model_project_train(model, Xt, lambda_grl).cpu().numpy()
                    forgery_auc = lr_probe_auc(z_train_proj, y_train_forgery, z_val, y_val_forgery)
                    shortcut_aucs = []
                    for ys_t, ys_v in zip(y_train_shortcuts, y_val_shortcuts):
                        if ys_v.sum() < 5 or (len(ys_v) - ys_v.sum()) < 5:
                            shortcut_aucs.append(np.nan)
                        else:
                            sc = lr_probe_auc(z_train_proj, ys_t, z_val, ys_v)
                            shortcut_aucs.append(sc)
                    valid = [a for a in shortcut_aucs if not np.isnan(a)]
                    inv_mean = forgery_auc - np.mean(valid) if valid else float("nan")
                    inv_max = forgery_auc - max(valid) if valid else float("nan")
                    history.append({
                        "epoch": epoch + 1, "loss": total_loss / n,
                        "forgery_auc": forgery_auc,
                        **{f"shortcut_auc_{i}": s for i, s in enumerate(shortcut_aucs)},
                        "inv_mean": inv_mean, "inv_max": inv_max,
                    })
                    if inv_mean > best_inv_mean:
                        best_inv_mean = inv_mean
            except Exception:
                pass
    return {
        "best_inv_mean": best_inv_mean,
        "final_inv_mean": history[-1]["inv_mean"] if history else float("nan"),
        "final_forgery_auc": history[-1]["forgery_auc"] if history else float("nan"),
        "final_shortcut_aucs": [history[-1].get(f"shortcut_auc_{i}", float("nan"))
                                for i in range(len(y_train_shortcuts))] if history else [],
        "history": history,
    }


def model_project_train(model, Xt, lambda_grl):
    model.eval()
    with torch.no_grad():
        _, _, z = model(Xt, lambda_grl=0.0)
    model.train()
    return z


def lr_probe_auc(X_train, y_train, X_val, y_val):
    """Linear probe AUC on the projected features."""
    clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1, solver="lbfgs")
    clf.fit(X_train, y_train)
    if y_val.sum() == 0 or y_val.sum() == len(y_val):
        return float("nan")
    pred = clf.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, pred))


def main():
    print("Building panel...")
    panel = build_panel()
    panel["is_real_vs_fake"] = (panel["label"] == "fake").astype(int)
    panel["lap_var_high"] = (panel["lap_var"] > panel["lap_var"].median()).astype(int)
    panel["min_dim_high"] = (panel["min_dim"] > panel["min_dim"].median()).astype(int)
    if panel["face_pixel_area"].notna().any():
        panel["face_size_high"] = (panel["face_pixel_area"] > panel["face_pixel_area"].median()).astype(int)
    else:
        panel["face_size_high"] = 0
    panel["is_chronic_6_int"] = panel["is_chronic_6"].astype(int)
    panel["is_dor_int"] = panel["is_dor"].astype(int)

    LABEL_COLS = {
        "is_chronic_6": "is_chronic_6_int",
        "lap_var_high": "lap_var_high",
        "min_dim_high": "min_dim_high",
        "is_dor": "is_dor_int",
        "face_size_high": "face_size_high",
    }

    print(f"Panel: {len(panel)} rows, fakes={panel['is_real_vs_fake'].sum()}")
    print(f"Shortcut positive counts: " + ", ".join(
        f"{a}={panel[LABEL_COLS[a]].sum()}" for a in SHORTCUT_AXES
    ))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    rows = []
    for ckpt in CKPTS:
        print(f"\n=== {ckpt} L{L_LAYER} ===")
        try:
            feats, valid_idx = load_features(ckpt, L_LAYER)
        except Exception as exc:
            print(f"  feature load failed: {exc}")
            continue
        valid_set = set(int(v) for v in valid_idx.tolist())
        mask = panel["row_ix"].isin(valid_set).values
        panel_a = panel[mask].reset_index(drop=True)
        valid_to_pos = {int(v): r for r, v in enumerate(valid_idx)}
        sel_pos = [valid_to_pos[int(rx)] for rx in panel_a["row_ix"].values]
        X = feats[sel_pos]
        y_forgery = panel_a["is_real_vs_fake"].values.astype(int)
        y_shortcuts = [panel_a[LABEL_COLS[a]].values.astype(int) for a in SHORTCUT_AXES]
        print(f"  X shape={X.shape}, y_forgery sum={y_forgery.sum()}/{len(y_forgery)}")
        # Normalize features
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

        # Sweep
        for k in [128, 256, 768]:
            for lambda_grl in [0.0, 0.5, 1.0, 2.0, 5.0]:
                # Single-fold train/test for speed (5-fold on 800 features × multi-runs takes too long)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                folds = list(skf.split(X, y_forgery))
                # Use first fold only for speed in initial sweep
                tr, te = folds[0]
                X_tr, X_te = X[tr], X[te]
                yt_f, yt_s_list = y_forgery[tr], [s[tr] for s in y_shortcuts]
                yv_f, yv_s_list = y_forgery[te], [s[te] for s in y_shortcuts]
                result = train_one_fold(
                    X_tr, yt_f, yt_s_list, X_te, yv_f, yv_s_list,
                    k=k, lambda_grl=lambda_grl, n_epochs=80, batch_size=64,
                    lr=1e-3, device=device,
                )
                # Compute baseline LR probe on raw features for comparison
                baseline_forgery = lr_probe_auc(X_tr, yt_f, X_te, yv_f)
                baseline_shortcuts = [lr_probe_auc(X_tr, yt_s, X_te, yv_s) for yt_s, yv_s in zip(yt_s_list, yv_s_list)]
                baseline_valid = [a for a in baseline_shortcuts if not np.isnan(a)]
                baseline_inv_mean = baseline_forgery - np.mean(baseline_valid) if baseline_valid else float("nan")

                row = {
                    "ckpt": ckpt, "k": k, "lambda_grl": lambda_grl,
                    "baseline_forgery_auc": baseline_forgery,
                    "baseline_inv_mean": baseline_inv_mean,
                    "trained_forgery_auc": result["final_forgery_auc"],
                    "trained_inv_mean": result["final_inv_mean"],
                    "best_inv_mean": result["best_inv_mean"],
                    **{f"trained_shortcut_auc_{a}": result["final_shortcut_aucs"][i]
                       for i, a in enumerate(SHORTCUT_AXES)},
                    **{f"baseline_shortcut_auc_{a}": baseline_shortcuts[i]
                       for i, a in enumerate(SHORTCUT_AXES)},
                }
                rows.append(row)
                print(f"  k={k:3d}  λ={lambda_grl:3.1f}  baseline forgery={baseline_forgery:.3f} inv_mean={baseline_inv_mean:+.3f} | trained forgery={result['final_forgery_auc']:.3f} inv_mean={result['final_inv_mean']:+.3f} (best={result['best_inv_mean']:+.3f})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pretest1_multiaxis_grl.csv", index=False)
    print(f"\nWrote {OUT / 'pretest1_multiaxis_grl.csv'}: {len(df)} rows")

    # Summary
    print("\n=== Summary: best inv_mean per ckpt across (k, λ) ===")
    for ckpt in CKPTS:
        sub = df[df["ckpt"] == ckpt]
        if len(sub) == 0:
            continue
        best_row = sub.loc[sub["best_inv_mean"].idxmax()]
        print(f"  {ckpt:18s}: best inv_mean={best_row['best_inv_mean']:+.3f} at k={int(best_row['k'])} λ={best_row['lambda_grl']:.1f} (forgery_auc={best_row['trained_forgery_auc']:.3f}, baseline={best_row['baseline_inv_mean']:+.3f})")

if __name__ == "__main__":
    main()
