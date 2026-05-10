"""Pre-test 2 — L6 vs L11 head prototype.

Question: would a head reading from L6 (where forgery saturates first, before
shortcut amplification at L11) produce a comparable forgery model with
better invariance?

Method:
  1. Train a small classifier on L6 cached features for {P8A, E2B, T3_S1_step1500}
  2. Compare to a similarly-trained classifier on L11 features
  3. Measure forgery_AUC + shortcut_AUCs of trained head's hidden representation
  4. Cross-validation 5-fold

Outcome:
  - If L6 head matches L11 head on forgery_AUC AND has better inv_mean → architectural variant
  - If L6 head loses forgery_AUC significantly → L11 amplification is doing useful work

Output: outputs/pretest2_l6_vs_l11.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

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

CKPTS = ["P8A", "E2B", "T3_S1_step1500", "T3_S1_step2500", "MCLIOEXB"]
LAYERS = [6, 9, 11]
SHORTCUT_AXES = ["is_chronic_6", "lap_var_high", "min_dim_high", "is_dor", "face_size_high"]


class SmallHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, x):
        z = self.net(x)
        return self.classifier(z), z


def lr_probe_auc(X_tr, y_tr, X_va, y_va):
    if y_tr.sum() < 5 or y_va.sum() < 5 or (len(y_tr) - y_tr.sum()) < 5:
        return float("nan")
    clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    return float(roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1]))


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
        "is_chronic_6": "is_chronic_6_int", "lap_var_high": "lap_var_high",
        "min_dim_high": "min_dim_high", "is_dor": "is_dor_int", "face_size_high": "face_size_high",
    }

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    rows = []
    for ckpt in CKPTS:
        for layer in LAYERS:
            try:
                feats, valid_idx = load_features(ckpt, layer)
            except Exception as exc:
                print(f"  {ckpt} L{layer}: load failed {exc}")
                continue
            valid_set = set(int(v) for v in valid_idx.tolist())
            mask = panel["row_ix"].isin(valid_set).values
            panel_a = panel[mask].reset_index(drop=True)
            valid_to_pos = {int(v): r for r, v in enumerate(valid_idx)}
            sel_pos = [valid_to_pos[int(rx)] for rx in panel_a["row_ix"].values]
            X = feats[sel_pos]
            y_forgery = panel_a["is_real_vs_fake"].values.astype(int)
            y_shortcuts = {a: panel_a[LABEL_COLS[a]].values.astype(int) for a in SHORTCUT_AXES}
            X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

            # 5-fold CV: train MLP head on (X, y_forgery), measure (forgery_auc, shortcut_aucs) on the trained hidden representation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            forgery_aucs, shortcut_aucs_per_fold = [], {a: [] for a in SHORTCUT_AXES}
            raw_forgery_aucs, raw_shortcut_aucs = [], {a: [] for a in SHORTCUT_AXES}
            for fold, (tr, te) in enumerate(skf.split(X_n, y_forgery)):
                X_tr, X_te = X_n[tr], X_n[te]
                yt_f = y_forgery[tr]
                yv_f = y_forgery[te]
                # Train MLP head on forgery
                model = SmallHead(X_tr.shape[1]).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                Xt_tensor = torch.tensor(X_tr, dtype=torch.float32, device=device)
                yt_tensor = torch.tensor(yt_f, dtype=torch.long, device=device)
                Xv_tensor = torch.tensor(X_te, dtype=torch.float32, device=device)
                for epoch in range(80):
                    model.train()
                    perm = torch.randperm(len(Xt_tensor), device=device)
                    for i in range(0, len(Xt_tensor), 64):
                        idx = perm[i:i+64]
                        logits, _ = model(Xt_tensor[idx])
                        loss = F.cross_entropy(logits, yt_tensor[idx])
                        opt.zero_grad(); loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    val_logits, val_z = model(Xv_tensor)
                    train_logits, train_z = model(Xt_tensor)
                val_z_np = val_z.cpu().numpy()
                train_z_np = train_z.cpu().numpy()
                # Forgery AUC from MLP head's softmax
                forgery_pred = F.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
                f_auc = float(roc_auc_score(yv_f, forgery_pred))
                forgery_aucs.append(f_auc)
                # Shortcut AUCs via LR probe on trained hidden representation
                for a in SHORTCUT_AXES:
                    yt_s = y_shortcuts[a][tr]
                    yv_s = y_shortcuts[a][te]
                    sc_auc = lr_probe_auc(train_z_np, yt_s, val_z_np, yv_s)
                    shortcut_aucs_per_fold[a].append(sc_auc)
                # Raw feature baselines
                raw_f = lr_probe_auc(X_tr, yt_f, X_te, yv_f)
                raw_forgery_aucs.append(raw_f)
                for a in SHORTCUT_AXES:
                    yt_s = y_shortcuts[a][tr]
                    yv_s = y_shortcuts[a][te]
                    raw_sc = lr_probe_auc(X_tr, yt_s, X_te, yv_s)
                    raw_shortcut_aucs[a].append(raw_sc)
            mean_forgery = np.mean(forgery_aucs)
            mean_shortcuts = {a: np.nanmean(shortcut_aucs_per_fold[a]) for a in SHORTCUT_AXES}
            mean_inv = mean_forgery - np.mean(list(mean_shortcuts.values()))
            raw_mean_forgery = np.mean(raw_forgery_aucs)
            raw_mean_shortcuts = {a: np.nanmean(raw_shortcut_aucs[a]) for a in SHORTCUT_AXES}
            raw_inv = raw_mean_forgery - np.mean(list(raw_mean_shortcuts.values()))
            row = {
                "ckpt": ckpt, "layer": layer,
                "trained_head_forgery_auc": mean_forgery,
                "trained_head_inv_mean": mean_inv,
                "raw_features_forgery_auc": raw_mean_forgery,
                "raw_features_inv_mean": raw_inv,
                **{f"trained_head_shortcut_{a}": mean_shortcuts[a] for a in SHORTCUT_AXES},
                **{f"raw_shortcut_{a}": raw_mean_shortcuts[a] for a in SHORTCUT_AXES},
            }
            rows.append(row)
            print(f"  {ckpt:18s} L{layer:02d}: trained_head forgery={mean_forgery:.3f} inv={mean_inv:+.3f} | raw forgery={raw_mean_forgery:.3f} inv={raw_inv:+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pretest2_l6_vs_l11.csv", index=False)
    print(f"\nWrote {OUT / 'pretest2_l6_vs_l11.csv'}: {len(df)} rows")

    # Markdown summary
    md = ["# Pre-test 2 — L6 vs L11 head prototype",
          "",
          "Trained MLP head (linear→256→256→2) on cached layer features. Forgery AUC measured from MLP softmax. Shortcut AUCs measured via LR probe on the MLP's penultimate hidden representation.",
          "",
          "## Forgery AUC + Inv mean by layer",
          ""]
    md.append("| ckpt | L6 forgery / inv_mean | L9 forgery / inv_mean | L11 forgery / inv_mean |")
    md.append("|---|---|---|---|")
    for ckpt in CKPTS:
        sub = df[df["ckpt"] == ckpt]
        cells = []
        for layer in LAYERS:
            srow = sub[sub["layer"] == layer]
            if len(srow):
                cells.append(f"{srow['trained_head_forgery_auc'].iloc[0]:.3f} / {srow['trained_head_inv_mean'].iloc[0]:+.3f}")
            else:
                cells.append("-")
        md.append(f"| {ckpt} | {' | '.join(cells)} |")
    (OUT / "pretest2_PROFILE.md").write_text("\n".join(md))
    print(f"Wrote {OUT / 'pretest2_PROFILE.md'}")

if __name__ == "__main__":
    main()
