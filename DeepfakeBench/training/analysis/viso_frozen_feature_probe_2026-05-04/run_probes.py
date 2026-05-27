#!/usr/bin/env python3
"""CPU Job 6: Frozen-feature linear probe on viso fakes — caught vs uncaught.

Inputs:
  - analysis/clip_vs_p8a_viso_2026-05-03/outputs/p8a__features.npz
  - analysis/clip_vs_p8a_viso_2026-05-03/outputs/clip_b16_raw__features.npz
  - analysis/clip_vs_p8a_viso_2026-05-03/outputs/sample_manifest.csv
  - analysis/cpu_followups_2026-05-04/outputs/viso_per_frame_with_all_ckpts.csv
  - analysis/score_distribution_2026-05-02/outputs/crop_attributes.csv
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = ROOT / "analysis/viso_frozen_feature_probe_2026-05-04/outputs"
OUT.mkdir(parents=True, exist_ok=True)

P8A_NPZ = ROOT / "analysis/clip_vs_p8a_viso_2026-05-03/outputs/p8a__features.npz"
CLIP_NPZ = ROOT / "analysis/clip_vs_p8a_viso_2026-05-03/outputs/clip_b16_raw__features.npz"
PER_FRAME = ROOT / "analysis/cpu_followups_2026-05-04/outputs/viso_per_frame_with_all_ckpts.csv"
CROP_ATTR = ROOT / "analysis/score_distribution_2026-05-02/outputs/crop_attributes.csv"

RNG = np.random.RandomState(42)


def load_features(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    return {
        "X": np.asarray(z["features"], dtype=np.float32),
        "y": np.asarray(z["label"], dtype=np.int32),
        "frame_path": np.asarray(z["frame_path"]),
        "source": np.asarray(z["source"]),
        "family_key": np.asarray(z["family_key"]),
        "local_path": np.asarray(z["local_path"]),
    }


def basename_of(p):
    return os.path.basename(str(p))


def viso_subtype_from_name(fname):
    if "visomaster_enhanced_raw__" in fname:
        return "raw"
    if "visomaster_enhanced_teams__" in fname:
        return "teams"
    return "other"


def kfold_probe(X, y, C_values, n_splits=5, seed=42):
    """Run StratifiedKFold logistic regression on features, return per-C summary."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for C in C_values:
        accs, aucs, aps = [], [], []
        for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xte = scaler.transform(X[te])
            clf = LogisticRegression(
                max_iter=2000, n_jobs=1, C=C, solver="lbfgs", random_state=seed
            )
            clf.fit(Xtr, y[tr])
            pred = clf.predict(Xte)
            prob = clf.predict_proba(Xte)[:, 1]
            accs.append(accuracy_score(y[te], pred))
            try:
                aucs.append(roc_auc_score(y[te], prob))
            except ValueError:
                aucs.append(np.nan)
            aps.append(average_precision_score(y[te], prob))
        rows.append(
            dict(
                C=C,
                accuracy_mean=float(np.mean(accs)),
                accuracy_std=float(np.std(accs)),
                auc_mean=float(np.nanmean(aucs)),
                auc_std=float(np.nanstd(aucs)),
                ap_mean=float(np.mean(aps)),
            )
        )
    return rows


def main():
    print("[1] Loading frozen feature caches ...")
    p8a = load_features(P8A_NPZ)
    clip = load_features(CLIP_NPZ)

    # Sanity: cached frame_path arrays must align across the two caches
    assert np.array_equal(p8a["frame_path"], clip["frame_path"]), (
        "P8A and CLIP caches not aligned by frame_path"
    )
    assert np.array_equal(p8a["y"], clip["y"]), "label mismatch"
    print(
        f"    P8A X={p8a['X'].shape}  CLIP X={clip['X'].shape}  "
        f"label counts={np.bincount(p8a['y'])}  source counts={np.unique(p8a['source'], return_counts=True)}"
    )

    # Restrict to viso fakes (550)
    is_viso_fake = p8a["source"] == "viso_fake"
    n_viso = int(is_viso_fake.sum())
    print(f"[2] Cached viso fakes: {n_viso}")

    fpath_viso = p8a["frame_path"][is_viso_fake]
    fpath_basename = np.array([basename_of(p) for p in fpath_viso])

    # Load per-frame ckpt CSV
    print("[3] Loading per-frame ckpt CSV ...")
    df_pf = pd.read_csv(PER_FRAME)
    df_pf["basename"] = df_pf["frame_path"].apply(basename_of)
    print(f"    per-frame rows: {len(df_pf)}  unique basenames: {df_pf['basename'].nunique()}")

    bn_pf = set(df_pf["basename"].tolist())
    bn_cache = set(fpath_basename.tolist())
    overlap = bn_cache & bn_pf
    only_cache = bn_cache - bn_pf
    only_pf = bn_pf - bn_cache
    print(
        f"    overlap: {len(overlap)} | only-in-cache: {len(only_cache)} | only-in-per-frame: {len(only_pf)}"
    )

    # Build aligned table for the cached viso-fake frames
    df_pf_idx = df_pf.set_index("basename")
    keep_mask = np.array([bn in df_pf_idx.index for bn in fpath_basename])
    n_drop = int((~keep_mask).sum())
    print(f"    dropping {n_drop} cached frames not in per-frame CSV")

    # Indices into the original cache
    viso_idx_in_cache = np.where(is_viso_fake)[0]
    keep_global_idx = viso_idx_in_cache[keep_mask]
    kept_basenames = fpath_basename[keep_mask]
    n_keep = len(keep_global_idx)
    print(f"    n_keep viso fakes: {n_keep}")

    # Pull ckpt info aligned to kept_basenames
    rows = []
    for bn in kept_basenames:
        r = df_pf_idx.loc[bn]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        rows.append(
            dict(
                basename=bn,
                P8A_score=float(r["P8A_score"]),
                E2B_3200_score=float(r["E2B_3200_score"]),
                E3_6600_score=float(r["E3_6600_score"]),
                P8A_caught=int(r["P8A_caught_at_fpr10"]),
                E2B_3200_caught=int(r["E2B_3200_caught_at_fpr10"]),
                E3_6600_caught=int(r["E3_6600_caught_at_fpr10"]),
                caught_by=str(r["caught_by_subset"]),
                n_ckpts_caught=int(r["n_ckpts_caught"]),
                method=str(r["method"]),
            )
        )
    df_align = pd.DataFrame(rows)
    df_align["subtype"] = df_align["basename"].apply(viso_subtype_from_name)
    df_align["is_caught"] = (df_align["n_ckpts_caught"] >= 1).astype(int)

    # Load crop_attributes for laplacian quartile
    print("[4] Loading crop_attributes ...")
    df_attr = pd.read_csv(CROP_ATTR)
    df_attr["basename"] = df_attr["filename"].apply(basename_of)
    bn_attr_idx = df_attr.set_index("basename")
    lap = []
    for bn in df_align["basename"]:
        if bn in bn_attr_idx.index:
            r = bn_attr_idx.loc[bn]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            lap.append(float(r["laplacian_var"]))
        else:
            lap.append(np.nan)
    df_align["laplacian_var"] = lap
    n_lap_missing = int(np.isnan(df_align["laplacian_var"]).sum())
    print(f"    laplacian_var missing for {n_lap_missing} frames")
    # Quartile bins
    lap_arr = df_align["laplacian_var"].values
    finite = np.isfinite(lap_arr)
    q = np.full(lap_arr.shape, -1, dtype=int)
    if finite.sum() > 4:
        try:
            qs = pd.qcut(
                pd.Series(lap_arr[finite]),
                4,
                labels=[0, 1, 2, 3],
                duplicates="drop",
            )
            q[finite] = np.asarray(qs.astype(int).values)
        except Exception as e:  # pragma: no cover
            print("    qcut failed:", e)
    df_align["laplacian_var_quartile"] = q

    # Aligned features
    Xp = p8a["X"][keep_global_idx]
    Xc = clip["X"][keep_global_idx]

    # Targets
    y_caught = df_align["is_caught"].values.astype(int)
    print(f"[5] is_caught counts: {np.bincount(y_caught)}")

    # ---- Step 2: caught-vs-uncaught probe (P8A & CLIP) ----
    C_VALUES = [1.0, 0.1, 0.01, 0.001]
    print("[6] Probe A: P8A frozen features, target=is_caught")
    rows_pa = kfold_probe(Xp, y_caught, C_VALUES)
    print("[7] Probe B: CLIP-B16 raw features, target=is_caught")
    rows_cb = kfold_probe(Xc, y_caught, C_VALUES)

    # ---- Step 3: fake-vs-real probe on full 1100 ----
    print("[8] Probe sanity: fake-vs-real on viso_fake + teams_real_dev (1100)")
    Xp_full = p8a["X"]
    Xc_full = clip["X"]
    y_fr = p8a["y"]  # 0 / 1
    rows_pa_fr = kfold_probe(Xp_full, y_fr, C_VALUES)
    rows_cb_fr = kfold_probe(Xc_full, y_fr, C_VALUES)

    # ---- Step 5: subtype-stratified ----
    print("[9] Subtype-stratified caught probes (P8A only, C=1.0/0.1/0.01)")
    sub_rows = []
    for subtype in ["raw", "teams"]:
        mask = (df_align["subtype"] == subtype).values
        n = int(mask.sum())
        if n < 30:
            continue
        Xp_s = Xp[mask]
        y_s = y_caught[mask]
        if len(np.unique(y_s)) < 2:
            sub_rows.append(
                dict(subtype=subtype, n=n, note="single-class, skipped")
            )
            continue
        rows_s = kfold_probe(Xp_s, y_s, C_VALUES)
        for r in rows_s:
            r2 = dict(r)
            r2["subtype"] = subtype
            r2["n"] = n
            r2["caught_count"] = int(y_s.sum())
            sub_rows.append(r2)

    # ---- Build probe summary CSV/JSON ----
    summary_rows = []
    for r in rows_pa:
        summary_rows.append(
            dict(model="P8A", target="is_caught", n_train=int(len(y_caught)), **r)
        )
    for r in rows_cb:
        summary_rows.append(
            dict(model="CLIP_B16_raw", target="is_caught", n_train=int(len(y_caught)), **r)
        )
    for r in rows_pa_fr:
        summary_rows.append(
            dict(model="P8A", target="fake_vs_real", n_train=int(len(y_fr)), **r)
        )
    for r in rows_cb_fr:
        summary_rows.append(
            dict(model="CLIP_B16_raw", target="fake_vs_real", n_train=int(len(y_fr)), **r)
        )
    for r in sub_rows:
        summary_rows.append(
            dict(
                model="P8A",
                target=f"is_caught[subtype={r.get('subtype')}]",
                n_train=int(r.get("n", 0)),
                **{k: v for k, v in r.items() if k not in ("subtype", "n")},
            )
        )
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT / "probe_summary.csv", index=False)
    print("    wrote probe_summary.csv")

    # Verdict
    auc_pa_c1 = next(r["auc_mean"] for r in rows_pa if r["C"] == 1.0)
    auc_cb_c1 = next(r["auc_mean"] for r in rows_cb if r["C"] == 1.0)
    auc_pa_c001 = next(r["auc_mean"] for r in rows_pa if r["C"] == 0.01)
    delta = auc_pa_c1 - auc_cb_c1

    if auc_pa_c1 > 0.85:
        verdict = "head_only_lever_live"
        verdict_text = (
            f"P8A linear probe AUC at C=1.0 is {auc_pa_c1:.3f} > 0.85: caught-vs-uncaught is "
            "strongly linearly separable in frozen P8A features. A different head boundary "
            "could plausibly catch the uncaught fakes -> head-only retrain lever is LIVE."
        )
    elif auc_pa_c1 < 0.65:
        verdict = "representation_gap"
        verdict_text = (
            f"P8A linear probe AUC at C=1.0 is {auc_pa_c1:.3f} < 0.65: caught-vs-uncaught is "
            "NOT linearly separable in frozen P8A features. Encoder representation must change "
            "(head-only fix unlikely to bite)."
        )
    else:
        verdict = "ambiguous"
        verdict_text = (
            f"P8A linear probe AUC at C=1.0 is {auc_pa_c1:.3f} (in 0.65-0.85 band). "
            "Partial linear separability — head-only retrain may help but not silver bullet."
        )

    # ---- Step 4: t-SNE (UMAP fallback) on viso fakes only, P8A features ----
    print("[10] t-SNE on 550 viso fakes (P8A) — UMAP not installed, fallback to t-SNE")
    from sklearn.manifold import TSNE

    scaler = StandardScaler()
    Xp_std = scaler.fit_transform(Xp)
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        n_jobs=1,
        init="pca",
        learning_rate="auto",
    )
    emb = tsne.fit_transform(Xp_std)

    df_align["x"] = emb[:, 0]
    df_align["y"] = emb[:, 1]

    df_emb_out = pd.DataFrame(
        dict(
            frame_path=df_align["basename"],
            x=df_align["x"],
            y=df_align["y"],
            caught_by=df_align["caught_by"],
            subtype=df_align["subtype"],
            laplacian_var_quartile=df_align["laplacian_var_quartile"],
        )
    )
    df_emb_out.to_csv(OUT / "umap_embedding.csv", index=False)

    # Plots
    def scatter_by(col, title, fname, cmap=None):
        fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
        cats = df_align[col].astype(str).values
        unique = sorted(set(cats))
        for cat in unique:
            m = cats == cat
            ax.scatter(df_align["x"][m], df_align["y"][m], s=8, alpha=0.7, label=str(cat))
        ax.set_title(title)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=7, markerscale=1.5, loc="best")
        fig.tight_layout()
        fig.savefig(OUT / fname)
        plt.close(fig)

    scatter_by("caught_by", "viso fakes — caught_by subset (P8A t-SNE)", "umap_caught_by.png")
    scatter_by("subtype", "viso fakes — subtype (P8A t-SNE)", "umap_subtype.png")
    scatter_by(
        "laplacian_var_quartile",
        "viso fakes — laplacian_var quartile (P8A t-SNE)",
        "umap_laplacian.png",
    )

    # Region purity on caught_by, k=10 nearest in feature space (Euclidean on standardized P8A)
    print("[11] Region purity (k=10) ...")
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=11, n_jobs=1)
    nn.fit(Xp_std)
    _, idx = nn.kneighbors(Xp_std)
    idx = idx[:, 1:]  # drop self

    cb_arr = df_align["caught_by"].astype(str).values
    purity = []
    for i in range(len(cb_arr)):
        neigh_labels = cb_arr[idx[i]]
        match = (neigh_labels == cb_arr[i]).sum()
        purity.append(match / 10.0)
    df_align["purity_k10"] = purity

    purity_per_class = (
        df_align.groupby("caught_by")["purity_k10"].agg(["mean", "std", "count"]).reset_index()
    )

    # Same metric but binary is_caught (sometimes more interpretable)
    is_caught_arr = df_align["is_caught"].values
    bin_purity = []
    for i in range(len(is_caught_arr)):
        neigh_labels = is_caught_arr[idx[i]]
        match = (neigh_labels == is_caught_arr[i]).sum()
        bin_purity.append(match / 10.0)
    df_align["binary_purity_k10"] = bin_purity
    bin_purity_per = (
        df_align.groupby("is_caught")["binary_purity_k10"].agg(["mean", "std", "count"]).reset_index()
    )

    region_purity_doc = {
        "k": 10,
        "metric": "fraction of k=10 nearest neighbours sharing the same caught_by label",
        "feature_space": "P8A 512-d, StandardScaler, Euclidean k-NN",
        "per_caught_by": purity_per_class.to_dict(orient="records"),
        "binary_is_caught": bin_purity_per.to_dict(orient="records"),
        "overall_caught_by_purity_mean": float(np.mean(purity)),
        "overall_binary_purity_mean": float(np.mean(bin_purity)),
        "interpretation_threshold": (
            "purity ≈ class prior => interleaved (decision-boundary problem); "
            "purity >> prior => clustered region (representation supports head-only fix in principle, but "
            "would need supervision signal)."
        ),
    }
    with open(OUT / "region_purity.json", "w") as f:
        json.dump(region_purity_doc, f, indent=2, default=float)

    # ---- probe_summary.json ----
    out_json = {
        "cache": {
            "p8a_npz": str(P8A_NPZ.relative_to(ROOT)),
            "clip_npz": str(CLIP_NPZ.relative_to(ROOT)),
            "n_total": int(p8a["X"].shape[0]),
            "n_viso_fake_cached": int((p8a["source"] == "viso_fake").sum()),
            "n_teams_real_dev_cached": int((p8a["source"] == "teams_real_dev").sum()),
            "n_per_frame_csv_rows": int(len(df_pf)),
            "overlap_basenames": int(len(overlap)),
            "drop_cached_frames_not_in_per_frame_csv": int(n_drop),
            "n_aligned_viso_fakes_used": int(n_keep),
            "n_caught": int(np.sum(y_caught == 1)),
            "n_uncaught": int(np.sum(y_caught == 0)),
            "subtype_counts": (
                df_align["subtype"].value_counts().to_dict()
            ),
        },
        "probe_is_caught": {
            "P8A": rows_pa,
            "CLIP_B16_raw": rows_cb,
            "delta_auc_C1_P8A_minus_CLIP": float(delta),
        },
        "probe_fake_vs_real": {
            "P8A": rows_pa_fr,
            "CLIP_B16_raw": rows_cb_fr,
        },
        "probe_is_caught_subtype_stratified": sub_rows,
        "verdict_load_bearing": {
            "label": verdict,
            "auc_pa_C1": auc_pa_c1,
            "auc_cb_C1": auc_cb_c1,
            "delta": float(delta),
            "auc_pa_C001": auc_pa_c001,
            "text": verdict_text,
        },
        "region_purity_summary": region_purity_doc,
        "umap_fallback": "t-SNE (umap-learn not installed)",
    }
    with open(OUT / "probe_summary.json", "w") as f:
        json.dump(out_json, f, indent=2, default=float)

    print(
        f"[done] verdict={verdict}  AUC P8A@C=1.0={auc_pa_c1:.3f}  CLIP@C=1.0={auc_cb_c1:.3f}  Δ={delta:+.3f}"
    )
    print(f"region_purity overall (caught_by 6-way) = {np.mean(purity):.3f}")
    print(f"region_purity overall (binary is_caught) = {np.mean(bin_purity):.3f}")


if __name__ == "__main__":
    main()
