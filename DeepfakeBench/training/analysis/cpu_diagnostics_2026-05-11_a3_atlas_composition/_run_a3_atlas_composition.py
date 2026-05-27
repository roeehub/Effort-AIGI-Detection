"""A3 — Triptych composition audit + atlas substrate decomposition.

Goal: explain why T4_L1_step10500 atlas inv_mean = +0.0414 (highest across 30
ckpts) yet promotion-contract lockbox AUC dropped 0.174 absolute vs P8A
(0.9355 -> 0.7619).

Hypothesis: the 800-frame triptych is dev-leaning; atlas inv_mean lift reflects
dev-substrate invariance while lockbox-substrate degradation goes uncaptured.

Outputs:
  triptych_substrate_breakdown.csv
  per_substrate_inv_mean.csv
  atlas_lockbox_overlap.csv

Method:
  1. Read the 800-frame triptych manifest (first 800 rows of
     analysis/embedding_triptych_2026-04-30/.../sampled_frames.csv).
  2. Tally by split + label + chronic-6 + identity.
  3. Re-fit 5-fold CV LR probes on cached L11 features per substrate slice:
     - dev_only (n_dev=713)
     - lockbox_only (n_lockbox=87)
     - dev_real, dev_fake, lockbox_real, lockbox_fake (one-vs-rest constructions
       where applicable - we treat each as a label subset for is_real_vs_fake)
  4. For inv_mean we restrict the rows used to fit BOTH the is_real_vs_fake
     probe AND the shortcut probes to the subset, then compute the inv_mean
     definition: forgery_AUC - mean(shortcut_AUCs).
  5. Overlap audit: GCS URI intersection between triptych rows that are
     'lockbox' split and the 1419 real + 280 fake frames in the lockbox suite
     reports.

CPU-only. n_jobs=1 per memory feedback.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = REPO / "analysis" / "cpu_diagnostics_2026-05-11_a3_atlas_composition"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLED_CSV = REPO / "analysis" / "embedding_triptych_2026-04-30" / "outputs" / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"

# Layer 11 feature caches (all 4 ckpts use the same 800-frame ordering)
FEAT_CACHE_2026_05_08 = REPO / "analysis" / "iq_perlayer_probe_2026-05-08" / "_cache"

CKPTS = {
    "P8A":             FEAT_CACHE_2026_05_08 / "intermediate__P8A__layer11__n800.npz",
    "E2B":             FEAT_CACHE_2026_05_08 / "intermediate__E2B__layer11__n800.npz",
    "T4_L1_step10500": FEAT_CACHE_2026_05_08 / "intermediate__T4_L1_step10500__layer11__n800.npz",
    "T3_S1_step1500":  FEAT_CACHE_2026_05_08 / "intermediate__T3_S1_step1500__layer11__n800.npz",
}

LOCKBOX_SCORECARD = REPO / "analysis" / "cpu_diagnostics_2026-05-10" / "_t4_scorecard_local"

CHRONIC_6_PATTERNS = [
    "Roy_D", "PC_Generator", "bla_bla_chow",
    "Md_noyn_Sharker", "dor_shkedi", "healthy_dor",
]

PRIMARY_6_IQ = ["lap_var", "min_dim", "luma_mean", "color_b_dev", "edge_mag", "skin_frac"]

ATLAS_PARQUET = REPO / "analysis" / "iq_data_atlas_2026-05-08" / "outputs" / "per_frame.parquet"


def chronic_match(s) -> bool:
    if not isinstance(s, str):
        return False
    sl = s.lower()
    return any(p.lower() in sl for p in CHRONIC_6_PATTERNS)


def is_dor_fn(identity_key) -> bool:
    if not isinstance(identity_key, str):
        return False
    return "dor" in identity_key.lower()


def build_panel():
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

    # Attach IQ features
    try:
        atlas = pd.read_parquet(ATLAS_PARQUET)[["frame_path"] + PRIMARY_6_IQ].copy()
        atlas = atlas.drop_duplicates(subset=["frame_path"], keep="first")
        df = df.merge(atlas, left_on="gcs_uri", right_on="frame_path", how="left")
        assert len(df) == 800, f"merge inflated panel size to {len(df)} (should be 800)"
    except Exception as exc:
        print(f"WARNING: atlas not joinable: {exc}", file=sys.stderr)
        for c in PRIMARY_6_IQ:
            df[c] = np.nan

    # Inline IQ for missing rows
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


def write_substrate_breakdown(panel: pd.DataFrame):
    rows = []

    # By split
    for split in ["dev", "lockbox"]:
        sub = panel[panel["split"] == split]
        rows.append({"slice": f"split={split}", "n": len(sub),
                     "n_real": int((sub["label"] == "real").sum()),
                     "n_fake": int((sub["label"] == "fake").sum()),
                     "n_chronic_6": int(sub["is_chronic_6"].sum()),
                     "n_dor": int(sub["is_dor"].sum()),
                     "n_identities": sub["identity_key"].nunique()})

    # By split x label
    for split in ["dev", "lockbox"]:
        for lab in ["real", "fake"]:
            sub = panel[(panel["split"] == split) & (panel["label"] == lab)]
            rows.append({"slice": f"split={split}/label={lab}", "n": len(sub),
                         "n_real": int((sub["label"] == "real").sum()),
                         "n_fake": int((sub["label"] == "fake").sum()),
                         "n_chronic_6": int(sub["is_chronic_6"].sum()),
                         "n_dor": int(sub["is_dor"].sum()),
                         "n_identities": sub["identity_key"].nunique()})

    # Chronic-6 yes/no
    for v in [0, 1]:
        sub = panel[panel["is_chronic_6"] == v]
        rows.append({"slice": f"is_chronic_6={v}", "n": len(sub),
                     "n_real": int((sub["label"] == "real").sum()),
                     "n_fake": int((sub["label"] == "fake").sum()),
                     "n_chronic_6": int(sub["is_chronic_6"].sum()),
                     "n_dor": int(sub["is_dor"].sum()),
                     "n_identities": sub["identity_key"].nunique()})

    # Method-level rollup
    method_tbl = panel.groupby("method", dropna=False).size().reset_index(name="n").sort_values("n", ascending=False)
    method_tbl.to_csv(OUT / "triptych_method_breakdown.csv", index=False)

    # Whole pool
    rows.append({"slice": "all", "n": len(panel),
                 "n_real": int((panel["label"] == "real").sum()),
                 "n_fake": int((panel["label"] == "fake").sum()),
                 "n_chronic_6": int(panel["is_chronic_6"].sum()),
                 "n_dor": int(panel["is_dor"].sum()),
                 "n_identities": panel["identity_key"].nunique()})

    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "triptych_substrate_breakdown.csv", index=False)
    return tbl


def load_features(npz_path: Path):
    blob = np.load(npz_path)
    feats = blob["features"].astype(np.float32)
    valid_idx = blob["valid_idx"].astype(np.int64)
    return feats, valid_idx


def fit_probe_auc(X, y, n_splits=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y, dtype=int)
    if len(y) < 20:
        return float("nan"), len(y), int(y.sum())
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return float("nan"), len(y), int(y.sum())

    # Try 5-fold; fall back to 3-fold for small slices.
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


def per_substrate_inv_mean(panel: pd.DataFrame):
    """Recompute inv_mean per ckpt per substrate slice.

    inv_mean = forgery_AUC - mean(shortcut_AUCs)
    shortcuts: is_dor, is_chronic_6, lap_var_high, min_dim_high, face_size_high.

    For substrate-specific slices, all probes use the same subset of frames so the
    inv_mean reflects "how well does this ckpt's L11 representation separate
    real-vs-fake AS COMPARED TO substrate shortcuts WITHIN this slice".
    """
    # Build binary shortcut columns (medians over the FULL 800-frame pool to match
    # the original atlas methodology, then we subset rows downstream).
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
        "full_n800":           pd.Series(True, index=panel.index),
        "dev_only":            (panel["split"] == "dev"),
        "lockbox_only":        (panel["split"] == "lockbox"),
        "chronic_6_only":      (panel["is_chronic_6"] == 1),
        "non_chronic_only":    (panel["is_chronic_6"] == 0),
    }

    rows = []
    for ckpt_name, npz_path in CKPTS.items():
        feats, valid_idx = load_features(npz_path)
        # Map row_ix -> position in feats
        valid_to_pos = {int(v): r for r, v in enumerate(valid_idx)}
        valid_set = set(valid_to_pos.keys())

        for slice_name, slice_mask in SLICES.items():
            sub = panel[slice_mask].copy()
            sub = sub[sub["row_ix"].isin(valid_set)].reset_index(drop=True)
            if len(sub) < 30:
                print(f"  [{ckpt_name}] slice={slice_name} too small (n={len(sub)}), skipping")
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
                auc, n, n_pos = fit_probe_auc(X, y)
                sig_aucs[sig_label] = auc

            forgery_auc = sig_aucs.get("is_real_vs_fake", float("nan"))
            shortcut_vals = [sig_aucs.get(s, float("nan")) for s in SHORTCUTS]
            shortcut_vals_clean = [v for v in shortcut_vals if not (v is None or np.isnan(v))]
            mean_shortcut = float(np.mean(shortcut_vals_clean)) if shortcut_vals_clean else float("nan")
            inv_mean = (forgery_auc - mean_shortcut) if (not np.isnan(forgery_auc) and not np.isnan(mean_shortcut)) else float("nan")

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
            print(f"  [{ckpt_name}] slice={slice_name:18s} n={len(sub):3d} forgery={forgery_auc:.3f} inv_mean={inv_mean:+.4f}")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "per_substrate_inv_mean.csv", index=False)
    return tbl


def lockbox_overlap_audit(panel: pd.DataFrame):
    """Check whether triptych frames are also in the lockbox suite reports."""
    # Pull GCS URIs from the lockbox suite reports.
    real_report = pd.read_csv(LOCKBOX_SCORECARD / "teams_real_all_lockbox_p8a_reference_step5000_frames_report.csv")
    fake_report = pd.read_csv(LOCKBOX_SCORECARD / "teams_fake_all_lockbox_p8a_reference_step5000_frames_report.csv")
    lockbox_real_uris = set(real_report["frame_path"].dropna().unique())
    lockbox_fake_uris = set(fake_report["frame_path"].dropna().unique())
    lockbox_all_uris = lockbox_real_uris | lockbox_fake_uris

    triptych_uris = set(panel["gcs_uri"].dropna().unique())
    triptych_lockbox_split_uris = set(panel[panel["split"] == "lockbox"]["gcs_uri"].dropna().unique())
    triptych_dev_split_uris = set(panel[panel["split"] == "dev"]["gcs_uri"].dropna().unique())

    rows = [
        {"comparison": "triptych_n_unique_uris", "n": len(triptych_uris)},
        {"comparison": "triptych_split=lockbox_n_unique_uris", "n": len(triptych_lockbox_split_uris)},
        {"comparison": "triptych_split=dev_n_unique_uris", "n": len(triptych_dev_split_uris)},
        {"comparison": "lockbox_suite_real_frames_n", "n": len(lockbox_real_uris)},
        {"comparison": "lockbox_suite_fake_frames_n", "n": len(lockbox_fake_uris)},
        {"comparison": "lockbox_suite_all_frames_n", "n": len(lockbox_all_uris)},
        # Critical: do triptych frames appear in the lockbox suite? (data overlap audit)
        {"comparison": "triptych_split=lockbox ∩ lockbox_suite_real_frames",
         "n": len(triptych_lockbox_split_uris & lockbox_real_uris)},
        {"comparison": "triptych_split=lockbox ∩ lockbox_suite_fake_frames",
         "n": len(triptych_lockbox_split_uris & lockbox_fake_uris)},
        {"comparison": "triptych_split=lockbox ∩ lockbox_suite_all",
         "n": len(triptych_lockbox_split_uris & lockbox_all_uris)},
        {"comparison": "triptych_split=lockbox NOT in lockbox_suite (dropped or different bucket)",
         "n": len(triptych_lockbox_split_uris - lockbox_all_uris)},
        # And dev-split triptych frames should NOT be in lockbox suite
        {"comparison": "triptych_split=dev ∩ lockbox_suite_all (should be 0)",
         "n": len(triptych_dev_split_uris & lockbox_all_uris)},
    ]
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "atlas_lockbox_overlap.csv", index=False)
    return tbl


def main():
    print("Building 800-frame panel + IQ inline merge...")
    panel = build_panel()
    print(f"Panel: {len(panel)} rows")

    print("\n[1/3] Substrate breakdown ...")
    bd = write_substrate_breakdown(panel)
    print(bd.to_string(index=False))

    print("\n[2/3] Per-substrate inv_mean recomputation ...")
    inv = per_substrate_inv_mean(panel)
    print("\nPer-substrate inv_mean wide:")
    wide = inv.pivot(index="slice", columns="ckpt", values="inv_mean")
    print(wide.to_string(float_format="%+.4f"))

    print("\n[3/3] Lockbox-suite overlap audit ...")
    overlap = lockbox_overlap_audit(panel)
    print(overlap.to_string(index=False))

    print(f"\nWrote outputs under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
