"""Paired-transport feature consistency probe — 2026-05-06.

Question: Does the encoder (P8A) already encode (raw, teams) viso pairs to
nearly-identical feature vectors? If yes, the failure is downstream
(head/threshold) and AugMix-on-frozen-encoder + new head is a viable lever.
If no, the encoder itself shifts between pairs and we need encoder-level
fine-tuning (PD-class) or paradigm shift (SBI / amplitude-aug).

Inputs:
  - analysis/clip_vs_p8a_viso_2026-05-03/outputs/p8a__features.npz
  - analysis/clip_vs_p8a_viso_2026-05-03/outputs/clip_b16_raw__features.npz
  - analysis/clip_vs_p8a_viso_2026-05-03/outputs/sample_manifest.csv
  - analysis/cpu_followups_2026-05-04/raw_reports/visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv
  - analysis/cpu_followups_2026-05-04/raw_reports/visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv

Pairing: parses subtype + seq_id from frame filename
`visomaster_enhanced_(raw|teams)__frame_NNNNNN_seqMMMM.png`. Pairs are
seq_ids appearing in both subtypes (n=275; identical scheme to prior
analysis/pair_loss_effect_verification_2026-05-05/).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
)
CLIP_DIR = ROOT / "analysis" / "clip_vs_p8a_viso_2026-05-03" / "outputs"
P8A_NPZ = CLIP_DIR / "p8a__features.npz"
CLIP_NPZ = CLIP_DIR / "clip_b16_raw__features.npz"
MANIFEST = CLIP_DIR / "sample_manifest.csv"

P8A_REPORT = (
    ROOT
    / "analysis"
    / "cpu_followups_2026-05-04"
    / "raw_reports"
    / "visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv"
)
E2B_REPORT = (
    ROOT
    / "analysis"
    / "cpu_followups_2026-05-04"
    / "raw_reports"
    / "visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv"
)

OUT = ROOT / "analysis" / "paired_feature_consistency_2026-05-06" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

FNAME_RE = re.compile(r"visomaster_enhanced_(raw|teams)__frame_(\d+)_seq(\d+)\.png")


def parse_frame_path(p: str) -> tuple[str, int, int] | None:
    fname = p.rsplit("/", 1)[-1]
    m = FNAME_RE.match(fname)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n


def stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "n": int(len(arr)),
    }


def main() -> None:
    print(f"Output dir: {OUT}")
    # ---- Load features ----
    print("\n[1] Loading features ...")
    p8a_npz = np.load(P8A_NPZ, allow_pickle=True)
    clip_npz = np.load(CLIP_NPZ, allow_pickle=True)
    p8a_feats = p8a_npz["features"]  # (1100, 512) float32
    clip_feats = clip_npz["features"]  # (1100, 512) float32
    p8a_paths = [str(x) for x in p8a_npz["frame_path"]]
    clip_paths = [str(x) for x in clip_npz["frame_path"]]
    p8a_label = p8a_npz["label"]
    print(
        f"  P8A features {p8a_feats.shape}, label counts {dict(zip(*np.unique(p8a_label, return_counts=True)))}"
    )
    print(f"  CLIP features {clip_feats.shape}")
    assert (
        p8a_paths == clip_paths
    ), "P8A and CLIP frame_path arrays must be aligned (they are by construction)"

    # ---- Build per-subtype seq_id index restricted to fakes (label==1) ----
    print("\n[2] Building paired seq_id index (subset to fakes) ...")
    p8a_index: dict[str, dict[int, int]] = {"raw": {}, "teams": {}}
    for i, (lbl, p) in enumerate(zip(p8a_label, p8a_paths)):
        if int(lbl) != 1:
            continue
        rec = parse_frame_path(p)
        if rec is None:
            continue
        subtype, _frame_num, seq_id = rec
        if seq_id not in p8a_index[subtype]:
            p8a_index[subtype][seq_id] = i

    raw_seqs = set(p8a_index["raw"].keys())
    teams_seqs = set(p8a_index["teams"].keys())
    pair_seqs = sorted(raw_seqs & teams_seqs)
    print(
        f"  fake-frame seq counts: raw={len(raw_seqs)}, teams={len(teams_seqs)}, paired={len(pair_seqs)}"
    )
    if len(pair_seqs) == 0:
        raise SystemExit("No pairs found — pairing regex likely mismatched.")

    # ---- Load score reports and index by (subtype, seq_id) ----
    print("\n[3] Loading P8A and E2B per-frame reports ...")
    df_p8a = pd.read_csv(P8A_REPORT)
    df_e2b = pd.read_csv(E2B_REPORT)
    print(f"  P8A report rows: {len(df_p8a)}; E2B report rows: {len(df_e2b)}")

    def index_report(df: pd.DataFrame) -> dict[str, dict[int, float]]:
        df = df[df["label"] == 1].copy()
        rec = df["frame_path"].apply(parse_frame_path)
        df = df[rec.notna()].copy()
        df["subtype"] = rec[rec.notna()].apply(lambda x: x[0])
        df["seq_id"] = rec[rec.notna()].apply(lambda x: x[2])
        out: dict[str, dict[int, float]] = {"raw": {}, "teams": {}}
        for _, row in df.iterrows():
            out[row["subtype"]][int(row["seq_id"])] = float(row["frame_prob"])
        return out

    p8a_score_idx = index_report(df_p8a)
    e2b_score_idx = index_report(df_e2b)
    print(
        f"  P8A scores: raw={len(p8a_score_idx['raw'])}, teams={len(p8a_score_idx['teams'])}"
    )
    print(
        f"  E2B scores: raw={len(e2b_score_idx['raw'])}, teams={len(e2b_score_idx['teams'])}"
    )

    # ---- Compute per-pair distances and lookup scores ----
    print("\n[4] Computing per-pair distances and gathering scores ...")
    rows = []
    n_missing_score = 0
    for sid in pair_seqs:
        i_raw = p8a_index["raw"][sid]
        i_teams = p8a_index["teams"][sid]
        # P8A features
        p_raw = p8a_feats[i_raw].astype(np.float64)
        p_teams = p8a_feats[i_teams].astype(np.float64)
        # CLIP-B16 features (same indices because frame_path arrays are aligned)
        c_raw = clip_feats[i_raw].astype(np.float64)
        c_teams = clip_feats[i_teams].astype(np.float64)

        # Cosine
        def cos_sim(a, b):
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        feat_cos_p8a = cos_sim(p_raw, p_teams)
        feat_cos_clip = cos_sim(c_raw, c_teams)
        # Cosine distance = 1 - cosine similarity
        feat_cos_dist_p8a = 1.0 - feat_cos_p8a
        feat_cos_dist_clip = 1.0 - feat_cos_clip
        # L2 distance after L2-normalization. Note: ||a/||a|| - b/||b|||| = sqrt(2*(1 - cos))
        feat_l2_p8a = float(np.sqrt(max(0.0, 2.0 * feat_cos_dist_p8a)))
        feat_l2_clip = float(np.sqrt(max(0.0, 2.0 * feat_cos_dist_clip)))

        # Scores
        s_p8a_raw = p8a_score_idx["raw"].get(sid, np.nan)
        s_p8a_teams = p8a_score_idx["teams"].get(sid, np.nan)
        s_e2b_raw = e2b_score_idx["raw"].get(sid, np.nan)
        s_e2b_teams = e2b_score_idx["teams"].get(sid, np.nan)
        if any(np.isnan([s_p8a_raw, s_p8a_teams, s_e2b_raw, s_e2b_teams])):
            n_missing_score += 1

        rows.append(
            {
                "pair_id": sid,
                "raw_npz_idx": i_raw,
                "teams_npz_idx": i_teams,
                "feat_cos_p8a": feat_cos_p8a,
                "feat_cos_dist_p8a": feat_cos_dist_p8a,
                "feat_l2_p8a": feat_l2_p8a,
                "feat_cos_clip": feat_cos_clip,
                "feat_cos_dist_clip": feat_cos_dist_clip,
                "feat_l2_clip": feat_l2_clip,
                "score_p8a_raw": s_p8a_raw,
                "score_p8a_teams": s_p8a_teams,
                "score_p8a_delta": float(s_p8a_raw - s_p8a_teams),
                "abs_score_p8a_delta": float(abs(s_p8a_raw - s_p8a_teams)),
                "score_e2b_raw": s_e2b_raw,
                "score_e2b_teams": s_e2b_teams,
                "score_e2b_delta": float(s_e2b_raw - s_e2b_teams),
                "abs_score_e2b_delta": float(abs(s_e2b_raw - s_e2b_teams)),
            }
        )
    df_pair = pd.DataFrame(rows)
    print(f"  built {len(df_pair)} pairs; n_missing_score={n_missing_score}")
    df_pair.to_csv(OUT / "pair_distances.csv", index=False)
    print(f"  wrote {OUT / 'pair_distances.csv'}")

    # ---- Distribution stats ----
    print("\n[5] Distribution stats ...")
    feat_cos_dist_p8a = df_pair["feat_cos_dist_p8a"].to_numpy()
    feat_l2_p8a = df_pair["feat_l2_p8a"].to_numpy()
    feat_cos_dist_clip = df_pair["feat_cos_dist_clip"].to_numpy()
    feat_l2_clip = df_pair["feat_l2_clip"].to_numpy()
    abs_dp8a = df_pair["abs_score_p8a_delta"].to_numpy()
    abs_de2b = df_pair["abs_score_e2b_delta"].to_numpy()

    dist_summary_rows = []

    def add(metric, arr):
        s = stats(arr)
        s["metric"] = metric
        dist_summary_rows.append(s)

    add("feat_cos_dist_p8a", feat_cos_dist_p8a)
    add("feat_l2_p8a", feat_l2_p8a)
    add("feat_cos_dist_clip_b16", feat_cos_dist_clip)
    add("feat_l2_clip_b16", feat_l2_clip)
    add("abs_score_delta_p8a", abs_dp8a)
    add("abs_score_delta_e2b", abs_de2b)
    add("score_p8a_raw", df_pair["score_p8a_raw"].to_numpy())
    add("score_p8a_teams", df_pair["score_p8a_teams"].to_numpy())
    add("score_e2b_raw", df_pair["score_e2b_raw"].to_numpy())
    add("score_e2b_teams", df_pair["score_e2b_teams"].to_numpy())
    df_dist = pd.DataFrame(dist_summary_rows)
    df_dist = df_dist[
        ["metric", "n", "mean", "std", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
    ]
    df_dist.to_csv(OUT / "distribution_summary.csv", index=False)
    print(f"  wrote {OUT / 'distribution_summary.csv'}")
    for _, r in df_dist.iterrows():
        print(
            f"    {r['metric']:<28s}  n={int(r['n']):4d}  mean={r['mean']:+.4f}  p25={r['p25']:+.4f}  p50={r['p50']:+.4f}  p75={r['p75']:+.4f}  p95={r['p95']:+.4f}"
        )

    # ---- Correlation + regression ----
    print("\n[6] Pearson r and OLS regression score_delta ~ feat_distance ...")
    # We use cosine DISTANCE (1 - cos sim) as the feature-distance metric.
    # P8A vs P8A scores
    r_p8a_abs, p_p8a_abs = pearsonr(feat_cos_dist_p8a, abs_dp8a)
    r_p8a_signed, p_p8a_signed = pearsonr(
        feat_cos_dist_p8a, df_pair["score_p8a_delta"].to_numpy()
    )
    # P8A feat dist vs E2B scores
    r_e2b_abs, p_e2b_abs = pearsonr(feat_cos_dist_p8a, abs_de2b)
    r_e2b_signed, p_e2b_signed = pearsonr(
        feat_cos_dist_p8a, df_pair["score_e2b_delta"].to_numpy()
    )
    # CLIP feat dist vs P8A and E2B scores (cross-encoder probe)
    r_clip_p8a_abs, p_clip_p8a_abs = pearsonr(feat_cos_dist_clip, abs_dp8a)
    r_clip_e2b_abs, p_clip_e2b_abs = pearsonr(feat_cos_dist_clip, abs_de2b)

    # OLS regressions
    def ols_r2(x, y):
        x = np.asarray(x).reshape(-1, 1)
        y = np.asarray(y).reshape(-1)
        m = LinearRegression(n_jobs=2)
        m.fit(x, y)
        yhat = m.predict(x)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            "slope": float(m.coef_[0]),
            "intercept": float(m.intercept_),
            "r2": float(r2),
            "n": int(len(y)),
        }

    ols_p8a_abs = ols_r2(feat_cos_dist_p8a, abs_dp8a)
    ols_p8a_signed = ols_r2(feat_cos_dist_p8a, df_pair["score_p8a_delta"].to_numpy())
    ols_e2b_abs = ols_r2(feat_cos_dist_p8a, abs_de2b)
    ols_e2b_signed = ols_r2(feat_cos_dist_p8a, df_pair["score_e2b_delta"].to_numpy())
    ols_clip_p8a = ols_r2(feat_cos_dist_clip, abs_dp8a)
    ols_clip_e2b = ols_r2(feat_cos_dist_clip, abs_de2b)

    regression = {
        "feat_metric": "cosine_distance (1 - cos)",
        "p8a_feat_vs_p8a_abs_score_delta": {
            "pearson_r": float(r_p8a_abs),
            "pearson_p": float(p_p8a_abs),
            **ols_p8a_abs,
        },
        "p8a_feat_vs_p8a_signed_score_delta": {
            "pearson_r": float(r_p8a_signed),
            "pearson_p": float(p_p8a_signed),
            **ols_p8a_signed,
        },
        "p8a_feat_vs_e2b_abs_score_delta": {
            "pearson_r": float(r_e2b_abs),
            "pearson_p": float(p_e2b_abs),
            **ols_e2b_abs,
        },
        "p8a_feat_vs_e2b_signed_score_delta": {
            "pearson_r": float(r_e2b_signed),
            "pearson_p": float(p_e2b_signed),
            **ols_e2b_signed,
        },
        "clip_feat_vs_p8a_abs_score_delta": {
            "pearson_r": float(r_clip_p8a_abs),
            "pearson_p": float(p_clip_p8a_abs),
            **ols_clip_p8a,
        },
        "clip_feat_vs_e2b_abs_score_delta": {
            "pearson_r": float(r_clip_e2b_abs),
            "pearson_p": float(p_clip_e2b_abs),
            **ols_clip_e2b,
        },
    }
    with open(OUT / "regression_score_on_feat.json", "w") as f:
        json.dump(regression, f, indent=2)
    print(f"  wrote {OUT / 'regression_score_on_feat.json'}")
    print(
        f"    P8A feat dist  vs  P8A |Δscore|  : r={r_p8a_abs:+.4f} (p={p_p8a_abs:.3g})  R^2={ols_p8a_abs['r2']:+.4f}"
    )
    print(
        f"    P8A feat dist  vs  E2B |Δscore|  : r={r_e2b_abs:+.4f} (p={p_e2b_abs:.3g})  R^2={ols_e2b_abs['r2']:+.4f}"
    )
    print(
        f"    CLIP feat dist vs  P8A |Δscore|  : r={r_clip_p8a_abs:+.4f} (p={p_clip_p8a_abs:.3g})  R^2={ols_clip_p8a['r2']:+.4f}"
    )
    print(
        f"    CLIP feat dist vs  E2B |Δscore|  : r={r_clip_e2b_abs:+.4f} (p={p_clip_e2b_abs:.3g})  R^2={ols_clip_e2b['r2']:+.4f}"
    )

    # ---- Compute per-pair P8A vs CLIP feat-distance ratio ----
    median_p8a = float(np.median(feat_cos_dist_p8a))
    median_clip = float(np.median(feat_cos_dist_clip))
    delta_median = median_p8a - median_clip  # negative => P8A is more invariant
    pct_pairs_p8a_smaller = float(
        np.mean(feat_cos_dist_p8a < feat_cos_dist_clip) * 100.0
    )
    print("\n[7] P8A-vs-CLIP per-pair feature distance comparison:")
    print(f"  median P8A cosine distance  = {median_p8a:.4f}")
    print(f"  median CLIP cosine distance = {median_clip:.4f}")
    print(
        f"  delta(P8A - CLIP) at median = {delta_median:+.4f}  "
        f"({'P8A more invariant' if delta_median < 0 else 'CLIP more invariant'})"
    )
    print(
        f"  fraction of pairs where P8A dist < CLIP dist: {pct_pairs_p8a_smaller:.1f}%"
    )

    # ---- Decision rule ----
    if median_p8a < 0.05 and median_clip > median_p8a:
        verdict = "ENCODER_INVARIANT_HEAD_FAILURE"
        recommendation = (
            "Encoder gained pair-invariance via FT (median pair cosine distance < 0.05 "
            "and CLIP-B16 baseline higher). Failure point is the head/threshold. "
            "Recommend AugMix on frozen encoder + new head, plus HSIC penalty if PD shifts."
        )
    elif median_p8a > 0.15:
        verdict = "ENCODER_NOT_INVARIANT"
        recommendation = (
            "Encoder itself shifts substantially between (raw, teams) pairs "
            "(median pair cosine distance > 0.15). Recommend SBI / amplitude-aug "
            "paradigm shift or PD-class encoder fine-tuning."
        )
    else:
        verdict = "ENCODER_PARTIAL_INVARIANCE"
        recommendation = (
            "Encoder is partially invariant. Both head-axis (AugMix on frozen encoder + "
            "new head) and encoder-axis (PD-class FT, SBI) levers are worth pursuing."
        )

    decision = {
        "median_pair_feat_cosine_distance_p8a": median_p8a,
        "median_pair_feat_cosine_distance_clip_b16": median_clip,
        "delta_median_p8a_minus_clip": delta_median,
        "pct_pairs_p8a_dist_smaller_than_clip": pct_pairs_p8a_smaller,
        "n_pairs": int(len(df_pair)),
        "thresholds": {
            "encoder_invariant_max_median_cos_dist": 0.05,
            "encoder_not_invariant_min_median_cos_dist": 0.15,
        },
        "verdict": verdict,
        "recommendation": recommendation,
    }
    with open(OUT / "decision.json", "w") as f:
        json.dump(decision, f, indent=2)
    print(f"\n[8] Verdict: {verdict}")
    print(f"   wrote {OUT / 'decision.json'}")

    # ---- Hand back summary numbers for the FINDINGS draft ----
    summary = {
        "n_pairs": int(len(df_pair)),
        "median_pair_cosine_distance_p8a": median_p8a,
        "median_pair_cosine_distance_clip_b16": median_clip,
        "p25_p75_p95_p8a_cos_dist": [
            float(np.percentile(feat_cos_dist_p8a, 25)),
            float(np.percentile(feat_cos_dist_p8a, 75)),
            float(np.percentile(feat_cos_dist_p8a, 95)),
        ],
        "p25_p75_p95_clip_cos_dist": [
            float(np.percentile(feat_cos_dist_clip, 25)),
            float(np.percentile(feat_cos_dist_clip, 75)),
            float(np.percentile(feat_cos_dist_clip, 95)),
        ],
        "median_p8a_l2": float(np.median(feat_l2_p8a)),
        "median_clip_l2": float(np.median(feat_l2_clip)),
        "pct_pairs_p8a_dist_smaller_than_clip": pct_pairs_p8a_smaller,
        "regression": regression,
        "verdict": verdict,
        "recommendation": recommendation,
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   wrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
