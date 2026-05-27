"""Pair-loss effect verification — viso enhanced macro dev.

Analyzes whether a pair-aware consistency loss has meaningful effect potential
by quantifying:
  Q1 — feature gap between paired (raw, teams) frames vs random within subtype
  Q2 — score-gap vs feature-gap correlation
  Q3 — upper-bound recall lift if pair loss perfectly equated teams to raw
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = ROOT / "analysis" / "pair_loss_effect_verification_2026-05-05"
OUT.mkdir(parents=True, exist_ok=True)

NPZ_PATH = ROOT / "analysis" / "job7_head_retrain_2026-05-04" / "frozen_features" / "visomaster_enhanced_macro_dev_p8a_features.npz"
E2B_REPORT = ROOT / "analysis" / "cpu_followups_2026-05-04" / "raw_reports" / "visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv"
P8A_REPORT = ROOT / "analysis" / "cpu_followups_2026-05-04" / "raw_reports" / "visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv"
E3_REPORT = ROOT / "analysis" / "cpu_followups_2026-05-04" / "raw_reports" / "visomaster_enhanced_macro_dev_e3_top_n_step6600_frames_report.csv"

# Filenames: visomaster_enhanced_(raw|teams)__frame_NNNNNN_seqMMMM.png
FNAME_RE = re.compile(r"visomaster_enhanced_(raw|teams)__frame_(\d+)_seq(\d+)\.png")


def parse_frame_path(p: str) -> tuple[str, int, int] | None:
    """Return (subtype, frame_num, seq_id) or None."""
    fname = p.rsplit("/", 1)[-1]
    m = FNAME_RE.match(fname)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cos_sim_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine sim between A[i] and B[i]."""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return np.sum(An * Bn, axis=1)


def main() -> None:
    print(f"Loading P8A features from {NPZ_PATH.name} ...")
    npz = np.load(NPZ_PATH, allow_pickle=True)
    feats = npz["features"]  # (550, 512)
    paths = npz["frame_paths"]  # (550,)
    p8a_scores = npz["scores_p8a_existing"]
    print(f"  features shape: {feats.shape}")
    print(f"  paths n={len(paths)}, p8a_scores n={len(p8a_scores)}")

    # Build per-frame index: subtype -> {seq_id -> (idx_in_npz, frame_num)}
    npz_index: dict[str, dict[int, tuple[int, int]]] = {"raw": {}, "teams": {}}
    parsed = []
    for i, p in enumerate(paths):
        rec = parse_frame_path(str(p))
        parsed.append(rec)
        if rec is None:
            continue
        subtype, frame_num, seq_id = rec
        if seq_id in npz_index[subtype]:
            # If duplicate seq, prefer earlier; should not happen if pairs are 1:1
            continue
        npz_index[subtype][seq_id] = (i, frame_num)

    n_raw_npz = len(npz_index["raw"])
    n_teams_npz = len(npz_index["teams"])
    print(f"  npz unique seq counts: raw={n_raw_npz}, teams={n_teams_npz}")

    # Load E2B per-frame report
    print(f"\nLoading E2B report from {E2B_REPORT.name} ...")
    df_e2b = pd.read_csv(E2B_REPORT)
    print(f"  rows: {len(df_e2b)}, cols: {list(df_e2b.columns)}")

    # Filter to fakes (label==1)
    df_e2b = df_e2b[df_e2b["label"] == 1].copy()
    df_e2b["parsed"] = df_e2b["frame_path"].apply(parse_frame_path)
    df_e2b = df_e2b[df_e2b["parsed"].notna()].copy()
    df_e2b["subtype"] = df_e2b["parsed"].apply(lambda x: x[0])
    df_e2b["frame_num"] = df_e2b["parsed"].apply(lambda x: x[1])
    df_e2b["seq_id"] = df_e2b["parsed"].apply(lambda x: x[2])
    df_e2b = df_e2b.drop(columns=["parsed"])
    print(f"  parsed fake rows: {len(df_e2b)}, subtypes: {df_e2b['subtype'].value_counts().to_dict()}")

    # Build E2B subtype -> seq_id -> score map
    e2b_index: dict[str, dict[int, float]] = {"raw": {}, "teams": {}}
    for _, row in df_e2b.iterrows():
        e2b_index[row["subtype"]][int(row["seq_id"])] = float(row["frame_prob"])

    # Find paired seq_ids: present in both subtypes in BOTH npz and e2b
    raw_seqs = set(npz_index["raw"].keys()) & set(e2b_index["raw"].keys())
    teams_seqs = set(npz_index["teams"].keys()) & set(e2b_index["teams"].keys())
    pair_seqs = sorted(raw_seqs & teams_seqs)
    print(f"\nPaired seq_ids (in both subtypes, both sources): {len(pair_seqs)}")

    # Build per-pair table
    rows = []
    for sid in pair_seqs:
        raw_idx, raw_frame = npz_index["raw"][sid]
        teams_idx, teams_frame = npz_index["teams"][sid]
        f_raw = feats[raw_idx]
        f_teams = feats[teams_idx]
        cs = cos_sim(f_raw, f_teams)
        e2b_raw = e2b_index["raw"][sid]
        e2b_teams = e2b_index["teams"][sid]
        rows.append({
            "seq_id": sid,
            "raw_frame_num": raw_frame,
            "teams_frame_num": teams_frame,
            "raw_score_e2b": e2b_raw,
            "teams_score_e2b": e2b_teams,
            "score_gap": e2b_raw - e2b_teams,
            "abs_score_gap": abs(e2b_raw - e2b_teams),
            "raw_score_p8a": float(p8a_scores[raw_idx]),
            "teams_score_p8a": float(p8a_scores[teams_idx]),
            "feature_cosine_p8a": cs,
            "feature_distance": 1.0 - cs,
        })
    df_pair = pd.DataFrame(rows)
    df_pair.to_csv(OUT / "per_pair_analysis.csv", index=False)
    print(f"  wrote per_pair_analysis.csv with {len(df_pair)} pairs")

    # ---- Q1: feature gap (paired vs random) ----
    paired_cos = df_pair["feature_cosine_p8a"].to_numpy()

    def stats(arr: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
            "n": int(len(arr)),
        }

    paired_stats = stats(paired_cos)
    print(f"\nQ1 — paired (raw,teams) cosine similarity: mean={paired_stats['mean']:.4f}, p50={paired_stats['p50']:.4f}, p10={paired_stats['p10']:.4f}, p90={paired_stats['p90']:.4f}")

    # Random baseline: 200 random pairs within each subtype, paths must NOT share seq_id
    rng = np.random.default_rng(42)
    rand_records = []
    for subtype in ("raw", "teams"):
        seq_to_idx = npz_index[subtype]
        seqs = list(seq_to_idx.keys())
        n_target = 200
        attempts = 0
        max_attempts = n_target * 50
        while len(rand_records) < (n_target if subtype == "raw" else 2 * n_target) and attempts < max_attempts:
            i, j = rng.integers(0, len(seqs), size=2)
            if i == j:
                attempts += 1
                continue
            si, sj = seqs[i], seqs[j]
            if si == sj:
                attempts += 1
                continue
            idx_i = seq_to_idx[si][0]
            idx_j = seq_to_idx[sj][0]
            cs = cos_sim(feats[idx_i], feats[idx_j])
            rand_records.append({
                "subtype": subtype,
                "seq_a": si,
                "seq_b": sj,
                "cosine": cs,
            })
            attempts += 1
    df_rand = pd.DataFrame(rand_records)
    df_rand.to_csv(OUT / "random_baseline.csv", index=False)
    rand_cos = df_rand["cosine"].to_numpy()
    rand_stats = stats(rand_cos)
    rand_stats_per_subtype = {
        "raw": stats(df_rand[df_rand["subtype"] == "raw"]["cosine"].to_numpy()),
        "teams": stats(df_rand[df_rand["subtype"] == "teams"]["cosine"].to_numpy()),
    }
    print(f"  random baseline (within-subtype): mean={rand_stats['mean']:.4f}, p50={rand_stats['p50']:.4f}")
    print(f"    raw-within mean={rand_stats_per_subtype['raw']['mean']:.4f}, teams-within mean={rand_stats_per_subtype['teams']['mean']:.4f}")

    # Cross-subtype random baseline (raw vs teams of DIFFERENT seq_id)
    cross_records = []
    raw_seqs_list = list(npz_index["raw"].keys())
    teams_seqs_list = list(npz_index["teams"].keys())
    attempts = 0
    while len(cross_records) < 200 and attempts < 200 * 50:
        i = rng.integers(0, len(raw_seqs_list))
        j = rng.integers(0, len(teams_seqs_list))
        sr = raw_seqs_list[i]
        st = teams_seqs_list[j]
        if sr == st:
            attempts += 1
            continue
        ir = npz_index["raw"][sr][0]
        it = npz_index["teams"][st][0]
        cs = cos_sim(feats[ir], feats[it])
        cross_records.append({
            "seq_raw": sr,
            "seq_teams": st,
            "cosine": cs,
        })
        attempts += 1
    df_cross = pd.DataFrame(cross_records)
    df_cross.to_csv(OUT / "random_baseline_cross_subtype.csv", index=False)
    cross_cos = df_cross["cosine"].to_numpy()
    cross_stats = stats(cross_cos)
    print(f"  cross-subtype random (raw vs teams, different seq): mean={cross_stats['mean']:.4f}, p50={cross_stats['p50']:.4f}")

    # ---- Q2: score↔feature correlation ----
    feat_dist = df_pair["feature_distance"].to_numpy()
    score_gap_abs = df_pair["abs_score_gap"].to_numpy()
    score_gap_signed = df_pair["score_gap"].to_numpy()
    r_abs, p_abs = pearsonr(feat_dist, score_gap_abs)
    r_signed, p_signed = pearsonr(feat_dist, score_gap_signed)
    print(f"\nQ2 — Pearson r(feature_distance, |score_gap|) = {r_abs:.4f} (p={p_abs:.4g})")
    print(f"     Pearson r(feature_distance, score_gap signed [raw-teams]) = {r_signed:.4f} (p={p_signed:.4g})")

    # ---- Q3: upper-bound recall lift ----
    # Cohort: raw_score > 0.5 AND teams_score < 0.5 (E2B catches clean, misses transported)
    cohort = df_pair[(df_pair["raw_score_e2b"] > 0.5) & (df_pair["teams_score_e2b"] < 0.5)]
    n_cohort = len(cohort)
    n_pairs = len(df_pair)
    n_total_viso_fakes_in_pairs = 2 * n_pairs  # both raw + teams frames
    cohort_mean_raw = float(cohort["raw_score_e2b"].mean()) if n_cohort > 0 else 0.0
    cohort_mean_teams = float(cohort["teams_score_e2b"].mean()) if n_cohort > 0 else 0.0
    upper_bound_pp_against_pairs = (n_cohort / n_pairs) * 100.0  # in pair-terms
    # Also against the "275 total" framing in the prompt — n_pairs is our actual count
    print(f"\nQ3 — missed-teams cohort (E2B raw>0.5 & teams<0.5):")
    print(f"     n_cohort = {n_cohort} / {n_pairs} pairs ({100.0*n_cohort/n_pairs:.1f}%)")
    print(f"     mean raw_score (E2B) in cohort = {cohort_mean_raw:.4f}")
    print(f"     mean teams_score (E2B) in cohort = {cohort_mean_teams:.4f}")
    print(f"     upper-bound recall lift if perfect transfer = {upper_bound_pp_against_pairs:.1f} pp (of total viso pairs)")

    # Also report expected realistic lift assuming 30-60% transfer efficiency
    realistic_lift_lo = 0.30 * upper_bound_pp_against_pairs
    realistic_lift_hi = 0.60 * upper_bound_pp_against_pairs

    # ---- Sanity: how many cohort frames have small vs large feature distance? ----
    if n_cohort > 0:
        feat_dist_cohort = cohort["feature_distance"].to_numpy()
        cohort_feat_stats = stats(feat_dist_cohort)
    else:
        cohort_feat_stats = None

    # Verdict
    verdict_q1 = "STRONG" if paired_stats["mean"] < 0.75 else ("MODERATE" if paired_stats["mean"] < 0.95 else "LOW")
    verdict_q2 = "STRONG" if r_abs > 0.4 else ("MODERATE" if r_abs > 0.2 else "LOW")
    verdict_q3 = "STRONG" if upper_bound_pp_against_pairs > 30.0 else ("MODERATE" if upper_bound_pp_against_pairs > 10.0 else "LOW")
    verdicts = [verdict_q1, verdict_q2, verdict_q3]
    if all(v == "STRONG" for v in verdicts):
        overall = "STRONG"
    elif any(v == "STRONG" for v in verdicts) or sum(v != "LOW" for v in verdicts) >= 2:
        overall = "MODERATE"
    else:
        overall = "LOW"

    summary = {
        "n_pairs": int(n_pairs),
        "n_npz_total": int(len(paths)),
        "n_raw_npz": n_raw_npz,
        "n_teams_npz": n_teams_npz,
        "Q1_paired_cosine_p8a": paired_stats,
        "Q1_random_within_subtype_cosine_p8a": rand_stats,
        "Q1_random_within_subtype_cosine_p8a_per_subtype": rand_stats_per_subtype,
        "Q1_random_cross_subtype_cosine_p8a": cross_stats,
        "Q1_paired_minus_random_within": paired_stats["mean"] - rand_stats["mean"],
        "Q1_paired_minus_random_cross": paired_stats["mean"] - cross_stats["mean"],
        "Q1_verdict": verdict_q1,
        "Q2_pearson_r_feat_dist_vs_abs_score_gap": float(r_abs),
        "Q2_pearson_p_feat_dist_vs_abs_score_gap": float(p_abs),
        "Q2_pearson_r_feat_dist_vs_signed_score_gap_raw_minus_teams": float(r_signed),
        "Q2_pearson_p_signed": float(p_signed),
        "Q2_verdict": verdict_q2,
        "Q3_cohort_n": int(n_cohort),
        "Q3_cohort_pct_of_pairs": float(100.0 * n_cohort / n_pairs),
        "Q3_cohort_mean_raw_score_e2b": cohort_mean_raw,
        "Q3_cohort_mean_teams_score_e2b": cohort_mean_teams,
        "Q3_upper_bound_recall_lift_pp_pair_basis": upper_bound_pp_against_pairs,
        "Q3_realistic_lift_pp_30pct": realistic_lift_lo,
        "Q3_realistic_lift_pp_60pct": realistic_lift_hi,
        "Q3_cohort_feature_distance_stats": cohort_feat_stats,
        "Q3_verdict": verdict_q3,
        "overall_verdict": overall,
    }

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary.json — overall verdict: {overall}")

    # FINDINGS.md
    findings = []
    findings.append("# Pair-loss effect verification — visomaster enhanced macro dev (E2B + P8A frozen features)\n")
    findings.append(f"_Date: 2026-05-05_\n")
    findings.append(f"## Verdict: **{overall}**\n")
    findings.append(f"- Q1 (feature gap): **{verdict_q1}**\n")
    findings.append(f"- Q2 (score↔feature correlation): **{verdict_q2}**\n")
    findings.append(f"- Q3 (upper-bound recall lift): **{verdict_q3}**\n")
    findings.append("\n---\n")
    findings.append("## Setup\n")
    findings.append(f"- N pairs (seq_id present in both raw & teams subtypes, both in npz and E2B report): **{n_pairs}**\n")
    findings.append(f"- npz raw seqs: {n_raw_npz}, teams seqs: {n_teams_npz}; total npz frames: {len(paths)}\n")
    findings.append(f"- Features: P8A frozen penultimate (512-d) from `visomaster_enhanced_macro_dev_p8a_features.npz`\n")
    findings.append(f"- Scores: E2B step3200 (`visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv`)\n")
    findings.append(f"- Pair extraction regex: `visomaster_enhanced_(raw|teams)__frame_NNNNNN_seqMMMM\\.png`\n")
    findings.append(f"- Caveat: P8A features are a PROXY for E2B features (E2B features not cached on disk)\n")
    findings.append("\n## Q1 — Feature gap: paired (raw,teams) vs random within subtype\n")
    findings.append(f"| stat | paired (raw,teams) | random within subtype | random cross-subtype (raw vs teams of diff seq) |\n")
    findings.append(f"|---|---|---|---|\n")
    for k in ("mean", "p10", "p25", "p50", "p75", "p90", "p99"):
        findings.append(f"| {k} | {paired_stats[k]:.4f} | {rand_stats[k]:.4f} | {cross_stats[k]:.4f} |\n")
    findings.append(f"\n**Δ mean(paired − random_within) = {summary['Q1_paired_minus_random_within']:+.4f}** "
                    f"(positive = paired closer than random)\n")
    findings.append(f"**Δ mean(paired − random_cross) = {summary['Q1_paired_minus_random_cross']:+.4f}**\n")
    findings.append(f"\nInterpretation thresholds: <0.75 STRONG, 0.75–0.95 MODERATE, >0.95 LOW.\n")
    findings.append(f"→ Paired mean cosine = **{paired_stats['mean']:.4f}** → **{verdict_q1}**.\n")
    findings.append("\n## Q2 — Score↔feature correlation\n")
    findings.append(f"- Pearson r(feature_distance, |E2B_score_gap|) = **{r_abs:.4f}** (p = {p_abs:.4g})\n")
    findings.append(f"- Pearson r(feature_distance, signed E2B raw−teams) = **{r_signed:.4f}** (p = {p_signed:.4g})\n")
    findings.append(f"\nInterpretation thresholds: r>0.4 STRONG, 0.2–0.4 MODERATE, <0.2 LOW.\n")
    findings.append(f"→ |r| = **{r_abs:.4f}** → **{verdict_q2}**.\n")
    findings.append(f"\n*Caveat:* feature distance comes from P8A; score gap comes from E2B. ")
    findings.append("If E2B features differ from P8A in their geometry, this correlation could under- or over-estimate the true E2B feature↔score linkage. ")
    findings.append("Phase-1 audits show E2B and P8A track different shortcut axes (image quality vs identity-cluster), so r should be read as a lower bound on the within-model linkage in P8A and an unknown (probably weaker) estimator for E2B.\n")
    findings.append("\n## Q3 — Upper-bound recall lift\n")
    findings.append(f"- Cohort definition: pairs where E2B raw_score > 0.5 AND teams_score < 0.5\n")
    findings.append(f"- n_cohort = **{n_cohort}** out of **{n_pairs}** pairs ({100.0*n_cohort/n_pairs:.1f}% of pairs)\n")
    findings.append(f"- Mean raw_score (E2B) in cohort = **{cohort_mean_raw:.4f}**\n")
    findings.append(f"- Mean teams_score (E2B) in cohort = **{cohort_mean_teams:.4f}**\n")
    findings.append(f"- Implied per-frame score lift if perfect transfer = {cohort_mean_raw - cohort_mean_teams:.4f}\n")
    findings.append(f"- **Upper-bound recall lift on viso (pair basis): {upper_bound_pp_against_pairs:.1f} pp**\n")
    findings.append(f"- Realistic lift band (30–60% transfer efficiency): **{realistic_lift_lo:.1f}–{realistic_lift_hi:.1f} pp**\n")
    findings.append(f"\nInterpretation thresholds: >30 pp STRONG, 10–30 pp MODERATE, <10 pp LOW.\n")
    findings.append(f"→ Upper-bound = **{upper_bound_pp_against_pairs:.1f} pp** → **{verdict_q3}**.\n")
    findings.append(f"\n*Caveat:* this assumes the cohort's frames CAN be lifted by representation alignment alone. ")
    findings.append("If the missed-teams frames are missed because of irreversible information loss (e.g. severe sharpness collapse making the face genuinely unreadable), pair loss can't recover them. ")
    findings.append(f"Cohort feature_distance stats: {cohort_feat_stats}\n")
    findings.append("\n## Recommendation\n")
    if overall == "STRONG":
        findings.append("Launch the pair-loss packet. All three signals (feature gap, score-feat correlation, upper-bound headroom) point in the same direction.\n")
    elif overall == "MODERATE":
        findings.append("Mixed signals. Suggest one of: (a) extract E2B features (1 GPU hour) and re-run Q2/Q3 directly on E2B; (b) launch a TINY pair-loss probe (e.g. 1–2k steps from P8A base, λ_pair=0.1, single seed) before committing the full 6h+$87 packet.\n")
    else:
        findings.append("Skip the pair-loss packet. The cached signals do not support meaningful recall lift. Reallocate the 6h+$87 budget to: substrate cleaning (Job 14 follow-up), threshold-relaxed deployment, or out-of-stream router work.\n")
    with open(OUT / "FINDINGS.md", "w") as f:
        f.write("".join(findings))
    print(f"Wrote FINDINGS.md\n")

    # Console final summary
    print("=" * 70)
    print(f"OVERALL VERDICT: {overall}")
    print(f"  Q1 paired cosine mean = {paired_stats['mean']:.4f} (>0.95 LOW, 0.75-0.95 MOD, <0.75 STRONG) → {verdict_q1}")
    print(f"  Q2 |r| = {r_abs:.4f} (>0.4 STRONG, 0.2-0.4 MOD, <0.2 LOW) → {verdict_q2}")
    print(f"  Q3 upper bound = {upper_bound_pp_against_pairs:.1f} pp (>30 STRONG, 10-30 MOD, <10 LOW) → {verdict_q3}")


if __name__ == "__main__":
    main()
