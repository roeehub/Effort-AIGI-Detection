"""Job beta: P8A signature decomposition.

Question: P8A is a 5-stage FT chain. If it does something better on viso, we
should be able to (a) describe what it catches that scratch ckpts (E2B B16
scratch+CE+aug, E3 L14 scratch+CE+aug) don't, and (b) hypothesize a from-
scratch recipe that captures the same lesson.

Cohorts on viso fakes:
    A: caught by all three (broad signal)
    B: caught by P8A AND E3, missed by E2B (B16-scratch loses; CLIP-init OR L14
       capacity preserves)
    C: caught by P8A only, missed by E2B AND E3 (P8A-unique signal)
    D: missed by all (truly hard)

For each cohort: profile IQ attributes, subtype (raw vs teams), score
distribution. The B cohort is the most informative for the user's question
(what survives in P8A but dies in B16-scratch).

Inputs:
- per-frame reports for P8A/E2B/E3 on viso, deeplive, teams
- crop_attributes.csv (IQ tags by seq_id+subtype+frame_num)
- full_tags_2026-04-27.parquet (for any extra coverage)

Outputs:
- viso_cohort_assignments.csv
- viso_cohort_iq_profile.csv
- viso_cohort_per_subtype.csv
- viso_cohort_per_seq.csv (which sequences fall into each cohort)
- summary.json
"""

import os
import re
import json
import pandas as pd
import numpy as np

ROOT = "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
RAW = f"{ROOT}/analysis/cpu_followups_2026-05-04/raw_reports"
CROP_ATTRS = f"{ROOT}/analysis/score_distribution_2026-05-02/outputs/crop_attributes.csv"
TAGS = f"{ROOT}/analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
OUT = f"{ROOT}/analysis/p8a_signature_decomposition_2026-05-05"

os.makedirs(OUT, exist_ok=True)

# tau values from f1_recall_2026-05-04 (F0 calibrated at FPR=10%)
TAU_F0 = {"P8A": 0.70503, "E2B_3200": 0.50644, "E3_6600": 0.85196}

VISO_FILES = {
    "P8A":      f"{RAW}/visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv",
    "E2B_3200": f"{RAW}/visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv",
    "E3_6600":  f"{RAW}/visomaster_enhanced_macro_dev_e3_top_n_step6600_frames_report.csv",
}

# ----------------------------------------------------------------------
# 1. Load + join viso fake scores across 3 ckpts
# ----------------------------------------------------------------------
print("Loading viso fake reports...")
dfs = {}
for ck, path in VISO_FILES.items():
    d = pd.read_csv(path)
    d = d.rename(columns={"frame_prob": f"score_{ck}"})
    dfs[ck] = d[["frame_path", "label", "video_id", "group_key", "family_key", f"score_{ck}"]]

m = dfs["P8A"]
for ck in ["E2B_3200", "E3_6600"]:
    m = m.merge(dfs[ck][["frame_path", f"score_{ck}"]], on="frame_path", how="inner")
print(f"  joined viso fakes: {m.shape[0]} frames (expected ~550)")

# Restrict to fakes (label=1) — should already be all-fake but be safe
m = m[m["label"] == 1].copy()
print(f"  fake-only: {m.shape[0]}")

# ----------------------------------------------------------------------
# 2. Cohort assignment
# ----------------------------------------------------------------------
m["caught_P8A"] = m["score_P8A"] >= TAU_F0["P8A"]
m["caught_E2B"] = m["score_E2B_3200"] >= TAU_F0["E2B_3200"]
m["caught_E3"]  = m["score_E3_6600"]  >= TAU_F0["E3_6600"]

def cohort(row):
    p, e2, e3 = row["caught_P8A"], row["caught_E2B"], row["caught_E3"]
    if p and e2 and e3: return "A_caught_all"
    if p and not e2 and e3: return "B_caught_P8A_E3_not_E2B"
    if p and not e2 and not e3: return "C_caught_P8A_only"
    if not p and e2 and not e3: return "X_caught_E2B_only"
    if not p and not e2 and e3: return "Y_caught_E3_only"
    if p and e2 and not e3: return "Z_caught_P8A_E2B_not_E3"
    if not p and e2 and e3: return "W_caught_E2B_E3_not_P8A"
    return "D_missed_all"

m["cohort"] = m.apply(cohort, axis=1)
print("\nCohort sizes (out of {} total fakes):".format(len(m)))
cohort_counts = m["cohort"].value_counts()
print(cohort_counts.to_string())
print(f"\nP8A-positive: {m['caught_P8A'].sum()}")
print(f"E2B-positive: {m['caught_E2B'].sum()}")
print(f"E3-positive:  {m['caught_E3'].sum()}")

# ----------------------------------------------------------------------
# 3. Extract seq_id + subtype from filename
# ----------------------------------------------------------------------
def parse_filename(frame_path: str) -> dict:
    base = os.path.basename(frame_path)
    # patterns: visomaster_enhanced_raw__frame_NNNNNN_seqMMMM.png
    #           visomaster_enhanced_teams__frame_NNNNNN_seqMMMM.png
    mtch = re.match(r"visomaster_enhanced_(raw|teams)__frame_(\d+)_seq(\d+)\.png$", base)
    if mtch:
        subtype = mtch.group(1)
        frame_num = int(mtch.group(2))
        seq_id = f"seq{mtch.group(3)}"
        return {"subtype": subtype, "frame_num": frame_num, "seq_id": seq_id, "filename": base}
    return {"subtype": "UNKNOWN", "frame_num": -1, "seq_id": "UNKNOWN", "filename": base}

parsed = m["frame_path"].apply(parse_filename).apply(pd.Series)
m = pd.concat([m, parsed], axis=1)
print("\nSubtype distribution (full sample):")
print(m["subtype"].value_counts())

# ----------------------------------------------------------------------
# 4. Cohort × subtype
# ----------------------------------------------------------------------
ct = pd.crosstab(m["cohort"], m["subtype"], margins=True)
print("\nCohort × subtype:")
print(ct)
ct.to_csv(f"{OUT}/cohort_x_subtype.csv")

# ----------------------------------------------------------------------
# 5. Join IQ attributes
# ----------------------------------------------------------------------
print("\nLoading IQ attributes...")
iq = pd.read_csv(CROP_ATTRS)
print(f"  attrs shape: {iq.shape}; coverage on dev viso fakes: see crop_attributes.csv subtype field")
# crop_attributes has filename column; join on filename
m_iq = m.merge(iq, on=["filename"], how="left", suffixes=("", "_iq"))
covered = m_iq["luma_mean"].notna().sum()
print(f"  IQ coverage on viso fakes: {covered}/{len(m_iq)} = {covered/len(m_iq):.1%}")

# ----------------------------------------------------------------------
# 6. Per-cohort IQ profile (mean + std per attribute, per cohort)
# ----------------------------------------------------------------------
IQ_COLS = ["luma_mean", "luma_p10", "luma_p90", "laplacian_var", "sobel_edge_mean",
           "saturation_mean", "skin_frac"]

profile_rows = []
for ch, g in m_iq.groupby("cohort"):
    g_iq = g[g["luma_mean"].notna()]
    if len(g_iq) == 0:
        continue
    row = {"cohort": ch, "n_total": len(g), "n_with_iq": len(g_iq)}
    for col in IQ_COLS:
        row[f"{col}_mean"] = float(g_iq[col].mean())
        row[f"{col}_p50"]  = float(g_iq[col].quantile(0.50))
    profile_rows.append(row)
profile_df = pd.DataFrame(profile_rows)
print("\nPer-cohort IQ profile (mean):")
print(profile_df[["cohort", "n_total", "n_with_iq",
                  "laplacian_var_mean", "sobel_edge_mean_mean",
                  "luma_mean_mean", "saturation_mean_mean", "skin_frac_mean"]].to_string(index=False))
profile_df.to_csv(f"{OUT}/per_cohort_iq_profile.csv", index=False)

# ----------------------------------------------------------------------
# 7. Hypothesis tests: B-cohort (P8A+E3 catch, E2B miss) vs A-cohort (all catch)
#                     C-cohort (P8A only) vs A-cohort
# ----------------------------------------------------------------------
from scipy import stats
pair_rows = []
for target_cohort in ["B_caught_P8A_E3_not_E2B", "C_caught_P8A_only", "D_missed_all"]:
    a = m_iq[(m_iq["cohort"] == "A_caught_all") & m_iq["luma_mean"].notna()]
    b = m_iq[(m_iq["cohort"] == target_cohort) & m_iq["luma_mean"].notna()]
    if len(b) < 5 or len(a) < 5:
        print(f"  skip {target_cohort}: n_a={len(a)}, n_b={len(b)}")
        continue
    for col in IQ_COLS:
        ks_stat, ks_p = stats.ks_2samp(a[col].dropna(), b[col].dropna())
        pair_rows.append({
            "cohort": target_cohort,
            "vs": "A_caught_all",
            "feature": col,
            "n_target": len(b),
            "n_a": len(a),
            "mean_target": float(b[col].mean()),
            "mean_a": float(a[col].mean()),
            "delta_mean": float(b[col].mean() - a[col].mean()),
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
        })
pair_df = pd.DataFrame(pair_rows)
print("\nKS tests cohort-vs-A on each IQ feature:")
print(pair_df.to_string(index=False))
pair_df.to_csv(f"{OUT}/cohort_vs_A_ks_tests.csv", index=False)

# ----------------------------------------------------------------------
# 8. Per-cohort score distribution shape (for E2B and E3 on the cohort)
#    e.g., on B cohort, what does E2B's score distribution look like? (it
#    misses, but where in [0,1]?)
# ----------------------------------------------------------------------
score_rows = []
for ch, g in m.groupby("cohort"):
    if len(g) == 0:
        continue
    for ck, score_col in [("P8A", "score_P8A"), ("E2B_3200", "score_E2B_3200"), ("E3_6600", "score_E3_6600")]:
        score_rows.append({
            "cohort": ch,
            "ckpt": ck,
            "n": len(g),
            "mean": float(g[score_col].mean()),
            "p10": float(g[score_col].quantile(0.10)),
            "p25": float(g[score_col].quantile(0.25)),
            "p50": float(g[score_col].quantile(0.50)),
            "p75": float(g[score_col].quantile(0.75)),
            "p90": float(g[score_col].quantile(0.90)),
            "tau_F0": TAU_F0[ck],
        })
score_dist = pd.DataFrame(score_rows)
print("\nScore distributions per cohort × ckpt:")
print(score_dist.to_string(index=False))
score_dist.to_csv(f"{OUT}/per_cohort_score_distribution.csv", index=False)

# ----------------------------------------------------------------------
# 9. Sequence-level: which sequences contribute to which cohort?
# ----------------------------------------------------------------------
seq_x_cohort = pd.crosstab(m["seq_id"], m["cohort"])
seq_x_cohort.to_csv(f"{OUT}/seq_x_cohort.csv")
print(f"\nUnique seq_ids: {m['seq_id'].nunique()}")
print(f"Top sequences by frame count (showing cohort distribution):")
seq_totals = m.groupby("seq_id").size().sort_values(ascending=False)
print(seq_totals.head(15).to_string())

# ----------------------------------------------------------------------
# 10. Save merged frame-level table for the viewer
# ----------------------------------------------------------------------
keep_cols = ["frame_path", "video_id", "filename", "seq_id", "subtype", "frame_num",
             "score_P8A", "score_E2B_3200", "score_E3_6600",
             "caught_P8A", "caught_E2B", "caught_E3", "cohort",
             "luma_mean", "luma_p10", "luma_p90", "laplacian_var",
             "sobel_edge_mean", "saturation_mean", "skin_frac"]
keep_cols = [c for c in keep_cols if c in m_iq.columns]
m_iq[keep_cols].to_csv(f"{OUT}/viso_cohort_assignments.csv", index=False)

# ----------------------------------------------------------------------
# 11. Summary
# ----------------------------------------------------------------------
summary = {
    "n_total_viso_fakes": int(len(m)),
    "cohort_counts": cohort_counts.to_dict(),
    "p8a_caught_fraction": float(m["caught_P8A"].mean()),
    "e2b_caught_fraction": float(m["caught_E2B"].mean()),
    "e3_caught_fraction":  float(m["caught_E3"].mean()),
    "B_cohort_size": int((m["cohort"] == "B_caught_P8A_E3_not_E2B").sum()),
    "C_cohort_size": int((m["cohort"] == "C_caught_P8A_only").sum()),
    "B_cohort_subtype_split": m.loc[m["cohort"] == "B_caught_P8A_E3_not_E2B", "subtype"].value_counts().to_dict(),
    "C_cohort_subtype_split": m.loc[m["cohort"] == "C_caught_P8A_only", "subtype"].value_counts().to_dict(),
    "interpretation_note": (
        "B_cohort = caught by both CLIP-init-FT (P8A) and L14-scratch (E3), but missed by B16-scratch (E2B). "
        "If B is large and has a coherent IQ pattern, that pattern is the lesson a from-scratch B16 needs to learn. "
        "C_cohort = caught by P8A alone — uniquely attributable to the 5-stage FT chain or CLIP-init."
    ),
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary:\n{json.dumps(summary, indent=2)}")
print(f"\nDone. Outputs in {OUT}")
