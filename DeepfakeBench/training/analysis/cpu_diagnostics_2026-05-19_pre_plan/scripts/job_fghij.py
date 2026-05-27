"""
Jobs F, G, H, I, J combined — independent quick diagnostics on local data only.

F: Slot β NEW failure modes in IQ space (frame-level)
G: anchor_aware rescue identities × bimodal partition cross-reference
H: Pocket spec verification (apply §4.2 spec to unified_tags)
I: Roy_D train-manifest grep (closes loop `roy-d-color-b-dev-mechanism`)
J: Slot β vs T5C disagreement structure
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402


def job_f() -> None:
    print("\n\n========================================")
    print("Job F: Slot β NEW failure modes in IQ space")
    print("========================================")
    df_lock = common.load_lockbox_frames("teams_real_all_lockbox")
    df_real = df_lock[df_lock["label"] == 0].copy()
    tau_b = 0.816038
    tau_t5c = 0.830874

    df_real["slot_b_over"] = df_real["prob_SLOT_B_6AXIS_GRL_STEP3500"] >= tau_b
    df_real["t5c_over"] = df_real["prob_T5C_PERIODIC_STEP3500"] >= tau_t5c

    # Categorize
    def cat(r):
        if r["slot_b_over"] and not r["t5c_over"]:
            return "slot_b_only"
        if r["t5c_over"] and not r["slot_b_over"]:
            return "t5c_only"
        if r["slot_b_over"] and r["t5c_over"]:
            return "both"
        return "neither"
    df_real["category"] = df_real.apply(cat, axis=1)
    print(f"\nCategory distribution:")
    print(df_real["category"].value_counts())

    # Slot β NEW failures = slot_b_only category
    new_failures = df_real[df_real["category"] == "slot_b_only"].copy()
    print(f"\nSlot β NEW failures (over-fire AND T5C does NOT): {len(new_failures)}")
    print(f"Their identity_key distribution (top 15):")
    print(new_failures["identity_key"].value_counts().head(15).to_string())

    # Bimodal partition lookup
    bim = common.load_bimodal_partition()
    new_with_bim = new_failures.merge(
        bim[["identity_key", "median_ratio", "coverage_class"]],
        on="identity_key", how="left",
    )
    print(f"\nNEW failures coverage class distribution:")
    cov_counts = new_with_bim["coverage_class"].value_counts(dropna=False)
    print(cov_counts.to_string())
    cov_counts.to_csv(common.OUT / "job_f_new_failures_by_coverage.csv")

    new_with_bim.to_csv(common.OUT / "job_f_slot_b_new_failures.csv", index=False)
    print(f"\nSaved: {common.OUT / 'job_f_slot_b_new_failures.csv'}")

    # Reverse: T5C-only failures (Slot β CORRECTLY classifies as real)
    t5c_only = df_real[df_real["category"] == "t5c_only"].copy()
    print(f"\nT5C-only failures (where Slot β correctly does NOT over-fire): {len(t5c_only)}")
    if len(t5c_only) > 0:
        print(f"Their identity_key (top 10): {t5c_only['identity_key'].value_counts().head(10).to_dict()}")


def job_g() -> None:
    print("\n\n========================================")
    print("Job G: anchor_aware rescue identities × bimodal partition")
    print("========================================")
    bim = common.load_bimodal_partition()

    # Per memory `project_band_shortcut_ood_hypothesis_2026-05-16`:
    # Slot A v2 (anchor_aware T5C-base) rescued:
    #  - Chikara_Takahashi: 26% → 0%
    #  - PC_Generator: 28% → 0%
    # Regressed:
    #  - Roy_D: 29% → 81%
    # New failure:
    #  - bla_bla_chow: 0% → 16%
    rescued = ["Chikara_Takahashi", "PC_Generator"]
    regressed = ["Roy_D"]
    new_failure = ["bla_bla_chow"]

    rows = []
    for identity_pattern, status in (
        [(p, "rescued") for p in rescued]
        + [(p, "regressed") for p in regressed]
        + [(p, "new_failure") for p in new_failure]
    ):
        matches = bim[bim["identity_key"].str.contains(identity_pattern, case=False, na=False)]
        for _, m in matches.iterrows():
            rows.append({
                "pattern": identity_pattern,
                "anchor_aware_status": status,
                "identity_key": m["identity_key"],
                "n": m["n"],
                "bucket": m["bucket"],
                "median_ratio": m["median_ratio"],
                "coverage_class": m["coverage_class"],
            })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(common.OUT / "job_g_anchor_aware_rescue_x_bimodal.csv", index=False)

    # Roy_D specifically
    print(f"\nRoy_D in bimodal partition: {bim[bim['identity_key'].str.contains('Roy_D', case=False)]}")
    print("(memory note: Roy_D absent from full_tags so unmeasurable in joint-marginal audit)")


def job_h() -> None:
    print("\n\n========================================")
    print("Job H: pocket spec verification")
    print("========================================")
    tags = common.load_unified_tags()
    print(f"Unified tags: {tags.shape}")

    # §4.2 spec: min_dim > 350 AND (color_a_dev < 4 OR color_a_dev > 12) AND skin_frac < 0.35
    spec_mask = (
        (tags["min_dim"] > 350)
        & ((tags["color_a_dev"] < 4) | (tags["color_a_dev"] > 12))
        & (tags["skin_frac"] < 0.35)
    )
    print(f"\nFrames matching §4.2 pocket spec: {spec_mask.sum()} / {len(tags)} ({spec_mask.mean():.2%})")

    # Per-bucket breakdown
    by_bucket = tags.groupby("bucket")["min_dim"].count().to_dict()
    pocket_by_bucket = tags[spec_mask].groupby("bucket")["min_dim"].count().to_dict()
    print(f"\nPocket frames per bucket:")
    for b in by_bucket:
        n_total = by_bucket[b]
        n_pocket = pocket_by_bucket.get(b, 0)
        print(f"  {b:20s}: {n_pocket:6d} / {n_total:6d} ({n_pocket/n_total:.2%})")

    # Per-identity (chronic) in pocket
    if "identity_key" in tags.columns:
        chronic_tags = tags[tags["identity_key"].notna()].copy()
        chronic_tags["in_pocket"] = spec_mask[chronic_tags.index]
        per_ident = chronic_tags.groupby("identity_key")["in_pocket"].agg(["count", "sum"]).rename(columns={"count": "n", "sum": "n_in_pocket"})
        per_ident["frac_in_pocket"] = per_ident["n_in_pocket"] / per_ident["n"]
        per_ident = per_ident.sort_values("frac_in_pocket", ascending=False)
        print(f"\nTop 15 identities by pocket fraction (n ≥ 20):")
        view = per_ident[per_ident["n"] >= 20].head(15)
        print(view.to_string())
        per_ident.to_csv(common.OUT / "job_h_per_identity_in_pocket.csv")

    # Also test variants of the spec
    print("\n=== Variant spec exploration ===")
    variants = [
        ("§4.2 as-written", spec_mask),
        ("min_dim > 350 alone", tags["min_dim"] > 350),
        ("color_a OR-tail alone", (tags["color_a_dev"] < 4) | (tags["color_a_dev"] > 12)),
        ("skin_frac < 0.35 alone", tags["skin_frac"] < 0.35),
        ("min_dim > 350 AND skin_frac < 0.35", (tags["min_dim"] > 350) & (tags["skin_frac"] < 0.35)),
        ("color_a > 12 only (high cast)", tags["color_a_dev"] > 12),
        ("color_a < 4 only (low cast)", tags["color_a_dev"] < 4),
    ]
    rows = []
    for name, m in variants:
        rows.append({"variant": name, "n_total": int(m.sum()), "frac": m.mean()})
        per_b = tags[m].groupby("bucket").size().to_dict()
        for b in ["train_real", "dev_real", "lockbox_real"]:
            rows[-1][f"in_{b}"] = per_b.get(b, 0)
    var_df = pd.DataFrame(rows)
    print(var_df.to_string(index=False))
    var_df.to_csv(common.OUT / "job_h_spec_variants.csv", index=False)


def job_i() -> None:
    print("\n\n========================================")
    print("Job I: Roy_D train-manifest grep")
    print("========================================")
    root = common.ROOT
    patterns = ["Roy_D", "roy_d", "RoyD", "roy.d"]
    print(f"Searching for Roy_D references in train-side manifests and yamls...")

    # 1) grep training yamls
    search_paths = [
        root / "experiments",
        root / "data" / "validation_sources.py",
        root / "data" / "sources",
        root / "arena" / "manifests",
        root / "arena" / "checkpoint_maps",
    ]

    results = []
    for p in search_paths:
        if not p.exists():
            continue
        for pat in ["Roy_D", "roy_d"]:
            try:
                proc = subprocess.run(
                    ["grep", "-r", "-l", pat, str(p)],
                    capture_output=True, text=True, timeout=30,
                )
                for f in proc.stdout.strip().split("\n"):
                    if f and not f.endswith(".pyc"):
                        results.append({"pattern": pat, "file": f})
            except Exception as e:
                print(f"grep error on {p}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        print(f"\nMatches:")
        print(df.to_string(index=False))
    else:
        print("\nNo matches in experiments/ or training-source paths.")
    df.to_csv(common.OUT / "job_i_roy_d_grep.csv", index=False)

    # 2) Check the unified_tags for Roy_D location
    tags = common.load_unified_tags()
    roy_mask = tags["identity_key"].astype(str).str.contains("Roy_D|roy_d", case=False, na=False, regex=True)
    print(f"\nRoy_D frames in unified_tags: {roy_mask.sum()}")
    if roy_mask.any():
        print(f"Roy_D by bucket: {tags[roy_mask].groupby('bucket').size().to_dict()}")

    # 3) Check chronic_full_tags
    chronic_tags = pd.read_parquet(common.ROOT / "analysis/joint_marginal_audit_2026-05-19/artifacts/chronic_full_tags.parquet")
    roy_chronic = chronic_tags[chronic_tags["identity_key"].astype(str).str.contains("Roy_D|roy_d", case=False, na=False, regex=True)]
    print(f"\nRoy_D frames in chronic_full_tags: {len(roy_chronic)}")
    if len(roy_chronic) > 0:
        print(f"Roy_D buckets: {roy_chronic.groupby('bucket').size().to_dict()}")

    # 4) Lockbox tagging parquet
    lb_full = common.load_lockbox_full_tags()
    if "identity_key" in lb_full.columns:
        roy_lb = lb_full[lb_full["identity_key"].astype(str).str.contains("Roy_D|roy_d", case=False, na=False, regex=True)]
        print(f"\nRoy_D frames in lockbox full_tags: {len(roy_lb)}")
        if len(roy_lb) > 0:
            print(f"Roy_D in lockbox: split={roy_lb['split'].value_counts().to_dict() if 'split' in roy_lb.columns else 'n/a'}")


def job_j() -> None:
    print("\n\n========================================")
    print("Job J: Slot β vs T5C disagreement structure")
    print("========================================")
    df_lock = common.load_lockbox_frames("teams_real_all_lockbox")
    df_real = df_lock[df_lock["label"] == 0].copy()
    print(f"Lockbox real frames: {len(df_real)}")

    pearson_real = df_real["prob_SLOT_B_6AXIS_GRL_STEP3500"].corr(df_real["prob_T5C_PERIODIC_STEP3500"])
    spearman_real = df_real[["prob_SLOT_B_6AXIS_GRL_STEP3500", "prob_T5C_PERIODIC_STEP3500"]].corr(method="spearman").iloc[0, 1]
    print(f"\nReal-frame correlation Slot β vs T5C:")
    print(f"  Pearson:  {pearson_real:.4f}")
    print(f"  Spearman: {spearman_real:.4f}")

    # Disagreement at each ckpt's τ
    df_real["slot_b_over"] = df_real["prob_SLOT_B_6AXIS_GRL_STEP3500"] >= 0.816038
    df_real["t5c_over"] = df_real["prob_T5C_PERIODIC_STEP3500"] >= 0.830874

    table = pd.crosstab(df_real["slot_b_over"], df_real["t5c_over"], margins=True, margins_name="Total")
    print(f"\nDisagreement table (Slot β rows × T5C cols, at each selected τ):")
    print(table.to_string())
    table.to_csv(common.OUT / "job_j_disagreement_table.csv")

    # Also pearson on fakes
    df_fake = df_lock[df_lock["label"] == 1].copy()
    if len(df_fake) > 0:
        pearson_fake = df_fake["prob_SLOT_B_6AXIS_GRL_STEP3500"].corr(df_fake["prob_T5C_PERIODIC_STEP3500"])
        print(f"\nFake-frame Pearson: {pearson_fake:.4f}")

    # All-pair correlations across 5 ckpts on real
    print("\n=== ALL-CKPT PEARSON ON LOCKBOX REALS ===")
    score_cols = [f"prob_{c[0]}" for c in common.CKPTS]
    corr = df_real[score_cols].corr().round(3)
    print(corr.to_string())
    corr.to_csv(common.OUT / "job_j_all_ckpt_corr_real.csv")


def main() -> int:
    log_f = common.LOGS / "job_fghij.log"
    sys.stdout = open(log_f, "w", buffering=1)
    sys.stderr = sys.stdout

    job_f()
    job_g()
    job_h()
    job_i()
    job_j()
    print("\n=== Jobs F,G,H,I,J complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
