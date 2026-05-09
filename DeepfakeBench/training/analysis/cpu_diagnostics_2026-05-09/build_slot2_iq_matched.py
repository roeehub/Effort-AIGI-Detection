"""
Slot 2 — IQ-matched reals.

Subsample reals (FF++ DF40 paired + teams_real combined) so that their
aggregate `lap_var` distribution matches the active training fake
distribution. Stratification uses FAKE-distribution quartile edges; within
each quartile, we sample reals at the same proportion as fakes' contribution.

Outputs:
  - dataset/df40_pairs/df40-pair-matching__iq_matched_2026-05-09.json
  - analysis/cpu_diagnostics_2026-05-09/slot2_teams_real_keep_list_2026-05-09.csv
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_slot1_teams_keep_list import load_active_fake_lap_var  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("slot2")

RANDOM_STATE = 9102
ACTIVE_METHODS = (
    "simswap", "facedancer", "blendface", "e4s",
    "inswap", "mobileswap", "uniface",
)


def main() -> int:
    pair_json_path = ROOT / "dataset/df40_pairs/df40-pair-matching.json"
    out_pair_json = ROOT / "dataset/df40_pairs/df40-pair-matching__iq_matched_2026-05-09.json"
    out_keep_csv = ROOT / "analysis/cpu_diagnostics_2026-05-09/slot2_teams_real_keep_list_2026-05-09.csv"

    ff_df = pd.read_parquet(ROOT / "analysis/cpu_diagnostics_2026-05-09/ff_per_identity_lap_var_2026-05-09.parquet")
    teams_df = pd.read_parquet(ROOT / "analysis/cpu_diagnostics_2026-05-09/_cache/teams_real_uri_lap_var.parquet")
    teams_df = teams_df.dropna(subset=["lap_var"]).reset_index(drop=True)
    logger.info("FF++ identities: %d, Teams real frames: %d", len(ff_df), len(teams_df))

    fake_lv = load_active_fake_lap_var()
    if len(fake_lv) == 0:
        raise RuntimeError("Active fake atlas pools missing.")
    logger.info(
        "Active fakes: N=%d, lap_var p10=%.2f / p50=%.2f / p90=%.2f",
        len(fake_lv),
        np.percentile(fake_lv, 10), np.percentile(fake_lv, 50), np.percentile(fake_lv, 90),
    )

    # Pool the reals into a single dataframe with kind label.
    ff_pool = ff_df[["identity", "lap_var", "frame_uri"]].copy()
    ff_pool["kind"] = "ff_identity"
    teams_pool = teams_df[["frame_uri", "lap_var"]].copy()
    teams_pool["identity"] = ""
    teams_pool["kind"] = "teams_frame"
    real_pool = pd.concat([ff_pool, teams_pool], ignore_index=True)

    # Use 20 fake-distribution bins (vingtiles) for finer matching than quartiles.
    # Each bin covers 5% of the fake mass; we sample reals at the same per-bin
    # proportion. The min-bin count over reals binds the global scale.
    n_bins = 20
    q_edges = np.linspace(0.0, 1.0, n_bins + 1)
    raw_edges = [float(np.quantile(fake_lv, q)) for q in q_edges]
    # Replace endpoints with -inf/+inf so all reals are captured.
    edges = np.array([-np.inf] + raw_edges[1:-1] + [np.inf])
    logger.info("Fake %d-bin edges (%d total): %s", n_bins, len(edges), edges.round(2).tolist())

    real_pool["bucket"] = pd.cut(
        real_pool["lap_var"],
        bins=edges,
        labels=list(range(n_bins)),
        include_lowest=True,
    ).astype(int)
    bucket_counts = real_pool.groupby("bucket").size().to_dict()
    logger.info("Real-pool counts per bucket (before sampling): %s", bucket_counts)

    # Each fake bin holds ~25% / n_bins of fakes; uniform-bin matching means
    # equal reals per bucket. The smallest real-bucket binds the scale.
    nonzero = [c for c in bucket_counts.values() if c > 0]
    if not nonzero:
        raise RuntimeError("Real pool empty after binning.")
    target_per_bucket = min(nonzero)
    logger.info("Target count per bucket (= min nonzero): %d", target_per_bucket)

    rng = np.random.default_rng(RANDOM_STATE)
    sampled_chunks = []
    for b in range(n_bins):
        sub = real_pool[real_pool["bucket"] == b]
        if len(sub) == 0:
            continue
        n = min(len(sub), target_per_bucket)
        idx = rng.choice(len(sub), size=n, replace=False)
        sampled_chunks.append(sub.iloc[idx])
    sampled = pd.concat(sampled_chunks, ignore_index=True)

    sampled_lv = sampled["lap_var"].to_numpy()
    from scipy.stats import ks_2samp
    ks = ks_2samp(sampled_lv, fake_lv).statistic
    logger.info(
        "Sampled %d reals; lap_var p10=%.2f p50=%.2f p90=%.2f; KS vs fakes=%.4f",
        len(sampled),
        np.percentile(sampled_lv, 10),
        np.percentile(sampled_lv, 50),
        np.percentile(sampled_lv, 90),
        ks,
    )

    # Split sampled into FF++ identities (whitelist) and teams_real frames.
    ff_kept_identities = set(sampled[sampled["kind"] == "ff_identity"]["identity"].tolist())
    teams_kept_uris = set(sampled[sampled["kind"] == "teams_frame"]["frame_uri"].tolist())
    logger.info("FF++ identities kept: %d / %d", len(ff_kept_identities), len(ff_df))
    logger.info("Teams real frames kept: %d / %d", len(teams_kept_uris), len(teams_df))

    # Build filtered pair JSON: drop pairs whose real-identity is not in kept.
    with open(pair_json_path, "r") as f:
        data = json.load(f)
    new_pairs = []
    method_counts: Dict[str, int] = {}
    for pair in data["pairs"]:
        if pair["method"] not in ACTIVE_METHODS:
            new_pairs.append(pair)
            continue
        real = pair["real"]
        ident = f"df40_{real['source']}_{real['identity']}"
        if ident in ff_kept_identities:
            new_pairs.append(pair)
            method_counts[pair["method"]] = method_counts.get(pair["method"], 0) + 1
    new_data = dict(data)
    new_data["pairs"] = new_pairs
    new_summary = dict(data.get("summary", {}))
    new_summary["pairs_per_method__after_iq_matched_2026-05-09"] = method_counts
    new_summary["total_pairs__after_iq_matched_2026-05-09"] = len(new_pairs)
    new_data["summary"] = new_summary
    new_data.setdefault("filters", []).append({
        "filter": "iq_matched_2026-05-09",
        "random_state": RANDOM_STATE,
        "ff_identity_count_in": int(len(ff_df)),
        "ff_identity_count_kept": int(len(ff_kept_identities)),
        "active_methods": list(ACTIVE_METHODS),
    })
    out_pair_json.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic JSON: sort_keys for filter records would corrupt pair order;
    # we keep original pair order, but write with consistent indent.
    with open(out_pair_json, "w") as f:
        json.dump(new_data, f, indent=2)
        f.write("\n")
    logger.info("Wrote %s (%d pairs)", out_pair_json, len(new_pairs))

    # Build teams keep-list (sorted, deterministic).
    teams_keep_df = teams_df[teams_df["frame_uri"].isin(teams_kept_uris)][["frame_uri", "lap_var"]]
    teams_keep_df = teams_keep_df.sort_values(["frame_uri"]).reset_index(drop=True)
    out_keep_csv.parent.mkdir(parents=True, exist_ok=True)
    teams_keep_df.to_csv(out_keep_csv, index=False, lineterminator="\n", float_format="%.6f")
    logger.info("Wrote %s (%d rows)", out_keep_csv, len(teams_keep_df))

    print("====== SLOT 2 SUMMARY ======")
    print(f"input_real_frames (ff_id + teams):  {len(ff_df) + len(teams_df)}")
    print(f"output_real_frames:                 {len(sampled)}")
    print(f"  ff_identities_kept:               {len(ff_kept_identities)}")
    print(f"  teams_frames_kept:                {len(teams_kept_uris)}")
    print(f"KS(real_matched, fake):             {ks:.4f}")
    print(f"output_pair_json:                   {out_pair_json}")
    print(f"output_teams_keep_csv:              {out_keep_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
