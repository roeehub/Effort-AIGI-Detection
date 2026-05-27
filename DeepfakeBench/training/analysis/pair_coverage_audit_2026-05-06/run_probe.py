"""PAIR_COVERAGE_AUDIT_2026-05-06 — pair-coverage of training data.

Purpose
-------
Quantify what fraction of the active training-data lanes (per
``experiments/phase2_round13/R13_VISO_CORR_PENALTY.yaml``, the current
production-candidate ship config) yields *paired* (real, fake) samples
that ``PE_PAIR_RANK_DRO`` could compute its rank loss on.

Pairing semantics (cross-checked against ``data/sources/combined_paired.py``):
  - For DF40 / DeepLive / VisoMaster / VisoMaster Enhanced /
    VisoMaster TeamsEnhanced / proper_data / Teams-passthrough lanes the
    iterator yields a *real frame* and a *fake frame* per ``frame_idx``
    drawn from the SAME ``sample_id`` (= same source video / capture).
    These are 1:1 matched pairs by ``(sample_id, frame_idx)``.
  - For ``external_training_reals`` (e.g. VCD, AVSpeech) the loader emits
    ``UnifiedUnpairedRealSample`` — REAL frames only, no fake counterpart.
    The yaml has ``external_real`` family weight 2.0 and ``realpool_real``
    weight 1.5 — a measurable share of every batch is real-only.
  - The ``visomaster_teams_enhanced`` track resolves only ~5.4% (54/999)
    of base samples to a Teams-v2 companion; the remaining ~94.6% fall
    back to the clean companion bucket. So the lane is paired (real+fake
    same source) but the *transport-conjunction* slice within it is
    narrow.

This script ENUMERATES from cached metadata only (no GPU, no GCS calls,
no model loading). Counts are reported in *base sample units* (one
``UnifiedPairedSample``); each base sample yields ``2 * frames_per_video``
frame-level samples per epoch (real + fake, matched by frame_idx).

Inputs
------
  - ``dataset/df40_pairs/df40-pair-matching.json``           (DF40 pairs)
  - ``analysis/deeplive_face_geometry_2026-05-05/summary.json`` (DeepLive
    strategy×label frame counts; sample = pair, so n_label0 ≈ n_label1)
  - ``analysis/p16_split_audit_2026-04-30/_cache/enhanced_visomaster_resolver_2026-04-06.json``
    (visomaster_teams_enhanced resolver: 999 base samples,
    teams_v2_companion vs clean_companion_only split)
  - ``analysis/inventory_audit_2026-05-04/inventory_summary.json``
    (proper-viso lane counts, NOT used by R13_VISO_CORR_PENALTY ship
    yaml — recorded as informational reference for the proper_data lane)
  - ``experiments/phase2_round13/R13_VISO_CORR_PENALTY.yaml``
    (current ship config; defines which lanes are enabled)

Outputs
-------
  outputs/coverage_by_method.csv
  outputs/coverage_by_enhancer.csv
  outputs/coverage_by_transport.csv
  outputs/coverage_by_identity.csv
  outputs/coverage_summary.json
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
)
OUT_DIR = REPO_ROOT / "analysis" / "pair_coverage_audit_2026-05-06" / "outputs"


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Lane-by-lane base-sample counts (one base sample = one
    # UnifiedPairedSample = one matched (real, fake) source video pair).
    #
    # Counts are conservative — we use cached/discovered evidence; we do
    # NOT hit GCS in this audit. Where a count is approximate it is
    # tagged ``approx=True``.
    # ----------------------------------------------------------------------
    lanes: Dict[str, dict] = {}

    # ===== DF40 lane =====
    # Each pair = (real video, fake video) = paired on every frame_idx.
    df40_path = REPO_ROOT / "dataset" / "df40_pairs" / "df40-pair-matching.json"
    with df40_path.open() as f:
        df40 = json.load(f)
    df40_per_method = df40["summary"]["pairs_per_method"]
    # The R13_VISO_CORR_PENALTY ship enables only a subset of DF40 methods.
    df40_enabled = {
        "simswap", "facedancer", "blendface", "e4s",
        "inswap", "mobileswap", "uniface",
    }
    df40_active_pairs = sum(
        n for m, n in df40_per_method.items() if m in df40_enabled
    )
    df40_active_identities = set()
    for p in df40["pairs"]:
        if p["method"] in df40_enabled:
            df40_active_identities.add(p["target_identity"])

    lanes["df40"] = {
        "n_paired_base_samples": df40_active_pairs,
        "n_unpaired_base_samples": 0,
        "n_unique_identities": len(df40_active_identities),
        "method_family": "df40",
        "enhancer_family": "none",  # df40 has no GFPGAN-style enhancers
        "transport": "raw_clean",
        "fake_per_base": 1,
        "real_per_base": 1,
        "pair_keys": ["pair_id", "frame_idx"],
        "iterator": "_iterate_df40_sample",
        "method_breakdown": {
            m: n for m, n in df40_per_method.items() if m in df40_enabled
        },
    }

    # ===== DeepLive lane =====
    # The face_area summary covers all 5 enabled strategies. n_label=0
    # frames = n_label=1 frames (within +/-1 — deeplive is paired).
    dl_summary_path = (
        REPO_ROOT
        / "analysis"
        / "deeplive_face_geometry_2026-05-05"
        / "summary.json"
    )
    with dl_summary_path.open() as f:
        dl_summary = json.load(f)
    dl_strategies_enabled = {
        "edge_cases", "minimal_processing", "quality_enhancement",
        "edge_cases_enhanced", "minimal_processing_enhanced",
    }
    dl_method_breakdown: Dict[str, int] = {}
    dl_total_frames_real = 0
    dl_total_frames_fake = 0
    # frames_per_video=8 per the yaml — but the underlying sample is one
    # video with ~16 frames (anchor_indices 8 of them used). We count
    # real frames per strategy as a *proxy* for base-sample count, then
    # divide by frames_per_video=8 to recover videos.
    frames_per_video_dl = 8
    for strat in dl_strategies_enabled:
        n0 = dl_summary["by_strategy_label"][f"{strat}|label=0"]["n_frames"]
        n1 = dl_summary["by_strategy_label"][f"{strat}|label=1"]["n_frames"]
        dl_total_frames_real += n0
        dl_total_frames_fake += n1
        # base-sample = video count; n0/anchor_count
        n_videos = n0 // frames_per_video_dl
        dl_method_breakdown[f"deeplive_{strat}"] = n_videos
    # Each deeplive sample has 1 real + 1 fake video, paired by frame_idx.
    dl_total_videos = sum(dl_method_breakdown.values())

    lanes["deeplive"] = {
        "n_paired_base_samples": dl_total_videos,
        "n_unpaired_base_samples": 0,
        "n_unique_identities": dl_total_videos,  # 1:1 ~ (memory note)
        "method_family": "deeplive",
        "enhancer_family": "split",  # 3 strategies clean + 2 enhanced
        "transport": "raw_clean",
        "fake_per_base": 1,
        "real_per_base": 1,
        "pair_keys": ["sample_id", "frame_idx"],
        "iterator": "_iterate_deeplive_sample",
        "method_breakdown": dl_method_breakdown,
        "n_frames_real": dl_total_frames_real,
        "n_frames_fake": dl_total_frames_fake,
        "approx_identity_count": True,
    }

    # ===== VisoMaster (V1 base, no enhancer, no Teams) =====
    # The R13_VISO_CORR_PENALTY ship enables 9 swap models. We don't have
    # the live discovery cache locally; the count is estimated from the
    # proper-viso (V1) lockbox+dev manifest as a proxy:
    #   proper_visomaster_clean videos = 342 (split: 262 dev + 80 lockbox)
    # This is the closest local proxy for the V1 base bucket. The
    # `data_inventory_identity_diversity` memory says "~430 viso" base
    # identities for training; cross-checks with the 342 figure (a
    # post-quality-filter subset). We use 342 here as a conservative
    # lower bound and note the memory upper bound.
    inv_path = (
        REPO_ROOT / "analysis" / "inventory_audit_2026-05-04" / "inventory_summary.json"
    )
    with inv_path.open() as f:
        inv_summary = json.load(f)
    proper_viso_clean_total = (
        inv_summary["proper_viso_breakdown_by_lane_split"]["proper_visomaster_clean__dev"]["videos"]
        + inv_summary["proper_viso_breakdown_by_lane_split"]["proper_visomaster_clean__lockbox"]["videos"]
    )
    # NOTE: the V1 base bucket (live-deepfake-methods-real-and-fake-frames-cropped)
    # for R13_VISO_CORR_PENALTY is BROADER than the proper-data dev/lockbox
    # subset above. The yaml enables 9 swap_models. Memory notes ~430 viso
    # identities. We record the conservative estimate at 342 and the
    # memory upper at 430.
    viso_v1_base_estimate = 342  # lower bound (proper-viso clean count)
    viso_v1_base_upper = 430     # memory: data_inventory_identity_diversity
    lanes["visomaster_v1_base"] = {
        "n_paired_base_samples": viso_v1_base_estimate,
        "n_paired_base_samples_upper": viso_v1_base_upper,
        "n_unpaired_base_samples": 0,
        "n_unique_identities": 229 + 63,  # proper-viso clean unique ids
        "method_family": "visomaster",
        "enhancer_family": "none",
        "transport": "raw_clean",
        "fake_per_base": 1,
        "real_per_base": 1,
        "pair_keys": ["sample_id", "frame_idx"],
        "iterator": "_iterate_visomaster_sample",
        "method_breakdown": {
            "visomaster_CSCS": "approx",
            "visomaster_GhostFace-v1": "approx",
            "visomaster_GhostFace-v2": "approx",
            "visomaster_GhostFace-v3": "approx",
            "visomaster_InStyleSwapper256-A": "approx",
            "visomaster_InStyleSwapper256-B": "approx",
            "visomaster_InStyleSwapper256-C": "approx",
            "visomaster_Inswapper128": "approx",
            "visomaster_SimSwap512": "approx",
        },
        "approx_identity_count": True,
    }

    # ===== VisoMaster Enhanced (V1 enhanced bucket, clean transport, enhancer pass) =====
    # Same source identities, post-hoc enhancer pass. 8 enhancers × ~430
    # base = potentially 3000+ enhanced samples; in practice the
    # `exclude_tiers: ARTIFACT` filter and bucket coverage trims this.
    # The proper-viso "enhanced_clean" lane has 1484 dev+lockbox videos
    # which we use as a reasonable proxy for the post-hoc enhanced
    # paired-sample count.
    proper_viso_enh_clean_total = (
        inv_summary["proper_viso_breakdown_by_lane_split"]["proper_visomaster_enhanced_clean__dev"]["videos"]
        + inv_summary["proper_viso_breakdown_by_lane_split"]["proper_visomaster_enhanced_clean__lockbox"]["videos"]
    )
    lanes["visomaster_enhanced"] = {
        "n_paired_base_samples": proper_viso_enh_clean_total,
        "n_unpaired_base_samples": 0,
        "n_unique_identities": 544 + 145,
        "method_family": "visomaster",
        "enhancer_family": "enhanced",  # 8 enhancers
        "transport": "raw_clean",
        "fake_per_base": 1,
        "real_per_base": 1,
        "pair_keys": ["sample_id", "frame_idx"],
        "iterator": "_iterate_visomaster_enhanced_sample",
        "method_breakdown": "see proper_visomaster_enhanced_clean by-method counts",
        "approx_identity_count": True,
    }

    # ===== VisoMaster Teams Enhanced (V1 enhanced × Teams-v2 transport) =====
    # The local resolver cache from 2026-04-06 quantifies this lane:
    #   - 999 base samples claimed Teams-v2 transport.
    #   - 54 RESOLVED to teams_v2_companion (true Teams-transported).
    #   - 943 fell back to clean_companion_only.
    #   - 2 missing companion (skipped at iteration time).
    #
    # The iterator picks a fake branch (original or one of N enhancers)
    # at iteration time per sample with p_original=0.5. Real side is the
    # same source video frame regardless. So every yielded item still
    # forms a (real, fake) pair — the only structural variation is the
    # *real* frame's transport: teams_v2 vs clean.
    resolver_path = (
        REPO_ROOT
        / "analysis"
        / "p16_split_audit_2026-04-30"
        / "_cache"
        / "enhanced_visomaster_resolver_2026-04-06.json"
    )
    with resolver_path.open() as f:
        resolver = json.load(f)
    resolver_summary = resolver["summary"]
    n_teams_v2 = resolver_summary["resolved_to_teams_v2_count"]
    n_clean_fallback = resolver_summary["resolved_to_clean_count"]
    n_missing = resolver_summary["missing_companion_count"]
    n_total_resolver = (
        resolver_summary["sample_count"] - n_missing
    )
    lanes["visomaster_teams_enhanced"] = {
        "n_paired_base_samples": n_total_resolver,
        "n_paired_teams_v2": n_teams_v2,
        "n_paired_clean_fallback": n_clean_fallback,
        "n_unpaired_base_samples": 0,
        "n_unique_identities": "approx 700+",  # spans the V1 enhanced identity space
        "method_family": "visomaster_enhanced",
        "enhancer_family": "enhanced + original branch",  # iterator picks 1 of (N+1) branches per epoch
        "transport_breakdown": {
            "teams_v2": n_teams_v2,
            "clean_fallback": n_clean_fallback,
        },
        "fake_per_base": "1 of N+1 branches per epoch",
        "real_per_base": 1,
        "pair_keys": ["sample_id", "frame_idx"],
        "iterator": "_iterate_visomaster_teams_enhanced_sample",
        "p_original": 0.5,
        "approx_identity_count": True,
    }

    # ===== Teams passthrough (deeplive identities, real+fake post-Teams) =====
    # The Teams passthrough bucket
    # ``live-deepfake-methods-real-and-fake-frames-cropped-teams`` provides
    # paired real+fake JPGs played through Teams. Discovery is GCS-side
    # so we don't have a local count cache. Memory says deeplive has
    # ~1900 base identities; not all are recaptured through Teams.
    # Conservative estimate based on ``deeplive_teams_fake`` family
    # weight 7.0 (most heavily weighted family in R13_VISO_CORR_PENALTY)
    # implies the lane is sizable. We mark this as "approx — GCS-side".
    lanes["deeplive_teams"] = {
        "n_paired_base_samples": 1300,  # rough estimate; needs GCS listing to confirm
        "n_unpaired_base_samples": 0,
        "n_unique_identities": "approx 1300",
        "method_family": "deeplive",
        "enhancer_family": "split (5 strategies)",
        "transport": "teams_v2",
        "fake_per_base": 1,
        "real_per_base": 1,
        "pair_keys": ["sample_id", "frame_idx"],
        "iterator": "_iterate_teams_sample",
        "approx_identity_count": True,
        "approx_base_count": True,
        "note": (
            "Discovery requires GCS listing; not available in this audit. "
            "Family weight 7.0 (highest in yaml) implies lane is large."
        ),
    }

    # ===== external_training_reals (UNPAIRED — REAL ONLY) =====
    # The yaml configures a single source: VCD reals from
    # ``effort-collected-data/real/VCD`` with
    # max_total_samples=1200, identity_train_fraction=0.40,
    # max_frames_per_identity=15. Each ``UnifiedUnpairedRealSample``
    # represents one identity. No fake counterpart. PAIR-RANK CANNOT
    # FIRE on these samples.
    lanes["external_vcd_real"] = {
        "n_paired_base_samples": 0,
        "n_unpaired_base_samples": 1200,  # max_total_samples cap
        "n_unique_identities": "approx 80",  # 0.40 * (~200 distinct identities)
        "method_family": "external_real",
        "enhancer_family": "n/a",
        "transport": "webcam_codec",
        "fake_per_base": 0,
        "real_per_base": 1,
        "pair_keys": "n/a — REAL-ONLY",
        "iterator": "_iterate_unpaired_real_sample",
        "approx_identity_count": True,
    }

    # ===== visomaster_hints + visomaster_hints_teams =====
    # DISABLED in R13_VISO_CORR_PENALTY (verified bad data per memory).
    lanes["visomaster_hints"] = {
        "n_paired_base_samples": 0,
        "n_unpaired_base_samples": 0,
        "enabled_in_ship_yaml": False,
        "note": "Disabled — bad data per project_visomaster_hints_lanes_bad_data.md",
    }

    # ===== proper_data lane (NOT in R13_VISO_CORR_PENALTY) =====
    # Recorded for reference only. R13_VISO_CORR_PENALTY does not enable
    # combined_paired.proper_data.* — the iterator path is dead in this
    # ship. P8A and earlier P-series did use this lane (4 lanes: viso
    # clean, viso teams, viso enhanced clean, viso enhanced teams).
    lanes["proper_data_INACTIVE_in_ship"] = {
        "n_paired_base_samples": 0,
        "n_unpaired_base_samples": 0,
        "enabled_in_ship_yaml": False,
        "note": (
            "proper_data lane not enabled in R13_VISO_CORR_PENALTY. "
            "If reactivated for future packets, would add ~1826 paired "
            "real captures × 6 lane families."
        ),
    }

    # ----------------------------------------------------------------------
    # Aggregate
    # ----------------------------------------------------------------------
    total_paired = 0
    total_unpaired = 0
    for k, v in lanes.items():
        if isinstance(v.get("n_paired_base_samples"), int):
            total_paired += v["n_paired_base_samples"]
        if isinstance(v.get("n_unpaired_base_samples"), int):
            total_unpaired += v["n_unpaired_base_samples"]

    paired_fraction_base = total_paired / (total_paired + total_unpaired)

    # Frame-level fraction. Each paired base sample yields ~2*F frames
    # per epoch (F=8 anchor frames, real+fake). Each unpaired real
    # sample yields up to ``frames_per_sample`` (default ~6) frames.
    # Default frames_per_sample is 6 in the iterator (see code review).
    F_paired = 8
    F_unpaired = 6
    paired_frames_per_epoch = total_paired * 2 * F_paired
    unpaired_frames_per_epoch = total_unpaired * F_unpaired
    paired_fraction_frame = paired_frames_per_epoch / (
        paired_frames_per_epoch + unpaired_frames_per_epoch
    )

    # Effective fraction in a batch — depends on family_weights.
    # R13_VISO_CORR_PENALTY family weights (real-side share):
    #   df40_real        0.5
    #   realpool_real    1.5
    #   external_real    2.0
    #   (plus the fake-side families which are all paired)
    # Identity-balanced sampler picks ONE base-sample per identity per
    # epoch (or per worker rotation). Every batch has roughly:
    #   2.0 weight on external (UNPAIRED real)
    #   non-external real shares (1.5 + 0.5 = 2.0) all PAIRED
    #   plus all the fake-side families (paired)
    # External real frames are roughly 2/(2 + 1.5 + 0.5 + 0.2 + 4 + 4 + 2.5 + 3 + 7 + 5)
    #   = 2.0 / 29.7 = 6.7% of batch by family-weight share.
    real_only_batch_share = 2.0 / (
        0.2 + 4.0 + 4.0 + 2.5 + 3.0 + 7.0 + 5.0 + 0.5 + 1.5 + 2.0
    )
    paired_batch_share = 1.0 - real_only_batch_share

    # ----------------------------------------------------------------------
    # CSV — by method
    # ----------------------------------------------------------------------
    rows_method = []
    # DF40 per-method
    for m, n in lanes["df40"]["method_breakdown"].items():
        rows_method.append({
            "lane": "df40",
            "method": f"df40_{m}",
            "n_paired_samples": n,
            "n_unpaired_samples": 0,
            "paired_fraction": 1.0,
            "transport": "raw_clean",
            "enhancer_family": "none",
        })
    # DeepLive per-strategy
    for m, n in lanes["deeplive"]["method_breakdown"].items():
        is_enhanced = m.endswith("_enhanced")
        rows_method.append({
            "lane": "deeplive",
            "method": m,
            "n_paired_samples": n,
            "n_unpaired_samples": 0,
            "paired_fraction": 1.0,
            "transport": "raw_clean",
            "enhancer_family": "enhanced" if is_enhanced else "clean",
        })
    rows_method.append({
        "lane": "visomaster_v1_base",
        "method": "visomaster_*9_swap_models_aggregated",
        "n_paired_samples": viso_v1_base_estimate,
        "n_unpaired_samples": 0,
        "paired_fraction": 1.0,
        "transport": "raw_clean",
        "enhancer_family": "none",
    })
    rows_method.append({
        "lane": "visomaster_enhanced",
        "method": "visomaster_enhanced_*8_enhancers_aggregated",
        "n_paired_samples": proper_viso_enh_clean_total,
        "n_unpaired_samples": 0,
        "paired_fraction": 1.0,
        "transport": "raw_clean",
        "enhancer_family": "enhanced",
    })
    rows_method.append({
        "lane": "visomaster_teams_enhanced",
        "method": "visomaster_teams_enhanced_aggregated_999",
        "n_paired_samples": n_total_resolver,
        "n_unpaired_samples": 0,
        "paired_fraction": 1.0,
        "transport": "mixed_(teams_v2_5pct_+_clean_95pct)",
        "enhancer_family": "enhanced+original_per_epoch_branch",
    })
    rows_method.append({
        "lane": "deeplive_teams",
        "method": "teams_passthrough_*5_strategies",
        "n_paired_samples": lanes["deeplive_teams"]["n_paired_base_samples"],
        "n_unpaired_samples": 0,
        "paired_fraction": 1.0,
        "transport": "teams_v2",
        "enhancer_family": "split",
    })
    rows_method.append({
        "lane": "external_vcd_real",
        "method": "external_vcd_real",
        "n_paired_samples": 0,
        "n_unpaired_samples": 1200,
        "paired_fraction": 0.0,
        "transport": "webcam_codec",
        "enhancer_family": "n/a",
    })

    _write_csv(
        OUT_DIR / "coverage_by_method.csv",
        rows_method,
        [
            "lane", "method", "n_paired_samples", "n_unpaired_samples",
            "paired_fraction", "transport", "enhancer_family",
        ],
    )

    # ----------------------------------------------------------------------
    # CSV — by enhancer family
    # ----------------------------------------------------------------------
    enhancer_agg: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"n_paired": 0, "n_unpaired": 0}
    )
    for r in rows_method:
        ef = r["enhancer_family"]
        enhancer_agg[ef]["n_paired"] += r["n_paired_samples"]
        enhancer_agg[ef]["n_unpaired"] += r["n_unpaired_samples"]

    rows_enh = []
    for ef, agg in enhancer_agg.items():
        denom = agg["n_paired"] + agg["n_unpaired"] or 1
        rows_enh.append({
            "enhancer_family": ef,
            "n_paired_samples": agg["n_paired"],
            "n_unpaired_samples": agg["n_unpaired"],
            "paired_fraction": agg["n_paired"] / denom,
        })
    _write_csv(
        OUT_DIR / "coverage_by_enhancer.csv",
        rows_enh,
        ["enhancer_family", "n_paired_samples", "n_unpaired_samples", "paired_fraction"],
    )

    # ----------------------------------------------------------------------
    # CSV — by transport
    # ----------------------------------------------------------------------
    trans_agg: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"n_paired": 0, "n_unpaired": 0}
    )
    for r in rows_method:
        t = r["transport"]
        trans_agg[t]["n_paired"] += r["n_paired_samples"]
        trans_agg[t]["n_unpaired"] += r["n_unpaired_samples"]
    # Decompose viso teams_enhanced mixed transport
    trans_agg["teams_v2"]["n_paired"] += n_teams_v2
    trans_agg["raw_clean"]["n_paired"] += n_clean_fallback
    trans_agg.pop("mixed_(teams_v2_5pct_+_clean_95pct)", None)

    rows_trans = []
    for t, agg in trans_agg.items():
        denom = agg["n_paired"] + agg["n_unpaired"] or 1
        rows_trans.append({
            "transport": t,
            "n_paired_samples": agg["n_paired"],
            "n_unpaired_samples": agg["n_unpaired"],
            "paired_fraction": agg["n_paired"] / denom,
        })
    _write_csv(
        OUT_DIR / "coverage_by_transport.csv",
        rows_trans,
        ["transport", "n_paired_samples", "n_unpaired_samples", "paired_fraction"],
    )

    # ----------------------------------------------------------------------
    # CSV — by identity (lane-level)
    # ----------------------------------------------------------------------
    rows_id = [
        {
            "lane": "df40",
            "n_unique_identities_with_paired_data": len(df40_active_identities),
            "n_unique_identities_real_only": 0,
            "identity_prefix": "df40_",
        },
        {
            "lane": "deeplive",
            "n_unique_identities_with_paired_data": dl_total_videos,
            "n_unique_identities_real_only": 0,
            "identity_prefix": "realpool_",
            "note": "1:1 video-to-identity (approx)",
        },
        {
            "lane": "visomaster_v1_base",
            "n_unique_identities_with_paired_data": 229 + 63,
            "n_unique_identities_real_only": 0,
            "identity_prefix": "realpool_",
            "note": "shares identity space with deeplive (realpool_*)",
        },
        {
            "lane": "visomaster_enhanced",
            "n_unique_identities_with_paired_data": 544 + 145,
            "n_unique_identities_real_only": 0,
            "identity_prefix": "realpool_",
            "note": "shares identity space with viso_v1_base + deeplive",
        },
        {
            "lane": "visomaster_teams_enhanced",
            "n_unique_identities_with_paired_data": 999,
            "n_unique_identities_real_only": 0,
            "identity_prefix": "realpool_",
            "note": "shares identity space with above; ~5% have teams_v2 real companion",
        },
        {
            "lane": "deeplive_teams",
            "n_unique_identities_with_paired_data": 1300,  # approx
            "n_unique_identities_real_only": 0,
            "identity_prefix": "realpool_",
            "note": "approx; needs GCS listing to confirm",
        },
        {
            "lane": "external_vcd_real",
            "n_unique_identities_with_paired_data": 0,
            "n_unique_identities_real_only": 80,
            "identity_prefix": "external_vcd_",
            "note": "REAL ONLY — pair-rank does not fire",
        },
    ]
    _write_csv(
        OUT_DIR / "coverage_by_identity.csv",
        rows_id,
        [
            "lane", "n_unique_identities_with_paired_data",
            "n_unique_identities_real_only", "identity_prefix", "note",
        ],
    )

    # ----------------------------------------------------------------------
    # Summary JSON
    # ----------------------------------------------------------------------
    summary = {
        "audit_date": "2026-05-06",
        "audit_name": "PAIR_COVERAGE_AUDIT_2026-05-06",
        "ship_yaml": "experiments/phase2_round13/R13_VISO_CORR_PENALTY.yaml",
        "lanes": lanes,
        "totals": {
            "n_paired_base_samples": total_paired,
            "n_unpaired_base_samples": total_unpaired,
            "paired_fraction_base_unit": round(paired_fraction_base, 4),
            "paired_frames_per_epoch_estimate": paired_frames_per_epoch,
            "unpaired_frames_per_epoch_estimate": unpaired_frames_per_epoch,
            "paired_fraction_frame_unit": round(paired_fraction_frame, 4),
            "real_only_batch_share_by_family_weight": round(real_only_batch_share, 4),
            "paired_batch_share_by_family_weight": round(paired_batch_share, 4),
        },
        "key_gaps": {
            "external_vcd_real": (
                "1200 unpaired real samples (~80 identities) — "
                "pair-rank cannot fire; multi-axis GroupDRO real-side "
                "should still group by source/transport/quality_band."
            ),
            "visomaster_teams_enhanced_transport_conjunction": (
                f"Only {n_teams_v2}/{n_total_resolver} ({n_teams_v2 * 100 / n_total_resolver:.1f}%) "
                "of viso teams_enhanced base samples have a true Teams-v2 real "
                "companion; the rest fall back to clean transport. The "
                "*conjunction* slice (viso enhanced × teams transport) is narrow."
            ),
            "deeplive_teams_count_unverified": (
                "Discovery requires GCS listing; not in audit. Family weight 7.0 "
                "implies lane is large but exact count needs separate GCS probe."
            ),
        },
        "conclusions": {
            "paired_fraction_headline": (
                f"~{paired_fraction_base * 100:.1f}% of training BASE samples "
                f"are paired (real+fake same source). "
                f"~{paired_batch_share * 100:.1f}% of effective BATCH content "
                f"is paired by family-weight share."
            ),
            "lanes_pair_rank_can_apply": [
                "df40", "deeplive", "visomaster_v1_base", "visomaster_enhanced",
                "visomaster_teams_enhanced", "deeplive_teams",
            ],
            "lanes_pair_rank_cannot_apply": ["external_vcd_real"],
            "recommendation": (
                "Apply pair-rank UNIFORMLY to all paired lanes. The unpaired "
                "external_real lane already requires special handling (no "
                "pair-rank loss term — only CE) since it has no fake counterpart. "
                "Restricting pair-rank to a subset of paired lanes is NOT "
                "recommended: (a) all 6 paired lanes share the same "
                "(real, fake same-source) structure that pair-rank is designed "
                "for; (b) Teams passthrough lane (deeplive_teams) is high-value "
                "for the binding metric and should not be excluded."
            ),
        },
        "pairing_semantics_note": "see pairing_semantics_notes.md",
    }
    with (OUT_DIR / "coverage_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["totals"], indent=2))
    print()
    print("Lanes pair-rank applies to:",
          ", ".join(summary["conclusions"]["lanes_pair_rank_can_apply"]))
    print("Lanes pair-rank does NOT apply to:",
          ", ".join(summary["conclusions"]["lanes_pair_rank_cannot_apply"]))


if __name__ == "__main__":
    main()
