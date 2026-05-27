#!/usr/bin/env python3
"""Build the viso bucket overview page + supporting data files.

Aggregates existing data into a single viewer-friendly digest:
  - bucket_metadata.json              (suite contents, n, methods)
  - recall_through_chain.csv          (viso recall × ckpt × FPR floor)
  - per_packet_verdicts.csv           (one row per packet w/ verdict + viso headline)
  - INDEX.html                        (single-page report; viewer serves via static-report)

This is data plumbing, not analysis. The interpretive layer comes from the three
CPU diagnostic streams (see ../{p8a_param_audit,clip_vs_p8a_viso,skin_frac_viso_gap}_2026-05-03/).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
TRAINING_DIR = HERE.parent.parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True, parents=True)

CHAIN_CSV = TRAINING_DIR / "analysis/s1_s2_2026-05-02_planning/probes/outputs/05_chain_joint_summary.csv"
P8A_VISO_FRAMES = TRAINING_DIR / "analysis/score_distribution_2026-05-02/raw_reports/visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv"

# ────────────────────────────────────────────────────────────────────────
# 1. Bucket metadata (from FACTS §9 + per-frame CSV)
# ────────────────────────────────────────────────────────────────────────
def build_bucket_metadata() -> dict:
    methods = defaultdict(int)
    families = defaultdict(int)
    n_total = 0
    bucket_uri = None
    with P8A_VISO_FRAMES.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n_total += 1
            methods[row["method"]] += 1
            families[row["family_key"]] += 1
            if bucket_uri is None and row.get("frame_path", "").startswith("gs://"):
                bucket_uri = row["frame_path"].rsplit("/", 1)[0]

    meta = {
        "suite_name": "visomaster_enhanced_macro_dev",
        "n_frames": n_total,
        "label": "fake (all)",
        "methods": dict(methods),
        "families": dict(families),
        "gcs_bucket_dir": bucket_uri,
        "manifest_source": "arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json",
        "training_source_for_viso": [
            "live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_*",
            "(visomaster_teams_enhanced is the conjunction match — disabled in P8A; enabled in P16/xan4dfto/R13_P14_DATA_FIX)",
        ],
        "swap_methods_in_viso": [
            "CSCS", "GhostFace-v1", "GhostFace-v2", "GhostFace-v3",
            "InStyleSwapper256-A", "InStyleSwapper256-B", "InStyleSwapper256-C",
            "Inswapper128", "SimSwap512",
        ],
        "notes_2026_05_03": (
            "All 550 frames are recaptures-through-Teams of GAN-enhanced viso fakes. "
            "The training data has the enhancer-pass viso WITHOUT the Teams transport — "
            "this is the bucket gap (~2× delta-from-1 vs deeplive). See PSERIES_FACTS §1.3, §6.9.1."
        ),
    }
    return meta


# ────────────────────────────────────────────────────────────────────────
# 2. Recall through the R13 chain (viso column from joint table)
# ────────────────────────────────────────────────────────────────────────
def build_recall_through_chain() -> list:
    rows: list = []
    with CHAIN_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            ckpt = r["ckpt"]
            floor = float(r["floor"])
            rows.append({
                "ckpt": ckpt,
                "joint_fpr_floor": floor,
                "viso_recall": float(r["visomaster_enhanced_macro_dev_recall"]),
                "deeplive_recall": float(r["deeplive_enhanced_dev_recall"]),
                "teams_fake_dev_recall": float(r["teams_fake_all_dev_recall"]),
                "teams_fake_lockbox_recall": float(r["teams_fake_all_lockbox_recall"]),
                "selected_tau": float(r["tau"]),
                "joint_compliant": r["joint_compliant"] == "True",
            })
    return rows


# ────────────────────────────────────────────────────────────────────────
# 3. Per-packet verdicts (manually curated from FACTS doc + memory)
# ────────────────────────────────────────────────────────────────────────
def build_per_packet_verdicts() -> list:
    return [
        {
            "packet": "P8A_REFERENCE_STEP5000",
            "lever": "MLP-SVD + visual.proj + ln_post unfrozen (FT from RLP7_02)",
            "viso_recall_fpr10": 0.270,
            "viso_recall_fpr20": 0.576,
            "verdict": "BASELINE",
            "memory_ref": "project_p8a_breakthrough.md",
        },
        {
            "packet": "P14_FT_FROM_P8A (anti-shortcut bundle)",
            "lever": "anchor_aware + pipeline_random + face_scale_jitter (stacked)",
            "viso_recall_fpr10": 0.167,
            "viso_recall_fpr20": None,
            "verdict": "FAILED — bundle lost vs single-lever",
            "memory_ref": "PSERIES_FACTS §2.2",
        },
        {
            "packet": "P14_FACE_SCALE_JITTER_ISOLATED (mclioexb)",
            "lever": "face_scale_jitter scale_limit=0.50 ONLY",
            "viso_recall_fpr10": 0.013,
            "viso_recall_fpr20": None,
            "verdict": "FAILED contract — trainer composite 0.661 didn't translate",
            "memory_ref": "project_mclioexb_does_not_promote_2026-04-30.md",
        },
        {
            "packet": "P14_DATA_FIX (xan4dfto)",
            "lever": "visomaster_teams_enhanced fw=8.0 + bundle",
            "viso_recall_fpr10": None,
            "viso_recall_fpr20": None,
            "verdict": "COLLAPSED — value_composite 0.126",
            "memory_ref": "project_data_axis_lever_pulled_twice_no_lift.md",
        },
        {
            "packet": "P15_GRL_FROM_P8A",
            "lever": "static GRL λ=0.20 quality-domain head",
            "viso_recall_fpr10": None,
            "viso_recall_fpr20": None,
            "verdict": "FAILED — value_composite 0.516",
            "memory_ref": "PSERIES_FACTS §3.1",
        },
        {
            "packet": "P16_DATA_AXIS (rmic6wrc)",
            "lever": "visomaster_teams_enhanced fw=2.0 (no bundle)",
            "viso_recall_fpr10": None,
            "viso_recall_fpr20": None,
            "verdict": "FAILED — viso 0.7% at deployment τ",
            "memory_ref": "project_p16_data_axis_does_not_promote_2026-04-30.md",
        },
        {
            "packet": "P17_LAYER3_HEAD",
            "lever": "Trained head on intermediate layer (ArcFace + LINEAR variants)",
            "viso_recall_fpr10": None,
            "viso_recall_fpr20": None,
            "verdict": "FAILED — destroyed substrate-invariance",
            "memory_ref": "project_p17_trained_head_destroys_substrate_invariance.md",
        },
        {
            "packet": "P18T_GRL_TREATMENT (12-class method-conditional GRL)",
            "lever": "Method-domain GRL with ramped λ→0.95",
            "viso_recall_fpr10": None,
            "viso_recall_fpr20": None,
            "verdict": "FAILED — defensive against FT regression, NOT additive",
            "memory_ref": "project_p18_diagnostics_complete_2026-05-02.md",
        },
        {
            "packet": "P22_AUG_CURRICULUM (dot1buye, step1k)",
            "lever": "Pipeline-random aug curriculum + ArcFace s annealing",
            "viso_recall_fpr10": 0.249,
            "viso_recall_fpr20": 0.387,
            "verdict": "MIXED — deeplive solved (63%) but viso unchanged",
            "memory_ref": "project_p22_cpu_followups_reframe_2026-05-02.md",
        },
        {
            "packet": "P22_AUG_CURRICULUM (step8k)",
            "lever": "Same as above, longer training",
            "viso_recall_fpr10": 0.135,
            "viso_recall_fpr20": 0.220,
            "verdict": "DEGRADED — score variance collapsed 140×; deeplive 92% but viso regressed",
            "memory_ref": "project_p22_cpu_followups_reframe_2026-05-02.md",
        },
        {
            "packet": "S1_REDUX_SHORT (padjfsoq)",
            "lever": "Training cap 1000 steps, same s_end",
            "viso_recall_fpr10": 0.180,
            "viso_recall_fpr20": 0.453,
            "verdict": "FAILED — 0/3 falsifiers",
            "memory_ref": "project_s1_s2_s3_2026-05-03.md",
        },
        {
            "packet": "S2_EARLIER_BASE (evm4y66r, step600)",
            "lever": "FT from P8A_step2500 (less-saturated base) + curriculum",
            "viso_recall_fpr10": 0.118,
            "viso_recall_fpr20": 0.360,
            "verdict": "MIXED — viso FAIL but lockbox transfer 91.5% at FPR=10% (chain best)",
            "memory_ref": "project_s1_s2_s3_2026-05-03.md",
        },
        {
            "packet": "S3_VISO_WEIGHT (1sakmkv4)",
            "lever": "S2 + viso family weight 4.0→8.0",
            "viso_recall_fpr10": 0.138,
            "viso_recall_fpr20": 0.318,
            "verdict": "FAILED — class_sep peak 4.97 didn't translate",
            "memory_ref": "project_s1_s2_s3_2026-05-03.md",
        },
    ]


# ────────────────────────────────────────────────────────────────────────
# 4. Write all artifacts
# ────────────────────────────────────────────────────────────────────────
def main() -> None:
    meta = build_bucket_metadata()
    chain = build_recall_through_chain()
    verdicts = build_per_packet_verdicts()

    (OUT / "bucket_metadata.json").write_text(json.dumps(meta, indent=2))

    with (OUT / "recall_through_chain.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(chain[0].keys()))
        writer.writeheader()
        writer.writerows(chain)

    with (OUT / "per_packet_verdicts.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(verdicts[0].keys()))
        writer.writeheader()
        writer.writerows(verdicts)

    print(f"Wrote bucket_metadata.json: {meta['n_frames']} frames, {len(meta['methods'])} methods, {len(meta['families'])} families")
    print(f"Wrote recall_through_chain.csv: {len(chain)} rows ({len(set(r['ckpt'] for r in chain))} ckpts × {len(set(r['joint_fpr_floor'] for r in chain))} floors)")
    print(f"Wrote per_packet_verdicts.csv: {len(verdicts)} packets")


if __name__ == "__main__":
    main()
