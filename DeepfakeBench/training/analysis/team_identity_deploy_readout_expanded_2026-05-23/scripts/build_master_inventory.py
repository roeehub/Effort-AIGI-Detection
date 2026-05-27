"""Build the master inventory for the expanded team-identity deploy readout.

Output: outputs/master_inventory.csv with one row per frame:
    base_identity, suite, bucket, frame_path, label, human, device,
    deploy_relevant, role, score_P8A_cached, score_E2B_cached, score_T5C_cached,
    needs_fresh_score, sampled

Sampling: REAL_CAP=150, FAKE_CAP=100 per base_identity (random_state=42 for
reproducibility). Cohorts smaller than the cap are used in full.

The set of base_identities of interest is the canonical 5 humans (Roee on
Windows, dor, Noyn, Xiang, Xinhe) plus Mac-Roee informational cohorts plus
all fake-attack cohorts whose swap target is one of the 5 humans.

This is the input feed for the scoring scripts below. Re-run idempotently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv"
OUTPUT_DIR = REPO_ROOT / "analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs"
OUT_CSV = OUTPUT_DIR / "master_inventory.csv"

REAL_CAP = 150
FAKE_CAP = 100
SEED = 42

# (base_identity, human, device, deploy_relevant, role)
# role = "real" means a real-frame cohort
# role = "fake_target_<human>" means a fake-attack cohort with this swap target
REAL_COHORTS = [
    # Roee Windows (deploy-relevant) — verified "tester tester" name signature in frame paths
    ("roee_tester_real_2026-03-24", "Roee_Windows", "Windows", True),
    ("tester_roee_real_2026-03-06", "Roee_Windows", "Windows", True),
    ("team_may5__Roee", "Roee_Windows", "Windows", True),
    # Roee Mac (NOT deploy-relevant) — verified "Roy D"/Mac-bucket signatures
    # NOTE: prior agent placed `royd_real_2026-03-06` in Roee_Windows. REVISED to Roee_Mac
    # because the frame names start with "Roy D " (Mac label per memory), not "tester tester".
    # Only include cohorts with CACHED P8A/E2B/T5C scores — Mac-Roee is info-only, not deploy-relevant,
    # so we do NOT spend MPS time freshly scoring it. extra_roy_d and royd_real_2026-03-06 are
    # excluded for this reason (no cached scores).
    ("Roy_D", "Roee_Mac", "Mac", False),
    ("bla_bla_chow", "Roee_Mac", "Mac", False),
    ("bla_bla_chow__s1", "Roee_Mac", "Mac", False),
    ("bla_bla_chow__s2", "Roee_Mac", "Mac", False),
    # dor (deploy-relevant)
    ("dor_shkedi", "dor", "mixed", True),
    ("dor_shkedi__s16", "dor", "mixed", True),
    ("real_dor", "dor", "mixed", True),
    ("dor_morning", "dor", "local", True),
    ("dor_evening", "dor", "local", True),
    ("team_may5__Dor", "dor", "real-teams-dor-roee", True),
    # Noyn (= Noyn Sharker; deploy-relevant)
    ("Md_noyn_Sharker__s15", "Noyn", "teams-faces-data-test", True),
    ("team_may5__Noyn", "Noyn", "real-teams-dor-roee", True),
    # Xiang (deploy-relevant)
    ("Xiang_Xiang2_Feng", "Xiang", "teams-faces-data-test", True),
    ("Xiang_Xiang2_Feng__s23", "Xiang", "teams-faces-data-test", True),
    ("xiang", "Xiang", "teams-faces-data-test", True),
    ("extra_xiang", "Xiang", "local", True),
    ("team_may5__Xiang", "Xiang", "real-teams-dor-roee", True),
    # Xinhe (deploy-relevant)
    ("team_may5__Xinhe", "Xinhe", "real-teams-dor-roee", True),
    ("extra_xinghe", "Xinhe", "local", True),  # xinhe spelling variant
]


def map_fake_target(bi: str) -> str | None:
    """Determine which team-human is the swap target of a fake cohort."""
    if bi.startswith("live_prod__xinhe-fake-"):
        return "Xinhe"
    if bi.startswith("live_prod__xiang-fake-"):
        return "Xiang"
    if bi.startswith("dor_fake_"):
        return "dor"
    # Xiang_Xiang2_Feng (the fake half) is Xiang swap-target
    # extra_xiang (the fake half) is Xiang swap-target
    # extra_xinghe (the fake half) is Xinhe swap-target
    # dor_shkedi__s16 (the fake half) is dor swap-target
    if bi == "Xiang_Xiang2_Feng":
        return "Xiang"
    if bi == "extra_xiang":
        return "Xiang"
    if bi == "extra_xinghe":
        return "Xinhe"
    if bi == "dor_shkedi__s16":
        return "dor"
    return None


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST, low_memory=False)

    # Collect all base_identities of interest
    real_bis = [c[0] for c in REAL_COHORTS]
    fake_bis = sorted(set(
        bi for bi in df.base_identity.dropna().unique()
        if (
            bi.startswith("live_prod__xinhe-fake-")
            or bi.startswith("live_prod__xiang-fake-")
            or bi.startswith("dor_fake_")
        )
    ))
    all_set = sorted(set(real_bis + fake_bis))

    # Per-cohort sampling
    sampled_rows = []
    for bi in all_set:
        sub = df[df.base_identity == bi]
        for lbl in [0, 1]:
            sub_lbl = sub[sub.label == lbl]
            if len(sub_lbl) == 0:
                continue
            cap = FAKE_CAP if lbl == 1 else REAL_CAP
            if len(sub_lbl) > cap:
                sub_lbl = sub_lbl.sample(n=cap, random_state=SEED)
            sampled_rows.append(sub_lbl)
    s = pd.concat(sampled_rows, ignore_index=True)

    # Attribute human/device/deploy_relevant
    cohort_meta = {c[0]: c for c in REAL_COHORTS}

    def attribute(row):
        bi = row["base_identity"]
        lbl = row["label"]
        if bi in cohort_meta and lbl == 0:
            _, human, device, deploy = cohort_meta[bi]
            return pd.Series({
                "human": human, "device": device, "deploy_relevant": deploy,
                "role": "real",
            })
        # Fake half
        ft = map_fake_target(bi)
        if ft is not None and lbl == 1:
            return pd.Series({
                "human": ft, "device": "fake_target", "deploy_relevant": True,
                "role": f"fake_target_{ft}",
            })
        # real half of a mixed cohort (e.g., Xiang_Xiang2_Feng real)
        if bi in cohort_meta and lbl == 0:
            _, human, device, deploy = cohort_meta[bi]
            return pd.Series({
                "human": human, "device": device, "deploy_relevant": deploy,
                "role": "real",
            })
        # Unknown — shouldn't happen for the curated set
        return pd.Series({
            "human": "UNKNOWN", "device": "UNKNOWN", "deploy_relevant": False,
            "role": "UNKNOWN",
        })

    attrib = s.apply(attribute, axis=1)
    s = pd.concat([s, attrib], axis=1)

    # Sanity check: any UNKNOWN?
    n_unknown = (s.human == "UNKNOWN").sum()
    print(f"[warn] {n_unknown} frames could not be attributed", file=sys.stderr)

    # needs_fresh_score = score_P8A is NaN (none of the 4 cached cols populated)
    s["needs_fresh_score"] = s.score_P8A.isna()
    s["sampled"] = True

    # Select & order columns
    keep_cols = [
        "base_identity", "suite", "bucket", "frame_path", "label",
        "human", "device", "deploy_relevant", "role",
        "video_id", "score_P8A", "score_E2B", "score_T5C", "score_P2D",
        "needs_fresh_score", "sampled",
        "width", "height", "face_size", "quality", "gate_status",
        "face_area_ratio", "is_low_quality", "is_no_face",
    ]
    keep_cols = [c for c in keep_cols if c in s.columns]
    s = s[keep_cols]

    s.to_csv(OUT_CSV, index=False)
    print(f"[ok] wrote {len(s)} rows to {OUT_CSV}")

    # Print summary
    print("\n=== Per-human counts ===")
    grp = s.groupby(["human", "role"]).size().unstack(fill_value=0)
    print(grp)

    print("\n=== Per-cohort counts ===")
    grp = s.groupby(["human", "base_identity", "role"]).size()
    print(grp.to_string())

    print("\n=== Needs-fresh-score summary ===")
    print(s.groupby(["human", "needs_fresh_score"]).size().unstack(fill_value=0))

    return 0


if __name__ == "__main__":
    sys.exit(main())
