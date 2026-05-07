"""
Build the 800-frame teams_chronic_diverse canary for in-training monitoring.

Composition (target = 800 rows):
  REALS (600):
    - Chronic-6 cohort (50 each = 300):
        chronic_PCGen_s22, chronic_PCGen_s45, chronic_Q_s6,
        chronic_bla_bla_chow, chronic_bla_bla_chow_s2, chronic_Roy_D
    - Healthy diverse reals (50 each = 250):
        healthy_test_cam, healthy_md_noyn_sharker,
        healthy_xiang_xiang2_feng, healthy_dor, healthy_dor_shkedi
    - HDTF clean reals (50): hdtf_clean_real
  FAKES (200):
    - lockbox_fake (100, video_id-stratified)
    - viso_fake (50)
    - deeplive_fake (50)

Sampling: video_id-stratified random with seed=42 where applicable.
Chronic-id matching: prefix-on-raw-video_id (case-insensitive),
mirrors analysis/p1_pe_eval_2026-05-07/phase_d/run_chronic_filter.py.

Output:
  arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet
  arena/canaries/teams_chronic_diverse_800_2026-05-07.README.md

Reproducibility: deterministic given pandas + seed=42 only.
CPU only; no parallelism (n_jobs=1).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- paths
ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
)
PHASE_A = ROOT / "analysis/p1_pe_eval_2026-05-07/raw_reports/phase_a"
PHASE_C = ROOT / "analysis/p1_pe_eval_2026-05-07/raw_reports/phase_c"
OUT_DIR = ROOT / "arena/canaries"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = OUT_DIR / "teams_chronic_diverse_800_2026-05-07.parquet"
README_PATH = OUT_DIR / "teams_chronic_diverse_800_2026-05-07.README.md"

P8A_TAG = "p8a_reference_step5000"
SEED = 42

# ----------------------------------------------------------------- logging
logger = logging.getLogger("build_canary")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
logger.addHandler(sh)


# ----------------------------------------------------------------- helpers
def video_matches_cid(video_id: str, cid: str) -> bool:
    """Prefix-match cid against raw video_id (case-insensitive).

    Mirrors phase_d/run_chronic_filter.py.video_matches_cid:
    'PC_Generator__s22' matches 'PC_Generator__s22__seg_488.2__real',
    'bla_bla_chow' matches 'bla_bla_chow__seq...' AND 'bla_bla_chow__s2__seg...'.
    Caller is responsible for excluding sub-prefixes (e.g. 'bla_bla_chow' vs
    'bla_bla_chow__s2', 'dor' vs 'dor_shkedi') if disjointness is required.
    """
    if not isinstance(video_id, str):
        return False
    vl = video_id.lower()
    cl = cid.lower()
    return vl == cl or vl.startswith(cl + "_") or vl.startswith(cl + "__")


def load_phase_a(suite: str) -> pd.DataFrame:
    p = PHASE_A / f"{suite}_{P8A_TAG}_frames_report.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df["__source_csv"] = str(p)
    return df


def load_phase_c(suite: str) -> pd.DataFrame:
    p = PHASE_C / f"{suite}_{P8A_TAG}_frames_report.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df["__source_csv"] = str(p)
    return df


def stratified_video_sample(
    df: pd.DataFrame,
    n_target: int,
    seed: int,
) -> pd.DataFrame:
    """Sample up to n_target frames stratified by video_id.

    Strategy:
      - If len(df) <= n_target: return df.
      - Else: distribute target evenly across videos, then top up with
        randomized leftovers. Frame-level random_state=seed.
    """
    if len(df) <= n_target:
        return df.copy()

    df = df.copy()
    rng = np.random.default_rng(seed)
    video_ids = df["video_id"].unique().tolist()
    rng.shuffle(video_ids)
    n_videos = len(video_ids)

    # Even distribution: each video gets at least floor(n_target / n_videos),
    # then the first (n_target % n_videos) videos get +1.
    base = n_target // n_videos
    remainder = n_target - base * n_videos

    selected_idx: list[int] = []
    for i, vid in enumerate(video_ids):
        k = base + (1 if i < remainder else 0)
        if k <= 0:
            continue
        sub = df[df["video_id"] == vid]
        if len(sub) <= k:
            selected_idx.extend(sub.index.tolist())
        else:
            sampled = sub.sample(n=k, random_state=seed + i, replace=False)
            selected_idx.extend(sampled.index.tolist())
    out = df.loc[selected_idx].reset_index(drop=True)

    # Could be short if some videos had fewer than k available; top up with
    # uniform random from the remaining unsampled rows (still seed=42).
    if len(out) < n_target:
        leftover = df.drop(index=selected_idx)
        n_short = n_target - len(out)
        if len(leftover) > 0:
            top_up = leftover.sample(
                n=min(n_short, len(leftover)), random_state=seed, replace=False
            )
            out = pd.concat([out, top_up], ignore_index=True)
    return out


# ----------------------------------------------------------------- cohorts
def cohort_chronic(
    teams_real: pd.DataFrame,
    cid: str,
    cohort_name: str,
    base_identity: str,
    n_target: int = 50,
    exclude_prefixes: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Filter teams_real by chronic prefix, exclude sub-prefixes (e.g. s2),
    then video_id-stratified sample n_target frames."""
    mask_in = teams_real["video_id"].apply(lambda v: video_matches_cid(v, cid))
    sub = teams_real[mask_in].copy()
    for excl in exclude_prefixes:
        mask_out = sub["video_id"].apply(lambda v: video_matches_cid(v, excl))
        sub = sub[~mask_out]
    if len(sub) == 0:
        logger.warning("cohort %s: 0 candidate frames after prefix filter", cohort_name)
    sampled = stratified_video_sample(sub, n_target, seed=SEED)
    sampled = sampled.copy()
    sampled["cohort"] = cohort_name
    sampled["base_identity"] = base_identity
    sampled["suite"] = "teams_real_all_dev"
    return sampled


def cohort_dor_shkedi(
    n_target: int = 50,
) -> pd.DataFrame:
    """Take all 50 frames from teams_real_dor_dev (the 50-frame dor suite)."""
    df = load_phase_a("teams_real_dor_dev")
    if len(df) > n_target:
        df = stratified_video_sample(df, n_target, seed=SEED)
    df = df.copy()
    df["cohort"] = "healthy_dor_shkedi"
    df["base_identity"] = "dor_shkedi"
    df["suite"] = "teams_real_dor_dev"
    return df


def cohort_hdtf_clean(
    n_target: int = 50,
) -> pd.DataFrame:
    df = load_phase_c("proper_real_clean_lockbox")
    sampled = stratified_video_sample(df, n_target, seed=SEED)
    sampled = sampled.copy()
    sampled["cohort"] = "hdtf_clean_real"
    sampled["base_identity"] = "hdtf_clean"
    sampled["suite"] = "proper_real_clean_lockbox"
    return sampled


def cohort_lockbox_fake(n_target: int = 100) -> pd.DataFrame:
    df = load_phase_a("teams_fake_all_lockbox")
    sampled = stratified_video_sample(df, n_target, seed=SEED)
    sampled = sampled.copy()
    sampled["cohort"] = "lockbox_fake"
    sampled["base_identity"] = "teams_fake_lockbox"
    sampled["suite"] = "teams_fake_all_lockbox"
    return sampled


def cohort_viso_fake(n_target: int = 50) -> pd.DataFrame:
    df = load_phase_a("visomaster_enhanced_macro_dev")
    sampled = stratified_video_sample(df, n_target, seed=SEED)
    sampled = sampled.copy()
    sampled["cohort"] = "viso_fake"
    sampled["base_identity"] = "visomaster_enhanced_macro"
    sampled["suite"] = "visomaster_enhanced_macro_dev"
    return sampled


def cohort_deeplive_fake(n_target: int = 50) -> pd.DataFrame:
    df = load_phase_a("deeplive_enhanced_dev")
    sampled = stratified_video_sample(df, n_target, seed=SEED)
    sampled = sampled.copy()
    sampled["cohort"] = "deeplive_fake"
    sampled["base_identity"] = "deeplive_enhanced"
    sampled["suite"] = "deeplive_enhanced_dev"
    return sampled


# ----------------------------------------------------------------- main
def main() -> int:
    logger.info("loading teams_real_all_dev (P8A reference)")
    teams_real = load_phase_a("teams_real_all_dev")
    logger.info("teams_real_all_dev: %d rows", len(teams_real))

    cohorts: list[pd.DataFrame] = []

    # ---- chronic-6 (300 frames)
    logger.info("--- chronic-6 ---")
    cohorts.append(
        cohort_chronic(
            teams_real,
            cid="PC_Generator__s22",
            cohort_name="chronic_PCGen_s22",
            base_identity="PC_Generator__s22",
        )
    )
    cohorts.append(
        cohort_chronic(
            teams_real,
            cid="PC_Generator__s45",
            cohort_name="chronic_PCGen_s45",
            base_identity="PC_Generator__s45",
        )
    )
    cohorts.append(
        cohort_chronic(
            teams_real,
            cid="Q__s6",
            cohort_name="chronic_Q_s6",
            base_identity="Q__s6",
        )
    )
    # bla_bla_chow without __s2
    cohorts.append(
        cohort_chronic(
            teams_real,
            cid="bla_bla_chow",
            cohort_name="chronic_bla_bla_chow",
            base_identity="bla_bla_chow",
            exclude_prefixes=("bla_bla_chow__s2",),
        )
    )
    cohorts.append(
        cohort_chronic(
            teams_real,
            cid="bla_bla_chow__s2",
            cohort_name="chronic_bla_bla_chow_s2",
            base_identity="bla_bla_chow__s2",
        )
    )
    cohorts.append(
        cohort_chronic(
            teams_real,
            cid="Roy_D",
            cohort_name="chronic_Roy_D",
            base_identity="Roy_D",
        )
    )

    # ---- healthy diverse (250 frames)
    logger.info("--- healthy diverse ---")
    cohorts.append(
        cohort_chronic(
            teams_real,
            cid="Test_Cam",
            cohort_name="healthy_test_cam",
            base_identity="Test_Cam",
        )
    )
    cohorts.append(
        cohort_chronic(
            teams_real,
            cid="Md_noyn_Sharker",
            cohort_name="healthy_md_noyn_sharker",
            base_identity="Md_noyn_Sharker",
        )
    )
    cohorts.append(
        cohort_chronic(
            teams_real,
            cid="Xiang_Xiang2_Feng",
            cohort_name="healthy_xiang_xiang2_feng",
            base_identity="Xiang_Xiang2_Feng",
        )
    )
    # 'dor' (the Dor identity) — exclude dor_shkedi to keep cohorts disjoint
    cohorts.append(
        cohort_chronic(
            teams_real,
            cid="dor",
            cohort_name="healthy_dor",
            base_identity="dor",
            exclude_prefixes=("dor_shkedi",),
        )
    )
    # dor_shkedi (50-frame suite) — separate phase_a CSV
    cohorts.append(cohort_dor_shkedi())

    # ---- HDTF clean reals (50 frames)
    logger.info("--- hdtf clean ---")
    cohorts.append(cohort_hdtf_clean())

    # ---- fakes (200 frames)
    logger.info("--- fakes ---")
    cohorts.append(cohort_lockbox_fake())
    cohorts.append(cohort_viso_fake())
    cohorts.append(cohort_deeplive_fake())

    # ---- assemble
    all_df = pd.concat(cohorts, ignore_index=True)
    logger.info("pre-clean total rows: %d", len(all_df))

    # frame_path / frame_prob sanity
    n_before = len(all_df)
    all_df = all_df.dropna(subset=["frame_path", "frame_prob"])
    all_df = all_df[all_df["frame_path"].astype(str).str.len() > 0]
    n_after = len(all_df)
    if n_before != n_after:
        logger.warning("dropped %d rows missing frame_path or frame_prob", n_before - n_after)

    # Build final canary frame
    canary = pd.DataFrame(
        {
            "frame_idx": np.arange(len(all_df), dtype=np.int64),
            "frame_path": all_df["frame_path"].astype(str).values,
            "label": all_df["label"].astype(np.int64).values,
            "cohort": all_df["cohort"].astype(str).values,
            "base_identity": all_df["base_identity"].astype(str).values,
            "suite": all_df["suite"].astype(str).values,
            "p8a_reference_score": all_df["frame_prob"].astype(np.float64).values,
        }
    )

    # Validation
    cohort_counts = canary["cohort"].value_counts().to_dict()
    label_counts = canary["label"].value_counts().to_dict()
    logger.info("=== VALIDATION ===")
    logger.info("total rows: %d (target=800)", len(canary))
    logger.info("label counts: %s (target reals=600 fakes=200)", label_counts)
    logger.info("cohort counts: %s", cohort_counts)
    expected_cohorts = {
        "chronic_PCGen_s22",
        "chronic_PCGen_s45",
        "chronic_Q_s6",
        "chronic_bla_bla_chow",
        "chronic_bla_bla_chow_s2",
        "chronic_Roy_D",
        "healthy_test_cam",
        "healthy_md_noyn_sharker",
        "healthy_xiang_xiang2_feng",
        "healthy_dor",
        "healthy_dor_shkedi",
        "hdtf_clean_real",
        "lockbox_fake",
        "viso_fake",
        "deeplive_fake",
    }
    found_cohorts = set(canary["cohort"].unique())
    missing = expected_cohorts - found_cohorts
    if missing:
        logger.warning("missing expected cohorts: %s", missing)

    # Write parquet
    canary.to_parquet(PARQUET_PATH, engine="pyarrow", index=False)
    logger.info("wrote parquet -> %s", PARQUET_PATH)

    # ---- README
    write_readme(canary)
    logger.info("wrote readme  -> %s", README_PATH)

    return 0


def write_readme(canary: pd.DataFrame) -> None:
    """Write the README with composition + provenance + P8A stats per cohort."""
    cohort_order = [
        "chronic_PCGen_s22",
        "chronic_PCGen_s45",
        "chronic_Q_s6",
        "chronic_bla_bla_chow",
        "chronic_bla_bla_chow_s2",
        "chronic_Roy_D",
        "healthy_test_cam",
        "healthy_md_noyn_sharker",
        "healthy_xiang_xiang2_feng",
        "healthy_dor",
        "healthy_dor_shkedi",
        "hdtf_clean_real",
        "lockbox_fake",
        "viso_fake",
        "deeplive_fake",
    ]
    cohort_target = {
        "chronic_PCGen_s22": 50,
        "chronic_PCGen_s45": 50,
        "chronic_Q_s6": 50,
        "chronic_bla_bla_chow": 50,
        "chronic_bla_bla_chow_s2": 50,
        "chronic_Roy_D": 50,
        "healthy_test_cam": 50,
        "healthy_md_noyn_sharker": 50,
        "healthy_xiang_xiang2_feng": 50,
        "healthy_dor": 50,
        "healthy_dor_shkedi": 50,
        "hdtf_clean_real": 50,
        "lockbox_fake": 100,
        "viso_fake": 50,
        "deeplive_fake": 50,
    }
    suite_by_cohort: dict[str, str] = {
        c: canary.loc[canary["cohort"] == c, "suite"].iloc[0]
        if (canary["cohort"] == c).any()
        else "(missing)"
        for c in cohort_order
    }

    rows: list[str] = []
    for c in cohort_order:
        sub = canary[canary["cohort"] == c]
        n = int(len(sub))
        target = cohort_target[c]
        if n == 0:
            mean_s = float("nan")
            p50_s = float("nan")
            p95_s = float("nan")
        else:
            mean_s = float(sub["p8a_reference_score"].mean())
            p50_s = float(sub["p8a_reference_score"].median())
            p95_s = float(sub["p8a_reference_score"].quantile(0.95))
        rows.append(
            f"| {c} | {target} | {n} | {n - target:+d} | {mean_s:.4f} | "
            f"{p50_s:.4f} | {p95_s:.4f} | {suite_by_cohort[c]} |"
        )

    n_total = int(len(canary))
    n_real = int((canary["label"] == 0).sum())
    n_fake = int((canary["label"] == 1).sum())

    body = f"""# teams_chronic_diverse_800 canary (2026-05-07)

In-training monitoring canary for the 3-slot from-scratch GPU experiment
(P2 packet, launch 2026-05-07 PM). Forward-passed every 1000 steps.

- Parquet: `arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet`
- Total frames: {n_total} (target 800)
- Reals: {n_real} | Fakes: {n_fake}
- Sampling seed: {SEED} (deterministic)

## Source CSVs

All `p8a_reference_score` values are P8A_REFERENCE_STEP5000 `frame_prob`
(per-frame `prob_fake`-equivalent column in the Phase A/C reports).

- Phase A (per-suite frame reports):
  `analysis/p1_pe_eval_2026-05-07/raw_reports/phase_a/<suite>_p8a_reference_step5000_frames_report.csv`
- Phase C (HDTF / proper_real_clean_lockbox):
  `analysis/p1_pe_eval_2026-05-07/raw_reports/phase_c/proper_real_clean_lockbox_p8a_reference_step5000_frames_report.csv`

## Composition

| cohort | target_n | actual_n | delta | p8a_mean | p8a_p50 | p8a_p95 | source_suite |
|---|---:|---:|---:|---:|---:|---:|---|
""" + "\n".join(rows) + f"""

## Cohort definitions

- **chronic_PCGen_s22 / chronic_PCGen_s45 / chronic_Q_s6 /
  chronic_bla_bla_chow / chronic_bla_bla_chow_s2 / chronic_Roy_D**:
  six chronic FP-tail identities from
  `analysis/p1_pe_eval_2026-05-07/JOINT_TAU_SWEEP_FACTS_2026-05-07.md` /
  Phase D `chronic_flag_definition.json`. Drawn from
  `teams_real_all_dev` via prefix-on-raw-video_id matching (mirrors
  `analysis/p1_pe_eval_2026-05-07/phase_d/run_chronic_filter.py`,
  `video_matches_cid`). `chronic_bla_bla_chow` excludes
  `bla_bla_chow__s2` rows so the cohorts are disjoint.

- **healthy_test_cam, healthy_md_noyn_sharker,
  healthy_xiang_xiang2_feng**: healthy diverse reals from
  `teams_real_all_dev`, prefix-matched.

- **healthy_dor**: from `teams_real_all_dev`, prefix `dor` with
  `dor_shkedi` excluded (since the prefix would otherwise match the
  shkedi rows).

- **healthy_dor_shkedi**: 50 frames from `teams_real_dor_dev` (the
  pre-built 50-frame dor suite). The full suite is taken if size
  permits.

- **hdtf_clean_real**: 50 frames from `proper_real_clean_lockbox`
  (HDTF-style cross-substrate reals, F4-relevant). Phase C report.

- **lockbox_fake**: 100 frames from `teams_fake_all_lockbox`
  (F1 deployment-relevant fakes), video_id-stratified.

- **viso_fake**: 50 frames from `visomaster_enhanced_macro_dev`
  (the structural ceiling cohort — refer to
  `project_viso_ceiling_unbroken_10_packets.md`).

- **deeplive_fake**: 50 frames from `deeplive_enhanced_dev`
  (deeplive method).

## Sampling rule

`stratified_video_sample(seed=42)`:
- If candidate set <= target: take all.
- Else: distribute the target evenly across video_ids
  (floor + remainder), top-up uniformly from leftovers if some videos
  had insufficient frames.

## Schema

| column | dtype | description |
|---|---|---|
| frame_idx | int64 | 0..n-1 |
| frame_path | str | gs:// URL into the existing storage substrate |
| label | int64 | 0 = real, 1 = fake |
| cohort | str | one of the {len(cohort_order)} cohort tags above |
| base_identity | str | identity tag for grouping |
| suite | str | source suite name |
| p8a_reference_score | float64 | P8A_REFERENCE_STEP5000 `frame_prob` on this frame |

## Sanity

- Rows missing `frame_path` or with NaN `p8a_reference_score` are
  excluded by the build script.
- Build script: `arena/canaries/build_canary_2026-05-07.py`.
- Run wall-clock: < 30 s (CPU-only).
"""
    README_PATH.write_text(body)


if __name__ == "__main__":
    sys.exit(main())
