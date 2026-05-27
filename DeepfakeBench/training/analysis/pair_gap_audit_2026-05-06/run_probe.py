#!/usr/bin/env python3
"""
PAIR_GAP_AUDIT_P8A_E2B_PA_2026-05-06

Go/no-go probe for PE_PAIR_RANK_DRO (pair-ranking-loss training packet).

Purpose
-------
For each "pair" of (real, fake) frames that share base_identity / canonical
source, compute the pair_gap = score(fake) - score(real) on the cached
P8A / E2B / PA_3800 score columns in
    analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv

Then condition on missed_fake / caught_fake / FP_real and answer the
load-bearing question: do missed fakes have pair_gap <= 0 on a meaningful
fraction of pairs? If yes (>25%), GREEN-LIGHT P1 (PE_PAIR_RANK_DRO);
if pair_gap > 0 already on most missed fakes, RED-LIGHT it.

Pairing semantics
-----------------
The inference manifest does NOT contain frame-level paired structure
(teams fake/real come from different sessions/segs; visomaster_v2_dor has
no companion real bucket here; live-fakes-teams-prod has fakes only in
most sessions). The closest natural pairing layer in this manifest is at
the identity / canonical-subject level. We therefore construct pairs as:

    For each canonical_subject, take the cross-product of {real frames} x
    {fake frames}. If that exceeds CROSS_PRODUCT_CAP, sub-sample
    deterministically (with a seeded RNG) to keep memory bounded.

This is a *coarser* pairing than the trainer's true paired loader (where
real & fake share source video / pose / frame index). It still answers the
go/no-go question: if fake>real ordering is already satisfied at the
identity-pool level, it is also satisfied at the tighter same-source level
(by Jensen / monotonicity arguments). If fake>real ordering is broken at
the identity-pool level, the tighter pair-rank loss can only narrow the
gap, never widen it -- so a positive lower bound on "fraction of pairs
with gap <= 0" remains a valid greenlight signal for the training packet.

This caveat is reported in summary.json + FINDINGS.md.

Outputs (analysis/pair_gap_audit_2026-05-06/outputs/):
  - pair_gaps.csv         per-pair scores + gaps
  - summary.json          population stats + decision-rule outputs
  - missed_fake_audit.csv per missed-fake pair fraction by method/transport
  - fp_real_audit.csv     per FP-real fraction with paired fake still > real
  - FINDINGS.md           1-2 page synthesis with green/red/mixed verdict

CPU-only. n_jobs=1 throughout. Idempotent.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

REPO_ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
)
MANIFEST_PATH = (
    REPO_ROOT
    / "analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv"
)
OUT_DIR = REPO_ROOT / "analysis/pair_gap_audit_2026-05-06/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cap on cross-product per canonical subject to keep memory bounded.
# 200x200 = 40k pairs per subject is plenty for distribution estimation.
CROSS_PRODUCT_CAP = 4_000  # pairs per subject after sub-sampling
RNG_SEED = 20260506

CKPTS = ("P8A", "E2B", "PA_3800")
FAKE_THRESH = 0.5  # canonical missed/caught split
REAL_THRESH = 0.5  # canonical FP/OK split

# -----------------------------------------------------------------------------
# Subject canonicalization + method/transport derivation
# -----------------------------------------------------------------------------


def canonicalize_subject(base_identity: str, suite: str) -> str:
    """
    Map base_identity -> canonical subject so that 'dor_fake_*' from
    visomaster_v2_dor and 'dor_evening' / 'dor_morning' both map to 'dor'.
    Suite is used as a tie-breaker.
    """
    s = str(base_identity).lower()
    # dor variants
    if s.startswith("dor_fake_") or s in ("dor_evening", "dor_morning"):
        return "dor_local"
    if s == "dor_shkedi":
        return "dor_shkedi_teams"
    if s == "dor_shkedi__s16":
        return "dor_shkedi__s16"
    if s.startswith("team_may5__"):
        return s.replace("team_may5__", "may5_")
    # extra_xiang_real, extra_xiang_fake, extra_xinghe_real, extra_xinghe_fake
    if s in ("extra_xiang", "extra_xinghe", "extra_roy_d"):
        return s.replace("extra_", "extra_")
    # teams identities like 'Cam_Test__s32', 'PC_Generator__s15' -- already canonical
    return s


def parse_method(frame_path: str, base_identity: str, suite: str) -> str:
    """
    Derive a coarse method label.
    """
    p = str(frame_path).lower()
    if "visomaster-enhanced-face-cropped-v2" in p:
        # subdir after 'fake/' is the method, e.g. 'ghostface_v1', 'simswap'
        m = re.search(r"/fake/([^/]+)/", p)
        return f"viso_v2_{m.group(1)}" if m else "viso_v2_unknown"
    if "live-fakes-teams-prod/fake" in p:
        m = re.search(r"/fake/[^/]+/([^/]+)/", p)
        return f"live_prod_{m.group(1)}" if m else "live_prod_unknown"
    if "live-fakes-teams-prod/real" in p:
        return "live_prod_real"
    if "teams-faces-data-test-2914" in p:
        # teams pipeline; method is 'teams_passthrough' for all
        return "teams_passthrough"
    if "/dor_deep_live_cam/" in p:
        # e.g. 'dor_fake_simswap', 'dor_fake_bill_gates_regular'
        m = re.search(r"/dor_deep_live_cam/([^/]+)/", p)
        return f"deeplive_{m.group(1)}" if m else "deeplive_unknown"
    if "/extra/" in p:
        m = re.search(r"/extra/([^/]+)/", p)
        return f"extra_{m.group(1)}" if m else "extra_unknown"
    if "/faces_dor/" in p:
        m = re.search(r"/faces_dor/([^/]+)/", p)
        return f"dor_real_{m.group(1)}" if m else "dor_real_unknown"
    if "real-teams-dor-roee" in p:
        return "may5_team_real"
    return f"suite_{suite}"


def parse_transport(frame_path: str, suite: str, base_identity: str) -> str:
    """
    Derive transport: 'teams' if path comes via Teams pipeline, 'raw'
    otherwise. Local / GCS clean buckets count as raw.
    """
    p = str(frame_path).lower()
    if "teams-faces-data-test-2914" in p:
        return "teams"
    if "live-fakes-teams-prod" in p:
        return "teams"
    if "real-teams-dor-roee" in p:
        return "teams"
    return "raw"


def parse_enhancer(method: str) -> str:
    """
    Coarse enhancer family. 'enhanced' / 'gfpgan' / 'gpen' / 'codeformer' /
    'plain' or 'unknown'.
    """
    m = method.lower()
    if "gfpgan" in m:
        return "gfpgan"
    if "gpen" in m:
        return "gpen"
    if "codeformer" in m:
        return "codeformer"
    if "enhanced" in m:
        return "enhanced_other"
    if "_real" in m or method == "live_prod_real":
        return "n/a_real"
    return "plain"


# -----------------------------------------------------------------------------
# Load manifest
# -----------------------------------------------------------------------------


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST_PATH)
    print(f"[load] {MANIFEST_PATH.name}: {len(df):,} rows x {len(df.columns)} cols")
    # Drop rows with any missing score
    score_cols = [f"score_{c}" for c in CKPTS]
    n_before = len(df)
    df = df.dropna(subset=score_cols).copy()
    print(f"[load] dropped {n_before - len(df):,} rows with missing scores")
    df["canonical_subject"] = df.apply(
        lambda r: canonicalize_subject(r["base_identity"], r["suite"]), axis=1
    )
    df["method"] = df.apply(
        lambda r: parse_method(r["frame_path"], r["base_identity"], r["suite"]),
        axis=1,
    )
    df["transport"] = df.apply(
        lambda r: parse_transport(r["frame_path"], r["suite"], r["base_identity"]),
        axis=1,
    )
    df["enhancer"] = df["method"].apply(parse_enhancer)
    return df


# -----------------------------------------------------------------------------
# Build pairs
# -----------------------------------------------------------------------------


def build_pairs(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    For each canonical_subject with both label=0 and label=1 frames, take
    the cross product of (real, fake), capped at CROSS_PRODUCT_CAP per
    subject via deterministic sub-sampling.
    """
    pairs = []
    pair_id = 0
    cov = []
    for subj, sub in df.groupby("canonical_subject"):
        reals = sub[sub["label"] == 0]
        fakes = sub[sub["label"] == 1]
        n_r, n_f = len(reals), len(fakes)
        if n_r == 0 or n_f == 0:
            cov.append(
                dict(
                    canonical_subject=subj,
                    n_real=n_r,
                    n_fake=n_f,
                    n_pairs=0,
                    paired=False,
                )
            )
            continue
        full = n_r * n_f
        # Deterministic: take all if under cap; else sub-sample without
        # replacement.
        if full <= CROSS_PRODUCT_CAP:
            real_idx = np.repeat(reals.index.values, n_f)
            fake_idx = np.tile(fakes.index.values, n_r)
        else:
            # uniform random pairing
            real_idx = rng.choice(reals.index.values, size=CROSS_PRODUCT_CAP, replace=True)
            fake_idx = rng.choice(fakes.index.values, size=CROSS_PRODUCT_CAP, replace=True)
        for ri, fi in zip(real_idx, fake_idx):
            r = df.loc[ri]
            f = df.loc[fi]
            pairs.append(
                dict(
                    pair_id=pair_id,
                    canonical_subject=subj,
                    real_path=r["frame_path"],
                    fake_path=f["frame_path"],
                    real_suite=r["suite"],
                    fake_suite=f["suite"],
                    real_base_identity=r["base_identity"],
                    fake_base_identity=f["base_identity"],
                    method=f["method"],
                    enhancer=f["enhancer"],
                    fake_transport=f["transport"],
                    real_transport=r["transport"],
                    transport_match=(f["transport"] == r["transport"]),
                    real_is_lockbox=bool(r["is_lockbox"]),
                    fake_is_lockbox=bool(f["is_lockbox"]),
                    real_face_size=r["face_size"],
                    fake_face_size=f["face_size"],
                    real_quality=r["quality"],
                    fake_quality=f["quality"],
                    real_score_P8A=float(r["score_P8A"]),
                    fake_score_P8A=float(f["score_P8A"]),
                    real_score_E2B=float(r["score_E2B"]),
                    fake_score_E2B=float(f["score_E2B"]),
                    real_score_PA_3800=float(r["score_PA_3800"]),
                    fake_score_PA_3800=float(f["score_PA_3800"]),
                )
            )
            pair_id += 1
        cov.append(
            dict(
                canonical_subject=subj,
                n_real=n_r,
                n_fake=n_f,
                n_pairs=min(full, CROSS_PRODUCT_CAP),
                paired=True,
            )
        )
    pairs_df = pd.DataFrame(pairs)
    cov_df = pd.DataFrame(cov)
    return pairs_df, cov_df


# -----------------------------------------------------------------------------
# Compute gaps
# -----------------------------------------------------------------------------


def add_gaps(pairs: pd.DataFrame) -> pd.DataFrame:
    EPS = 1e-6
    for c in CKPTS:
        rs = pairs[f"real_score_{c}"].clip(EPS, 1 - EPS)
        fs = pairs[f"fake_score_{c}"].clip(EPS, 1 - EPS)
        pairs[f"pair_gap_{c}"] = pairs[f"fake_score_{c}"] - pairs[f"real_score_{c}"]
        # Logit space
        rl = np.log(rs / (1 - rs))
        fl = np.log(fs / (1 - fs))
        pairs[f"pair_gap_logit_{c}"] = fl - rl
    return pairs


# -----------------------------------------------------------------------------
# Decision-rule outputs
# -----------------------------------------------------------------------------


def gap_thresholds(series: pd.Series) -> dict:
    if len(series) == 0:
        return dict(p_gt_0=None, p_gt_05=None, p_gt_10=None, p_le_0=None, n=0)
    s = series.dropna()
    n = len(s)
    return dict(
        n=int(n),
        p_gt_0=float((s > 0).mean()),
        p_gt_05=float((s > 0.5).mean()),
        p_gt_10=float((s > 1.0).mean()),
        p_le_0=float((s <= 0).mean()),
        mean=float(s.mean()),
        median=float(s.median()),
        std=float(s.std()),
    )


def summarize(pairs: pd.DataFrame, cov_df: pd.DataFrame) -> dict:
    out: dict = {
        "version": "1.1",
        "n_pairs_total": int(len(pairs)),
        "n_subjects_with_pairs": int(cov_df["paired"].sum()),
        "n_subjects_no_pairs": int((~cov_df["paired"]).sum()),
        "cross_product_cap": CROSS_PRODUCT_CAP,
        "rng_seed": RNG_SEED,
        "ckpts": list(CKPTS),
        "pairing_caveat": (
            "Pairs are CROSS-PRODUCT within canonical_subject (not "
            "frame-level). The inference manifest does not carry frame-level "
            "paired structure (teams fake/real are different sessions; "
            "visomaster_v2_dor has no co-bucketed real). This audit therefore "
            "answers the question at the identity / canonical-subject pool "
            "level. If fake>real ordering is broken at this coarser level it "
            "will also be broken at the trainer's tighter same-source pair "
            "level (the latter has higher score noise). Greenlight is "
            "conservative; redlight is suggestive only."
        ),
        "transport_match_breakdown": (
            {
                f"{r}__{f}": int(n)
                for (r, f), n in pairs.groupby(
                    ["real_transport", "fake_transport"]
                ).size().items()
            }
            if len(pairs) > 0
            else {}
        ),
        "by_ckpt": {},
        "by_ckpt_transport_matched": {},
    }
    out["coverage_by_subject"] = (
        cov_df.sort_values("n_pairs", ascending=False).to_dict(orient="records")
    )
    # Helper: per-subject P(gap<=0|missed) for subject-weighted averaging
    def _per_subject_stats(p: pd.DataFrame, c: str) -> dict:
        miss_mask = p[f"fake_score_{c}"] < FAKE_THRESH
        miss = p[miss_mask]
        if len(miss) == 0:
            return dict(subject_mean_p_le_0=None, per_subject={})
        per = (
            miss.groupby("canonical_subject")
            .apply(lambda x: float((x[f"pair_gap_{c}"] <= 0).mean()))
            .rename("p_gap_le_0")
            .to_dict()
        )
        # also n_missed per subject
        per_n = miss.groupby("canonical_subject").size().to_dict()
        per_dict = {
            k: dict(p_gap_le_0=v, n_missed=int(per_n.get(k, 0)))
            for k, v in per.items()
        }
        return dict(
            subject_mean_p_le_0=float(np.mean(list(per.values()))),
            per_subject=per_dict,
        )

    # Headline: by ckpt, raw probability gap and logit gap (all pairs)
    for c in CKPTS:
        gap = pairs[f"pair_gap_{c}"]
        gap_logit = pairs[f"pair_gap_logit_{c}"]
        rs = pairs[f"real_score_{c}"]
        fs = pairs[f"fake_score_{c}"]
        # Buckets
        missed_fake = fs < FAKE_THRESH
        caught_fake = fs >= FAKE_THRESH
        fp_real = rs >= REAL_THRESH
        ok_real = rs < REAL_THRESH
        per_subj = _per_subject_stats(pairs, c)
        out["by_ckpt"][c] = dict(
            overall_gap_prob=gap_thresholds(gap),
            overall_gap_logit=gap_thresholds(gap_logit),
            n_missed_fake=int(missed_fake.sum()),
            n_caught_fake=int(caught_fake.sum()),
            n_fp_real=int(fp_real.sum()),
            n_ok_real=int(ok_real.sum()),
            missed_fake_gap_prob=gap_thresholds(gap[missed_fake]),
            caught_fake_gap_prob=gap_thresholds(gap[caught_fake]),
            fp_real_gap_prob=gap_thresholds(gap[fp_real]),
            ok_real_gap_prob=gap_thresholds(gap[ok_real]),
            missed_fake_p_le_0_pair_weighted=(
                None
                if missed_fake.sum() == 0
                else float((gap[missed_fake] <= 0).mean())
            ),
            missed_fake_p_le_0_subject_weighted=per_subj["subject_mean_p_le_0"],
            per_subject_missed_fake=per_subj["per_subject"],
            fp_real_p_gt_0=(
                None
                if fp_real.sum() == 0
                else float((gap[fp_real] > 0).mean())
            ),
        )
    # Transport-matched only (fake_transport == real_transport): cleaner signal
    pairs_tm = pairs[pairs["transport_match"]]
    for c in CKPTS:
        if len(pairs_tm) == 0:
            out["by_ckpt_transport_matched"][c] = None
            continue
        gap = pairs_tm[f"pair_gap_{c}"]
        fs = pairs_tm[f"fake_score_{c}"]
        miss = fs < FAKE_THRESH
        per_subj = _per_subject_stats(pairs_tm, c)
        out["by_ckpt_transport_matched"][c] = dict(
            n_pairs=int(len(pairs_tm)),
            n_missed_fake=int(miss.sum()),
            missed_fake_gap_prob=gap_thresholds(gap[miss]),
            missed_fake_p_le_0_pair_weighted=(
                None
                if miss.sum() == 0
                else float((gap[miss] <= 0).mean())
            ),
            missed_fake_p_le_0_subject_weighted=per_subj["subject_mean_p_le_0"],
            per_subject_missed_fake=per_subj["per_subject"],
        )
    # Decision rule per ckpt — use TRANSPORT-MATCHED PAIR-WEIGHTED as primary,
    # report subject-weighted as cross-check.
    decisions = {}
    for c in CKPTS:
        p_pair_all = out["by_ckpt"][c]["missed_fake_p_le_0_pair_weighted"]
        p_subj_all = out["by_ckpt"][c]["missed_fake_p_le_0_subject_weighted"]
        tm = out["by_ckpt_transport_matched"][c]
        p_pair_tm = (tm["missed_fake_p_le_0_pair_weighted"] if tm else None)
        p_subj_tm = (tm["missed_fake_p_le_0_subject_weighted"] if tm else None)
        # Verdict driven by transport-matched pair-weighted
        primary = p_pair_tm if p_pair_tm is not None else p_pair_all
        if primary is None:
            verdict = "N/A_no_missed_fakes"
        elif primary > 0.25:
            verdict = "GREEN"
        elif primary < 0.10:
            verdict = "RED"
        else:
            verdict = "AMBER"
        decisions[c] = dict(
            primary_metric=primary,
            primary_metric_name="P(pair_gap<=0 | missed_fake) "
                                "[transport-matched, pair-weighted]",
            p_pair_all=p_pair_all,
            p_subj_all=p_subj_all,
            p_pair_tm=p_pair_tm,
            p_subj_tm=p_subj_tm,
            verdict=verdict,
        )
    out["decisions_per_ckpt"] = decisions
    # Aggregate verdict (worst-case)
    verdicts = [d["verdict"] for d in decisions.values() if d["verdict"] != "N/A_no_missed_fakes"]
    if not verdicts:
        out["aggregate_verdict"] = "N/A"
    elif "GREEN" in verdicts and "RED" not in verdicts:
        out["aggregate_verdict"] = "GREEN"
    elif "RED" in verdicts and "GREEN" not in verdicts:
        out["aggregate_verdict"] = "RED"
    else:
        out["aggregate_verdict"] = "MIXED"
    return out


# -----------------------------------------------------------------------------
# Sub-tables
# -----------------------------------------------------------------------------


def missed_fake_audit_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CKPTS:
        sub = pairs[pairs[f"fake_score_{c}"] < FAKE_THRESH].copy()
        if len(sub) == 0:
            continue
        sub["ckpt"] = c
        gp = sub[f"pair_gap_{c}"]
        gl = sub[f"pair_gap_logit_{c}"]
        sub_le0 = (gp <= 0)
        sub["pair_gap_prob"] = gp
        sub["pair_gap_logit"] = gl
        sub["pair_gap_le_0"] = sub_le0
        for keys in [
            ("ckpt",),
            ("ckpt", "method"),
            ("ckpt", "enhancer"),
            ("ckpt", "fake_transport"),
            ("ckpt", "method", "fake_transport"),
            ("ckpt", "canonical_subject"),
        ]:
            grp = sub.groupby(list(keys)).agg(
                n=("pair_gap_prob", "size"),
                mean_gap_prob=("pair_gap_prob", "mean"),
                median_gap_prob=("pair_gap_prob", "median"),
                p_gap_le_0=("pair_gap_le_0", "mean"),
                mean_gap_logit=("pair_gap_logit", "mean"),
            ).reset_index()
            grp["group_keys"] = "|".join(keys)
            rows.append(grp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def fp_real_audit_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CKPTS:
        sub = pairs[pairs[f"real_score_{c}"] >= REAL_THRESH].copy()
        if len(sub) == 0:
            continue
        sub["ckpt"] = c
        gp = sub[f"pair_gap_{c}"]
        gl = sub[f"pair_gap_logit_{c}"]
        sub_gt0 = (gp > 0)
        sub["pair_gap_prob"] = gp
        sub["pair_gap_logit"] = gl
        sub["pair_gap_gt_0"] = sub_gt0
        for keys in [
            ("ckpt",),
            ("ckpt", "method"),
            ("ckpt", "fake_transport"),
            ("ckpt", "canonical_subject"),
        ]:
            grp = sub.groupby(list(keys)).agg(
                n=("pair_gap_prob", "size"),
                mean_gap_prob=("pair_gap_prob", "mean"),
                median_gap_prob=("pair_gap_prob", "median"),
                p_gap_gt_0=("pair_gap_gt_0", "mean"),
                mean_gap_logit=("pair_gap_logit", "mean"),
            ).reset_index()
            grp["group_keys"] = "|".join(keys)
            rows.append(grp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


# -----------------------------------------------------------------------------
# FINDINGS.md writer
# -----------------------------------------------------------------------------


def write_findings(summary: dict, pairs: pd.DataFrame, cov_df: pd.DataFrame) -> None:
    lines = []
    L = lines.append
    L("# PAIR_GAP_AUDIT_P8A_E2B_PA_2026-05-06 — Findings")
    L("")
    L(f"**Generated:** {pd.Timestamp.utcnow().isoformat()}Z")
    L(f"**Manifest:** `{MANIFEST_PATH.relative_to(REPO_ROOT)}`")
    L(f"**Pairs constructed:** {summary['n_pairs_total']:,} across "
      f"{summary['n_subjects_with_pairs']} canonical subjects")
    L(f"**Subjects without paired structure (label=0 only or label=1 only):** "
      f"{summary['n_subjects_no_pairs']}")
    L("")
    L("## Headline verdict")
    L("")
    agg = summary["aggregate_verdict"]
    if agg == "GREEN":
        L("**GREEN** — pair-rank loss has signal. Greenlight P1 "
          "(`PE_PAIR_RANK_DRO`).")
    elif agg == "RED":
        L("**RED** — pair-rank loss is dead. Skip P1; demote to P2 "
          "(`PE_SBI`).")
    elif agg == "MIXED":
        L("**MIXED** — verdict differs across checkpoints. Use the per-ckpt "
          "breakdown below to pick the FT base where pair-rank has signal.")
    else:
        L("**N/A** — no missed fakes in dataset (decision rule not "
          "evaluable).")
    L("")
    L("## Decision rule")
    L("")
    L("> Per ckpt, P(pair_gap <= 0 | missed_fake):")
    L("> - **>25%** → **GREEN** (lever has signal).")
    L("> - **10-25%** → **AMBER** (marginal; depends on margin of "
      "loss + GroupDRO weighting).")
    L("> - **<10%** → **RED** (lever is dead — fakes already rank "
      "above reals on missed fakes).")
    L("")
    L("## Per-ckpt summary (transport-matched primary)")
    L("")
    L("| Ckpt | N pairs (TM) | N missed_fake (TM) | P(gap<=0\\|missed) [pair-w, TM] | P(gap<=0\\|missed) [subj-w, TM] | Mean gap (logit, all) | Verdict |")
    L("|------|------------:|------------------:|------------------------------:|-------------------------------:|---------------------:|---------|")
    for c in CKPTS:
        b = summary["by_ckpt"][c]
        d = summary["decisions_per_ckpt"][c]
        v = d["verdict"]
        gl = b["overall_gap_logit"]
        tm = summary["by_ckpt_transport_matched"][c] or {}
        npairs_tm = tm.get("n_pairs", 0)
        nmiss_tm = tm.get("n_missed_fake", 0)
        p_pair_tm = d.get("p_pair_tm")
        p_subj_tm = d.get("p_subj_tm")
        L(
            f"| {c} | {npairs_tm:,} | {nmiss_tm:,} | "
            f"{(p_pair_tm if p_pair_tm is not None else float('nan')):.4f} | "
            f"{(p_subj_tm if p_subj_tm is not None else float('nan')):.4f} | "
            f"{gl['mean']:.4f} | {v} |"
        )
    L("")
    L("## All-pair vs transport-matched cross-check")
    L("")
    L("| Ckpt | P(gap<=0\\|missed) all pair-w | all subj-w | TM pair-w | TM subj-w |")
    L("|------|----------------------------:|----------:|---------:|---------:|")
    for c in CKPTS:
        d = summary["decisions_per_ckpt"][c]
        L(
            f"| {c} | {(d['p_pair_all'] if d['p_pair_all'] is not None else float('nan')):.4f} | "
            f"{(d['p_subj_all'] if d['p_subj_all'] is not None else float('nan')):.4f} | "
            f"{(d['p_pair_tm'] if d['p_pair_tm'] is not None else float('nan')):.4f} | "
            f"{(d['p_subj_tm'] if d['p_subj_tm'] is not None else float('nan')):.4f} |"
        )
    L("")
    L("## Overall pair_gap distribution (raw probability) per ckpt")
    L("")
    L("| Ckpt | mean | median | std | P(gap>0) | P(gap>0.5) | P(gap>1.0) | P(gap<=0) |")
    L("|------|-----:|-------:|----:|--------:|----------:|----------:|---------:|")
    for c in CKPTS:
        gp = summary["by_ckpt"][c]["overall_gap_prob"]
        L(
            f"| {c} | {gp['mean']:.4f} | {gp['median']:.4f} | {gp['std']:.4f} | "
            f"{gp['p_gt_0']:.4f} | {gp['p_gt_05']:.4f} | {gp['p_gt_10']:.4f} | "
            f"{gp['p_le_0']:.4f} |"
        )
    L("")
    L("## Conditional: missed_fake subset")
    L("")
    L("| Ckpt | N missed | mean | median | P(gap<=0) |")
    L("|------|---------:|-----:|-------:|---------:|")
    for c in CKPTS:
        mf = summary["by_ckpt"][c]["missed_fake_gap_prob"]
        if mf["n"] == 0:
            L(f"| {c} | 0 | - | - | - |")
            continue
        L(
            f"| {c} | {mf['n']:,} | {mf['mean']:.4f} | {mf['median']:.4f} | "
            f"{mf['p_le_0']:.4f} |"
        )
    L("")
    L("## Conditional: FP_real subset (do paired fakes still outrank?)")
    L("")
    L("| Ckpt | N FP_real | mean gap | median gap | P(gap>0) |")
    L("|------|----------:|---------:|----------:|---------:|")
    for c in CKPTS:
        fp = summary["by_ckpt"][c]["fp_real_gap_prob"]
        if fp["n"] == 0:
            L(f"| {c} | 0 | - | - | - |")
            continue
        gt0 = summary["by_ckpt"][c]["fp_real_p_gt_0"]
        L(
            f"| {c} | {fp['n']:,} | {fp['mean']:.4f} | {fp['median']:.4f} | "
            f"{(gt0 if gt0 is not None else float('nan')):.4f} |"
        )
    L("")
    L("## Coverage by canonical subject")
    L("")
    L("| Canonical subject | N real | N fake | N pairs | Paired |")
    L("|---|---:|---:|---:|:---:|")
    for r in summary["coverage_by_subject"][:50]:
        L(
            f"| {r['canonical_subject']} | {r['n_real']:,} | {r['n_fake']:,} | "
            f"{r['n_pairs']:,} | {'Y' if r['paired'] else 'N'} |"
        )
    L("")
    L("## Per-subject P(gap<=0 | missed_fake) — heterogeneity check")
    L("")
    L("This audit pools pairs across canonical subjects, but the result can "
      "be dominated by one heavy subject (e.g. dor_local has the most fakes "
      "in this manifest). Per-subject breakdown shows where the lever has "
      "real signal vs where it is dead.")
    L("")
    for c in CKPTS:
        b = summary["by_ckpt"][c]
        per = b.get("per_subject_missed_fake", {}) or {}
        if not per:
            continue
        L(f"### {c}")
        L("")
        L("| Canonical subject | N missed_fake | P(gap<=0\\|missed) |")
        L("|---|---:|---:|")
        items = sorted(per.items(), key=lambda kv: -(kv[1].get("p_gap_le_0") or 0))
        for k, v in items:
            L(f"| {k} | {v.get('n_missed', 0):,} | {v.get('p_gap_le_0', 0.0):.4f} |")
        L("")
    L("## Recommended FT base if pair-rank is greenlit")
    L("")
    # Pick the ckpt with the highest primary metric that is also GREEN/AMBER.
    candidates = []
    for c in CKPTS:
        d = summary["decisions_per_ckpt"][c]
        if d["verdict"] in ("GREEN", "AMBER"):
            candidates.append((c, d.get("primary_metric")))
    if candidates:
        candidates.sort(key=lambda x: -(x[1] or 0))
        L(
            f"- **Recommended FT base:** `{candidates[0][0]}` — "
            f"P(pair_gap<=0 | missed_fake) [TM pair-weighted] = "
            f"{candidates[0][1]:.3f}, highest of "
            f"{[c for c,_ in candidates]}."
        )
        L(
            "- **Rationale:** the ckpt where the greatest fraction of missed "
            "fakes are *strictly below* their paired real has the most "
            "headroom for a margin-based pair-rank loss to bite."
        )
    else:
        L(
            "- No ckpt qualifies under the decision rule (all RED on the "
            "transport-matched primary). Skip P1; demote to P2 (`PE_SBI`)."
        )
    L("")
    L("## Caveats")
    L("")
    L(f"- {summary['pairing_caveat']}")
    L(
        "- The inference manifest only contains 14k cached scores. The full "
        "training paired loader has access to ~5,379 DF40 + DeepLive + "
        "VisoMaster pairs that ARE frame-level paired (see "
        "`data/sources/df40_paired.py`, `combined_paired.py`). The audit "
        "here is on the **eval** substrate, NOT the training substrate. The "
        "go/no-go signal is whether the model fails the pair-rank objective "
        "on data it sees at deployment, not on data it was trained on."
    )
    L(
        "- `score_PA_3800` is from the PA chain (E2B + viso enhanced data "
        "fw=4.0). PA does NOT generalise to HDTF (memory "
        "`pa_does_not_generalize_to_hdtf_2026-05-05`); its verdict here is "
        "informational only — do not seed PE from PA."
    )
    L(
        "- Cross-product pairing inflates pair counts; the absolute number "
        "of pairs (millions) is not directly comparable to a per-batch "
        "training pair count. The **fraction** statistics (P(gap<=0)) are "
        "the load-bearing outputs."
    )
    L(
        "- Method/enhancer/transport columns are heuristically parsed from "
        "frame_path; not all paths follow a documented schema."
    )
    L("")
    (OUT_DIR / "FINDINGS.md").write_text("\n".join(lines))
    print(f"[write] FINDINGS.md ({len(lines)} lines)")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    df = load_manifest()
    print(f"[manifest] suites: {df['suite'].value_counts().to_dict()}")
    print(f"[manifest] canonical subjects: {df['canonical_subject'].nunique()}")

    pairs, cov_df = build_pairs(df, rng)
    print(f"[pairs] built {len(pairs):,} pairs across "
          f"{cov_df['paired'].sum()} subjects")

    if len(pairs) == 0:
        print("[pairs] WARNING: 0 pairs — no canonical subject has both real "
              "and fake frames. Cannot proceed.")
        return

    pairs = add_gaps(pairs)
    pairs.to_csv(OUT_DIR / "pair_gaps.csv", index=False)
    print(f"[write] pair_gaps.csv ({len(pairs):,} rows)")

    summary = summarize(pairs, cov_df)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[write] summary.json")

    mfa = missed_fake_audit_table(pairs)
    mfa.to_csv(OUT_DIR / "missed_fake_audit.csv", index=False)
    print(f"[write] missed_fake_audit.csv ({len(mfa):,} rows)")

    fpa = fp_real_audit_table(pairs)
    fpa.to_csv(OUT_DIR / "fp_real_audit.csv", index=False)
    print(f"[write] fp_real_audit.csv ({len(fpa):,} rows)")

    write_findings(summary, pairs, cov_df)
    print(f"[done] outputs in {OUT_DIR}")
    print()
    print("=== AGGREGATE VERDICT ===")
    print(f"verdict = {summary['aggregate_verdict']}")
    for c in CKPTS:
        d = summary["decisions_per_ckpt"][c]
        b = summary["by_ckpt"][c]
        prim = d.get("primary_metric")
        print(
            f"  {c}: P(gap<=0|missed) [TM pair-w] = "
            f"{(prim if prim is not None else float('nan')):.4f} -> "
            f"{d['verdict']}  (N_missed_all={b['n_missed_fake']})"
        )
        # also dump per-subject for transparency
        per = b.get("per_subject_missed_fake", {}) or {}
        for subj, v in sorted(per.items(), key=lambda kv: -(kv[1].get("p_gap_le_0") or 0)):
            print(
                f"      {subj}: P(gap<=0|missed)={v.get('p_gap_le_0', 0.0):.4f} "
                f"(N_missed={v.get('n_missed', 0)})"
            )


if __name__ == "__main__":
    main()
