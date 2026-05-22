"""CPU-2 Pair-loss re-verification on Slot A v2 step3500 (2026-05-23).

Goal: does the 2026-05-04 sign-of-effect refutation
(mean(raw) < mean(teams) on E2B viso fakes) still hold on Slot A v2 step3500?

Method:
1. Load the 275 paired (raw, teams) viso fake seq_ids from
   `analysis/pair_loss_effect_verification_2026-05-05/per_pair_analysis.csv`.
2. Resolve to GCS frame URIs using the visomaster_enhanced_macro_dev manifest
   that already lives in `analysis/face_pool_scorecard_2026-05-22/_tmp/`.
3. Score on Slot A v2 step3500 with the CLS-pool head (apples-to-apples).
4. Compute per-pair: mean(raw_pair_probs), mean(teams_pair_probs), Δ.
   Aggregate: overall mean(raw) vs mean(teams), Wilcoxon signed-rank test,
   per-pair cohort partition at τ ∈ {0.05, 0.10, 0.20, 0.50}.
5. Comparison table vs E2B summary.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from batch_inference_gcs import CLIP_MEAN, CLIP_STD, load_model  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
DETECTOR_CFG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CFG = REPO_ROOT / "config/train_config.yaml"

CKPT_LOCAL = (
    REPO_ROOT / "analysis/manual_canary_2026-05-20/ckpts/"
    "periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth"
)

PAIR_CSV = (
    REPO_ROOT / "analysis/pair_loss_effect_verification_2026-05-05/per_pair_analysis.csv"
)
E2B_SUMMARY_JSON = (
    REPO_ROOT / "analysis/pair_loss_effect_verification_2026-05-05/summary.json"
)
VISO_MANIFEST = (
    REPO_ROOT / "analysis/face_pool_scorecard_2026-05-22/_tmp/"
    "visomaster_enhanced_macro_dev_frames.csv"
)

RESOLUTION = 224

TAU_GRID = [0.05, 0.10, 0.20, 0.50]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / "_cpu2.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("cpu2")


FNAME_RE = re.compile(r"visomaster_enhanced_(raw|teams)__frame_(\d+)_seq(\d+)\.png")


def parse_frame_path(p: str):
    fname = p.rsplit("/", 1)[-1]
    m = FNAME_RE.match(fname)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def _parse_gs_uri(uri: str):
    s = uri.removeprefix("gs://")
    bucket, _, blob = s.partition("/")
    return bucket, blob


def build_pair_table() -> pd.DataFrame:
    """Map seq_id -> (raw_uri, teams_uri) using the manifest."""
    pair_df = pd.read_csv(PAIR_CSV)
    seq_ids = pair_df["seq_id"].astype(int).tolist()
    logger.info("pair csv: %d rows, %d unique seq_ids", len(pair_df), len(set(seq_ids)))

    manifest = pd.read_csv(VISO_MANIFEST)
    by_seq: Dict[int, Dict[str, str]] = {}
    for _, row in manifest.iterrows():
        rec = parse_frame_path(str(row["frame_path"]))
        if rec is None:
            continue
        subtype, _frame_num, seq_id = rec
        if seq_id not in by_seq:
            by_seq[seq_id] = {}
        by_seq[seq_id][subtype] = str(row["frame_path"])

    rows: List[Dict] = []
    missing = 0
    for seq_id in seq_ids:
        d = by_seq.get(seq_id, {})
        if "raw" in d and "teams" in d:
            rows.append({
                "seq_id": seq_id,
                "raw_uri": d["raw"],
                "teams_uri": d["teams"],
            })
        else:
            missing += 1
            logger.warning("seq %d missing raw or teams in manifest", seq_id)
    df = pd.DataFrame(rows)
    logger.info(
        "built pair table: %d / %d pairs resolvable (%d missing)",
        len(df), len(seq_ids), missing,
    )
    return df


def stream_frames(uris: List[str]) -> Tuple[torch.Tensor, np.ndarray]:
    """Stream a list of GCS URIs into a tensor; return (tensor, valid_mask)."""
    from google.cloud import storage
    from torchvision import transforms as T

    normalize = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    client = storage.Client()
    bucket_cache: Dict[str, object] = {}
    n = len(uris)
    out = torch.zeros(n, 3, RESOLUTION, RESOLUTION, dtype=torch.float32)
    valid = np.zeros(n, dtype=bool)
    t0 = time.time()
    fails = 0
    for i, uri in enumerate(uris):
        bucket_name, blob_path = _parse_gs_uri(uri)
        if bucket_name not in bucket_cache:
            bucket_cache[bucket_name] = client.bucket(bucket_name)
        try:
            blob = bucket_cache[bucket_name].blob(blob_path)
            img_bytes = blob.download_as_bytes()
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                fails += 1
                continue
            img_bgr = cv2.resize(img_bgr, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            out[i] = normalize(img_rgb)
            valid[i] = True
        except Exception as e:
            fails += 1
            if fails <= 3:
                logger.warning("frame %d (%s) failed: %s", i, uri, e)
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d streamed (%.1fs, %d fails)", i + 1, n, time.time() - t0, fails)
    logger.info("[stream] %d/%d valid frames in %.1fs", int(valid.sum()), n, time.time() - t0)
    return out, valid


@torch.no_grad()
def score_frames(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device,
                 batch_size: int = 16) -> np.ndarray:
    model.eval()
    n = tensor.shape[0]
    probs: List[float] = []
    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = tensor[start:end].to(device, non_blocking=True)
        out = model({"image": batch}, inference=True)
        if isinstance(out, dict):
            if "prob" in out:
                p = out["prob"]
            else:
                logits = None
                for k in ("cls", "raw_logits", "logits"):
                    if k in out:
                        logits = out[k]
                        break
                if logits.dim() == 3:
                    logits = logits.mean(dim=1)
                p = torch.softmax(logits, dim=-1)[:, 1]
        else:
            p = torch.softmax(out, dim=-1)[:, 1]
        probs.extend(p.float().cpu().tolist())
        if (start // batch_size) % 5 == 0:
            logger.info("  scoring %d/%d (%.1fs)", end, n, time.time() - t0)
    return np.asarray(probs, dtype=np.float64)


def cohort_partition(raw_probs: np.ndarray, teams_probs: np.ndarray, tau: float) -> Dict[str, object]:
    """target = teams > τ AND raw <= τ
       wrong_way = raw > τ AND teams <= τ
       both_caught = both > τ
       both_missed = both <= τ
    """
    target_mask = (teams_probs > tau) & (raw_probs <= tau)
    wrong_mask = (raw_probs > tau) & (teams_probs <= tau)
    both_caught = (raw_probs > tau) & (teams_probs > tau)
    both_missed = (raw_probs <= tau) & (teams_probs <= tau)
    return {
        "tau": tau,
        "target_n": int(target_mask.sum()),
        "wrong_way_n": int(wrong_mask.sum()),
        "both_caught_n": int(both_caught.sum()),
        "both_missed_n": int(both_missed.sum()),
        "target_mean_raw": float(raw_probs[target_mask].mean()) if target_mask.any() else float("nan"),
        "target_mean_teams": float(teams_probs[target_mask].mean()) if target_mask.any() else float("nan"),
        "wrong_way_mean_raw": float(raw_probs[wrong_mask].mean()) if wrong_mask.any() else float("nan"),
        "wrong_way_mean_teams": float(teams_probs[wrong_mask].mean()) if wrong_mask.any() else float("nan"),
    }


def compute_verdict(slot_mean_raw: float, slot_mean_teams: float,
                    cohorts: List[Dict]) -> Tuple[str, str]:
    """alpha: sign reversed (mean(raw) > mean(teams)) AND target cohort > wrong-way at τ=0.20
       beta: sign same direction as E2B (mean(raw) < mean(teams)) → pair-loss refuted
       gamma: signs cancel (|Δ| < 0.02) → indeterminate
    """
    delta = slot_mean_teams - slot_mean_raw  # if positive → teams > raw, same as E2B (premise refuted)
    cohort_tau_020 = next(c for c in cohorts if c["tau"] == 0.20)
    if abs(delta) < 0.02:
        verdict = "gamma"
        summary = (
            f"signs cancel: mean(teams)-mean(raw) = {delta:+.4f} (|Δ|<0.02); "
            f"BACKBONE runs with reduced λ_pair=0.15"
        )
        return verdict, summary
    if slot_mean_raw > slot_mean_teams and cohort_tau_020["target_n"] > cohort_tau_020["wrong_way_n"]:
        verdict = "alpha"
        summary = (
            f"sign reversed on Slot A v2: mean(raw)={slot_mean_raw:.4f} > mean(teams)={slot_mean_teams:.4f} "
            f"(Δ={delta:+.4f}); target_τ020={cohort_tau_020['target_n']} > wrong_way_τ020={cohort_tau_020['wrong_way_n']} "
            f"→ pair-loss fulcrum exists on Slot A v2 reals"
        )
        return verdict, summary
    if slot_mean_raw < slot_mean_teams:
        verdict = "beta"
        summary = (
            f"sign same direction as E2B on Slot A v2: mean(raw)={slot_mean_raw:.4f} < mean(teams)={slot_mean_teams:.4f} "
            f"(Δ={delta:+.4f}); pair-loss premise does not hold; pivot to GroupDRO substrate-balanced"
        )
        return verdict, summary
    # alpha condition partial (sign reversed but cohort tied) -> gamma
    verdict = "gamma"
    summary = (
        f"sign reversed but cohort partition: target_τ020={cohort_tau_020['target_n']} "
        f"vs wrong_way_τ020={cohort_tau_020['wrong_way_n']}; deferred to gamma"
    )
    return verdict, summary


def write_facts_doc(pair_table: pd.DataFrame, raw_probs: np.ndarray, teams_probs: np.ndarray,
                    cohorts: List[Dict], e2b_summary: Dict,
                    wilcoxon_stat: float, wilcoxon_p: float,
                    wall_seconds: float, verdict: str, summary: str,
                    out_path: Path) -> None:
    mean_raw = float(raw_probs.mean())
    mean_teams = float(teams_probs.mean())
    median_raw = float(np.median(raw_probs))
    median_teams = float(np.median(teams_probs))
    n = len(raw_probs)
    e2b_raw = e2b_summary["sign_of_effect"]["e2b_mean_raw"]
    e2b_teams = e2b_summary["sign_of_effect"]["e2b_mean_teams"]
    e2b_wstat = e2b_summary["sign_of_effect"]["e2b_wilcoxon_teams_vs_raw_stat"]
    e2b_wp = e2b_summary["sign_of_effect"]["e2b_wilcoxon_teams_vs_raw_p"]
    p8a_raw = e2b_summary["sign_of_effect"]["p8a_mean_raw"]
    p8a_teams = e2b_summary["sign_of_effect"]["p8a_mean_teams"]

    lines = []
    lines.append("# CPU-2 Pair-Loss Re-Verification on Slot A v2 — FACTS (2026-05-23)")
    lines.append("")
    lines.append("> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, ship, kill, best, worst, unfortunately, remarkably, lucky, confirmed, refuted, shortcut-aligned, gap-is-wide, gap-is-narrow.")
    lines.append("")
    lines.append("## 1. Method")
    lines.append("")
    lines.append(f"- Checkpoint: Slot A v2 step3500 (`{CKPT_LOCAL.name}`).")
    lines.append(f"- Source: 275 paired (raw, teams) viso fake seq_ids from `analysis/pair_loss_effect_verification_2026-05-05/per_pair_analysis.csv`.")
    lines.append(f"- Frame URIs resolved via `analysis/face_pool_scorecard_2026-05-22/_tmp/visomaster_enhanced_macro_dev_frames.csv` (275 raw + 275 teams = 550 frames).")
    lines.append(f"- Scoring: CLS-pool head readout (apples-to-apples with E2B 2026-05-04 analysis).")
    lines.append(f"- N pairs scored = {n}.")
    lines.append(f"- Wall time: {wall_seconds:.0f}s.")
    lines.append("")
    lines.append("## 2. Headline table — Slot A v2 vs E2B vs P8A")
    lines.append("")
    lines.append("| Model | mean(raw) | mean(teams) | Δ (teams−raw) | direction |")
    lines.append("|---|---:|---:|---:|---|")
    lines.append(
        f"| Slot A v2 step3500 (new) | {mean_raw:.4f} | {mean_teams:.4f} | "
        f"{mean_teams - mean_raw:+.4f} | "
        f"{'teams > raw' if mean_teams > mean_raw else 'raw > teams' if mean_raw > mean_teams else 'equal'} |"
    )
    lines.append(f"| E2B step3200 (2026-05-04) | {e2b_raw:.4f} | {e2b_teams:.4f} | {e2b_teams - e2b_raw:+.4f} | teams > raw |")
    lines.append(f"| P8A step5000 (2026-05-04) | {p8a_raw:.4f} | {p8a_teams:.4f} | {p8a_teams - p8a_raw:+.4f} | raw > teams |")
    lines.append("")
    lines.append(f"Median(raw) = {median_raw:.4f}, median(teams) = {median_teams:.4f}.")
    lines.append(f"Wilcoxon signed-rank (teams vs raw): statistic = {wilcoxon_stat:.1f}, p = {wilcoxon_p:.4g}.")
    lines.append(f"(E2B for reference: statistic = {e2b_wstat}, p = {e2b_wp:.4g}.)")
    lines.append("")
    lines.append("## 3. Per-pair cohort partition")
    lines.append("")
    lines.append("target cohort = teams > τ AND raw ≤ τ (pair-loss would help)")
    lines.append("wrong_way cohort = raw > τ AND teams ≤ τ (pair-loss would hurt)")
    lines.append("")
    lines.append("| τ | target | wrong_way | both_caught | both_missed | target Δ | wrong_way Δ |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for c in cohorts:
        t_delta = c["target_mean_teams"] - c["target_mean_raw"] if not np.isnan(c["target_mean_raw"]) else float("nan")
        w_delta = c["wrong_way_mean_teams"] - c["wrong_way_mean_raw"] if not np.isnan(c["wrong_way_mean_raw"]) else float("nan")
        lines.append(
            f"| {c['tau']:.2f} | {c['target_n']} | {c['wrong_way_n']} | "
            f"{c['both_caught_n']} | {c['both_missed_n']} | "
            f"{t_delta:+.4f} | {w_delta:+.4f} |"
        )
    lines.append("")
    lines.append("Comparison to E2B Q3_threshold_sensitivity_e2b_symmetric (raw_caught_teams_missed = target; teams_caught_raw_missed = wrong_way):")
    lines.append("")
    e2b_thresh = e2b_summary.get("Q3_threshold_sensitivity_e2b_symmetric", {})
    lines.append("| τ | E2B target | E2B wrong_way |")
    lines.append("|---:|---:|---:|")
    for tau_key, tau_val in (("tau_0.05", 0.05), ("tau_0.1", 0.10), ("tau_0.2", 0.20), ("tau_0.5", 0.50)):
        if tau_key in e2b_thresh:
            d = e2b_thresh[tau_key]
            lines.append(f"| {tau_val:.2f} | {d['n_cohort_raw_caught_teams_missed']} | {d['n_opposite_teams_caught_raw_missed']} |")
    lines.append("")
    lines.append("**Note on cohort label semantics:** the 2026-05-04 doc labelled `target` as `raw_caught_teams_missed` (because that paper's pair-loss premise was raw→teams alignment). In the current plan the pair-loss is teams→raw alignment, so `target` = `teams_caught_raw_missed` here. This re-verification reports BOTH cohorts. The decision rule applies to the asymmetry between them at τ=0.20.")
    lines.append("")
    lines.append("## 4. Close criterion verdict")
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append(f"Summary: {summary}")
    lines.append("")
    lines.append("Decision rule:")
    lines.append("- alpha: sign reversed (mean(raw) > mean(teams)) AND target cohort > wrong_way cohort at τ=0.20 → pair-loss fulcrum exists → BACKBONE proceeds with substrate-pair-orthogonal loss")
    lines.append("- beta: sign same direction as E2B (mean(raw) < mean(teams)) → pair-loss premise on Slot A v2 too → BACKBONE pivots to GroupDRO substrate-balanced")
    lines.append("- gamma: signs cancel (|Δ| < 0.02) → indeterminate; BACKBONE runs with reduced λ_pair=0.15")
    lines.append("")
    lines.append("## 5. Output artifacts")
    lines.append("")
    lines.append(f"- `pair_scores_slot_a_v2.csv` — per-pair scores")
    lines.append(f"- `RESULTS_FACTS_2026-05-23.md` — this file")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("device=%s", device)

    # Build pair table
    pair_table = build_pair_table()
    n = len(pair_table)
    if n == 0:
        logger.error("no pairs resolvable — aborting")
        return 2

    # Stream raw + teams frames
    raw_uris = pair_table["raw_uri"].tolist()
    teams_uris = pair_table["teams_uri"].tolist()
    cache_path = OUT_DIR / "_cache_pair_frames.pt"
    cache_meta = OUT_DIR / "_cache_pair_meta.parquet"
    if cache_path.exists() and cache_meta.exists():
        logger.info("[cache hit] %s", cache_path)
        all_tensor = torch.load(cache_path, map_location="cpu")
        if all_tensor.dtype != torch.float32:
            all_tensor = all_tensor.float()
        cached_meta = pd.read_parquet(cache_meta)
        # sanity check that order matches
        if len(cached_meta) == 2 * n:
            n_valid_raw = cached_meta.iloc[:n]["valid"].sum()
            n_valid_teams = cached_meta.iloc[n:]["valid"].sum()
            logger.info("[cache] n_valid_raw=%d, n_valid_teams=%d", n_valid_raw, n_valid_teams)
            raw_tensor = all_tensor[:n]
            teams_tensor = all_tensor[n:]
            raw_valid = cached_meta.iloc[:n]["valid"].to_numpy()
            teams_valid = cached_meta.iloc[n:]["valid"].to_numpy()
        else:
            logger.warning("cache shape mismatch; rebuilding")
            cache_path.unlink()
            return main()  # rerun
    else:
        logger.info("[stream] raw side")
        raw_tensor, raw_valid = stream_frames(raw_uris)
        logger.info("[stream] teams side")
        teams_tensor, teams_valid = stream_frames(teams_uris)
        all_tensor = torch.cat([raw_tensor, teams_tensor], dim=0)
        meta_rows = []
        for i in range(n):
            meta_rows.append({"side": "raw", "seq_id": int(pair_table.iloc[i]["seq_id"]),
                              "uri": raw_uris[i], "valid": bool(raw_valid[i])})
        for i in range(n):
            meta_rows.append({"side": "teams", "seq_id": int(pair_table.iloc[i]["seq_id"]),
                              "uri": teams_uris[i], "valid": bool(teams_valid[i])})
        cached_meta = pd.DataFrame(meta_rows)
        cached_meta.to_parquet(cache_meta)
        torch.save(all_tensor.half(), cache_path)
        logger.info("[cache] saved %s", cache_path)

    # Load model
    logger.info("loading model")
    t_m = time.time()
    model = load_model(str(CKPT_LOCAL), str(DETECTOR_CFG), str(TRAIN_CFG), device)
    logger.info("  loaded in %.1fs", time.time() - t_m)

    # Score
    logger.info("scoring raw side")
    raw_probs = score_frames(model, raw_tensor, device, args.batch_size)
    logger.info("scoring teams side")
    teams_probs = score_frames(model, teams_tensor, device, args.batch_size)

    # Only keep pairs where both sides decoded
    both_valid = raw_valid & teams_valid
    n_valid = int(both_valid.sum())
    logger.info("kept %d / %d pairs (both sides valid)", n_valid, n)
    raw_probs = raw_probs[both_valid]
    teams_probs = teams_probs[both_valid]
    pair_table_valid = pair_table[both_valid].reset_index(drop=True)

    # Per-pair CSV
    out_csv = OUT_DIR / "pair_scores_slot_a_v2.csv"
    pd.DataFrame({
        "seq_id": pair_table_valid["seq_id"].values,
        "raw_uri": pair_table_valid["raw_uri"].values,
        "teams_uri": pair_table_valid["teams_uri"].values,
        "raw_prob": raw_probs,
        "teams_prob": teams_probs,
        "delta_teams_minus_raw": teams_probs - raw_probs,
    }).to_csv(out_csv, index=False)
    logger.info("wrote %s", out_csv)

    # Aggregate metrics
    mean_raw = float(raw_probs.mean())
    mean_teams = float(teams_probs.mean())
    logger.info("mean(raw)=%.4f mean(teams)=%.4f delta=%.4f",
                mean_raw, mean_teams, mean_teams - mean_raw)

    # Wilcoxon signed-rank on teams vs raw
    try:
        ws_res = stats.wilcoxon(teams_probs, raw_probs, zero_method="wilcox", alternative="two-sided")
        ws_stat, ws_p = float(ws_res.statistic), float(ws_res.pvalue)
    except Exception as e:
        logger.warning("Wilcoxon failed: %s", e)
        ws_stat, ws_p = float("nan"), float("nan")

    # Cohort partition at τ ∈ {0.05, 0.10, 0.20, 0.50}
    cohorts = [cohort_partition(raw_probs, teams_probs, tau) for tau in TAU_GRID]
    for c in cohorts:
        logger.info("  τ=%.2f target=%d wrong_way=%d both_caught=%d both_missed=%d",
                    c["tau"], c["target_n"], c["wrong_way_n"], c["both_caught_n"], c["both_missed_n"])

    # E2B summary
    with open(E2B_SUMMARY_JSON) as f:
        e2b_summary = json.load(f)

    # Verdict
    verdict, summary = compute_verdict(mean_raw, mean_teams, cohorts)
    logger.info("VERDICT: %s — %s", verdict, summary)

    wall = time.time() - t0
    write_facts_doc(
        pair_table_valid, raw_probs, teams_probs, cohorts, e2b_summary,
        ws_stat, ws_p, wall, verdict, summary,
        OUT_DIR / "RESULTS_FACTS_2026-05-23.md",
    )

    sentinel = OUT_DIR / "_cpu2_complete.json"
    with open(sentinel, "w") as f:
        json.dump({
            "status": "done",
            "wall_seconds": int(wall),
            "verdict": verdict,
            "verdict_summary": summary,
            "n_pairs": n_valid,
            "mean_raw": mean_raw,
            "mean_teams": mean_teams,
            "delta_teams_minus_raw": mean_teams - mean_raw,
            "wilcoxon_stat": ws_stat,
            "wilcoxon_p": ws_p,
            "cohorts": cohorts,
        }, f, indent=2)
    logger.info("DONE %.1fs", wall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
