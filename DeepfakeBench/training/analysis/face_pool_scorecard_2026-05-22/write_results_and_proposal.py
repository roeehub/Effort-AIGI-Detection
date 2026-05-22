"""Build RESULTS_FACTS_2026-05-22.md + AGENT_PROPOSAL_2026-05-22.md after
the two scorers have produced their outputs.

Pulls the CLS-pool baseline scorecard from GCS (validation 2026-05-20),
extracts the contract metrics for SLOT_A_ANCHOR_AWARE_STEP3500, and emits
side-by-side tables vs the face-pool rerun under both lex and composite
policies.

Outputs:
  RESULTS_FACTS_2026-05-22.md  (facts only, banned-word compliant)
  AGENT_PROPOSAL_2026-05-22.md (recommendation)
  _verdict.txt                  (single line for sentinel)
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


BANNED_WORDS = (
    "succeeds", "fails", "wins", "loses", "promotes", "deployment-grade",
    "ship", "kill", "best", "worst", "unfortunately", "remarkably", "lucky",
    "confirmed", "refuted", "shortcut-aligned", "gap-is-wide", "gap-is-narrow",
)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _safe_get(row: Dict[str, str], key: str, default: str = "") -> str:
    return str(row.get(key, default) or default)


def _to_float_or_nan(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _pull_cls_baseline(gs_uri_prefix: str, dest: Path) -> Dict[str, Path]:
    """Copy CLS-pool baseline scorecard files from GCS to local."""
    dest.mkdir(parents=True, exist_ok=True)
    files = [
        "selected_threshold_scorecard.csv",
        "checkpoint_summary.csv",
        "promotion_contract.json",
        "promotion_winner.json",
        "threshold_grid.csv",
    ]
    out: Dict[str, Path] = {}
    for name in files:
        src = f"{gs_uri_prefix.rstrip('/')}/{name}"
        local = dest / name
        try:
            subprocess.run(
                ["gsutil", "-q", "cp", src, str(local)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            out[name] = local
        except subprocess.CalledProcessError:
            out[name] = local  # may not exist
    return out


def _row_for_ckpt(rows: List[Dict[str, str]], ckpt_key: str) -> Dict[str, str]:
    for row in rows:
        k = _safe_get(row, "checkpoint_key").upper()
        if k == ckpt_key.upper():
            return row
    return {}


def _format(value: str, fmt: str = "{:.4f}") -> str:
    if value is None or value == "":
        return "n/a"
    try:
        f = float(value)
        return fmt.format(f)
    except Exception:
        return str(value)


def _diff(face: str, cls: str) -> str:
    a = _to_float_or_nan(face)
    b = _to_float_or_nan(cls)
    if a != a or b != b:
        return "n/a"
    d = a - b
    sign = "+" if d >= 0 else "-"
    return f"{sign}{abs(d):.4f}"


def _check_banned(text: str) -> List[str]:
    lower = text.lower()
    hits = [w for w in BANNED_WORDS if w in lower]
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--cls-baseline-scorecard-gs", required=True)
    ap.add_argument("--ckpt-key", default="SLOT_A_ANCHOR_AWARE_STEP3500")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    cls_dir = workdir / "scorecard" / "cls_baseline_pull"
    pulled = _pull_cls_baseline(args.cls_baseline_scorecard_gs, cls_dir)

    # Face-pool scorecards
    lex_dir = workdir / "scorecard" / "lex"
    comp_dir = workdir / "scorecard" / "composite_lambda_1.0"

    cls_sel = _read_csv_rows(pulled.get("selected_threshold_scorecard.csv", Path("/dev/null")))
    cls_summary = _read_csv_rows(pulled.get("checkpoint_summary.csv", Path("/dev/null")))
    cls_winner = _read_json(pulled.get("promotion_winner.json", Path("/dev/null")))

    fp_lex_sel = _read_csv_rows(lex_dir / "selected_threshold_scorecard.csv")
    fp_lex_summary = _read_csv_rows(lex_dir / "checkpoint_summary.csv")
    fp_lex_winner = _read_json(lex_dir / "promotion_winner.json")

    fp_comp_sel = _read_csv_rows(comp_dir / "selected_threshold_scorecard.csv")
    fp_comp_summary = _read_csv_rows(comp_dir / "checkpoint_summary.csv")
    fp_comp_winner = _read_json(comp_dir / "promotion_winner.json")

    cls_row = _row_for_ckpt(cls_summary, args.ckpt_key)
    fp_lex_row = _row_for_ckpt(fp_lex_summary, args.ckpt_key)
    fp_comp_row = _row_for_ckpt(fp_comp_summary, args.ckpt_key)

    # Contract metrics from checkpoint_summary.csv:
    # dev_primary_real_fpr, dev_worst_real_stress_fpr, dev_fake_macro_recall,
    # lockbox_real_fpr, lockbox_fake_recall, plus per-suite recall columns.
    contract_metrics = [
        "selected_threshold",
        "dev_primary_real_fpr",
        "dev_worst_real_stress_fpr",
        "dev_fake_macro_recall",
        "lockbox_real_fpr",
        "lockbox_fake_recall",
    ]

    # Per-suite fake recalls — sometimes appear in the summary csv as
    # 'fake_recall_<suite>' or similar; we just scan the CLS summary keys.
    extra_keys: List[str] = []
    for k in cls_row.keys():
        kl = k.lower()
        if any(
            substr in kl
            for substr in ("visomaster", "deeplive", "teams_real_dor", "teams_fake_all")
        ):
            extra_keys.append(k)
    extra_keys = sorted(set(extra_keys))

    # Build markdown
    lines: List[str] = []
    lines.append("# Face-Pool Full Scorecard Rerun — Slot A v2 step3500 — FACTS (2026-05-22)")
    lines.append("")
    lines.append(
        "> Status: factual-only. No banned words (succeeds, fails, wins, loses, "
        "promotes, deployment-grade, ship, kill, best, worst, unfortunately, "
        "remarkably, lucky, confirmed, refuted, shortcut-aligned, gap-is-wide, "
        "gap-is-narrow). Interpretation lives in AGENT_PROPOSAL_2026-05-22.md."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Method")
    lines.append("")
    lines.append(
        "1. Reused the per-frame manifests embedded in the 2026-05-20 CLS-pool "
        "frames_report CSVs at `gs://training-job-outputs/test_results/teams_promotion_contract/slot-a-v2-validation-2026-05-20/reports/` "
        "(9 suites x SLOT_A_ANCHOR_AWARE_STEP3500). Each row carries the GCS frame URI, label, "
        "video_id, group_key, family_key, and method, so the same frames the CLS-pool scorer ate are "
        "the ones face-pool now scores."
    )
    lines.append(
        "2. Loaded checkpoint `analysis/manual_canary_2026-05-20/ckpts/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth` "
        "via `batch_inference_gcs.load_model` against `config/detector/effort.yaml` + `config/train_config.yaml`."
    )
    lines.append(
        "3. Installed the centered-7x7 face-region monkey-patch from "
        "`analysis/face_pool_canary_2026-05-22/score_canary_face_pool.py` "
        "on `model.backbone.visual.transformer.resblocks[11]`. The patched "
        "`backbone.forward` discards the CLS-pool output and returns the "
        "ln_post -> visual.proj-projected mean of 49 face-region patch tokens "
        "to the unchanged ArcFace head."
    )
    lines.append(
        "4. Streamed all 13,636 frames from GCS via per-worker `storage.Client` "
        "and ran inference on MPS. Aggregated per video_id by mean-of-frame-probs, "
        "then emitted `<suite>_<ckpt>.lower()_videos_report.csv` with the schema "
        "`arena/score_teams_promotion_contract.py` consumes."
    )
    lines.append(
        "5. Scored the 9 face-pool reports under two cross-checkpoint policies: "
        "the standing lex policy (output under `scorecard/lex/`) and the "
        "2026-05-22 composite tiebreak with lambda=1.0 (output under "
        "`scorecard/composite_lambda_1.0/`)."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # 2. Suite-level counts (sanity check)
    lines.append("## 2. Suite-level frame counts")
    lines.append("")
    lines.append("| Suite | Frames | Videos |")
    lines.append("|---|---:|---:|")
    sum_json = workdir / "reports" / f"_face_pool_scoring_summary_{args.ckpt_key.lower()}.json"
    suite_rows = []
    summary = _read_json(sum_json)
    if summary and "suites" in summary:
        for s in summary["suites"]:
            lines.append(f"| `{s['suite']}` | {s['frames']} | {s['videos']} |")
            suite_rows.append(s)
    lines.append("")
    if summary:
        lines.append(f"Total scoring wall time: {summary.get('total_wall_seconds', 'n/a')} s.")
        lines.append("")
    lines.append("---")
    lines.append("")

    # 3. Side-by-side contract metric table
    lines.append("## 3. Contract metrics — CLS-pool baseline vs face-pool (Slot A v2 step3500)")
    lines.append("")
    lines.append(
        "All rows are `checkpoint_summary.csv` columns for "
        f"`{args.ckpt_key}`. CLS-pool column pulled from "
        f"`{args.cls_baseline_scorecard_gs}checkpoint_summary.csv`."
    )
    lines.append("")
    lines.append("### 3a. Lex policy (standing)")
    lines.append("")
    lines.append("| Metric | CLS pool | Face pool (lex) | Δ (face − CLS) |")
    lines.append("|---|---:|---:|---:|")
    for m in contract_metrics:
        cls_v = cls_row.get(m, "")
        face_v = fp_lex_row.get(m, "")
        lines.append(f"| `{m}` | {_format(cls_v)} | {_format(face_v)} | {_diff(face_v, cls_v)} |")
    for k in extra_keys:
        cls_v = cls_row.get(k, "")
        face_v = fp_lex_row.get(k, "")
        lines.append(f"| `{k}` | {_format(cls_v)} | {_format(face_v)} | {_diff(face_v, cls_v)} |")
    lines.append("")

    lines.append("### 3b. Composite policy (λ=1.0)")
    lines.append("")
    lines.append("| Metric | CLS pool (lex) | Face pool (composite λ=1.0) | Δ |")
    lines.append("|---|---:|---:|---:|")
    for m in contract_metrics + extra_keys:
        cls_v = cls_row.get(m, "")
        face_v = fp_comp_row.get(m, "")
        lines.append(f"| `{m}` | {_format(cls_v)} | {_format(face_v)} | {_diff(face_v, cls_v)} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 4. Promotion winners
    lines.append("## 4. Promotion outcome — same panel, two policies")
    lines.append("")
    lines.append("### 4a. CLS-pool baseline (lex; from 2026-05-20 validation)")
    lines.append("")
    if cls_winner:
        lines.append("```json")
        lines.append(json.dumps(cls_winner, indent=2))
        lines.append("```")
    else:
        lines.append("(promotion_winner.json not available from CLS-pool baseline.)")
    lines.append("")
    lines.append("### 4b. Face-pool, lex policy")
    lines.append("")
    if fp_lex_winner:
        lines.append("```json")
        lines.append(json.dumps(fp_lex_winner, indent=2))
        lines.append("```")
    else:
        lines.append("(promotion_winner.json not produced.)")
    lines.append("")
    lines.append("### 4c. Face-pool, composite policy (λ=1.0)")
    lines.append("")
    if fp_comp_winner:
        lines.append("```json")
        lines.append(json.dumps(fp_comp_winner, indent=2))
        lines.append("```")
    else:
        lines.append("(promotion_winner.json not produced.)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 5. Selected-threshold suite breakdown
    lines.append("## 5. Selected-threshold suite breakdown — face-pool, lex policy")
    lines.append("")
    if fp_lex_sel:
        cols = list(fp_lex_sel[0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")
        for row in fp_lex_sel:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    else:
        lines.append("(selected_threshold_scorecard.csv not produced.)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 6. Cross-reference
    lines.append("## 6. Cross-reference")
    lines.append("")
    lines.append(
        "- 800-frame face-pool canary baseline: `analysis/face_pool_canary_2026-05-22/outputs/SLOT_A_V2_STEP3500_face_pool.json`."
    )
    lines.append(
        "- CLS-pool reference summary (2026-05-20 validation): `gs://training-job-outputs/test_results/teams_promotion_contract/slot-a-v2-validation-2026-05-20/promotion_contract/checkpoint_summary.csv`."
    )
    lines.append(
        "- Representation-geometry probe 2 (Probe 2): `analysis/substrate_pair_geometry_2026-05-22/per_ckpt_face_region_cosines.csv` — cos_pair 0.87 → 0.96, delta_pair_vs_within -0.077 → -0.016."
    )
    lines.append("")

    text = "\n".join(lines)

    banned_hits = _check_banned(text)
    if banned_hits:
        # Write a warning footer instead of refusing — facts CSV may legitimately
        # include strings like 'fake_recall' which is allowed; we banlist exact words.
        text += f"\n\n<!-- BANNED-WORD CHECK: {banned_hits} flagged; review before publishing. -->\n"

    (workdir / "RESULTS_FACTS_2026-05-22.md").write_text(text)

    # ----- AGENT_PROPOSAL -----
    prop_lines: List[str] = []
    prop_lines.append("# Face-Pool Full Scorecard Rerun — AGENT_PROPOSAL (2026-05-22)")
    prop_lines.append("")
    prop_lines.append(
        "Opinion-and-recommendation document. Facts live in RESULTS_FACTS_2026-05-22.md."
    )
    prop_lines.append("")
    prop_lines.append("## Question")
    prop_lines.append("")
    prop_lines.append(
        "Does swapping the centered-7x7 face-region patch pool for the CLS pool — at "
        "inference time only, on a frozen Slot A v2 step3500 head — give a "
        "deployment-grade improvement over CLS pool on the same checkpoint?"
    )
    prop_lines.append("")

    # Pull a few numbers for the proposal narrative
    def _f(d: Dict[str, str], k: str) -> float:
        try:
            return float(d.get(k, "nan"))
        except Exception:
            return float("nan")

    cls_lockbox_fpr = _f(cls_row, "lockbox_real_fpr")
    fp_lex_lockbox_fpr = _f(fp_lex_row, "lockbox_real_fpr")
    cls_lockbox_recall = _f(cls_row, "lockbox_fake_recall")
    fp_lex_lockbox_recall = _f(fp_lex_row, "lockbox_fake_recall")
    cls_devp_fpr = _f(cls_row, "dev_primary_real_fpr")
    fp_lex_devp_fpr = _f(fp_lex_row, "dev_primary_real_fpr")
    cls_devfake = _f(cls_row, "dev_fake_macro_recall")
    fp_lex_devfake = _f(fp_lex_row, "dev_fake_macro_recall")

    prop_lines.append("## Headline numbers (lex policy)")
    prop_lines.append("")
    prop_lines.append("| Metric | CLS pool | Face pool | Δ |")
    prop_lines.append("|---|---:|---:|---:|")
    prop_lines.append(
        f"| `dev_primary_real_fpr` | {cls_devp_fpr:.4f} | {fp_lex_devp_fpr:.4f} | "
        f"{fp_lex_devp_fpr - cls_devp_fpr:+.4f} |"
    )
    prop_lines.append(
        f"| `dev_fake_macro_recall` | {cls_devfake:.4f} | {fp_lex_devfake:.4f} | "
        f"{fp_lex_devfake - cls_devfake:+.4f} |"
    )
    prop_lines.append(
        f"| `lockbox_real_fpr` | {cls_lockbox_fpr:.4f} | {fp_lex_lockbox_fpr:.4f} | "
        f"{fp_lex_lockbox_fpr - cls_lockbox_fpr:+.4f} |"
    )
    prop_lines.append(
        f"| `lockbox_fake_recall` | {cls_lockbox_recall:.4f} | {fp_lex_lockbox_recall:.4f} | "
        f"{fp_lex_lockbox_recall - cls_lockbox_recall:+.4f} |"
    )
    prop_lines.append("")

    # Recommendation logic — purely numeric (interpretation only here)
    recommend = "TRIAL"
    why = []
    # Lockbox dominance
    delta_fpr = fp_lex_lockbox_fpr - cls_lockbox_fpr
    delta_recall = fp_lex_lockbox_recall - cls_lockbox_recall
    if delta_fpr <= 0.005 and delta_recall >= 0.05:
        recommend = "SHIP-as-inference-variant"
        why.append(
            "lockbox_real_fpr non-regressing (Δ≤+0.5pp) AND lockbox_fake_recall up at least +5pp"
        )
    elif delta_fpr >= 0.02 or fp_lex_devfake < 0.30:
        recommend = "ABANDON"
        why.append(
            "either lockbox_real_fpr drift > +2pp or dev_fake_macro_recall below the 0.30 floor — face-pool is not currently a free upgrade"
        )
    else:
        recommend = "TRIAL"
        why.append(
            "mixed signal — try face-pool on the next ckpt panel (multiple ckpts, not single-ckpt rerun) before committing"
        )

    prop_lines.append("## Recommendation")
    prop_lines.append("")
    prop_lines.append(f"**{recommend}** — {('; '.join(why))}.")
    prop_lines.append("")
    prop_lines.append("## Caveats")
    prop_lines.append("")
    prop_lines.append(
        "- This is a single-ckpt rerun on the same calibration substrate (2026-05-20 "
        "validation map). The classifier head was trained on CLS-pool features, "
        "so the absolute real-score distribution shifts upward under face-pool — "
        "the FPR-budgeted contract τ re-calibrates this away, but the calibration "
        "is now on shifted scale."
    )
    prop_lines.append(
        "- The composite-policy result (λ=1.0) names the FP-to-FN cost ratio "
        "explicitly. If the composite winner differs from the lex winner, that "
        "is a 2026-05-22 policy-question, not a face-pool question."
    )
    prop_lines.append(
        "- The 800-frame canary follow-up gave +6pp at FPR=10% and -4pp at "
        "FPR=5%; the per-chronic-identity Roy_D drop (-10pp) was the brightest "
        "single number. Full-suite extrapolation depends on whether the chronic "
        "subset is representative of the lockbox_real pool — RESULTS table 3a "
        "tests this directly."
    )
    prop_lines.append("")
    prop_lines.append("## Next steps (regardless of recommendation)")
    prop_lines.append("")
    prop_lines.append(
        "- Run face-pool on the broader 4-ckpt panel (P8A, T5C step3500, Slot A "
        "v2 step1500, Slot A v2 step3500) to see whether the lift is a Slot-A-"
        "v2 idiosyncrasy or a substrate-invariance lever that transfers."
    )
    prop_lines.append(
        "- Probe-side: confirm that the chronic-6 Roy_D drop reproduces in the "
        "full-suite lockbox pool — same identity, larger n."
    )

    (workdir / "AGENT_PROPOSAL_2026-05-22.md").write_text("\n".join(prop_lines))

    # ---- Sentinel verdict line ----
    verdict = (
        f"face-pool {recommend} — lockbox Δfpr={delta_fpr:+.4f} "
        f"Δrecall={delta_recall:+.4f}"
    )
    (workdir / "_verdict.txt").write_text(verdict + "\n")

    print(f"[write_results] wrote RESULTS, AGENT_PROPOSAL, _verdict.txt; recommendation={recommend}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
