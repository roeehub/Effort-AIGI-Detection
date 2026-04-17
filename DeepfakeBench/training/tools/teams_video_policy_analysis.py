#!/usr/bin/env python3
"""
WT-D video-report analysis for threshold sweeps and abstain-band evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from teams_decision_report_utils import (  # noqa: E402
    ContractConfig,
    VideoScore,
    csv_list,
    join_path,
    load_video_scores,
    optional_mean,
    quantile,
    read_text_from_path,
    report_path_for_job,
    resolve_checkpoints,
    resolve_requested_checkpoint_keys,
    round_float,
    safe_rate,
    score_decisions,
    sort_number,
    write_dict_rows_to_csv,
    write_text_to_path,
)


def _candidate_thresholds(
    report_rows_by_suite: Dict[str, List[VideoScore]],
    contract: ContractConfig,
) -> List[float]:
    values = {0.0, 1.0}
    for suite_name in contract.dev_suites():
        for row in report_rows_by_suite.get(suite_name, []):
            values.add(round(float(row.prob), 8))
    return sorted(values)


def _suite_metrics(rows: Sequence[VideoScore], threshold: float) -> Dict[str, Any]:
    labels = [int(row.label) for row in rows]
    decisions = [1 if float(row.prob) >= threshold else 0 for row in rows]
    metrics = score_decisions(labels, decisions)
    probs = [float(row.prob) for row in rows]
    metrics.update(
        {
            "threshold": round_float(threshold),
            "mean_prob": round_float(sum(probs) / float(len(probs)) if probs else 0.0),
            "p50_prob": round_float(quantile(probs, 0.50)),
            "p90_prob": round_float(quantile(probs, 0.90)),
        }
    )
    return metrics


def _suite_metrics_with_abstain(
    rows: Sequence[VideoScore],
    threshold: float,
    margin: float,
) -> Dict[str, Any]:
    lower = threshold - margin
    upper = threshold + margin

    labels = [int(row.label) for row in rows]
    decisions: List[int | None] = []
    probs = [float(row.prob) for row in rows]
    for prob in probs:
        if prob >= upper:
            decisions.append(1)
        elif prob < lower:
            decisions.append(0)
        else:
            decisions.append(None)

    metrics = score_decisions(labels, decisions)
    metrics.update(
        {
            "threshold": round_float(threshold),
            "abstain_margin": round_float(margin),
            "abstain_lower": round_float(lower),
            "abstain_upper": round_float(upper),
            "mean_prob": round_float(sum(probs) / float(len(probs)) if probs else 0.0),
            "p50_prob": round_float(quantile(probs, 0.50)),
            "p90_prob": round_float(quantile(probs, 0.90)),
        }
    )
    return metrics


def _build_threshold_grid_row(
    checkpoint_key: str,
    checkpoint_path: str,
    threshold: float,
    report_rows_by_suite: Dict[str, List[VideoScore]],
    contract: ContractConfig,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "checkpoint_key": checkpoint_key,
        "checkpoint_path": checkpoint_path,
        "threshold": round_float(threshold),
    }

    dev_fake_recalls: List[float | None] = []
    dev_real_stress_fprs: List[float | None] = []

    for suite_name in contract.dev_suites():
        metrics = _suite_metrics(report_rows_by_suite[suite_name], threshold)
        row[f"{suite_name}__n_videos"] = metrics["n_videos"]
        row[f"{suite_name}__real_fpr"] = metrics["real_fpr"]
        row[f"{suite_name}__fake_recall"] = metrics["fake_recall"]
        row[f"{suite_name}__accuracy"] = metrics["accuracy"]
        if suite_name == contract.dev_real_suite:
            row["dev_primary_real_fpr"] = metrics["real_fpr"]
        if suite_name in contract.dev_real_stress_suites:
            dev_real_stress_fprs.append(metrics["real_fpr"])
        if suite_name in contract.dev_fake_suites:
            dev_fake_recalls.append(metrics["fake_recall"])

    row["dev_worst_real_stress_fpr"] = round_float(
        max(value for value in dev_real_stress_fprs if value is not None)
        if any(value is not None for value in dev_real_stress_fprs)
        else None
    )
    row["dev_fake_macro_recall"] = round_float(optional_mean(dev_fake_recalls))
    return row


def _threshold_sort_key(row: Dict[str, Any], contract: ContractConfig) -> Tuple[float, ...]:
    parts: List[float] = [
        sort_number(row.get("dev_primary_real_fpr")),
        sort_number(row.get("dev_worst_real_stress_fpr")),
    ]
    for suite_name in contract.dev_fake_suites:
        parts.append(sort_number(row.get(f"{suite_name}__fake_recall"), higher_is_better=True))
    parts.append(sort_number(row.get("threshold"), higher_is_better=True))
    return tuple(parts)


def _promotion_summary_sort_key(row: Dict[str, Any], contract: ContractConfig) -> Tuple[float, ...]:
    parts: List[float] = []
    if contract.lockbox_real_suite:
        parts.append(sort_number(row.get("lockbox_real_fpr")))
    if contract.lockbox_fake_suite:
        parts.append(sort_number(row.get("lockbox_fake_recall"), higher_is_better=True))
    parts.extend(
        [
            sort_number(row.get("dev_primary_real_fpr")),
            sort_number(row.get("dev_worst_real_stress_fpr")),
        ]
    )
    for suite_name in contract.dev_fake_suites:
        parts.append(sort_number(row.get(f"{suite_name}__fake_recall"), higher_is_better=True))
    parts.append(sort_number(row.get("selected_threshold"), higher_is_better=True))
    return tuple(parts)


def _abstain_summary_sort_key(row: Dict[str, Any], contract: ContractConfig) -> Tuple[float, ...]:
    parts: List[float] = []
    if contract.lockbox_real_suite:
        parts.append(sort_number(row.get("lockbox_real_fpr")))
    parts.append(sort_number(row.get("dev_primary_real_fpr")))
    parts.append(sort_number(row.get("dev_worst_real_stress_fpr")))
    if contract.lockbox_fake_suite:
        parts.append(sort_number(row.get("lockbox_fake_recall"), higher_is_better=True))
    parts.append(sort_number(row.get("dev_fake_macro_recall"), higher_is_better=True))
    parts.append(sort_number(row.get("macro_uncertain_rate")))
    parts.append(sort_number(row.get("abstain_margin")))
    return tuple(parts)


def _build_scorecard_rows(
    checkpoint_key: str,
    checkpoint_path: str,
    threshold: float,
    report_rows_by_suite: Dict[str, List[VideoScore]],
    report_paths_by_suite: Dict[str, str],
    contract: ContractConfig,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for suite_name in contract.all_suites():
        metrics = _suite_metrics(report_rows_by_suite[suite_name], threshold)
        rows.append(
            {
                "checkpoint_key": checkpoint_key,
                "checkpoint_path": checkpoint_path,
                "suite_name": suite_name,
                "report_path": report_paths_by_suite[suite_name],
                **metrics,
            }
        )
    return rows


def _build_abstain_rows(
    checkpoint_key: str,
    checkpoint_path: str,
    threshold: float,
    margins: Sequence[float],
    report_rows_by_suite: Dict[str, List[VideoScore]],
    report_paths_by_suite: Dict[str, str],
    contract: ContractConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    scorecard_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for margin in margins:
        suite_rows: List[Dict[str, Any]] = []
        for suite_name in contract.all_suites():
            metrics = _suite_metrics_with_abstain(report_rows_by_suite[suite_name], threshold, margin)
            suite_row = {
                "checkpoint_key": checkpoint_key,
                "checkpoint_path": checkpoint_path,
                "selected_threshold": round_float(threshold),
                "suite_name": suite_name,
                "report_path": report_paths_by_suite[suite_name],
                **metrics,
            }
            suite_rows.append(suite_row)
        scorecard_rows.extend(suite_rows)

        suite_map = {row["suite_name"]: row for row in suite_rows}
        summary_row: Dict[str, Any] = {
            "checkpoint_key": checkpoint_key,
            "checkpoint_path": checkpoint_path,
            "selected_threshold": round_float(threshold),
            "abstain_margin": round_float(margin),
            "macro_uncertain_rate": round_float(
                optional_mean(row.get("uncertain_rate") for row in suite_rows)
            ),
            "macro_decided_rate": round_float(
                optional_mean(row.get("decided_rate") for row in suite_rows)
            ),
            "dev_primary_real_fpr": suite_map.get(contract.dev_real_suite, {}).get("real_fpr"),
            "dev_worst_real_stress_fpr": round_float(
                max(
                    value
                    for value in (
                        suite_map.get(suite_name, {}).get("real_fpr")
                        for suite_name in contract.dev_real_stress_suites
                    )
                    if value is not None
                )
                if any(
                    suite_map.get(suite_name, {}).get("real_fpr") is not None
                    for suite_name in contract.dev_real_stress_suites
                )
                else None
            ),
            "dev_fake_macro_recall": round_float(
                optional_mean(
                    suite_map.get(suite_name, {}).get("fake_recall")
                    for suite_name in contract.dev_fake_suites
                )
            ),
        }
        for suite_name in contract.dev_fake_suites:
            summary_row[f"{suite_name}__fake_recall"] = suite_map.get(suite_name, {}).get("fake_recall")
        if contract.lockbox_real_suite:
            summary_row["lockbox_real_fpr"] = suite_map.get(contract.lockbox_real_suite, {}).get("real_fpr")
            summary_row["lockbox_real_uncertain_rate"] = suite_map.get(contract.lockbox_real_suite, {}).get("uncertain_rate")
        if contract.lockbox_fake_suite:
            summary_row["lockbox_fake_recall"] = suite_map.get(contract.lockbox_fake_suite, {}).get("fake_recall")
            summary_row["lockbox_fake_uncertain_rate"] = suite_map.get(contract.lockbox_fake_suite, {}).get("uncertain_rate")
        summary_rows.append(summary_row)

    summary_rows.sort(key=lambda row: _abstain_summary_sort_key(row, contract))
    for rank, row in enumerate(summary_rows, 1):
        row["abstain_rank"] = rank

    return scorecard_rows, summary_rows


def analyze_video_reports(
    *,
    report_root: str,
    checkpoint_map_path: str,
    checkpoints_arg: str,
    contract: ContractConfig,
    abstain_margins: Sequence[float],
) -> Dict[str, Any]:
    selected_keys = resolve_requested_checkpoint_keys(checkpoints_arg, checkpoint_map_path)
    checkpoints = resolve_checkpoints(selected_keys, checkpoint_map_path)

    threshold_grid_rows: List[Dict[str, Any]] = []
    selected_threshold_scorecard_rows: List[Dict[str, Any]] = []
    checkpoint_summary_rows: List[Dict[str, Any]] = []
    abstain_band_scorecard_rows: List[Dict[str, Any]] = []
    abstain_band_summary_rows: List[Dict[str, Any]] = []

    for checkpoint_key in selected_keys:
        checkpoint_path = checkpoints[checkpoint_key]
        report_rows_by_suite: Dict[str, List[VideoScore]] = {}
        report_paths_by_suite: Dict[str, str] = {}
        for suite_name in contract.all_suites():
            report_path = report_path_for_job(report_root, checkpoint_key, suite_name, "videos_report")
            report_paths_by_suite[suite_name] = report_path
            report_rows_by_suite[suite_name] = load_video_scores(report_path)

        candidate_thresholds = _candidate_thresholds(report_rows_by_suite, contract)
        checkpoint_grid_rows = [
            _build_threshold_grid_row(
                checkpoint_key=checkpoint_key,
                checkpoint_path=checkpoint_path,
                threshold=threshold,
                report_rows_by_suite=report_rows_by_suite,
                contract=contract,
            )
            for threshold in candidate_thresholds
        ]
        checkpoint_grid_rows.sort(key=lambda row: _threshold_sort_key(row, contract))
        best_threshold_row = checkpoint_grid_rows[0]
        selected_threshold = float(best_threshold_row["threshold"])

        threshold_grid_rows.extend(checkpoint_grid_rows)

        scorecard_rows = _build_scorecard_rows(
            checkpoint_key=checkpoint_key,
            checkpoint_path=checkpoint_path,
            threshold=selected_threshold,
            report_rows_by_suite=report_rows_by_suite,
            report_paths_by_suite=report_paths_by_suite,
            contract=contract,
        )
        selected_threshold_scorecard_rows.extend(scorecard_rows)

        suite_row_map = {row["suite_name"]: row for row in scorecard_rows}
        summary_row: Dict[str, Any] = {
            "checkpoint_key": checkpoint_key,
            "checkpoint_path": checkpoint_path,
            "selected_threshold": round_float(selected_threshold),
            "threshold_candidate_count": len(candidate_thresholds),
            "dev_primary_real_fpr": best_threshold_row.get("dev_primary_real_fpr"),
            "dev_worst_real_stress_fpr": best_threshold_row.get("dev_worst_real_stress_fpr"),
            "dev_fake_macro_recall": best_threshold_row.get("dev_fake_macro_recall"),
        }
        for suite_name in contract.dev_fake_suites:
            summary_row[f"{suite_name}__fake_recall"] = best_threshold_row.get(f"{suite_name}__fake_recall")
        if contract.lockbox_real_suite:
            summary_row["lockbox_real_suite"] = contract.lockbox_real_suite
            summary_row["lockbox_real_n_videos"] = suite_row_map[contract.lockbox_real_suite]["n_videos"]
            summary_row["lockbox_real_fpr"] = suite_row_map[contract.lockbox_real_suite].get("real_fpr")
        if contract.lockbox_fake_suite:
            summary_row["lockbox_fake_suite"] = contract.lockbox_fake_suite
            summary_row["lockbox_fake_n_videos"] = suite_row_map[contract.lockbox_fake_suite]["n_videos"]
            summary_row["lockbox_fake_recall"] = suite_row_map[contract.lockbox_fake_suite].get("fake_recall")
        checkpoint_summary_rows.append(summary_row)

        abstain_rows, abstain_summary = _build_abstain_rows(
            checkpoint_key=checkpoint_key,
            checkpoint_path=checkpoint_path,
            threshold=selected_threshold,
            margins=abstain_margins,
            report_rows_by_suite=report_rows_by_suite,
            report_paths_by_suite=report_paths_by_suite,
            contract=contract,
        )
        abstain_band_scorecard_rows.extend(abstain_rows)
        abstain_band_summary_rows.extend(abstain_summary)

    checkpoint_summary_rows.sort(key=lambda row: _promotion_summary_sort_key(row, contract))
    for rank, row in enumerate(checkpoint_summary_rows, 1):
        row["promotion_rank"] = rank

    return {
        "contract": {
            "dev_real_suite": contract.dev_real_suite,
            "dev_real_stress_suites": list(contract.dev_real_stress_suites),
            "dev_fake_suites": list(contract.dev_fake_suites),
            "lockbox_real_suite": contract.lockbox_real_suite,
            "lockbox_fake_suite": contract.lockbox_fake_suite,
        },
        "threshold_grid_rows": threshold_grid_rows,
        "selected_threshold_scorecard_rows": selected_threshold_scorecard_rows,
        "checkpoint_summary_rows": checkpoint_summary_rows,
        "abstain_band_scorecard_rows": abstain_band_scorecard_rows,
        "abstain_band_summary_rows": abstain_band_summary_rows,
        "winner": checkpoint_summary_rows[0] if checkpoint_summary_rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WT-D Teams video policy analysis from *_videos_report.csv artifacts"
    )
    parser.add_argument("--report_root", required=True)
    parser.add_argument("--checkpoint_map", required=True)
    parser.add_argument("--checkpoints", default="ALL")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dev_real_suite", default="teams_real_all_dev")
    parser.add_argument(
        "--dev_real_stress_suites",
        default="teams_real_poor_quality_dev,teams_real_lighting_extreme_dev",
    )
    parser.add_argument(
        "--dev_fake_suites",
        default="teams_fake_all_dev,visomaster_enhanced_macro_dev,deeplive_enhanced_dev",
    )
    parser.add_argument("--lockbox_real_suite", default="teams_real_all_lockbox")
    parser.add_argument("--lockbox_fake_suite", default="teams_fake_all_lockbox")
    parser.add_argument("--abstain_margins", default="0.0,0.02,0.05")
    args = parser.parse_args()

    contract = ContractConfig(
        dev_real_suite=str(args.dev_real_suite).strip(),
        dev_real_stress_suites=csv_list(args.dev_real_stress_suites),
        dev_fake_suites=csv_list(args.dev_fake_suites),
        lockbox_real_suite=str(args.lockbox_real_suite).strip(),
        lockbox_fake_suite=str(args.lockbox_fake_suite).strip(),
    )
    abstain_margins = [float(value) for value in csv_list(args.abstain_margins)]

    payload = analyze_video_reports(
        report_root=str(args.report_root).strip(),
        checkpoint_map_path=str(args.checkpoint_map).strip(),
        checkpoints_arg=str(args.checkpoints).strip(),
        contract=contract,
        abstain_margins=abstain_margins,
    )

    output_root = str(args.output_dir).rstrip("/")
    threshold_grid_csv = join_path(output_root, "threshold_grid.csv")
    scorecard_csv = join_path(output_root, "selected_threshold_scorecard.csv")
    summary_csv = join_path(output_root, "checkpoint_summary.csv")
    abstain_scorecard_csv = join_path(output_root, "abstain_band_scorecard.csv")
    abstain_summary_csv = join_path(output_root, "abstain_band_summary.csv")
    payload_json = join_path(output_root, "video_policy_analysis.json")

    write_dict_rows_to_csv(threshold_grid_csv, payload["threshold_grid_rows"])
    write_dict_rows_to_csv(scorecard_csv, payload["selected_threshold_scorecard_rows"])
    write_dict_rows_to_csv(summary_csv, payload["checkpoint_summary_rows"])
    write_dict_rows_to_csv(abstain_scorecard_csv, payload["abstain_band_scorecard_rows"])
    write_dict_rows_to_csv(abstain_summary_csv, payload["abstain_band_summary_rows"])
    write_text_to_path(payload_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    winner = payload["winner"]
    print("WT-D video policy analysis complete.")
    print(f"  threshold_grid.csv          : {threshold_grid_csv}")
    print(f"  checkpoint_summary.csv      : {summary_csv}")
    print(f"  abstain_band_summary.csv    : {abstain_summary_csv}")
    print(f"  video_policy_analysis.json  : {payload_json}")
    if winner:
        print(
            "  winner                      : "
            f"{winner['checkpoint_key']} "
            f"(threshold={winner['selected_threshold']}, "
            f"dev_real_fpr={winner['dev_primary_real_fpr']}, "
            f"lockbox_real_fpr={winner.get('lockbox_real_fpr')})"
        )


if __name__ == "__main__":
    main()
