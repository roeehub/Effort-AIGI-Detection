#!/usr/bin/env python3
"""
WT-D frame-report analysis for stability, temporal aggregation, and hysteresis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from teams_decision_report_utils import (  # noqa: E402
    ContractConfig,
    FrameScore,
    csv_list,
    format_float_slug,
    group_frame_scores_by_video,
    join_path,
    load_frame_scores,
    optional_mean,
    report_path_for_job,
    resolve_checkpoints,
    resolve_requested_checkpoint_keys,
    round_float,
    score_decisions,
    sort_number,
    stddev,
    write_dict_rows_to_csv,
    write_text_to_path,
)


@dataclass(frozen=True)
class PolicyConfig:
    policy_name: str
    policy_family: str
    alpha: float | None = None
    run_length: int | None = None
    hysteresis_margin: float | None = None
    raise_run: int | None = None
    clear_run: int | None = None


def _load_threshold_map(summary_csv: str | None, default_threshold: float) -> Dict[str, float]:
    threshold_map: Dict[str, float] = {}
    if not summary_csv:
        return threshold_map
    with Path(summary_csv).open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        checkpoint_key = str(row.get("checkpoint_key", "")).strip().upper()
        selected_threshold = row.get("selected_threshold")
        if checkpoint_key and selected_threshold:
            threshold_map[checkpoint_key] = float(selected_threshold)
    return threshold_map


def _clip_threshold(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _build_policy_configs(args: argparse.Namespace) -> List[PolicyConfig]:
    configs: List[PolicyConfig] = []
    families = {part.strip().lower() for part in csv_list(args.policy_families)}

    if "mean" in families:
        configs.append(PolicyConfig(policy_name="mean", policy_family="mean"))
    if "median" in families:
        configs.append(PolicyConfig(policy_name="median", policy_family="median"))
    if "majority" in families:
        configs.append(PolicyConfig(policy_name="majority", policy_family="majority"))
    if "ema_last" in families:
        for alpha in (float(value) for value in csv_list(args.ema_alphas)):
            configs.append(
                PolicyConfig(
                    policy_name=f"ema_last_a{format_float_slug(alpha)}",
                    policy_family="ema_last",
                    alpha=alpha,
                )
            )
    if "consecutive_positive" in families:
        for run_length in (int(value) for value in csv_list(args.positive_run_lengths)):
            configs.append(
                PolicyConfig(
                    policy_name=f"consecutive_positive_k{run_length}",
                    policy_family="consecutive_positive",
                    run_length=run_length,
                )
            )
    if "hysteresis" in families:
        margins = [float(value) for value in csv_list(args.hysteresis_margins)]
        raise_runs = [int(value) for value in csv_list(args.hysteresis_raise_runs)]
        clear_runs = [int(value) for value in csv_list(args.hysteresis_clear_runs)]
        for margin in margins:
            for raise_run in raise_runs:
                for clear_run in clear_runs:
                    configs.append(
                        PolicyConfig(
                            policy_name=(
                                f"hysteresis_m{format_float_slug(margin)}"
                                f"_r{raise_run}_c{clear_run}"
                            ),
                            policy_family="hysteresis",
                            hysteresis_margin=margin,
                            raise_run=raise_run,
                            clear_run=clear_run,
                        )
                    )
    if not configs:
        raise ValueError("No policy configurations were generated.")
    return configs


def _evaluate_policy(
    frames: Sequence[FrameScore],
    threshold: float,
    policy: PolicyConfig,
) -> Tuple[int | None, float | None]:
    probs = [float(frame.prob) for frame in frames]
    if not probs:
        return None, None

    if policy.policy_family == "mean":
        score = sum(probs) / float(len(probs))
        return (1 if score >= threshold else 0), score

    if policy.policy_family == "median":
        ordered = sorted(probs)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            score = ordered[mid]
        else:
            score = (ordered[mid - 1] + ordered[mid]) / 2.0
        return (1 if score >= threshold else 0), score

    if policy.policy_family == "majority":
        fake_votes = sum(1 for prob in probs if prob >= threshold)
        score = fake_votes / float(len(probs))
        return (1 if fake_votes > len(probs) / 2.0 else 0), score

    if policy.policy_family == "ema_last":
        assert policy.alpha is not None
        alpha = float(policy.alpha)
        ema = probs[0]
        for prob in probs[1:]:
            ema = alpha * prob + (1.0 - alpha) * ema
        return (1 if ema >= threshold else 0), ema

    if policy.policy_family == "consecutive_positive":
        assert policy.run_length is not None
        current_run = 0
        max_run = 0
        for prob in probs:
            if prob >= threshold:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return (1 if max_run >= int(policy.run_length) else 0), float(max_run)

    if policy.policy_family == "hysteresis":
        assert policy.hysteresis_margin is not None
        assert policy.raise_run is not None
        assert policy.clear_run is not None

        raise_threshold = _clip_threshold(threshold + float(policy.hysteresis_margin))
        clear_threshold = _clip_threshold(threshold - float(policy.hysteresis_margin))
        state_fake = False
        ever_alerted = False
        fake_state_frames = 0
        positive_run = 0
        negative_run = 0

        for prob in probs:
            if not state_fake:
                if prob >= raise_threshold:
                    positive_run += 1
                else:
                    positive_run = 0
                if positive_run >= int(policy.raise_run):
                    state_fake = True
                    ever_alerted = True
                    positive_run = 0
                    negative_run = 0
            else:
                if prob <= clear_threshold:
                    negative_run += 1
                else:
                    negative_run = 0
                if negative_run >= int(policy.clear_run):
                    state_fake = False
                    positive_run = 0
                    negative_run = 0
                else:
                    fake_state_frames += 1

        score = fake_state_frames / float(len(probs))
        return (1 if ever_alerted else 0), score

    raise ValueError(f"Unsupported policy family: {policy.policy_family}")


def _flip_rate(probs: Sequence[float], threshold: float) -> Tuple[int, float]:
    if len(probs) < 2:
        return 0, 0.0
    decisions = [1 if prob >= threshold else 0 for prob in probs]
    flips = sum(1 for idx in range(1, len(decisions)) if decisions[idx] != decisions[idx - 1])
    return flips, flips / float(len(decisions) - 1)


def _stability_rows_for_suite(
    checkpoint_key: str,
    checkpoint_path: str,
    suite_name: str,
    grouped_frames: Dict[str, List[FrameScore]],
    base_threshold: float,
    threshold_offsets: Sequence[float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    per_video_rows: List[Dict[str, Any]] = []
    all_diffs: List[float] = []
    total_frame_pairs = 0
    flip_counts = {offset: 0 for offset in threshold_offsets}

    for video_id, frames in grouped_frames.items():
        probs = [float(frame.prob) for frame in frames]
        if len(probs) < 2:
            continue
        diffs = [abs(probs[idx] - probs[idx - 1]) for idx in range(1, len(probs))]
        all_diffs.extend(diffs)
        total_frame_pairs += len(diffs)

        row: Dict[str, Any] = {
            "checkpoint_key": checkpoint_key,
            "checkpoint_path": checkpoint_path,
            "suite_name": suite_name,
            "video_id": video_id,
            "label": frames[0].label,
            "method": frames[0].method,
            "n_frames": len(probs),
            "mean_prob": round_float(sum(probs) / float(len(probs))),
            "prob_std": round_float(stddev(probs)),
            "mean_jitter": round_float(sum(diffs) / float(len(diffs))),
            "max_jitter": round_float(max(diffs)),
            "std_jitter": round_float(stddev(diffs)),
        }

        for offset in threshold_offsets:
            threshold_value = _clip_threshold(base_threshold + float(offset))
            flips, flip_rate = _flip_rate(probs, threshold_value)
            flip_counts[offset] += flips
            slug = format_float_slug(offset)
            row[f"flip_count_offset_{slug}"] = flips
            row[f"flip_rate_offset_{slug}"] = round_float(flip_rate)
            row[f"threshold_offset_{slug}"] = round_float(threshold_value)

        per_video_rows.append(row)

    if not all_diffs:
        summary_row = {
            "checkpoint_key": checkpoint_key,
            "checkpoint_path": checkpoint_path,
            "suite_name": suite_name,
            "base_threshold": round_float(base_threshold),
            "n_videos": 0,
            "n_frame_pairs": 0,
            "mean_jitter": None,
            "median_jitter": None,
            "p95_jitter": None,
            "max_jitter": None,
            "std_jitter": None,
        }
        return summary_row, per_video_rows

    ordered_diffs = sorted(all_diffs)
    p95_index = max(0, math.ceil(0.95 * len(ordered_diffs)) - 1)
    summary_row = {
        "checkpoint_key": checkpoint_key,
        "checkpoint_path": checkpoint_path,
        "suite_name": suite_name,
        "base_threshold": round_float(base_threshold),
        "n_videos": len(per_video_rows),
        "n_frame_pairs": total_frame_pairs,
        "mean_jitter": round_float(sum(all_diffs) / float(len(all_diffs))),
        "median_jitter": round_float(ordered_diffs[len(ordered_diffs) // 2]),
        "p95_jitter": round_float(ordered_diffs[p95_index]),
        "max_jitter": round_float(max(all_diffs)),
        "std_jitter": round_float(stddev(all_diffs)),
    }
    for offset in threshold_offsets:
        slug = format_float_slug(offset)
        rate = flip_counts[offset] / float(total_frame_pairs) if total_frame_pairs > 0 else 0.0
        summary_row[f"flip_rate_offset_{slug}"] = round_float(rate)
        summary_row[f"threshold_offset_{slug}"] = round_float(_clip_threshold(base_threshold + float(offset)))

    return summary_row, per_video_rows


def _policy_summary_sort_key(row: Dict[str, Any], contract: ContractConfig) -> Tuple[float, ...]:
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
    parts.append(sort_number(row.get("policy_name")))
    return tuple(parts)


def analyze_frame_reports(
    *,
    report_root: str,
    checkpoint_map_path: str,
    checkpoints_arg: str,
    contract: ContractConfig,
    default_threshold: float,
    threshold_summary_csv: str | None,
    threshold_offsets: Sequence[float],
    policies: Sequence[PolicyConfig],
) -> Dict[str, Any]:
    selected_keys = resolve_requested_checkpoint_keys(checkpoints_arg, checkpoint_map_path)
    checkpoints = resolve_checkpoints(selected_keys, checkpoint_map_path)
    threshold_map = _load_threshold_map(threshold_summary_csv, default_threshold)

    stability_summary_rows: List[Dict[str, Any]] = []
    stability_per_video_rows: List[Dict[str, Any]] = []
    policy_suite_metric_rows: List[Dict[str, Any]] = []
    policy_checkpoint_summary_rows: List[Dict[str, Any]] = []

    for checkpoint_key in selected_keys:
        checkpoint_path = checkpoints[checkpoint_key]
        base_threshold = float(threshold_map.get(checkpoint_key, default_threshold))
        grouped_by_suite: Dict[str, Dict[str, List[FrameScore]]] = {}

        for suite_name in contract.all_suites():
            report_path = report_path_for_job(report_root, checkpoint_key, suite_name, "frames_report")
            grouped_frames = group_frame_scores_by_video(load_frame_scores(report_path))
            grouped_by_suite[suite_name] = grouped_frames

            stability_row, per_video_rows = _stability_rows_for_suite(
                checkpoint_key=checkpoint_key,
                checkpoint_path=checkpoint_path,
                suite_name=suite_name,
                grouped_frames=grouped_frames,
                base_threshold=base_threshold,
                threshold_offsets=threshold_offsets,
            )
            stability_summary_rows.append(stability_row)
            stability_per_video_rows.extend(per_video_rows)

        for policy in policies:
            suite_rows: List[Dict[str, Any]] = []
            for suite_name in contract.all_suites():
                grouped_frames = grouped_by_suite[suite_name]
                labels: List[int] = []
                decisions: List[int | None] = []
                policy_scores: List[float] = []

                for frames in grouped_frames.values():
                    labels.append(int(frames[0].label))
                    decision, policy_score = _evaluate_policy(frames, base_threshold, policy)
                    decisions.append(decision)
                    if policy_score is not None:
                        policy_scores.append(float(policy_score))

                metrics = score_decisions(labels, decisions)
                suite_row: Dict[str, Any] = {
                    "checkpoint_key": checkpoint_key,
                    "checkpoint_path": checkpoint_path,
                    "suite_name": suite_name,
                    "base_threshold": round_float(base_threshold),
                    "policy_name": policy.policy_name,
                    "policy_family": policy.policy_family,
                    "mean_policy_score": round_float(
                        sum(policy_scores) / float(len(policy_scores)) if policy_scores else None
                    ),
                    **metrics,
                }
                suite_rows.append(suite_row)
            policy_suite_metric_rows.extend(suite_rows)

            suite_map = {row["suite_name"]: row for row in suite_rows}
            summary_row: Dict[str, Any] = {
                "checkpoint_key": checkpoint_key,
                "checkpoint_path": checkpoint_path,
                "base_threshold": round_float(base_threshold),
                "policy_name": policy.policy_name,
                "policy_family": policy.policy_family,
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
            if contract.lockbox_fake_suite:
                summary_row["lockbox_fake_recall"] = suite_map.get(contract.lockbox_fake_suite, {}).get("fake_recall")
            policy_checkpoint_summary_rows.append(summary_row)

    policy_checkpoint_summary_rows.sort(key=lambda row: _policy_summary_sort_key(row, contract))
    for rank, row in enumerate(policy_checkpoint_summary_rows, 1):
        row["policy_rank"] = rank

    return {
        "contract": {
            "dev_real_suite": contract.dev_real_suite,
            "dev_real_stress_suites": list(contract.dev_real_stress_suites),
            "dev_fake_suites": list(contract.dev_fake_suites),
            "lockbox_real_suite": contract.lockbox_real_suite,
            "lockbox_fake_suite": contract.lockbox_fake_suite,
        },
        "stability_summary_rows": stability_summary_rows,
        "stability_per_video_rows": stability_per_video_rows,
        "policy_suite_metric_rows": policy_suite_metric_rows,
        "policy_checkpoint_summary_rows": policy_checkpoint_summary_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WT-D Teams frame policy analysis from *_frames_report.csv artifacts"
    )
    parser.add_argument("--report_root", required=True)
    parser.add_argument("--checkpoint_map", required=True)
    parser.add_argument("--checkpoints", default="ALL")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_summary_csv", default="")
    parser.add_argument("--default_threshold", type=float, default=0.5)
    parser.add_argument("--flip_threshold_offsets", default="-0.05,0.0,0.05")
    parser.add_argument("--policy_families", default="mean,median,majority,ema_last,consecutive_positive,hysteresis")
    parser.add_argument("--ema_alphas", default="0.35")
    parser.add_argument("--positive_run_lengths", default="2,3")
    parser.add_argument("--hysteresis_margins", default="0.02,0.05")
    parser.add_argument("--hysteresis_raise_runs", default="2")
    parser.add_argument("--hysteresis_clear_runs", default="1,2")
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
    args = parser.parse_args()

    contract = ContractConfig(
        dev_real_suite=str(args.dev_real_suite).strip(),
        dev_real_stress_suites=csv_list(args.dev_real_stress_suites),
        dev_fake_suites=csv_list(args.dev_fake_suites),
        lockbox_real_suite=str(args.lockbox_real_suite).strip(),
        lockbox_fake_suite=str(args.lockbox_fake_suite).strip(),
    )
    threshold_offsets = [float(value) for value in csv_list(args.flip_threshold_offsets)]
    policies = _build_policy_configs(args)

    payload = analyze_frame_reports(
        report_root=str(args.report_root).strip(),
        checkpoint_map_path=str(args.checkpoint_map).strip(),
        checkpoints_arg=str(args.checkpoints).strip(),
        contract=contract,
        default_threshold=float(args.default_threshold),
        threshold_summary_csv=str(args.checkpoint_summary_csv).strip() or None,
        threshold_offsets=threshold_offsets,
        policies=policies,
    )

    output_root = str(args.output_dir).rstrip("/")
    stability_summary_csv = join_path(output_root, "stability_summary.csv")
    stability_per_video_csv = join_path(output_root, "stability_per_video.csv")
    policy_suite_metrics_csv = join_path(output_root, "policy_suite_metrics.csv")
    policy_summary_csv = join_path(output_root, "policy_checkpoint_summary.csv")
    payload_json = join_path(output_root, "frame_policy_analysis.json")

    write_dict_rows_to_csv(stability_summary_csv, payload["stability_summary_rows"])
    write_dict_rows_to_csv(stability_per_video_csv, payload["stability_per_video_rows"])
    write_dict_rows_to_csv(policy_suite_metrics_csv, payload["policy_suite_metric_rows"])
    write_dict_rows_to_csv(policy_summary_csv, payload["policy_checkpoint_summary_rows"])
    write_text_to_path(payload_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("WT-D frame policy analysis complete.")
    print(f"  stability_summary.csv        : {stability_summary_csv}")
    print(f"  policy_checkpoint_summary.csv: {policy_summary_csv}")
    print(f"  frame_policy_analysis.json   : {payload_json}")


if __name__ == "__main__":
    main()
