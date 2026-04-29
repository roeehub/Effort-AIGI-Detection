"""Tests for the calibrated Teams promotion-contract scorer."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(module_name: str, relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


promotion = _load_module(
    "teams_promotion_contract_test_module",
    "arena/score_teams_promotion_contract.py",
)


def _write_report(
    report_root: Path,
    suite_name: str,
    checkpoint_key: str,
    rows: list[tuple[int, float]],
) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"{suite_name}_{checkpoint_key.lower()}_videos_report.csv"
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "label", "video_id", "avg_video_prob", "prediction", "group_key", "family_key"],
        )
        writer.writeheader()
        for idx, (label, prob) in enumerate(rows, 1):
            writer.writerow(
                {
                    "method": suite_name,
                    "label": label,
                    "video_id": f"{suite_name}_{idx:03d}",
                    "avg_video_prob": f"{prob:.6f}",
                    "prediction": int(prob >= 0.5),
                    "group_key": suite_name,
                    "family_key": suite_name,
                }
            )


def test_score_promotion_contract_freezes_dev_threshold_and_ranks_lockbox(tmp_path):
    checkpoint_map_path = tmp_path / "checkpoint_map.yaml"
    report_root = tmp_path / "reports"

    checkpoint_map_path.write_text(
        "\n".join(
            [
                'r12_g_fp32: "gs://bucket/r12_g_fp32.pth"',
                'r13_ft7_fp32: "gs://bucket/r13_ft7_fp32.pth"',
            ]
        )
    )

    fixtures = {
        "R12_G_FP32": {
            "teams_real_all_dev": [(0, 0.10), (0, 0.15), (0, 0.20)],
            "teams_real_poor_quality_dev": [(0, 0.18), (0, 0.22)],
            "teams_real_lighting_extreme_dev": [(0, 0.23), (0, 0.25)],
            "teams_fake_all_dev": [(1, 0.55), (1, 0.60)],
            "visomaster_enhanced_macro_dev": [(1, 0.58)],
            "deeplive_enhanced_dev": [(1, 0.57)],
            "teams_real_all_lockbox": [(0, 0.30), (0, 0.40)],
            "teams_fake_all_lockbox": [(1, 0.52), (1, 0.60)],
        },
        "R13_FT7_FP32": {
            "teams_real_all_dev": [(0, 0.05), (0, 0.08), (0, 0.12)],
            "teams_real_poor_quality_dev": [(0, 0.09), (0, 0.14)],
            "teams_real_lighting_extreme_dev": [(0, 0.11), (0, 0.13)],
            "teams_fake_all_dev": [(1, 0.40), (1, 0.45)],
            "visomaster_enhanced_macro_dev": [(1, 0.42)],
            "deeplive_enhanced_dev": [(1, 0.43)],
            "teams_real_all_lockbox": [(0, 0.20), (0, 0.42)],
            "teams_fake_all_lockbox": [(1, 0.60), (1, 0.65)],
        },
    }

    for checkpoint_key, suite_map in fixtures.items():
        for suite_name, rows in suite_map.items():
            _write_report(report_root, suite_name, checkpoint_key, rows)

    payload = promotion.score_promotion_contract(
        report_root=str(report_root),
        checkpoint_map_path=str(checkpoint_map_path),
        checkpoints_arg="R12_G_FP32,R13_FT7_FP32",
        contract=promotion.ContractConfig(
            dev_real_suite="teams_real_all_dev",
            dev_real_stress_suites=("teams_real_poor_quality_dev", "teams_real_lighting_extreme_dev"),
            dev_fake_suites=("teams_fake_all_dev", "visomaster_enhanced_macro_dev", "deeplive_enhanced_dev"),
            lockbox_real_suite="teams_real_all_lockbox",
            lockbox_fake_suite="teams_fake_all_lockbox",
        ),
    )

    assert payload["winner"]["checkpoint_key"] == "R12_G_FP32"

    summary = {row["checkpoint_key"]: row for row in payload["checkpoint_summary_rows"]}
    assert summary["R12_G_FP32"]["promotion_rank"] == 1
    assert summary["R13_FT7_FP32"]["promotion_rank"] == 2

    assert summary["R12_G_FP32"]["selected_threshold"] == 0.55
    assert summary["R13_FT7_FP32"]["selected_threshold"] == 0.4

    assert summary["R12_G_FP32"]["lockbox_real_fpr"] == 0.0
    assert summary["R13_FT7_FP32"]["lockbox_real_fpr"] == 0.5

    scorecard_rows = {
        (row["checkpoint_key"], row["suite_name"]): row
        for row in payload["selected_threshold_scorecard_rows"]
    }
    assert scorecard_rows[("R12_G_FP32", "teams_fake_all_lockbox")]["fake_recall"] == 0.5
    assert scorecard_rows[("R13_FT7_FP32", "teams_fake_all_lockbox")]["fake_recall"] == 1.0


def test_write_promotion_contract_outputs_persists_expected_files(tmp_path):
    checkpoint_map_path = tmp_path / "checkpoint_map.yaml"
    report_root = tmp_path / "reports"

    checkpoint_map_path.write_text('r12_g_fp32: "gs://bucket/r12_g_fp32.pth"\n')

    fixtures = {
        "teams_real_all_dev": [(0, 0.10), (0, 0.15), (0, 0.20)],
        "teams_real_poor_quality_dev": [(0, 0.18), (0, 0.22)],
        "teams_real_lighting_extreme_dev": [(0, 0.23), (0, 0.25)],
        "teams_fake_all_dev": [(1, 0.55), (1, 0.60)],
        "visomaster_enhanced_macro_dev": [(1, 0.58)],
        "deeplive_enhanced_dev": [(1, 0.57)],
        "teams_real_all_lockbox": [(0, 0.30), (0, 0.40)],
        "teams_fake_all_lockbox": [(1, 0.52), (1, 0.60)],
    }
    for suite_name, rows in fixtures.items():
        _write_report(report_root, suite_name, "R12_G_FP32", rows)

    payload = promotion.score_promotion_contract(
        report_root=str(report_root),
        checkpoint_map_path=str(checkpoint_map_path),
        checkpoints_arg="R12_G_FP32",
        contract=promotion.ContractConfig(
            dev_real_suite="teams_real_all_dev",
            dev_real_stress_suites=("teams_real_poor_quality_dev", "teams_real_lighting_extreme_dev"),
            dev_fake_suites=("teams_fake_all_dev", "visomaster_enhanced_macro_dev", "deeplive_enhanced_dev"),
            lockbox_real_suite="teams_real_all_lockbox",
            lockbox_fake_suite="teams_fake_all_lockbox",
        ),
    )

    output_paths = promotion.write_promotion_contract_outputs(
        str(tmp_path / "promotion_contract"),
        payload,
    )

    summary_path = Path(output_paths["checkpoint_summary_csv"])
    winner_path = Path(output_paths["promotion_winner_json"])
    assert summary_path.exists()
    assert winner_path.exists()

    summary_rows = list(csv.DictReader(summary_path.open()))
    assert len(summary_rows) == 1
    assert summary_rows[0]["checkpoint_key"] == "R12_G_FP32"
    assert summary_rows[0]["promotion_rank"] == "1"

    winner_payload = json.loads(winner_path.read_text())
    assert winner_payload["winner"]["checkpoint_key"] == "R12_G_FP32"


def _base_contract(**overrides):
    base = dict(
        dev_real_suite="teams_real_all_dev",
        dev_real_stress_suites=("teams_real_poor_quality_dev", "teams_real_lighting_extreme_dev"),
        dev_fake_suites=("teams_fake_all_dev", "visomaster_enhanced_macro_dev", "deeplive_enhanced_dev"),
        lockbox_real_suite="teams_real_all_lockbox",
        lockbox_fake_suite="teams_fake_all_lockbox",
    )
    base.update(overrides)
    return promotion.ContractConfig(**base)


def test_promotion_summary_sort_key_demotes_low_recall_with_floor():
    """Cross-checkpoint ranker: with target_fake_recall_min > 0, a checkpoint
    whose dev_fake_macro_recall is below the floor must rank below any
    checkpoint that satisfies the floor, regardless of lockbox FPR. Without
    the floor, the degenerate (low-FPR / low-recall) checkpoint wins on
    lockbox FPR alone — the bug from project_contract_policy_bug.md.
    """
    no_floor = _base_contract(target_fake_recall_min=0.0)
    with_floor = _base_contract(target_fake_recall_min=0.30)

    degenerate = {
        "lockbox_real_fpr": 0.0,
        "lockbox_fake_recall": 0.0,
        "dev_primary_real_fpr": 0.0,
        "dev_worst_real_stress_fpr": 0.0,
        "dev_fake_macro_recall": 0.05,
        "teams_fake_all_dev__fake_recall": 0.05,
        "visomaster_enhanced_macro_dev__fake_recall": 0.05,
        "deeplive_enhanced_dev__fake_recall": 0.05,
    }
    healthy = {
        "lockbox_real_fpr": 0.05,
        "lockbox_fake_recall": 0.50,
        "dev_primary_real_fpr": 0.05,
        "dev_worst_real_stress_fpr": 0.05,
        "dev_fake_macro_recall": 0.50,
        "teams_fake_all_dev__fake_recall": 0.50,
        "visomaster_enhanced_macro_dev__fake_recall": 0.50,
        "deeplive_enhanced_dev__fake_recall": 0.50,
    }

    key_deg_no_floor = promotion._promotion_summary_sort_key(degenerate, no_floor)
    key_healthy_no_floor = promotion._promotion_summary_sort_key(healthy, no_floor)
    assert key_deg_no_floor < key_healthy_no_floor, (
        "without floor: degenerate ckpt wins on lower lockbox_real_fpr"
    )

    key_deg_floor = promotion._promotion_summary_sort_key(degenerate, with_floor)
    key_healthy_floor = promotion._promotion_summary_sort_key(healthy, with_floor)
    assert key_deg_floor[0] == 1, "degenerate (recall < floor) sits in tier 1"
    assert key_healthy_floor[0] == 0, "healthy (recall ≥ floor) sits in tier 0"
    assert key_healthy_floor < key_deg_floor, (
        "with floor: healthy ckpt wins despite higher lockbox_real_fpr"
    )


def test_recall_floor_changes_winner_in_score_promotion_contract(tmp_path):
    """End-to-end: same fixtures, two policies. Without the recall floor the
    degenerate ckpt wins (it has lower lockbox_real_fpr); with the floor the
    healthy ckpt wins (only it satisfies dev_fake_macro_recall ≥ floor).
    """
    checkpoint_map_path = tmp_path / "checkpoint_map.yaml"
    report_root = tmp_path / "reports"
    checkpoint_map_path.write_text(
        'degenerate_low_fpr: "gs://bucket/degenerate.pth"\n'
        'good_recall: "gs://bucket/good_recall.pth"\n'
    )

    # DEGENERATE_LOW_FPR: dev reals cluster very high (>0.45) and dev fakes
    # cluster very low (<0.40). Only τ=1.0 satisfies the FPR budget; at τ=1.0
    # macro_recall=0. Lockbox real probs are all very low → lockbox FPR=0.
    # GOOD_RECALL: classic separable distribution; τ≈0.55 gives macro_recall=1.0
    # with FPR=0 on dev. Lockbox real probs include one above τ → lockbox FPR>0.
    fixtures = {
        "DEGENERATE_LOW_FPR": {
            "teams_real_all_dev": [(0, 0.45), (0, 0.46), (0, 0.47)],
            "teams_real_poor_quality_dev": [(0, 0.10), (0, 0.15)],
            "teams_real_lighting_extreme_dev": [(0, 0.10), (0, 0.15)],
            "teams_fake_all_dev": [(1, 0.30), (1, 0.35)],
            "visomaster_enhanced_macro_dev": [(1, 0.25)],
            "deeplive_enhanced_dev": [(1, 0.32)],
            "teams_real_all_lockbox": [(0, 0.10), (0, 0.15)],
            "teams_fake_all_lockbox": [(1, 0.20), (1, 0.30)],
        },
        "GOOD_RECALL": {
            "teams_real_all_dev": [(0, 0.10), (0, 0.20), (0, 0.30)],
            "teams_real_poor_quality_dev": [(0, 0.10), (0, 0.20)],
            "teams_real_lighting_extreme_dev": [(0, 0.10), (0, 0.20)],
            "teams_fake_all_dev": [(1, 0.60), (1, 0.70)],
            "visomaster_enhanced_macro_dev": [(1, 0.65)],
            "deeplive_enhanced_dev": [(1, 0.55)],
            "teams_real_all_lockbox": [(0, 0.30), (0, 0.60)],
            "teams_fake_all_lockbox": [(1, 0.55), (1, 0.65)],
        },
    }
    for ckpt, suite_map in fixtures.items():
        for suite_name, rows in suite_map.items():
            _write_report(report_root, suite_name, ckpt, rows)

    common = dict(
        report_root=str(report_root),
        checkpoint_map_path=str(checkpoint_map_path),
        checkpoints_arg="DEGENERATE_LOW_FPR,GOOD_RECALL",
    )

    payload_no_floor = promotion.score_promotion_contract(
        **common,
        contract=_base_contract(target_fake_recall_min=0.0),
    )
    assert payload_no_floor["winner"]["checkpoint_key"] == "DEGENERATE_LOW_FPR"

    payload_with_floor = promotion.score_promotion_contract(
        **common,
        contract=_base_contract(target_fake_recall_min=0.30),
    )
    assert payload_with_floor["winner"]["checkpoint_key"] == "GOOD_RECALL"
    summary = {row["checkpoint_key"]: row for row in payload_with_floor["checkpoint_summary_rows"]}
    assert summary["GOOD_RECALL"]["promotion_rank"] == 1
    assert summary["DEGENERATE_LOW_FPR"]["promotion_rank"] == 2
    assert summary["DEGENERATE_LOW_FPR"]["dev_fake_macro_recall"] == 0.0
    assert summary["GOOD_RECALL"]["dev_fake_macro_recall"] == 1.0


def test_default_contract_config_has_active_recall_floor():
    """Regression guard: the default ContractConfig must keep
    target_fake_recall_min ≥ 0.70. The recall-floor gating in
    _threshold_sort_key / _promotion_summary_sort_key only activates when
    the floor is strictly > 0.0, so a default of 0.0 silently re-enters the
    τ-tail-collapse failure mode on any wrapper that omits the CLI flag.
    See docs/packet_retrospectives/threads/contract_policy_bug.md.
    """
    contract = _base_contract()
    assert contract.target_fake_recall_min >= 0.70, (
        f"Default ContractConfig has target_fake_recall_min="
        f"{contract.target_fake_recall_min}; must be ≥ 0.70 — a Teams-deployment "
        "detector with macro fake recall below 70 % is not deployment-grade, "
        "and a default below 0.70 silently re-enables τ-tail-collapse."
    )


def test_wrapper_and_scorer_cli_defaults_match_contract_default():
    """Regression guard: the argparse defaults for both the standalone scorer
    (--target_fake_recall_min) and the sequential validation wrapper
    (--promotion_target_fake_recall_min) must match the dataclass default.

    The wrapper passes args.promotion_target_fake_recall_min directly into
    ContractConfig, so the wrapper CLI default is the load-bearing one for
    arena/launch_*.sh production scorecards. Drift between the dataclass
    default and the CLI defaults silently re-enables the τ-tail-collapse bug.
    """
    import re

    contract_default = _base_contract().target_fake_recall_min

    scorer_text = (ROOT / "arena/score_teams_promotion_contract.py").read_text()
    scorer_match = re.search(
        r'add_argument\(\s*"--target_fake_recall_min"\s*,\s*type=float\s*,\s*default=([^,\)]+)',
        scorer_text,
    )
    assert scorer_match is not None, (
        "could not locate --target_fake_recall_min argparse default in scorer"
    )
    scorer_default = float(scorer_match.group(1).strip())
    assert scorer_default == contract_default, (
        f"Scorer CLI --target_fake_recall_min default ({scorer_default}) does "
        f"not match ContractConfig default ({contract_default})."
    )

    wrapper_text = (ROOT / "arena/run_target_domain_validation_sequential.py").read_text()
    wrapper_match = re.search(
        r'add_argument\(\s*"--promotion_target_fake_recall_min"\s*,\s*type=float\s*,\s*default=([^,\s\)]+)',
        wrapper_text,
    )
    assert wrapper_match is not None, (
        "could not locate --promotion_target_fake_recall_min argparse default in wrapper"
    )
    wrapper_default = float(wrapper_match.group(1).strip())
    assert wrapper_default == contract_default, (
        f"Wrapper CLI --promotion_target_fake_recall_min default ({wrapper_default}) "
        f"does not match ContractConfig default ({contract_default})."
    )
