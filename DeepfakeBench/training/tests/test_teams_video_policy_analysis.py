from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
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


video_analysis = _load_module(
    "teams_video_policy_analysis_test_module",
    "tools/teams_video_policy_analysis.py",
)


def _write_video_report(
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

class TeamsVideoPolicyAnalysisTest(unittest.TestCase):
    def test_video_policy_analysis_matches_contract_and_emits_abstain_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
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
                    _write_video_report(report_root, suite_name, checkpoint_key, rows)

            payload = video_analysis.analyze_video_reports(
                report_root=str(report_root),
                checkpoint_map_path=str(checkpoint_map_path),
                checkpoints_arg="R12_G_FP32,R13_FT7_FP32",
                contract=video_analysis.ContractConfig(
                    dev_real_suite="teams_real_all_dev",
                    dev_real_stress_suites=("teams_real_poor_quality_dev", "teams_real_lighting_extreme_dev"),
                    dev_fake_suites=("teams_fake_all_dev", "visomaster_enhanced_macro_dev", "deeplive_enhanced_dev"),
                    lockbox_real_suite="teams_real_all_lockbox",
                    lockbox_fake_suite="teams_fake_all_lockbox",
                ),
                abstain_margins=(0.0, 0.05),
            )

            self.assertEqual(payload["winner"]["checkpoint_key"], "R12_G_FP32")

            summary = {row["checkpoint_key"]: row for row in payload["checkpoint_summary_rows"]}
            self.assertEqual(summary["R12_G_FP32"]["promotion_rank"], 1)
            self.assertEqual(summary["R13_FT7_FP32"]["promotion_rank"], 2)
            self.assertEqual(summary["R12_G_FP32"]["selected_threshold"], 0.55)
            self.assertEqual(summary["R13_FT7_FP32"]["selected_threshold"], 0.4)

            abstain_rows = [
                row
                for row in payload["abstain_band_summary_rows"]
                if row["checkpoint_key"] == "R12_G_FP32" and row["abstain_margin"] == 0.05
            ]
            self.assertEqual(len(abstain_rows), 1)
            self.assertGreater(abstain_rows[0]["macro_uncertain_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
