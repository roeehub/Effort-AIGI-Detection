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


frame_analysis = _load_module(
    "teams_frame_policy_analysis_test_module",
    "tools/teams_frame_policy_analysis.py",
)


def _write_frame_report(
    report_root: Path,
    suite_name: str,
    checkpoint_key: str,
    videos: dict[str, tuple[int, list[float]]],
) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"{suite_name}_{checkpoint_key.lower()}_frames_report.csv"
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "label", "video_id", "frame_path", "frame_prob", "group_key", "family_key"],
        )
        writer.writeheader()
        for video_id, (label, probs) in videos.items():
            for idx, prob in enumerate(probs):
                writer.writerow(
                    {
                        "method": suite_name,
                        "label": label,
                        "video_id": video_id,
                        "frame_path": f"{video_id}_frame_{idx:04d}.jpg",
                        "frame_prob": f"{prob:.6f}",
                        "group_key": suite_name,
                        "family_key": suite_name,
                    }
                )

class TeamsFramePolicyAnalysisTest(unittest.TestCase):
    def test_frame_policy_analysis_prefers_hysteresis_over_mean_on_spiky_reals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            checkpoint_map_path = tmp_path / "checkpoint_map.yaml"
            threshold_summary_path = tmp_path / "checkpoint_summary.csv"
            report_root = tmp_path / "reports"

            checkpoint_map_path.write_text('r12_g_fp32: "gs://bucket/r12_g_fp32.pth"\n')
            threshold_summary_path.write_text("checkpoint_key,selected_threshold\nR12_G_FP32,0.5\n")

            fixtures = {
                "teams_real_all_dev": {
                    "real_dev_a": (0, [0.10, 0.20, 0.60, 0.10]),
                    "real_dev_b": (0, [0.45, 0.55, 0.45, 0.55]),
                },
                "teams_real_poor_quality_dev": {
                    "real_poor_a": (0, [0.40, 0.52, 0.48, 0.53]),
                },
                "teams_real_lighting_extreme_dev": {
                    "real_light_a": (0, [0.42, 0.54, 0.44, 0.56]),
                },
                "teams_fake_all_dev": {
                    "fake_dev_a": (1, [0.60, 0.62, 0.63, 0.64]),
                },
                "visomaster_enhanced_macro_dev": {
                    "fake_vm_a": (1, [0.61, 0.63, 0.65, 0.67]),
                },
                "deeplive_enhanced_dev": {
                    "fake_dl_a": (1, [0.66, 0.68, 0.69, 0.70]),
                },
                "teams_real_all_lockbox": {
                    "real_lock_a": (0, [0.43, 0.57, 0.43, 0.57]),
                },
                "teams_fake_all_lockbox": {
                    "fake_lock_a": (1, [0.59, 0.60, 0.61, 0.62]),
                },
            }

            for suite_name, videos in fixtures.items():
                _write_frame_report(report_root, suite_name, "R12_G_FP32", videos)

            policies = [
                frame_analysis.PolicyConfig(policy_name="mean", policy_family="mean"),
                frame_analysis.PolicyConfig(
                    policy_name="hysteresis_m0p05_r2_c1",
                    policy_family="hysteresis",
                    hysteresis_margin=0.05,
                    raise_run=2,
                    clear_run=1,
                ),
            ]

            payload = frame_analysis.analyze_frame_reports(
                report_root=str(report_root),
                checkpoint_map_path=str(checkpoint_map_path),
                checkpoints_arg="R12_G_FP32",
                contract=frame_analysis.ContractConfig(
                    dev_real_suite="teams_real_all_dev",
                    dev_real_stress_suites=("teams_real_poor_quality_dev", "teams_real_lighting_extreme_dev"),
                    dev_fake_suites=("teams_fake_all_dev", "visomaster_enhanced_macro_dev", "deeplive_enhanced_dev"),
                    lockbox_real_suite="teams_real_all_lockbox",
                    lockbox_fake_suite="teams_fake_all_lockbox",
                ),
                default_threshold=0.5,
                threshold_summary_csv=str(threshold_summary_path),
                threshold_offsets=(0.0,),
                policies=policies,
            )

            summary_rows = {
                row["policy_name"]: row
                for row in payload["policy_checkpoint_summary_rows"]
                if row["checkpoint_key"] == "R12_G_FP32"
            }
            self.assertLess(
                summary_rows["hysteresis_m0p05_r2_c1"]["dev_primary_real_fpr"],
                summary_rows["mean"]["dev_primary_real_fpr"],
            )
            self.assertLess(
                summary_rows["hysteresis_m0p05_r2_c1"]["lockbox_real_fpr"],
                summary_rows["mean"]["lockbox_real_fpr"],
            )

            stability_rows = {
                row["suite_name"]: row
                for row in payload["stability_summary_rows"]
                if row["checkpoint_key"] == "R12_G_FP32"
            }
            self.assertGreater(stability_rows["teams_real_all_dev"]["mean_jitter"], 0.0)
            self.assertGreater(stability_rows["teams_real_all_dev"]["flip_rate_offset_0"], 0.0)


if __name__ == "__main__":
    unittest.main()
