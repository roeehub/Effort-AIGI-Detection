from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run"
    / "prepare_training_speedup_validation.py"
)


def _load_validation_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_training_speedup_validation",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_apply_short_validation_overrides_disables_eval_and_ood():
    module = _load_validation_module()

    source = {
        "name": "R13_B_k64_capacity",
        "description": "Capacity test",
        "seed": 737,
        "evaluate_every_steps": 500,
        "ood_monitoring_enabled": True,
        "ood_monitoring_start_step": 5000,
        "ood_monitoring_every_steps": 1000,
        "early_stopping_enabled": True,
        "wandb": {"tags": ["phase2-round13"]},
        "checkpointing": {"save_every_steps": 200, "keep_last_n": 3},
        "combined_paired": {
            "ood_monitoring": {
                "enabled": True,
                "external_real_sources": [{"method": "zoom_vcd_real", "max_videos": 300}],
            }
        },
    }

    derived = module.apply_short_validation_overrides(
        source,
        label="Speedup",
        max_train_steps=120,
        log_progress_steps=1,
    )

    assert source["name"] == "R13_B_k64_capacity"
    assert source["combined_paired"]["ood_monitoring"]["enabled"] is True

    assert derived["name"] == "R13_B_k64_capacity__speedval_speedup"
    assert derived["seed"] == 737
    assert derived["max_train_steps"] == 120
    assert derived["evaluate_every_steps"] == 1120
    assert derived["ood_monitoring_enabled"] is False
    assert derived["ood_monitoring_start_step"] == 1120
    assert derived["ood_monitoring_every_steps"] == 1120
    assert derived["early_stopping_enabled"] is False
    assert derived["wandb"]["log_progress_steps"] == 1
    assert "training-speedup-validation" in derived["wandb"]["tags"]
    assert "fixed-seed" in derived["wandb"]["tags"]
    assert "speedup" in derived["wandb"]["tags"]
    assert derived["checkpointing"]["save_every_steps"] == 1120
    assert derived["checkpointing"]["keep_last_n"] == 1
    assert derived["combined_paired"]["ood_monitoring"]["enabled"] is False


def test_main_writes_validation_yaml(tmp_path):
    module = _load_validation_module()

    input_path = tmp_path / "source.yaml"
    output_path = tmp_path / "derived.yaml"
    input_path.write_text(
        yaml.safe_dump(
            {
                "name": "R12_A_scratch_aug_fix",
                "description": "Scratch run",
                "seed": 737,
                "wandb": {"tags": ["phase2-round12"]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    rc = module.main(
        [
            "--input-config",
            str(input_path),
            "--output-config",
            str(output_path),
            "--label",
            "main",
            "--max-train-steps",
            "64",
        ]
    )

    assert rc == 0
    saved = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert saved["name"] == "R12_A_scratch_aug_fix__speedval_main"
    assert saved["max_train_steps"] == 64
    assert saved["ood_monitoring_enabled"] is False
    assert saved["wandb"]["log_progress_steps"] == 1
