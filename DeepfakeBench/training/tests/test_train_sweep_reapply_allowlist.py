"""Regression test for the wandb-flattening config bug class.

Background: ``train_sweep.py`` calls ``wandb.init(config=single_cfg)`` which
flattens nested dicts into dot-notation top-level keys. After that call,
``wandb.config.get('anchor_aware')`` returns ``None`` even if the yaml had
``anchor_aware: {enabled: true, ...}``. ``train_sweep.py`` works around this
by re-applying an *allowlist* of nested dict blocks straight from
``single_cfg`` back into ``config`` / ``data_config``.

Bug class: any new top-level nested-dict yaml block that is consumed by the
trainer via ``self.config.get('<key>')`` MUST also be added to the
``train_sweep.py`` re-apply allowlist. Otherwise the trainer reads ``None``
and silently falls back to defaults — which is exactly what bit
``anchor_aware``, ``face_scale_jitter`` and ``periodic_saves`` on
2026-04-28.

This test enforces, for each known trainer-consumed nested-dict key, that
``train_sweep.py`` contains a ``'<key>' in single_cfg`` membership test
inside the bypass-flattening section.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# All nested-dict yaml keys the trainer expects to read at runtime.
# Add a new entry here when you add a new top-level nested-dict block to
# any experiment yaml AND consume it via self.config.get('<key>') in the
# trainer or any other downstream component.
TRAINER_NESTED_KEYS = [
    "anchor_aware",
    "face_scale_jitter",
    "periodic_saves",
    "value_composite",
    "augmentation",
    "combined_paired",
    "group_dro_params",
    "backbone",
    "checkpointing",
    "lesson_data_control",
    "lesson_gate",
    "dataset_methods",
    "correlation_penalty",
]


def test_train_sweep_reapplies_all_trainer_nested_keys():
    src = (ROOT / "train_sweep.py").read_text()
    missing = [k for k in TRAINER_NESTED_KEYS if f"'{k}' in single_cfg" not in src]
    assert not missing, (
        f"train_sweep.py is missing wandb-flattening re-apply for: {missing}. "
        "Add a block like `if '<key>' in single_cfg: config['<key>'] = single_cfg['<key>']` "
        "in the bypass-flattening section (around lines 176-322)."
    )


def test_p13_yaml_nested_blocks_are_all_reapplied():
    """Belt-and-suspenders: parse the P13 yaml top-level nested keys, ensure
    every one of them is either re-applied in train_sweep.py or in the
    documented exempt set (wandb / data_source / flat-key blocks)."""
    yaml_path = ROOT / "experiments" / "phase2_round13" / "R13_P13_FROM_SCRATCH.yaml"
    if not yaml_path.exists():
        # P13 yaml may move; skip rather than fail.
        return
    text = yaml_path.read_text()
    # Top-level keys are lines starting at column 0 that end in a colon and
    # whose next non-blank line is indented (signal of a nested dict).
    top_level_dict_keys = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line or line.startswith("#") or line.startswith(" "):
            continue
        if not line.endswith(":"):
            continue
        key = line[:-1].strip()
        # Look ahead for an indented line.
        for nxt in lines[i + 1 : i + 6]:
            stripped = nxt.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if nxt.startswith(" ") or nxt.startswith("\t"):
                top_level_dict_keys.append(key)
            break

    src = (ROOT / "train_sweep.py").read_text()
    # Keys that legitimately live outside the re-apply allowlist:
    #   wandb       — managed by wandb itself
    #   data_source — gets manual override elsewhere
    #   gcs_assets  — handled in a separate post-wandb override block
    #   deeplive / visomaster — handled when present (legacy path)
    EXEMPT = {"wandb", "data_source", "gcs_assets", "deeplive", "visomaster"}
    missing = []
    for k in top_level_dict_keys:
        if k in EXEMPT:
            continue
        if f"'{k}' in single_cfg" not in src:
            missing.append(k)
    assert not missing, (
        f"P13 yaml has nested-dict blocks not re-applied in train_sweep.py: {missing}. "
        "Either add them to the bypass-flattening allowlist or to the EXEMPT set in this test."
    )


if __name__ == "__main__":
    import traceback

    fns = [
        test_train_sweep_reapplies_all_trainer_nested_keys,
        test_p13_yaml_nested_blocks_are_all_reapplied,
    ]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed} passed, {failed} failed")
    raise SystemExit(failed)
