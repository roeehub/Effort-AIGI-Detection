"""
Smoke test for trainer/trainer.py periodic_saves resolution.

Validates the bug fix at trainer/trainer.py:2326-2342 (committed 2026-04-28
after P12_HEAVY_LONG produced no periodic ckpts).

Confirms:
  1. The yaml at experiments/phase2_round13/R13_P12_HEAVY_LONG.yaml has
     periodic_saves.enabled=True and a non-empty step_list.
  2. yaml.safe_load returns a plain dict for periodic_saves; isinstance(dict)=True.
  3. The fix's isinstance(dict) guard:
       a. preserves dict behavior (.get('enabled') returns True),
       b. defaults to empty when fed a non-dict wrapper (no crash, no false-True).
  4. step_cnt=1000 (yaml's step_list[0]) is in the resolved step_list.

Run from the training/ directory:  python3 scripts/smoke_periodic_saves_2026-04-28.py
Exit code 0 = pass.  Non-zero = fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml


YAML_PATH = Path("experiments/phase2_round13/R13_P12_HEAVY_LONG.yaml")


def resolve(config: dict, key: str = "periodic_saves") -> dict:
    """Mirror the fix at trainer/trainer.py:2329-2330 exactly."""
    raw = config.get(key)
    return (raw or {}) if isinstance(raw, dict) else {}


class FakeWandbConfigSubdict:
    """
    Mimic the failure mode hypothesized for wandb.config nested values:
    has a .get() method (so the OLD buggy code's ``.get('enabled', False)``
    didn't raise), but isinstance(x, dict) is False.
    """

    def __init__(self, payload: dict):
        self._payload = payload

    def get(self, key, default=None):
        return self._payload.get(key, default)


def main() -> int:
    if not YAML_PATH.exists():
        print(f"FAIL: yaml not found at {YAML_PATH.resolve()}")
        return 1

    with YAML_PATH.open() as f:
        cfg = yaml.safe_load(f)

    # Test 1 — yaml top-level shape
    raw_ps = cfg.get("periodic_saves")
    print(f"[1] raw type from yaml.safe_load: {type(raw_ps).__name__}")
    assert isinstance(raw_ps, dict), "yaml.safe_load did not return a dict"
    print(f"[1] keys: {list(raw_ps.keys())}")
    assert raw_ps.get("enabled") is True, f"enabled not True: {raw_ps.get('enabled')!r}"
    step_list = raw_ps.get("step_list") or []
    print(f"[1] step_list: {step_list}")
    assert step_list, "step_list is empty"

    # Test 2 — fix path on dict input
    resolved = resolve(cfg, "periodic_saves")
    print(f"[2] resolved keys after isinstance(dict) gate: {list(resolved.keys())}")
    assert resolved.get("enabled") is True
    assert resolved.get("step_list") == step_list

    # Test 3 — fix path on non-dict wrapper input (the failure mode)
    fake_cfg = {"periodic_saves": FakeWandbConfigSubdict(raw_ps)}
    resolved_wrapped = resolve(fake_cfg, "periodic_saves")
    print(
        f"[3] non-dict wrapper input → resolved keys: "
        f"{list(resolved_wrapped.keys()) if resolved_wrapped else []}"
    )
    # Important: the fix downgrades to {}; it does NOT magically extract from
    # the wrapper. The diagnostic log added beside the fix is what reveals
    # this case at runtime, not the fix itself.
    assert resolved_wrapped == {}, (
        "isinstance(dict) guard should fall through to {} on non-dict input; "
        f"got {resolved_wrapped!r}"
    )

    # Test 4 — step_cnt membership
    step_cnt = 1000
    print(f"[4] step_cnt={step_cnt} in step_list={step_list}: {step_cnt in step_list}")
    assert step_cnt in step_list, "step_cnt=1000 expected in step_list"

    # Test 5 — old buggy code on the wrapper (would silently skip)
    old_periodic_cfg = fake_cfg.get("periodic_saves") or {}
    old_enabled = old_periodic_cfg.get("enabled", False)
    print(f"[5] OLD code on wrapper: .get('enabled', False) → {old_enabled}")
    # Old code path on the wrapper returns True via passthrough .get since the
    # FakeWandbConfigSubdict above honors get(). So the wrapper hypothesis
    # alone does not explain a silent skip — see notes in PRINTED summary.

    print()
    print("PASS — all smoke checks passed.")
    print()
    print("Caveats (read these):")
    print(
        "  * This test does NOT exercise actual wandb.config behavior — wandb "
        "isn't initialized here.\n"
        "  * The fix's isinstance(dict) guard is defense-in-depth; it does NOT "
        "by itself reveal which exact upstream object is responsible if the "
        "bug recurs.\n"
        "  * The diagnostic log added at trainer/trainer.py:2331-2340 is what "
        "exposes the runtime state — its output in the next training run is "
        "the empirical truth."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
