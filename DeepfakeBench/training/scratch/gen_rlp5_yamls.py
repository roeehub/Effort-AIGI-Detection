"""
Generator for R13 Packet-5 yamls.

Base = experiments/phase2_round13/R13_RLP4_06_arcface_m015_spatial.yaml
(spatial aug 0.08/0.40, arcface_m=0.15, anneal_steps=8000, value_composite 0.03/0.05/p95).

Applies targeted string edits per-slot. Verifies each expected substitution occurs
exactly once (--replace-count) to fail loud on any drift in the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE_PATH = ROOT / "experiments" / "phase2_round13" / "R13_RLP4_06_arcface_m015_spatial.yaml"
OUT_DIR = ROOT / "experiments" / "phase2_round13"


BASE_NAME_LINE = 'name: "R13_RLP4_06_arcface_m015_spatial"'
BASE_DESCRIPTION_LINE = (
    'description: "Packet 4 core stack: P3.5 leader arcface_m=0.15 x P3 leader spatial aug. '
    'Strongest candidate for new leader if the two real wins add cleanly under the NEW '
    '(0.03, 0.05, p95) gates."'
)

# Exactly this string must appear once in baseline (top-level seed line, no indent).
BASE_TOP_SEED = "\nseed: 737\n"

BASE_INCLUDE_LANES = """  proper_data:
    enabled: true
    inventory_path: "arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml"
    manifest_path: "arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json"
    include_lanes:
      - "proper_visomaster_clean"
      - "proper_visomaster_teams"
    anchor_indices: [0, 2, 4, 6, 8, 10, 12, 14]"""

BASE_FAMILY_WEIGHTS = """  sampling:
    strategy: "identity_resample_weighted"
    family_weights:
      df40_fake: 0.15
      deeplive_non_enhanced_fake: 2.5
      deeplive_enhanced_fake: 3.0
      deeplive_teams_fake: 5.0
      proper_visomaster_clean_fake: 1.0
      proper_visomaster_teams_fake: 1.0
      deeplive_teams_real: 4.0
      df40_real: 0.4
      realpool_real: 2.5
      external_real: 2.5"""


def build_include_lanes_block(add_enhanced: bool) -> str:
    lanes = ['      - "proper_visomaster_clean"', '      - "proper_visomaster_teams"']
    if add_enhanced:
        lanes.append('      - "proper_visomaster_enhanced_teams"')
    return (
        "  proper_data:\n"
        "    enabled: true\n"
        '    inventory_path: "arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml"\n'
        '    manifest_path: "arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json"\n'
        "    include_lanes:\n"
        + "\n".join(lanes)
        + "\n    anchor_indices: [0, 2, 4, 6, 8, 10, 12, 14]"
    )


def build_family_weights_block(
    *,
    teams_fake: float,
    enhanced_teams_fake: float | None,
    deeplive_non_enhanced: float = 2.5,
    deeplive_enhanced: float = 3.0,
    deeplive_teams: float = 5.0,
) -> str:
    lines = [
        '  sampling:',
        '    strategy: "identity_resample_weighted"',
        '    family_weights:',
        '      df40_fake: 0.15',
        f'      deeplive_non_enhanced_fake: {deeplive_non_enhanced}',
        f'      deeplive_enhanced_fake: {deeplive_enhanced}',
        f'      deeplive_teams_fake: {deeplive_teams}',
        '      proper_visomaster_clean_fake: 1.0',
        f'      proper_visomaster_teams_fake: {teams_fake}',
    ]
    if enhanced_teams_fake is not None:
        lines.append(f'      proper_visomaster_enhanced_teams_fake: {enhanced_teams_fake}')
    lines.extend([
        '      deeplive_teams_real: 4.0',
        '      df40_real: 0.4',
        '      realpool_real: 2.5',
        '      external_real: 2.5',
    ])
    return "\n".join(lines)


@dataclass
class Slot:
    num: str
    name: str
    description: str
    include_enhanced_lane: bool
    teams_fake: float
    enhanced_teams_fake: float | None  # None = key not set
    seed: int = 737  # top-level seed (everything else stays 737)
    # Optional deeplive reweight (slot 6 only)
    deeplive_non_enhanced: float = 2.5
    deeplive_enhanced: float = 3.0
    deeplive_teams: float = 5.0
    extra_tags: list[str] = field(default_factory=list)


SLOTS: list[Slot] = [
    Slot(
        num="01",
        name="R13_RLP5_01_E3_teams2_5",
        description="Packet 5 E3 core — enable proper_visomaster_enhanced_teams lane + teams_fake 1.0->2.5. Primary new-leader candidate.",
        include_enhanced_lane=True,
        teams_fake=2.5,
        enhanced_teams_fake=1.0,
        extra_tags=["packet-5", "rlp5", "e3-core", "enhanced-teams-lane", "teams-fake-2_5", "enhanced-teams-fake-1_0"],
    ),
    Slot(
        num="02",
        name="R13_RLP5_02_E3_teams4_0",
        description="Packet 5 E3 teams ceiling — enhanced lane + teams_fake 4.0. Probe upper end of teams reweight.",
        include_enhanced_lane=True,
        teams_fake=4.0,
        enhanced_teams_fake=1.0,
        extra_tags=["packet-5", "rlp5", "e3", "enhanced-teams-lane", "teams-fake-4_0", "enhanced-teams-fake-1_0"],
    ),
    Slot(
        num="03",
        name="R13_RLP5_03_E3_both2_5",
        description="Packet 5 E3 matched — enhanced lane + teams_fake 2.5 + enhanced_teams_fake 2.5. Equal weight on both proper-teams lanes.",
        include_enhanced_lane=True,
        teams_fake=2.5,
        enhanced_teams_fake=2.5,
        extra_tags=["packet-5", "rlp5", "e3", "enhanced-teams-lane", "teams-fake-2_5", "enhanced-teams-fake-2_5"],
    ),
    Slot(
        num="04",
        name="R13_RLP5_04_E3_enh4_0",
        description="Packet 5 E3 enhanced-maximal — enhanced lane + teams_fake 2.5 + enhanced_teams_fake 4.0. Heavy weight on the previously-excluded lane.",
        include_enhanced_lane=True,
        teams_fake=2.5,
        enhanced_teams_fake=4.0,
        extra_tags=["packet-5", "rlp5", "e3", "enhanced-teams-lane", "teams-fake-2_5", "enhanced-teams-fake-4_0"],
    ),
    Slot(
        num="05",
        name="R13_RLP5_05_E3_both_heavy",
        description="Packet 5 E3 kill-or-cure — enhanced lane + teams_fake 4.0 + enhanced_teams_fake 3.0. Both levers pushed hard.",
        include_enhanced_lane=True,
        teams_fake=4.0,
        enhanced_teams_fake=3.0,
        extra_tags=["packet-5", "rlp5", "e3", "enhanced-teams-lane", "teams-fake-4_0", "enhanced-teams-fake-3_0"],
    ),
    Slot(
        num="06",
        name="R13_RLP5_06_E3_deeplive_up",
        description="Packet 5 E3 + DeepLive-up — enhanced lane + teams_fake 2.5 + enhanced_teams_fake 1.0; deeplive_teams 5->6, deeplive_enhanced 3->4, deeplive_non_enhanced 2.5->3.0.",
        include_enhanced_lane=True,
        teams_fake=2.5,
        enhanced_teams_fake=1.0,
        deeplive_non_enhanced=3.0,
        deeplive_enhanced=4.0,
        deeplive_teams=6.0,
        extra_tags=["packet-5", "rlp5", "e3", "enhanced-teams-lane", "teams-fake-2_5", "enhanced-teams-fake-1_0", "deeplive-up"],
    ),
    Slot(
        num="07",
        name="R13_RLP5_07_E3_seedB",
        description="Packet 5 E3 core seed replication — identical to RLP5_01 but top-level seed 737->742 (split_seed and identity_split_seed stay 737 to preserve the train/held-out identity partition).",
        include_enhanced_lane=True,
        teams_fake=2.5,
        enhanced_teams_fake=1.0,
        seed=742,
        extra_tags=["packet-5", "rlp5", "e3-core", "enhanced-teams-lane", "teams-fake-2_5", "enhanced-teams-fake-1_0", "seed-742"],
    ),
    Slot(
        num="08",
        name="R13_RLP5_08_E1_teams2_5",
        description="Packet 5 E1 pure control — NO enhanced lane; teams_fake 1.0->2.5 only. Isolates weight lever from lane-enablement lever.",
        include_enhanced_lane=False,
        teams_fake=2.5,
        enhanced_teams_fake=None,
        extra_tags=["packet-5", "rlp5", "e1-pure", "teams-fake-2_5"],
    ),
]


BASE_WANDB_TAGS = """wandb:
  tags: ["phase2-round13", "rlp4", "packet-4", "fine-tune", "vit-b-16", "wt-b", "wt-f",
         "data-composition", "no-hints", "proper-data", "proper-unenhanced",
         "policy-aware-teams", "live-discovery", "cacheless-sources",
         "r12g-fp32-base", "ft-scorecard-base", "seed-737",
         "truthful-aug-baseline", "rlp4-instrumented",
         "arcface-m-015", "spatial-aug", "value-composite-35-gate", "p95-stability"]"""


def build_wandb_tags(slot: Slot) -> str:
    base_tags = [
        "phase2-round13",
        "fine-tune", "vit-b-16", "wt-b", "wt-f",
        "data-composition", "proper-data",
        "policy-aware-teams", "live-discovery", "cacheless-sources",
        "r12g-fp32-base", "ft-scorecard-base",
        f"seed-{slot.seed}",
        "truthful-aug-baseline",
        "arcface-m-015", "spatial-aug", "value-composite-35-gate", "p95-stability",
    ]
    tags = base_tags + slot.extra_tags
    quoted = ", ".join(f'"{t}"' for t in tags)
    return f"wandb:\n  tags: [{quoted}]"


BASE_HEADER = """# ============================================================================
# R13_RLP35_02_arcface_m015
# Phase 2, Round 13 - relaunch packet 3.5 / ArcFace margin probe (middle)
# Single-lever delta from RLP3_02: arcface_m 0.0 -> 0.15.
# Global changes (applied to every packet-3.5 slot):
#   - value_composite.target_mean_fpr: 0.02 -> 0.03  (user-approved relaxed gate)
#   - value_composite.max_pool_fpr:    0.04 -> 0.05  (user-approved relaxed gate)
#   - value_composite.stability_jitter_stat: "max" -> "p95"
#   - anneal_steps: 15000 -> 8000  (so ArcFace anneal completes inside 10k training steps)
# ============================================================================"""


def build_header(slot: Slot) -> str:
    return f"""# ============================================================================
# {slot.name}
# Phase 2, Round 13 - relaunch packet 5.
# Base = R13_RLP3_05_FT_proper_low_arcface_spatial (spatial 0.08/0.40, arcface_m=0.15,
# anneal_steps=8000, value_composite gate 0.03/0.05/p95).
# Motivated by: docs/relaunch_handoffs/R13_RELAUNCH_VISOMASTER_TEAMS_POOL_DIAGNOSTIC_FINDINGS_2026-04-23.md
#   - in-training proper_visomaster_teams_fake accuracy ~67% (vs teams_ood_fake 91%),
#   - proper_visomaster_enhanced_teams lane (1484 captures) was NEVER in training.
# Delta from base (this slot):
# {slot.description}
# ============================================================================"""


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"ERROR while rendering {label}: expected exactly 1 occurrence of substitution target, got {count}."
        )
    return text.replace(old, new, 1)


def render_slot(base: str, slot: Slot) -> str:
    out = base

    # Header block.
    out = replace_once(out, BASE_HEADER, build_header(slot), label=f"slot{slot.num} header")

    # Name + description.
    out = replace_once(
        out,
        BASE_NAME_LINE,
        f'name: "{slot.name}"',
        label=f"slot{slot.num} name",
    )
    out = replace_once(
        out,
        BASE_DESCRIPTION_LINE,
        f'description: "{slot.description}"',
        label=f"slot{slot.num} description",
    )

    # wandb.tags block.
    out = replace_once(out, BASE_WANDB_TAGS, build_wandb_tags(slot), label=f"slot{slot.num} tags")

    # Top-level seed (must NOT touch split_seed or identity_split_seed).
    if slot.seed != 737:
        out = replace_once(
            out,
            BASE_TOP_SEED,
            f"\nseed: {slot.seed}\n",
            label=f"slot{slot.num} top seed",
        )

    # include_lanes + anchor_indices block.
    out = replace_once(
        out,
        BASE_INCLUDE_LANES,
        build_include_lanes_block(add_enhanced=slot.include_enhanced_lane),
        label=f"slot{slot.num} include_lanes",
    )

    # family_weights block.
    out = replace_once(
        out,
        BASE_FAMILY_WEIGHTS,
        build_family_weights_block(
            teams_fake=slot.teams_fake,
            enhanced_teams_fake=slot.enhanced_teams_fake,
            deeplive_non_enhanced=slot.deeplive_non_enhanced,
            deeplive_enhanced=slot.deeplive_enhanced,
            deeplive_teams=slot.deeplive_teams,
        ),
        label=f"slot{slot.num} family_weights",
    )

    return out


def main() -> int:
    base = BASE_PATH.read_text()
    # Sanity: baseline must contain all the anchor strings.
    for anchor, label in [
        (BASE_HEADER, "header"),
        (BASE_NAME_LINE, "name"),
        (BASE_DESCRIPTION_LINE, "description"),
        (BASE_WANDB_TAGS, "tags"),
        (BASE_TOP_SEED, "top seed"),
        (BASE_INCLUDE_LANES, "include_lanes"),
        (BASE_FAMILY_WEIGHTS, "family_weights"),
    ]:
        count = base.count(anchor)
        if count != 1:
            print(f"Baseline anchor mismatch: {label} occurrences = {count} (expected 1)", file=sys.stderr)
            return 2

    written = []
    for slot in SLOTS:
        rendered = render_slot(base, slot)
        out_path = OUT_DIR / f"R13_RLP5_{slot.num}_{slot.name.split('_', 2)[-1]}.yaml"
        # Use the exact slot.name to avoid accidental path-name mismatch.
        out_path = OUT_DIR / f"{slot.name}.yaml"
        out_path.write_text(rendered)
        written.append(out_path)

    print(f"Wrote {len(written)} files:")
    for p in written:
        print(f"  {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
