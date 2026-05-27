"""
Multi-axis GroupDRO group_id construction — recommended by
analysis/group_id_design_audit_2026-05-06.

Fake-side scheme: F-B
Real-side scheme: R-D

Drop this into the data loader / collate function. Returns a single string
group_id per row that the existing trainer/mixins/group_dro.py can use after
extending to accept arbitrary str keys (the current code uses int method_id
via data_params.method_mapping; you must build a dict {group_str -> int} at
config-load time and pass it as `data_params.group_id_mapping`).

Chronic identity list comes from
memory/project_chronic_offenders_partition_per_ckpt_2026-05-04.md (Job 11,
2026-05-04).
"""

CHRONIC_IDENTITIES = [
    "bla_bla_chow",
    "bla_bla_chow__s2",
    "PC_Generator__s22",
    "PC_Generator__s45",
    "roy_d",
    "Q__s6",
]


def is_chronic(base_identity: str) -> bool:
    if not isinstance(base_identity, str):
        return False
    if base_identity in CHRONIC_IDENTITIES:
        return True
    s = base_identity.lower()
    return any(cid.lower() in s for cid in CHRONIC_IDENTITIES)


def quality_band(quality: str | None) -> str:
    if isinstance(quality, str) and quality.lower() in {"hi-q", "lo-q"}:
        return quality.lower()
    return "unknown"


def make_group_id(row: dict) -> str | None:
    """Build the asymmetric group_id for a manifest row.

    Returns None when the row should be skipped from the GroupDRO term entirely
    (currently: never — even external_vcd_real is grouped as a real lane). The
    PAIR-RANK loss separately skips external_vcd_real via is_unpaired_real.

    Required row keys:
      label                 0 or 1
      method_family         e.g. 'deeplive', 'inswapper', 'real_or_unknown'
      enhancer_family       e.g. 'gpen', 'gfpgan', 'none'
      transport             e.g. 'raw_capture', 'teams_capture', 'visomaster'
      quality               'hi-q' / 'lo-q' / None
      source                coarse lane id (e.g. 'teams_real_dev', 'visomaster_v2')
      base_identity         exact identity string

    Skip rule for pair-rank: an upstream collate flag `is_unpaired_real` (set on
    rows from external_vcd_real, where there is no paired fake) must suppress
    the pair-rank loss contribution. The GroupDRO term still applies.
    """
    label = int(row["label"])
    qb = quality_band(row.get("quality"))
    transport = row.get("transport", "raw_capture")
    if label == 1:
        # Fake-side: F-B
        return (
            f"fake|{row.get('method_family', 'unknown')}"
            f"|{row.get('enhancer_family', 'none')}"
            f"|{transport}"
            f"|{qb}"
        )
    # Real-side: R-D
    chronic_tag = "chronic" if is_chronic(row.get("base_identity", "")) else "regular"
    return (
        f"real|{row.get('source', 'unknown')}"
        f"|{transport}"
        f"|{qb}"
        f"|{chronic_tag}"
    )


def build_group_id_mapping(manifest_rows) -> dict:
    """Pre-pass at config-load time. Walk all training rows, collect distinct
    group_id strings, return dict[str -> int] for trainer/mixins/group_dro.py.
    Pass this as data_params.group_id_mapping (replaces method_mapping)."""
    seen = sorted({make_group_id(r) for r in manifest_rows if make_group_id(r) is not None})
    return {g: i for i, g in enumerate(seen)}
