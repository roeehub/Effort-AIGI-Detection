"""D2 — Paired-data infrastructure audit.

Question: how many tuples does combined_paired.proper_data + DF40 paired +
DeepLive actually have at scale, with same-identity / same-source /
different-method semantics?

Reads:
  - dataset/df40_pairs/df40-pair-matching.json (or wherever it lives)
  - The proper-data inventory artifacts produced by `arena.build_visomaster_proper_data_artifacts`

Outputs:
  outputs/d2_paired_data_inventory.csv  — per-source counts
  outputs/d2_paired_identity_diversity.csv  — identity counts per source × split
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
OUTPUTS = THIS_DIR / "outputs"

logger = logging.getLogger("d2-paired-audit")


def find_df40_pair_json() -> Path | None:
    candidates = [
        REPO_ROOT / "dataset" / "df40_pairs" / "df40-pair-matching.json",
        REPO_ROOT / "data" / "sources" / "df40-pair-matching.json",
        REPO_ROOT / "dataset" / "df40-pair-matching.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(REPO_ROOT.rglob("df40-pair-matching.json"))
    return matches[0] if matches else None


def find_proper_inventories() -> list[Path]:
    return list(REPO_ROOT.rglob("proper_data_inventory*.yaml"))


def find_proper_manifests() -> list[Path]:
    return list((REPO_ROOT / "arena").rglob("proper_data*.yaml"))


def audit_df40_pairs() -> dict:
    p = find_df40_pair_json()
    if p is None:
        return {"path": None, "error": "df40 pair-matching json not found"}
    try:
        with open(p) as f:
            data = json.load(f)
    except Exception as exc:
        return {"path": str(p), "error": str(exc)}
    n_entries = 0
    method_counter = Counter()
    identity_counter = Counter()
    if isinstance(data, list):
        for entry in data:
            n_entries += 1
            method_counter.update([entry.get("method", "unknown")])
            identity_counter.update([entry.get("identity", "unknown")])
    elif isinstance(data, dict):
        n_entries = len(data)
        for k, v in list(data.items())[:5]:
            logger.info("df40 sample entry key=%s val_type=%s", k, type(v).__name__)
    return {
        "path": str(p),
        "n_entries": n_entries,
        "n_unique_methods": len(method_counter),
        "n_unique_identities": len(identity_counter),
        "top_methods": dict(method_counter.most_common(10)),
    }


def audit_proper_data_inventories() -> list[dict]:
    rows = []
    inventories = find_proper_inventories()
    logger.info("found %d proper_data inventory candidates", len(inventories))
    import yaml
    for inv in inventories[:20]:
        try:
            with open(inv) as f:
                obj = yaml.safe_load(f)
            if not isinstance(obj, dict):
                continue
            entries = []
            if "entries" in obj:
                entries = obj["entries"]
            elif "captures" in obj:
                entries = obj["captures"]
            elif isinstance(obj, list):
                entries = obj
            n = len(entries) if isinstance(entries, list) else 0
            identity_set = set()
            method_set = set()
            transport_set = set()
            for e in (entries or [])[:5000]:
                if isinstance(e, dict):
                    if "identity_id" in e:
                        identity_set.add(e.get("identity_id"))
                    if "method" in e:
                        method_set.add(e.get("method"))
                    if "transport" in e:
                        transport_set.add(e.get("transport"))
            rows.append({
                "path": str(inv.relative_to(REPO_ROOT)),
                "n_entries": n,
                "n_identities": len(identity_set),
                "n_methods": len(method_set),
                "transports": ",".join(sorted(transport_set)) if transport_set else "",
            })
        except Exception as exc:
            rows.append({"path": str(inv.relative_to(REPO_ROOT)), "error": str(exc)})
    return rows


def audit_combined_paired_recipes() -> list[dict]:
    """Find experimental yamls that reference combined_paired and parse the
    paired data they pull from (identifies which packets used paired-only)."""
    rows = []
    import yaml
    for yaml_path in (REPO_ROOT / "experiments").rglob("*.yaml"):
        try:
            with open(yaml_path) as f:
                obj = yaml.safe_load(f)
            if not isinstance(obj, dict):
                continue
            text = json.dumps(obj)
            if "combined_paired" in text or "proper_data" in text:
                rows.append({
                    "yaml": str(yaml_path.relative_to(REPO_ROOT)),
                    "uses_combined_paired": "combined_paired" in text,
                    "uses_proper_data": "proper_data" in text,
                })
        except Exception:
            continue
    return rows


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    summary = {}

    summary["df40_pairs"] = audit_df40_pairs()
    logger.info("df40_pairs: %s", summary["df40_pairs"])

    inv = audit_proper_data_inventories()
    summary["proper_data_inventories"] = inv
    pd.DataFrame(inv).to_csv(OUTPUTS / "d2_proper_data_inventories.csv", index=False)
    logger.info("found %d proper_data inventories", len(inv))

    combined = audit_combined_paired_recipes()
    summary["combined_paired_yamls"] = combined
    pd.DataFrame(combined).to_csv(OUTPUTS / "d2_combined_paired_yamls.csv", index=False)
    logger.info("found %d yamls referencing combined_paired/proper_data", len(combined))

    with open(OUTPUTS / "d2_paired_data_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("wrote %s", OUTPUTS / "d2_paired_data_summary.json")

    print("\n" + "=" * 80)
    print("D2 — Paired data inventory")
    print("=" * 80)
    print(f"DF40 pair file: {summary['df40_pairs'].get('path')}")
    if "n_entries" in summary["df40_pairs"]:
        print(f"  entries={summary['df40_pairs']['n_entries']}")
        print(f"  identities={summary['df40_pairs']['n_unique_identities']}")
        print(f"  methods={summary['df40_pairs']['n_unique_methods']}")
    if inv:
        print(f"\nProper-data inventories: {len(inv)}")
        for row in inv[:10]:
            print(f"  {row}")
    if combined:
        print(f"\nyamls referencing combined_paired/proper_data: {len(combined)}")
        for row in combined[:15]:
            print(f"  {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
