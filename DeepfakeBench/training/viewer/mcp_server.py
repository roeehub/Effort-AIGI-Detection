"""
MCP Server for the Training Data Viewer.

Exposes the training dataset to LLM agents (Copilot, Claude) via
Model Context Protocol tools. Wraps the existing Flask REST API.

Usage:
    python -m viewer.mcp_server                    # default: http://127.0.0.1:8501
    python -m viewer.mcp_server --api-url http://localhost:8501

Requires the Flask viewer server to be running.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

# ── Server Setup ─────────────────────────────────────────────────────────────

mcp = FastMCP(
    "training-data-viewer",
    instructions=textwrap.dedent("""\
        Training Data Viewer MCP Server — Deepfake/AIGI detection dataset explorer.

        This server provides tools to inspect the training data for the Effort
        deepfake detector (ICML 2025). The dataset composition depends on the
        active config plus any policy overlays applied by the local viewer. In
        particular, some historical bad `visomaster_*` rows may be relabeled
        into weak-signal lanes such as `visomaster_hints` and
        `visomaster_hints_teams` instead of being exposed as normal
        method-faithful VisoMaster data. Each sample is a video clip with
        multiple cropped face frames (real or fake).

        **Workflow for exploring the data:**
        1. Start with `get_dataset_overview` to understand the overall composition
        2. Use `list_methods` or `get_data_health` to find methods of interest
        3. Drill into a method with `get_method_detail` for identity/split/tier stats
        4. Browse samples with `browse_samples` (filterable by source, method, label, split)
        5. Inspect a single sample + its real↔fake pair with `get_sample_detail`

        **Key concepts:**
        - **Source**: The training lane exposed by the viewer (df40, deeplive,
          visomaster_hints, deeplive_teams, etc.)
        - **Method**: The deepfake generation technique (simswap, blendface, visomaster_CSCS, etc.)
        - **Family**: Sampling group with assigned weight for training balance
        - **Identity**: Person identity — used for train/val/test splitting
        - **Tier**: Quality tier (visomaster only): STRONG, USABLE, ARTIFACT
        - **Pair**: Most fake samples have a paired real counterpart from the same identity
    """),
)

API_BASE = "http://127.0.0.1:8501"


def _api_get(path: str, params: Optional[Dict[str, str]] = None) -> Any:
    """Call the Flask viewer API and return parsed JSON."""
    url = f"{API_BASE}{path}"
    if params:
        qs = "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in params.items() if v)
        if qs:
            url += f"?{qs}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        return {"error": f"Cannot reach viewer server at {API_BASE}. Is it running? ({e})"}
    except Exception as e:
        return {"error": str(e)}


def _check_server() -> Optional[str]:
    """Return an error message if the server isn't ready, else None."""
    status = _api_get("/api/discovery-status")
    if "error" in status:
        return status["error"]
    if not status.get("done"):
        return "Discovery still in progress. Try again shortly."
    return None


# ── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_dataset_overview() -> str:
    """Get a high-level overview of the training dataset: total samples,
    breakdown by source, label, family, split, training weights, and
    effective sampling distribution. Start here to understand the data."""
    err = _check_server()
    if err:
        return err

    stats = _api_get("/api/stats")
    if "error" in stats:
        return json.dumps(stats)

    lines = [
        f"# Training Dataset Overview",
        f"**Total samples:** {stats['total']:,}",
        "",
        "## By Source",
    ]
    for src, cnt in sorted(stats["by_source"].items()):
        label_info = stats["by_source_label"].get(src, {})
        ids = stats["identities_per_source"].get(src, 0)
        lines.append(f"- **{src}**: {cnt:,} samples ({label_info.get('real', 0)} real, "
                      f"{label_info.get('fake', 0)} fake) — {ids} identities")

    lines += ["", "## By Label"]
    bl = stats["by_label"]
    lines.append(f"- Real: {bl.get('real', 0):,}  |  Fake: {bl.get('fake', 0):,}")

    lines += ["", "## By Split"]
    for split, cnt in sorted(stats["by_split"].items()):
        if split:
            lines.append(f"- {split}: {cnt:,}")

    lines += ["", "## Family Weights (training sampling)"]
    for fam, w in sorted(stats.get("family_weights", {}).items()):
        cnt = stats.get("by_family", {}).get(fam, 0)
        eff = stats.get("effective_pct", {}).get(fam, 0)
        lines.append(f"- **{fam}**: weight={w}, count={cnt:,}, effective={eff:.1f}%")

    lines += [
        "",
        f"## Target Domain",
        f"- Target domain effective %: {stats.get('target_domain_pct', 0):.1f}%",
        f"- Teams coverage: {stats.get('teams_coverage_pct', 0):.1f}% of realpool identities",
        f"- Multi-source identities: {stats.get('multi_source_identities', 0)}",
    ]

    if stats.get("swap_model_counts"):
        lines += ["", "## Swap Models"]
        for sm, cnt in sorted(stats["swap_model_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"- {sm}: {cnt:,}")

    if stats.get("tier_counts"):
        lines += ["", "## Quality Tiers (VisoMaster)"]
        for tier, cnt in sorted(stats["tier_counts"].items()):
            lines.append(f"- {tier}: {cnt:,}")

    if stats.get("enhancer_counts"):
        lines += ["", "## Enhancers"]
        for enh, cnt in sorted(stats["enhancer_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"- {enh}: {cnt:,}")

    return "\n".join(lines)


@mcp.tool()
def list_methods() -> str:
    """List all generation methods in the dataset with sample counts,
    source, family, training weight, and health status.
    Use this to see what deepfake methods are available."""
    err = _check_server()
    if err:
        return err

    stats = _api_get("/api/stats")
    if "error" in stats:
        return json.dumps(stats)

    health = _api_get("/api/method-health")
    health_map = {}
    if "methods" in health:
        health_map = {m["method"]: m for m in health["methods"]}

    lines = [f"# Methods ({len(stats.get('by_method', {}))} total)", ""]
    lines.append("| Method | Source | Family | Total | Real | Fake | Weight | Health |")
    lines.append("|--------|--------|--------|------:|-----:|-----:|-------:|--------|")

    for method, count in sorted(stats.get("by_method", {}).items(), key=lambda x: -x[1]):
        h = health_map.get(method, {})
        src = h.get("source", "?")
        fam = h.get("family", "?")
        real = h.get("real", 0)
        fake = h.get("fake", 0)
        weight = h.get("weight", 1.0)
        status = h.get("health", "?")
        icon = {"green": "✅", "yellow": "⚠️", "red": "🔴"}.get(status, "?")
        lines.append(f"| {method} | {src} | {fam} | {count:,} | {real:,} | {fake:,} "
                      f"| {weight} | {icon} {status} |")

    return "\n".join(lines)


@mcp.tool()
def get_method_detail(method: str) -> str:
    """Get detailed statistics for a single deepfake generation method:
    identity count, train/val/test split, frame stats, tier distribution,
    balance ratio, health issues, and swap model breakdown.

    Args:
        method: The method name (e.g. 'simswap', 'visomaster_CSCS', 'deeplive_edge_cases')
    """
    err = _check_server()
    if err:
        return err

    health = _api_get("/api/method-health")
    if "error" in health:
        return json.dumps(health)

    target = None
    for m in health.get("methods", []):
        if m["method"] == method:
            target = m
            break

    if not target:
        available = [m["method"] for m in health.get("methods", [])]
        return f"Method '{method}' not found. Available: {', '.join(sorted(available))}"

    m = target
    lines = [
        f"# Method: {m['method']}",
        f"- **Source**: {m['source']}",
        f"- **Family**: {m['family']} (weight: {m['weight']})",
        f"- **Health**: {m['health']} — issues: {', '.join(m['issues']) if m['issues'] else 'none'}",
        "",
        "## Sample Counts",
        f"- Total: {m['total']:,} (Real: {m['real']:,}, Fake: {m['fake']:,})",
        f"- Balance ratio: {m['balance_ratio']:.3f} (0=balanced, 1=one-sided)",
        f"- Train: {m['train']:,} | Val: {m['val']:,} | Test: {m['test']:,}",
        "",
        "## Identity & Frame Stats",
        f"- Unique identities: {m['n_identities']}",
        f"- Samples per identity: {m['samples_per_identity']}",
        f"- Average frames: {m['avg_frames']} (min: {m['min_frames']}, max: {m['max_frames']})",
    ]

    if m.get("tier_counts"):
        lines += ["", "## Quality Tiers"]
        for tier, cnt in sorted(m["tier_counts"].items()):
            lines.append(f"- {tier}: {cnt}")

    if m.get("swap_model_counts"):
        lines += ["", "## Swap Models"]
        for sm, cnt in sorted(m["swap_model_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"- {sm}: {cnt}")

    if m.get("enhancer_counts"):
        lines += ["", "## Enhancers"]
        for enh, cnt in sorted(m["enhancer_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"- {enh}: {cnt}")

    return "\n".join(lines)


@mcp.tool()
def get_data_health() -> str:
    """Get a data quality health report across all methods.
    Shows health status (green/yellow/red) and specific issues for each method.
    Useful for identifying data quality problems."""
    err = _check_server()
    if err:
        return err

    health = _api_get("/api/method-health")
    if "error" in health:
        return json.dumps(health)

    methods = health.get("methods", [])
    red = [m for m in methods if m["health"] == "red"]
    yellow = [m for m in methods if m["health"] == "yellow"]
    green = [m for m in methods if m["health"] == "green"]

    lines = [
        f"# Data Health Report",
        f"**Total methods**: {len(methods)} — 🔴 {len(red)} red, ⚠️ {len(yellow)} yellow, ✅ {len(green)} green",
        "",
    ]

    if red:
        lines.append("## 🔴 Critical Issues")
        for m in red:
            lines.append(f"- **{m['method']}** ({m['source']}, {m['total']:,} samples): "
                          f"{', '.join(m['issues'])}")
        lines.append("")

    if yellow:
        lines.append("## ⚠️ Warnings")
        for m in yellow:
            lines.append(f"- **{m['method']}** ({m['source']}, {m['total']:,} samples): "
                          f"{', '.join(m['issues'])}")
        lines.append("")

    if green:
        lines.append("## ✅ Healthy")
        for m in green:
            lines.append(f"- **{m['method']}** ({m['source']}, {m['total']:,} samples)")

    return "\n".join(lines)


@mcp.tool()
def browse_samples(
    method: str = "",
    source: str = "",
    label: str = "",
    split: str = "",
    identity: str = "",
    search: str = "",
    page: int = 1,
    per_page: int = 20,
) -> str:
    """Browse training samples with filters. Returns a paginated list of samples.

    Args:
        method: Filter by method name (e.g. 'simswap', 'visomaster_CSCS')
        source: Filter by source (df40, deeplive, visomaster, deeplive_teams, visomaster_teams_enhanced)
        label: Filter by label ('0' for real, '1' for fake)
        split: Filter by split (train, val_in_dist, test)
        identity: Filter by identity substring (e.g. 'df40_001', 'realpool_00197')
        search: Free-text search across sample_id, method, identity
        page: Page number (default 1)
        per_page: Results per page (default 20, max 50)
    """
    err = _check_server()
    if err:
        return err

    per_page = min(per_page, 50)
    params = {
        "method": method,
        "source": source,
        "label": label,
        "split": split,
        "identity": identity,
        "search": search,
        "page": str(page),
        "per_page": str(per_page),
    }
    data = _api_get("/api/samples", {k: v for k, v in params.items() if v})
    if "error" in data:
        return json.dumps(data)

    total = data.get("total", 0)
    samples = data.get("samples", [])

    lines = [
        f"# Samples (page {page}, showing {len(samples)} of {total:,} matching)",
        "",
    ]

    if not samples:
        lines.append("No samples match these filters.")
        return "\n".join(lines)

    for s in samples:
        label_str = "FAKE" if s.get("label") == 1 else "REAL"
        extra = s.get("extra", {})
        fam = extra.get("effective_family", "")
        swap = s.get("swap_model", "")
        tier_str = f" tier={s['tier']}" if s.get("tier") else ""
        swap_str = f" swap={swap}" if swap else ""
        enh = s.get("enhancer", "")
        enh_str = f" enhancer={enh}" if enh else ""
        pair_str = " [paired]" if s.get("has_pair") else ""

        lines.append(
            f"- **{s['sample_id']}** [{label_str}] — {s['method']} | "
            f"{s['source']} | {s.get('split', '?')} | "
            f"identity={s['identity']} | frames={s.get('frame_count', '?')}"
            f"{tier_str}{swap_str}{enh_str}{pair_str}"
        )

    if data.get("has_more"):
        lines.append(f"\n*… {total - page * per_page:,} more results. Use page={page + 1} to continue.*")

    return "\n".join(lines)


@mcp.tool()
def get_sample_detail(sample_id: str) -> str:
    """Get detailed info about a specific sample including all metadata and
    its paired real↔fake counterpart (if available).

    Args:
        sample_id: The sample ID (e.g. 'simswap__737_719__fake', 'visomaster_CSCS_00197__fake')
    """
    err = _check_server()
    if err:
        return err

    data = _api_get(f"/api/sample-detail/{sample_id}", {"frames_per": "8"})
    if "error" in data:
        return json.dumps(data)

    def _fmt_sample(s: dict, heading: str) -> list:
        lines = [
            f"## {heading}: {s['sample_id']}",
            f"- **Label**: {'FAKE' if s.get('label') == 'fake' else 'REAL'}",
            f"- **Method**: {s['method']}",
            f"- **Source**: {s['source']}",
            f"- **Identity**: {s['identity']}",
            f"- **Split**: {s.get('split', 'N/A')}",
            f"- **Frame count**: {s.get('frame_count', '?')}",
        ]
        if s.get("tier"):
            lines.append(f"- **Tier**: {s['tier']}")
        if s.get("swap_model"):
            lines.append(f"- **Swap model**: {s['swap_model']}")
        if s.get("enhancer"):
            lines.append(f"- **Enhancer**: {s['enhancer']}")
        if s.get("strategy"):
            lines.append(f"- **Strategy**: {s['strategy']}")
        if s.get("original_video_name"):
            lines.append(f"- **Original video**: {s['original_video_name']}")
        n_frames = len(s.get("frame_urls", []))
        if n_frames:
            lines.append(f"- **Frame URLs available**: {n_frames}")
        return lines

    lines = [f"# Sample Detail"]
    lines += _fmt_sample(data, "Main Sample")

    pair = data.get("pair")
    if pair:
        lines += ["", "---"]
        lines += _fmt_sample(pair, "Paired Counterpart")
    else:
        lines += ["", "*No paired counterpart found.*"]

    return "\n".join(lines)


@mcp.tool()
def get_filter_options() -> str:
    """Get all available filter values: sources, methods, families, splits,
    tiers, and labels. Use this to know what valid filter values exist
    before calling browse_samples."""
    err = _check_server()
    if err:
        return err

    data = _api_get("/api/filter-options")
    if "error" in data:
        return json.dumps(data)

    lines = ["# Available Filters", ""]
    for key in ["sources", "methods", "families", "splits", "tiers", "labels"]:
        vals = data.get(key, [])
        lines.append(f"## {key.title()} ({len(vals)})")
        for v in vals:
            lines.append(f"- {v}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def get_method_gallery(
    method: str,
    label: str = "",
    sort: str = "random",
    page: int = 1,
    per_page: int = 10,
) -> str:
    """Browse samples from a specific method with multi-frame detail.
    Shows more frame info per sample than browse_samples.

    Args:
        method: Method name (e.g. 'simswap', 'visomaster_CSCS')
        label: Filter by 'real' or 'fake' (empty for both)
        sort: Sort order: 'random', 'identity', or 'frames'
        page: Page number
        per_page: Samples per page (default 10, max 24)
    """
    err = _check_server()
    if err:
        return err

    per_page = min(per_page, 24)
    params = {
        "label": label,
        "sort": sort,
        "page": str(page),
        "per_page": str(per_page),
        "frames_per": "4",
        "seed": "42",
    }
    data = _api_get(f"/api/method-gallery/{method}", {k: v for k, v in params.items() if v})
    if "error" in data:
        if data.get("error") == "Discovery not complete":
            return "Discovery not complete. Wait for the viewer server to finish loading."
        return json.dumps(data)

    total = data.get("total", 0)
    samples = data.get("samples", [])

    lines = [
        f"# {method} Gallery (page {page}/{data.get('total_pages', 1)}, "
        f"{len(samples)} of {total:,} samples)",
        "",
    ]

    for s in samples:
        label_str = s.get("label", "?").upper()
        tier_str = f" | tier={s['tier']}" if s.get("tier") else ""
        swap_str = f" | swap={s['swap_model']}" if s.get("swap_model") else ""
        enh_str = f" | enhancer={s['enhancer']}" if s.get("enhancer") else ""
        n_frames = len(s.get("frame_urls", []))
        lines.append(
            f"### {s['sample_id']} [{label_str}]"
        )
        lines.append(
            f"identity={s['identity']} | frames={s.get('frame_count', '?')} "
            f"(showing {n_frames}) | split={s.get('split', '?')}"
            f"{tier_str}{swap_str}{enh_str}"
        )
        lines.append("")

    if data.get("has_more"):
        remaining = total - page * per_page
        lines.append(f"*… {remaining:,} more. Use page={page + 1}.*")

    return "\n".join(lines)


# ── Resources ────────────────────────────────────────────────────────────────

@mcp.resource("training-data://overview")
def resource_overview() -> str:
    """Static overview of the training dataset composition."""
    return get_dataset_overview()


@mcp.resource("training-data://methods")
def resource_methods() -> str:
    """List of all methods in the training dataset."""
    return list_methods()


@mcp.resource("training-data://health")
def resource_health() -> str:
    """Data quality health report."""
    return get_data_health()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Training Data Viewer – MCP Server")
    parser.add_argument("--api-url", default="http://127.0.0.1:8501",
                        help="Base URL of the Flask viewer server (default: %(default)s)")
    args = parser.parse_args()

    global API_BASE
    API_BASE = args.api_url.rstrip("/")

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
