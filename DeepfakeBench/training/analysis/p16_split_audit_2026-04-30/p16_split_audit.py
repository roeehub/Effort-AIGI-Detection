"""§10.5 split-audit gate for R13_P16_DATA_AXIS (2026-04-30).

Run BEFORE image rebuild + Vertex launch. Blocks the launch if any of the
mandatory disjointness checks fails.

Scope (limited by what local manifests permit):

The eval suite ``visomaster_enhanced_macro_dev`` lives at
``arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json``,
under slice ``visomaster_enhanced_macro``, split ``dev``. Its 550 videos sit
in bucket ``teams-faces-data-test-2914-fake-4420-real-feb-28`` with frames
named ``visomaster_enhanced_{raw,teams}__frame_*_seq*.png``. Each frame
carries a ``sequence_id`` (e.g. ``seq12349``) and a numeric-hash ``identity``
field, but NOT a person-identity label — ``identity_key`` is the lane name
(``visomaster_enhanced_raw`` / ``_teams``), not a per-person ID. PLAN.md §10.5
specifies ``(identity, capture_session, source_bucket)`` as the joint group
key; this manifest doesn't carry the person-identity component, so the
audit here is the strongest *local* check we can build:

    1. **bucket disjointness** (mandatory). Training sources for
       ``visomaster_teams_enhanced`` resolve to companion buckets
       ``enhanced-visomaster-cropped`` and ``live-...-teams-v2`` style. None
       of these may equal the eval bucket
       ``teams-faces-data-test-2914-fake-4420-real-feb-28``.
    2. **sequence_id disjointness** (mandatory). No training sample_id may
       contain any eval ``sequence_id`` string (``seq12349`` etc.). This
       catches the case where a Teams play of the eval segments was
       accidentally repackaged into the training source.
    3. **frame_path exact disjointness** (mandatory). Self-explanatory; the
       intersection of training and eval frame paths must be empty.

The training resolver manifest at
``gs://training-job-outputs/cache/visomaster/enhanced_visomaster_resolver_2026-04-06.json``
must be cached locally at
``analysis/p16_split_audit_2026-04-30/_cache/enhanced_visomaster_resolver_2026-04-06.json``
before this script runs; fetch with::

    gsutil cp gs://training-job-outputs/cache/visomaster/enhanced_visomaster_resolver_2026-04-06.json \\
      analysis/p16_split_audit_2026-04-30/_cache/

If the resolver cache is missing, the script reports SKIPPED for the
training-side checks and exits 2. Exit codes:

    0 — all checks passed; launch is permitted.
    1 — at least one check failed; launch BLOCKED.
    2 — resolver manifest not cached; rerun after gsutil cp.

Outputs:
    analysis/p16_split_audit_2026-04-30/outputs/audit_report.json
    analysis/p16_split_audit_2026-04-30/outputs/audit_report.md (human-readable)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_MANIFEST = REPO_ROOT / "arena" / "manifests" / "teams_target_domain_manifest_2026-04-06_frozen.json"
RESOLVER_CACHE = (
    REPO_ROOT / "analysis" / "p16_split_audit_2026-04-30" / "_cache" / "enhanced_visomaster_resolver_2026-04-06.json"
)
OUTPUT_DIR = REPO_ROOT / "analysis" / "p16_split_audit_2026-04-30" / "outputs"

EVAL_SLICE = "visomaster_enhanced_macro"
EVAL_SPLIT = "dev"

logger = logging.getLogger("p16-split-audit")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    overlap_examples: list = field(default_factory=list)


def _bucket_from_gs_uri(uri: str) -> str:
    if not uri.startswith("gs://"):
        return ""
    return uri[len("gs://") :].split("/", 1)[0]


def load_eval_payload() -> dict:
    if not EVAL_MANIFEST.exists():
        raise FileNotFoundError(f"Eval-suite manifest missing: {EVAL_MANIFEST}")
    with open(EVAL_MANIFEST, "r") as f:
        return json.load(f)


def filter_eval_videos(payload: dict) -> list:
    videos = payload.get("videos") or []
    out = []
    for v in videos:
        slices = v.get("slices") or []
        if EVAL_SLICE in slices and v.get("split") == EVAL_SPLIT:
            out.append(v)
    return out


def extract_eval_sequence_ids(videos: Iterable[dict]) -> set:
    out = set()
    for v in videos:
        seq = v.get("sequence_id")
        if seq:
            out.add(str(seq).strip())
    return out


def extract_eval_buckets(videos: Iterable[dict]) -> set:
    out = set()
    for v in videos:
        for fp in v.get("frame_paths") or []:
            b = _bucket_from_gs_uri(fp)
            if b:
                out.add(b)
    return out


def extract_eval_frame_paths(videos: Iterable[dict]) -> set:
    out = set()
    for v in videos:
        for fp in v.get("frame_paths") or []:
            out.add(fp.strip())
    return out


def load_resolver_payload() -> dict | None:
    if not RESOLVER_CACHE.exists():
        return None
    with open(RESOLVER_CACHE, "r") as f:
        return json.load(f)


def extract_train_sample_ids(payload: dict) -> set:
    rows = payload.get("rows") or []
    out = set()
    for row in rows:
        sid = str(row.get("sample_id") or "").strip()
        if sid:
            out.add(sid)
    return out


def extract_train_buckets(payload: dict) -> set:
    rows = payload.get("rows") or []
    out = set()
    for row in rows:
        b = str(row.get("resolved_companion_bucket") or "").strip()
        if b:
            out.add(b)
        # Defensive: also any frame_path-bucket fields
        for k in ("real_frame_paths", "fake_frame_paths"):
            paths = row.get(k) or []
            for p in paths:
                bb = _bucket_from_gs_uri(str(p))
                if bb:
                    out.add(bb)
    return out


def extract_train_frame_paths(payload: dict) -> set:
    rows = payload.get("rows") or []
    out = set()
    for row in rows:
        for k in ("real_frame_paths", "fake_frame_paths"):
            for p in row.get(k) or []:
                if p:
                    out.add(str(p).strip())
    return out


def check_bucket_disjointness(
    train_buckets: set, eval_buckets: set
) -> CheckResult:
    overlap = train_buckets & eval_buckets
    return CheckResult(
        name="bucket_disjointness",
        passed=(len(overlap) == 0),
        detail=(
            f"train_buckets={len(train_buckets)}, eval_buckets={len(eval_buckets)}, "
            f"overlap={len(overlap)}"
        ),
        overlap_examples=sorted(overlap)[:10],
    )


def check_sequence_id_disjointness(
    train_sample_ids: set, eval_sequence_ids: set
) -> CheckResult:
    """No training sample_id may contain any eval sequence_id as a substring."""
    bad = []
    for seq in eval_sequence_ids:
        for sid in train_sample_ids:
            if seq in sid:
                bad.append((seq, sid))
                if len(bad) >= 20:
                    break
        if len(bad) >= 20:
            break
    return CheckResult(
        name="sequence_id_disjointness",
        passed=(len(bad) == 0),
        detail=(
            f"n_eval_sequence_ids={len(eval_sequence_ids)}, "
            f"n_train_sample_ids={len(train_sample_ids)}, "
            f"matched_pairs={'>=20 (truncated)' if len(bad) >= 20 else len(bad)}"
        ),
        overlap_examples=[f"{seq} in sample_id {sid}" for seq, sid in bad[:10]],
    )


def check_frame_path_disjointness(train_paths: set, eval_paths: set) -> CheckResult:
    overlap = train_paths & eval_paths
    return CheckResult(
        name="frame_path_disjointness",
        passed=(len(overlap) == 0),
        detail=(
            f"train_frame_paths={len(train_paths)}, eval_frame_paths={len(eval_paths)}, "
            f"overlap={len(overlap)}"
        ),
        overlap_examples=sorted(overlap)[:10],
    )


def write_outputs(results: list[CheckResult], context: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "audit": "p16_split_audit",
        "date": "2026-04-30",
        "eval_manifest": str(EVAL_MANIFEST.relative_to(REPO_ROOT)),
        "eval_slice": EVAL_SLICE,
        "eval_split": EVAL_SPLIT,
        "resolver_cache": str(RESOLVER_CACHE.relative_to(REPO_ROOT)),
        "context": context,
        "checks": [
            {
                "name": r.name,
                "passed": r.passed,
                "detail": r.detail,
                "overlap_examples": r.overlap_examples,
            }
            for r in results
        ],
        "verdict": "PASS" if all(r.passed for r in results) else "FAIL",
    }
    with open(OUTPUT_DIR / "audit_report.json", "w") as f:
        json.dump(payload, f, indent=2)

    md_lines = [
        "# P16 Split Audit Report (§10.5)",
        "",
        f"**Date**: 2026-04-30",
        f"**Eval manifest**: `{EVAL_MANIFEST.name}`",
        f"**Eval slice/split**: `{EVAL_SLICE}` / `{EVAL_SPLIT}`",
        f"**Resolver cache**: `{RESOLVER_CACHE.name}`",
        "",
        f"## Verdict: **{payload['verdict']}**",
        "",
    ]
    for k, v in context.items():
        md_lines.append(f"- {k}: {v}")
    md_lines.append("")
    md_lines.append("## Checks")
    md_lines.append("")
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        md_lines.append(f"### {status} — {r.name}")
        md_lines.append("")
        md_lines.append(f"{r.detail}")
        if r.overlap_examples:
            md_lines.append("")
            md_lines.append("Overlap examples (truncated to 10):")
            md_lines.append("")
            for ex in r.overlap_examples:
                md_lines.append(f"- `{ex}`")
        md_lines.append("")
    with open(OUTPUT_DIR / "audit_report.md", "w") as f:
        f.write("\n".join(md_lines))


def run_audit() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    logger.info("Loading eval manifest: %s", EVAL_MANIFEST)
    eval_payload = load_eval_payload()
    eval_videos = filter_eval_videos(eval_payload)
    if not eval_videos:
        logger.error("No eval videos matched slice=%s split=%s", EVAL_SLICE, EVAL_SPLIT)
        return 1

    eval_seq_ids = extract_eval_sequence_ids(eval_videos)
    eval_buckets = extract_eval_buckets(eval_videos)
    eval_frame_paths = extract_eval_frame_paths(eval_videos)
    logger.info(
        "eval: n_videos=%d, n_seq_ids=%d, n_buckets=%d, n_frame_paths=%d",
        len(eval_videos), len(eval_seq_ids), len(eval_buckets), len(eval_frame_paths),
    )

    if not RESOLVER_CACHE.exists():
        logger.error("Resolver manifest cache missing: %s", RESOLVER_CACHE)
        logger.error(
            "Fetch via: gsutil cp gs://training-job-outputs/cache/visomaster/"
            "enhanced_visomaster_resolver_2026-04-06.json %s/",
            RESOLVER_CACHE.parent,
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_DIR / "audit_report.json", "w") as f:
            json.dump({"verdict": "SKIPPED", "reason": "resolver-cache-missing"}, f, indent=2)
        return 2

    resolver_payload = load_resolver_payload()
    if resolver_payload is None:
        logger.error("Resolver payload empty/invalid")
        return 2

    train_sample_ids = extract_train_sample_ids(resolver_payload)
    train_buckets = extract_train_buckets(resolver_payload)
    train_frame_paths = extract_train_frame_paths(resolver_payload)
    logger.info(
        "train: n_sample_ids=%d, n_buckets=%d, n_frame_paths=%d",
        len(train_sample_ids), len(train_buckets), len(train_frame_paths),
    )

    results: list[CheckResult] = [
        check_bucket_disjointness(train_buckets, eval_buckets),
        check_sequence_id_disjointness(train_sample_ids, eval_seq_ids),
        check_frame_path_disjointness(train_frame_paths, eval_frame_paths),
    ]

    context = {
        "n_eval_videos": len(eval_videos),
        "n_eval_sequence_ids": len(eval_seq_ids),
        "n_eval_buckets": len(eval_buckets),
        "n_eval_frame_paths": len(eval_frame_paths),
        "n_train_sample_ids": len(train_sample_ids),
        "n_train_buckets": len(train_buckets),
        "n_train_frame_paths": len(train_frame_paths),
    }
    write_outputs(results, context)

    print()
    print("=" * 78)
    print("P16 SPLIT AUDIT (§10.5)  --  eval={}/{}  resolver={}".format(
        EVAL_SLICE, EVAL_SPLIT, RESOLVER_CACHE.name))
    print("=" * 78)
    for r in results:
        marker = "PASS" if r.passed else "FAIL"
        print(f"  [{marker}]  {r.name}: {r.detail}")
        for ex in r.overlap_examples[:3]:
            print(f"           e.g. {ex}")
    verdict = "PASS" if all(r.passed for r in results) else "FAIL"
    print("-" * 78)
    print(f"  VERDICT: {verdict}")
    print(f"  outputs in {OUTPUT_DIR}")
    print("=" * 78)
    return 0 if verdict == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="P16 split audit (§10.5)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    return run_audit()


if __name__ == "__main__":
    raise SystemExit(main())
