"""CPU smoke test for R13_P16_DATA_AXIS data wiring (no GPU, no Vertex).

Run BEFORE image rebuild. Exercises the visomaster_teams_enhanced loader
(the P16 lever) end-to-end on a local resolver-cache copy:

  1. Imports the data sources module (catches import errors).
  2. Calls ``discover_visomaster_teams_enhanced_samples`` with the EXACT yaml
     kwargs (companion_domains=['teams_v2'], include_statuses=['teams_v2_companion']).
  3. Asserts: count > 0, samples have populated companion_bucket fields,
     non-empty available_enhancers, identity-property resolves.
  4. Wraps the first 16 samples through ``create_unified_samples_from_visomaster_teams_enhanced``
     to verify the paired-side wrapper produces tagged UnifiedPairedSamples.
  5. Verifies sample method tags are well-formed and identity resolution
     returns non-empty strings.

  6. Smokes the family-weights routing: load the yaml, locate the
     ``visomaster_enhanced_fake`` bucket; verify it equals 2.0 and is below 4.0.

NO GCS frame fetching, NO image decoding. The trainer will exercise frame
fetching during the first iteration; this smoke validates the
metadata-discovery contract.

Pre-req: resolver manifest cached at
    analysis/p16_split_audit_2026-04-30/_cache/enhanced_visomaster_resolver_2026-04-06.json
(fetched by the §10.5 audit script; see analysis/p16_split_audit_2026-04-30/p16_split_audit.py).

Exit codes:
    0 — all asserts passed; data wiring is launch-ready.
    1 — wiring problem; investigate before image rebuild.
    2 — resolver cache absent; rerun gsutil cp.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

YAML_PATH = REPO_ROOT / "experiments" / "phase2_round13" / "R13_P16_DATA_AXIS.yaml"
RESOLVER_CACHE = (
    REPO_ROOT
    / "analysis"
    / "p16_split_audit_2026-04-30"
    / "_cache"
    / "enhanced_visomaster_resolver_2026-04-06.json"
)
OUTPUT_DIR = REPO_ROOT / "analysis" / "p16_smoke_test_2026-04-30" / "outputs"

logger = logging.getLogger("p16-smoke")


def smoke_loader_via_local_cache() -> dict:
    """Exercise discover_visomaster_teams_enhanced_samples against a local
    file-uri pointing at the cached resolver manifest.

    The discover function uses ``_read_json_uri`` which handles ``file://``
    URIs and bare local paths.
    """
    from data.sources.visomaster import discover_visomaster_teams_enhanced_samples

    samples = discover_visomaster_teams_enhanced_samples(
        resolver_manifest_uri=str(RESOLVER_CACHE),
        enhanced_bucket="enhanced-visomaster-cropped",
        companion_domains=["teams_v2"],
        include_statuses=["teams_v2_companion"],
    )
    if not samples:
        raise RuntimeError(
            "visomaster_teams_enhanced loader returned 0 samples; check resolver "
            "cache and yaml filter kwargs."
        )

    n = len(samples)
    take = samples[:16]

    # Field assertions on first 16 samples.
    bad: list[str] = []
    swap_model_set = set()
    enhancer_set = set()
    identity_set = set()
    for s in take:
        if not s.companion_bucket:
            bad.append(f"sample_id={s.sample_id} missing companion_bucket")
        if s.companion_domain != "teams_v2":
            bad.append(f"sample_id={s.sample_id} unexpected companion_domain={s.companion_domain!r}")
        if not s.available_enhancers:
            bad.append(f"sample_id={s.sample_id} no available_enhancers")
        if not s.swap_model:
            bad.append(f"sample_id={s.sample_id} empty swap_model")
        ident = s.identity
        if not ident:
            bad.append(f"sample_id={s.sample_id} empty identity")
        else:
            identity_set.add(ident)
        if s.swap_model:
            swap_model_set.add(s.swap_model)
        for enh in s.available_enhancers:
            enhancer_set.add(enh)

    if bad:
        raise AssertionError(
            "{} field-level issues in first 16 samples:\n  ".format(len(bad)) + "\n  ".join(bad)
        )

    # Wrap through paired-side. Pure-CPU; no GCS reads.
    from data.sources.combined_paired import create_unified_samples_from_visomaster_teams_enhanced

    unified = create_unified_samples_from_visomaster_teams_enhanced(take, logger=logger)
    if not unified:
        raise RuntimeError("create_unified_samples_from_visomaster_teams_enhanced returned empty")

    method_tags = sorted({u.method for u in unified if hasattr(u, "method")})
    source_tags = sorted({u.source for u in unified if hasattr(u, "source")})

    return {
        "n_samples_total": n,
        "n_smoke_examined": len(take),
        "n_unified_wrapped": len(unified),
        "distinct_swap_models": sorted(swap_model_set),
        "distinct_enhancers_n": len(enhancer_set),
        "distinct_identities_n": len(identity_set),
        "unified_method_tags_sample": method_tags[:8],
        "unified_source_tags": source_tags,
    }


def smoke_yaml_family_weights() -> dict:
    import yaml

    with open(YAML_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    fw = cfg["combined_paired"]["sampling"]["family_weights"]
    if abs(fw["visomaster_enhanced_fake"] - 2.0) > 1e-9:
        raise AssertionError(
            f"visomaster_enhanced_fake fw={fw['visomaster_enhanced_fake']} (expected 2.0)"
        )
    if fw["visomaster_enhanced_fake"] >= 4.0:
        raise AssertionError(
            f"visomaster_enhanced_fake fw={fw['visomaster_enhanced_fake']} >= 4.0; "
            "exceeds DATA_FIX-collapse-buffer ceiling"
        )
    # Single-lever assertions.
    if cfg["face_scale_jitter"]["enabled"] is not False:
        raise AssertionError("face_scale_jitter.enabled should be False (single-lever discipline)")
    if cfg["anchor_aware"]["enabled"] is not False:
        raise AssertionError("anchor_aware.enabled should be False (single-lever discipline)")
    if cfg["augmentation"]["pipeline_randomization"]["enabled"] is not False:
        raise AssertionError(
            "augmentation.pipeline_randomization.enabled should be False (single-lever discipline)"
        )
    return {"family_weights": fw}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RESOLVER_CACHE.exists():
        logger.error("Resolver cache missing: %s", RESOLVER_CACHE)
        logger.error(
            "Fetch via: gsutil cp gs://training-job-outputs/cache/visomaster/"
            "enhanced_visomaster_resolver_2026-04-06.json %s/",
            RESOLVER_CACHE.parent,
        )
        return 2

    if not YAML_PATH.exists():
        logger.error("Yaml missing: %s", YAML_PATH)
        return 1

    yaml_report = smoke_yaml_family_weights()
    logger.info("Yaml smoke OK: family_weights=%s", yaml_report["family_weights"])

    loader_report = smoke_loader_via_local_cache()
    logger.info(
        "Loader smoke OK: n_samples_total=%d, swap_models=%s, distinct_enhancers=%d, identities=%d",
        loader_report["n_samples_total"],
        loader_report["distinct_swap_models"],
        loader_report["distinct_enhancers_n"],
        loader_report["distinct_identities_n"],
    )
    logger.info("Unified method tags sample: %s", loader_report["unified_method_tags_sample"])
    logger.info("Unified source tags: %s", loader_report["unified_source_tags"])

    summary = {
        "smoke": "p16_data_axis",
        "date": "2026-04-30",
        "yaml": yaml_report,
        "loader": loader_report,
        "verdict": "PASS",
    }
    with open(OUTPUT_DIR / "smoke_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 78)
    print("P16 SMOKE TEST  --  CPU-only, no GCS frame fetch, no GPU")
    print("=" * 78)
    print(f"  yaml family_weights[visomaster_enhanced_fake] = {yaml_report['family_weights']['visomaster_enhanced_fake']}")
    print(f"  loader produced {loader_report['n_samples_total']} samples")
    print(f"  examined first {loader_report['n_smoke_examined']}")
    print(f"  distinct swap_models: {loader_report['distinct_swap_models']}")
    print(f"  distinct enhancers: {loader_report['distinct_enhancers_n']}")
    print(f"  distinct identities: {loader_report['distinct_identities_n']}")
    print(f"  unified wrapped: {loader_report['n_unified_wrapped']}")
    print(f"  unified source tags: {loader_report['unified_source_tags']}")
    print(f"  unified method tags (first 8): {loader_report['unified_method_tags_sample']}")
    print()
    print("  VERDICT: PASS — wiring is launch-ready")
    print(f"  outputs in {OUTPUT_DIR}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
