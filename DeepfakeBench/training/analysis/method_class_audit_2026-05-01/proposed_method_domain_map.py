"""
Proposed METHOD_DOMAIN_MAP for Phase 3 method-conditional GRL.

Phase 2C audit output (2026-05-01). 12 buckets total.

Replaces / extends the existing 4-class QUALITY_DOMAIN_MAP in:
    - data/sources/combined_paired.py:66
    - detectors/effort_detector.py:280  (QualityDomainHead.DOMAIN_MAP)

The existing map collapses all of deeplive_* + visomaster_* into domain 2
('studio_capture'), so the dor_shkedi / deeplive_enhanced cluster the trained
P17 head latched onto (Phase 1A finding) sits *inside* one undifferentiated
bucket and the GRL has nothing to attack.

This map splits:
  - deeplive  → 3 buckets (basic / enhanced / teams)        — IMPORTANT
  - visomaster → 3 buckets (inswapper / ghost / other)      — by swap-model family
  - df40       → 1 bucket  (academic clean)
  - real-side  → external_vcd_real + realpool_real
  - currently-unused enhanced/teams-recap/proper-clean lanes → reserved buckets
    so the map doesn't need re-versioning when those sources are turned on.

Lookup convention (different from QUALITY_DOMAIN_MAP):
    - QUALITY_DOMAIN_MAP keys on `sample.source` (one of:
      df40, deeplive, visomaster, deeplive_teams, external,
      visomaster_enhanced, visomaster_teams_enhanced, proper_visomaster_*).
    - METHOD_DOMAIN_MAP must key on `sample.method` (one of:
      simswap, deeplive_edge_cases, deeplive_quality_enhancement,
      visomaster_Inswapper128, deeplive_teams_<strategy>, etc.).

Therefore the lookup is a two-step:
    1. Match by exact method string in METHOD_DOMAIN_EXACT.
    2. If not found, match by method-prefix in METHOD_DOMAIN_PREFIX.
    3. Fall back to source-string in METHOD_DOMAIN_SOURCE_FALLBACK.
"""

# Bucket index → human-readable name.
# Used by the GRL classifier head as `quality_domain_count`. Pass
# `quality_domain_count: 12` in the Phase 3 yaml.
METHOD_DOMAIN_NAMES = {
    0:  "df40",                         # academic clean
    1:  "deeplive_basic",                # edge_cases + minimal_processing
    2:  "deeplive_enhanced",             # quality_enhancement + *_enhanced
    3:  "deeplive_teams",                # Teams passthrough
    4:  "visomaster_inswapper",          # Inswapper128 + SimSwap512
    5:  "visomaster_ghost",              # GhostFace-v1/v2/v3
    6:  "visomaster_other",              # CSCS + InStyleSwapper256-A/B/C
    7:  "visomaster_enhanced",           # visomaster_enhanced_<enhancer>
    8:  "visomaster_teams_recap",        # visomaster_teams_enhanced + proper_*_teams
    9:  "proper_visomaster_clean",       # proper_visomaster_clean + enhanced_clean
    10: "external_vcd_real",             # VCD webcam reals
    11: "realpool_real",                 # DeepLive/VisoMaster reals merged
}

# ============================================================================
# Step 1: exact-method-string lookup
# ============================================================================
# Format: sample.method (lowercased+normalized via normalize_method_name) →
# bucket index. df40 method names + visomaster swap_models go here.

METHOD_DOMAIN_EXACT = {
    # df40 (bucket 0); P14/P15 enables 7 of 17 methods. All df40 methods → bucket 0.
    "simswap":     0,
    "facedancer":  0,
    "blendface":   0,
    "e4s":         0,
    "inswap":      0,
    "mobileswap":  0,
    "uniface":     0,
    "faceswap":    0,
    "mraa":        0,
    "danet":       0,
    "facevid2vid": 0,
    "fomm":        0,
    "fsgan":       0,
    "lia":         0,
    "mcnet":       0,
    "one_shot_free": 0,
    "pirender":    0,

    # deeplive_basic (bucket 1)
    "deeplive_edge_cases":          1,
    "deeplive_minimal_processing":  1,

    # deeplive_enhanced (bucket 2)
    "deeplive_quality_enhancement":          2,
    "deeplive_edge_cases_enhanced":          2,
    "deeplive_minimal_processing_enhanced":  2,

    # visomaster_inswapper (bucket 4) — Inswapper128 + SimSwap512 are most-used
    # commercial swaps; cluster them.
    "visomaster_inswapper128":  4,
    "visomaster_simswap512":    4,

    # visomaster_ghost (bucket 5) — GhostFace family (3 versions)
    "visomaster_ghostface_v1":  5,
    "visomaster_ghostface_v2":  5,
    "visomaster_ghostface_v3":  5,

    # visomaster_other (bucket 6) — CSCS + InStyleSwapper256 family
    "visomaster_cscs":              6,
    "visomaster_instyleswapper256_a":  6,
    "visomaster_instyleswapper256_b":  6,
    "visomaster_instyleswapper256_c":  6,

    # external_vcd_real (bucket 10) — reals only
    "external_vcd_real":  10,
}


# ============================================================================
# Step 2: prefix-based lookup
# ============================================================================
# Format: method.startswith(prefix) → bucket index. Used for variable suffixes
# (deeplive_teams_<strategy>, visomaster_enhanced_<enhancer>, proper_*).
# IMPORTANT: order matters — longer/more-specific prefixes must be checked
# BEFORE shorter ones (e.g. visomaster_enhanced_ before visomaster_).

METHOD_DOMAIN_PREFIX = [
    # deeplive_teams_<strategy> → bucket 3
    ("deeplive_teams_",                  3),

    # visomaster_enhanced_<enhancer> → bucket 7
    ("visomaster_enhanced_",             7),

    # visomaster_teams_enhanced (companion to v2 + Teams-recapture) → bucket 8
    ("visomaster_teams_enhanced",        8),

    # proper_visomaster_*_teams variants → bucket 8 (Teams transport)
    ("proper_visomaster_teams__",        8),
    ("proper_visomaster_enhanced_teams__", 8),

    # proper_visomaster_*_clean variants → bucket 9 (clean transport)
    ("proper_visomaster_clean__",        9),
    ("proper_visomaster_enhanced_clean__", 9),
]


# ============================================================================
# Step 3: source-string fallback (for samples where method doesn't match exact
# or prefix, fall back to bucket by source).
# ============================================================================

METHOD_DOMAIN_SOURCE_FALLBACK = {
    "df40":                          0,
    "deeplive":                      1,   # ambiguous default; prefer exact lookup
    "visomaster":                    6,   # ambiguous default; prefer exact lookup
    "deeplive_teams":                3,
    "visomaster_enhanced":           7,
    "visomaster_teams_enhanced":     8,
    "proper_visomaster_clean":       9,
    "proper_visomaster_enhanced_clean": 9,
    "proper_visomaster_teams":       8,
    "proper_visomaster_enhanced_teams": 8,
    "external":                      10,
}


# ============================================================================
# Real-side reverse mapping: every fake bucket has a corresponding real bucket
# determined at iterator time, NOT here. The iterators in combined_paired.py
# emit real frames with `source` set to the same source string as the fake;
# the real-frame method-domain lookup follows the same map but maps to:
#   - external (label=0) → bucket 10 (external_vcd_real)
#   - df40 (label=0) → bucket 0 (df40)  -- the bucket carries paired real+fake
#   - all other (label=0) → bucket 11 (realpool_real)
# This convention should be enforced in the iterator wrapper, not in this map.
# ============================================================================


def lookup_method_domain(method: str, source: str) -> int:
    """
    Map a (method, source) pair to a bucket index in 0..11.

    Args:
        method: sample.method string (e.g., "deeplive_quality_enhancement",
                "visomaster_GhostFace-v1", "simswap", "deeplive_teams_dor_shkedi_s16").
        source: sample.source string (e.g., "deeplive", "visomaster",
                "deeplive_teams", "df40", "external").

    Returns:
        Bucket index 0..11 per METHOD_DOMAIN_NAMES.
    """
    # Normalize: lowercase + replace spaces/hyphens with underscores. Matches
    # utils.grouping.normalize_method_name behavior.
    method_norm = (method or "").strip().lower().replace(" ", "_").replace("-", "_")
    source_norm = (source or "").strip().lower().replace(" ", "_").replace("-", "_")

    # Step 1: exact-method match
    if method_norm in METHOD_DOMAIN_EXACT:
        return METHOD_DOMAIN_EXACT[method_norm]

    # Step 2: prefix match (longest-first iteration in METHOD_DOMAIN_PREFIX)
    for prefix, bucket in METHOD_DOMAIN_PREFIX:
        if method_norm.startswith(prefix):
            return bucket

    # Step 3: source fallback
    if source_norm in METHOD_DOMAIN_SOURCE_FALLBACK:
        return METHOD_DOMAIN_SOURCE_FALLBACK[source_norm]

    # Last-resort: bucket 0 (clean academic) — should not happen if data
    # discovery is well-typed; trainer should log unknown-method warnings.
    return 0


# Keep a sorted list of (method_pattern, bucket) for documentation / unit
# testing.
__all__ = [
    "METHOD_DOMAIN_NAMES",
    "METHOD_DOMAIN_EXACT",
    "METHOD_DOMAIN_PREFIX",
    "METHOD_DOMAIN_SOURCE_FALLBACK",
    "lookup_method_domain",
]
