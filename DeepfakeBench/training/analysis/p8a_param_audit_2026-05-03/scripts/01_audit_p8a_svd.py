"""
P8A SVD-Residual Parameter Audit  (2026-05-03)
================================================

Hypothesis under test:
    The current champion checkpoint P8A (run 9lmvb5b4, step 5000, trained
    2026-04-24) was produced BEFORE the in_proj-SVD silent-zero-gradient bug
    was fixed (2026-04-26 commit 2feea58). Did the in_proj-SVD residuals
    (svd_q / svd_k / svd_v) actually learn anything, or did they sit at
    near-zero values throughout training?

Method:
    Load P8A state_dict directly with torch.load (no model class
    instantiation). For every transformer block i in [0..11], inspect the
    six SVD slots that the architecture emits:

        attn.out_proj                       (always trained: bug-free path)
        attn._svd_in_proj.svd_q             (suspect: silent-zero-grad bug)
        attn._svd_in_proj.svd_k             (suspect: silent-zero-grad bug)
        attn._svd_in_proj.svd_v             (suspect: silent-zero-grad bug)
        mlp.c_fc                            (always trained: bug-free path)
        mlp.c_proj                          (always trained: bug-free path)

    For each slot, report:
        - L2 norm of S_residual                  (the singular-value scalar)
        - L2 norm of U_residual                  (left factor)
        - L2 norm of V_residual                  (right factor)
        - L2 norm of contribution = U @ diag(S) @ V   (the actual delta to weight)
        - Ratio of contribution to the layer's out_proj contribution
        - is_zero flag (contribution L2 < 1e-6)

Verdicts:
    PASSED   = in_proj-SVD residuals are non-zero across >=80% of layers
               (lever was load-bearing; bug did not silently zero them out)
    FAILED   = in_proj residuals are zero across >=80% of layers
               (lever was effectively a no-op for P8A; explains the FT ceiling)
    PARTIAL  = mixed (some layers learned, some didn't)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
)

P8A_CKPT = (
    REPO_ROOT
    / "analysis/_features_cache_2026-04-30/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"
)

AUDIT_DIR = REPO_ROOT / "analysis/p8a_param_audit_2026-05-03"
OUT_DIR = AUDIT_DIR / "outputs"
LOG_PATH = AUDIT_DIR / "run.log"
CSV_PATH = OUT_DIR / "svd_layer_audit.csv"
JSON_PATH = OUT_DIR / "audit_summary.json"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("p8a_svd_audit")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

fh = logging.FileHandler(LOG_PATH, mode="w")
fh.setFormatter(fmt)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The six SVD slots that we expect to find on every transformer block, with
# their key suffixes. These are the keys *inside* the state_dict.
SLOTS: List[Tuple[str, str]] = [
    ("out_proj", "attn.out_proj"),
    ("svd_q", "attn._svd_in_proj.svd_q"),
    ("svd_k", "attn._svd_in_proj.svd_k"),
    ("svd_v", "attn._svd_in_proj.svd_v"),
    ("mlp.c_fc", "mlp.c_fc"),
    ("mlp.c_proj", "mlp.c_proj"),
]

# Bucket each slot into "in_proj-SVD" (the suspect / bug-affected family) vs
# "non in_proj-SVD" (the bug-free reference family). The verdict is computed
# only over the first family.
IN_PROJ_SLOTS = {"svd_q", "svd_k", "svd_v"}

ZERO_THRESHOLD = 1e-6


def unwrap_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """Handle the three common wrappers: raw state_dict, {'state_dict': ...},
    or {'model_state_dict': ...}."""
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                logger.info("checkpoint wrapped under top-level key '%s'", key)
                return obj[key]
        # Heuristic: if the dict's values are all tensors, treat it as raw.
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            logger.info("checkpoint is a raw state_dict (no wrapper)")
            return obj
    raise ValueError(
        f"Could not extract state_dict from checkpoint; top-level type={type(obj).__name__}"
    )


def base_key(layer_idx: int, slot_path: str) -> str:
    """The state_dict key prefix for a given (layer_idx, slot_path)."""
    return f"backbone.visual.transformer.resblocks.{layer_idx}.{slot_path}"


def fmt_shape(t: torch.Tensor) -> str:
    return "x".join(str(d) for d in t.shape)


def slot_metrics(
    sd: Dict[str, torch.Tensor], layer_idx: int, slot_name: str, slot_path: str
) -> Dict[str, Any]:
    """Compute the audit metrics for a single SVD slot."""
    base = base_key(layer_idx, slot_path)
    u_key = f"{base}.U_residual"
    v_key = f"{base}.V_residual"
    s_key = f"{base}.S_residual"

    if u_key not in sd or v_key not in sd or s_key not in sd:
        return {
            "layer_idx": layer_idx,
            "proj_kind": slot_name,
            "U_norm": None,
            "V_norm": None,
            "S_norm": None,
            "S_min": None,
            "S_max": None,
            "S_mean": None,
            "S_abs_mean": None,
            "S_sign_flip_frac": None,
            "contribution_l2": None,
            "shape_str": "MISSING",
            "is_zero": True,
            "exists": False,
        }

    U = sd[u_key].float()
    V = sd[v_key].float()
    S = sd[s_key].float()

    # Contribution to weight = U @ diag(S) @ V (k-rank-32 in P8A).
    contribution = U @ torch.diag(S) @ V
    contribution_l2 = float(contribution.norm())

    # Discriminator: a properly-trained SVD residual has all-non-negative
    # S_residual (since these initialise from the SVD tail, which is by
    # definition non-negative). If S has substantial sign flips, that means
    # S got pushed through zero by stochastic noise — i.e., it received no
    # useful gradient signal but did get optimiser noise / weight-decay /
    # a tiny amount of orthogonal-loss leakage.
    sign_flip_frac = float((S < 0).sum().item()) / max(1, S.numel())

    # Shape string of the contribution (= shape of weight delta).
    shape_str = f"U:{fmt_shape(U)}, V:{fmt_shape(V)}, S:{fmt_shape(S)}"

    return {
        "layer_idx": layer_idx,
        "proj_kind": slot_name,
        "U_norm": float(U.norm()),
        "V_norm": float(V.norm()),
        "S_norm": float(S.norm()),
        "S_min": float(S.min()),
        "S_max": float(S.max()),
        "S_mean": float(S.mean()),
        "S_abs_mean": float(S.abs().mean()),
        "S_sign_flip_frac": sign_flip_frac,
        "contribution_l2": contribution_l2,
        "shape_str": shape_str,
        "is_zero": contribution_l2 < ZERO_THRESHOLD,
        "exists": True,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logger.info("=" * 78)
    logger.info("P8A SVD-Residual Parameter Audit  (2026-05-03)")
    logger.info("=" * 78)
    logger.info("checkpoint: %s", P8A_CKPT)
    if not P8A_CKPT.exists():
        logger.error("checkpoint not found at %s", P8A_CKPT)
        return 2

    ckpt_size_mb = P8A_CKPT.stat().st_size / 1024 / 1024
    logger.info("checkpoint size: %.1f MB", ckpt_size_mb)

    raw = torch.load(P8A_CKPT, map_location="cpu", weights_only=False)
    sd = unwrap_state_dict(raw)
    logger.info("state_dict has %d keys", len(sd))

    if isinstance(raw, dict):
        for sidecar in ("epoch", "auc", "eer", "training_step", "model_checksum"):
            if sidecar in raw:
                logger.info("checkpoint sidecar | %s = %r", sidecar, raw[sidecar])

    # ---- per-slot metrics ----
    rows: List[Dict[str, Any]] = []
    for layer_idx in range(12):
        for slot_name, slot_path in SLOTS:
            row = slot_metrics(sd, layer_idx, slot_name, slot_path)
            rows.append(row)

    # ---- emit CSV ----
    fieldnames = [
        "layer_idx",
        "proj_kind",
        "exists",
        "U_norm",
        "V_norm",
        "S_norm",
        "S_min",
        "S_max",
        "S_mean",
        "S_abs_mean",
        "S_sign_flip_frac",
        "contribution_l2",
        "shape_str",
        "is_zero",
    ]
    with open(CSV_PATH, "w", newline="") as fh_csv:
        writer = csv.DictWriter(fh_csv, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
    logger.info("wrote per-layer audit CSV -> %s", CSV_PATH)

    # ---- per-layer summary tables (logged) ----
    logger.info("")
    logger.info("Per-layer contribution L2 (= ||U @ diag(S) @ V||_F):")
    logger.info(
        "  %-3s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s",
        "L",
        "out_proj",
        "svd_q",
        "svd_k",
        "svd_v",
        "mlp.c_fc",
        "mlp.c_proj",
    )
    for layer_idx in range(12):
        cells = []
        for slot_name, _ in SLOTS:
            r = next(
                row
                for row in rows
                if row["layer_idx"] == layer_idx and row["proj_kind"] == slot_name
            )
            v = r["contribution_l2"]
            cells.append("MISSING" if v is None else f"{v:.3e}")
        logger.info(
            "  %-3d | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s",
            layer_idx,
            *cells,
        )

    logger.info("")
    logger.info("Per-layer |S_residual| mean (lower = trained less / not at all):")
    logger.info(
        "  %-3s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s",
        "L",
        "out_proj",
        "svd_q",
        "svd_k",
        "svd_v",
        "mlp.c_fc",
        "mlp.c_proj",
    )
    for layer_idx in range(12):
        cells = []
        for slot_name, _ in SLOTS:
            r = next(
                row
                for row in rows
                if row["layer_idx"] == layer_idx and row["proj_kind"] == slot_name
            )
            v = r["S_abs_mean"]
            cells.append("MISSING" if v is None else f"{v:.3e}")
        logger.info(
            "  %-3d | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s",
            layer_idx,
            *cells,
        )

    logger.info("")
    logger.info("Per-layer S_residual sign-flip fraction (high = optimiser noise only):")
    logger.info(
        "  %-3s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s",
        "L",
        "out_proj",
        "svd_q",
        "svd_k",
        "svd_v",
        "mlp.c_fc",
        "mlp.c_proj",
    )
    for layer_idx in range(12):
        cells = []
        for slot_name, _ in SLOTS:
            r = next(
                row
                for row in rows
                if row["layer_idx"] == layer_idx and row["proj_kind"] == slot_name
            )
            v = r["S_sign_flip_frac"]
            cells.append("MISSING" if v is None else f"{v:.3f}")
        logger.info(
            "  %-3d | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s",
            layer_idx,
            *cells,
        )

    # ---- aggregate stats ----
    existing_rows = [r for r in rows if r["exists"]]
    in_proj_rows = [r for r in existing_rows if r["proj_kind"] in IN_PROJ_SLOTS]
    other_rows = [r for r in existing_rows if r["proj_kind"] not in IN_PROJ_SLOTS]

    n_total = len(existing_rows)
    n_nonzero = sum(1 for r in existing_rows if not r["is_zero"])
    n_in_proj_total = len(in_proj_rows)
    n_in_proj_nonzero = sum(1 for r in in_proj_rows if not r["is_zero"])

    # Trainable backbone params per non-zero residual layer
    def _params(r: Dict[str, Any]) -> int:
        # state_dict shapes: U_residual (out, k), V_residual (k, in), S_residual (k,)
        # We don't have explicit numel here, but the shape_str encodes it.
        # Recompute via the actual tensors.
        layer_idx = r["layer_idx"]
        slot_name = r["proj_kind"]
        slot_path = next(p for s, p in SLOTS if s == slot_name)
        base = base_key(layer_idx, slot_path)
        u = sd[f"{base}.U_residual"]
        v = sd[f"{base}.V_residual"]
        s = sd[f"{base}.S_residual"]
        return int(u.numel() + v.numel() + s.numel())

    trainable_params_nonzero = sum(_params(r) for r in existing_rows if not r["is_zero"])
    trainable_params_in_proj_nonzero = sum(
        _params(r) for r in in_proj_rows if not r["is_zero"]
    )
    trainable_params_other_nonzero = sum(
        _params(r) for r in other_rows if not r["is_zero"]
    )

    # ---- discriminator: bug-affected vs control comparison ----
    # Compute median contribution L2 for in_proj group vs out_proj group
    def median(vals: List[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    out_proj_contribs = [
        r["contribution_l2"] for r in existing_rows if r["proj_kind"] == "out_proj"
    ]
    in_proj_contribs = [r["contribution_l2"] for r in in_proj_rows]
    in_proj_s_abs_means = [r["S_abs_mean"] for r in in_proj_rows]
    out_proj_s_abs_means = [
        r["S_abs_mean"] for r in existing_rows if r["proj_kind"] == "out_proj"
    ]

    median_in_proj_contrib = median(in_proj_contribs)
    median_out_proj_contrib = median(out_proj_contribs)
    contribution_ratio = (
        median_in_proj_contrib / median_out_proj_contrib
        if median_out_proj_contrib > 0
        else float("inf")
    )

    median_in_proj_s_abs = median(in_proj_s_abs_means)
    median_out_proj_s_abs = median(out_proj_s_abs_means)

    # Sign-flip discriminator: a learned SVD residual should retain SVD-tail
    # ordering (all-positive S). A residual that received only noise will
    # have ~50% sign-flips. We flag layers with >25% sign flips as "noise-
    # dominated" (i.e., effectively unlearned).
    in_proj_noise_dominated = [
        r for r in in_proj_rows if r["S_sign_flip_frac"] > 0.25
    ]
    other_noise_dominated = [
        r for r in other_rows if r["S_sign_flip_frac"] > 0.25
    ]

    # Headline verdict logic uses BOTH:
    #  (a) contribution magnitude (zero-threshold), and
    #  (b) the much-more-sensitive sign-flip diagnostic
    n_in_proj_effectively_unlearned = sum(
        1
        for r in in_proj_rows
        if r["is_zero"] or r["S_sign_flip_frac"] > 0.25
    )
    frac_in_proj_unlearned = n_in_proj_effectively_unlearned / max(1, n_in_proj_total)

    if frac_in_proj_unlearned >= 0.80:
        verdict = "FAILED"
        verdict_explanation = (
            f"in_proj-SVD residuals are effectively unlearned in "
            f"{n_in_proj_effectively_unlearned}/{n_in_proj_total} "
            f"({frac_in_proj_unlearned:.0%}) of slots. "
            "The bug zeroed the gradient as predicted. "
            "P8A trained as if SVD coverage was 1× (out_proj only)."
        )
    elif frac_in_proj_unlearned <= 0.20:
        verdict = "PASSED"
        verdict_explanation = (
            f"Only {n_in_proj_effectively_unlearned}/{n_in_proj_total} "
            f"({frac_in_proj_unlearned:.0%}) of in_proj-SVD slots are "
            "effectively unlearned. P8A used the full 4× SVD capacity."
        )
    else:
        verdict = "PARTIAL"
        verdict_explanation = (
            f"{n_in_proj_effectively_unlearned}/{n_in_proj_total} "
            f"({frac_in_proj_unlearned:.0%}) of in_proj-SVD slots are "
            "effectively unlearned. Mixed signal."
        )

    summary: Dict[str, Any] = {
        "checkpoint": str(P8A_CKPT),
        "checkpoint_size_mb": round(ckpt_size_mb, 1),
        "training_step": raw.get("training_step") if isinstance(raw, dict) else None,
        "training_epoch": raw.get("epoch") if isinstance(raw, dict) else None,
        "training_auc": raw.get("auc") if isinstance(raw, dict) else None,
        "training_eer": raw.get("eer") if isinstance(raw, dict) else None,
        "n_resblocks_inspected": 12,
        "n_svd_slots_per_block": len(SLOTS),
        "total_svd_residual_layers_found": n_total,
        "total_nonzero_residual_layers": n_nonzero,
        "in_proj_svd_layers_total": n_in_proj_total,
        "in_proj_svd_layers_nonzero": n_in_proj_nonzero,
        "in_proj_svd_layers_effectively_unlearned": n_in_proj_effectively_unlearned,
        "frac_in_proj_unlearned": round(frac_in_proj_unlearned, 4),
        "trainable_params_nonzero_total": trainable_params_nonzero,
        "trainable_params_nonzero_in_proj": trainable_params_in_proj_nonzero,
        "trainable_params_nonzero_other": trainable_params_other_nonzero,
        "median_contribution_l2_in_proj": median_in_proj_contrib,
        "median_contribution_l2_out_proj": median_out_proj_contrib,
        "in_proj_to_out_proj_contribution_ratio": contribution_ratio,
        "median_S_abs_mean_in_proj": median_in_proj_s_abs,
        "median_S_abs_mean_out_proj": median_out_proj_s_abs,
        "in_proj_layers_with_sign_flip_frac_gt_25pct": [
            (r["layer_idx"], r["proj_kind"], round(r["S_sign_flip_frac"], 3))
            for r in in_proj_noise_dominated
        ],
        "other_layers_with_sign_flip_frac_gt_25pct": [
            (r["layer_idx"], r["proj_kind"], round(r["S_sign_flip_frac"], 3))
            for r in other_noise_dominated
        ],
        "verdict": verdict,
        "verdict_explanation": verdict_explanation,
        "discriminator_method": (
            "A slot is 'effectively unlearned' if its weight-contribution L2 "
            "(||U @ diag(S) @ V||_F) is < 1e-6 OR its S_residual sign-flip "
            "fraction exceeds 0.25 (a properly-trained SVD residual retains "
            "SVD-tail ordering: all-positive S). Verdict aggregates over the "
            "36 in_proj-SVD slots (12 layers x {svd_q, svd_k, svd_v})."
        ),
    }

    with open(JSON_PATH, "w") as fh_json:
        json.dump(summary, fh_json, indent=2, default=str)
    logger.info("wrote summary JSON -> %s", JSON_PATH)

    # ---- final stdout banner ----
    logger.info("")
    logger.info("=" * 78)
    logger.info("VERDICT: %s", verdict)
    logger.info("=" * 78)
    logger.info(verdict_explanation)
    logger.info(
        "median in_proj contribution L2: %.3e   (vs out_proj %.3e ; ratio %.3f)",
        median_in_proj_contrib,
        median_out_proj_contrib,
        contribution_ratio,
    )
    logger.info(
        "median in_proj |S| mean       : %.3e   (vs out_proj %.3e)",
        median_in_proj_s_abs,
        median_out_proj_s_abs,
    )
    logger.info(
        "in_proj slots effectively unlearned: %d/%d (%.0f%%)",
        n_in_proj_effectively_unlearned,
        n_in_proj_total,
        100.0 * frac_in_proj_unlearned,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
