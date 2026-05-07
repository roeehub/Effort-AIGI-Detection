"""
Phase E weight-delta diagnostic — P1 (PE_PAIR_RANK_DRO) ckpts vs P8A
2026-05-07.

Question: did `apply_svd_to_in_proj` fire under PE_PAIR_RANK_DRO loss?

Design (from session decision 2026-05-07): direct weight-delta beats
CE-grad-audit. CE-grad-audit replicates packets/P8A.md:108 C-ablation
exactly — confirms the bug fix is preserved (true and trivial). Direct
training-time payoff under the actual loss class needs weight-delta.

Method:
  For each layer with SVD residuals (U_residual, S_residual, V_residual),
  reconstruct Δ = U @ diag(S) @ V at both checkpoints; Frobenius norm of
  the difference (W_p1 − W_p8a) is the "training-time movement" of that
  layer. Group by layer category:

    in_proj_qkv  — the lever under audit (was zero-gradient pre-`2feea58`)
    out_proj     — adjacent SVD'd layer (control: also unfrozen, classifier
                   gradient was flowing here pre-fix)
    mlp          — SVD'd MLP layers if `apply_svd_to_mlp=true` (P1 yamls have it)

Read:
  Ratio in_proj_qkv / out_proj movement.
    ≈ 1 → lever fires similarly to other unfrozen layers (active under PE_PAIR_RANK_DRO).
    ≈ 0 → lever inert under PE_PAIR_RANK_DRO loss class (consistent with the
            P10_SYM-on-P8A C-ablation null in packets/P8A.md:108).
    >> 1 → lever moved much more than control (active and dominant).

Inputs are public GCS URIs from arena/checkpoint_maps/teams_target_domain.p1_pe_pair_rank_2026-05-07.yaml
plus the P8A reference. Trajectory: step 500/1000/4000 (BUNDLE) and
step 500/1000/6750 (PAIRRANK).

Usage:
  cd analysis/p1_pe_eval_2026-05-07/weight_delta
  python compute_weight_delta.py
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

# ---------------------------------------------------------------------- paths

THIS_DIR = Path(__file__).resolve().parent
CACHE_DIR = THIS_DIR / "ckpt_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = THIS_DIR.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
log = logging.getLogger("weight_delta")

# ------------------------------------------------------------------- ckpts

BASELINE_URI = "gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"

CKPTS: dict[str, str] = {
    "p1_bundle_step500":     "gs://training-job-outputs/best_checkpoints/tznuar61/periodic_effort_20260506_step500_auc0.9784_eer0.0735.pth",
    "p1_bundle_step1000":    "gs://training-job-outputs/best_checkpoints/tznuar61/periodic_effort_20260506_step1000_auc0.9869_eer0.0189.pth",
    "p1_bundle_step4000":    "gs://training-job-outputs/best_checkpoints/tznuar61/top_n_effort_20260507_step4000_auc0.9951_eer0.0210.pth",
    "p1_pairrank_step500":   "gs://training-job-outputs/best_checkpoints/s2mp5fxm/periodic_effort_20260506_step500_auc0.9848_eer0.0500.pth",
    "p1_pairrank_step1000":  "gs://training-job-outputs/best_checkpoints/s2mp5fxm/periodic_effort_20260506_step1000_auc0.9853_eer0.0476.pth",
    "p1_pairrank_step6750":  "gs://training-job-outputs/best_checkpoints/s2mp5fxm/top_n_effort_20260507_step6750_auc0.9949_eer0.0119.pth",
}

# ----------------------------------------------------------------- helpers

def cached_path(uri: str) -> Path:
    local = CACHE_DIR / Path(uri).name
    if local.exists():
        return local
    log.info("downloading %s -> %s", uri, local)
    subprocess.run(["gcloud", "storage", "cp", uri, str(local)], check=True)
    return local


def load_state_dict(uri: str) -> dict:
    local = cached_path(uri)
    # weights_only=False because R13 ckpts pickle numpy scalars in metadata.
    # Source is our own GCS bucket (train-cvit2 project) — trusted.
    raw = torch.load(local, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
            raw = raw[key]
            break
    if not isinstance(raw, dict):
        raise RuntimeError(f"could not extract state_dict from {uri}")
    return raw


def find_residual_bases(sd: dict) -> list[str]:
    """Return module-base keys that have all 3 of {U,S,V}_residual."""
    bases: dict[str, set[str]] = {}
    for k in sd:
        for suffix in ("U_residual", "S_residual", "V_residual"):
            tail = "." + suffix
            if k.endswith(tail):
                base = k[: -len(tail)]
                bases.setdefault(base, set()).add(suffix)
    complete = sorted(b for b, parts in bases.items()
                      if {"U_residual", "S_residual", "V_residual"} <= parts)
    return complete


def reconstruct_residual(sd: dict, base: str) -> torch.Tensor:
    U = sd[f"{base}.U_residual"]
    S = sd[f"{base}.S_residual"]
    V = sd[f"{base}.V_residual"]
    if S is None or U is None or V is None:
        return torch.zeros(1)
    return U @ torch.diag(S) @ V


def classify_layer(base: str) -> str:
    b = base.lower()
    if "in_proj" in b:
        return "in_proj_qkv"
    if "out_proj" in b:
        return "out_proj"
    if "mlp" in b or "c_fc" in b or "c_proj" in b or "fc1" in b or "fc2" in b:
        return "mlp"
    return "other"


# ----------------------------------------------------------------- main

def main() -> None:
    log.info("loading P8A baseline state_dict")
    p8a = load_state_dict(BASELINE_URI)
    bases = find_residual_bases(p8a)
    log.info("found %d SVD'd layers in P8A", len(bases))

    if not bases:
        log.error("P8A has no SVD residuals — wrong ckpt format?")
        sys.exit(2)

    cat_counts = {}
    for b in bases:
        cat_counts[classify_layer(b)] = cat_counts.get(classify_layer(b), 0) + 1
    log.info("P8A SVD layer category breakdown: %s", cat_counts)

    rows: list[dict] = []
    for ckpt_name, uri in CKPTS.items():
        log.info("=== %s ===", ckpt_name)
        ck = load_state_dict(uri)
        ck_bases = set(find_residual_bases(ck))
        common = [b for b in bases if b in ck_bases]
        log.info("%d / %d layers in common with P8A", len(common), len(bases))

        for base in common:
            try:
                W_p8a = reconstruct_residual(p8a, base)
                W_ck = reconstruct_residual(ck, base)
                delta = (W_ck - W_p8a).norm().item()
                p8a_norm = W_p8a.norm().item()
                ck_norm = W_ck.norm().item()
                rel = delta / max(p8a_norm, 1e-9)
                rows.append({
                    "ckpt": ckpt_name,
                    "layer": base,
                    "category": classify_layer(base),
                    "delta_fnorm": delta,
                    "p8a_residual_fnorm": p8a_norm,
                    "ckpt_residual_fnorm": ck_norm,
                    "relative_delta": rel,
                })
            except Exception as exc:  # noqa: BLE001
                log.warning("failed on %s: %s", base, exc)

    df = pd.DataFrame(rows)
    per_layer = OUT_DIR / "weight_delta_per_layer.csv"
    df.to_csv(per_layer, index=False)
    log.info("wrote %s (%d rows)", per_layer, len(df))

    # Aggregate by ckpt × category
    agg = (df.groupby(["ckpt", "category"])["delta_fnorm"]
             .agg(["count", "mean", "median", "max", "sum"])
             .reset_index())
    by_cat = OUT_DIR / "weight_delta_by_category.csv"
    agg.to_csv(by_cat, index=False)
    log.info("wrote %s", by_cat)

    # Verdict ratio: in_proj_qkv vs out_proj movement, per ckpt
    verdict_rows: list[dict] = []
    for ckpt_name in CKPTS:
        sub = agg[agg["ckpt"] == ckpt_name].set_index("category")
        if "in_proj_qkv" not in sub.index or "out_proj" not in sub.index:
            continue
        qkv = sub.loc["in_proj_qkv", "mean"]
        op = sub.loc["out_proj", "mean"]
        mlp = sub.loc["mlp", "mean"] if "mlp" in sub.index else None
        verdict_rows.append({
            "ckpt": ckpt_name,
            "qkv_mean_fnorm": qkv,
            "out_proj_mean_fnorm": op,
            "mlp_mean_fnorm": mlp,
            "qkv_over_out_proj": qkv / max(op, 1e-9),
            "qkv_over_mlp": qkv / max(mlp, 1e-9) if mlp else None,
        })

    verdict_df = pd.DataFrame(verdict_rows)
    verdict = OUT_DIR / "weight_delta_verdict.csv"
    verdict_df.to_csv(verdict, index=False)
    log.info("wrote %s", verdict)
    print()
    print("=" * 70)
    print("VERDICT (ratio qkv / out_proj):")
    print("=" * 70)
    print(verdict_df.to_string(index=False))
    print()
    print("Read:  ratio ≈ 1 → in_proj_svd lever fires under PE_PAIR_RANK_DRO loss")
    print("       ratio ≈ 0 → lever inert (consistent with P10_SYM C-ablation null)")
    print("       ratio >> 1 → lever active and dominant")


if __name__ == "__main__":
    main()
