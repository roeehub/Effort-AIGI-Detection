"""
F3 untargeted-axis audit extension — color_b_dev (B-channel std).

For each (ckpt, suite), compute Pearson r between per-frame `frame_prob`
(from Phase A reports at raw_reports/phase_a/) and color_b_dev of the source
frame at `frame_path`. Reports Δ |r| pct vs P8A on real-side and fake-side.

Output:
  per_frame_color_b_dev.csv     — frame_path, color_b_dev, suite
  correlations_per_ckpt_suite.csv — ckpt, suite, kind, n, pearson_r
  abs_pearson_summary.csv       — ckpt, suite_kind, mean_abs_r, max_abs_r
  F3_COLOR_B_DEV_RESULTS.md
"""
from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from google.cloud import storage

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
log = logging.getLogger("color_b_dev")

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
PHASE_A = ROOT / "analysis/p1_pe_eval_2026-05-07/raw_reports/phase_a"
CACHE_DIR = THIS_DIR / "_frame_cache"
CACHE_DIR.mkdir(exist_ok=True)
DOR_CACHE_DIR = ROOT / "analysis/p1_pe_eval_2026-05-07/dor_invariance_2026-05-07/_axis_cache"

# 8 ckpts
CKPTS = {
    "p8a_reference_step5000": "p8a",
    "e2b_top_n_step3200": "e2b",
    "p1_bundle_periodic_step500": "p1_bundle_periodic_500",
    "p1_bundle_top_n_step3750": "p1_bundle_top_n_3750",
    "p1_bundle_top_n_step4000": "p1_bundle_top_n_4000",
    "p1_pairrank_periodic_step500": "p1_pairrank_periodic_500",
    "p1_pairrank_top_n_step6000": "p1_pairrank_top_n_6000",
    "p1_pairrank_top_n_step6750": "p1_pairrank_top_n_6750",
}

# Suites and their classification (real vs fake) for aggregation
SUITES = {
    "visomaster_enhanced_macro_dev": "fake",
    "deeplive_enhanced_dev": "fake",
    "teams_fake_all_dev": "fake",
    "teams_real_all_dev": "real",
}


def parse_gcs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(uri)
    bucket, blob = uri[5:].split("/", 1)
    return bucket, blob


def cache_key(uri: str) -> Path:
    """Use bucket__blob path with / -> __ filesystem-safe."""
    bucket, blob = parse_gcs(uri)
    safe = (bucket + "__" + blob).replace("/", "__").replace(" ", "_")
    return CACHE_DIR / safe


def dor_cache_key(uri: str) -> Path:
    """Pre-existing dor cache uses just blob path with / -> __."""
    _, blob = parse_gcs(uri)
    safe = blob.replace("/", "__").replace(" ", "_")
    return DOR_CACHE_DIR / safe


def find_local(uri: str) -> Path | None:
    p = cache_key(uri)
    if p.exists() and p.stat().st_size > 0:
        return p
    p2 = dor_cache_key(uri)
    if p2.exists() and p2.stat().st_size > 0:
        return p2
    return None


def download_one(client: storage.Client, uri: str) -> tuple[str, Path | None, str | None]:
    out = cache_key(uri)
    if out.exists() and out.stat().st_size > 0:
        return uri, out, None
    # Allow re-use from dor cache
    fall = dor_cache_key(uri)
    if fall.exists() and fall.stat().st_size > 0:
        return uri, fall, None
    bucket, blob = parse_gcs(uri)
    try:
        b = client.bucket(bucket)
        bl = b.blob(blob)
        data = bl.download_as_bytes()
        out.write_bytes(data)
        return uri, out, None
    except Exception as e:
        return uri, None, f"{type(e).__name__}: {e}"


def download_all(uris: list[str], max_workers: int = 24) -> tuple[dict[str, Path], dict[str, str]]:
    client = storage.Client()
    # First, check what we already have
    todo = []
    have: dict[str, Path] = {}
    for u in uris:
        lp = find_local(u)
        if lp is not None:
            have[u] = lp
        else:
            todo.append(u)
    log.info("cache hits: %d / %d, downloading: %d", len(have), len(uris), len(todo))
    failed: dict[str, str] = {}
    if not todo:
        return have, failed
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(download_one, client, u): u for u in todo}
        n_done = 0
        for fut in as_completed(futs):
            uri, lp, err = fut.result()
            n_done += 1
            if err is not None:
                failed[uri] = err
            else:
                have[uri] = lp
            if n_done % 500 == 0 or n_done == len(todo):
                log.info("downloaded %d / %d (%.1fs)", n_done, len(todo), time.time() - t0)
    return have, failed


def compute_color_b_dev(local_path: Path) -> float | None:
    """std of B-channel (RGB index 2) on uint8 0-255 image."""
    try:
        data = np.frombuffer(local_path.read_bytes(), dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        # cv2 imdecode is BGR, so RGB[:,:,2] == BGR[:,:,0] == B-channel.
        b = img_bgr[:, :, 0]  # B-channel (BGR layout)
        return float(np.std(b))
    except Exception:
        return None


def pearson_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return float("nan"), n
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), n
    return float(np.corrcoef(x, y)[0, 1]), n


def main() -> None:
    # Step 1: unique URIs across the 4 suites (use P8A reports as canonical;
    # frame sets are identical across ckpts for the same suite — verified).
    suite_to_uris: dict[str, list[str]] = {}
    for suite in SUITES:
        rep = pd.read_csv(PHASE_A / f"{suite}_p8a_reference_step5000_frames_report.csv")
        suite_to_uris[suite] = list(rep["frame_path"].unique())
        log.info("suite=%s n_frames=%d", suite, len(suite_to_uris[suite]))

    all_unique_uris = []
    seen: set[str] = set()
    for s, uris in suite_to_uris.items():
        for u in uris:
            if u not in seen:
                all_unique_uris.append(u)
                seen.add(u)
    log.info("total unique URIs: %d", len(all_unique_uris))

    # Step 2: download (with cache reuse)
    have, failed = download_all(all_unique_uris)
    log.info("downloaded/cached: %d  failed: %d", len(have), len(failed))

    # Step 3: compute color_b_dev for each unique URI
    log.info("computing color_b_dev for %d frames", len(have))
    uri_to_dev: dict[str, float] = {}
    decode_failed: list[str] = []
    t0 = time.time()
    for n, (u, lp) in enumerate(have.items(), 1):
        v = compute_color_b_dev(lp)
        if v is None:
            decode_failed.append(u)
        else:
            uri_to_dev[u] = v
        if n % 1000 == 0 or n == len(have):
            log.info("computed %d / %d (%.1fs)", n, len(have), time.time() - t0)
    log.info("decode failures: %d", len(decode_failed))

    # Step 4: emit per_frame_color_b_dev.csv with rows = (frame_path, color_b_dev, suite)
    pf_rows = []
    for suite, uris in suite_to_uris.items():
        for u in uris:
            v = uri_to_dev.get(u, float("nan"))
            pf_rows.append({"frame_path": u, "color_b_dev": v, "suite": suite})
    pf_df = pd.DataFrame(pf_rows)
    pf_path = THIS_DIR / "per_frame_color_b_dev.csv"
    pf_df.to_csv(pf_path, index=False)
    log.info("wrote %s (n=%d)", pf_path, len(pf_df))

    # Step 5: per (ckpt, suite) Pearson r between frame_prob and color_b_dev
    corr_rows = []
    coverage_rows = []
    for tag, ckpt in CKPTS.items():
        for suite, kind in SUITES.items():
            rp = PHASE_A / f"{suite}_{tag}_frames_report.csv"
            df = pd.read_csv(rp)
            df["color_b_dev"] = df["frame_path"].map(uri_to_dev)
            sub = df[["frame_prob", "color_b_dev"]].dropna()
            r, n_eff = pearson_safe(sub["frame_prob"].values, sub["color_b_dev"].values)
            corr_rows.append({
                "ckpt": ckpt, "suite": suite, "kind": kind, "n": n_eff, "pearson_r": r,
            })
            coverage_rows.append({
                "ckpt": ckpt, "suite": suite, "n_total": len(df),
                "n_with_color_b_dev": int(df["color_b_dev"].notna().sum()),
            })
    corr_df = pd.DataFrame(corr_rows)
    corr_path = THIS_DIR / "correlations_per_ckpt_suite.csv"
    corr_df.to_csv(corr_path, index=False)
    log.info("wrote %s (n=%d)", corr_path, len(corr_df))

    cov_df = pd.DataFrame(coverage_rows)
    cov_path = THIS_DIR / "coverage_per_ckpt_suite.csv"
    cov_df.to_csv(cov_path, index=False)
    log.info("wrote %s", cov_path)

    # Step 6: abs_pearson_summary by ckpt × suite_kind (real / fake)
    corr_df["abs_r"] = corr_df["pearson_r"].abs()
    summary = corr_df.groupby(["ckpt", "kind"]).agg(
        mean_abs_r=("abs_r", "mean"),
        max_abs_r=("abs_r", "max"),
        n_suites=("suite", "count"),
    ).reset_index().rename(columns={"kind": "suite_kind"})
    summary_path = THIS_DIR / "abs_pearson_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("wrote %s (n=%d)", summary_path, len(summary))

    # Step 7: Markdown report
    # Per-ckpt mean |r| on real and fake sides
    pivot = summary.pivot(index="ckpt", columns="suite_kind", values="mean_abs_r")
    # Δ |r| pct vs P8A
    p8a_real = pivot.loc["p8a", "real"]
    p8a_fake = pivot.loc["p8a", "fake"]
    delta = pd.DataFrame({
        "ckpt": pivot.index,
        "mean_abs_r_real": pivot["real"].values,
        "delta_real_pct_vs_p8a": ((pivot["real"].values / p8a_real) - 1.0) * 100.0 if p8a_real > 0 else float("nan"),
        "mean_abs_r_fake": pivot["fake"].values,
        "delta_fake_pct_vs_p8a": ((pivot["fake"].values / p8a_fake) - 1.0) * 100.0 if p8a_fake > 0 else float("nan"),
    })
    delta_path = THIS_DIR / "delta_vs_p8a.csv"
    delta.to_csv(delta_path, index=False)
    log.info("wrote %s", delta_path)

    md_lines: list[str] = []
    md_lines.append("# F3 untargeted-axis audit — color_b_dev (B-channel std)")
    md_lines.append("")
    md_lines.append("## Method")
    md_lines.append("")
    md_lines.append(
        f"Read {sum(len(v) for v in suite_to_uris.values())} (ckpt-shared) per-frame Phase A "
        f"reports across 8 ckpts × 4 suites at "
        f"`analysis/p1_pe_eval_2026-05-07/raw_reports/phase_a/`. "
        f"Downloaded {len(have)} unique source frames into local cache "
        f"(`_frame_cache/` and reused `dor_invariance_2026-05-07/_axis_cache/`); "
        f"decoded with `cv2.imdecode` (BGR uint8 0-255) and computed "
        f"`color_b_dev = std(B_channel)`. "
        f"Per (ckpt, suite) Pearson r between `frame_prob` and `color_b_dev` "
        f"computed via `numpy.corrcoef` after dropping NaNs."
    )
    md_lines.append("")
    md_lines.append("No subsampling was performed (full per-suite frame sets used).")
    md_lines.append("")

    # Per-ckpt mean |r| table
    md_lines.append("## Per-ckpt mean |r| on real-side and fake-side")
    md_lines.append("")
    md_lines.append("| ckpt | mean_abs_r_real | mean_abs_r_fake |")
    md_lines.append("| --- | --- | --- |")
    # Order: p8a first, then e2b, then p1 ckpts in file order
    ck_order = ["p8a", "e2b",
                "p1_bundle_periodic_500", "p1_bundle_top_n_3750", "p1_bundle_top_n_4000",
                "p1_pairrank_periodic_500", "p1_pairrank_top_n_6000", "p1_pairrank_top_n_6750"]
    for ck in ck_order:
        if ck not in pivot.index:
            continue
        rr = pivot.loc[ck, "real"] if "real" in pivot.columns else float("nan")
        fr = pivot.loc[ck, "fake"] if "fake" in pivot.columns else float("nan")
        md_lines.append(f"| {ck} | {rr:.4f} | {fr:.4f} |")
    md_lines.append("")

    # Δ |r| pct vs P8A
    md_lines.append("## Δ |r| pct vs P8A")
    md_lines.append("")
    md_lines.append("| ckpt | delta_real_pct_vs_p8a | delta_fake_pct_vs_p8a |")
    md_lines.append("| --- | --- | --- |")
    for ck in ck_order:
        if ck not in pivot.index:
            continue
        if ck == "p8a":
            md_lines.append(f"| {ck} | 0.00 | 0.00 |")
            continue
        dr = ((pivot.loc[ck, "real"] / p8a_real) - 1.0) * 100.0 if p8a_real > 0 else float("nan")
        df_ = ((pivot.loc[ck, "fake"] / p8a_fake) - 1.0) * 100.0 if p8a_fake > 0 else float("nan")
        md_lines.append(f"| {ck} | {dr:+.2f} | {df_:+.2f} |")
    md_lines.append("")

    # Per-(ckpt × suite) raw Pearson r
    md_lines.append("## Per-(ckpt × suite) Pearson r")
    md_lines.append("")
    md_lines.append("| ckpt | suite | kind | n | pearson_r |")
    md_lines.append("| --- | --- | --- | --- | --- |")
    suite_order = ["visomaster_enhanced_macro_dev", "deeplive_enhanced_dev",
                   "teams_fake_all_dev", "teams_real_all_dev"]
    for ck in ck_order:
        for s in suite_order:
            row = corr_df[(corr_df["ckpt"] == ck) & (corr_df["suite"] == s)]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            md_lines.append(f"| {ck} | {s} | {row['kind']} | {row['n']} | {row['pearson_r']:+.4f} |")
    md_lines.append("")

    # Subsampling note
    md_lines.append("## Subsampling")
    md_lines.append("")
    md_lines.append("None. All Phase A frames per suite were used.")
    md_lines.append("")

    # Caveats
    md_lines.append("## Coverage and caveats")
    md_lines.append("")
    md_lines.append(f"- Total unique URIs across 4 suites: **{len(all_unique_uris)}**")
    md_lines.append(f"- Successfully fetched: **{len(have)}**")
    md_lines.append(f"- Download failures: **{len(failed)}** ({100*len(failed)/max(1,len(all_unique_uris)):.2f}%)")
    md_lines.append(f"- Decode failures (image corrupt / unsupported): **{len(decode_failed)}**")
    if failed:
        # Group by suite
        per_suite_fail = {s: 0 for s in SUITES}
        for u in failed:
            for s, uris in suite_to_uris.items():
                if u in set(uris):
                    per_suite_fail[s] += 1
                    break
        md_lines.append("")
        md_lines.append("Per-suite download failure counts:")
        for s, n in per_suite_fail.items():
            md_lines.append(f"  - {s}: {n}")
    md_lines.append("")

    # F3 close criterion: pass/fail mechanically
    md_lines.append("## F3 close criterion")
    md_lines.append("")
    md_lines.append("F3 close criterion: \"no untargeted axis amplifies +50%\". "
                    "Mechanical pass/fail per ckpt against `color_b_dev`:")
    md_lines.append("")
    md_lines.append("| ckpt | delta_real_pct_vs_p8a | delta_fake_pct_vs_p8a | real_amplifies_>50% | fake_amplifies_>50% |")
    md_lines.append("| --- | --- | --- | --- | --- |")
    for ck in ck_order:
        if ck == "p8a" or ck not in pivot.index:
            continue
        dr = ((pivot.loc[ck, "real"] / p8a_real) - 1.0) * 100.0 if p8a_real > 0 else float("nan")
        df_ = ((pivot.loc[ck, "fake"] / p8a_fake) - 1.0) * 100.0 if p8a_fake > 0 else float("nan")
        md_lines.append(f"| {ck} | {dr:+.2f} | {df_:+.2f} | {'YES' if dr > 50 else 'no'} | {'YES' if df_ > 50 else 'no'} |")
    md_lines.append("")

    md_path = THIS_DIR / "F3_COLOR_B_DEV_RESULTS.md"
    md_path.write_text("\n".join(md_lines))
    log.info("wrote %s", md_path)


if __name__ == "__main__":
    main()
