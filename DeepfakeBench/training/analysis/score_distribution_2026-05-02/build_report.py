"""Assemble a single HTML report aggregating every figure produced by the
score-distribution forensics for the 2026-05-02 D contract scorecard.

Output: outputs/REPORT.html (relative to ROOT)
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"

# Asset paths work both:
#  - file:// open of REPORT.html — relative path "figures/foo.png" resolves locally
#  - viewer-served via /api/model-runs/<rid>/static-report — <base href> below makes the same relative path resolve through the API asset endpoint
def img(rel: str, alt: str = "", w: str = "100%") -> str:
    return f'<img src="figures/{rel}" alt="{alt}" style="max-width:{w};height:auto;border:1px solid #ccc;border-radius:4px;margin:8px 0;" />'

def section(title: str, body: str, anchor: str = "") -> str:
    aid = f' id="{anchor}"' if anchor else ""
    return f'<section{aid}><h2>{title}</h2>{body}</section>\n'


def csv_table(path: Path, max_rows: int = 50) -> str:
    if not path.exists():
        return f'<p style="color:red">Missing: {path}</p>'
    df = pd.read_csv(path)
    return df.head(max_rows).to_html(index=False, classes="data-table", float_format="%.4f")


def main():
    fig_files = sorted(FIG.glob("*.png"))

    # --- Headline summary table
    headline_html = csv_table(OUT / "headline_summary.csv", max_rows=200)
    method_breakdown_html = csv_table(OUT / "method_breakdown.csv", max_rows=200)
    quantiles_html = csv_table(OUT / "quantiles_frame.csv", max_rows=200)
    feasibility_html = csv_table(OUT / "operating_point_feasibility.csv", max_rows=200)
    crop_sig_html = csv_table(OUT / "crop_attribute_significance.csv", max_rows=200)
    threshold_grid_html = csv_table(OUT / "threshold_grid.csv", max_rows=200)
    crossings_html = csv_table(OUT / "threshold_crossings.csv", max_rows=200)
    # Batch 2 tables
    sharpness_val_html = csv_table(OUT / "sharpness_validation.csv", max_rows=200)
    identity_fpr_html = csv_table(OUT / "identity_fpr_breakdown.csv", max_rows=200)
    lockbox_method_html = csv_table(OUT / "lockbox_fake_method_summary.csv", max_rows=200)
    pose_sig_html = csv_table(OUT / "pose_significance.csv", max_rows=200)
    train_attr_summary_html = csv_table(OUT / "train_vs_eval_attribute_summary.csv", max_rows=200)
    # Batch 3+4 tables
    lockbox_attr_html = csv_table(OUT / "lockbox_attribute_significance.csv", max_rows=200)
    cross_suite_summary_html = csv_table(OUT / "cross_suite_attribute_summary.csv", max_rows=200)
    score_corr_html = csv_table(OUT / "score_attribute_correlations.csv", max_rows=200)
    identity_per_suite_html = csv_table(OUT / "identity_per_suite_fpr.csv", max_rows=200)

    # --- Per-suite hist sections
    fake_suite_grid = ""
    for suite in ["visomaster_enhanced_macro_dev", "deeplive_enhanced_dev",
                  "teams_fake_all_dev", "teams_fake_all_lockbox"]:
        row = ""
        for model in ["P8A", "P18T", "P18C"]:
            f = f"hist_{suite}_{model}.png"
            if (FIG / f).exists():
                row += f'<div style="flex:1;min-width:300px">{img(f, f"{suite} {model}", w="100%")}</div>'
        fake_suite_grid += f'<h3>{suite}</h3><div style="display:flex;gap:8px;flex-wrap:wrap">{row}</div>'

    real_suite_grid = ""
    for suite in ["teams_real_all_dev", "teams_real_all_lockbox", "teams_real_dor_dev",
                  "teams_real_lighting_extreme_dev", "teams_real_poor_quality_dev"]:
        row = ""
        for model in ["P8A", "P18T", "P18C"]:
            f = f"hist_{suite}_{model}.png"
            if (FIG / f).exists():
                row += f'<div style="flex:1;min-width:300px">{img(f, f"{suite} {model}", w="100%")}</div>'
        real_suite_grid += f'<h3>{suite}</h3><div style="display:flex;gap:8px;flex-wrap:wrap">{row}</div>'

    # --- Compose
    html = textwrap.dedent(f"""
    <!doctype html>
    <html><head>
      <meta charset="utf-8">
      <title>Score Distribution Forensics — 2026-05-02 D contract</title>
      <script>
        // When served via the viewer endpoint (path starts with /api/), point relative URLs
        // at the static-report-asset endpoint so figures/ resolves correctly.
        (function() {{
          if (location.pathname.indexOf('/api/') === 0) {{
            var b = document.createElement('base');
            b.href = '/api/static-report-asset/score_distribution_2026-05-02/outputs/';
            document.head.appendChild(b);
          }}
        }})();
      </script>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
                max-width: 1400px; margin: 24px auto; padding: 0 24px; color: #222; line-height: 1.45; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
        h2 {{ margin-top: 32px; border-bottom: 1px solid #ddd; padding-bottom: 6px; color: #224; }}
        h3 {{ color: #444; margin-top: 24px; }}
        nav {{ background: #f5f5f5; padding: 12px; border-radius: 6px; margin: 16px 0; }}
        nav a {{ display: inline-block; margin-right: 12px; color: #036; }}
        section {{ margin-bottom: 28px; }}
        .data-table {{ border-collapse: collapse; margin: 10px 0; font-size: 11px; }}
        .data-table th, .data-table td {{ padding: 4px 8px; border: 1px solid #ccc; text-align: right; font-family: "SF Mono", Menlo, monospace; }}
        .data-table th {{ background: #e8e8e8; }}
        .callout {{ background: #fffae6; border-left: 4px solid #f0c000; padding: 10px 14px; margin: 12px 0; border-radius: 4px; }}
        .callout.fact {{ background: #e6f7ff; border-left-color: #1890ff; }}
        .callout.warn {{ background: #fff1f0; border-left-color: #ff4d4f; }}
        code {{ background: #eee; padding: 1px 5px; border-radius: 3px; font-size: 90%; }}
        details {{ margin: 8px 0; }}
        summary {{ cursor: pointer; color: #036; font-weight: 500; }}
      </style>
    </head><body>
      <h1>Score Distribution Forensics — 2026-05-02 D contract</h1>
      <p style="color: #666;">
        Built from the 27 per-frame reports produced by the D Vertex contract-scorecard run
        (<code>p18-corrective-contract-20260502-103127</code>): P8A, P18T_GRL_TREATMENT, P18C_NO_GRL_CONTROL,
        each scored on 9 suites covering 40,908 frames total.
        All numbers are pure data; this document deliberately omits framings.
      </p>

      <nav>
        <strong>Jump to:</strong>
        <a href="#headline">Headline</a>
        <a href="#identity-comparison">Identity comparison ●</a>
        <a href="#identity-overlap-analysis">Identity overlap ●</a>
        <a href="#lockbox-attr">Lockbox attr (cam vs pc) ●</a>
        <a href="#cross-suite-landscape">Cross-suite landscape ●</a>
        <a href="#score-attr-corr">Score-attr correlations ●</a>
        <a href="#sharpness-val">Sharpness per model ◆</a>
        <a href="#identity-fpr">Identity FPR ◆</a>
        <a href="#lockbox-walk">Lockbox walkthrough ◆</a>
        <a href="#train-attrs">Train attr distribution ◆</a>
        <a href="#pose-audit">Pose audit ◆</a>
        <a href="#embedding">Feature embedding ◆</a>
        <a href="#operating-point">Operating-point feasibility ★</a>
        <a href="#crop-attr">Crop attribute audit ★</a>
        <a href="#train-vs-eval">Train vs eval visomaster ★</a>
        <a href="#viso-paired">Viso paired (raw vs teams)</a>
        <a href="#viso-pixel">Viso pixel diff</a>
        <a href="#fake-hist">Fake hist grid</a>
        <a href="#real-hist">Real hist grid</a>
        <a href="#kde">KDE overlays</a>
        <a href="#recall-curves">Recall vs τ</a>
        <a href="#methods">Per-method recall</a>
        <a href="#tables">Tables</a>
      </nav>
      <p style="color:#999;font-size:11px">★ = PM follow-up batch 1 &nbsp;&nbsp; ◆ = PM follow-up batch 2 &nbsp;&nbsp; ● = PM follow-up batch 3 (cross-suite, identity overlap, identity comparison)</p>

      <div class="callout fact">
        <strong>Headline (frame-level, at deployed τ):</strong>
        <ul>
          <li><strong>visomaster_enhanced_macro_dev (n=550):</strong> P8A 1.1% / P18T 1.8% / P18C 1.5%. p50 score for P8A = <strong>0.17</strong> (half the slice scored as confidently real). 144/275 paired sequences (52%) are missed by ALL THREE models in BOTH substrates.</li>
          <li><strong>deeplive_enhanced_dev (n=545, single dor identity):</strong> P8A 2.4% / P18T 7.5% / P18C 19.4%. Different shape: P18C piles distribution at p50=0.984 but can't push past τ.</li>
          <li><strong>Viso slice is 50/50 raw/teams substrate</strong>, paired sequences. Visually identical pairs differ by L1 ~33/255 per pixel, +34 luminance shift, 2.5× HF energy reduction.</li>
        </ul>
        <strong>Headline (added in PM follow-up batch 1):</strong>
        <ul>
          <li>★ <strong>Operating-point unlock:</strong> if real-FPR floor relaxes from 2% → 10%, P18C deeplive recall jumps 19% → 77%, teams_fake_all 39% → 74%. The contract τ is what's making the model look bad.</li>
          <li>★ <strong>The 108 cross-model "never-caught" pairs are blurrier:</strong> Laplacian variance 43.6 vs 71.4 (p &#60; 0.0001). Sharpness IS the structural difference; brightness is not.</li>
          <li>★ <strong>Train vs eval mismatch is real and large:</strong> training viso is 224×224, eval viso is 369×369 (2.7× more pixels), different GCS buckets entirely.</li>
        </ul>
        <strong>Headline (added in PM follow-up batch 2):</strong>
        <ul>
          <li>◆ <strong>Sharpness predicts P8A catches reliably; P18T teams_only catches are NOT sharper</strong> than missed (delta +12, p=0.547) — GRL training catches a different mode than blur-conditioned.</li>
          <li>◆ <strong>13 unique identities in teams_real_all_dev; 96% of P8A's FPs are in 3 identities.</strong> Roy_D and Q (n=130 + n=54) drive everything. dor in dev = 0% FPR; the dor regression is purely lockbox.</li>
          <li>◆ <strong>Lockbox fake failure is concentrated in cam_test_s33:</strong> 315 of 334 frames (94%) missed by ALL three models at deployment τ. pc_generator_s15 (n=91) is largely solved.</li>
          <li>◆ <strong>Training Laplacian variance is 132-407; eval Laplacian variance is 36-60.</strong> Eval is 4-11× less sharp than training on the very metric that predicts catchability.</li>
          <li>◆ <strong>Pose features RULE OUT pose as the structural difference:</strong> yaw, pitch, roll, eye distance, mouth aspect — none significant. Image quality (sharpness/edges) is the only discriminating axis.</li>
          <li>◆ <strong>ResNet-50 ImageNet features show pair-distance pattern:</strong> raw_only (catch) pairs are FURTHER APART in feature space than both_missed pairs.</li>
        </ul>
        <strong>Headline (added in PM follow-up batch 3):</strong>
        <ul>
          <li>● <strong>dor identity is THE SAME PERSON across dev real, lockbox real, and dev fake suites</strong> (named dor / dor_shkedi / real_dor / deeplive_dor in different splits). Identity-leakage across the held-out boundary.</li>
          <li>● <strong>3 identities span dev↔lockbox</strong> (PC_Generator, bla_bla_chow, dor_shkedi); ZERO of the dev FPR-driver identities exist in the training bucket.</li>
          <li>● <strong>teams_fake_all_lockbox is the LEAST sharp suite</strong> in the entire scorecard (Laplacian median 10), 13-40× lower than training data.</li>
          <li>● <strong>Within cam_test_s33, sharpness does NOT distinguish caught vs missed</strong> — once in the low-sharpness regime, sharpness is no longer predictive.</li>
          <li>● <strong>Score-attribute correlations:</strong> brighter / sharper / more-skin frames → LOWER fake score. The model has learned that darker, less-sharp, less-skin-dominated images are more fake-like.</li>
        </ul>
      </div>

      {section("Headline summary (per suite × model)", '<details open><summary>Show table</summary>' + headline_html + '</details>', anchor="headline")}

      {section("★ Operating-point feasibility — what recall do we unlock at looser FPR?",
        '<div class="callout fact"><strong>Key finding:</strong> the 2% real-FPR floor is what makes the model look bad. At 10% FPR (still strict), P8A gets 70% teams_fake + 54% lockbox + 42% deeplive. P18C gets 74% teams + 77% deeplive. The model is much closer to usable than the contract scorecard implies.</div>'
        + img("operating_point_relaxation.png", "Operating-point relaxation curves")
        + '<details open><summary>Show feasibility table</summary>' + feasibility_html + '</details>'
        + '<details><summary>Show full threshold grid</summary>' + threshold_grid_html + '</details>'
        + '<details><summary>Show threshold crossings (at what τ does each suite cross floor X?)</summary>' + crossings_html + '</details>',
        anchor="operating-point")}

      {section("★ Crop attribute audit — what makes 144 viso pairs uncatchable?",
        '<div class="callout fact"><strong>Key finding:</strong> 108 of 275 viso pairs are NEVER caught by any model in either substrate at τ=0.5. They are systematically less sharp (Laplacian variance 43.6 vs 71.4, p &#60; 0.0001) and have less edge content (Sobel 27.6 vs 29.8, p &#60; 0.0001). Brightness, skin coverage, and saturation are essentially the same as caught pairs.</div>'
        + img("crop_attr_never_vs_ever.png", "Distribution: never_caught vs ever_caught per attribute")
        + img("crop_attr_vs_score_p8a.png", "P8A score vs each crop attribute")
        + '<details open><summary>Show significance table</summary>' + crop_sig_html + '</details>',
        anchor="crop-attr")}

      {section("★ Train vs eval visomaster comparison",
        '<div class="callout warn"><strong>Mechanical fact:</strong> Training viso lives at 224×224 (50k px²). Eval viso lives at 369×369 (136k px², 2.7× more pixels). Eval files are ~2× the bytes. The eval substrate is also literally a different bucket (gs://teams-faces-data-test-... vs gs://live-deepfake-methods-real-and-fake-frames-cropped/ vs gs://visomaster-enhanced-face-cropped/). Eval brightness (raw=148, teams=183) sits within the training-data range (CSCS=73, codeformer=183, etc).</div>'
        + img("train_vs_eval_grid.png", "Visual grid — training vs eval visomaster samples")
        + img("train_vs_eval_luma.png", "Mean luminance per group"),
        anchor="train-vs-eval")}

      {section("◆ Sharpness validation per model",
        '<div class="callout fact"><strong>Key finding:</strong> raw_only catches are reliably sharper than both_missed across all three models (P8A delta +43, P18T +42, P18C +43; all p &#60; 0.0001). However, P18T teams_only catches (n=54, the new wins from method-conditional GRL) show NO significant sharpness elevation vs both_missed (delta +12, p=0.547). Sharpness predicts P8A catches strongly; weakens for P18T/C, especially on the teams substrate.</div>'
        + img("sharpness_per_model.png", "Per-model laplacian variance per pair category")
        + '<details open><summary>Show validation table</summary>' + sharpness_val_html + '</details>',
        anchor="sharpness-val")}

      {section("◆ Identity-level FPR breakdown for teams_real_all_dev",
        '<div class="callout fact"><strong>Key finding:</strong> teams_real_all_dev (n=4564) decomposes into only 13 unique identities. P8A 122 false positives concentrate 96% in the top 3 identities. The dor identity in dev (n=269) has 0% FPR for all three models — the documented "dor problem" lives entirely in the LOCKBOX dor_shkedi identity, not the dev dor identity. Roy_D and Q are the actual FPR drivers and get progressively worse from P8A → P18T → P18C.</div>'
        + img("identity_fpr_concentration.png", "Cumulative FP concentration")
        + img("identity_fpr_top20.png", "Top 20 identities by FPR per model")
        + '<details open><summary>Show identity FPR table</summary>' + identity_fpr_html + '</details>',
        anchor="identity-fpr")}

      {section("◆ Lockbox fake walkthrough — n=425 frames across 2 methods",
        '<div class="callout fact"><strong>Key finding:</strong> Lockbox fake-recall headline is dominated by teams_capture_cam_test_s33 failure: 315 of 334 frames (94.3%) are missed by ALL THREE models at deployed τ. teams_capture_pc_generator_s15 (n=91) is largely solved (P8A catches 89% at deployed τ). Visual inspection shows cam_test_s33 is one specific identity in various poses, all reasonable-quality face captures.</div>'
        + img("lockbox_fake_gallery_teams_capture_cam_test_s33.png", "Lockbox fake gallery — cam_test_s33 (the missed subset)")
        + img("lockbox_fake_gallery_teams_capture_pc_generator_s15.png", "Lockbox fake gallery — pc_generator_s15 (the caught subset)")
        + '<details open><summary>Show per-method recall summary</summary>' + lockbox_method_html + '</details>',
        anchor="lockbox-walk")}

      {section("◆ Train data attribute distribution vs eval",
        '<div class="callout warn"><strong>Mechanical fact:</strong> Training viso has Laplacian variance 132-407 (mean 165 base, 407 enhanced). Eval viso has Laplacian variance 36-60. The eval distribution is roughly 4-11× LESS SHARP than training on the very metric that the crop-attribute audit identified as predicting catchability. Image height: training 224, eval 379 (~1.7× larger). Density distributions for laplacian_var are essentially non-overlapping between train and eval populations.</div>'
        + img("train_vs_eval_attr_overlay.png", "Train vs eval — per-feature density distributions")
        + img("train_vs_eval_attr_box.png", "Train vs eval — per-feature box plots")
        + '<details><summary>Show train+eval attribute summary</summary>' + train_attr_summary_html + '</details>',
        anchor="train-attrs")}

      {section("◆ Pose / face-detection audit on viso pairs",
        '<div class="callout fact"><strong>Key finding:</strong> MediaPipe face detection succeeds on 98% (539/550) of frames. NO pose feature (yaw, pitch, roll, eye distance, mouth aspect ratio, face bbox area) significantly distinguishes never_caught from ever_caught pairs (lowest p=0.061, all non-significant). This rules out pose/orientation as the structural difference between catchable and uncatchable pairs. Combined with the sharpness audit (p &#60; 0.0001), image-quality features are the only discriminating axis.</div>'
        + img("pose_attr_dist.png", "Pose attribute distribution — never_caught vs ever_caught")
        + '<details open><summary>Show pose significance table</summary>' + pose_sig_html + '</details>',
        anchor="pose-audit")}

      {section("◆ ResNet-50 ImageNet feature embedding",
        '<div class="callout fact"><strong>Key finding:</strong> 2048-dim ResNet-50 ImageNet features for the 550 viso frames. Per-pair raw↔teams feature distance is largest for raw_only category (median 4.00) and smallest for both_missed (median 3.68). Pairs (same seq_id) are NOT consistently close in t-SNE space — the substrate transformation moves them apart even in pretrained-feature space. PCA explains 25% + 13% of variance on first two components.</div>'
        + img("embedding_pca_2d.png", "PCA 2D projection of ResNet50 features")
        + img("embedding_tsne_2d.png", "t-SNE 2D projection")
        + img("embedding_pair_distance.png", "Per-pair raw↔teams feature distance by P8A category"),
        anchor="embedding")}

      {section("● Identity comparison gallery (visual identity check)",
        '<div class="callout warn"><strong>Visual finding:</strong> dor / dor_shkedi / real_dor / deeplive_dor are visually the SAME PERSON across dev real, lockbox real, and dev fake suites. The dev↔lockbox identity split is NOT held out by person — only by per-frame label string. Roy_D, Q, Cam_Test, bla_bla_chow are 4 distinct individuals, none visually similar to each other or to dor. Per-identity score histograms show P18C right-shifts dor lockbox real significantly vs P8A.</div>'
        + img("identity_comparison_gallery.png", "Identity comparison gallery")
        + img("identity_score_histograms.png", "Per-identity prob_fake distributions"),
        anchor="identity-comparison")}

      {section("● Identity overlap & training-bucket cross-reference",
        '<div class="callout warn"><strong>Mechanical fact:</strong> 3 identity strings appear in BOTH dev and lockbox real suites: PC_Generator (835/29), bla_bla_chow (491/68), dor_shkedi (31/1170). ZERO of the dev FPR-driver identities (PC_Generator, dor_shkedi, bla_bla_chow, Roy_D, Q, Test_Cam, Md_noyn_Sharker, Cam_Test) appear in the training bucket gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/. The training bucket is structured by ALGORITHM, not by identity.</div>'
        + img("identity_overlap_heatmap.png", "Per-identity FPR per real-suite heatmap")
        + '<details><summary>Show per-identity FPR per suite table</summary>' + identity_per_suite_html + '</details>',
        anchor="identity-overlap-analysis")}

      {section("● Lockbox cam_test_s33 vs pc_generator_s15 attribute comparison",
        '<div class="callout fact"><strong>Mechanical fact:</strong> All 425 lockbox fake frames analysed. cam_test_s33 (n=334, mostly missed) is 5× LESS sharp than pc_generator_s15 (n=91, mostly caught) — Laplacian variance 9.5 vs 46.1 (p &#60; 0.0001). Also: 22 luminance units darker, 7.8 luma_std lower, 0.30 more skin_frac, 172 pixels taller. Within cam_test_s33 (caught vs missed), sharpness does NOT distinguish (p=0.39) — once in the low-sharpness regime, sharpness is no longer predictive.</div>'
        + img("lockbox_cam_vs_pc_attrs.png", "cam_test_s33 vs pc_generator_s15 attribute distributions")
        + '<details open><summary>Show lockbox attribute significance table</summary>' + lockbox_attr_html + '</details>',
        anchor="lockbox-attr")}

      {section("● Cross-suite attribute landscape",
        '<div class="callout warn"><strong>Mechanical fact:</strong> teams_fake_all_lockbox is the LEAST sharp suite in the entire scorecard (Laplacian median 10). deeplive_enhanced_dev is one of the SHARPEST fake suites (median 245). Real suites cluster at Laplacian 200-300. Training viso has Laplacian 132-407. The eval lockbox fake distribution sits 13-40× below the training data on the very metric that predicts catchability.</div>'
        + img("cross_suite_attr_box.png", "Cross-suite attribute box distributions")
        + '<details open><summary>Show cross-suite attribute summary</summary>' + cross_suite_summary_html + '</details>',
        anchor="cross-suite-landscape")}

      {section("● Score-attribute correlations across suites",
        '<div class="callout fact"><strong>Mechanical fact:</strong> Pearson r between frame_prob and crop attributes per (suite, model). Negative correlations dominate: brighter / sharper / more-skin → LOWER fake score. For real suites this means correctly-classified-as-real (good). For fake suites this means missed (bad). Strongest |r| values: teams_real_dor_dev / P18T / laplacian_var = -0.57; teams_fake_all_dev / P8A / luma_std = -0.55; teams_fake_all_lockbox / P8A / h = -0.50.</div>'
        + img("score_attribute_corr_heatmap_P8A.png", "Score-attribute correlation heatmap (P8A)")
        + '<details open><summary>Show full correlation table</summary>' + score_corr_html + '</details>',
        anchor="score-attr-corr")}

      {section("Viso paired raw vs teams", img("viso_paired_scatter.png", "Paired scatter") + img("viso_paired_delta.png", "Δ score histogram") + img("viso_video_score_strip.png", "Per-video sorted strip") + '<h3>Visual pair grids — same source, two substrates</h3>' + img("viso_paired_grid_P8A.png", "P8A pair grid") + img("viso_paired_grid_P18T.png", "P18T pair grid") + img("viso_paired_grid_P18C.png", "P18C pair grid"), anchor="viso-paired")}

      {section("Viso pixel-level diff (the mechanical signature of the Teams transport)",
        '<div class="callout warn"><strong>Mechanical fact:</strong> Across 275 paired sequences: mean RGB L1 = 33.0 (per pixel), mean luma shift = +33.85, frac pixels changed = 1.000, mean HF-band-ratio delta = -0.0083 (raw 0.013 → teams 0.005, ~2.5× reduction), mean filesize ratio = 0.88. The teams substrate is uniformly brighter, smoother, and smaller-on-disk. Per-channel: R=+34.6 G=+41.8 B=+38.1 (G shifted most). Visual face-shape pattern in |diff|.</div>'
        + img("pixel_diff_hist.png", "Pixel-diff histograms")
        + img("single_pair_seq5402_diff.png", "seq5402 raw vs teams diff (raw-only winner pair)")
        + img("pixel_diff_vs_score_delta.png", "Score delta vs pixel-diff metrics"), anchor="viso-pixel")}

      {section("Fake suites — per (suite × model) histograms", fake_suite_grid, anchor="fake-hist")}

      {section("Real suites — per (suite × model) histograms", real_suite_grid, anchor="real-hist")}

      {section("KDE overlays — all suites in one panel per model",
        '<h3>Fake suites</h3>' + img("kde_overlay_fake_P8A.png", "Fake KDE P8A") + img("kde_overlay_fake_P18T.png", "Fake KDE P18T") + img("kde_overlay_fake_P18C.png", "Fake KDE P18C")
        + '<h3>Real suites</h3>' + img("kde_overlay_real_P8A.png", "Real KDE P8A") + img("kde_overlay_real_P18T.png", "Real KDE P18T") + img("kde_overlay_real_P18C.png", "Real KDE P18C"), anchor="kde")}

      {section("Suite recall vs τ — fake suites", img("suite_recall_curves.png", "Fake suite recall curves") + img("real_fpr_curves.png", "Real suite FPR curves"), anchor="recall-curves")}

      {section("Per-method recall on teams_fake_all_dev (which contains viso + deeplive + teams_capture)",
        img("method_recall_curves_P8A.png", "Per-method recall P8A") + img("method_recall_curves_P18T.png", "Per-method recall P18T") + img("method_recall_curves_P18C.png", "Per-method recall P18C"), anchor="methods")}

      {section("Per-method breakdown table (teams_fake_all_dev composition + per-method recall)",
        '<details open><summary>Show table</summary>' + method_breakdown_html + '</details>', anchor="methods-table")}

      {section("Quantile table per (suite × model) — frame-level prob_fake",
        '<details><summary>Show table</summary>' + quantiles_html + '</details>', anchor="tables")}

      <hr>
      <p style="color:#999;font-size:11px">
        Generated by analysis/score_distribution_2026-05-02/build_report.py.
        Source: gs://training-job-outputs/test_results/teams_promotion_contract/p18-corrective-contract-20260502-103127.
        Per-frame data: combined_frames.parquet (40,908 rows). Paired viso analysis: viso_pairs.csv (825 rows).
        Pixel diff: pixel_diff.csv (275 unique sequences).
      </p>
    </body></html>
    """)

    out_path = OUT / "REPORT.html"
    out_path.write_text(html)
    print(f"[done] wrote {out_path}")
    print(f"[open] file://{out_path}")


if __name__ == "__main__":
    main()
