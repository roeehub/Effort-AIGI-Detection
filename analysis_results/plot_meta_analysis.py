#!/usr/bin/env python3
"""
Meta-Analysis Visualization: Why Enhanced Face-Swaps Fool All Models
====================================================================
Creates comprehensive plots comparing WMA enhanced images vs training data sources.

Outputs saved to analysis_results/plots/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Color scheme: distinguish fake sources, real sources, and WMA
SOURCE_COLORS = {
    'wma_enhanced':                      '#e74c3c',  # Red - the problem source
    'deeplive_quality_enhancement_fake': '#c0392b',  # Dark red - DL QE fakes
    'deeplive_minimal_processing_fake':  '#d35400',  # Burnt orange - DL MP fakes
    'deeplive_edge_cases_fake':          '#f39c12',  # Orange - DL edge cases fakes
    'visomaster_fake':                   '#e67e22',  # Dark orange - VisoMaster fakes
    'df40_fake':                         '#9b59b6',  # Purple - DF40 fakes
    'deeplive_quality_enhancement_real': '#27ae60',  # Dark green - DL QE reals
    'deeplive_minimal_processing_real':  '#16a085',  # Teal - DL MP reals
    'df40_real':                         '#2ecc71',  # Green - DF40 reals
    'external_youtube_real':             '#3498db',  # Blue - YouTube reals
}

SOURCE_LABELS = {
    'wma_enhanced':                     'WMA Enhanced\n(test, ALL fake)',
    'deeplive_quality_enhancement_fake':'DL QualEnhance\n(train, fake)',
    'deeplive_minimal_processing_fake': 'DL MinProc\n(train, fake)',
    'deeplive_edge_cases_fake':         'DL EdgeCases\n(train, fake)',
    'visomaster_fake':                  'VisoMaster\n(train, fake)',
    'df40_fake':                        'DF40\n(train, fake)',
    'deeplive_quality_enhancement_real':'DL QualEnhance\n(train, real)',
    'deeplive_minimal_processing_real': 'DL MinProc\n(train, real)',
    'df40_real':                        'DF40\n(train, real)',
    'external_youtube_real':            'YouTube Ext.\n(train, real)',
}

SOURCE_ORDER = [
    'wma_enhanced',
    'deeplive_quality_enhancement_fake',
    'deeplive_minimal_processing_fake',
    'deeplive_edge_cases_fake',
    'visomaster_fake',
    'df40_fake',
    'deeplive_quality_enhancement_real',
    'deeplive_minimal_processing_real',
    'df40_real',
    'external_youtube_real',
]


def load_data():
    """Load CSV properties and NPZ features."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'meta_analysis_properties.csv'))
    npz = np.load(os.path.join(RESULTS_DIR, 'meta_analysis_features.npz'), allow_pickle=True)
    return df, npz


# ═══════════════════════════════════════════════════════════════════════
# PLOT 1: Model Probability Distributions
# ═══════════════════════════════════════════════════════════════════════
def plot_probability_distributions(df):
    """Histogram of model fake probabilities per source — THE key failure plot."""
    n_sources = len([s for s in SOURCE_ORDER if s in df['source'].unique()])
    ncols = min(5, n_sources)
    nrows = (n_sources + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Model Fake Probability Distributions by Source\n'
                 '(threshold = 0.5: right = "detected as fake")',
                 fontsize=16, fontweight='bold')

    # Filter to sources that exist in data
    active_sources = [s for s in SOURCE_ORDER if s in df['source'].unique()]
    for idx, source in enumerate(active_sources):
        ax = axes[idx // ncols, idx % ncols]
        subset = df[df['source'] == source]['model_fake_prob'].dropna()

        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(SOURCE_LABELS.get(source, source))
            continue

        ax.hist(subset, bins=50, color=SOURCE_COLORS[source], alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='threshold')

        # Stats annotation
        mean_p = subset.mean()
        median_p = subset.median()
        is_fake = not source.endswith('_real')
        if is_fake:
            acc = (subset > 0.5).mean() * 100
        else:
            acc = (subset <= 0.5).mean() * 100

        stats_text = f'mean={mean_p:.3f}\nmedian={median_p:.3f}\nacc={acc:.1f}%'
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_title(SOURCE_LABELS.get(source, source), fontsize=10, fontweight='bold')
        ax.set_xlabel('Fake Probability')
        ax.set_ylabel('Count')
        ax.set_xlim(0, 1)

    # Hide unused axes
    for idx in range(len(active_sources), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '01_probability_distributions.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 2: Image Sharpness Comparison (THE smoking gun)
# ═══════════════════════════════════════════════════════════════════════
def plot_sharpness_comparison(df):
    """Box + violin plots of sharpness metrics across sources."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Sharpness / Texture Metrics: WMA vs Training Data',
                 fontsize=16, fontweight='bold')

    metrics = [
        ('sharpness_laplacian_var', 'Laplacian Variance\n(higher = sharper)'),
        ('sharpness_tenengrad', 'Tenengrad\n(gradient magnitude)'),
        ('edge_density', 'Edge Density\n(fraction of edge pixels)'),
    ]

    for i, (col, title) in enumerate(metrics):
        ax = axes[i]
        data_by_source = []
        labels = []
        colors = []
        for source in SOURCE_ORDER:
            subset = df[df['source'] == source][col].dropna()
            if len(subset) > 0:
                data_by_source.append(subset.values)
                labels.append(SOURCE_LABELS.get(source, source))
                colors.append(SOURCE_COLORS[source])

        bp = ax.boxplot(data_by_source, labels=labels, patch_artist=True,
                       showfliers=False, widths=0.6)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Overlay medians as text
        for j, d in enumerate(data_by_source):
            med = np.median(d)
            ax.text(j + 1, med, f'{med:.1f}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold', color='black')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '02_sharpness_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 3: Frequency Domain Analysis
# ═══════════════════════════════════════════════════════════════════════
def plot_frequency_analysis(df):
    """Frequency band energy distribution — reveals GAN/enhancer smoothing."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Frequency Domain Analysis: Energy Distribution',
                 fontsize=16, fontweight='bold')

    freq_metrics = [
        ('freq_low_energy_ratio', 'Low-Freq Energy Ratio\n(smooth content)'),
        ('freq_high_energy_ratio', 'High-Freq Energy Ratio\n(fine detail / noise)'),
        ('freq_high_to_low_ratio', 'High/Low Frequency Ratio\n(detail sharpness)'),
    ]

    for i, (col, title) in enumerate(freq_metrics):
        ax = axes[i]
        data_by_source = []
        labels = []
        colors = []
        for source in SOURCE_ORDER:
            subset = df[df['source'] == source][col].dropna()
            if len(subset) > 0:
                data_by_source.append(subset.values)
                labels.append(SOURCE_LABELS.get(source, source))
                colors.append(SOURCE_COLORS[source])

        bp = ax.boxplot(data_by_source, labels=labels, patch_artist=True,
                       showfliers=False, widths=0.6)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '03_frequency_analysis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 4: Noise & Texture Characteristics
# ═══════════════════════════════════════════════════════════════════════
def plot_noise_texture(df):
    """Noise estimate and texture variance — fingerprints of generation method."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Noise & Texture Properties',
                 fontsize=16, fontweight='bold')

    metrics = [
        ('noise_estimate', 'Noise Estimate\n(estimated noise σ)'),
        ('texture_local_var_mean', 'Texture Local Variance\n(mean local variation)'),
        ('jpeg_compressibility', 'JPEG Compressibility\n(ratio: smaller = more compressible)'),
    ]

    for i, (col, title) in enumerate(metrics):
        ax = axes[i]
        data_by_source = []
        labels = []
        colors = []
        for source in SOURCE_ORDER:
            subset = df[df['source'] == source][col].dropna()
            if len(subset) > 0:
                data_by_source.append(subset.values)
                labels.append(SOURCE_LABELS.get(source, source))
                colors.append(SOURCE_COLORS[source])

        bp = ax.boxplot(data_by_source, labels=labels, patch_artist=True,
                       showfliers=False, widths=0.6)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '04_noise_texture.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 5: Resolution & Aspect Ratio
# ═══════════════════════════════════════════════════════════════════════
def plot_resolution(df):
    """Resolution scatter — reveals the 224×224 vs arbitrary resolution gap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Image Resolution: Training Data (224×224) vs WMA (variable)',
                 fontsize=16, fontweight='bold')

    # Scatter: width vs height
    ax = axes[0]
    for source in SOURCE_ORDER:
        subset = df[df['source'] == source]
        ax.scatter(subset['width'], subset['height'],
                  c=SOURCE_COLORS[source], alpha=0.5, s=20,
                  label=SOURCE_LABELS.get(source, source).replace('\n', ' '))
    ax.set_xlabel('Width (px)', fontsize=12)
    ax.set_ylabel('Height (px)', fontsize=12)
    ax.set_title('Width vs Height per Source', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    # Box: total pixel count
    ax = axes[1]
    data_by_source = []
    labels = []
    colors = []
    for source in SOURCE_ORDER:
        subset = df[df['source'] == source]['num_pixels'].dropna()
        if len(subset) > 0:
            data_by_source.append(subset.values)
            labels.append(SOURCE_LABELS.get(source, source))
            colors.append(SOURCE_COLORS[source])

    bp = ax.boxplot(data_by_source, labels=labels, patch_artist=True,
                   showfliers=False, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(224*224, color='gray', linestyle='--', alpha=0.5, label='224×224 (model input)')
    ax.set_title('Total Pixels per Source', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pixels')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '05_resolution_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 6: Color Statistics
# ═══════════════════════════════════════════════════════════════════════
def plot_color_stats(df):
    """Color channel means and saturation — does the enhancer shift color?"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Color & Luminance Properties',
                 fontsize=16, fontweight='bold')

    metrics = [
        ('luminance_mean', 'Luminance (mean)', axes[0, 0]),
        ('contrast_rms', 'RMS Contrast', axes[0, 1]),
        ('saturation_mean', 'Saturation (mean)', axes[1, 0]),
        ('color_r_mean', 'Red Channel Mean', axes[1, 1]),
    ]

    for col, title, ax in metrics:
        data_by_source = []
        labels = []
        colors = []
        for source in SOURCE_ORDER:
            subset = df[df['source'] == source][col].dropna()
            if len(subset) > 0:
                data_by_source.append(subset.values)
                labels.append(SOURCE_LABELS.get(source, source))
                colors.append(SOURCE_COLORS[source])

        bp = ax.boxplot(data_by_source, labels=labels, patch_artist=True,
                       showfliers=False, widths=0.6)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '06_color_statistics.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 7: t-SNE of CLIP Features (the most important plot)
# ═══════════════════════════════════════════════════════════════════════
def plot_tsne_features(npz):
    """t-SNE of 512-dim CLIP features colored by source — where does WMA cluster?"""
    features = npz['features']
    source_labels = npz['source_labels']
    probs = npz['probs']

    print(f"  Running t-SNE on {features.shape[0]} × {features.shape[1]} features (this takes ~30s)...")

    # PCA first for speed
    pca = PCA(n_components=50, random_state=42)
    features_pca = pca.fit_transform(features)
    pca_var = pca.explained_variance_ratio_.sum()
    print(f"  PCA: 50 components explain {pca_var:.1%} variance")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, learning_rate='auto')
    coords = tsne.fit_transform(features_pca)

    # --- Plot 7a: colored by source ---
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle('CLIP Feature Space (t-SNE): Where Do WMA Enhanced Images Land?',
                 fontsize=16, fontweight='bold')

    ax = axes[0]
    # Plot training data first, WMA on top
    for source in reversed(SOURCE_ORDER):
        mask = source_labels == source
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=SOURCE_COLORS[source], alpha=0.5, s=15 if source != 'wma_enhanced' else 8,
                  label=SOURCE_LABELS.get(source, source).replace('\n', ' '),
                  edgecolors='none')
    ax.set_title('Colored by Data Source', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best', markerscale=2)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(alpha=0.2)

    # --- Plot 7b: colored by model probability ---
    ax = axes[1]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=probs, cmap='RdYlGn_r',
                   alpha=0.5, s=8, edgecolors='none', vmin=0, vmax=1)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Model Fake Probability', fontsize=11)
    ax.set_title('Colored by Model Prediction', fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '07_tsne_clip_features.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return coords  # Return for reuse


# ═══════════════════════════════════════════════════════════════════════
# PLOT 8: Correlation Scatter — Sharpness vs Model Probability
# ═══════════════════════════════════════════════════════════════════════
def plot_sharpness_vs_prob(df):
    """Scatter of sharpness vs model probability — test if model uses sharpness as proxy."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Does the Model Use Sharpness as a Shortcut for "Fake"?',
                 fontsize=16, fontweight='bold')

    metrics = [
        ('sharpness_laplacian_var', 'Laplacian Variance'),
        ('edge_density', 'Edge Density'),
        ('freq_high_energy_ratio', 'High-Freq Energy Ratio'),
    ]

    for i, (col, xlabel) in enumerate(metrics):
        ax = axes[i]
        for source in SOURCE_ORDER:
            subset = df[df['source'] == source]
            ax.scatter(subset[col], subset['model_fake_prob'],
                      c=SOURCE_COLORS[source], alpha=0.3, s=10,
                      label=SOURCE_LABELS.get(source, source).replace('\n', ' '))

        ax.axhline(0.5, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Model Fake Probability', fontsize=12)
        ax.set_title(f'Fake Prob vs {xlabel}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc='best', markerscale=2)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '08_sharpness_vs_probability.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 9: Summary Dashboard
# ═══════════════════════════════════════════════════════════════════════
def plot_summary_dashboard(df):
    """Single-page summary of the most important findings."""
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

    fig.suptitle('META-ANALYSIS: Why Enhanced Face-Swaps Fool the Model',
                 fontsize=18, fontweight='bold', y=0.98)

    # 1. Accuracy bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    accs = []
    src_labels = []
    src_colors = []
    for source in SOURCE_ORDER:
        subset = df[df['source'] == source]['model_fake_prob'].dropna()
        is_fake = not source.endswith('_real')
        if is_fake:
            acc = (subset > 0.5).mean() * 100
        else:
            acc = (subset <= 0.5).mean() * 100
        accs.append(acc)
        src_labels.append(SOURCE_LABELS.get(source, source).replace('\n', ' '))
        src_colors.append(SOURCE_COLORS.get(source, '#95a5a6'))

    bars = ax1.barh(range(len(accs)), accs, color=src_colors, alpha=0.8)
    ax1.set_yticks(range(len(accs)))
    ax1.set_yticklabels(src_labels, fontsize=8)
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('Model Accuracy by Source', fontweight='bold', fontsize=11)
    ax1.axvline(50, color='gray', linestyle='--', alpha=0.5)
    for i, v in enumerate(accs):
        ax1.text(v + 1, i, f'{v:.0f}%', va='center', fontsize=9, fontweight='bold')
    ax1.set_xlim(0, 110)

    # 2. Sharpness comparison
    ax2 = fig.add_subplot(gs[0, 1])
    medians = []
    for source in SOURCE_ORDER:
        subset = df[df['source'] == source]['sharpness_laplacian_var'].dropna()
        medians.append(subset.median() if len(subset) > 0 else 0)
    bars = ax2.barh(range(len(medians)), medians, color=src_colors, alpha=0.8)
    ax2.set_yticks(range(len(medians)))
    ax2.set_yticklabels(src_labels, fontsize=8)
    ax2.set_xlabel('Laplacian Variance (median)')
    ax2.set_title('Sharpness by Source', fontweight='bold', fontsize=11)
    for i, v in enumerate(medians):
        ax2.text(v + 1, i, f'{v:.1f}', va='center', fontsize=9)

    # 3. Edge density
    ax3 = fig.add_subplot(gs[0, 2])
    medians = []
    for source in SOURCE_ORDER:
        subset = df[df['source'] == source]['edge_density'].dropna()
        medians.append(subset.median() if len(subset) > 0 else 0)
    bars = ax3.barh(range(len(medians)), medians, color=src_colors, alpha=0.8)
    ax3.set_yticks(range(len(medians)))
    ax3.set_yticklabels(src_labels, fontsize=8)
    ax3.set_xlabel('Edge Density (median)')
    ax3.set_title('Edge Density by Source', fontweight='bold', fontsize=11)
    for i, v in enumerate(medians):
        ax3.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

    # 4. Noise estimate distribution
    ax4 = fig.add_subplot(gs[1, 0])
    data_by_source = []
    labels_short = []
    colors_list = []
    for source in SOURCE_ORDER:
        subset = df[df['source'] == source]['noise_estimate'].dropna()
        if len(subset) > 0:
            data_by_source.append(subset.values)
            labels_short.append(source.replace('deeplive_edge_cases_', 'DL_EC_').replace('external_youtube_', 'YT_').replace('visomaster_', 'VM_').replace('wma_enhanced', 'WMA'))
            colors_list.append(SOURCE_COLORS[source])
    bp = ax4.boxplot(data_by_source, labels=labels_short, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.set_title('Noise Estimate σ', fontweight='bold', fontsize=11)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)

    # 5. High-freq energy
    ax5 = fig.add_subplot(gs[1, 1])
    data_by_source = []
    labels_short = []
    colors_list = []
    for source in SOURCE_ORDER:
        subset = df[df['source'] == source]['freq_high_energy_ratio'].dropna()
        if len(subset) > 0:
            data_by_source.append(subset.values)
            labels_short.append(source.replace('deeplive_edge_cases_', 'DL_EC_').replace('external_youtube_', 'YT_').replace('visomaster_', 'VM_').replace('wma_enhanced', 'WMA'))
            colors_list.append(SOURCE_COLORS[source])
    bp = ax5.boxplot(data_by_source, labels=labels_short, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax5.set_title('High-Frequency Energy', fontweight='bold', fontsize=11)
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(axis='y', alpha=0.3)

    # 6. Resolution scatter
    ax6 = fig.add_subplot(gs[1, 2])
    for source in SOURCE_ORDER:
        subset = df[df['source'] == source]
        ax6.scatter(subset['width'], subset['height'],
                   c=SOURCE_COLORS[source], alpha=0.4, s=10,
                   label=source.replace('deeplive_edge_cases_', 'DL_EC_').replace('external_youtube_', 'YT_').replace('visomaster_', 'VM_').replace('wma_enhanced', 'WMA'))
    ax6.set_xlabel('Width')
    ax6.set_ylabel('Height')
    ax6.set_title('Resolution Distribution', fontweight='bold', fontsize=11)
    ax6.legend(fontsize=7, loc='upper left')
    ax6.grid(alpha=0.3)

    # 7. Probability distribution — WMA vs DeepLive QE vs DeepLive EC
    ax7 = fig.add_subplot(gs[2, 0:2])
    overlay_sources = ['wma_enhanced', 'deeplive_quality_enhancement_fake',
                       'deeplive_minimal_processing_fake', 'deeplive_edge_cases_fake',
                       'visomaster_fake']
    for source in overlay_sources:
        subset = df[df['source'] == source]['model_fake_prob'].dropna()
        if len(subset) == 0:
            continue
        ax7.hist(subset, bins=50, alpha=0.4, color=SOURCE_COLORS.get(source, 'gray'),
                label=SOURCE_LABELS.get(source, source).replace('\n', ' '), density=True)
    ax7.axvline(0.5, color='black', linestyle='--', alpha=0.7)
    ax7.set_xlabel('Fake Probability', fontsize=11)
    ax7.set_ylabel('Density')
    ax7.set_title('Probability Distribution: WMA vs Training Fakes', fontweight='bold', fontsize=11)
    ax7.legend(fontsize=9)

    # 8. Key findings text
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    findings = [
        "KEY FINDINGS",
        "─" * 35,
        "",
        "1. WMA images are 3-8× LESS sharp",
        "   than training fakes",
        "",
        "2. WMA has ~3× less edge density",
        "   (GFPGAN smooths edges)",
        "",
        "3. WMA noise σ spread is tight",
        "   (GFPGAN uniform output)",
        "",
        "4. Training data is ALL 224×224;",
        "   WMA is ~333×430 (variable)",
        "",
        "5. Compare WMA vs DL QualEnhance:",
        "   same pipeline, different crops",
        "",
        "6. DF40 reals: high false positive",
        "   (model thinks reals are fake!)",
    ]
    ax8.text(0.05, 0.95, '\n'.join(findings), transform=ax8.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    path = os.path.join(PLOTS_DIR, '09_summary_dashboard.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 10: Radar / Spider Chart — Property Profiles
# ═══════════════════════════════════════════════════════════════════════
def plot_radar_profiles(df):
    """Radar chart showing normalized property profiles per source."""
    properties = [
        'sharpness_laplacian_var', 'edge_density', 'freq_high_energy_ratio',
        'noise_estimate', 'texture_local_var_mean', 'saturation_mean',
        'contrast_rms', 'luminance_mean'
    ]
    prop_labels = [
        'Sharpness', 'Edge Density', 'High-Freq', 'Noise σ',
        'Texture Var', 'Saturation', 'Contrast', 'Luminance'
    ]

    # Compute median per source, then min-max normalize across sources
    medians = {}
    for source in SOURCE_ORDER:
        subset = df[df['source'] == source]
        medians[source] = [subset[p].median() for p in properties]

    medians_arr = np.array(list(medians.values()))
    mins = medians_arr.min(axis=0)
    maxs = medians_arr.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # avoid div by zero

    normalized = {s: (np.array(v) - mins) / ranges for s, v in medians.items()}

    # Plot
    N = len(properties)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.suptitle('Property Profiles (Normalized Medians)',
                 fontsize=14, fontweight='bold', y=1.02)

    for source in SOURCE_ORDER:
        values = normalized[source].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, alpha=0.7,
               color=SOURCE_COLORS[source],
               label=SOURCE_LABELS.get(source, source).replace('\n', ' '))
        ax.fill(angles, values, alpha=0.05, color=SOURCE_COLORS[source])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(prop_labels, fontsize=10)
    ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.1))

    path = os.path.join(PLOTS_DIR, '10_radar_profiles.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("META-ANALYSIS: Enhanced Face-Swap Detection Failure")
    print("=" * 70)

    print("\nLoading data...")
    df, npz = load_data()
    print(f"  CSV: {len(df)} rows, {len(df.columns)} columns")
    print(f"  NPZ: {npz['features'].shape[0]} samples, {npz['features'].shape[1]}-dim features")
    print(f"  Sources: {df['source'].nunique()} ({', '.join(df['source'].unique())})")

    # Quick stats
    print("\n" + "─" * 70)
    print("QUICK STATS PER SOURCE:")
    print("─" * 70)
    for source in SOURCE_ORDER:
        subset = df[df['source'] == source]
        probs = subset['model_fake_prob'].dropna()
        is_fake = not source.endswith('_real')
        if is_fake:
            acc = (probs > 0.5).mean() * 100
        else:
            acc = (probs <= 0.5).mean() * 100
        sharp = subset['sharpness_laplacian_var'].median()
        edge = subset['edge_density'].median()
        noise = subset['noise_estimate'].median()
        w = subset['width'].median()
        h = subset['height'].median()
        print(f"  {source:30s} | n={len(subset):4d} | acc={acc:5.1f}% | "
              f"prob={probs.median():.3f} | sharp={sharp:7.1f} | edge={edge:.4f} | "
              f"noise={noise:.3f} | res={w:.0f}×{h:.0f}")

    print("\n" + "─" * 70)
    print("Generating plots...")
    print("─" * 70)

    plot_probability_distributions(df)
    plot_sharpness_comparison(df)
    plot_frequency_analysis(df)
    plot_noise_texture(df)
    plot_resolution(df)
    plot_color_stats(df)
    plot_tsne_features(npz)
    plot_sharpness_vs_prob(df)
    plot_summary_dashboard(df)
    plot_radar_profiles(df)

    print("\n" + "=" * 70)
    print(f"All plots saved to: {PLOTS_DIR}")
    print("=" * 70)

    # Print conclusions
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
1. SHARPNESS SHORTCUT (ROOT CAUSE): The model uses sharpness/edge density
   as a primary proxy for real/fake classification. Low-sharpness images
   are classified as "real" regardless of truth. WMA (18.7), DF40 fake
   (37.9), and DF40 real (15.4) all share low sharpness and are all
   misclassified. Training fakes (54-129) and YouTube reals (226) have
   high sharpness and are classified correctly.

2. RESOLUTION MISMATCH (AMPLIFIER): WMA images are ~342x435, downscaled
   to 224x224 at inference. This additional downscaling further reduces
   sharpness and destroys high-frequency content. Training data is
   uniformly 224x224 (pre-cropped). The model has never seen
   inference-time downscaling during training.

3. GFPGAN SMOOTHING (AMPLIFIER): GFPGAN removes high-frequency GAN
   artifacts, but the model handles GFPGAN-enhanced images fine when
   pre-cropped (DL QE: 98% accuracy). The issue is GFPGAN smoothing
   COMBINED with resolution mismatch.

4. EDGE DENSITY CORROBORATION: WMA fakes and DF40 reals have IDENTICAL
   edge density (0.0118) and are both badly misclassified. The model
   uses edge density as a co-feature with sharpness.

5. DF40 REAL FAILURE (89% FPR): DF40 reals have sharpness 15.4 (lowest
   in dataset) and are misclassified as fake with 89% rate. This confirms
   the model has NOT learned genuine fake-detection features.

6. QUALITY-ROBUST AUGMENTATION MAKES IT WORSE: R3_FT1/FT2 (quality
   augmented) get 8-9% accuracy on WMA vs R25_F1's 21% and B16_old's
   55.7%. The augmentation destroys detection ability rather than helping.
""")


if __name__ == '__main__':
    main()
