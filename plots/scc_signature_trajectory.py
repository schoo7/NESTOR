"""
SCC signature trajectory analysis along NESTOR Z progression.

Filtered to 5 signatures with meaningful enrichment patterns:
  PanC_SCC, Glioma_Stem, Stem-like-1, Stem-like-2, Primed hESC

Removed (negative or no center enrichment): Melanoma_SCC, ACS Stem-like,
AML_Stem, Naive hESC.

Outputs:
  publication/02_ne_progression/scc_trajectory/
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

from plotting_utils.save_pdf import save_publication_pdf
from plotting_utils.themes import set_publication_theme

set_publication_theme()

# ── Configuration ──────────────────────────────────────────────────────────

SIGNATURES = [
    'PanC_SCC',
    'Glioma_Stem',
    'Stem-like-1',
    'Stem-like-2',
    'Primed hESC signature',
]

SIG_SHORT = {
    'PanC_SCC': 'PanC SCC',
    'Glioma_Stem': 'Glioma Stem',
    'Stem-like-1': 'Stem-like 1',
    'Stem-like-2': 'Stem-like 2',
    'Primed hESC signature': 'Primed hESC',
}

CANCER_SIGS = ['PanC_SCC', 'Glioma_Stem']
STEM_SIGS = ['Stem-like-1', 'Stem-like-2']
PLURI_SIGS = ['Primed hESC signature']

SIG_COLORS = {
    'PanC_SCC': '#7EB5D6',
    'Glioma_Stem': '#A8C8A0',
    'Stem-like-1': '#E07888',
    'Stem-like-2': '#F4B8C1',
    'Primed hESC signature': '#C8A87A',
}

CATEGORY_COLORS = {
    'cancer': '#7EB5D6',
    'stem': '#E07888',
    'pluripotency': '#C8A87A',
}

INPUT_PATH = 'outputs/scc_unbiased_study/scc_9sig_scores.parquet'
OUTPUT_DIR = 'publication/02_ne_progression/scc_trajectory/'

Z_SMOOTH_WINDOW = 300


# ── Helpers ──────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_parquet(INPUT_PATH)
    df = df.sort_values('z_progression').reset_index(drop=True)
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for col in df.columns:
        cleaned = col.replace('ï', 'i').replace('ì', 'i')
        if cleaned != col:
            rename[col] = cleaned
    if rename:
        df = df.rename(columns=rename)
    return df


def smooth_trajectory(z, values, window=Z_SMOOTH_WINDOW):
    order = np.argsort(z)
    z_sorted = z[order]
    v_sorted = values[order]
    smoothed = uniform_filter1d(v_sorted.astype(float), size=window, mode='nearest')
    return z_sorted, smoothed


def compute_center_metrics(z, values, n_bins=50):
    bins = pd.cut(pd.Series(z), bins=n_bins)
    df_temp = pd.DataFrame({'z': z, 'val': values, 'bin': bins})
    binned = df_temp.groupby('bin', observed=False)['val'].mean()
    z_mids = np.array([(i.left + i.right) / 2 for i in binned.index])

    peak_idx = np.argmax(binned.values)
    peak_z = z_mids[peak_idx]

    center_idx = (z_mids >= 0.4) & (z_mids <= 0.6)
    edge_low_idx = z_mids <= 0.15
    edge_high_idx = z_mids >= 0.85

    center_mean = binned.values[center_idx].mean()
    edge_low = binned.values[edge_low_idx].mean()
    edge_high = binned.values[edge_high_idx].mean()
    edge_mean = (edge_low + edge_high) / 2

    is_dome = center_mean > edge_low and center_mean > edge_high
    center_enrichment = center_mean - edge_mean
    rho, pval = spearmanr(z, values)

    return {
        'peak_z': peak_z,
        'center_mean': center_mean,
        'edge_low': edge_low,
        'edge_high': edge_high,
        'center_enrichment': center_enrichment,
        'is_dome': is_dome,
        'spearman_rho': rho,
        'spearman_p': pval,
        'in_center': 0.4 <= peak_z <= 0.6,
    }


def sig_category(sig):
    if sig in CANCER_SIGS:
        return 'cancer'
    if sig in STEM_SIGS:
        return 'stem'
    return 'pluripotency'


def build_consensus(df):
    """Z-score each signature and average across all 5."""
    zscored = pd.DataFrame()
    for sig in SIGNATURES:
        col = f'ssgsea__{sig}'
        vals = df[col].values
        zscored[sig] = (vals - vals.mean()) / (vals.std() + 1e-10)
    return zscored.mean(axis=1)


def build_consensus_3(df):
    """Consensus of the 3 center-peaking signatures only."""
    best = ['Stem-like-1', 'Stem-like-2', 'Primed hESC signature']
    zscored = pd.DataFrame()
    for sig in best:
        col = f'ssgsea__{sig}'
        vals = df[col].values
        zscored[sig] = (vals - vals.mean()) / (vals.std() + 1e-10)
    return zscored.mean(axis=1)


# ── Figure 1: Individual Signature Trajectories ───────────────────────

def plot_individual_trajectories(df):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))

    for i, sig in enumerate(SIGNATURES):
        ax = axes[i]
        col = f'ssgsea__{sig}'
        z, smooth = smooth_trajectory(df['z_progression'].values, df[col].values)
        metrics = compute_center_metrics(df['z_progression'].values, df[col].values)

        ax.fill_between(z, smooth, alpha=0.15, color=SIG_COLORS[sig])
        ax.plot(z, smooth, color=SIG_COLORS[sig], linewidth=1.8)

        ax.axvspan(0.4, 0.6, alpha=0.08, color='#666666', zorder=0)
        ax.axvline(0.5, color='#999999', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(metrics['peak_z'], color=SIG_COLORS[sig], linewidth=0.8,
                    linestyle=':', alpha=0.7)

        status = ''
        if metrics['is_dome'] and metrics['in_center']:
            status = ' *'
        elif metrics['is_dome']:
            status = ' ^'

        rho_str = f"rho={metrics['spearman_rho']:.3f}"
        ce_str = f"CE={metrics['center_enrichment']:+.3f}"
        ax.set_title(f"{SIG_SHORT[sig]}{status}", fontsize=10, fontweight='bold')
        ax.text(0.02, 0.95, rho_str, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', color='#555555')
        ax.text(0.02, 0.85, ce_str, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', color='#555555')

        ax.set_xlim(0, 1)
        ax.set_xlabel('Z Progression', fontsize=8)
        if i == 0:
            ax.set_ylabel('ssGSEA Score', fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle('SCC Signature Enrichment Along NESTOR Z Progression (ssGSEA)',
                 fontsize=12, fontweight='bold', y=1.03)
    fig.tight_layout()
    save_publication_pdf(fig, OUTPUT_DIR + 'scc_individual_ssgsea_trajectory.pdf')
    plt.close(fig)
    print("Figure 1: Individual trajectories saved")


# ── Figure 2: Single-Panel Category Overlay with Consensus ────────────

def plot_category_overlay_with_consensus(df):
    """Single panel: all 5 signatures + consensus, normalized."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for sig in SIGNATURES:
        col = f'ssgsea__{sig}'
        z, smooth = smooth_trajectory(df['z_progression'].values, df[col].values)
        smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-10)
        cat = sig_category(sig)
        ax.plot(z, smooth_norm, color=SIG_COLORS[sig], linewidth=1.5,
                alpha=0.7, label=SIG_SHORT[sig])

    # Consensus of all 5
    consensus = build_consensus(df)
    z, smooth = smooth_trajectory(df['z_progression'].values, consensus.values)
    smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-10)
    ax.fill_between(z, smooth_norm, alpha=0.15, color='#555555')
    ax.plot(z, smooth_norm, color='#333333', linewidth=2.5,
            label='Consensus (all 5)', linestyle='-')

    ax.text(0.05, 0.02, 'Adenocarcinoma', fontsize=9, color='#5A9BBF',
            transform=ax.transAxes, fontstyle='italic')
    ax.text(0.82, 0.02, 'Neuroendocrine', fontsize=9, color='#8A78B8',
            transform=ax.transAxes, fontstyle='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1)
    ax.set_xlabel('NE Progression (Z)', fontsize=11)
    ax.set_ylabel('Normalized ssGSEA Score', fontsize=11)
    ax.set_title('SCC Signatures Along NESTOR Z Progression',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper center', ncol=3, framealpha=0.9)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    save_publication_pdf(fig, OUTPUT_DIR + 'scc_category_overlay_consensus.pdf')
    plt.close(fig)
    print("Figure 2: Category overlay with consensus saved")


# ── Figure 3: Enrichment Landscape with Z Distribution ─────────────────

def plot_enrichment_landscape(df):
    """
    Instead of peak Z location, show Z distribution of enrichment.
    Each combination plotted as its mean ssGSEA score in 5 Z-bins,
    visualized as a heatmap-style strip.
    """
    ssgsea_cols = {sig: f'ssgsea__{sig}' for sig in SIGNATURES}
    zscored = pd.DataFrame()
    for sig in SIGNATURES:
        col = ssgsea_cols[sig]
        vals = df[col].values
        zscored[sig] = (vals - vals.mean()) / (vals.std() + 1e-10)

    # Build all combinations
    all_combos = []

    # Individual
    for sig in SIGNATURES:
        all_combos.append((sig, [sig]))

    # Pairs
    for pair in combinations(SIGNATURES, 2):
        name = ' + '.join([SIG_SHORT[s] for s in pair])
        all_combos.append((name, list(pair)))

    # Triplets
    for trip in combinations(SIGNATURES, 3):
        name = ' + '.join([SIG_SHORT[s] for s in trip])
        all_combos.append((name, list(trip)))

    # Quadruplets
    for quad in combinations(SIGNATURES, 4):
        name = ' + '.join([SIG_SHORT[s] for s in quad])
        all_combos.append((name, list(quad)))

    # Full consensus
    all_combos.append(('Consensus (all 5)', list(SIGNATURES)))

    # Compute Z-binned enrichment for each combination
    z_bins_edges = np.linspace(0, 1, 11)  # 10 bins
    z_mids = (z_bins_edges[:-1] + z_bins_edges[1:]) / 2

    # For each combo, compute score and bin averages
    combo_scores = []
    combo_names = []
    combo_cats = []
    combo_ce = []

    for name, sigs in all_combos:
        score = zscored[sigs].mean(axis=1)
        binned = []
        for lo, hi in zip(z_bins_edges[:-1], z_bins_edges[1:]):
            mask = (df['z_progression'].values >= lo) & (df['z_progression'].values < hi)
            binned.append(score[mask].mean())
        combo_scores.append(binned)
        combo_names.append(name)

        # Category
        cats = sorted(set(sig_category(s) for s in sigs))
        combo_cats.append('+'.join(cats))

        m = compute_center_metrics(df['z_progression'].values, score.values)
        combo_ce.append(m['center_enrichment'])

    scores_matrix = np.array(combo_scores)

    # Sort by center enrichment descending
    sort_idx = np.argsort(combo_ce)[::-1]
    scores_matrix = scores_matrix[sort_idx]
    combo_names = [combo_names[i] for i in sort_idx]
    combo_cats = [combo_cats[i] for i in sort_idx]
    combo_ce = [combo_ce[i] for i in sort_idx]

    # Color rows by category
    cat_row_colors = {
        'cancer': '#7EB5D6',
        'stem': '#E07888',
        'pluripotency': '#C8A87A',
        'cancer+stem': '#B8A0CC',
        'cancer+pluripotency': '#A8C8A0',
        'stem+pluripotency': '#D4A8D8',
        'cancer+stem+pluripotency': '#888888',
    }

    n_combos = len(combo_names)
    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=(12, max(6, n_combos * 0.25)),
        gridspec_kw={'width_ratios': [1, 4], 'wspace': 0.05}
    )

    # Left panel: center enrichment bars
    bar_colors = [cat_row_colors.get(c, '#888888') for c in combo_cats]
    ax_bar.barh(range(n_combos), combo_ce, color=bar_colors,
                edgecolor='white', linewidth=0.3, height=0.8)
    ax_bar.set_yticks(range(n_combos))
    ax_bar.set_yticklabels(combo_names, fontsize=6)
    ax_bar.axvline(0, color='#CCCCCC', linewidth=0.5)
    ax_bar.set_xlabel('Center\nEnrichment', fontsize=8)
    ax_bar.tick_params(labelsize=6)
    ax_bar.invert_yaxis()

    # Right panel: Z distribution heatmap
    vmax = max(abs(scores_matrix.min()), abs(scores_matrix.max()))
    im = ax_heat.imshow(scores_matrix, cmap='RdBu_r', aspect='auto',
                         vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax_heat.set_xticks(range(len(z_mids)))
    ax_heat.set_xticklabels([f'{z:.1f}' for z in z_mids], fontsize=7)
    ax_heat.set_yticks([])
    ax_heat.set_xlabel('Z Progression', fontsize=9)
    ax_heat.set_title('Score Distribution Across Z Progression', fontsize=10,
                       fontweight='bold')

    # Center zone markers
    for idx in [4, 5]:  # Z bins 0.4-0.5 and 0.5-0.6
        ax_heat.axvline(idx + 0.5, color='#333333', linewidth=0.3, alpha=0.3)

    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.6, label='Mean z-scored ssGSEA')

    fig.suptitle('SCC Signature Enrichment Landscape Along NESTOR Z Progression',
                 fontsize=11, fontweight='bold', y=1.0)
    fig.tight_layout()
    save_publication_pdf(fig, OUTPUT_DIR + 'scc_enrichment_landscape.pdf')
    plt.close(fig)
    print("Figure 3: Enrichment landscape saved")


# ── Figure 4: Consensus Dome (publication-ready) ─────────────────────

def plot_consensus_dome(df):
    """Best combination: Stem-like-1 + Stem-like-2 + Primed hESC consensus."""
    best_sigs = ['Stem-like-1', 'Stem-like-2', 'Primed hESC signature']

    zscored = pd.DataFrame()
    for sig in best_sigs:
        col = f'ssgsea__{sig}'
        vals = df[col].values
        zscored[sig] = (vals - vals.mean()) / (vals.std() + 1e-10)

    consensus = zscored.mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))

    for sig in best_sigs:
        col = f'ssgsea__{sig}'
        z, smooth = smooth_trajectory(df['z_progression'].values, df[col].values)
        smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-10)
        ax.plot(z, smooth_norm, color=SIG_COLORS[sig], linewidth=0.8,
                alpha=0.4, linestyle='--', label=SIG_SHORT[sig])

    z, smooth = smooth_trajectory(df['z_progression'].values, consensus.values)
    smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-10)
    ax.fill_between(z, smooth_norm, alpha=0.2, color='#C8A87A')
    ax.plot(z, smooth_norm, color='#C8A87A', linewidth=2.5,
            label='Consensus')

    ax.axvspan(0.4, 0.6, alpha=0.1, color='#666666', zorder=0)
    ax.axvline(0.5, color='#666666', linewidth=1, linestyle='--', alpha=0.6)

    ax.text(0.05, 0.02, 'Adenocarcinoma', fontsize=9, color='#5A9BBF',
            transform=ax.transAxes, fontstyle='italic')
    ax.text(0.82, 0.02, 'Neuroendocrine', fontsize=9, color='#8A78B8',
            transform=ax.transAxes, fontstyle='italic')
    ax.text(0.46, 0.02, 'Transition', fontsize=9, color='#666666',
            transform=ax.transAxes, fontstyle='italic')

    metrics = compute_center_metrics(df['z_progression'].values, consensus.values)
    ax.text(0.02, 0.95, f"Spearman rho = {metrics['spearman_rho']:.3f}",
            transform=ax.transAxes, fontsize=8, verticalalignment='top', color='#555555')
    ax.text(0.02, 0.88, f"Center enrichment = {metrics['center_enrichment']:+.3f}",
            transform=ax.transAxes, fontsize=8, verticalalignment='top', color='#555555')

    ax.set_xlim(0, 1)
    ax.set_xlabel('NE Progression (Z)', fontsize=11)
    ax.set_ylabel('Normalized Enrichment Score', fontsize=11)
    ax.set_title('Stem-like and Pluripotency Signatures Peak at Transition Zone',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper center')

    fig.tight_layout()
    save_publication_pdf(fig, OUTPUT_DIR + 'scc_consensus_dome_transition.pdf')
    plt.close(fig)
    print("Figure 4: Consensus dome saved")


# ── Figure 5: Center Enrichment Bar Chart ──────────────────────────────

def plot_center_enrichment_barplot(df):
    metrics_list = []
    for sig in SIGNATURES:
        col = f'ssgsea__{sig}'
        m = compute_center_metrics(df['z_progression'].values, df[col].values)
        m['signature'] = sig
        m['short'] = SIG_SHORT[sig]
        m['category'] = sig_category(sig)
        metrics_list.append(m)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.sort_values('center_enrichment', ascending=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))

    colors = [CATEGORY_COLORS[c] for c in metrics_df['category']]
    ax.barh(range(len(metrics_df)), metrics_df['center_enrichment'],
            color=colors, edgecolor='white', linewidth=0.5, height=0.7)

    for i, (idx, row) in enumerate(metrics_df.iterrows()):
        val = row['center_enrichment']
        ax.text(val + 0.003 if val >= 0 else val - 0.003, i,
                f'{val:+.3f}', ha='left' if val >= 0 else 'right',
                va='center', fontsize=8, color='#333333')

    ax.set_yticks(range(len(metrics_df)))
    ax.set_yticklabels(metrics_df['short'], fontsize=9)
    ax.axvline(0, color='#CCCCCC', linewidth=0.5)
    ax.set_xlabel('Center Enrichment (Z 0.4-0.6 vs edges)', fontsize=10)
    ax.set_title('ssGSEA Center Enrichment by Signature', fontsize=12, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS['cancer'], label='Cancer-specific'),
        Patch(facecolor=CATEGORY_COLORS['stem'], label='Stem-like'),
        Patch(facecolor=CATEGORY_COLORS['pluripotency'], label='Pluripotency'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

    fig.tight_layout()
    save_publication_pdf(fig, OUTPUT_DIR + 'scc_center_enrichment_barplot.pdf')
    plt.close(fig)
    print("Figure 5: Center enrichment barplot saved")


# ── Figure 6: Method Comparison ──────────────────────────────────────

def plot_method_comparison(df):
    """ssGSEA vs mean_zscore for all 5 filtered signatures."""
    method_colors = {'ssgsea': '#7EB5D6', 'mean_zscore': '#E07888'}

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))

    for i, sig in enumerate(SIGNATURES):
        ax = axes[i]
        for method, color, label in [
            ('ssgsea', method_colors['ssgsea'], 'ssGSEA'),
            ('mean_zscore', method_colors['mean_zscore'], 'Mean z-score'),
        ]:
            col = f'{method}__{sig}'
            if col not in df.columns:
                continue
            z, smooth = smooth_trajectory(df['z_progression'].values, df[col].values)
            smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-10)
            ax.plot(z, smooth_norm, color=color, linewidth=1.8, label=label)

        ax.axvspan(0.4, 0.6, alpha=0.08, color='#666666', zorder=0)
        ax.axvline(0.5, color='#999999', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_title(SIG_SHORT[sig], fontsize=10, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Z Progression', fontsize=8)
        if i == 0:
            ax.set_ylabel('Normalized Score', fontsize=8)
            ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=7)

    fig.suptitle('ssGSEA vs Mean z-score: Normalized Trajectory Comparison',
                 fontsize=11, fontweight='bold', y=1.03)
    fig.tight_layout()
    save_publication_pdf(fig, OUTPUT_DIR + 'scc_method_comparison_ssgsea_vs_zscore.pdf')
    plt.close(fig)
    print("Figure 6: Method comparison saved")


# ── Figure 7: Best Combinations ──────────────────────────────────────

def plot_best_combinations(df):
    """Top center-peaking combinations from filtered 5 signatures."""
    ssgsea_cols = {sig: f'ssgsea__{sig}' for sig in SIGNATURES}
    zscored = pd.DataFrame()
    for sig in SIGNATURES:
        col = ssgsea_cols[sig]
        vals = df[col].values
        zscored[sig] = (vals - vals.mean()) / (vals.std() + 1e-10)

    combos = []
    for n in [2, 3, 4]:
        for combo in combinations(SIGNATURES, n):
            score = zscored[list(combo)].mean(axis=1)
            m = compute_center_metrics(df['z_progression'].values, score.values)
            m['name'] = ' + '.join([SIG_SHORT[s] for s in combo])
            m['sigs'] = combo
            m['scores'] = score.values
            m['type'] = f'{n}-sig'
            combos.append(m)

    center_domes = [c for c in combos if c['is_dome'] and c['in_center']]
    center_domes.sort(key=lambda x: x['center_enrichment'], reverse=True)
    top = center_domes[:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, combo in enumerate(top):
        ax = axes[i]
        z, smooth = smooth_trajectory(df['z_progression'].values, combo['scores'])

        has_stem = any(s in STEM_SIGS for s in combo['sigs'])
        has_pluri = any(s in PLURI_SIGS for s in combo['sigs'])

        if has_stem and has_pluri:
            line_color = '#C8A87A'
        elif has_stem:
            line_color = '#E07888'
        elif has_pluri:
            line_color = '#C8A87A'
        else:
            line_color = '#7EB5D6'

        ax.fill_between(z, smooth, alpha=0.15, color=line_color)
        ax.plot(z, smooth, color=line_color, linewidth=2)

        ax.axvspan(0.4, 0.6, alpha=0.12, color='#666666', zorder=0)
        ax.axvline(0.5, color='#999999', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(combo['peak_z'], color=line_color, linewidth=0.8,
                    linestyle=':', alpha=0.7)

        ax.set_title(f"{combo['name']}\n({combo['type']}, CE={combo['center_enrichment']:+.3f})",
                     fontsize=8, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Z Progression', fontsize=8)
        ax.set_ylabel('Combined Score', fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle('Top Center-Peaking Signature Combinations (ssGSEA)',
                 fontsize=12, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_publication_pdf(fig, OUTPUT_DIR + 'scc_best_combinations_center_peak.pdf')
    plt.close(fig)
    print("Figure 7: Best combinations saved")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    import argparse

    global INPUT_PATH, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="SCC signature trajectory analysis along NESTOR progression")
    parser.add_argument('--input', type=str, default=INPUT_PATH,
                        help='Path to SCC scores parquet file')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for figures')
    args = parser.parse_args()

    INPUT_PATH = args.input
    OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} samples, {len(df.columns)} columns")

    # Fix column name encoding
    col_rename = {}
    for c in df.columns:
        fixed = c.replace('ï', 'i').replace('ì', 'i')
        if fixed != c:
            col_rename[c] = fixed
    if col_rename:
        df = df.rename(columns=col_rename)

    # Verify all signatures present
    for sig in SIGNATURES:
        col = f'ssgsea__{sig}'
        if col not in df.columns:
            print(f"  WARNING: {col} not found")
        else:
            print(f"  Found: {sig}")

    print("\nGenerating figures...")
    plot_individual_trajectories(df)
    plot_category_overlay_with_consensus(df)
    plot_enrichment_landscape(df)
    plot_consensus_dome(df)
    plot_center_enrichment_barplot(df)
    plot_method_comparison(df)
    plot_best_combinations(df)

    print("\n" + "=" * 60)
    print("SUMMARY: 5 Filtered Signatures (ssGSEA)")
    print("=" * 60)
    for sig in SIGNATURES:
        col = f'ssgsea__{sig}'
        m = compute_center_metrics(df['z_progression'].values, df[col].values)
        dome = "DOME" if m['is_dome'] else "    "
        center = "CENTER" if m['in_center'] else "      "
        print(f"  {dome} {center} {SIG_SHORT[sig]:20s} "
              f"peak_z={m['peak_z']:.3f}  CE={m['center_enrichment']:+.4f}")

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
