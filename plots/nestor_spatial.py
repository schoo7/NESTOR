#!/usr/bin/env python3
"""
Generate NESTOR NE progression spatial plots for pan-cancer analysis.

Creates per-sample spatial plots showing NESTOR z_progression scores
mapped onto tissue sections. Uses the project's centralized color scheme.

Plot types:
- 2D spatial: NESTOR score heatmap on tissue coordinates
- 3D density: Cancer cell density surface with NESTOR score overlay
- Cancer fraction: Spatial distribution of cancer cell abundance

Usage:
    python Pan-cancer/scripts/nestor_spatial.py prostate
    python Pan-cancer/scripts/nestor_spatial.py breast
    python Pan-cancer/scripts/nestor_spatial.py lung
    python Pan-cancer/scripts/nestor_spatial.py all
"""

import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='scanpy')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LightSource, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scanpy as sc
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from config._colors import (
    CANCER_SUBGROUP_COLORS, TISSUE_GROUP_CMAPS, CORRELATION_COLORS,
    UI_COLORS, NON_TUMOR_SPOT_COLOR, DIVERGING_ENDPOINTS,
)
from plotting_utils.save_pdf import save_publication_pdf
from plotting_utils.rasterize_3d import draw_contour_3d, rasterize_3d_axes


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIRS = {
    'prostate': os.path.join(BASE_DIR, 'Pan-cancer', 'nestor_results', 'prostate'),
    'breast': os.path.join(BASE_DIR, 'Pan-cancer', 'nestor_results', 'breast'),
    'lung': os.path.join(BASE_DIR, 'Pan-cancer', 'nestor_results', 'lung'),
}

FIGURE_DIRS = {
    'prostate': {
        'spatial': os.path.join(BASE_DIR, 'figures', 'pan_cancer', 'nestor', 'prostate', 'spatial'),
        '3d': os.path.join(BASE_DIR, 'figures', 'pan_cancer', 'nestor', 'prostate', '3d_density'),
    },
    'breast': {
        'spatial': os.path.join(BASE_DIR, 'figures', 'pan_cancer', 'nestor', 'breast', 'spatial'),
        '3d': os.path.join(BASE_DIR, 'figures', 'pan_cancer', 'nestor', 'breast', '3d_density'),
    },
    'lung': {
        'spatial': os.path.join(BASE_DIR, 'figures', 'pan_cancer', 'nestor', 'lung', 'spatial'),
        '3d': os.path.join(BASE_DIR, 'figures', 'pan_cancer', 'nestor', 'lung', '3d_density'),
    },
}

# NESTOR diverging colormap: blue -> white -> coral (matches summary figure)
NESTOR_CMAP_2D = LinearSegmentedColormap.from_list(
    "nestor",
    [DIVERGING_ENDPOINTS["low"], DIVERGING_ENDPOINTS["mid"], DIVERGING_ENDPOINTS["high"]],
    N=256,
)
NESTOR_CMAP_3D = 'plasma'
NESTOR_VMIN = 0.0
NESTOR_VMAX = 0.5
SPOT_SIZE = 8.0
ALPHA_SPOT = 0.85
ALPHA_BG = 0.12
FIG_DPI = 300

# Nature-style global font defaults
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C2L_PREFIX = 'q05cell_abundance_w_sf_'

# Load cancer cell types from authoritative JSON (single source of truth)
_CELL_TYPES_JSON = os.path.join(BASE_DIR, 'Pan-cancer', 'cancer_cell_types.json')
with open(_CELL_TYPES_JSON) as _f:
    _cell_types_data = json.load(_f)
CANCER_CELL_TYPES = {
    'prostate': _cell_types_data['prostate']['cancer_cell_types'],
    'breast': _cell_types_data['breast']['cancer_cell_types'],
    'lung': _cell_types_data['lung']['cancer_cell_types'],
}

CANCER_TYPE_LABELS = {
    'prostate': 'Prostate Cancer',
    'breast': 'Breast Cancer',
    'lung': 'Lung Cancer',
}


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def plot_nestor_spatial(adata, sample_name, cancer_type, output_dir):
    """Plot NESTOR z_progression as spatial heatmap.

    Single-panel design matching the summary figure style:
    blue-white-coral diverging colormap, Nature font sizing, scale bar,
    annotation box, no axis clutter.
    """
    if 'spatial' not in adata.obsm:
        return False, f"{sample_name}: No spatial coordinates"

    if 'nestor_z_progression' not in adata.obs.columns:
        return False, f"{sample_name}: No NESTOR results"

    coords = adata.obsm['spatial']
    z_prog = adata.obs['nestor_z_progression'].values
    cancer_mask = adata.obs['is_cancer_spot'].values if 'is_cancer_spot' in adata.obs.columns else ~np.isnan(z_prog)

    n_cancer = cancer_mask.sum()
    if n_cancer == 0:
        return False, f"{sample_name}: No cancer spots with NESTOR scores"

    # Use only valid (non-NaN) coordinates for figure sizing
    valid_mask = ~np.isnan(coords).any(axis=1)
    valid_coords = coords[valid_mask]
    x_range = valid_coords[:, 0].max() - valid_coords[:, 0].min()
    y_range = valid_coords[:, 1].max() - valid_coords[:, 1].min()
    aspect = x_range / y_range if y_range > 0 else 1.0
    fig_w = max(6, min(10, 6 * aspect))
    fig_h = fig_w / aspect if aspect > 1 else fig_w * aspect

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), facecolor="white")

    # Non-cancer spots as visible grey background
    non_cancer_mask = ~cancer_mask & valid_mask
    if non_cancer_mask.sum() > 0:
        ax.scatter(
            coords[non_cancer_mask, 0], coords[non_cancer_mask, 1],
            c=NON_TUMOR_SPOT_COLOR, s=SPOT_SIZE * 0.6, alpha=0.45,
            edgecolors="none", rasterized=True,
        )

    # Cancer spots colored by NESTOR z_progression
    valid = cancer_mask & ~np.isnan(z_prog)
    sc_plot = None
    if valid.sum() > 0:
        sc_plot = ax.scatter(
            coords[valid, 0], coords[valid, 1],
            c=z_prog[valid], cmap=NESTOR_CMAP_2D,
            vmin=NESTOR_VMIN, vmax=NESTOR_VMAX,
            s=SPOT_SIZE, alpha=ALPHA_SPOT,
            edgecolors="none", rasterized=True,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()

    # Scale bar with round um value
    target_frac = 0.2
    raw_len = target_frac * x_range
    nice_vals = [200, 500, 1000, 2000, 5000, 10000]
    bar_um = min(nice_vals, key=lambda v: abs(v - raw_len))
    bar_len_data = bar_um
    bar_x_start = valid_coords[:, 0].max() - bar_len_data - x_range * 0.05
    bar_y = valid_coords[:, 1].min() + y_range * 0.06
    ax.plot(
        [bar_x_start, bar_x_start + bar_len_data],
        [bar_y, bar_y],
        color=UI_COLORS['text_dark'], linewidth=2, solid_capstyle="butt",
    )
    ax.plot(
        [bar_x_start, bar_x_start],
        [bar_y - y_range * 0.008, bar_y + y_range * 0.008],
        color=UI_COLORS['text_dark'], linewidth=1.5,
    )
    ax.plot(
        [bar_x_start + bar_len_data, bar_x_start + bar_len_data],
        [bar_y - y_range * 0.008, bar_y + y_range * 0.008],
        color=UI_COLORS['text_dark'], linewidth=1.5,
    )
    ax.text(
        bar_x_start + bar_len_data / 2, bar_y + y_range * 0.018,
        f"{int(bar_um)} um", ha="center", va="bottom", fontsize=7,
        color=UI_COLORS['text_dark'], fontweight="bold",
    )

    # Annotation box (upper-left)
    z_valid = z_prog[valid]
    mean_z = np.nanmean(z_valid)
    max_z = np.nanmax(z_valid)
    pct_010 = 100 * np.nansum(z_valid >= 0.10) / len(z_valid)
    label = (f"{sample_name}\n"
             f"{n_cancer} cancer spots\n"
             f"mean = {mean_z:.3f}, max = {max_z:.3f}\n"
             f">= 0.10: {pct_010:.1f}%")
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes, fontsize=7,
        va="top", ha="left",
        color=UI_COLORS['text_dark'],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.85, edgecolor=UI_COLORS['border']),
    )

    # Colorbar
    if sc_plot is not None:
        cbar = fig.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.02,
                            shrink=0.8, aspect=25)
        cbar.set_label("NESTOR z_progression", fontsize=8, labelpad=3)
        cbar.ax.tick_params(labelsize=7, length=2)
        cbar.outline.set_linewidth(0.5)

    # Title
    ax.set_title(
        f"{CANCER_TYPE_LABELS[cancer_type]} - {sample_name}",
        fontsize=10, fontweight="bold", color=UI_COLORS['text_dark'], pad=6,
    )

    fig.tight_layout()
    save_publication_pdf(fig, f'02_pan_cancer_ne/nestor/{sample_name}_nestor_spatial.pdf')
    plt.close(fig)

    return True, (
        f"{sample_name}: {n_cancer} cancer spots, "
        f"NESTOR [{z_valid.min():.3f}, {z_valid.max():.3f}]"
    )


def plot_nestor_3d(adata, sample_name, cancer_type, output_dir, grid_resolution=200):
    """Plot 3D cancer density surface colored by NESTOR NE progression score.

    Matches the reference AdPCa/NEPCa 3D function style:
    - Rasterized surface content for Illustrator compatibility
    - Shadow projection at Z=0, contour lines at base
    - No panes, no grid, figure sized to tissue aspect ratio
    - Two-pass Gaussian smoothing, fallback interpolation
    - Compact colorbar, legend, proper 3D proportions
    """
    if 'spatial' not in adata.obsm:
        return False, f"{sample_name}: No spatial coordinates"

    if 'nestor_z_progression' not in adata.obs.columns:
        return False, f"{sample_name}: No NESTOR results"

    coords = adata.obsm['spatial']
    z_prog = adata.obs['nestor_z_progression'].values
    cancer_mask = adata.obs['is_cancer_spot'].values if 'is_cancer_spot' in adata.obs.columns else ~np.isnan(z_prog)

    # Filter NaN coordinates
    valid_mask = ~np.isnan(coords).any(axis=1)
    cancer_mask = cancer_mask & valid_mask

    if cancer_mask.sum() < 10:
        return False, f"{sample_name}: Too few cancer spots for 3D plot ({cancer_mask.sum()})"

    # Get cancer abundance
    cancer_abundance = None
    if 'q05_cell_abundance_w_sf' in adata.obsm:
        c2l = adata.obsm['q05_cell_abundance_w_sf']
        cancer_types = CANCER_CELL_TYPES[cancer_type]
        cancer_cols = [f'{C2L_PREFIX}{ct}' for ct in cancer_types
                       if f'{C2L_PREFIX}{ct}' in c2l.columns]
        if cancer_cols:
            cancer_abundance = c2l[cancer_cols].sum(axis=1).values

    if cancer_abundance is None:
        return False, f"{sample_name}: No cancer abundance data"

    # Use only valid (non-NaN) coordinates
    x_coords = coords[valid_mask, 0]
    y_coords = coords[valid_mask, 1]
    cancer_abundance = cancer_abundance[valid_mask]
    z_prog = z_prog[valid_mask]
    cancer_mask = cancer_mask[valid_mask]

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    aspect_ratio = x_range / y_range if y_range > 0 else 1.0

    # Grid resolution scaled to tissue aspect ratio
    effective_res = int(grid_resolution * 1.5)
    if aspect_ratio >= 1:
        x_res = effective_res
        y_res = int(effective_res / aspect_ratio)
    else:
        x_res = int(effective_res * aspect_ratio)
        y_res = effective_res
    x_res = max(x_res, 100)
    y_res = max(y_res, 100)

    xi = np.linspace(x_min, x_max, x_res)
    yi = np.linspace(y_min, y_max, y_res)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate cancer abundance onto grid
    try:
        density_grid = griddata(
            (x_coords, y_coords), cancer_abundance, (Xi, Yi),
            method='linear', fill_value=0,
        )
        density_grid = np.nan_to_num(density_grid, nan=0.0)
    except Exception:
        density_grid = griddata(
            (x_coords, y_coords), cancer_abundance, (Xi, Yi),
            method='nearest', fill_value=0,
        )
        density_grid = np.nan_to_num(density_grid, nan=0.0)

    # Interpolate NESTOR score (only for cancer spots)
    valid = cancer_mask & ~np.isnan(z_prog)
    if valid.sum() < 10:
        return False, f"{sample_name}: Too few NESTOR-scored spots"

    try:
        nestor_grid = griddata(
            (x_coords[valid], y_coords[valid]), z_prog[valid], (Xi, Yi),
            method='linear', fill_value=np.nan,
        )
        nestor_grid = np.nan_to_num(nestor_grid, nan=0.0)
    except Exception:
        nestor_grid = griddata(
            (x_coords[valid], y_coords[valid]), z_prog[valid], (Xi, Yi),
            method='nearest', fill_value=0,
        )
        nestor_grid = np.nan_to_num(nestor_grid, nan=0.0)

    # Two-pass Gaussian smoothing for smoother peaks
    sigma = max(1.0, min(x_res, y_res) / 150)
    density_grid = gaussian_filter(np.maximum(density_grid, 0), sigma=sigma)
    density_grid = gaussian_filter(np.maximum(density_grid, 0), sigma=sigma * 0.5)
    nestor_grid = gaussian_filter(nestor_grid, sigma=sigma)
    nestor_grid = gaussian_filter(nestor_grid, sigma=sigma * 0.5)

    density_grid = np.maximum(density_grid, 0)

    # Threshold low-density noise
    if np.any(density_grid > 0):
        threshold = np.percentile(density_grid[density_grid > 0], 5)
        density_grid[density_grid < threshold] = 0
        nestor_grid[density_grid < threshold] = 0

    if density_grid.max() == 0:
        return False, f"{sample_name}: No significant cancer density"

    # Figure sized to tissue aspect ratio, capped at 8 inches
    fig_w = min(8.0, 5.0 * max(aspect_ratio, 1.0))
    fig_h = min(8.0, 5.0 / min(aspect_ratio, 1.0))
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111, projection='3d')

    # No panes, no grid
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('none')
    ax.grid(False)

    display_max = max(density_grid.max() * 1.2, 10)
    density_norm = np.clip(density_grid / display_max, 0, 1)

    # Surface colored by NESTOR score with lighting
    nestor_cmap = mpl.colormaps[NESTOR_CMAP_3D]
    nestor_norm = Normalize(vmin=NESTOR_VMIN, vmax=NESTOR_VMAX)
    nestor_colors = nestor_cmap(nestor_norm(nestor_grid))

    ls = LightSource(azdeg=315, altdeg=45)
    illuminated = ls.shade_rgb(
        nestor_colors[:, :, :3], elevation=density_grid,
        vert_exag=0.8, blend_mode='overlay',
    )

    # Power-law alpha matching reference style
    alpha = np.power(density_norm, 0.7)
    alpha[density_grid <= 1e-8] = 0.0

    rgba = np.zeros((*illuminated.shape[:2], 4))
    rgba[:, :, :3] = illuminated
    rgba[:, :, 3] = alpha

    ax.plot_surface(
        Xi, Yi, density_grid, facecolors=rgba,
        antialiased=True, linewidth=0, shade=False,
        rstride=2, cstride=2, zorder=1,
    )

    # Shadow projection at Z=0
    if np.any(density_grid > 0):
        shadow_norm = np.clip(density_grid / density_grid.max(), 0, 1)
        shadow_alpha = shadow_norm * 0.15
        shadow_rgba = np.ones((*density_grid.shape, 4))
        shadow_rgba[:, :, :3] = 0.5
        shadow_rgba[:, :, 3] = shadow_alpha
        ax.plot_surface(
            Xi, Yi, np.zeros_like(Xi), facecolors=shadow_rgba,
            antialiased=False, linewidth=0, shade=False,
            rstride=4, cstride=4, zorder=0,
        )

    # Contour lines at base (rasterizable Line3D paths)
    if np.any(density_grid > 0):
        contour_levels = [l for l in np.linspace(0, display_max, 7) if l > 0]
        draw_contour_3d(ax, Xi, Yi, density_grid, contour_levels,
                       UI_COLORS['text_dark'], linewidth=0.3, alpha=0.6, z_offset=0)

    # Rasterize all 3D content for proportional Illustrator scaling
    _3d_dpi = rasterize_3d_axes(ax, dpi=600)

    # Axes styling
    ax.set_zlim(0, display_max)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_box_aspect([aspect_ratio, 1, 0.5])

    ax.set_xlabel('Spatial X', fontsize=8, labelpad=2)
    ax.set_ylabel('Spatial Y', fontsize=8, labelpad=2)
    ax.set_zlabel('Cancer Density', fontsize=8, labelpad=2)
    ax.tick_params(labelsize=6, pad=0)

    ax.xaxis.line.set_color('#CCCCCC')
    ax.yaxis.line.set_color('#CCCCCC')
    ax.zaxis.line.set_color('#CCCCCC')

    ax.view_init(elev=28, azim=135)
    ax.dist = 8
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Legend
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc=nestor_cmap(0.75), label='NESTOR Score'),
    ]
    ax.legend(
        handles=legend_elements, loc='upper left', fontsize=7,
        framealpha=0.9, edgecolor='#CCCCCC',
    )

    # Compact colorbar
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=nestor_cmap, norm=nestor_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.3, aspect=12, pad=0.02, location='right')
    cbar.set_label('NESTOR NE Progression', fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_edgecolor('#CCCCCC')

    fig.tight_layout()

    # Save with DPI for rasterized content, text stays vector
    save_publication_pdf(
        fig, f'02_pan_cancer_ne/nestor/{sample_name}_nestor_3d.pdf',
        dpi=_3d_dpi,
    )

    return True, f"{sample_name}: 3D plot generated, density max={density_grid.max():.1f}"


def process_cancer_type(cancer_type, skip_3d=False):
    """Generate all plots for a cancer type."""
    data_dir = DATA_DIRS[cancer_type]
    fig_dirs = FIGURE_DIRS[cancer_type]

    for subdir in fig_dirs.values():
        os.makedirs(subdir, exist_ok=True)

    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    h5ad_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.h5ad') and not f.startswith('._')
    ])

    if not h5ad_files:
        print(f"No H5AD files found in {data_dir}")
        return

    print(f"\n{'='*70}")
    print(f"Pan-Cancer NESTOR Visualization: {CANCER_TYPE_LABELS[cancer_type]}")
    print(f"{'='*70}")
    print(f"Input: {data_dir} ({len(h5ad_files)} files)")
    print(f"Spatial output: {fig_dirs['spatial']}")
    print(f"3D output: {fig_dirs['3d']}")
    print()

    ok_2d = 0
    ok_3d = 0
    failed = 0

    for i, f in enumerate(h5ad_files):
        filepath = os.path.join(data_dir, f)
        if os.path.getsize(filepath) < 1000:
            continue
        try:
            adata = sc.read_h5ad(filepath)
        except Exception:
            continue
        sample_name = f.replace('_nestor.h5ad', '').replace('_annotated.h5ad', '')

        # 2D spatial plot
        success, msg = plot_nestor_spatial(
            adata, sample_name, cancer_type, fig_dirs['spatial']
        )
        if success:
            ok_2d += 1
            print(f"  [{i+1}/{len(h5ad_files)}] 2D: {msg}")
        else:
            print(f"  [{i+1}/{len(h5ad_files)}] 2D SKIP: {msg}")

        # 3D density plot
        if not skip_3d:
            try:
                success_3d, msg_3d = plot_nestor_3d(
                    adata, sample_name, cancer_type, fig_dirs['3d']
                )
                if success_3d:
                    ok_3d += 1
                    print(f"  [{i+1}/{len(h5ad_files)}] 3D: {msg_3d}")
                else:
                    print(f"  [{i+1}/{len(h5ad_files)}] 3D SKIP: {msg_3d}")
            except Exception as e:
                print(f"  [{i+1}/{len(h5ad_files)}] 3D ERROR: {sample_name}: {e}")

    print(f"\nSummary for {CANCER_TYPE_LABELS[cancer_type]}:")
    print(f"  2D spatial plots: {ok_2d}/{len(h5ad_files)}")
    print(f"  3D density plots: {ok_3d}/{len(h5ad_files)}")


def main():
    global BASE_DIR, DATA_DIRS

    parser = argparse.ArgumentParser(
        description='Pan-cancer NESTOR spatial visualization')
    parser.add_argument(
        'cancer_type',
        choices=['prostate', 'breast', 'lung', 'all'],
        help='Cancer type to visualize')
    parser.add_argument(
        '--skip-3d', action='store_true',
        help='Skip 3D density plots (faster)')
    parser.add_argument(
        '--base-dir', type=str, default=BASE_DIR,
        help='Base directory containing Pan-cancer/nestor_results/')
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    DATA_DIRS = {
        'prostate': os.path.join(BASE_DIR, 'Pan-cancer', 'nestor_results', 'prostate'),
        'breast': os.path.join(BASE_DIR, 'Pan-cancer', 'nestor_results', 'breast'),
        'lung': os.path.join(BASE_DIR, 'Pan-cancer', 'nestor_results', 'lung'),
    }

    if args.cancer_type == 'all':
        for ct in ['prostate', 'breast', 'lung']:
            process_cancer_type(ct, skip_3d=args.skip_3d)
    else:
        process_cancer_type(args.cancer_type, skip_3d=args.skip_3d)


if __name__ == '__main__':
    main()
