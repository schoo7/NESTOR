"""Generate Nature-level NESTOR pan-cancer summary figure.

Creates a multi-panel publication figure showing NESTOR z_progression
spatial mapping and distribution analysis across prostate, breast, and
lung cancer Visium datasets.

Panels:
    A-C: Representative spatial maps (one per cancer type)
    D:   Violin plot comparing z_progression distributions
    E:   Threshold enrichment bar chart
    F:   Per-sample swarm dot plot
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config._colors import DIVERGING_ENDPOINTS, PAN_CANCER_TYPE_COLORS, UI_COLORS
from plotting_utils.save_pdf import save_publication_pdf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIGURE_DIR = Path("figures/pan_cancer/nestor")

NESTOR_DIR = Path("Pan-cancer/nestor_results")

# Nature double-column width: 183mm, full page: 247mm
FIG_WIDTH_MM = 247
FIG_HEIGHT_MM = 280
MM_TO_INCH = 1 / 25.4

# Macaron palette cancer type colors (centralized in config)
CANCER_COLORS = PAN_CANCER_TYPE_COLORS

CANCER_ORDER = ["Prostate", "Breast", "Lung"]

# Representative samples (manually curated for spatial diversity)
REPRESENTATIVE_SAMPLES = {
    "Prostate": "GSM8558024",
    "Breast": "GSM6433585_092A",
    "Lung": "GSM9226171_P2_LUAD",
}

# NESTOR spatial colormap: blue -> white -> coral
NESTOR_CMAP = LinearSegmentedColormap.from_list(
    "nestor",
    [DIVERGING_ENDPOINTS["low"], DIVERGING_ENDPOINTS["mid"], DIVERGING_ENDPOINTS["high"]],
    N=256,
)

# Nature style defaults
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
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

SPOT_SIZE = 8
ALPHA_SPOT = 0.85
ALPHA_BG = 0.12


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_samples():
    """Load all NESTOR H5AD files and aggregate cancer spot data."""
    all_data = []

    for cancer_type in CANCER_ORDER:
        ct_dir = NESTOR_DIR / cancer_type.lower()
        if not ct_dir.exists():
            continue

        for h5ad_file in sorted(ct_dir.glob("*_nestor.h5ad")):
            if h5ad_file.stat().st_size < 1000:
                continue
            sample_id = h5ad_file.stem.replace("_nestor", "")
            try:
                adata = sc.read_h5ad(h5ad_file)
            except Exception:
                continue

            z = adata.obs["nestor_z_progression"].values.copy()
            cancer_col = adata.obs.get("is_cancer_spot")
            coords = adata.obsm.get("spatial")

            if cancer_col is not None:
                cancer_mask = cancer_col.values.astype(bool)
            else:
                cancer_mask = ~np.isnan(z)

            cancer_z = z[cancer_mask]
            n_cancer = cancer_mask.sum()
            n_total = len(z)

            if n_cancer < 5:
                continue

            all_data.append({
                "cancer_type": cancer_type,
                "sample_id": sample_id,
                "adata": adata,
                "cancer_z": cancer_z,
                "n_cancer": n_cancer,
                "n_total": n_total,
                "cancer_mask": cancer_mask,
                "mean_z": np.nanmean(cancer_z),
                "max_z": np.nanmax(cancer_z),
                "pct_010": 100 * np.nansum(cancer_z >= 0.10) / n_cancer,
                "pct_020": 100 * np.nansum(cancer_z >= 0.20) / n_cancer,
                "pct_030": 100 * np.nansum(cancer_z >= 0.30) / n_cancer,
                "pct_050": 100 * np.nansum(cancer_z >= 0.50) / n_cancer,
            })

    return all_data


# ---------------------------------------------------------------------------
# Panel plotting functions
# ---------------------------------------------------------------------------

def plot_spatial_map(ax, sample_info, cancer_type):
    """Plot a single spatial tissue map colored by NESTOR z_progression."""
    adata = sample_info["adata"]
    z = adata.obs["nestor_z_progression"].values.copy()
    cancer_mask = sample_info["cancer_mask"]
    coords = adata.obsm["spatial"]

    x_bg = coords[:, 0]
    y_bg = coords[:, 1]

    # Background (non-cancer spots) in light gray
    bg_mask = ~cancer_mask
    if bg_mask.any():
        ax.scatter(
            x_bg[bg_mask], y_bg[bg_mask],
            c=UI_COLORS["border"], s=SPOT_SIZE * 0.4, alpha=ALPHA_BG,
            edgecolors="none", rasterized=True,
        )

    # Cancer spots colored by z_progression
    valid = cancer_mask & ~np.isnan(z)
    if valid.any():
        sc_plot = ax.scatter(
            coords[valid, 0], coords[valid, 1],
            c=z[valid], cmap=NESTOR_CMAP,
            vmin=0, vmax=0.5,
            s=SPOT_SIZE, alpha=ALPHA_SPOT,
            edgecolors="none", rasterized=True,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()

    # Scale bar using axes-fraction coordinates for reliable sizing
    x_data_range = coords[:, 0].max() - coords[:, 0].min()
    # Pick a round scale bar length that is ~15-25% of tissue width
    target_frac = 0.2
    raw_len = target_frac * x_data_range
    # Round to nearest "nice" value: 200, 500, 1000, 2000, 5000
    nice_vals = [200, 500, 1000, 2000, 5000, 10000]
    bar_um = min(nice_vals, key=lambda v: abs(v - raw_len))
    bar_len_data = bar_um
    bar_x_start = coords[:, 0].max() - bar_len_data - x_data_range * 0.05
    y_range = coords[:, 1].max() - coords[:, 1].min()
    bar_y = coords[:, 1].min() + y_range * 0.06
    ax.plot(
        [bar_x_start, bar_x_start + bar_len_data],
        [bar_y, bar_y],
        color=UI_COLORS["grey_dark"], linewidth=2, solid_capstyle="butt",
    )
    ax.plot(
        [bar_x_start, bar_x_start],
        [bar_y - y_range * 0.008, bar_y + y_range * 0.008],
        color=UI_COLORS["grey_dark"], linewidth=1.5,
    )
    ax.plot(
        [bar_x_start + bar_len_data, bar_x_start + bar_len_data],
        [bar_y - y_range * 0.008, bar_y + y_range * 0.008],
        color=UI_COLORS["grey_dark"], linewidth=1.5,
    )
    ax.text(
        bar_x_start + bar_len_data / 2, bar_y + y_range * 0.018,
        f"{int(bar_um)} um", ha="center", va="bottom", fontsize=7,
        color=UI_COLORS["grey_dark"], fontweight="bold",
    )

    # Sample annotation
    mean_z = sample_info["mean_z"]
    max_z = sample_info["max_z"]
    n_cancer = sample_info["n_cancer"]
    label = f"{sample_info['sample_id']}\n{n_cancer} cancer spots\nmean = {mean_z:.3f}"
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes, fontsize=7,
        va="top", ha="left",
        color=UI_COLORS["grey_medium"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=UI_COLORS["border"]),
    )

    return sc_plot


def plot_violin_distributions(ax, all_data):
    """Plot half-violin distributions of z_progression per cancer type."""
    positions = []
    violin_data = []
    colors = []

    for i, ct in enumerate(CANCER_ORDER):
        ct_samples = [s for s in all_data if s["cancer_type"] == ct]
        all_z = np.concatenate([s["cancer_z"] for s in ct_samples])
        all_z = all_z[~np.isnan(all_z)]

        # Subsample for performance if needed
        if len(all_z) > 50000:
            rng = np.random.RandomState(42)
            all_z = rng.choice(all_z, 50000, replace=False)

        violin_data.append(all_z)
        positions.append(i)
        colors.append(CANCER_COLORS[ct])

    parts = ax.violinplot(
        violin_data,
        positions=positions,
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor(colors[i])
        body.set_alpha(0.4)
        # Make left half only (mirror effect)
        m = np.mean(body.get_paths()[0].vertices[:, 0])
        body.get_paths()[0].vertices[:, 0] = np.clip(
            body.get_paths()[0].vertices[:, 0], m, np.inf
        )

    # Add box plots (narrow, inside violins)
    bp_parts = ax.boxplot(
        violin_data,
        positions=positions,
        widths=0.15,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color=UI_COLORS["grey_dark"], linewidth=1.2),
        whiskerprops=dict(color=UI_COLORS["grey_medium"], linewidth=0.7),
        capprops=dict(color=UI_COLORS["grey_medium"], linewidth=0.7),
        boxprops=dict(linewidth=0.7),
    )
    for i, patch in enumerate(bp_parts["boxes"]):
        patch.set_facecolor("white")
        patch.set_edgecolor(colors[i])
        patch.set_linewidth(0.8)

    # Scatter individual sample means
    for i, ct in enumerate(CANCER_ORDER):
        ct_samples = [s for s in all_data if s["cancer_type"] == ct]
        means = [s["mean_z"] for s in ct_samples]
        jitter = np.random.RandomState(42).uniform(-0.12, 0.12, len(means))
        ax.scatter(
            [i + 0.22 + j for j in jitter], means,
            s=10, color=colors[i], alpha=0.6, zorder=5, edgecolors="none",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(CANCER_ORDER, fontsize=8.5, fontweight="bold")
    ax.set_ylabel("NESTOR z_progression", fontsize=8.5)
    ax.set_ylim(-0.02, 0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    # Sample count annotations
    for i, ct in enumerate(CANCER_ORDER):
        ct_samples = [s for s in all_data if s["cancer_type"] == ct]
        n_spots = sum(s["n_cancer"] for s in ct_samples)
        n_samples = len(ct_samples)
        ax.text(
            i, -0.055, f"n = {n_samples} samples\n{n_spots:,} spots",
            ha="center", va="top", fontsize=6.5, color=UI_COLORS["grey_muted"],
        )

    ax.axhline(y=0, color=UI_COLORS["border"], linewidth=0.3, linestyle="-")

    # Pairwise statistical comparisons (Mann-Whitney U)
    from scipy.stats import mannwhitneyu
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_labels = ["Prostate\nvs Breast", "Prostate\nvs Lung", "Breast\nvs Lung"]
    y_offsets = [0.55, 0.58, 0.61]
    for (i, j), label, y_off in zip(pairs, pair_labels, y_offsets):
        u_stat, p_val = mannwhitneyu(violin_data[i], violin_data[j], alternative="two-sided")
        if p_val < 1e-10:
            sig = "***"
        elif p_val < 1e-3:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"
        x_mid = (i + j) / 2
        ax.text(
            x_mid, y_off,
            f"p = {p_val:.1e} {sig}",
            ha="center", va="bottom", fontsize=6.5, color=UI_COLORS["grey_medium"],
        )


def plot_threshold_bars(ax, all_data):
    """Plot grouped bar chart of % cancer spots above thresholds."""
    thresholds = [0.05, 0.10, 0.20, 0.30]
    threshold_labels = [">=0.05", ">=0.10", ">=0.20", ">=0.30"]

    n_ct = len(CANCER_ORDER)
    n_thresh = len(thresholds)
    bar_width = 0.22
    group_width = n_ct * bar_width + 0.08

    for t_idx, thresh in enumerate(thresholds):
        for ct_idx, ct in enumerate(CANCER_ORDER):
            ct_samples = [s for s in all_data if s["cancer_type"] == ct]
            all_z = np.concatenate([s["cancer_z"] for s in ct_samples])
            pct = 100 * np.nansum(all_z >= thresh) / len(all_z[~np.isnan(all_z)])

            x = t_idx * group_width + ct_idx * bar_width
            bar = ax.bar(
                x, pct,
                width=bar_width - 0.02,
                color=CANCER_COLORS[ct],
                alpha=0.75,
                edgecolor="white",
                linewidth=0.3,
            )

            if pct > 3:
                ax.text(
                    x, pct + 1.5, f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=6,
                    color=UI_COLORS["grey_medium"],
                )

    ax.set_xticks(
        [t_idx * group_width + (n_ct - 1) * bar_width / 2 for t_idx in range(n_thresh)]
    )
    ax.set_xticklabels(threshold_labels, fontsize=7.5)
    ax.set_ylabel("% Cancer Spots", fontsize=8.5)
    ax.set_xlabel("z_progression Threshold", fontsize=8.5)
    ax.set_ylim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CANCER_COLORS[ct], alpha=0.75)
        for ct in CANCER_ORDER
    ]
    ax.legend(
        handles, CANCER_ORDER,
        loc="upper right", frameon=True,
        framealpha=0.9, edgecolor=UI_COLORS["border"],
        fontsize=7, ncol=3, columnspacing=0.8,
    )


def plot_sample_swarm(ax, all_data):
    """Plot per-sample mean z_progression as a dot plot."""
    # Collect per-sample data
    samples_by_ct = {ct: [] for ct in CANCER_ORDER}
    for s in all_data:
        samples_by_ct[s["cancer_type"]].append(s)

    # Sort each group by mean_z
    for ct in CANCER_ORDER:
        samples_by_ct[ct].sort(key=lambda x: x["mean_z"])

    # Layout: groups side by side
    group_gap = 1.5
    all_x = []
    all_y = []
    all_colors = []
    all_sizes = []
    group_centers = {}

    x_offset = 0
    for ct in CANCER_ORDER:
        samples = samples_by_ct[ct]
        n = len(samples)
        group_centers[ct] = x_offset + (n - 1) / 2

        for j, s in enumerate(samples):
            all_x.append(x_offset + j)
            all_y.append(s["mean_z"])
            all_colors.append(CANCER_COLORS[ct])
            all_sizes.append(max(12, min(60, s["n_cancer"] / 20)))
        x_offset += n + group_gap

    ax.scatter(
        all_x, all_y,
        c=all_colors, s=all_sizes,
        alpha=0.7, edgecolors="white", linewidths=0.3,
        zorder=5,
    )

    # Group labels
    for ct in CANCER_ORDER:
        samples = samples_by_ct[ct]
        center = group_centers[ct]
        means = [s["mean_z"] for s in samples]
        grand_mean = np.mean(means)

        # Horizontal line for group mean
        x_start = group_centers[ct] - len(samples) / 2 + 0.3
        x_end = group_centers[ct] + len(samples) / 2 - 0.3
        ax.plot(
            [x_start, x_end], [grand_mean, grand_mean],
            color=CANCER_COLORS[ct], linewidth=1.5, alpha=0.6, zorder=4,
        )

        ax.text(
            center, -0.025, ct,
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=CANCER_COLORS[ct],
        )
        ax.text(
            center, -0.045, f"n = {len(samples)}",
            ha="center", va="top", fontsize=6.5, color=UI_COLORS["grey_light"],
        )

    ax.set_ylabel("Mean z_progression per sample", fontsize=8.5)
    ax.set_ylim(-0.06, max(all_y) * 1.15)
    ax.set_xlim(-1, max(all_x) + 1)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    ax.axhline(y=0, color=UI_COLORS["border"], linewidth=0.3)

    # Size legend
    for sz, label in [(100, "100"), (500, "500"), (2000, "2000")]:
        ax.scatter([], [], s=max(15, sz / 30), c=UI_COLORS["grey_light"], alpha=0.5,
                   edgecolors="white", linewidths=0.3, label=f"{label} spots")
    leg = ax.legend(
        title="Cancer spots", loc="upper left",
        frameon=True, framealpha=0.9, edgecolor=UI_COLORS["border"],
        fontsize=6.5, title_fontsize=7,
    )


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------

def build_figure(all_data):
    """Assemble the complete Nature-level multi-panel figure."""
    fig_w = FIG_WIDTH_MM * MM_TO_INCH
    fig_h = FIG_HEIGHT_MM * MM_TO_INCH

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Grid: 4 rows
    # Row 0: Spatial maps (3 panels, taller)
    # Row 1: Violin plot + threshold bars (2 panels)
    # Row 2: Per-sample swarm (full width)
    gs = gridspec.GridSpec(
        3, 6,
        figure=fig,
        height_ratios=[1.3, 1.0, 0.8],
        hspace=0.35, wspace=0.45,
        left=0.06, right=0.94, top=0.91, bottom=0.06,
    )

    panel_label_style = dict(
        fontsize=12, fontweight="bold", color=UI_COLORS["grey_medium"],
        va="top", ha="left",
    )

    # ---- Row 0: Spatial maps (A, B, C) ----
    representative_samples = REPRESENTATIVE_SAMPLES
    sc_plots = []

    for i, ct in enumerate(CANCER_ORDER):
        ax = fig.add_subplot(gs[0, i * 2:(i + 1) * 2])
        sample_id = representative_samples[ct]
        sample_info = next(
            s for s in all_data
            if s["cancer_type"] == ct and s["sample_id"] == sample_id
        )

        sp = plot_spatial_map(ax, sample_info, ct)
        if sp is not None:
            sc_plots.append(sp)

        # Cancer type title
        ax.set_title(
            ct, fontsize=10, fontweight="bold",
            color=CANCER_COLORS[ct], pad=6,
        )

        # Panel label
        fig.text(
            ax.get_position().x0 - 0.01,
            ax.get_position().y1 + 0.005,
            chr(ord("A") + i),
            **panel_label_style,
        )

    # Shared colorbar for spatial maps
    if sc_plots:
        cbar_ax = fig.add_axes([0.35, 0.925, 0.30, 0.012])
        cbar = fig.colorbar(
            sc_plots[-1], cax=cbar_ax, orientation="horizontal",
        )
        cbar.set_label("NESTOR z_progression", fontsize=8, labelpad=3)
        cbar.ax.tick_params(labelsize=7, length=2)
        cbar.outline.set_linewidth(0.5)

    # ---- Row 1, Left: Violin distributions (D) ----
    ax_violin = fig.add_subplot(gs[1, :3])
    plot_violin_distributions(ax_violin, all_data)
    fig.text(
        ax_violin.get_position().x0 - 0.01,
        ax_violin.get_position().y1 + 0.005,
        "D",
        **panel_label_style,
    )

    # ---- Row 1, Right: Threshold bars (E) ----
    ax_bars = fig.add_subplot(gs[1, 3:])
    plot_threshold_bars(ax_bars, all_data)
    fig.text(
        ax_bars.get_position().x0 - 0.01,
        ax_bars.get_position().y1 + 0.005,
        "E",
        **panel_label_style,
    )

    # ---- Row 2: Per-sample swarm (F) ----
    ax_swarm = fig.add_subplot(gs[2, :])
    plot_sample_swarm(ax_swarm, all_data)
    fig.text(
        ax_swarm.get_position().x0 - 0.01,
        ax_swarm.get_position().y1 + 0.005,
        "F",
        **panel_label_style,
    )

    return fig


def main():
    import argparse

    global NESTOR_DIR, FIGURE_DIR

    parser = argparse.ArgumentParser(description="Generate NESTOR pan-cancer summary figure")
    parser.add_argument('--input-dir', type=Path, default=NESTOR_DIR,
                        help='Directory containing cancer-type subdirs with *_nestor.h5ad files')
    parser.add_argument('--output-dir', type=Path, default=FIGURE_DIR,
                        help='Output directory for the summary figure')
    args = parser.parse_args()

    NESTOR_DIR = args.input_dir
    FIGURE_DIR = args.output_dir
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NESTOR spatial data...")
    all_data = load_all_samples()

    # Summary
    for ct in CANCER_ORDER:
        ct_samples = [s for s in all_data if s["cancer_type"] == ct]
        if ct_samples:
            total_spots = sum(s["n_cancer"] for s in ct_samples)
            all_z = np.concatenate([s["cancer_z"] for s in ct_samples])
            print(f"  {ct}: {len(ct_samples)} samples, {total_spots:,} cancer spots, "
                  f"mean z={np.nanmean(all_z):.4f}")

    print("Building figure...")
    fig = build_figure(all_data)

    save_publication_pdf(fig, "02_pan_cancer_ne/summary/nestor_pan_cancer_spatial_summary.pdf")

    plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
