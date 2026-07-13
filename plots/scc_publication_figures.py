#!/usr/bin/env python3
"""
scc_publication_figures.py

Slow-Cycling Cancer cell (SCC) signature enrichment along NESTOR NE progression.

Scores all 9 SCC signatures from Faouzi et al. 2018 JCI (PMID: 29944140)
using multiple unbiased methods, evaluates enrichment at the transition zone
(z ~ 0.5), and tests consensus combinations.

SCC = Slow-Cycling Cancer cells (NOT squamous cell carcinoma).

Input:
    - neAI/data/processed/cbio_pan_cancer_ne_final_hgnc.h5ad
    - neAI/outputs/nestor_scores_tumor_only.parquet
    - TET2_analysis/data/SCC_signatures.csv

Output:
    - publication/02_ne_progression/scc_bulk/

Usage:
    python plots/scc_publication_figures.py \
        --input-anndata <path/to/cbio_pan_cancer_ne_final_hgnc.h5ad> \
        --input-nestor <path/to/nestor_scores_tumor_only.parquet> \
        --input-signatures <path/to/SCC_signatures.csv> \
        --output-dir <path/to/output>
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import anndata
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent

from config._colors import (
    CORRELATION_COLORS,
    DIVERGING_ENDPOINTS,
    DIVERGING_SEQUENTIAL,
    LOG2FC_COLORS,
    NE_CATEGORY_COLORS,
    SCC_SIGNATURE_COLORS,
    UI_COLORS,
)
from config._paths import DIR_NE_SCC_BULK
from plotting_utils.save_pdf import save_pdf
from plotting_utils.save_pdf import save_publication_pdf

# ---------------------------------------------------------------------------
# Global rcParams
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("publication/02_ne_progression/scc_bulk")
DATA_DIR = Path("outputs/scc_unbiased_study")

INPUT_ANNDATA = Path("data/processed/cbio_pan_cancer_ne_final_hgnc.h5ad")
INPUT_NESTOR = Path("outputs/nestor_scores_tumor_only.parquet")
INPUT_SIGNATURES = Path("data/SCC_signatures.csv")

CACHE_FILE = DATA_DIR / "scc_9sig_scores.parquet"
EVALUATION_FILE = DATA_DIR / "scc_9sig_evaluation.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_PATH = Path("logs") / "scc_publication_figures.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BINS = 30

SIGNATURE_DISPLAY = {
    "PanC_SCC": "Pan-Cancer SCC",
    "Melanoma_SCC": "Melanoma SCC",
    "Glioma_Stem": "Glioma Stem",
    "AML_Stem": "AML Stem",
    "Stem-like-1": "Stem-like 1",
    "Stem-like-2": "Stem-like 2",
    "Stem-like ACS signature": "Stem-like ACS",
    "Naive hESC signature": "Naive hESC",
    "Primed hESC signature": "Primed hESC",
}

CONSENSUS_DEFS = {
    "Full consensus": None,
    "PanC + cancer": ["PanC_SCC", "Melanoma_SCC", "Glioma_Stem", "AML_Stem"],
    "PanC + stem": ["PanC_SCC", "Stem-like-1", "Stem-like-2", "Stem-like ACS signature"],
    "PanC + pluripotency": ["PanC_SCC", "Naive hESC signature", "Primed hESC signature"],
    "Cancer specific": ["Melanoma_SCC", "Glioma_Stem", "AML_Stem"],
    "Stem + pluripotency": [
        "Stem-like-1", "Stem-like-2", "Stem-like ACS signature",
        "Naive hESC signature", "Primed hESC signature",
    ],
}

METHOD_COLORS = {
    "mean_zscore": SCC_SIGNATURE_COLORS.get("panc_scc", "#7EB5D6"),
    "ssgsea": LOG2FC_COLORS.get("increased", "#E8A0B0"),
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_signatures() -> dict[str, list[str]]:
    """Load all 9 SCC signatures from TET2_analysis/data/SCC_signatures.csv."""
    df = pd.read_csv(INPUT_SIGNATURES, encoding="utf-8-sig")
    sigs = {}
    for col in df.columns:
        col_clean = col.strip()
        genes = [g for g in df[col].dropna().tolist() if g and isinstance(g, str)]
        sigs[col_clean] = genes
        logger.info("  %s: %d genes", col_clean, len(genes))
    return sigs


def load_expression_and_nestor() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load expression matrix and NE progression scores. Returns (expr_df, z_prog, gene_names)."""
    logger.info("Loading AnnData: %s", INPUT_ANNDATA)
    adata = anndata.read_h5ad(INPUT_ANNDATA)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)

    ne = pd.read_parquet(INPUT_NESTOR)
    if "sample_id" in ne.columns:
        ne = ne.set_index("sample_id") if ne.index.name != "sample_id" else ne
    elif "__index_level_0__" in ne.columns:
        ne = ne.set_index("__index_level_0__")

    common = adata.obs.index.intersection(ne.index)
    idx = [list(adata.obs.index).index(s) for s in common]

    expr = pd.DataFrame(X[idx], index=common, columns=adata.var_names)
    z_prog = ne.loc[common, "NE_progression"]

    logger.info("Expression: %d samples x %d genes, z_progression range [%.3f, %.3f]",
                expr.shape[0], expr.shape[1], z_prog.min(), z_prog.max())
    return expr, z_prog, adata.var_names.tolist()


# ---------------------------------------------------------------------------
# Scoring methods
# ---------------------------------------------------------------------------


def score_mean_zscore(expr: pd.DataFrame, genes: list[str]) -> np.ndarray:
    """Mean z-score of signature genes per sample."""
    overlap = [g for g in genes if g in expr.columns]
    if not overlap:
        return np.full(len(expr), np.nan)
    return expr[overlap].mean(axis=1).values


def score_ssgsea(expr: pd.DataFrame, genes: list[str], sig_name: str) -> np.ndarray:
    """ssGSEA enrichment score per sample using gseapy. Passes full expression matrix for proper ranking."""
    import gseapy as gp

    overlap = [g for g in genes if g in expr.columns]
    if len(overlap) < 5:
        logger.info("  ssgsea: skipping %s (%d genes overlap, below min_size=5)", sig_name, len(overlap))
        return np.full(len(expr), np.nan)

    # ssGSEA needs the FULL expression matrix (genes as rows, samples as columns)
    # to compute proper per-sample gene rankings
    expr_full = expr.T

    try:
        result = gp.ssgsea(
            data=expr_full,
            gene_sets={sig_name: overlap},
            outdir=None,
            sample_norm_method="rank",
            no_plot=True,
            threads=4,
            seed=42,
            min_size=5,
            max_size=2000,
        )
        scores = result.res2d.pivot(index="Term", columns="Name", values="NES")
        if sig_name in scores.index:
            return scores.loc[sig_name].reindex(expr.index).values.astype(float)
    except Exception as e:
        logger.warning("  ssgsea: failed for %s: %s", sig_name, e)

    return np.full(len(expr), np.nan)


def compute_all_scores(
    expr: pd.DataFrame,
    z_prog: pd.Series,
    signatures: dict[str, list[str]],
) -> pd.DataFrame:
    """Score all signatures with all methods. Returns DataFrame indexed by sample_id."""
    if CACHE_FILE.exists():
        logger.info("Loading cached scores from %s", CACHE_FILE)
        return pd.read_parquet(CACHE_FILE)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = {"sample_id": expr.index.tolist(), "z_progression": z_prog.values}

    for sig_name, genes in signatures.items():
        logger.info("Scoring %s (%d genes)...", sig_name, len(genes))

        # Mean z-score
        mz = score_mean_zscore(expr, genes)
        records[f"mean_zscore__{sig_name}"] = mz
        overlap = [g for g in genes if g in expr.columns]
        logger.info("  mean_zscore: %d/%d genes, range [%.3f, %.3f]",
                     len(overlap), len(genes), np.nanmin(mz), np.nanmax(mz))

        # ssGSEA
        ss = score_ssgsea(expr, genes, sig_name)
        records[f"ssgsea__{sig_name}"] = ss
        valid_ss = ss[~np.isnan(ss)]
        if len(valid_ss) > 0:
            logger.info("  ssgsea: range [%.3f, %.3f]", valid_ss.min(), valid_ss.max())

    df = pd.DataFrame(records)

    # Compute consensus scores (z-score each individual score, then average)
    for consensus_name, sig_list in CONSENSUS_DEFS.items():
        if sig_list is None:
            sig_list = list(signatures.keys())

        for method in ["mean_zscore", "ssgsea"]:
            cols = [f"{method}__{s}" for s in sig_list]
            existing = [c for c in cols if c in df.columns]
            if len(existing) < 2:
                continue
            zscored = df[existing].apply(lambda x: (x - x.mean()) / x.std())
            df[f"{method}__consensus__{consensus_name}"] = zscored.mean(axis=1)

    df.to_parquet(CACHE_FILE, index=False)
    logger.info("Saved scores to %s (%d rows, %d columns)", CACHE_FILE, len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Trajectory plotting
# ---------------------------------------------------------------------------


def compute_bin_stats(
    df: pd.DataFrame,
    score_col: str,
    n_bins: int = N_BINS,
) -> pd.DataFrame:
    """Bin z_progression into equal-width bins and compute summary stats."""
    bins = pd.cut(df["z_progression"], bins=n_bins, labels=False)
    grouped = df.groupby(bins, observed=True)
    stats_df = grouped.agg(
        bin_center=("z_progression", "mean"),
        n=("z_progression", "size"),
        median=(score_col, "median"),
        mean_val=(score_col, "mean"),
        q1=(score_col, lambda x: x.quantile(0.25)),
        q3=(score_col, lambda x: x.quantile(0.75)),
    ).dropna()
    return stats_df


def gam_smooth(x, y, n_knots=10):
    sort_idx = np.argsort(x)
    x_s = x[sort_idx]
    y_s = y[sort_idx]
    sigma = max(len(y_s) / (n_knots * 3), 0.5)
    y_smooth = gaussian_filter1d(y_s, sigma=sigma)
    return x_s, y_smooth


def _draw_trajectory(ax, merged, score_col, color, title, ylim=None):
    """Draw trajectory: median + IQR ribbon + GAM smooth, no dots."""
    valid = merged[["z_progression", score_col]].dropna()
    if len(valid) < 50:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="gray")
        ax.set_title(title)
        return
    bin_stats = compute_bin_stats(valid, score_col)
    if len(bin_stats) < 3:
        ax.text(0.5, 0.5, "Too few bins", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="gray")
        ax.set_title(title)
        return

    ax.fill_between(
        bin_stats["bin_center"], bin_stats["q1"], bin_stats["q3"],
        alpha=0.25, color=color, linewidth=0, label="Median (IQR)",
    )
    ax.plot(
        bin_stats["bin_center"], bin_stats["median"],
        color=color, linewidth=1.5, solid_capstyle="round", label="Median",
    )
    x_gam, y_gam = gam_smooth(
        bin_stats["bin_center"].values, bin_stats["median"].values, n_knots=8,
    )
    ax.plot(x_gam, y_gam, color=color, linewidth=1.0, linestyle="--", alpha=0.7,
            label="GAM smooth")

    peak_idx = bin_stats["median"].idxmax()
    trough_idx = bin_stats["median"].idxmin()
    ax.plot(bin_stats.loc[peak_idx, "bin_center"], bin_stats.loc[peak_idx, "median"],
            "^", color=color, markersize=6, zorder=5)
    ax.plot(bin_stats.loc[trough_idx, "bin_center"], bin_stats.loc[trough_idx, "median"],
            "v", color=color, markersize=6, zorder=5, alpha=0.7)

    valid_mask = merged[score_col].notna() & merged["z_progression"].notna()
    rho, _ = stats.spearmanr(
        merged.loc[valid_mask, "z_progression"],
        merged.loc[valid_mask, score_col],
    )
    n_samples = valid_mask.sum()
    peak_z = bin_stats.loc[peak_idx, "bin_center"]
    peak_val = bin_stats.loc[peak_idx, "median"]

    ax.text(
        0.97, 0.97,
        f"rho = {rho:.3f}\nn = {n_samples:,}\nPeak: z={peak_z:.2f}",
        transform=ax.transAxes,
        fontsize=6.5, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=UI_COLORS["surface"],
                  edgecolor=UI_COLORS["border"], alpha=0.85),
    )

    ax.set_xlabel("NE Progression (z)")
    ax.set_ylabel(title + " Score")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.legend(loc="lower left", fontsize=6, framealpha=0.85,
              edgecolor=UI_COLORS["border"])


def plot_9sig_grid(merged, method):
    """3x3 grid of individual signature trajectories for one scoring method."""
    sig_names = list(SIGNATURE_DISPLAY.keys())
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    axes_flat = axes.flatten()

    color = METHOD_COLORS.get(method, "#7EB5D6")

    for i, sig_name in enumerate(sig_names):
        ax = axes_flat[i]
        col = f"{method}__{sig_name}"
        if col not in merged.columns:
            ax.set_visible(False)
            continue
        display = SIGNATURE_DISPLAY[sig_name]
        _draw_trajectory(ax, merged, col, color, display)

    fig.suptitle(f"Slow-Cycling Cancer Cell Signatures ({method})", fontsize=12, y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"scc_9sig_grid_{method}.pdf"
    save_publication_pdf(fig, f"02_ne_progression/scc_bulk/scc_9sig_grid_{method}.pdf")
    save_pdf(fig, str(out_path))
    logger.info("  Saved: %s", out_path)
    plt.close(fig)


def plot_consensus_grid(merged, method):
    """Grid of consensus combination trajectories for one scoring method."""
    n_cons = len(CONSENSUS_DEFS)
    ncols = 3
    nrows = (n_cons + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes_flat = axes.flatten()

    color = LOG2FC_COLORS.get("increased", "#E8A0B0")

    for i, (cons_name, _) in enumerate(CONSENSUS_DEFS.items()):
        ax = axes_flat[i]
        col = f"{method}__consensus__{cons_name}"
        if col not in merged.columns:
            ax.set_visible(False)
            continue
        _draw_trajectory(ax, merged, col, color, cons_name)

    for j in range(n_cons, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Consensus SCC Combinations ({method})", fontsize=12, y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"scc_consensus_{method}.pdf"
    save_publication_pdf(fig, f"02_ne_progression/scc_bulk/scc_consensus_{method}.pdf")
    save_pdf(fig, str(out_path))
    logger.info("  Saved: %s", out_path)
    plt.close(fig)


def plot_panc_scc_panels(merged):
    """Side-by-side PanC_SCC scored by mean_zscore and ssGSEA."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    col_mz = "mean_zscore__PanC_SCC"
    col_ss = "ssgsea__PanC_SCC"

    if col_mz in merged.columns:
        _draw_trajectory(ax1, merged, col_mz,
                         SCC_SIGNATURE_COLORS.get("panc_scc", "#7EB5D6"),
                         "Pan-Cancer SCC (Mean Z-Score)")
    if col_ss in merged.columns:
        _draw_trajectory(ax2, merged, col_ss,
                         LOG2FC_COLORS.get("increased", "#E8A0B0"),
                         "Pan-Cancer SCC (ssGSEA)")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "scc_trajectory_panc_scc.pdf"
    save_publication_pdf(fig, "02_ne_progression/scc_bulk/scc_trajectory_panc_scc.pdf")
    save_pdf(fig, str(out_path))
    logger.info("  Saved: %s", out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_all(merged) -> pd.DataFrame:
    """Compute evaluation metrics for every score column."""
    results = []
    score_cols = [c for c in merged.columns if "__" in c and c != "z_progression"]

    for col in score_cols:
        parts = col.split("__", 1)
        method = parts[0]
        sig_raw = parts[1]

        valid = merged[["z_progression", col]].dropna()
        if len(valid) < 100:
            continue

        rho, p = stats.spearmanr(valid["z_progression"], valid[col])
        bin_stats = compute_bin_stats(valid, col)
        peak_z = bin_stats.loc[bin_stats["median"].idxmax(), "bin_center"]
        peak_val = bin_stats["median"].max()
        in_tz = 0.4 <= peak_z <= 0.6

        results.append({
            "method": method,
            "signature": sig_raw,
            "spearman_rho": rho,
            "p_value": p,
            "peak_z": peak_z,
            "peak_value": peak_val,
            "in_transition_zone": in_tz,
            "n_samples": len(valid),
        })

    eval_df = pd.DataFrame(results)
    eval_df = eval_df.sort_values(["method", "spearman_rho"], ascending=[True, False])
    eval_df.to_csv(EVALUATION_FILE, index=False)
    logger.info("Evaluation saved to %s (%d rows)", EVALUATION_FILE, len(eval_df))
    return eval_df


def plot_evaluation_summary(eval_df):
    """Bar chart showing peak z location for each signature x method."""
    df = eval_df[~eval_df["signature"].str.startswith("consensus")].copy()
    df["display"] = df["signature"].map(SIGNATURE_DISPLAY).fillna(df["signature"])

    methods = df["method"].unique()
    n_methods = len(methods)
    sigs = df["display"].unique()
    n_sigs = len(sigs)

    fig, ax = plt.subplots(figsize=(max(10, n_sigs * 1.2), 5))
    bar_width = 0.8 / n_methods
    x = np.arange(n_sigs)

    for mi, method in enumerate(methods):
        subset = df[df["method"] == method].set_index("display").reindex(sigs)
        offset = (mi - (n_methods - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, subset["peak_z"].values, bar_width,
            label=method, alpha=0.85,
            color=METHOD_COLORS.get(method, "#999999"),
            edgecolor=UI_COLORS["border"], linewidth=0.5,
        )
        for bar, val in zip(bars, subset["peak_z"].values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=5.5)

    ax.axhspan(0.4, 0.6, alpha=0.1, color=LOG2FC_COLORS.get("increased", "#E8A0B0"),
               label="Transition zone (0.4-0.6)")
    ax.set_xticks(x)
    ax.set_xticklabels(sigs, rotation=35, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Peak NE Progression (z)")
    ax.set_title("Peak SCC Activity Location by Signature and Method")
    ax.legend(fontsize=7)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "scc_evaluation_summary.pdf"
    save_publication_pdf(fig, "02_ne_progression/scc_bulk/scc_evaluation_summary.pdf")
    save_pdf(fig, str(out_path))
    logger.info("  Saved: %s", out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argparse

    global INPUT_ANNDATA, INPUT_NESTOR, INPUT_SIGNATURES, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Generate SCC publication figures")
    parser.add_argument("--input-anndata", type=Path, default=INPUT_ANNDATA,
                        help="Path to pan-cancer AnnData (.h5ad)")
    parser.add_argument("--input-nestor", type=Path, default=INPUT_NESTOR,
                        help="Path to NESTOR scores (.parquet)")
    parser.add_argument("--input-signatures", type=Path, default=INPUT_SIGNATURES,
                        help="Path to SCC signatures (.csv)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory for figures")
    args = parser.parse_args()

    INPUT_ANNDATA = args.input_anndata
    INPUT_NESTOR = args.input_nestor
    INPUT_SIGNATURES = args.input_signatures
    OUTPUT_DIR = args.output_dir

    logger.info("=" * 60)
    logger.info("SCC (Slow-Cycling Cancer) Publication Figures")
    logger.info("Started: %s", datetime.now().isoformat())
    logger.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load signatures
    logger.info("Loading SCC signatures...")
    signatures = load_signatures()

    # Load expression and NE progression
    expr, z_prog, gene_names = load_expression_and_nestor()

    # Score all signatures with all methods
    logger.info("Computing scores...")
    merged = compute_all_scores(expr, z_prog, signatures)

    # Generate figures
    logger.info("Generating trajectory figures...")
    for method in ["mean_zscore", "ssgsea"]:
        plot_9sig_grid(merged, method)
        plot_consensus_grid(merged, method)

    plot_panc_scc_panels(merged)

    # Evaluate
    logger.info("Evaluating enrichment...")
    eval_df = evaluate_all(merged)
    plot_evaluation_summary(eval_df)

    # Print summary
    logger.info("")
    logger.info("EVALUATION SUMMARY")
    logger.info("-" * 60)
    tz_hits = eval_df[eval_df["in_transition_zone"]]
    if len(tz_hits) > 0:
        logger.info("Signatures peaking in transition zone (z 0.4-0.6):")
        for _, row in tz_hits.iterrows():
            logger.info("  %s / %s: peak=%.3f, rho=%.3f",
                        row["method"], row["signature"],
                        row["peak_z"], row["spearman_rho"])
    else:
        logger.info("No signatures peak in transition zone (z 0.4-0.6)")

    logger.info("")
    logger.info("Top signatures by |rho|:")
    for _, row in eval_df.head(10).iterrows():
        logger.info("  %s / %s: rho=%.4f, peak_z=%.3f",
                    row["method"], row["signature"],
                    row["spearman_rho"], row["peak_z"])

    logger.info("-" * 60)
    logger.info("COMPLETE. Figures written to: %s", OUTPUT_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
