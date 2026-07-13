#!/usr/bin/env python3
"""PanC SCC signature enrichment on treatment-induced NE progression.

Computes the PanC SCC (475-gene) signature score on pseudobulk expression
data from single-cell treatment datasets, then tests whether SCC enrichment
is associated with NESTOR-detected treatment-induced NE progression.

Target studies: GSE207422 (LUAD), GSE199333 (AML), 33664492 (CRPC),
GSE236581 (CRC).

Analysis pipeline:
  1. Load pseudobulk counts matrix and NESTOR z_progression scores
  2. TPM-normalize, log2-transform, z-score PanC SCC genes
  3. Compute mean z-score SCC score per sample
  4. Classify samples: Progressed / Non-progressed (within-study quartiles)
  5. Unbiased differential analysis (Wilcoxon, Cohen's d, logistic regression,
     permutation test, leave-one-study-out CV)
  6. Generate 4-panel publication figure

Input:
    - neAI/data/processed/cellres_pseudobulk/ (counts_matrix.mtx, gene_names.txt, ...)
    - neAI/outputs/sc_nestor_results/sc_nestor_results_tpm_log2_zscore.parquet
    - neAI/config/scc_signatures/panc_scc.txt

Output:
    - publication/02_ne_progression/scc_treatment_induced/
    - neAI/outputs/scc_treatment_induced/

Usage:
    cd <your_working_directory>
    python scripts/scc_treatment.py
"""

import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.io import mmread
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent

from config._colors import (
    DIVERGING_ENDPOINTS,
    LOG2FC_COLORS,
    TREATMENT_CANCER_TYPE_COLORS,
    TREATMENT_COMPARISON_COLORS,
    UI_COLORS,
)
from config._paths import DIR_NE_SCC_TREATMENT
from plotting_utils.save_pdf import save_publication_pdf

# ---------------------------------------------------------------------------
# rcParams (publication quality)
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

PSEUDOBULK_DIR = Path("data/processed/cellres_pseudobulk")
NESTOR_RESULTS = Path(
    "outputs/sc_nestor_results/sc_nestor_results_tpm_log2_zscore.parquet"
)
PANC_SCC_FILE = Path("config/scc_signatures/panc_scc.txt")
OUTPUT_FIG_DIR = Path("publication/02_ne_progression/scc_treatment")
OUTPUT_DATA_DIR = Path("outputs/scc_treatment_induced")

LOG_PATH = Path("logs") / "scc_treatment.log"

TARGET_STUDIES = ["GSE207422", "GSE199333", "33664492", "GSE236581"]
MIN_POST_SAMPLES = 8
N_PERMUTATIONS = 10000
RANDOM_STATE = 42

SCORE_COL = "panc_scc"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COL_PRE = TREATMENT_COMPARISON_COLORS["pre"]
COL_POST = TREATMENT_COMPARISON_COLORS["post"]
COL_HIGHLIGHT = TREATMENT_COMPARISON_COLORS["highlight"]
COL_PROGRESSSED = LOG2FC_COLORS["increased"]
COL_NONPROGRESSED = LOG2FC_COLORS["decreased"]

STUDY_COLORS = {
    "GSE207422": TREATMENT_CANCER_TYPE_COLORS["LUAD"],
    "GSE199333": TREATMENT_CANCER_TYPE_COLORS["AML"],
    "33664492": TREATMENT_CANCER_TYPE_COLORS["CRPC"],
    "GSE236581": TREATMENT_CANCER_TYPE_COLORS["CRC"],
}

STUDY_LABELS = {
    "GSE207422": "LUAD: Toripalimab + Chemo",
    "GSE199333": "AML: Gilteritinib",
    "33664492": "CRPC: Enzalutamide",
    "GSE236581": "CRC: Sintilimab + Chemo",
}

PROGRESSION_COLORS = {
    "Pre": COL_PRE,
    "Non-progressed": COL_NONPROGRESSED,
    "Progressed": COL_PROGRESSSED,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data loading
# ============================================================================


def load_panc_scc_genes(path: Path) -> list[str]:
    return [g for g in path.read_text().splitlines() if g.strip()]


def load_pseudobulk(data_dir: Path):
    logger.info("Loading pseudobulk counts matrix...")
    counts = mmread(str(data_dir / "counts_matrix.mtx"))
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    counts = np.asarray(counts, dtype=np.float64)

    gene_names = [
        l.strip()
        for l in (data_dir / "gene_names.txt").read_text().splitlines()
        if l.strip()
    ]
    sample_names = [
        l.strip()
        for l in (data_dir / "sample_names.txt").read_text().splitlines()
        if l.strip()
    ]
    logger.info(
        "Counts matrix: %d genes x %d samples", counts.shape[0], counts.shape[1]
    )
    return counts, gene_names, sample_names


def load_nestor_results(path: Path) -> pd.DataFrame:
    logger.info("Loading NESTOR results from %s", path)
    df = pd.read_parquet(path)
    logger.info("NESTOR results: %d samples", len(df))
    return df


# ============================================================================
# SCC score computation
# ============================================================================


def compute_scc_scores(
    counts: np.ndarray,
    gene_names: list[str],
    sample_names: list[str],
    scc_genes: list[str],
) -> pd.DataFrame:
    """Compute PanC SCC mean z-score on TPM-normalized pseudobulk data."""
    logger.info("Computing PanC SCC scores...")

    gene_set = set(gene_names)
    overlap = [g for g in scc_genes if g in gene_set]
    logger.info("PanC SCC genes in matrix: %d / %d", len(overlap), len(scc_genes))

    sig_idx = [gene_names.index(g) for g in overlap]
    X_sig = counts[sig_idx, :].copy()

    # TPM normalization (uniform 1kb gene length)
    tpm = X_sig / 1.0
    lib_sizes = tpm.sum(axis=0, keepdims=True)
    lib_sizes[lib_sizes == 0] = 1.0
    tpm = tpm / lib_sizes * 1e6

    # Log2(x+1)
    tpm = np.log2(tpm + 1.0)

    # Z-score each gene across samples
    means = tpm.mean(axis=1, keepdims=True)
    stds = tpm.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    X_z = (tpm - means) / stds

    # Mean z-score across signature genes
    scores = np.nanmean(X_z, axis=0)
    logger.info(
        "  PanC SCC score range [%.3f, %.3f]", scores.min(), scores.max()
    )

    result = pd.DataFrame({SCORE_COL: scores}, index=sample_names)
    result.index.name = "col_key"
    return result.reset_index()


# ============================================================================
# Study ID derivation and merging
# ============================================================================


def derive_study_id(rds_file: str) -> str:
    for prefix in TARGET_STUDIES:
        if rds_file.startswith(prefix):
            return prefix
    base = rds_file.split("_")[0]
    if base.startswith("GSE") or base[0].isdigit():
        return base
    return base


def build_merged_dataset(
    scc_scores: pd.DataFrame,
    nestor_df: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Merging SCC scores with NESTOR results...")
    merged = nestor_df.merge(scc_scores, on="col_key", how="left")
    merged["study_id"] = merged["rds_file"].apply(derive_study_id)
    merged["is_target"] = merged["study_id"].isin(TARGET_STUDIES)
    logger.info(
        "Merged dataset: %d samples, %d in target studies",
        len(merged),
        merged["is_target"].sum(),
    )
    return merged


# ============================================================================
# Progression classification
# ============================================================================


def classify_progression(df: pd.DataFrame) -> pd.DataFrame:
    """Classify post-treatment samples as Progressed or Non-progressed
    using within-study quartile thresholds on z_progression."""
    df = df.copy()
    df["progression_group"] = "Pre"

    for study in df["study_id"].unique():
        mask_study = df["study_id"] == study
        mask_post = df["treatment"] == "Post"
        post_idx = df.index[mask_study & mask_post]

        if len(post_idx) < MIN_POST_SAMPLES:
            continue

        z_vals = df.loc[post_idx, "z_progression"]
        q75 = z_vals.quantile(0.75)
        q25 = z_vals.quantile(0.25)

        df.loc[post_idx[z_vals >= q75], "progression_group"] = "Progressed"
        df.loc[post_idx[z_vals <= q25], "progression_group"] = "Non-progressed"

    n_prog = (df["progression_group"] == "Progressed").sum()
    n_non = (df["progression_group"] == "Non-progressed").sum()
    n_pre = (df["progression_group"] == "Pre").sum()
    logger.info(
        "Progression classification: %d Pre, %d Non-progressed, %d Progressed",
        n_pre, n_non, n_prog,
    )
    return df


# ============================================================================
# Statistical analysis
# ============================================================================


def compute_correlation(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=[SCORE_COL, "z_progression"])
    rho, p = sp_stats.spearmanr(valid[SCORE_COL], valid["z_progression"])
    return {"rho": rho, "p_value": p, "n": len(valid)}


def compute_differential(df: pd.DataFrame) -> dict:
    prog = df[df["progression_group"] == "Progressed"][SCORE_COL].dropna()
    non = df[df["progression_group"] == "Non-progressed"][SCORE_COL].dropna()
    if len(prog) < 2 or len(non) < 2:
        return {"error": "insufficient samples"}

    stat, p = sp_stats.mannwhitneyu(prog, non, alternative="two-sided")
    pooled_std = np.sqrt(
        (prog.std() ** 2 * (len(prog) - 1) + non.std() ** 2 * (len(non) - 1))
        / (len(prog) + len(non) - 2)
    )
    cohens_d = (prog.mean() - non.mean()) / pooled_std if pooled_std > 0 else 0.0

    binary = df[df["progression_group"].isin(["Progressed", "Non-progressed"])].copy()
    binary["label"] = (binary["progression_group"] == "Progressed").astype(int)
    valid = binary.dropna(subset=[SCORE_COL])
    logit_coef = np.nan
    logit_odds = np.nan
    if len(valid) > 5:
        lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        lr.fit(valid[[SCORE_COL]].values, valid["label"].values)
        logit_coef = lr.coef_[0][0]
        logit_odds = np.exp(logit_coef)

    return {
        "wilcoxon_stat": stat,
        "wilcoxon_p": p,
        "cohens_d": cohens_d,
        "n_progressed": len(prog),
        "n_non_progressed": len(non),
        "prog_mean": prog.mean(),
        "non_mean": non.mean(),
        "logit_coef": logit_coef,
        "logit_odds_ratio": logit_odds,
    }


def compute_permutation_test(
    df: pd.DataFrame, n_perm: int = N_PERMUTATIONS
) -> dict:
    prog = df[df["progression_group"] == "Progressed"][SCORE_COL].dropna().values
    non = df[df["progression_group"] == "Non-progressed"][SCORE_COL].dropna().values
    if len(prog) < 2 or len(non) < 2:
        return {"error": "insufficient samples"}

    observed, _ = sp_stats.mannwhitneyu(prog, non, alternative="two-sided")
    combined = np.concatenate([prog, non])
    rng = np.random.default_rng(RANDOM_STATE)
    null_stats = np.zeros(n_perm)
    for i in range(n_perm):
        rng.shuffle(combined)
        null_stats[i], _ = sp_stats.mannwhitneyu(
            combined[: len(prog)], combined[len(prog):], alternative="two-sided"
        )
    empirical_p = (null_stats >= observed).mean()
    return {
        "observed_stat": observed,
        "empirical_p": empirical_p,
        "n_permutations": n_perm,
        "null_mean": null_stats.mean(),
    }


def compute_leave_one_study_out(df: pd.DataFrame) -> pd.DataFrame:
    binary = df[df["progression_group"].isin(["Progressed", "Non-progressed"])].copy()
    binary["label"] = (binary["progression_group"] == "Progressed").astype(int)
    results = []

    for holdout in TARGET_STUDIES:
        train = binary[(binary["is_target"]) & (binary["study_id"] != holdout)]
        test = binary[(binary["is_target"]) & (binary["study_id"] == holdout)]
        train = train.dropna(subset=[SCORE_COL])
        test = test.dropna(subset=[SCORE_COL])

        if len(train) < 5 or len(test) < 2:
            results.append({
                "held_out_study": holdout,
                "auroc": np.nan,
                "n_test": len(test),
                "n_progressed_test": (test["label"] == 1).sum(),
            })
            continue

        lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        lr.fit(train[[SCORE_COL]].values, train["label"].values)
        probs = lr.predict_proba(test[[SCORE_COL]].values)[:, 1]

        auroc = np.nan
        if test["label"].nunique() >= 2:
            auroc = roc_auc_score(test["label"].values, probs)

        results.append({
            "held_out_study": holdout,
            "auroc": auroc,
            "n_test": len(test),
            "n_progressed_test": (test["label"] == 1).sum(),
        })

    return pd.DataFrame(results)


def run_all_analyses(df: pd.DataFrame) -> dict:
    logger.info("Running statistical analyses...")
    target_df = df[df["is_target"]]

    correlation = compute_correlation(target_df)
    differential = compute_differential(target_df)
    permutation = compute_permutation_test(target_df)
    loso = compute_leave_one_study_out(target_df)

    logger.info(
        "  PanC SCC vs z_progression: rho=%.3f, p=%.2e",
        correlation["rho"], correlation["p_value"],
    )
    if "error" not in differential:
        logger.info(
            "  Progressed vs Non-progressed: Wilcoxon p=%.2e, Cohen's d=%.2f",
            differential["wilcoxon_p"], differential["cohens_d"],
        )
    logger.info("  Permutation test: empirical p=%.4f", permutation.get("empirical_p", "N/A"))
    logger.info("  Leave-one-study-out:\n%s", loso.to_string(index=False))

    return {
        "correlation": correlation,
        "differential": differential,
        "permutation": permutation,
        "leave_one_out": loso,
    }


# ============================================================================
# Figure generation
# ============================================================================


def _smooth_trajectory(x, y, n_bins=30, sigma=1.5):
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    medians = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() > 0:
            medians[i] = np.nanmedian(y[mask])
    valid = ~np.isnan(medians)
    if valid.sum() < 3:
        return centers, medians
    smoothed = gaussian_filter1d(medians[valid], sigma=sigma)
    out = np.full_like(medians, np.nan)
    out[valid] = smoothed
    return centers, out


def plot_panel_a_scatter(df: pd.DataFrame, analyses: dict, output_dir: Path):
    """Scatter: PanC SCC vs z_progression with OLS regression line and 95% CI."""
    all_samples = df.dropna(subset=[SCORE_COL, "z_progression"]).copy()

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Color by cancer type using codebase palette
    cancer_types = sorted(all_samples["cancer_type"].unique())
    cmap = {}
    for ct in cancer_types:
        cmap[ct] = TREATMENT_CANCER_TYPE_COLORS.get(ct, UI_COLORS["grey_pale"])

    for ct in cancer_types:
        subset = all_samples[all_samples["cancer_type"] == ct]
        ax.scatter(
            subset["z_progression"], subset[SCORE_COL],
            c=cmap[ct], s=12, alpha=0.5, edgecolors="none",
            zorder=2, label=f"{ct} (n={len(subset)})",
            rasterized=True,
        )

    # OLS regression line with 95% CI on all samples
    x_all = all_samples["z_progression"].values
    y_all = all_samples[SCORE_COL].values
    slope, intercept, r_value, p_value_ols, std_err = sp_stats.linregress(x_all, y_all)
    x_line = np.linspace(x_all.min(), x_all.max(), 200)
    y_line = slope * x_line + intercept

    n = len(x_all)
    x_bar = x_all.mean()
    sxx = np.sum((x_all - x_bar) ** 2)
    residuals = y_all - (slope * x_all + intercept)
    se_y = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    t_crit = sp_stats.t.ppf(0.975, n - 2)
    ci_width = t_crit * se_y * np.sqrt(1.0 / n + (x_line - x_bar) ** 2 / sxx)

    ax.fill_between(x_line, y_line - ci_width, y_line + ci_width,
                    color="#333333", alpha=0.12, zorder=3)
    ax.plot(x_line, y_line, color="#333333", linewidth=1.8, zorder=4)

    rho = analyses["correlation"]["rho"]
    p = analyses["correlation"]["p_value"]
    n_total = analyses["correlation"]["n"]
    p_str = f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"
    ax.text(
        0.97, 0.97,
        f"Spearman $\\rho$ = {rho:.3f}\n{p_str}\nn = {n_total}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8,
                  edgecolor=UI_COLORS["border"]),
    )

    ax.set_ylim(-2, 2)
    ax.set_xlabel("NE Progression (NESTOR z-score)")
    ax.set_ylabel("PanC SCC Signature Score")
    ax.legend(loc="lower right", fontsize=5.5, framealpha=0.9, ncol=2)
    ax.set_title("PanC SCC Enrichment and NE Progression")

    fig.tight_layout()
    save_publication_pdf(fig, output_dir / "panel_a_scatter.pdf", dpi=300)
    plt.close(fig)
    logger.info("Panel A saved: panel_a_scatter.pdf")


def plot_panel_b_violin(df: pd.DataFrame, analyses: dict, output_dir: Path):
    """Violin: PanC SCC scores by progression group."""
    target = df[df["is_target"]].copy()
    target = target[target["progression_group"].isin(["Pre", "Non-progressed", "Progressed"])]

    groups = ["Pre", "Non-progressed", "Progressed"]
    data = [target[target["progression_group"] == g][SCORE_COL].dropna().values for g in groups]

    fig, ax = plt.subplots(figsize=(5, 5.5))

    positions = [0, 1, 2]
    parts = ax.violinplot(data, positions=positions, showmeans=False,
                          showmedians=False, showextrema=False)

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(PROGRESSION_COLORS[groups[i]])
        body.set_alpha(0.4)
        body.set_edgecolor(PROGRESSION_COLORS[groups[i]])

    for i, (d, g) in enumerate(zip(data, groups)):
        bp = ax.boxplot([d], positions=[positions[i]], widths=0.15,
                        showfliers=False, patch_artist=True)
        bp["boxes"][0].set_facecolor(PROGRESSION_COLORS[g])
        bp["boxes"][0].set_alpha(0.7)
        bp["medians"][0].set_color("black")
        bp["medians"][0].set_linewidth(1.5)

    rng = np.random.default_rng(RANDOM_STATE)
    for i, (d, g) in enumerate(zip(data, groups)):
        jitter = rng.uniform(-0.12, 0.12, len(d))
        ax.scatter(
            positions[i] + jitter, d,
            c=PROGRESSION_COLORS[g], s=6, alpha=0.4, edgecolors="none", zorder=5,
            rasterized=True,
        )

    diff = analyses["differential"]
    if "error" not in diff and len(data[1]) > 0 and len(data[2]) > 0:
        y_max = max(data[1].max(), data[2].max())
        bracket_y = y_max + 0.05 * abs(y_max) if y_max != 0 else 0.1
        ax.plot([1, 1, 2, 2], [bracket_y, bracket_y + 0.02, bracket_y + 0.02, bracket_y],
                color="black", linewidth=0.8)
        p = diff["wilcoxon_p"]
        p_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(1.5, bracket_y + 0.025,
                f"{p_str}\nCohen's d = {diff['cohens_d']:.2f}",
                ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("PanC SCC Signature Score")
    ax.set_title("PanC SCC by Progression Status")

    fig.tight_layout()
    save_publication_pdf(fig, output_dir / "panel_b_violin.pdf", dpi=300)
    plt.close(fig)
    logger.info("Panel B saved: panel_b_violin.pdf")


def plot_panel_c_prepost(df: pd.DataFrame, output_dir: Path):
    """2x2 grid: Pre vs Post PanC SCC scores per target study."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, study in enumerate(TARGET_STUDIES):
        ax = axes[i]
        subset = df[(df["study_id"] == study) & (df["is_target"])].copy()
        pre = subset[subset["treatment"] == "Pre"][SCORE_COL].dropna()
        post = subset[subset["treatment"] == "Post"][SCORE_COL].dropna()

        if len(pre) == 0 and len(post) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(STUDY_LABELS.get(study, study))
            continue

        all_data = []
        positions = []
        colors = []
        labels = []
        pos = 0

        if len(pre) > 0:
            all_data.append(pre.values)
            positions.append(pos)
            colors.append(COL_PRE)
            labels.append(f"Pre (n={len(pre)})")
            pos += 1

        if len(post) > 0:
            all_data.append(post.values)
            positions.append(pos)
            colors.append(COL_POST)
            labels.append(f"Post (n={len(post)})")
            pos += 1

        parts = ax.violinplot(all_data, positions=positions, showmeans=False,
                              showmedians=False, showextrema=False)
        for j, body in enumerate(parts["bodies"]):
            body.set_facecolor(colors[j])
            body.set_alpha(0.4)
            body.set_edgecolor(colors[j])

        for j, (d, c) in enumerate(zip(all_data, colors)):
            bp = ax.boxplot([d], positions=[positions[j]], widths=0.12,
                            showfliers=False, patch_artist=True)
            bp["boxes"][0].set_facecolor(c)
            bp["boxes"][0].set_alpha(0.7)
            bp["medians"][0].set_color("black")
            bp["medians"][0].set_linewidth(1.5)

        rng = np.random.default_rng(RANDOM_STATE)
        for j, (d, c) in enumerate(zip(all_data, colors)):
            jitter = rng.uniform(-0.1, 0.1, len(d))
            ax.scatter(positions[j] + jitter, d, c=c, s=8, alpha=0.4, edgecolors="none",
                       rasterized=True)

        pre_mean = pre.mean() if len(pre) > 0 else np.nan
        post_mean = post.mean() if len(post) > 0 else np.nan
        delta_scc = post_mean - pre_mean if not np.isnan(pre_mean) else np.nan

        pre_z = subset[subset["treatment"] == "Pre"]["z_progression"].mean() if len(pre) > 0 else np.nan
        post_z = subset[subset["treatment"] == "Post"]["z_progression"].mean() if len(post) > 0 else np.nan
        delta_z = post_z - pre_z if not np.isnan(pre_z) else np.nan

        ann_text = ""
        if not np.isnan(delta_scc):
            ann_text += f"$\\Delta$SCC = {delta_scc:+.3f}\n"
        if not np.isnan(delta_z):
            ann_text += f"$\\Delta$z_prog = {delta_z:+.3f}"
        if ann_text:
            ax.text(0.97, 0.97, ann_text.strip(), transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8,
                              edgecolor=UI_COLORS["border"]))

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("PanC SCC Score")
        ax.set_title(STUDY_LABELS.get(study, study))

    fig.suptitle("PanC SCC: Pre- vs Post-Treatment", fontsize=12, y=1.01)
    fig.tight_layout()
    save_publication_pdf(fig, output_dir / "panel_c_prepost.pdf", dpi=300)
    plt.close(fig)
    logger.info("Panel C saved: panel_c_prepost.pdf")


def plot_panel_d_heatmap(df: pd.DataFrame, output_dir: Path):
    """Heatmap: PanC SCC score for 4 target studies sorted by z_progression."""
    target = df[df["is_target"]].copy()
    target = target.dropna(subset=[SCORE_COL, "z_progression"])
    target = target.sort_values("z_progression")

    scores = target[SCORE_COL].values.copy()
    std = scores.std()
    if std > 0:
        scores = (scores - scores.mean()) / std
    score_mat = scores.reshape(-1, 1)

    n_samples = len(target)

    fig, ax = plt.subplots(figsize=(2.5, max(4, n_samples * 0.02)))

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "macaron_div",
        [DIVERGING_ENDPOINTS["low"], DIVERGING_ENDPOINTS["mid"], DIVERGING_ENDPOINTS["high"]],
    )
    vmax = np.nanpercentile(np.abs(score_mat), 98)
    im = ax.imshow(score_mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                   interpolation="nearest")

    ax.set_xticks([0])
    ax.set_xticklabels(["PanC SCC"], fontsize=8)
    ax.set_yticks([])
    ax.set_ylabel(f"Samples (n={n_samples}), sorted by NE progression", fontsize=8)

    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.05)
    cb.set_label("Z-scored PanC SCC", fontsize=7)

    # Treatment sidebar
    treatment_colors = [COL_PRE if t == "Pre" else COL_POST for t in target["treatment"]]
    for i, c in enumerate(treatment_colors):
        ax.plot(-0.8, i, marker="s", color=c, markersize=1.5, clip_on=False)

    ax.set_xlim(-0.5, 0.5)
    ax.set_title("PanC SCC Along NE Progression")

    fig.tight_layout()
    save_publication_pdf(fig, output_dir / "panel_d_heatmap.pdf")
    plt.close(fig)
    logger.info("Panel D saved: panel_d_heatmap.pdf")


def generate_composite_figure(
    df: pd.DataFrame, analyses: dict, output_dir: Path
):
    """Generate 4-panel composite figure."""
    target = df[df["is_target"]].copy()
    target = target[target["progression_group"].isin(["Pre", "Non-progressed", "Progressed"])]

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # --- Panel A: Scatter ---
    ax_a = fig.add_subplot(gs[0, 0])
    other = df[~df["is_target"]].dropna(subset=[SCORE_COL, "z_progression"])
    ax_a.scatter(
        other["z_progression"], other[SCORE_COL],
        c=UI_COLORS["grey_pale"], s=6, alpha=0.2, edgecolors="none", zorder=1,
        rasterized=True,
    )
    for treatment, color in [("Pre", COL_PRE), ("Post", COL_POST)]:
        mask = target["treatment"] == treatment
        ax_a.scatter(
            target.loc[mask, "z_progression"], target.loc[mask, SCORE_COL],
            c=color, s=15, alpha=0.6, edgecolors="none", zorder=2, label=treatment,
            rasterized=True,
        )
    for study in TARGET_STUDIES:
        mask = target["study_id"] == study
        ax_a.scatter(
            target.loc[mask, "z_progression"], target.loc[mask, SCORE_COL],
            facecolors="none", edgecolors=STUDY_COLORS[study],
            s=30, linewidths=0.7, zorder=3, label=study,
            rasterized=True,
        )
    x = target["z_progression"].values
    y = target[SCORE_COL].values
    cx, cy = _smooth_trajectory(x, y, n_bins=25, sigma=1.5)
    valid = ~np.isnan(cy)
    ax_a.plot(cx[valid], cy[valid], color=COL_HIGHLIGHT, linewidth=2, zorder=4)
    rho = analyses["correlation"]["rho"]
    p = analyses["correlation"]["p_value"]
    p_str = f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"
    ax_a.text(0.97, 0.97, f"Spearman $\\rho$ = {rho:.3f}\n{p_str}",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=8,
              fontstyle="italic",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8,
                        edgecolor=UI_COLORS["border"]))
    ax_a.set_xlabel("NE Progression (NESTOR z-score)")
    ax_a.set_ylabel("PanC SCC Signature Score")
    ax_a.set_title("A. PanC SCC Enrichment vs NE Progression")
    ax_a.legend(loc="lower right", fontsize=6, framealpha=0.9, ncol=2)

    # --- Panel B: Violin ---
    ax_b = fig.add_subplot(gs[0, 1])
    groups = ["Pre", "Non-progressed", "Progressed"]
    data_b = [target[target["progression_group"] == g][SCORE_COL].dropna().values for g in groups]
    positions_b = [0, 1, 2]
    parts_b = ax_b.violinplot(data_b, positions=positions_b, showmeans=False,
                              showmedians=False, showextrema=False)
    for i, body in enumerate(parts_b["bodies"]):
        body.set_facecolor(PROGRESSION_COLORS[groups[i]])
        body.set_alpha(0.4)
        body.set_edgecolor(PROGRESSION_COLORS[groups[i]])
    for i, (d, g) in enumerate(zip(data_b, groups)):
        bp = ax_b.boxplot([d], positions=[positions_b[i]], widths=0.15,
                          showfliers=False, patch_artist=True)
        bp["boxes"][0].set_facecolor(PROGRESSION_COLORS[g])
        bp["boxes"][0].set_alpha(0.7)
        bp["medians"][0].set_color("black")
        bp["medians"][0].set_linewidth(1.5)
    rng = np.random.default_rng(RANDOM_STATE)
    for i, (d, g) in enumerate(zip(data_b, groups)):
        jitter = rng.uniform(-0.12, 0.12, len(d))
        ax_b.scatter(positions_b[i] + jitter, d, c=PROGRESSION_COLORS[g],
                     s=6, alpha=0.4, edgecolors="none", zorder=5, rasterized=True)
    diff = analyses["differential"]
    if "error" not in diff and len(data_b[1]) > 0 and len(data_b[2]) > 0:
        y_max = max(data_b[1].max(), data_b[2].max())
        bracket_y = y_max + 0.08 * abs(y_max) if y_max != 0 else 0.1
        ax_b.plot([1, 1, 2, 2], [bracket_y, bracket_y + 0.02, bracket_y + 0.02, bracket_y],
                  color="black", linewidth=0.8)
        p_val = diff["wilcoxon_p"]
        p_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        ax_b.text(1.5, bracket_y + 0.025,
                  f"{p_str} (Cohen's d = {diff['cohens_d']:.2f})",
                  ha="center", va="bottom", fontsize=7.5)
    ax_b.set_xticks(positions_b)
    ax_b.set_xticklabels(groups, fontsize=8)
    ax_b.set_ylabel("PanC SCC Signature Score")
    ax_b.set_title("B. PanC SCC by Progression Status")

    # --- Panel C: Pre vs Post per study ---
    ax_c_main = fig.add_subplot(gs[1, 0])
    ax_c_main.axis("off")
    ax_c_main.set_title("C. Pre- vs Post-Treatment PanC SCC", fontsize=10)

    gs_c = gs[1, 0].subgridspec(2, 2, hspace=0.4, wspace=0.3)
    for idx, study in enumerate(TARGET_STUDIES):
        ax = fig.add_subplot(gs_c[idx // 2, idx % 2])
        subset = df[(df["study_id"] == study) & (df["is_target"])].copy()
        pre_s = subset[subset["treatment"] == "Pre"][SCORE_COL].dropna()
        post_s = subset[subset["treatment"] == "Post"][SCORE_COL].dropna()

        all_d = []
        pos_list = []
        cols_list = []
        pos = 0
        if len(pre_s) > 0:
            all_d.append(pre_s.values)
            pos_list.append(pos)
            cols_list.append(COL_PRE)
            pos += 1
        if len(post_s) > 0:
            all_d.append(post_s.values)
            pos_list.append(pos)
            cols_list.append(COL_POST)
            pos += 1

        if all_d:
            vp = ax.violinplot(all_d, positions=pos_list, showmeans=False,
                               showmedians=False, showextrema=False)
            for j, body in enumerate(vp["bodies"]):
                body.set_facecolor(cols_list[j])
                body.set_alpha(0.4)
                body.set_edgecolor(cols_list[j])
            for j, (d, c) in enumerate(zip(all_d, cols_list)):
                bp = ax.boxplot([d], positions=[pos_list[j]], widths=0.12,
                                showfliers=False, patch_artist=True)
                bp["boxes"][0].set_facecolor(c)
                bp["boxes"][0].set_alpha(0.7)
                bp["medians"][0].set_color("black")
                bp["medians"][0].set_linewidth(1.2)
            rng2 = np.random.default_rng(RANDOM_STATE + idx)
            for j, (d, c) in enumerate(zip(all_d, cols_list)):
                jitter = rng2.uniform(-0.1, 0.1, len(d))
                ax.scatter(pos_list[j] + jitter, d, c=c, s=6, alpha=0.4, edgecolors="none",
                           rasterized=True)

        ax.set_xticks(pos_list)
        lbls = []
        if len(pre_s) > 0:
            lbls.append(f"Pre\nn={len(pre_s)}")
        if len(post_s) > 0:
            lbls.append(f"Post\nn={len(post_s)}")
        ax.set_xticklabels(lbls, fontsize=6.5)
        ax.set_title(STUDY_LABELS.get(study, study), fontsize=8)
        ax.set_ylabel("PanC SCC Score", fontsize=7)

    # --- Panel D: Heatmap ---
    ax_d = fig.add_subplot(gs[1, 1])
    hm_data = df[df["is_target"]].dropna(
        subset=[SCORE_COL, "z_progression"]
    ).sort_values("z_progression")
    scores = hm_data[SCORE_COL].values.copy()
    std = scores.std()
    if std > 0:
        scores = (scores - scores.mean()) / std
    score_mat = scores.reshape(-1, 1)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "macaron_div",
        [DIVERGING_ENDPOINTS["low"], DIVERGING_ENDPOINTS["mid"], DIVERGING_ENDPOINTS["high"]],
    )
    vmax = np.nanpercentile(np.abs(score_mat), 98)
    im = ax_d.imshow(score_mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                     interpolation="nearest")
    ax_d.set_xticks([0])
    ax_d.set_xticklabels(["PanC SCC"], fontsize=7)
    ax_d.set_yticks([])
    ax_d.set_ylabel(f"Samples (n={len(hm_data)}), sorted by NE progression", fontsize=7)
    cb = fig.colorbar(im, ax=ax_d, shrink=0.6, pad=0.05)
    cb.set_label("Z-scored PanC SCC", fontsize=7)
    ax_d.set_title("D. PanC SCC Along NE Progression Axis")

    save_publication_pdf(fig, output_dir / "fig_scc_treatment_progression.pdf", dpi=300)
    plt.close(fig)
    logger.info("Composite figure saved: fig_scc_treatment_progression.pdf")


# ============================================================================
# Save results
# ============================================================================


def save_results(df: pd.DataFrame, analyses: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_dir / "scc_treatment_scores.parquet", index=False)

    corr = analyses["correlation"]
    diff = analyses["differential"]
    perm = analyses["permutation"]

    rows = [
        {"analysis": "spearman_correlation",
         "rho": corr["rho"], "p_value": corr["p_value"], "n": corr["n"]},
        {"analysis": "wilcoxon_differential",
         "wilcoxon_p": diff.get("wilcoxon_p"),
         "cohens_d": diff.get("cohens_d"),
         "n_progressed": diff.get("n_progressed"),
         "n_non_progressed": diff.get("n_non_progressed"),
         "prog_mean": diff.get("prog_mean"),
         "non_mean": diff.get("non_mean"),
         "logit_odds_ratio": diff.get("logit_odds_ratio")},
        {"analysis": "permutation_test",
         "empirical_p": perm.get("empirical_p"),
         "n_permutations": perm.get("n_permutations")},
    ]
    pd.DataFrame(rows).to_csv(output_dir / "statistical_results.csv", index=False)

    target = df[df["is_target"]]
    summary = target.groupby(["study_id", "cancer_type", "treatment"]).agg(
        n_samples=(SCORE_COL, "size"),
        scc_mean=(SCORE_COL, "mean"),
        scc_std=(SCORE_COL, "std"),
        z_prog_mean=("z_progression", "mean"),
        z_prog_std=("z_progression", "std"),
    ).reset_index()
    summary.to_csv(output_dir / "per_study_summary.csv", index=False)

    analyses["leave_one_out"].to_csv(output_dir / "leave_one_study_out.csv", index=False)

    logger.info("Results saved to %s", output_dir)


# ============================================================================
# Unbiased analysis helpers (SCC / TET2 driver analysis)
# ============================================================================


def extract_gene_expression(
    counts: np.ndarray,
    gene_names: list[str],
    sample_names: list[str],
    genes: list[str],
) -> pd.DataFrame:
    """Extract log2-TPM expression for specified genes from pseudobulk matrix."""
    gene_set = set(gene_names)
    overlap = [g for g in genes if g in gene_set]
    idx = [gene_names.index(g) for g in overlap]
    sub = counts[idx, :].copy()

    tpm = sub / 1.0
    lib_sizes = tpm.sum(axis=0, keepdims=True)
    lib_sizes[lib_sizes == 0] = 1.0
    tpm = tpm / lib_sizes * 1e6
    tpm = np.log2(tpm + 1.0)

    result = pd.DataFrame(tpm.T, columns=overlap, index=sample_names)
    result.index.name = "col_key"
    return result.reset_index()


def compute_gene_level_correlations(
    counts: np.ndarray,
    gene_names: list[str],
    sample_names: list[str],
    scc_genes: list[str],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Spearman correlation of each PanC SCC gene with z_progression."""
    target = df[df["is_target"]].dropna(subset=["z_progression"]).copy()
    gene_set = set(gene_names)
    overlap = [g for g in scc_genes if g in gene_set]

    idx = [gene_names.index(g) for g in overlap]
    sub = counts[idx, :].copy()

    tpm = sub / 1.0
    lib_sizes = tpm.sum(axis=0, keepdims=True)
    lib_sizes[lib_sizes == 0] = 1.0
    tpm = tpm / lib_sizes * 1e6
    tpm = np.log2(tpm + 1.0)

    means = tpm.mean(axis=1, keepdims=True)
    stds = tpm.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    X_z = (tpm - means) / stds

    sample_idx = {s: i for i, s in enumerate(sample_names)}
    t_indices = [sample_idx[k] for k in target["col_key"] if k in sample_idx]
    y = target.set_index("col_key").loc[[sample_names[i] for i in t_indices], "z_progression"].values
    X_t = X_z[:, t_indices].T

    rhos, pvals = [], []
    for j in range(X_t.shape[1]):
        col = X_t[:, j]
        if col.std() < 1e-10:
            rhos.append(0.0)
            pvals.append(1.0)
            continue
        r, p = sp_stats.spearmanr(col, y)
        rhos.append(r)
        pvals.append(float(np.clip(p, 0.0, 1.0)))

    pvals_arr = np.array(pvals)
    n = len(pvals_arr)
    rank = np.argsort(np.argsort(pvals_arr)) + 1
    padj = np.minimum(pvals_arr * n / rank, 1.0)
    order = np.argsort(pvals_arr)
    padj_sorted = padj[order].copy()
    for i in range(len(padj_sorted) - 2, -1, -1):
        padj_sorted[i] = min(padj_sorted[i], padj_sorted[i + 1])
    padj_final = np.empty_like(padj)
    padj_final[order] = padj_sorted

    return pd.DataFrame({"gene": overlap, "rho": rhos, "p": pvals, "padj": padj_final}).sort_values("rho", ascending=False)


def compute_partial_correlations(df: pd.DataFrame) -> dict:
    """Partial correlation decomposition: SCC vs TET contributions to z_progression."""
    target = df[df["is_target"]].dropna(subset=[SCORE_COL, "tet_score", "z_progression"])
    scc = target[SCORE_COL].values
    tet = target["tet_score"].values
    z = target["z_progression"].values

    # SCC | TET
    coef_tet = np.polyfit(tet, z, 1)
    z_resid_tet = z - np.polyval(coef_tet, tet)
    rho_scc_given_tet, p_scc_given_tet = sp_stats.spearmanr(scc, z_resid_tet)

    # TET | SCC
    coef_scc = np.polyfit(scc, z, 1)
    z_resid_scc = z - np.polyval(coef_scc, scc)
    rho_tet_given_scc, p_tet_given_scc = sp_stats.spearmanr(tet, z_resid_scc)

    return {
        "scc_given_tet": {"rho": rho_scc_given_tet, "p": p_scc_given_tet},
        "tet_given_scc": {"rho": rho_tet_given_scc, "p": p_tet_given_scc},
    }


def compute_variance_partitioning(df: pd.DataFrame) -> pd.DataFrame:
    """Sequential variance partitioning of z_progression."""
    target = df[df["is_target"]].dropna(subset=[SCORE_COL, "tet_score", "z_progression"])
    y = target["z_progression"].values

    scc = target[[SCORE_COL]].values
    tet = target[["tet_score"]].values
    both = target[[SCORE_COL, "tet_score"]].values

    le = None
    both_full = target[[SCORE_COL, "tet_score"]].values
    if "cancer_type" in target.columns and target["cancer_type"].nunique() > 1:
        le = LabelEncoder()
        ct = le.fit_transform(target["cancer_type"].values).reshape(-1, 1)
        nc = target[["n_cells"]].values
        both_full = np.column_stack([both, ct, nc])

    models = [
        ("SCC alone", scc),
        ("TET alone", tet),
        ("SCC + TET", both),
        ("SCC + TET + covariates", both_full),
    ]
    rows = []
    for name, X in models:
        lr = LinearRegression().fit(X, y)
        rows.append({"model": name, "r2": lr.score(X, y)})

    result = pd.DataFrame(rows)
    result["incremental_r2"] = result["r2"] - result["r2"].shift(1).fillna(0)
    return result


def compute_per_study_concordance(df: pd.DataFrame) -> pd.DataFrame:
    """Per-study Spearman rho between SCC and z_progression."""
    target = df[df["is_target"]].dropna(subset=[SCORE_COL, "z_progression"])
    rows = []
    for study in TARGET_STUDIES:
        sub = target[target["study_id"] == study]
        if len(sub) < 5:
            continue
        rho, p = sp_stats.spearmanr(sub[SCORE_COL], sub["z_progression"])
        n = len(sub)
        # Fisher z-transform for CI
        se = 1.0 / np.sqrt(n - 3)
        z_val = np.arctanh(rho)
        ci_low = np.tanh(z_val - 1.96 * se)
        ci_high = np.tanh(z_val + 1.96 * se)
        rows.append({"study": study, "rho": rho, "p": p, "n": n,
                      "ci_low": ci_low, "ci_high": ci_high})

    # Combined
    rho_all, p_all = sp_stats.spearmanr(target[SCORE_COL], target["z_progression"])
    n_all = len(target)
    se_all = 1.0 / np.sqrt(n_all - 3)
    z_all = np.arctanh(rho_all)
    rows.append({"study": "Combined", "rho": rho_all, "p": p_all, "n": n_all,
                  "ci_low": np.tanh(z_all - 1.96 * se_all),
                  "ci_high": np.tanh(z_all + 1.96 * se_all)})
    return pd.DataFrame(rows)


# ============================================================================
# Unbiased analysis figures (panels E-I)
# ============================================================================


def plot_panel_e_volcano(gene_corrs: pd.DataFrame, output_dir: Path):
    """Volcano plot: PanC SCC gene correlations with z_progression."""
    df = gene_corrs.copy()
    df["neg_log_p"] = -np.log10(df["padj"].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(7, 5.5))

    sig_pos = df[(df["padj"] < 0.05) & (df["rho"] > 0)]
    sig_neg = df[(df["padj"] < 0.05) & (df["rho"] < 0)]
    ns = df[df["padj"] >= 0.05]

    ax.scatter(ns["rho"], ns["neg_log_p"], c=UI_COLORS["grey_pale"], s=8, alpha=0.4,
               edgecolors="none", label=f"n.s. (n={len(ns)})")
    ax.scatter(sig_pos["rho"], sig_pos["neg_log_p"], c="#E07888", s=10, alpha=0.5,
               edgecolors="none", label=f"Positive (n={len(sig_pos)})")
    ax.scatter(sig_neg["rho"], sig_neg["neg_log_p"], c="#5A9BBF", s=10, alpha=0.5,
               edgecolors="none", label=f"Negative (n={len(sig_neg)})")

    # Label top genes
    for _, row in df.head(5).iterrows():
        ax.annotate(row["gene"], (row["rho"], row["neg_log_p"]),
                    fontsize=6, fontstyle="italic", ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")
    for _, row in df.tail(5).iterrows():
        ax.annotate(row["gene"], (row["rho"], row["neg_log_p"]),
                    fontsize=6, fontstyle="italic", ha="right", va="bottom",
                    xytext=(-3, 3), textcoords="offset points")

    # FDR threshold line
    ax.axhline(-np.log10(0.05), color=UI_COLORS["grey_medium"], linestyle="--", linewidth=0.7)
    ax.text(0.02, -np.log10(0.05) + 0.3, "FDR = 0.05", fontsize=6.5, color=UI_COLORS["grey_medium"])

    ax.set_xlabel("Spearman correlation with NE progression")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title("E. PanC SCC Gene-Level Association with NE Progression")
    ax.legend(fontsize=6.5, loc="upper left")

    fig.tight_layout()
    save_publication_pdf(fig, output_dir / "panel_e_volcano.pdf")
    plt.close(fig)
    logger.info("Panel E saved: panel_e_volcano.pdf")


def plot_panel_f_partial_corr(partial: dict, output_dir: Path):
    """Bar chart: partial correlation decomposition."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    labels = ["SCC | TET", "TET | SCC"]
    rhos = [partial["scc_given_tet"]["rho"], partial["tet_given_scc"]["rho"]]
    ps = [partial["scc_given_tet"]["p"], partial["tet_given_scc"]["p"]]
    colors = ["#E07888", "#5A9BBF"]

    bars = ax.bar(labels, rhos, color=colors, width=0.5, edgecolor="white", linewidth=0.5)

    for bar, rho, p in zip(bars, rhos, ps):
        p_str = f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"
        y = bar.get_height() + 0.01
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"rho = {rho:.3f}\n{p_str}",
                ha="center", va="bottom", fontsize=8, fontstyle="italic")

    ax.set_ylabel("Partial Spearman rho")
    ax.set_title("F. Partial Correlation: SCC vs TET Contributions")
    ax.axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    save_publication_pdf(fig, output_dir / "panel_f_partial_corr.pdf")
    plt.close(fig)
    logger.info("Panel F saved: panel_f_partial_corr.pdf")


def plot_panel_g_tet2_stratified(df: pd.DataFrame, output_dir: Path):
    """Two-panel scatter: SCC vs z_progression stratified by TET2 expression."""
    target = df[df["is_target"]].dropna(subset=[SCORE_COL, "z_progression", "TET2_expr"]).copy()
    median_tet2 = target["TET2_expr"].median()
    target["tet2_group"] = np.where(target["TET2_expr"] >= median_tet2, "TET2-high", "TET2-low")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, grp in zip(axes, ["TET2-high", "TET2-low"]):
        sub = target[target["tet2_group"] == grp]
        ax.scatter(sub["z_progression"], sub[SCORE_COL], c=UI_COLORS["grey_medium"],
                   s=12, alpha=0.5, edgecolors="none", rasterized=True)

        # OLS regression line + CI
        x = sub["z_progression"].values
        y = sub[SCORE_COL].values
        if len(x) > 3:
            slope, intercept, _, _, _ = sp_stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            n = len(x)
            x_bar = x.mean()
            sxx = np.sum((x - x_bar) ** 2)
            residuals = y - (slope * x + intercept)
            se_y = np.sqrt(np.sum(residuals ** 2) / (n - 2))
            t_crit = sp_stats.t.ppf(0.975, n - 2)
            ci = t_crit * se_y * np.sqrt(1.0 / n + (x_line - x_bar) ** 2 / sxx)
            ax.fill_between(x_line, y_line - ci, y_line + ci, color="#333333", alpha=0.12)
            ax.plot(x_line, y_line, color="#333333", linewidth=1.5)

        rho, p = sp_stats.spearmanr(sub[SCORE_COL], sub["z_progression"])
        p_str = f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.97, 0.97, f"rho = {rho:.3f}\n{p_str}\nn = {len(sub)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8,
                          edgecolor=UI_COLORS["border"]))
        ax.set_xlabel("NE Progression (NESTOR z-score)")
        ax.set_ylabel("PanC SCC Score")
        ax.set_ylim(-2, 2)
        ax.set_title(f"{grp} (TET2 expr {'>' if grp == 'TET2-high' else '<'} median)")

    fig.suptitle("G. SCC-Progression Link Stratified by TET2 Expression", fontsize=10)
    fig.tight_layout()
    save_publication_pdf(fig, output_dir / "panel_g_tet2_stratified.pdf")
    plt.close(fig)
    logger.info("Panel G saved: panel_g_tet2_stratified.pdf")


def plot_panel_h_variance_partition(vp: pd.DataFrame, output_dir: Path):
    """Stacked bar chart: incremental variance explained."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    labels = vp["model"].values
    r2_total = vp["r2"].values
    incr = vp["incremental_r2"].values
    residual = float(1.0 - r2_total[-1])

    colors = ["#E07888", "#5A9BBF", "#A8C8A0", "#F7D49A"]
    bottoms = np.zeros(len(labels))
    for i, (label, val) in enumerate(zip(labels, incr)):
        c = colors[i] if i < len(colors) else UI_COLORS["grey_pale"]
        ax.bar("Predictors", val, bottom=bottoms[i], color=c, width=0.4,
               edgecolor="white", linewidth=0.5, label=f"{label}\n  R2 = {r2_total[i]:.3f}")
        if val > 0.005:
            ax.text(0, bottoms[i] + val / 2, f"{val:.3f}", ha="center", va="center", fontsize=7)

    ax.bar("Predictors", residual, bottom=r2_total[-1], color="#DDDDDD", width=0.4,
           edgecolor="white", linewidth=0.5, label=f"Residual\n  R2 = {residual:.3f}")

    ax.set_ylabel("R-squared")
    ax.set_title("H. Variance Partitioning of NE Progression")
    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.35, 1.0))

    fig.tight_layout()
    save_publication_pdf(fig, output_dir / "panel_h_variance_partition.pdf")
    plt.close(fig)
    logger.info("Panel H saved: panel_h_variance_partition.pdf")


def plot_panel_i_per_study(concordance: pd.DataFrame, output_dir: Path):
    """Forest plot: per-study SCC-progression concordance."""
    fig, ax = plt.subplots(figsize=(6, 4))

    studies = concordance["study"].values
    rhos = concordance["rho"].values
    ci_low = concordance["ci_low"].values
    ci_high = concordance["ci_high"].values
    ns = concordance["n"].values

    colors = [STUDY_COLORS.get(s, "#333333") for s in studies[:-1]] + ["#333333"]

    y_pos = np.arange(len(studies))
    for i in range(len(studies)):
        ax.plot([ci_low[i], ci_high[i]], [y_pos[i], y_pos[i]], color=colors[i], linewidth=2)
        ax.plot(rhos[i], y_pos[i], "o", color=colors[i], markersize=6)
        ax.text(ci_high[i] + 0.03, y_pos[i], f"rho={rhos[i]:.2f}, n={ns[i]}",
                va="center", fontsize=7)

    ax.axvline(0, color=UI_COLORS["grey_pale"], linestyle="--", linewidth=0.7)
    combined_rho = rhos[-1]
    ax.axvline(combined_rho, color="#333333", linestyle=":", linewidth=0.7, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(studies, fontsize=8)
    ax.set_xlabel("Spearman rho (PanC SCC vs NE progression)")
    ax.set_title("I. Per-Study Concordance: SCC-Progression Link")

    fig.tight_layout()
    save_publication_pdf(fig, output_dir / "panel_i_per_study.pdf")
    plt.close(fig)
    logger.info("Panel I saved: panel_i_per_study.pdf")


def generate_unbiased_composite(
    gene_corrs: pd.DataFrame,
    partial: dict,
    df: pd.DataFrame,
    vp: pd.DataFrame,
    concordance: pd.DataFrame,
    output_dir: Path,
):
    """Composite 5-panel unbiased analysis figure."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.4)

    # E: Volcano (top-left, spans 2 cols)
    ax_e = fig.add_subplot(gs[0, 0:2])
    gdf = gene_corrs.copy()
    gdf["neg_log_p"] = -np.log10(gdf["padj"].clip(lower=1e-300))
    sig_pos = gdf[(gdf["padj"] < 0.05) & (gdf["rho"] > 0)]
    sig_neg = gdf[(gdf["padj"] < 0.05) & (gdf["rho"] < 0)]
    ns = gdf[gdf["padj"] >= 0.05]
    ax_e.scatter(ns["rho"], ns["neg_log_p"], c=UI_COLORS["grey_pale"], s=6, alpha=0.4,
                 edgecolors="none", label=f"n.s. (n={len(ns)})")
    ax_e.scatter(sig_pos["rho"], sig_pos["neg_log_p"], c="#E07888", s=8, alpha=0.5,
                 edgecolors="none", label=f"Positive (n={len(sig_pos)})")
    ax_e.scatter(sig_neg["rho"], sig_neg["neg_log_p"], c="#5A9BBF", s=8, alpha=0.5,
                 edgecolors="none", label=f"Negative (n={len(sig_neg)})")
    for _, row in gdf.head(5).iterrows():
        ax_e.annotate(row["gene"], (row["rho"], row["neg_log_p"]),
                      fontsize=5.5, fontstyle="italic", xytext=(3, 3), textcoords="offset points")
    for _, row in gdf.tail(3).iterrows():
        ax_e.annotate(row["gene"], (row["rho"], row["neg_log_p"]),
                      fontsize=5.5, fontstyle="italic", xytext=(-3, 3), textcoords="offset points")
    ax_e.axhline(-np.log10(0.05), color=UI_COLORS["grey_medium"], linestyle="--", linewidth=0.7)
    ax_e.set_xlabel("Spearman rho")
    ax_e.set_ylabel("-log10(FDR)")
    ax_e.set_title("E. PanC SCC Gene-Level Association with NE Progression")
    ax_e.legend(fontsize=6, loc="upper left")

    # F: Partial correlation (top-right)
    ax_f = fig.add_subplot(gs[0, 2])
    labels_f = ["SCC | TET", "TET | SCC"]
    rhos_f = [partial["scc_given_tet"]["rho"], partial["tet_given_scc"]["rho"]]
    ps_f = [partial["scc_given_tet"]["p"], partial["tet_given_scc"]["p"]]
    colors_f = ["#E07888", "#5A9BBF"]
    bars = ax_f.bar(labels_f, rhos_f, color=colors_f, width=0.5, edgecolor="white")
    for bar, rho, p in zip(bars, rhos_f, ps_f):
        p_str = f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"
        ax_f.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                  f"rho={rho:.3f}\n{p_str}", ha="center", va="bottom", fontsize=7, fontstyle="italic")
    ax_f.set_ylabel("Partial Spearman rho")
    ax_f.set_title("F. Partial Correlation Decomposition")
    ax_f.axhline(0, color="black", linewidth=0.5)

    # G: TET2-stratified (bottom-left, spans 2 cols)
    target = df[df["is_target"]].dropna(subset=[SCORE_COL, "z_progression", "TET2_expr"]).copy()
    med = target["TET2_expr"].median()
    gs_g = gs[1, 0:2].subgridspec(1, 2, wspace=0.3)
    for gi, (grp, thresh) in enumerate([("TET2-high", target["TET2_expr"] >= med),
                                         ("TET2-low", target["TET2_expr"] < med)]):
        ax = fig.add_subplot(gs_g[gi])
        sub = target[thresh]
        ax.scatter(sub["z_progression"], sub[SCORE_COL], c=UI_COLORS["grey_medium"],
                   s=10, alpha=0.5, edgecolors="none", rasterized=True)
        x, y = sub["z_progression"].values, sub[SCORE_COL].values
        if len(x) > 3:
            slope, intercept, _, _, _ = sp_stats.linregress(x, y)
            xl = np.linspace(x.min(), x.max(), 100)
            ax.plot(xl, slope * xl + intercept, color="#333333", linewidth=1.5)
        rho, p = sp_stats.spearmanr(sub[SCORE_COL], sub["z_progression"])
        p_str = f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"
        ax.text(0.97, 0.97, f"rho={rho:.3f}\n{p_str}\nn={len(sub)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8,
                          edgecolor=UI_COLORS["border"]))
        ax.set_xlabel("NE Progression")
        ax.set_ylabel("PanC SCC Score")
        ax.set_ylim(-2, 2)
        ax.set_title(f"{grp}")
    ax_g_title = fig.add_subplot(gs[1, 0:2])
    ax_g_title.axis("off")
    ax_g_title.set_title("G. SCC-Progression Stratified by TET2 Expression", fontsize=10, y=1.08)

    # I: Per-study concordance (bottom-right)
    ax_i = fig.add_subplot(gs[1, 2])
    studies = concordance["study"].values
    rhos_i = concordance["rho"].values
    ci_low_i = concordance["ci_low"].values
    ci_high_i = concordance["ci_high"].values
    y_pos = np.arange(len(studies))
    colors_i = [STUDY_COLORS.get(s, "#333333") for s in studies[:-1]] + ["#333333"]
    for i in range(len(studies)):
        ax_i.plot([ci_low_i[i], ci_high_i[i]], [y_pos[i], y_pos[i]], color=colors_i[i], linewidth=2)
        ax_i.plot(rhos_i[i], y_pos[i], "o", color=colors_i[i], markersize=5)
    ax_i.axvline(0, color=UI_COLORS["grey_pale"], linestyle="--", linewidth=0.7)
    ax_i.set_yticks(y_pos)
    ax_i.set_yticklabels(studies, fontsize=7)
    ax_i.set_xlabel("Spearman rho")
    ax_i.set_title("I. Per-Study Concordance")

    save_publication_pdf(fig, output_dir / "fig_scc_unbiased_analysis.pdf", dpi=300)
    plt.close(fig)
    logger.info("Unbiased composite saved: fig_scc_unbiased_analysis.pdf")


# ============================================================================
# Main
# ============================================================================


def main():
    import argparse

    global PSEUDOBULK_DIR, NESTOR_RESULTS, PANC_SCC_FILE, OUTPUT_FIG_DIR, OUTPUT_DATA_DIR

    parser = argparse.ArgumentParser(description="PanC SCC treatment-induced NE progression analysis")
    parser.add_argument('--pseudobulk-dir', type=Path, default=PSEUDOBULK_DIR,
                        help='Directory containing pseudobulk data')
    parser.add_argument('--nestor-results', type=Path, default=NESTOR_RESULTS,
                        help='Path to NESTOR results parquet')
    parser.add_argument('--panc-scc', type=Path, default=PANC_SCC_FILE,
                        help='Path to PanC SCC gene list')
    parser.add_argument('--output-fig-dir', type=Path, default=OUTPUT_FIG_DIR,
                        help='Output directory for figures')
    parser.add_argument('--output-data-dir', type=Path, default=OUTPUT_DATA_DIR,
                        help='Output directory for data')
    args = parser.parse_args()

    PSEUDOBULK_DIR = args.pseudobulk_dir
    NESTOR_RESULTS = args.nestor_results
    PANC_SCC_FILE = args.panc_scc
    OUTPUT_FIG_DIR = args.output_fig_dir
    OUTPUT_DATA_DIR = args.output_data_dir

    logger.info("=" * 60)
    logger.info("PanC SCC Treatment-Induced NE Progression Analysis")
    logger.info("=" * 60)

    # Load data
    counts, gene_names, sample_names = load_pseudobulk(PSEUDOBULK_DIR)
    nestor_df = load_nestor_results(NESTOR_RESULTS)
    scc_genes = load_panc_scc_genes(PANC_SCC_FILE)

    # Compute SCC scores
    scc_scores = compute_scc_scores(counts, gene_names, sample_names, scc_genes)

    # Build merged dataset
    merged = build_merged_dataset(scc_scores, nestor_df)

    # Add TET signature and key gene expression
    tet_genes = ["TET1", "TET2", "TET3"]
    tet_expr = extract_gene_expression(counts, gene_names, sample_names, tet_genes)
    merged = merged.merge(tet_expr, on="col_key", how="left")
    merged["tet_score"] = merged[tet_genes].mean(axis=1)

    key_genes = ["TET2", "TP63", "ASCL1", "DNMT3A", "SOX2", "KRT5", "CHGA", "NEUROD1"]
    key_expr = extract_gene_expression(counts, gene_names, sample_names, key_genes)
    for g in key_genes:
        if g in key_expr.columns:
            merged[f"{g}_expr"] = merged["col_key"].map(
                key_expr.set_index("col_key")[g].to_dict()
            )

    # Classify progression
    merged = classify_progression(merged)

    # Run analyses (on target studies only for statistics)
    analyses = run_all_analyses(merged)

    # Generate original figures (A-D)
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_panel_a_scatter(merged, analyses, OUTPUT_FIG_DIR)
    plot_panel_b_violin(merged, analyses, OUTPUT_FIG_DIR)
    plot_panel_c_prepost(merged, OUTPUT_FIG_DIR)
    plot_panel_d_heatmap(merged, OUTPUT_FIG_DIR)
    generate_composite_figure(merged, analyses, OUTPUT_FIG_DIR)

    # Unbiased driver analysis (panels E-I)
    logger.info("Running unbiased driver analysis...")
    gene_corrs = compute_gene_level_correlations(counts, gene_names, sample_names, scc_genes, merged)
    sig = (gene_corrs["padj"] < 0.05).sum()
    logger.info("  Gene-level: %d / %d significant (FDR < 0.05)", sig, len(gene_corrs))

    partial = compute_partial_correlations(merged)
    logger.info("  Partial corr SCC|TET: rho=%.3f, TET|SCC: rho=%.3f",
                partial["scc_given_tet"]["rho"], partial["tet_given_scc"]["rho"])

    vp = compute_variance_partitioning(merged)
    logger.info("  Variance partitioning:\n%s", vp.to_string(index=False))

    concordance = compute_per_study_concordance(merged)
    logger.info("  Per-study concordance:\n%s", concordance.to_string(index=False))

    plot_panel_e_volcano(gene_corrs, OUTPUT_FIG_DIR)
    plot_panel_f_partial_corr(partial, OUTPUT_FIG_DIR)
    plot_panel_g_tet2_stratified(merged, OUTPUT_FIG_DIR)
    plot_panel_h_variance_partition(vp, OUTPUT_FIG_DIR)
    plot_panel_i_per_study(concordance, OUTPUT_FIG_DIR)
    generate_unbiased_composite(gene_corrs, partial, merged, vp, concordance, OUTPUT_FIG_DIR)

    # Save results
    save_results(merged, analyses, OUTPUT_DATA_DIR)
    gene_corrs.to_csv(OUTPUT_DATA_DIR / "scc_gene_correlations.csv", index=False)

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
