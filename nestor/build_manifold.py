#!/usr/bin/env python3
"""
build_manifold.py

Build manifold preprocessing for NE-GATE training.

This script performs:
1. Seurat-style cell-cycle scoring (S-phase and G2M-phase)
2. Cell-cycle regression to remove proliferation signal
3. Optional SVD residuals to remove batch effects
4. k-NN graph construction using pynndescent

The output is used for manifold-guided training.

Usage:
    python scripts/build_manifold.py
    python scripts/build_manifold.py --use-svd  # Enable SVD batch correction

Input: data/processed/cbio_pan_cancer_ne_per_sample_zscore.h5ad
Output:
    - data/processed/manifold_neighbors.parquet (k-NN indices and distances)
    - data/processed/cellcycle_scores.parquet (S-score and G2M-score per sample)
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import anndata
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import issparse
from sklearn.linear_model import LinearRegression

# =============================================================================
# Named Constants
# =============================================================================
N_NEIGHBORS = 15  # Number of neighbors for k-NN graph
SVD_K = 5  # Number of SVD components to remove (if --use-svd)
RANDOM_STATE = 42

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cbio_pan_cancer_ne_per_sample_zscore.h5ad"
CELLCYCLE_GENES_PATH = PROJECT_ROOT / "config" / "seurat_cellcycle_genes.txt"
OUTPUT_NEIGHBORS_PATH = Path("data/manifold_neighbors.parquet")
OUTPUT_SCORES_PATH = Path("data/cellcycle_scores.parquet")
LOG_PATH = Path("logs") / "build_manifold.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH),
    ],
)
logger = logging.getLogger(__name__)


class CellCycleGenes(NamedTuple):
    """Container for cell-cycle gene sets."""

    s_genes: list[str]
    g2m_genes: list[str]


def parse_cellcycle_genes(filepath: Path) -> CellCycleGenes:
    """
    Parse Seurat-style cell-cycle gene file.

    The file format is:
        # S Phase Genes
        GENE1
        GENE2
        ...
        # G2M Phase Genes
        GENE1
        GENE2
        ...

    Args:
        filepath: Path to cell-cycle gene file

    Returns:
        CellCycleGenes with s_genes and g2m_genes lists
    """
    logger.info(f"Parsing cell-cycle genes from {filepath}")

    with open(filepath) as f:
        lines = f.read().strip().split("\n")

    s_genes = []
    g2m_genes = []
    current_phase = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("# S Phase"):
            current_phase = "S"
        elif stripped.startswith("# G2M Phase"):
            current_phase = "G2M"
        elif current_phase == "S":
            s_genes.append(stripped)
        elif current_phase == "G2M":
            g2m_genes.append(stripped)

    logger.info(f"Found {len(s_genes)} S-phase genes and {len(g2m_genes)} G2M-phase genes")

    return CellCycleGenes(s_genes=s_genes, g2m_genes=g2m_genes)


def compute_cellcycle_scores(
    X: np.ndarray,
    gene_names: np.ndarray,
    cc_genes: CellCycleGenes,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute Seurat-style cell-cycle scores.

    For each sample:
        S-score = mean(expression of S-phase genes)
        G2M-score = mean(expression of G2M-phase genes)

    Args:
        X: Expression matrix of shape (n_samples, n_genes)
        gene_names: Array of gene names
        cc_genes: CellCycleGenes container

    Returns:
        Tuple of (S-scores, G2M-scores, QC info dict)
    """
    logger.info("Computing cell-cycle scores...")

    # Map gene names to indices
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Find indices for each gene set
    s_indices = [gene_to_idx[g] for g in cc_genes.s_genes if g in gene_to_idx]
    g2m_indices = [gene_to_idx[g] for g in cc_genes.g2m_genes if g in gene_to_idx]

    logger.info(f"Found {len(s_indices)}/{len(cc_genes.s_genes)} S-phase genes in data")
    logger.info(f"Found {len(g2m_indices)}/{len(cc_genes.g2m_genes)} G2M-phase genes in data")

    if not s_indices or not g2m_indices:
        raise ValueError("Missing required cell-cycle genes in dataset")

    # Compute scores as mean expression of signature genes
    # Handle NaN by using nanmean
    s_scores = np.nanmean(X[:, s_indices], axis=1)
    g2m_scores = np.nanmean(X[:, g2m_indices], axis=1)

    # QC info
    qc_info = {
        "n_s_genes_found": len(s_indices),
        "n_g2m_genes_found": len(g2m_indices),
        "s_score_mean": float(np.nanmean(s_scores)),
        "s_score_std": float(np.nanstd(s_scores)),
        "g2m_score_mean": float(np.nanmean(g2m_scores)),
        "g2m_score_std": float(np.nanstd(g2m_scores)),
        "s_score_range": (float(np.nanmin(s_scores)), float(np.nanmax(s_scores))),
        "g2m_score_range": (float(np.nanmin(g2m_scores)), float(np.nanmax(g2m_scores))),
    }

    logger.info(f"S-score: mean={qc_info['s_score_mean']:.4f}, std={qc_info['s_score_std']:.4f}")
    logger.info(f"G2M-score: mean={qc_info['g2m_score_mean']:.4f}, std={qc_info['g2m_score_std']:.4f}")

    return s_scores, g2m_scores, qc_info


def regress_cell_cycle(
    X: np.ndarray,
    s_scores: np.ndarray,
    g2m_scores: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Regress out cell-cycle signal from expression data.

    Uses linear regression to remove the contribution of cell-cycle scores:
        X_cc_reg = X - beta_s * S_score - beta_g2m * G2M_score

    Args:
        X: Expression matrix of shape (n_samples, n_genes)
        s_scores: S-phase scores for each sample
        g2m_scores: G2M-phase scores for each sample

    Returns:
        Tuple of (regressed expression matrix, QC info dict)
    """
    logger.info("Regressing out cell-cycle signal...")

    n_samples, n_genes = X.shape

    # Build design matrix with S and G2M scores
    # Add intercept term
    covariates = np.column_stack([
        np.ones(n_samples),  # Intercept
        s_scores,
        g2m_scores,
    ])

    # Handle NaN in scores by filling with mean
    covariates = np.nan_to_num(covariates, nan=0.0)

    # Fit linear regression for each gene
    # This is equivalent to: X = covariates @ beta + epsilon
    # We compute residuals: X_cc_reg = X - covariates @ beta_hat

    # Use least squares: beta_hat = (X.T @ X)^-1 @ X.T @ y
    # For efficiency, we use sklearn's LinearRegression
    X_cc_reg = np.zeros_like(X)

    # Track coefficients for QC
    beta_s_all = []
    beta_g2m_all = []

    # Process in batches to avoid memory issues
    batch_size = 1000
    for start in range(0, n_genes, batch_size):
        end = min(start + batch_size, n_genes)
        X_batch = X[:, start:end]

        # Handle NaN in expression data
        X_batch_filled = np.nan_to_num(X_batch, nan=0.0)

        # Fit regression
        reg = LinearRegression(fit_intercept=False)
        reg.fit(covariates, X_batch_filled)

        # Compute residuals
        X_cc_reg[:, start:end] = X_batch_filled - covariates @ reg.coef_.T

        # Store coefficients
        beta_s_all.extend(reg.coef_[:, 1].tolist())
        beta_g2m_all.extend(reg.coef_[:, 2].tolist())

    # QC info
    qc_info = {
        "beta_s_mean": float(np.mean(beta_s_all)),
        "beta_s_std": float(np.std(beta_s_all)),
        "beta_g2m_mean": float(np.mean(beta_g2m_all)),
        "beta_g2m_std": float(np.std(beta_g2m_all)),
        "residual_mean": float(np.nanmean(X_cc_reg)),
        "residual_std": float(np.nanstd(X_cc_reg)),
    }

    logger.info(f"Cell-cycle regression complete")
    logger.info(f"  Beta S coefficient: mean={qc_info['beta_s_mean']:.4f}, std={qc_info['beta_s_std']:.4f}")
    logger.info(f"  Beta G2M coefficient: mean={qc_info['beta_g2m_mean']:.4f}, std={qc_info['beta_g2m_std']:.4f}")
    logger.info(f"  Residual: mean={qc_info['residual_mean']:.4f}, std={qc_info['residual_std']:.4f}")

    return X_cc_reg, qc_info


def compute_svd_residuals(
    X: np.ndarray,
    k: int = SVD_K,
) -> tuple[np.ndarray, dict]:
    """
    Compute residuals after removing top k SVD components.

    This removes low-rank batch effects:
        X_id = X - U_k @ S_k @ V_k.T

    Args:
        X: Expression matrix of shape (n_samples, n_genes)
        k: Number of SVD components to remove

    Returns:
        Tuple of (residual matrix, QC info dict)
    """
    logger.info(f"Computing SVD residuals (k={k})...")

    # Handle NaN by filling with 0 for SVD computation
    X_filled = np.nan_to_num(X, nan=0.0)

    # Compute SVD (using randomized SVD for efficiency)
    from sklearn.utils.extmath import randomized_svd

    U, S, Vt = randomized_svd(
        X_filled,
        n_components=k,
        random_state=RANDOM_STATE,
    )

    # Compute residuals
    X_svd = U @ np.diag(S) @ Vt
    X_residuals = X_filled - X_svd

    # Restore NaN positions
    nan_mask = np.isnan(X)
    X_residuals[nan_mask] = np.nan

    # QC info
    variance_explained = (S**2) / (S**2).sum()
    qc_info = {
        "k": k,
        "singular_values": S.tolist(),
        "variance_explained": variance_explained.tolist(),
        "total_variance_explained": float(variance_explained.sum()),
        "residual_mean": float(np.nanmean(X_residuals)),
        "residual_std": float(np.nanstd(X_residuals)),
    }

    logger.info(f"SVD residuals complete")
    logger.info(f"  Total variance explained by top {k} components: {qc_info['total_variance_explained']:.4f}")
    logger.info(f"  Residual: mean={qc_info['residual_mean']:.4f}, std={qc_info['residual_std']:.4f}")

    return X_residuals, qc_info


def build_knn_graph(
    X: np.ndarray,
    n_neighbors: int = N_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Build k-NN graph using pynndescent.

    Args:
        X: Expression matrix of shape (n_samples, n_genes)
        n_neighbors: Number of neighbors

    Returns:
        Tuple of (neighbor indices, distances, QC info dict)
    """
    logger.info(f"Building k-NN graph (k={n_neighbors})...")

    try:
        from pynndescent import NNDescent
    except ImportError as e:
        raise ImportError(
            "pynndescent is required for k-NN graph construction. "
            "Install with: pip install pynndescent"
        ) from e

    # Handle NaN by filling with 0 for k-NN computation
    X_filled = np.nan_to_num(X, nan=0.0)

    # Build k-NN index
    n_samples = X_filled.shape[0]

    # pynndescent requires n_neighbors < n_samples
    effective_k = min(n_neighbors, n_samples - 1)

    index = NNDescent(
        X_filled,
        n_neighbors=effective_k + 1,  # +1 because query includes self
        metric="euclidean",
        random_state=RANDOM_STATE,
    )

    # Query for neighbors
    neighbor_indices, neighbor_distances = index.query(X_filled, k=effective_k + 1)

    # Remove self from neighbors (first column)
    neighbor_indices = neighbor_indices[:, 1:]
    neighbor_distances = neighbor_distances[:, 1:]

    # QC info
    qc_info = {
        "n_neighbors": effective_k,
        "n_samples": n_samples,
        "mean_distance": float(np.mean(neighbor_distances)),
        "std_distance": float(np.std(neighbor_distances)),
        "median_distance": float(np.median(neighbor_distances)),
    }

    logger.info(f"k-NN graph complete")
    logger.info(f"  Mean neighbor distance: {qc_info['mean_distance']:.4f}")
    logger.info(f"  Median neighbor distance: {qc_info['median_distance']:.4f}")

    return neighbor_indices, neighbor_distances, qc_info


def save_outputs(
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    s_scores: np.ndarray,
    g2m_scores: np.ndarray,
    sample_ids: np.ndarray,
) -> None:
    """
    Save manifold outputs to parquet files.

    Args:
        neighbor_indices: k-NN indices (n_samples x n_neighbors)
        neighbor_distances: k-NN distances (n_samples x n_neighbors)
        s_scores: S-phase scores
        g2m_scores: G2M-phase scores
        sample_ids: Sample identifiers
    """
    logger.info("Saving outputs...")

    # Save neighbor data
    n_samples, n_neighbors = neighbor_indices.shape

    # Convert distances to weights using Gaussian kernel
    # w_ij = exp(-d_ij^2 / (2 * sigma^2)), then row-normalize
    sigma = np.median(neighbor_distances)  # Use median distance as bandwidth
    neighbor_weights = np.exp(-neighbor_distances**2 / (2 * sigma**2 + 1e-8))
    # Row-normalize
    row_sums = neighbor_weights.sum(axis=1, keepdims=True)
    neighbor_weights = neighbor_weights / (row_sums + 1e-8)

    # Create DataFrame with LIST columns for compatibility with training script
    neighbor_df = pd.DataFrame({
        "sample_idx": np.arange(n_samples),
        "neighbor_indices": [row.tolist() for row in neighbor_indices],
        "neighbor_weights": [row.tolist() for row in neighbor_weights],
        "neighbor_distances": [row.tolist() for row in neighbor_distances],
    })

    # Save to parquet
    neighbor_df.to_parquet(OUTPUT_NEIGHBORS_PATH, index=False)
    logger.info(f"Saved neighbor data to {OUTPUT_NEIGHBORS_PATH}")
    logger.info(f"  Shape: {neighbor_df.shape}")

    # Save cell-cycle scores
    scores_df = pd.DataFrame({
        "sample_idx": np.arange(n_samples),
        "unique_sample_id": sample_ids,
        "S_score": s_scores,
        "G2M_score": g2m_scores,
    })

    scores_df.to_parquet(OUTPUT_SCORES_PATH, index=False)
    logger.info(f"Saved cell-cycle scores to {OUTPUT_SCORES_PATH}")
    logger.info(f"  Shape: {scores_df.shape}")


def main() -> int:
    """
    Main entry point for manifold preprocessing.
    """
    parser = argparse.ArgumentParser(
        description="Build manifold preprocessing for NE-GATE training"
    )
    parser.add_argument(
        "--use-svd",
        action="store_true",
        help="Apply SVD batch correction (remove top k components)",
    )
    parser.add_argument(
        "--svd-k",
        type=int,
        default=SVD_K,
        help=f"Number of SVD components to remove (default: {SVD_K})",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=N_NEIGHBORS,
        help=f"Number of k-NN neighbors (default: {N_NEIGHBORS})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to per-sample z-scored expression AnnData (.h5ad)",
    )
    parser.add_argument(
        "--cellcycle-genes",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "seurat_cellcycle_genes.txt",
        help="Path to Seurat cell-cycle gene list",
    )
    parser.add_argument(
        "--output-neighbors",
        type=Path,
        default=Path("data/manifold_neighbors.parquet"),
        help="Output path for k-NN neighbours (.parquet)",
    )
    parser.add_argument(
        "--output-scores",
        type=Path,
        default=Path("data/cellcycle_scores.parquet"),
        help="Output path for cell-cycle scores (.parquet)",
    )
    args = parser.parse_args()

    # Override module-level paths with CLI values
    INPUT_PATH = args.input
    CELLCYCLE_GENES_PATH = args.cellcycle_genes
    OUTPUT_NEIGHBORS_PATH = args.output_neighbors
    OUTPUT_SCORES_PATH = args.output_scores
    OUTPUT_NEIGHBORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Manifold Preprocessing for NE-GATE")
    logger.info("=" * 60)
    logger.info(f"Input: {INPUT_PATH}")
    logger.info(f"Use SVD: {args.use_svd}")
    if args.use_svd:
        logger.info(f"SVD k: {args.svd_k}")
    logger.info(f"k-NN neighbors: {args.n_neighbors}")
    logger.info("=" * 60)

    # Check input file
    if not INPUT_PATH.exists():
        logger.error(f"Input file not found: {INPUT_PATH}")
        return 1

    if not CELLCYCLE_GENES_PATH.exists():
        logger.error(f"Cell-cycle gene file not found: {CELLCYCLE_GENES_PATH}")
        return 1

    # Load input AnnData
    logger.info(f"Loading input AnnData from {INPUT_PATH}")
    adata = anndata.read_h5ad(INPUT_PATH)
    logger.info(f"Loaded AnnData with shape {adata.shape}")

    # Get expression matrix
    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    logger.info(f"Expression matrix type: {type(X)}, dtype: {X.dtype}")

    # Get gene names and sample IDs (with fallback to index)
    gene_names = adata.var_names.values
    sample_ids = adata.obs["unique_sample_id"].values if "unique_sample_id" in adata.obs.columns else adata.obs.index.values

    # Parse cell-cycle genes
    cc_genes = parse_cellcycle_genes(CELLCYCLE_GENES_PATH)

    # Step 1: Compute cell-cycle scores
    logger.info("-" * 40)
    logger.info("Step 1: Computing cell-cycle scores")
    logger.info("-" * 40)
    s_scores, g2m_scores, cc_qc = compute_cellcycle_scores(X, gene_names, cc_genes)

    # Step 2: Regress out cell-cycle signal
    logger.info("-" * 40)
    logger.info("Step 2: Regressing cell-cycle signal")
    logger.info("-" * 40)
    X_cc_reg, reg_qc = regress_cell_cycle(X, s_scores, g2m_scores)

    # Step 3: Optional SVD residuals
    X_id = X_cc_reg
    svd_qc = None
    if args.use_svd:
        logger.info("-" * 40)
        logger.info("Step 3: Computing SVD residuals")
        logger.info("-" * 40)
        X_id, svd_qc = compute_svd_residuals(X_cc_reg, k=args.svd_k)

    # Step 4: Build k-NN graph
    logger.info("-" * 40)
    logger.info("Step 4: Building k-NN graph")
    logger.info("-" * 40)
    neighbor_indices, neighbor_distances, knn_qc = build_knn_graph(
        X_id, n_neighbors=args.n_neighbors
    )

    # Step 5: Save outputs
    logger.info("-" * 40)
    logger.info("Step 5: Saving outputs")
    logger.info("-" * 40)
    save_outputs(
        neighbor_indices,
        neighbor_distances,
        s_scores,
        g2m_scores,
        sample_ids,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Input samples: {adata.n_obs}")
    logger.info(f"Input genes: {adata.n_vars}")
    logger.info(f"S-phase genes used: {cc_qc['n_s_genes_found']}")
    logger.info(f"G2M-phase genes used: {cc_qc['n_g2m_genes_found']}")
    logger.info(f"SVD applied: {args.use_svd}")
    if svd_qc:
        logger.info(f"SVD variance explained: {svd_qc['total_variance_explained']:.4f}")
    logger.info(f"k-NN neighbors: {knn_qc['n_neighbors']}")
    logger.info(f"Mean neighbor distance: {knn_qc['mean_distance']:.4f}")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  Neighbors: {OUTPUT_NEIGHBORS_PATH}")
    logger.info(f"  Cell-cycle scores: {OUTPUT_SCORES_PATH}")
    logger.info("")
    logger.info("Manifold preprocessing complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
