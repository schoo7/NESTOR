#!/usr/bin/env python3
"""
identity_erosion.py

Compute GMM-based probabilistic erosion index measuring loss of tissue identity.

The erosion index uses a Gaussian Mixture Model to measure how far samples
have drifted from the NE-low "Rim". Samples that have lost their tissue-specific
gene expression pattern (e.g., through neuroendocrine transdifferentiation)
will have high erosion scores.

Algorithm:
1. Load per-sample z-scored data
2. Identify NE-low samples (bottom 50% by seed gene expression)
3. Fit GMM on NE-low samples (the Rim)
4. For all samples, compute log-likelihood under the GMM
5. Convert to erosion index [0, 1]:
   - High probability of belonging to NE-low = 0.0 erosion
   - Low probability (drifted away) = 1.0 erosion

Output files:
- outputs/erosion_indices.parquet: sample_id, erosion_index
- outputs/erosion_diagnostics.parquet: global statistics

Usage:
    python scripts/identity_erosion.py
"""

import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Configure logging
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "identity_erosion.log")
    ]
)
logger = logging.getLogger(__name__)

# Project paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = Path("outputs")
CONFIG_DIR = PROJECT_ROOT / "config"

def compute_probabilistic_erosion(
    X: np.ndarray,
    is_ne_low: pd.Series,
    n_components: int = 50,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute probabilistic erosion index using distance from NE-low centroid.

    This method uses a robust, simple approach that measures how far samples
    have drifted from the NE-low "Rim" using PCA + standardized distance.

    Algorithm:
    1. Reduce dimensionality with PCA (50 components)
    2. Compute centroid of NE-low samples in PCA space
    3. Compute standardized Euclidean distance (normalized by NE-low std)
    4. Convert to erosion index [0, 1]

    Args:
        X: Expression matrix (n_samples, n_genes)
        is_ne_low: Boolean Series indicating NE-low samples
        n_components: Number of PCA components (default 50)
        random_state: Random seed for reproducibility

    Returns:
        Array of erosion indices in [0, 1]
    """
    from sklearn.decomposition import PCA

    # Handle sparse matrices
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    # Handle NaN values by imputing with column means
    if np.any(np.isnan(X)):
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        X = X.copy()
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Get NE-low samples
    X_ne_low = X[is_ne_low.values]

    logger.info(f"Computing erosion using PCA + centroid distance on {X_ne_low.shape[0]} NE-low samples")

    # Step 1: Reduce dimensionality with PCA
    n_pca_components = min(n_components, 50, X.shape[1] - 1, X_ne_low.shape[0] - 1)
    pca = PCA(n_components=n_pca_components, random_state=random_state)
    pca.fit(X_ne_low)  # Fit only on NE-low samples for better Rim representation

    X_pca = pca.transform(X)
    X_ne_low_pca = X_pca[is_ne_low.values]

    logger.info(f"PCA explained variance (Rim-only): {pca.explained_variance_ratio_.sum():.3f}")

    # Step 2: Compute centroid of NE-low samples
    centroid = X_ne_low_pca.mean(axis=0)

    # Step 3: Compute standardized Euclidean distance
    # Compute standard deviation per dimension for NE-low samples
    std_per_dim = X_ne_low_pca.std(axis=0)
    std_per_dim = np.maximum(std_per_dim, 1e-8)  # Avoid division by zero

    # Standardize distances by dividing by std
    diff = X_pca - centroid
    distances = np.sqrt(np.sum((diff / std_per_dim) ** 2, axis=1))

    # Step 4: Convert to erosion index [0, 1]
    dist_min, dist_max = distances.min(), distances.max()
    erosion_index = (distances - dist_min) / (dist_max - dist_min + 1e-8)

    logger.info(f"Erosion index range: [{erosion_index.min():.4f}, {erosion_index.max():.4f}]")

    return erosion_index


def load_seed_genes(config_dir: Path) -> list[str]:
    """
    Load NE seed genes from config file.

    Args:
        config_dir: Path to config directory

    Returns:
        List of seed gene symbols
    """
    seed_file = config_dir / "ne_seed_genes.txt"
    if not seed_file.exists():
        raise FileNotFoundError(f"Seed genes file not found: {seed_file}")

    with open(seed_file) as f:
        genes = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(genes)} seed genes from {seed_file}")
    return genes


def compute_seed_gene_expression(
    adata: ad.AnnData,
    seed_genes: list[str]
) -> np.ndarray:
    """
    Compute mean expression of seed genes per sample.

    Args:
        adata: AnnData with expression data
        seed_genes: List of NE seed gene symbols

    Returns:
        Array of mean seed gene expression per sample
    """
    # Find available seed genes
    available_genes = [g for g in seed_genes if g in adata.var_names]
    if not available_genes:
        raise ValueError("No seed genes found in data")

    logger.info(f"Found {len(available_genes)}/{len(seed_genes)} seed genes in data")

    # Get expression matrix
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X)

    # Get column indices for seed genes
    gene_indices = [list(adata.var_names).index(g) for g in available_genes]

    # Compute mean expression (handle NaN)
    seed_expr = X[:, gene_indices]
    seed_mean = np.nanmean(seed_expr, axis=1)

    logger.info(f"Seed gene mean expression: [{np.nanmin(seed_mean):.4f}, {np.nanmax(seed_mean):.4f}]")

    return seed_mean


def identify_ne_low_samples(
    adata: ad.AnnData,
    seed_genes: list[str]
) -> pd.Series:
    """
    Identify NE-low samples (bottom 50% by seed gene expression).

    Args:
        adata: AnnData with expression data
        seed_genes: List of NE seed gene symbols

    Returns:
        Boolean Series indicating NE-low samples
    """
    seed_mean = compute_seed_gene_expression(adata, seed_genes)

    # Use median as threshold
    median_threshold = np.nanmedian(seed_mean)
    is_ne_low = seed_mean < median_threshold

    logger.info(f"NE-low threshold (median): {median_threshold:.4f}")
    logger.info(f"NE-low samples: {is_ne_low.sum()}/{len(is_ne_low)}")

    return pd.Series(is_ne_low, index=adata.obs.index)


def main() -> int:
    """Main entry point for GMM-based identity erosion analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute GMM-based identity erosion indices for NESTOR training"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to per-sample z-scored expression AnnData (.h5ad)",
    )
    parser.add_argument(
        "--seed-genes",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "ne_seed_genes.txt",
        help="Path to NE seed-gene list",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for erosion indices and diagnostics",
    )
    args = parser.parse_args()

    logger.info("Starting GMM-based identity erosion analysis")

    # Load data
    input_path = args.input
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    logger.info(f"Loading data from {input_path}")
    adata = ad.read_h5ad(input_path)
    logger.info(f"Loaded {adata.n_obs} samples, {adata.n_vars} genes")

    # Load seed genes
    seed_genes = load_seed_genes(args.seed_genes.parent)

    # Identify NE-low samples
    is_ne_low = identify_ne_low_samples(adata, seed_genes)

    # Get expression matrix
    X = adata.X

    # Compute GMM-based probabilistic erosion
    logger.info("Computing GMM-based probabilistic erosion...")
    erosion_index = compute_probabilistic_erosion(X, is_ne_low, n_components=15)

    # Build output DataFrame (no cancer_type dependency)
    erosion_df = pd.DataFrame({
        'sample_id': adata.obs['unique_sample_id'].values
                     if 'unique_sample_id' in adata.obs.columns
                     else adata.obs.index.values,
        'erosion_index': erosion_index
    })

    # Compute diagnostics (simplified - no cancer_type grouping)
    seed_mean = compute_seed_gene_expression(adata, seed_genes)
    diagnostics_df = pd.DataFrame({
        'metric': [
            'n_samples',
            'n_ne_low',
            'erosion_mean',
            'erosion_std',
            'seed_erosion_correlation'
        ],
        'value': [
            len(erosion_index),
            int(is_ne_low.sum()),
            float(erosion_index.mean()),
            float(erosion_index.std()),
            float(np.corrcoef(seed_mean, erosion_index)[0, 1])
        ]
    })

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save outputs
    erosion_path = args.output_dir / "erosion_indices.parquet"
    erosion_df.to_parquet(erosion_path, index=False)
    logger.info(f"Saved GMM erosion indices to {erosion_path}")

    diagnostics_path = args.output_dir / "erosion_diagnostics.parquet"
    diagnostics_df.to_parquet(diagnostics_path, index=False)
    logger.info(f"Saved erosion diagnostics to {diagnostics_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("GMM-Based Identity Erosion Analysis Summary")
    logger.info("=" * 60)
    logger.info(f"Total samples: {adata.n_obs}")
    logger.info(f"NE-low samples (Rim): {is_ne_low.sum()}")
    logger.info(f"Erosion mean: {erosion_index.mean():.4f}")
    logger.info(f"Seed-erosion correlation: {diagnostics_df[diagnostics_df['metric'] == 'seed_erosion_correlation']['value'].values[0]:.4f}")
    logger.info("=" * 60)
    logger.info("GMM-based erosion analysis complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
