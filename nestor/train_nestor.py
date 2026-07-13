#!/usr/bin/env python3
"""
train_nestor.py

Train NESTOR (NEuroendocrine Differentiation STate ORacle) model.

NESTOR is a deep learning model that quantifies neuroendocrine (NE) transdifferentiation
in cancer samples using pan-cancer gene expression data. The model discovers lineage
programs (ASCL1-driven, NEUROD1-driven, Amphicrine states) and measures progression
from non-NE to NE phenotype.

This script implements a tunnel-gated variational autoencoder with:
- Hard-Concrete (L0) gates for sparse gene selection
- Dedicated z_progression latent node as the NE differentiation axis [0,1]
- Manifold regularization for identity continuity
- Jacobian regularization for smooth gene decay
- Proliferation decorrelation to separate growth from identity
- Adaptive tunnel width for transitional samples

Architecture:
    Input x (genes) -> Gate m -> Gated input x*m -> Encoder -> z (with z_progression) -> Decoder -> x_hat

Loss components:
    - Reconstruction loss (MSE)
    - KL divergence for VAE regularization
    - Seed guidance correlation (decayed over epochs)
    - Sparse gate penalty (L0)
    - Manifold smoothness loss
    - Jacobian regularization
    - Proliferation decorrelation loss

Usage:
    python nestor/train_nestor.py --input <zscored.h5ad> --seed-genes <seed.txt> --output-dir <dir>

Output:
    - models/nestor/model.pt (trained model)
    - models/nestor/config.json (hyperparameters)
    - models/nestor/training_log.csv (per-epoch metrics)
    - outputs/nestor_scores.parquet (per-sample NE differentiation scores)
"""

import copy
import json
import logging
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Set CUBLAS workspace config for CUDA reproducibility (must be before torch operations)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Reproducibility seed
SEED = 42

# Configure logging
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = Path.cwd() / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Get the best available device: MPS (Mac), CUDA (NVIDIA), or CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# Generate timestamped log filename
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOGS_DIR / f"train_nestor_{log_timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)


def set_reproducibility(seed: int = 42):
    """
    Set all random seeds and deterministic modes for full reproducibility.

    This ensures identical results across multiple training runs with the same seed.
    Must be called at the start of training before any random operations.

    Args:
        seed: Random seed value (default: 42)
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # PyTorch global determinism (warn_only for operations without deterministic implementations)
    torch.use_deterministic_algorithms(True, warn_only=True)

    logger.info(f"Reproducibility configured with seed={seed}")
    logger.info(f"  Python random, NumPy, PyTorch (CPU/MPS/CUDA) seeds set")
    logger.info(f"  CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
    logger.info(f"  torch.use_deterministic_algorithms=True (warn_only=True)")
    return seed

# Default project paths (overridable via CLI arguments)
DATA_DIR = Path.cwd() / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = Path.cwd() / "models"
OUTPUTS_DIR = Path.cwd() / "outputs"

# Default hyperparameters
# NESTOR training parameters
DEFAULT_HIDDEN_DIM = 512  # Keep V2: Optimal for this dataset size
DEFAULT_LATENT_DIM = 16  # Total latent dim (4+4+8)
DEFAULT_LEARNING_RATE = 0.001  # Standard learning rate
DEFAULT_BATCH_SIZE = 2048
DEFAULT_N_EPOCHS = 500  # INCREASED: Compensate for larger batch size (fewer updates/epoch)
DEFAULT_SEED_GUIDANCE_WEIGHT = 1.0
DEFAULT_KL_WEIGHT = 0.5  # Keep V2: Optimal balance
DEFAULT_GATE_PENALTY_WEIGHT = 20.0  # INCREASED: Stronger pull for sparsity
DEFAULT_EARLY_STOPPING_PATIENCE = 500
DEFAULT_GATE_TEMPERATURE = 0.5  # Higher temperature keeps gates softer
DEFAULT_GATE_INIT_MEAN = 0.3  # Start with ~30% genes active, allow pruning during training
DEFAULT_SPARSITY_TARGET = 0.1  # Target 10% of genes active (~2000 genes)

# Latent space dimensions
DEFAULT_LATENT_STATE_DIM = 4  # Sufficient for NE master regulators
DEFAULT_LATENT_POTENTIAL_DIM = 4  # Sufficient for transition states
DEFAULT_LATENT_OTHER_DIM = 8  # Keep V2: Optimal for this heterogeneity level

# Pathway gating parameters
DEFAULT_LATENT_PATHWAY_DIM_MAX = 16  # Maximum pathway dimensions (will be pruned to D*)
DEFAULT_PATHWAY_GATE_TEMPERATURE = 0.3  # Lower temperature for sharper pathway gates
DEFAULT_PATHWAY_GATE_INIT_MEAN = 0.3  # Start with ~30% pathways active (~5 dimensions)
DEFAULT_PATHWAY_GATE_PENALTY = 0.1  # L1 penalty on pathway gates
DEFAULT_DECODER_ORTHO_WEIGHT = 0.3  # Weight for decoder orthogonality loss
DEFAULT_RARE_STATE_WEIGHT = 0.2  # Weight for rare state protection

# Regularization parameters
DEFAULT_KL_ANNEAL_EPOCHS = 50  # KL annealing period
DEFAULT_FREE_BITS = 0.1  # REDUCED: Force model to use z_other if needed
DEFAULT_DIVERSITY_WEIGHT = 0.1  # Weight for diversity loss

# IMPROVE: Learning rate warmup for training stability
DEFAULT_WARMUP_EPOCHS = 5  # Learning rate warmup period
DEFAULT_WARMUP_LR_FACTOR = 0.1  # Start at 10% of target learning rate


def get_learning_rate_with_warmup(epoch: int, base_lr: float) -> float:
    """
    IMPROVE: Compute learning rate with linear warmup.

    Prevents KL explosion at training start by gradually increasing
    learning rate over the warmup period.

    Args:
        epoch: Current epoch (0-indexed)
        base_lr: Target learning rate after warmup

    Returns:
        Current learning rate
    """
    if epoch < DEFAULT_WARMUP_EPOCHS:
        progress = (epoch + 1) / DEFAULT_WARMUP_EPOCHS
        return base_lr * (DEFAULT_WARMUP_LR_FACTOR + (1 - DEFAULT_WARMUP_LR_FACTOR) * progress)
    return base_lr


def load_seed_genes(config_path: Path) -> list[str]:
    """
    Load NE seed genes from config file.

    Args:
        config_path: Path to the seed genes text file

    Returns:
        List of gene symbols (uppercase)

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Seed genes config file not found: {config_path}")

    genes = []
    with open(config_path) as f:
        for line in f:
            gene = line.strip().upper()
            if gene:  # Skip empty lines
                genes.append(gene)

    logger.info(f"Loaded {len(genes)} seed genes from {config_path}")
    return genes


class HardConcreteGate(nn.Module):
    """
    Hard-Concrete gate for L0 regularization.

    Implements the Hard-Concrete distribution from:
    "Learning Sparse Neural Networks through L0 regularization" (Louizos et al., 2018)

    The gate samples from a stretched concrete distribution and hardens to exactly 0 or 1.
    """

    def __init__(
        self,
        n_features: int,
        temperature: float = 0.5,
        stretch_limits: tuple[float, float] = (-0.1, 1.1),
        init_mean: float = 0.5
    ):
        """
        Initialize Hard-Concrete gate.

        Args:
            n_features: Number of input features (genes)
            temperature: Temperature for concrete distribution
            stretch_limits: (lower, upper) limits for stretching
            init_mean: Initial mean for gate probabilities
        """
        super().__init__()

        self.n_features = n_features
        self.temperature = temperature
        self.lower, self.upper = stretch_limits

        # Initialize log_alpha to achieve desired initial mean
        # Account for the stretch_limits offset in get_probs()
        log_ratio = math.log(-self.lower / self.upper)
        init_log_alpha = math.log(init_mean / (1 - init_mean + 1e-8)) + log_ratio
        self.log_alpha = nn.Parameter(torch.full((n_features,), init_log_alpha))

    def forward(self, temperature: float | None = None) -> torch.Tensor:
        """
        Sample gate mask.

        Args:
            temperature: Optional temperature override

        Returns:
            Gate mask tensor of shape (n_features,)

        Raises:
            ValueError: If temperature is not positive
        """
        if temperature is None:
            temperature = self.temperature

        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        if self.training:
            # Sample from concrete distribution
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid(
                (torch.log(u) - torch.log(1 - u) + self.log_alpha) / temperature
            )
            # Stretch to [lower, upper]
            s = s * (self.upper - self.lower) + self.lower
            # Harden to [0, 1]
            mask = F.hardtanh(s, min_val=0.0, max_val=1.0)
        else:
            # Deterministic in eval mode
            mask = self.get_gates()

        return mask

    def get_probs(self) -> torch.Tensor:
        """
        Get the continuous gate probabilities (P(gate > 0)).
        """
        log_ratio = math.log(-self.lower / self.upper)
        return torch.sigmoid(self.log_alpha - log_ratio)

    def get_gates(self, k: int | None = None) -> torch.Tensor:
        """
        Get deterministic gate mask for evaluation.

        Args:
            k: If provided, apply Top-K hard selection to return exactly K active gates.
               If None, return continuous probabilities from get_probs().

        Returns:
            Deterministic mask values in [0, 1]
        """
        probs = self.get_probs()

        if k is not None and k < self.n_features:
            # Top-K hard selection: select exactly K genes with highest probability
            # Find threshold for K-th highest value
            sorted_probs, indices = torch.sort(probs, descending=True)
            threshold = sorted_probs[k - 1]
            return (probs >= threshold).float()

        return probs

    def l0_penalty(self, normalize: bool = True, target_sparsity: float = 0.1) -> torch.Tensor:
        """
        Compute L0 regularization penalty.

        Uses L1 regularization on gate probabilities to encourage sparsity,
        with a minimum activity constraint to prevent collapse.

        Args:
            normalize: If True, return normalized penalty
            target_sparsity: Target fraction of active gates (default 0.1 = 10%)

        Returns:
            L0 penalty (scalar)
        """
        probs = self.get_probs()
        active_fraction = probs.mean()

        # L1 penalty on gate probabilities (encourages sparsity)
        # This directly penalizes high probabilities
        l1_penalty = probs.mean()

        # Collapse prevention: heavy penalty if too few gates active
        min_active = 0.05  # Minimum 5% of genes must stay active
        if active_fraction < min_active:
            collapse_penalty = 100.0 * (min_active - active_fraction) ** 2
        else:
            collapse_penalty = 0.0

        return l1_penalty + collapse_penalty


class AdaptiveWidthMLP(nn.Module):
    """
    MLP to predict sample-specific tunnel width tau_i.

    Allows the tunnel to widen for transitional samples so they can
    reconstruct both AdPCa and NEPC programs without forcing a hard choice.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        tau_min: float = 0.1,
        tau_max: float = 2.0
    ):
        """
        Initialize adaptive width MLP.

        Args:
            input_dim: Dimension of input features (pathway encoder hidden dim)
            hidden_dim: Hidden layer dimension
            tau_min: Minimum tunnel width
            tau_max: Maximum tunnel width
        """
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict adaptive width tau_i for each sample.

        Args:
            h: (batch_size, input_dim) pathway encoder hidden state

        Returns:
            (batch_size, 1) adaptive widths clamped to [tau_min, tau_max]
        """
        normalized = self.mlp(h)
        return self.tau_min + normalized * (self.tau_max - self.tau_min)


class StratifiedBatchSampler:
    """
    Batch sampler that ensures each batch contains samples from multiple studies.

    This prevents study-specific learning by ensuring the model sees samples
    from diverse studies in each training batch. This was identified as the
    critical fix for improving generalization to held-out studies (Phase 3).

    Key insight from ablation experiments:
    - A3 (Study Stratification) achieved 0.79 correlation vs SOTA's 0.11
    - Root cause: random sampling allows batches dominated by single studies
    - Solution: stratified sampling ensures batch diversity across studies
    """

    def __init__(
        self,
        study_ids: np.ndarray,
        batch_size: int,
        studies_per_batch: int = 5,
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Initialize stratified batch sampler.

        Args:
            study_ids: Array of study IDs for each sample
            batch_size: Total batch size
            studies_per_batch: Minimum number of studies per batch
            shuffle: Whether to shuffle studies
            seed: Random seed for reproducibility
        """
        self.study_ids = study_ids
        self.batch_size = batch_size
        self.studies_per_batch = studies_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Group indices by study
        self.study_to_indices = {}
        for idx, study in enumerate(study_ids):
            if study not in self.study_to_indices:
                self.study_to_indices[study] = []
            self.study_to_indices[study].append(idx)

        self.studies = list(self.study_to_indices.keys())
        self.n_samples = len(study_ids)

        logger.info(f"StratifiedBatchSampler initialized:")
        logger.info(f"  Total samples: {self.n_samples}")
        logger.info(f"  Total studies: {len(self.studies)}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Studies per batch: {studies_per_batch}")
        logger.info(f"  Samples per study per batch: ~{batch_size // studies_per_batch}")

    def __iter__(self):
        """Generate batches with stratified sampling."""
        # Shuffle studies order
        studies = self.studies.copy()
        if self.shuffle:
            self.rng.shuffle(studies)

        # Shuffle indices within each study
        study_indices_shuffled = {}
        for study in studies:
            indices = self.study_to_indices[study].copy()
            if self.shuffle:
                self.rng.shuffle(indices)
            study_indices_shuffled[study] = indices

        # Track current position in each study's indices
        study_positions = {study: 0 for study in studies}

        # Calculate samples per study per batch
        samples_per_study = self.batch_size // self.studies_per_batch
        extra_samples = self.batch_size % self.studies_per_batch

        batches_generated = 0
        total_yielded = 0

        # Continue until all samples are used
        while total_yielded < self.n_samples:
            batch = []
            studies_in_batch = []

            # Select studies for this batch (cycle through all studies)
            start_study_idx = (batches_generated * self.studies_per_batch) % len(studies)

            for i in range(self.studies_per_batch):
                study_idx = (start_study_idx + i) % len(studies)
                study = studies[study_idx]

                # Get available indices for this study
                available = study_indices_shuffled[study]
                pos = study_positions[study]

                if pos < len(available):
                    # Take samples_per_study samples from this study
                    end_pos = min(pos + samples_per_study, len(available))
                    batch.extend(available[pos:end_pos])
                    study_positions[study] = end_pos
                    studies_in_batch.append(study)

            # Add extra samples if batch is not full
            if len(batch) < self.batch_size:
                remaining = self.batch_size - len(batch)
                for study in studies:
                    if remaining <= 0:
                        break
                    pos = study_positions[study]
                    available = study_indices_shuffled[study]
                    if pos < len(available):
                        take = min(remaining, len(available) - pos)
                        batch.extend(available[pos:pos + take])
                        study_positions[study] += take
                        remaining -= take

            # Yield batch if not empty
            if batch:
                # Truncate to batch_size if we have more
                batch = batch[:self.batch_size]
                yield batch
                total_yielded += len(batch)
                batches_generated += 1

    def __len__(self):
        """Return number of batches."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size


class ManifoldLoss(nn.Module):
    """
    Manifold smoothness loss for identity continuity.

    Vectorized implementation using a pre-computed sparse Laplacian matrix.
    L_manifold = lambda * z^T L z (where z is sparse vector of batch scores)
    """

    def __init__(self, laplacian: torch.Tensor, weight: float = 1.0):
        """
        Initialize manifold loss.

        Args:
            laplacian: Pre-computed (n_samples, n_samples) sparse COO Laplacian
            weight: Loss weight
        """
        super().__init__()
        self.register_buffer('L', laplacian)
        self.weight = weight

    def forward(self, z_progression: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute manifold loss for a batch using vectorized sparse operations.

        Args:
            z_progression: (batch_size, 1) progression scores
            batch_indices: (batch_size,) original sample indices

        Returns:
            Scalar manifold loss
        """
        n_samples = self.L.shape[0]
        batch_size = z_progression.shape[0]
        device = z_progression.device

        # Create sparse vector of batch scores in global coordinate space
        # v = zeros(n_samples, 1), v[batch_indices] = z_progression
        v = torch.zeros((n_samples, 1), device=device)
        v[batch_indices] = z_progression

        # Compute L @ v using sparse-dense multiplication
        # Lv: (n_samples, 1)
        Lv = torch.sparse.mm(self.L, v)

        # Compute v^T @ Lv (inner product)
        # Since v is zero outside batch_indices, we only sum those elements
        loss = (v[batch_indices] * Lv[batch_indices]).sum()

        # Normalize by batch size to keep loss scale stable
        return (loss / batch_size) * self.weight


class JacobianRegularization(nn.Module):
    """
    Jacobian norm regularization for decoder smoothness.

    Uses Hutchinson trace estimator with Rademacher probes to compute ||J_decoder||_F^2
    without explicit Jacobian computation.

    This enforces smooth gene decay curves along z_progression.
    """

    def __init__(self, n_probes: int = 2, weight: float = 0.1):
        """
        Initialize Jacobian regularization.

        Args:
            n_probes: Number of Rademacher probe vectors (1-2 recommended for efficiency)
            weight: Loss weight
        """
        super().__init__()
        self.n_probes = max(1, min(n_probes, 2))  # Clamp to 1-2 for efficiency
        self.weight = weight

    def forward(self, decoder_output: torch.Tensor, z_progression: torch.Tensor, gene_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute Jacobian regularization using Hutchinson estimator with Rademacher probes.

        Args:
            decoder_output: (batch_size, n_genes) decoder output
            z_progression: (batch_size, 1) progression scores (requires grad)
            gene_mask: Optional (n_genes,) mask to restrict to gate-active genes

        Returns:
            Scalar Jacobian regularization loss
        """
        batch_size = decoder_output.shape[0]

        # Ensure z_progression requires grad
        if not z_progression.requires_grad:
            z_progression = z_progression.detach().requires_grad_(True)

        # Apply gene mask if provided (restrict to gate-active genes)
        if gene_mask is not None:
            decoder_output = decoder_output * gene_mask.unsqueeze(0)

        # Sample Rademacher probes (+1 or -1 with equal probability)
        # This is more efficient than standard normal probes
        probes = torch.randint(0, 2, (batch_size, self.n_probes), device=decoder_output.device).float() * 2 - 1

        jvp_sum = torch.tensor(0.0, device=decoder_output.device)
        for i in range(self.n_probes):
            v = probes[:, i:i+1]  # (batch_size, 1)
            # Compute J^T @ v via backprop
            # J = d(decoder)/d(z_progression), shape (n_genes, 1)
            # We want ||J||_F^2 = trace(J^T @ J)
            grad_output = (decoder_output * v).sum()
            grad = torch.autograd.grad(
                grad_output,
                z_progression,
                retain_graph=True,
                create_graph=True
            )[0]
            jvp_sum = jvp_sum + (grad ** 2).sum()

        return (jvp_sum / self.n_probes) * self.weight


class NESTORModel(nn.Module):
    """
    NESTOR (NEuroendocrine Differentiation STate ORacle) Model.

    Lineage encoder with decoupled pathways, per-sample L0 gating, and VAE components.

    This model implements a tunnel-gated architecture for neuroendocrine differentiation analysis:
    - z_progression: 1D scalar [0,1] measuring NE progression (Rim to Center)
    - z_pathway: 16D pathway coordinates with per-sample L0 gating (learns optimal D*)
    - z_other: 8D VAE-style latent for reconstruction and biological variation

    Key features:
    - HardConcrete gate for sparse gene selection (~400 genes)
    - Per-sample pathway gating for automatic dimensionality discovery
    - Tunnel gate decoder: pathway contribution scaled by z_progression
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 512,
        max_pathways: int = 16,
        latent_other_dim: int = 8,
        gate_temperature: float = DEFAULT_GATE_TEMPERATURE,
        gate_init_mean: float = DEFAULT_GATE_INIT_MEAN
    ):
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.max_pathways = max_pathways
        self.latent_other_dim = latent_other_dim
        self._pathway_gate_probs = None  # For compatibility with training functions

        # 1. Input Gate (For Progression Only - isolates core genes)
        self.gene_gate = HardConcreteGate(
            n_genes,
            temperature=gate_temperature,
            init_mean=gate_init_mean
        )

        # 0. Input Normalization (The Shield)
        # Input is already z-scored; clamp outliers instead of learned normalization
        self.input_norm = None

        # 2. Progression Encoder (The Compass)
        self.prog_fc1 = nn.Linear(n_genes, hidden_dim)
        self.prog_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.prog_norm = nn.LayerNorm(hidden_dim)
        self.fc_progression = nn.Linear(hidden_dim, 1)

        # 3. Pathway Encoder (The Explorer - sees all genes)
        self.path_fc1 = nn.Linear(n_genes, hidden_dim)
        self.path_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_pathway = nn.Linear(hidden_dim, max_pathways)

        # PER-SAMPLE GATE LOGITS
        self.fc_pathway_gate_logits = nn.Linear(hidden_dim, max_pathways)

        # Adaptive Width MLP for per-sample tunnel width
        self.adaptive_width = AdaptiveWidthMLP(hidden_dim)

        # 4. VAE Components for z_other
        self.fc_other_mu = nn.Linear(hidden_dim, latent_other_dim)
        self.fc_other_logvar = nn.Linear(hidden_dim, latent_other_dim)

        # 5. Decoders
        self.decoder_pathway = nn.Linear(max_pathways, n_genes, bias=True)
        
        # Initialize pathway decoder with small weights to prevent NaN in gated_path
        nn.init.xavier_uniform_(self.decoder_pathway.weight, gain=0.1)
        
        self.decoder_base = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes)
        )
        self.decoder_vae = nn.Sequential(
            nn.Linear(latent_other_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes)
        )

        # Initialize pathway gate logits with positive bias to encourage pathway usage
        self._init_pathway_gates()

    def _init_pathway_gates(self):
        """Initialize pathway gate logits with positive bias.

        This ensures pathways start active (probability ~0.5-0.7)
        rather than inactive (probability ~0.0-0.1).
        The model must learn to turn pathways OFF rather than ON.
        """
        # Initialize bias to positive value (sigmoid(1.0) ≈ 0.73)
        nn.init.constant_(self.fc_pathway_gate_logits.bias, 1.0)
        # Initialize weights to small values to allow differentiation
        nn.init.xavier_normal_(self.fc_pathway_gate_logits.weight, gain=0.1)

    @property
    def gate(self):
        """Alias for gene_gate for compatibility with training functions."""
        return self.gene_gate

    @property
    def pathway_gate(self):
        """Compatibility wrapper for pathway gate access."""
        class PathwayGateWrapper:
            def __init__(self, parent):
                self._parent = parent
            def get_gates(self):
                """Return mean pathway gate probabilities across samples."""
                if hasattr(self._parent, '_pathway_gate_probs') and self._parent._pathway_gate_probs is not None:
                    return self._parent._pathway_gate_probs.mean(dim=0)
                else:
                    # Fallback: return uniform probabilities
                    return torch.ones(self._parent.max_pathways, device=next(self._parent.parameters()).device) * 0.5
        return PathwayGateWrapper(self)

    @property
    def pathway_gate_layer(self):
        """Alias for fc_pathway_gate_logits for compatibility."""
        return self.fc_pathway_gate_logits

    def forward(self, x: torch.Tensor, temperature: float = 0.5, k: int | None = None) -> dict:
        # 0. Input clamping (Prevents NaN from large z-score outliers)
        x_norm = torch.clamp(x, -10.0, 10.0)
        
        # A. Progression Branch (gated input)
        if k is not None:
            # Force Top-K hard selection (e.g., for final training epochs or inference)
            gene_mask = self.gene_gate.get_gates(k=k)
        else:
            gene_mask = self.gene_gate(temperature)
            
        x_prog = x_norm * gene_mask
        h_prog = F.relu(self.prog_fc1(x_prog))
        h_prog = F.relu(self.prog_fc2(h_prog))
        h_prog_normed = self.prog_norm(h_prog)
        
        # Stability: clamp progression logits before sigmoid to prevent saturation
        prog_logits = self.fc_progression(h_prog_normed)
        prog_logits = torch.clamp(prog_logits, min=-15.0, max=15.0)
        z_prog = torch.sigmoid(prog_logits)

        # B. Pathway Branch (ungated input)
        h_path = F.relu(self.path_fc1(x_norm))
        h_path = F.relu(self.path_fc2(h_path))
        # z_path_raw calculation remains the same
        z_path_raw = self.fc_pathway(h_path)
        
        # Stability: clamp raw pathway latent to prevent explosion
        z_path_raw = torch.clamp(z_path_raw, min=-20.0, max=20.0)

        # C. Per-Sample L0 Gating (Concrete Distribution)
        gate_logits = self.fc_pathway_gate_logits(h_path)
        # Clamp gate_logits to prevent numerical instability in sigmoid/log
        gate_logits = torch.clamp(gate_logits, min=-10.0, max=10.0)

        # Compute adaptive width tau_i for each sample
        tau = self.adaptive_width(h_path)  # (batch_size, 1), clamped to [tau_min, tau_max]

        # Store probabilities for get_gates() compatibility
        self._pathway_gate_probs = torch.sigmoid(gate_logits)

        if self.training:
            # Use adaptive width tau_i instead of global temperature
            u = torch.rand_like(gate_logits).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + gate_logits) / tau.clamp(min=0.01))
            s = s * 1.2 - 0.1
            path_mask = F.hardtanh(s, min_val=0.0, max_val=1.0)
        else:
            path_mask = (torch.sigmoid(gate_logits) > 0.5).float()

        z_path = z_path_raw * path_mask

        # D. VAE Latent
        mu_other = self.fc_other_mu(h_prog)
        logvar_other = self.fc_other_logvar(h_prog)
        # Stability: clamp logvar to reasonable range
        logvar_other = torch.clamp(logvar_other, min=-10.0, max=10.0)
        
        if self.training:
            std = torch.exp(0.5 * logvar_other)
            eps = torch.randn_like(std)
            z_other = mu_other + eps * std
        else:
            z_other = mu_other

        # E. Tunnel Gate Decoding
        base_recon = self.decoder_base(z_prog)
        path_recon = self.decoder_pathway(z_path)
        
        # Scale gated pathway by z_prog to create the "tunnel"
        # This ensures pathway-specific genes are only active when z_prog > 0
        gated_path = z_prog * path_recon
        vae_recon = self.decoder_vae(z_other)

        reconstruction = base_recon + gated_path + vae_recon

        # Numerical stability: final clamp on reconstruction
        reconstruction = torch.clamp(reconstruction, min=-50.0, max=50.0)

        return {
            "reconstruction": reconstruction,
            "z_progression": z_prog,
            "z_pathway": z_path,
            "z_pathway_raw": z_path_raw,
            "z_other": z_other,
            "mu_other": mu_other,
            "logvar_other": logvar_other,
            "path_mask": path_mask,
            "gate_logits": gate_logits,
            "decoder_weights": self.decoder_pathway.weight,
            "gated_input": x_prog,
            "gene_gate_mask": gene_mask,
            "active_pathways": (path_mask > 0.5).sum(dim=1).float().mean().item(),
            "adaptive_width_tau": tau,  # Sample-specific tunnel width
            "h_path": h_path,  # Pathway encoder hidden state for entropy regularization
        }

    def get_ne_progression(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.forward(x)
        return output["z_progression"]

    def get_gate_probabilities(self) -> torch.Tensor:
        return self.gene_gate.get_gates()

    def get_ne_scores(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            output = self.forward(x)
        return {
            "NE_state": output["z_progression"],
            "NE_potential": output["z_pathway"].mean(dim=1, keepdim=True)
        }

    def get_active_pathway_count(self, threshold: float = 0.5) -> int:
        """Return the number of active pathway dimensions based on gate probabilities."""
        if hasattr(self, '_pathway_gate_probs') and self._pathway_gate_probs is not None:
            mean_probs = self._pathway_gate_probs.mean(dim=0)
            return (mean_probs > threshold).sum().item()
        else:
            # Fallback: return max pathways
            return self.max_pathways




class ProgressionLoss(nn.Module):
    """
    Loss encouraging correlation between z_progression and NE seed gene expression.

    This loss ensures that high z_progression (near NE center) corresponds to
    high expression of known NE marker genes. It provides supervision for the
    radial progression axis.
    """

    def __init__(
        self,
        seed_gene_indices: list[int],
        n_genes: int,
        weight: float = 1.0
    ):
        """
        Initialize progression loss.

        Args:
            seed_gene_indices: Indices of NE seed genes in expression matrix
            n_genes: Total number of genes
            weight: Loss weight
        """
        super().__init__()

        self.seed_gene_indices = seed_gene_indices
        self.n_genes = n_genes
        self.weight = weight

        # Create mask for seed genes
        if len(seed_gene_indices) > 0:
            self.register_buffer(
                "seed_mask",
                torch.zeros(n_genes, dtype=torch.bool)
            )
            self.seed_mask[seed_gene_indices] = True
        else:
            self.register_buffer(
                "seed_mask",
                torch.zeros(n_genes, dtype=torch.bool)
            )

    def forward(self, z_progression: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute progression loss.

        Args:
            z_progression: Progression scores of shape (batch_size, 1)
            x: Original input expression of shape (batch_size, n_genes)

        Returns:
            Scalar loss value (negative correlation to maximize correlation)
        """
        if len(self.seed_gene_indices) == 0:
            return torch.tensor(0.0, device=x.device)

        # Extract seed gene expression
        seed_expr = x[:, self.seed_mask]

        # Compute mean seed gene expression per sample
        seed_mean = seed_expr.mean(dim=1, keepdim=True)

        # Compute Pearson correlation with variance check
        z_centered = z_progression - z_progression.mean()
        seed_centered = seed_mean - seed_mean.mean()

        # Check for zero variance to avoid division by zero
        z_norm = z_centered.norm()
        seed_norm = seed_centered.norm()
        if z_norm < 1e-8 or seed_norm < 1e-8:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        correlation = (
            (z_centered * seed_centered).sum() /
            (z_norm * seed_norm)
        )

        # Loss is negative correlation (maximize correlation)
        return -correlation * self.weight


class PathwayDiversityLoss(nn.Module):
    """
    Loss encouraging diversity in z_pathway to prevent mode collapse.

    This loss ensures that the pathway latent space is well-utilized and
    samples are spread across different pathways. Without this, all samples
    might collapse to the same pathway vector.

    Uses a contrastive approach: samples should be spread out in pathway space.
    """

    def __init__(self, temperature: float = 1.0, weight: float = 0.1):
        """
        Initialize pathway diversity loss.

        Args:
            temperature: Temperature for soft distance computation
            weight: Loss weight
        """
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def forward(self, z_pathway: torch.Tensor) -> torch.Tensor:
        """
        Compute pathway diversity loss.

        Args:
            z_pathway: Pathway vectors of shape (batch_size, pathway_dim)

        Returns:
            Scalar diversity loss (negative to encourage spread)
        """
        batch_size = z_pathway.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=z_pathway.device)

        # Compute pairwise squared distances
        # (batch, 1, dim) - (1, batch, dim) -> (batch, batch, dim) -> (batch, batch)
        diff = z_pathway.unsqueeze(1) - z_pathway.unsqueeze(0)
        dist_matrix = (diff ** 2).sum(dim=-1)

        # Mask out diagonal (self-distances)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=z_pathway.device)
        off_diag_dist = dist_matrix[mask]

        # Encourage spread: penalize small distances
        # Use soft minimum: -log(mean(exp(-dist/temp)))
        # This pushes samples apart in pathway space
        diversity = -torch.log(torch.mean(torch.exp(-off_diag_dist / self.temperature)) + 1e-8)

        return diversity * self.weight


class ProgressionDistributionLoss(nn.Module):
    """
    Loss encouraging z_progression to have a wide distribution across [0, 1].

    Without this, z_progression can collapse to a narrow range (e.g., all values
    near 1.0) while still maintaining correlation with seed genes. This loss
    ensures the progression axis spans the full range, making scores interpretable.

    IMPORTANT: This loss does NOT force a specific mean. The biological distribution
    is skewed toward low values (most samples are non-NE), so we should not force
    the mean to 0.5.

    Uses two mechanisms:
    1. Variance PUSH: encourage higher variance (not target-based)
    2. Range coverage: encourage at least some samples in low and high ranges
    """

    def __init__(self, min_variance: float = 0.04, weight: float = 0.5):
        """
        Initialize progression distribution loss.

        Args:
            min_variance: Minimum acceptable variance (push if below this)
            weight: Loss weight
        """
        super().__init__()
        self.min_variance = min_variance
        self.weight = weight

    def forward(self, z_progression: torch.Tensor) -> torch.Tensor:
        """
        Compute distribution loss for z_progression.

        Args:
            z_progression: Progression scores of shape (batch_size, 1)

        Returns:
            Scalar distribution loss
        """
        z_flat = z_progression.flatten()

        # Guard against empty or NaN inputs
        if z_flat.numel() == 0 or torch.isnan(z_flat).any():
            return torch.tensor(0.0, device=z_progression.device)

        # 1. Variance PUSH loss: penalize if variance is below minimum
        # This encourages spread without forcing a specific target
        current_var = z_flat.var()
        var_loss = F.relu(self.min_variance - current_var)  # Only penalize if below min

        # 2. Quantile-based coverage (more robust than fixed threshold)
        z_sorted = torch.sort(z_flat)[0]
        n = z_sorted.shape[0]

        p10_val = z_sorted[int(n * 0.1)]
        p90_val = z_sorted[int(n * 0.9)]

        # Push 10th percentile below 0.15, 90th percentile above 0.85
        # Give stronger weight to p90 to overcome biological sparsity of pure NE samples
        coverage_loss = (
            F.relu(p10_val - 0.15) +
            F.relu(0.85 - p90_val) * 3.0  # 3x stronger push for the high end
        )

        # Combined loss (no mean constraint - let biology determine distribution)
        # Clip to prevent explosion
        total_loss = (var_loss + coverage_loss) * self.weight
        return torch.clamp(total_loss, max=5.0)  # Cap maximum loss


class ErosionCorrelationLoss(nn.Module):
    """
    Loss encouraging correlation between z_progression and gene erosion.

    NE transdifferentiation involves "erosion" of adenocarcinoma genes
    (downregulation of AR signaling, prostate-specific markers, etc.).

    This loss ensures that high progression (near NE center) correlates with
    low expression of adenocarcinoma marker genes.
    """

    def __init__(
        self,
        erosion_gene_indices: list[int],
        n_genes: int,
        weight: float = 0.5
    ):
        """
        Initialize erosion correlation loss.

        Args:
            erosion_gene_indices: Indices of adenocarcinoma/AR pathway genes
            n_genes: Total number of genes
            weight: Loss weight
        """
        super().__init__()

        self.erosion_gene_indices = erosion_gene_indices
        self.n_genes = n_genes
        self.weight = weight

        # Create mask for erosion genes
        if len(erosion_gene_indices) > 0:
            self.register_buffer(
                "erosion_mask",
                torch.zeros(n_genes, dtype=torch.bool)
            )
            self.erosion_mask[erosion_gene_indices] = True
        else:
            self.register_buffer(
                "erosion_mask",
                torch.zeros(n_genes, dtype=torch.bool)
            )

    def forward(self, z_progression: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute erosion correlation loss.

        Args:
            z_progression: Progression scores of shape (batch_size, 1)
            x: Original input expression of shape (batch_size, n_genes)

        Returns:
            Scalar loss value (positive correlation = high progression, low erosion genes)
        """
        if len(self.erosion_gene_indices) == 0:
            return torch.tensor(0.0, device=x.device)

        # Extract erosion gene expression
        erosion_expr = x[:, self.erosion_mask]

        # Compute mean erosion gene expression per sample
        erosion_mean = erosion_expr.mean(dim=1, keepdim=True)

        # We want NEGATIVE correlation: high progression -> low erosion genes
        # So we maximize positive correlation between progression and NEGATIVE erosion
        neg_erosion = -erosion_mean

        z_centered = z_progression - z_progression.mean()
        erosion_centered = neg_erosion - neg_erosion.mean()

        # Check for zero variance to avoid division by zero
        z_norm = z_centered.norm()
        erosion_norm = erosion_centered.norm()
        if z_norm < 1e-8 or erosion_norm < 1e-8:
            return torch.tensor(0.0, device=z_progression.device, requires_grad=True)

        correlation = (
            (z_centered * erosion_centered).sum() /
            (z_norm * erosion_norm)
        )

        # Loss is negative correlation (maximize correlation with neg_erosion)
        return -correlation * self.weight


class ProliferationDecorrelationLoss(nn.Module):
    """
    Loss penalizing correlation between z_progression and proliferation scores.

    Optimized version: expects pre-standardized S/G2M scores (mean=0, std=1)
    so the loss becomes a simple dot-product correlation.
    """

    def __init__(self, weight: float = 1.0):
        """
        Initialize proliferation decorrelation loss.

        Args:
            weight: Loss weight
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        z_progression: torch.Tensor,
        s_scores: torch.Tensor,
        g2m_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute decorrelation loss using vectorized dot products.

        Args:
            z_progression: Progression scores (batch_size, 1)
            s_scores: Pre-standardized S-phase scores (batch_size,)
            g2m_scores: Pre-standardized G2M-phase scores (batch_size,)

        Returns:
            Scalar loss value (sum of squared correlations)
        """
        # Center and normalize z_progression
        z = z_progression.flatten()
        z_c = z - z.mean()
        z_std = z_c.std() + 1e-8
        z_norm = z_c / z_std

        # Since s and g2m are pre-standardized globally, they are roughly
        # mean=0, std=1 in the batch.
        # Correlation r = (1/N) * sum(z_norm * s_scores)
        corr_s = (z_norm * s_scores).mean()
        corr_g2m = (z_norm * g2m_scores).mean()

        # Penalize squared correlations
        loss = (corr_s**2 + corr_g2m**2)

        return loss * self.weight


class PrecomputedErosionLoss(nn.Module):
    """
    Erosion loss using pre-computed erosion indices from identity_erosion.py.

    This loss uses tissue-specific erosion indices that measure how much a sample
    has lost its lineage-specific identity (e.g., AR signaling in prostate,
    NKX2-1 in lung). High erosion indicates dedifferentiation toward NE phenotype.

    The target_erosion should be:
    - 1.0 for samples that have fully lost tissue identity (NE-high)
    - 0.0 for samples that retain tissue identity (Rim/non-NE)
    """

    def __init__(self, weight: float = 0.5):
        """
        Initialize precomputed erosion loss.

        Args:
            weight: Loss weight
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        z_progression: torch.Tensor,
        target_erosion: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute erosion correlation loss.

        Args:
            z_progression: Progression scores of shape (batch_size, 1) or (batch_size,)
            target_erosion: Pre-computed erosion indices of shape (batch_size,)

        Returns:
            Scalar loss value (negative correlation to maximize correlation)
        """
        # Flatten tensors
        z_progression = z_progression.flatten()
        target_erosion = target_erosion.flatten()

        # Guard against NaN or empty inputs
        if z_progression.numel() == 0 or target_erosion.numel() == 0:
            return torch.tensor(0.0, device=z_progression.device)
        if torch.isnan(z_progression).any() or torch.isnan(target_erosion).any():
            return torch.tensor(0.0, device=z_progression.device)

        # Center the vectors
        z_centered = z_progression - z_progression.mean()
        e_centered = target_erosion - target_erosion.mean()

        # Compute Pearson correlation with robust std
        z_std = z_centered.std()
        e_std = e_centered.std()

        # Guard against zero variance
        if z_std < 1e-8 or e_std < 1e-8:
            return torch.tensor(0.0, device=z_progression.device)

        correlation = (z_centered * e_centered).mean() / (z_std * e_std)

        # Clamp correlation to valid range
        correlation = torch.clamp(correlation, min=-1.0, max=1.0)

        # Return correlation (we want to maximize negative correlation with erosion)
        # Erosion index is inverted: high z_progression should correlate with low erosion
        return correlation * self.weight


class SeedGuidanceLoss(nn.Module):
    """
    Seed guidance loss encouraging correlation between z_ne and seed gene expression.

    This loss encourages the z_ne latent node to capture variance in known NE marker genes.
    """

    def __init__(
        self,
        seed_gene_indices: list[int],
        n_genes: int,
        weight: float = DEFAULT_SEED_GUIDANCE_WEIGHT
    ):
        """
        Initialize seed guidance loss.

        Args:
            seed_gene_indices: Indices of seed genes in expression matrix
            n_genes: Total number of genes
            weight: Loss weight
        """
        super().__init__()

        self.seed_gene_indices = seed_gene_indices
        self.n_genes = n_genes
        self.weight = weight

        # Create mask for seed genes
        if len(seed_gene_indices) > 0:
            self.register_buffer(
                "seed_mask",
                torch.zeros(n_genes, dtype=torch.bool)
            )
            self.seed_mask[seed_gene_indices] = True
        else:
            self.register_buffer(
                "seed_mask",
                torch.zeros(n_genes, dtype=torch.bool)
            )

    def forward(self, z_ne: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute seed guidance loss.

        FIX: Now handles multi-dimensional z_ne by aggregating to 1D first.

        Args:
            z_ne: NE scores of shape (batch_size, dim) or (batch_size, 1)
            x: Original input expression of shape (batch_size, n_genes)

        Returns:
            Scalar loss value
        """
        if len(self.seed_gene_indices) == 0:
            return torch.tensor(0.0, device=x.device)

        # FIX: Aggregate multi-dimensional z_ne to 1D if needed
        if z_ne.dim() > 1 and z_ne.shape[1] > 1:
            z_ne = z_ne.mean(dim=1, keepdim=True)

        # Extract seed gene expression
        seed_expr = x[:, self.seed_mask]

        # Compute correlation between z_ne and mean seed gene expression
        # Mean of seed genes per sample
        seed_mean = seed_expr.mean(dim=1, keepdim=True)

        # Pearson correlation with variance check
        z_ne_centered = z_ne - z_ne.mean()
        seed_centered = seed_mean - seed_mean.mean()

        # Check for zero variance to avoid division by zero
        z_norm = z_ne_centered.norm()
        seed_norm = seed_centered.norm()
        if z_norm < 1e-8 or seed_norm < 1e-8:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        correlation = (
            (z_ne_centered * seed_centered).sum() /
            (z_norm * seed_norm)
        )

        # Loss is negative correlation (maximize correlation)
        loss = -correlation * self.weight

        return loss


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient reversal layer for domain adversarial training.

    During forward pass, acts as identity function.
    During backward pass, reverses and scales gradients by -lambda_.
    This encourages the encoder to learn study-invariant representations.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """
        Forward pass: identity function.

        Args:
            ctx: Context object for saving information
            x: Input tensor
            lambda_: Gradient reversal strength

        Returns:
            Output tensor (same as input)
        """
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Backward pass: reverse gradients.

        Args:
            ctx: Context object with saved lambda_
            grad_output: Gradient from downstream

        Returns:
            Tuple of (reversed gradient, None for lambda_)
        """
        return -ctx.lambda_ * grad_output, None


class DomainClassifier(nn.Module):
    """
    Classifier to predict study_id from latent representation.

    Used in domain adversarial training to encourage study-invariant
    latent representations. The classifier tries to predict which study
    a sample came from, while the encoder tries to fool it.
    """

    def __init__(self, latent_dim: int, n_studies: int, hidden_dim: int = 64):
        """
        Initialize domain classifier.

        Args:
            latent_dim: Dimension of latent representation
            n_studies: Number of studies (output classes)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_studies)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict study_id from latent representation.

        Args:
            z: Latent representation (batch, latent_dim)

        Returns:
            Logits for study classification (batch, n_studies)
        """
        h = F.relu(self.fc1(z))
        return self.fc2(h)


def compute_domain_adversarial_loss(
    z: torch.Tensor,
    study_ids: torch.Tensor,
    classifier: DomainClassifier,
    lambda_grl: float = 1.0
) -> torch.Tensor:
    """
    Compute domain adversarial loss.

    Uses gradient reversal to encourage latent representations to be
    study-invariant. The loss trains the encoder to produce representations
    that cannot be used to predict which study a sample came from.

    Args:
        z: Latent representation (batch, latent_dim)
        study_ids: Study labels (batch,)
        classifier: Domain classifier network
        lambda_grl: Gradient reversal strength (default: 1.0)

    Returns:
        Domain adversarial loss (scalar)
    """
    # Apply gradient reversal
    z_reversed = GradientReversalFunction.apply(z, lambda_grl)

    # Predict study
    logits = classifier(z_reversed)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, study_ids)

    return loss


def compute_ranking_loss(
    z_state: torch.Tensor,
    is_anchor: torch.Tensor,
    margin: float = 0.5,
    confidence: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute ranking loss: anchors should score higher than unlabeled.

    This is a margin-based loss for weak supervision that ensures anchor samples
    (high-confidence NE annotations) score higher than unlabeled samples.

    FIX: Now handles multi-dimensional z_state by aggregating to 1D first.

    Args:
        z_state: State scores (batch, dim_state) or (batch, 1)
        is_anchor: Boolean tensor indicating anchors
        margin: Margin between anchor and unlabeled scores
        confidence: Optional confidence weights for anchors

    Returns:
        Scalar ranking loss
    """
    # FIX: Aggregate multi-dimensional z_state to 1D if needed
    if z_state.dim() > 1 and z_state.shape[1] > 1:
        z_state = z_state.mean(dim=1)

    anchor_scores = z_state[is_anchor].flatten()
    unlabeled_scores = z_state[~is_anchor].flatten()

    if len(anchor_scores) == 0 or len(unlabeled_scores) == 0:
        return torch.tensor(0.0, device=z_state.device)

    # Mean unlabeled score
    unlabeled_mean = unlabeled_scores.mean()

    # Margin-based ranking loss
    violations = margin - (anchor_scores - unlabeled_mean)
    violations = F.relu(violations)  # Only penalize violations

    # Apply confidence weights if provided
    if confidence is not None:
        confidence = confidence.to(z_state.device)
        violations = violations * confidence

    return violations.mean()


def compute_progression_loss(
    z_progression: torch.Tensor,
    seed_expression: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Compute progression loss: high z_progression should correlate with high NE seed expression.

    This loss anchors the progression axis to known NE biology by encouraging
    correlation between the learned progression score and seed gene expression.

    Args:
        z_progression: Progression latent scores (batch,) or (batch, 1)
        seed_expression: Mean expression of seed genes (batch,)
        temperature: Temperature for soft correlation computation

    Returns:
        Scalar progression loss (negative correlation to maximize)
    """
    # Flatten if needed
    z_progression = z_progression.flatten()
    seed_expression = seed_expression.flatten()

    # Guard against NaN or empty inputs
    if z_progression.numel() == 0 or seed_expression.numel() == 0:
        return torch.tensor(0.0, device=z_progression.device)
    if torch.isnan(z_progression).any() or torch.isnan(seed_expression).any():
        return torch.tensor(0.0, device=z_progression.device)

    # Center the vectors
    z_centered = z_progression - z_progression.mean()
    seed_centered = seed_expression - seed_expression.mean()

    # Compute Pearson correlation with robust std calculation
    z_std = z_centered.std()
    seed_std = seed_centered.std()

    # Guard against zero variance (all same values)
    if z_std < 1e-8 or seed_std < 1e-8:
        return torch.tensor(0.0, device=z_progression.device)

    correlation = (z_centered * seed_centered).mean() / (z_std * seed_std)

    # Clamp correlation to valid range
    correlation = torch.clamp(correlation, min=-1.0, max=1.0)

    # Return negative correlation (we want to maximize correlation)
    return -correlation


def compute_pathway_diversity_loss(
    z_pathway: torch.Tensor,
    z_progression: torch.Tensor,
    temperature: float = 1.0,
    progression_threshold: float = 0.5
) -> torch.Tensor:
    """
    Compute pathway diversity loss: samples at similar progression should have diverse pathways.

    CRITICAL: This loss maintains separation between ASCL1-type and NEUROD1-type
    NE tumors. DO NOT collapse pathways at high progression.

    The loss encourages samples with similar progression scores to spread out
    in the pathway dimension, allowing multiple NE subtypes to coexist.

    Args:
        z_pathway: Pathway latent scores (batch, dim_pathway)
        z_progression: Progression latent scores (batch,) or (batch, 1)
        temperature: Temperature for soft weighting
        progression_threshold: Minimum progression to include in diversity loss

    Returns:
        Scalar diversity loss (negative to encourage spread)
    """
    batch_size = z_pathway.shape[0]
    pathway_dim = z_pathway.shape[1]

    if batch_size < 2:
        return torch.tensor(0.0, device=z_pathway.device)

    # Guard against NaN inputs
    if torch.isnan(z_pathway).any() or torch.isnan(z_progression).any():
        return torch.tensor(0.0, device=z_pathway.device)

    # Flatten progression
    z_progression = z_progression.flatten()

    # Normalize progression to [0, 1] range for weighting
    prog_min = z_progression.min()
    prog_max = z_progression.max()
    prog_range = prog_max - prog_min + 1e-8
    prog_normalized = (z_progression - prog_min) / prog_range

    # Normalize pathway dimensions to prevent explosion with high dimensions
    # Use mean squared distance instead of sum to be dimension-invariant
    pathway_diff = z_pathway.unsqueeze(1) - z_pathway.unsqueeze(0)  # (batch, batch, dim)
    pathway_dist = (pathway_diff ** 2).mean(dim=-1)  # (batch, batch) - normalized by dim

    # Weight by progression similarity (samples at similar progression should be diverse)
    prog_diff = (z_progression.unsqueeze(1) - z_progression.unsqueeze(0)).abs()  # (batch, batch)
    prog_similarity = torch.exp(-prog_diff / temperature)  # Higher for similar progression

    # Mask out diagonal
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=z_pathway.device)

    # Focus on high-progression samples (where NE subtypes matter most)
    high_prog_mask = (prog_normalized > progression_threshold).float()
    weight_matrix = high_prog_mask.unsqueeze(1) * high_prog_mask.unsqueeze(0)

    # Combined weight: progression similarity * high progression focus
    combined_weight = prog_similarity * weight_matrix * mask.float()

    # Diversity loss: penalize small distances between similar-progression samples
    # (negative because we want to maximize distances)
    # Clip to prevent explosion
    diversity = -(combined_weight * pathway_dist).sum() / (combined_weight.sum() + 1e-8)
    diversity = torch.clamp(diversity, min=-1.0, max=0.0)  # Clip to reasonable range

    return diversity


def compute_pathway_orthogonality_loss(z_pathway: torch.Tensor) -> torch.Tensor:
    """
    Encourage pathway dimensions to be orthogonal (uncorrelated).

    This enforces that ASCL1-type and NEUROD1-type pathways are
    separable as independent axes in pathway space.

    Args:
        z_pathway: Pathway latent (batch, dim_pathway)

    Returns:
        Orthogonality loss (scalar) - lower is better
    """
    if z_pathway.shape[0] < 2 or z_pathway.shape[1] < 2:
        return torch.tensor(0.0, device=z_pathway.device)

    # Guard against NaN inputs
    if torch.isnan(z_pathway).any():
        return torch.tensor(0.0, device=z_pathway.device)

    # Center the pathway vectors
    z_centered = z_pathway - z_pathway.mean(dim=0, keepdim=True)

    # Normalize each dimension
    z_std = z_centered.std(dim=0, keepdim=True) + 1e-8
    z_norm = z_centered / z_std

    # Correlation matrix between pathway dimensions: (dim, dim)
    corr_matrix = torch.mm(z_norm.t(), z_norm) / z_norm.shape[0]

    # Penalize off-diagonal correlations (want them close to 0)
    # Use upper triangle (excluding diagonal)
    dim = corr_matrix.shape[0]
    mask = torch.triu(torch.ones_like(corr_matrix), diagonal=1).bool()
    off_diag_corr = corr_matrix[mask]

    # Penalize absolute correlation (want orthogonality)
    orthogonality_loss = off_diag_corr.abs().mean()

    return orthogonality_loss


def compute_decoder_orthogonality_loss(decoder_pathway: nn.Module) -> torch.Tensor:
    """
    Compute decoder weight orthogonality loss for pathway dimensions.

    This loss enforces that different pathway dimensions decode to orthogonal
    (uncorrelated) gene expression patterns. This prevents "rotational invariance"
    where the model uses different pathway dimensions for the same biological
    program across different batches.

    Mathematical formulation:
        L_ortho = mean(|W @ W.T - I|)
    Where W is the first layer weight matrix of decoder_pathway.

    Args:
        decoder_pathway: The pathway decoder module (Sequential with Linear first layer)

    Returns:
        Orthogonality loss (scalar) - lower is better (0 = perfectly orthogonal)
    """
    # Extract first layer weights from decoder_pathway
    # decoder_pathway can be Sequential(Linear, ReLU, Linear) or just Linear
    if isinstance(decoder_pathway, nn.Sequential):
        first_layer = decoder_pathway[0]  # Linear(in_features, hidden_dim)
        W = first_layer.weight  # Shape: (hidden_dim, pathway_dim)
    elif isinstance(decoder_pathway, nn.Linear):
        # Simple Linear layer: (pathway_dim, n_genes)
        W = decoder_pathway.weight  # Shape: (n_genes, pathway_dim)
    else:
        # Fallback: try to get weight attribute
        W = decoder_pathway.weight

    # Guard against NaN in weights
    if torch.isnan(W).any():
        return torch.tensor(0.0, device=W.device)

    # Guard against newly initialized weights with very small norms
    # With Kaiming initialization, max abs value ~sqrt(1/fan_in)
    # If weights are too small, the loss would be numerically unstable
    if W.abs().max() < 1e-4:
        return torch.tensor(0.0, device=W.device)

    # We want the COLUMNS of W to be orthogonal (one per pathway dimension)
    # So we work with W.T: (pathway_dim, hidden_dim)
    W_cols = W.T  # (pathway_dim, hidden_dim)

    pathway_dim = W_cols.shape[0]
    if pathway_dim < 2:
        return torch.tensor(0.0, device=W.device)

    # Compute column norms with clamping for numerical stability
    col_norms = W_cols.norm(dim=1, keepdim=True)
    # Clamp minimum norm to prevent division by very small numbers
    col_norms = col_norms.clamp(min=1e-3)

    # Normalize each column (pathway dimension)
    W_norm = W_cols / col_norms

    # Compute correlation matrix between pathway dimensions
    # (pathway_dim, pathway_dim)
    corr_matrix = torch.mm(W_norm, W_norm.t())

    # Create mask for off-diagonal elements
    mask = ~torch.eye(pathway_dim, dtype=torch.bool, device=W.device)
    off_diag = corr_matrix[mask]

    # Penalize non-orthogonality (want off-diagonal correlations close to 0)
    orthogonality_loss = (off_diag ** 2).mean()

    return orthogonality_loss


class ContrastiveMarginLoss(nn.Module):
    """
    Contrastive margin loss for protecting rare biological states.

    This loss ensures that rare phenotypes (e.g., Amphicrine cells with
    high AR + high SYP) are forced into unique pathway coordinates,
    preventing them from being pruned by sparsity penalties.

    Mathematical formulation:
        L_rare = max(0, margin - ||z_path_rare - z_path_anchor||)

    Where:
        - z_path_rare: pathway vectors of rare samples
        - z_path_anchor: pathway vectors of common samples (e.g., classical NE)
        - margin: minimum separation distance (default 1.0)
    """

    def __init__(
        self,
        margin: float = 1.0,
        rare_sample_indices: list[int] | None = None,
        anchor_sample_indices: list[int] | None = None
    ):
        """
        Initialize contrastive margin loss.

        Args:
            margin: Minimum separation distance between rare and anchor samples
            rare_sample_indices: Indices of rare samples in dataset
            anchor_sample_indices: Indices of anchor (common) samples
        """
        super().__init__()
        self.margin = margin
        self.rare_sample_indices = rare_sample_indices or []
        self.anchor_sample_indices = anchor_sample_indices or []

    def update_indices(
        self,
        rare_indices: list[int],
        anchor_indices: list[int]
    ) -> None:
        """Update sample indices after identification."""
        self.rare_sample_indices = rare_indices
        self.anchor_sample_indices = anchor_indices

    def forward(
        self,
        z_pathway: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive margin loss.

        Args:
            z_pathway: Pathway latent vectors (batch, pathway_dim)
            batch_indices: Original dataset indices for batch samples

        Returns:
            Contrastive loss (scalar) - lower is better
        """
        # Guard against NaN inputs
        if torch.isnan(z_pathway).any():
            return torch.tensor(0.0, device=z_pathway.device)

        if not self.rare_sample_indices or not self.anchor_sample_indices:
            return torch.tensor(0.0, device=z_pathway.device)

        # Find rare and anchor samples in current batch
        batch_indices_np = batch_indices.cpu().numpy()

        rare_mask = torch.tensor(
            [idx in self.rare_sample_indices for idx in batch_indices_np],
            dtype=torch.bool,
            device=z_pathway.device
        )
        anchor_mask = torch.tensor(
            [idx in self.anchor_sample_indices for idx in batch_indices_np],
            dtype=torch.bool,
            device=z_pathway.device
        )

        # Get pathway vectors for rare and anchor samples
        z_rare = z_pathway[rare_mask]
        z_anchor = z_pathway[anchor_mask]

        if z_rare.shape[0] == 0 or z_anchor.shape[0] == 0:
            return torch.tensor(0.0, device=z_pathway.device)

        # Compute pairwise distances between rare and anchor samples
        # z_rare: (n_rare, dim), z_anchor: (n_anchor, dim)
        # distances: (n_rare, n_anchor)
        diff = z_rare.unsqueeze(1) - z_anchor.unsqueeze(0)
        distances = (diff ** 2).sum(dim=-1).sqrt()

        # Contrastive loss: penalize if distance < margin
        # We want rare samples to be AT LEAST margin distance from anchors
        margin_violations = torch.clamp(self.margin - distances, min=0)

        # Average over all pairs
        loss = margin_violations.mean()

        return loss


def compute_erosion_correlation_loss(
    z_progression: torch.Tensor,
    erosion_index: torch.Tensor
) -> torch.Tensor:
    """
    Compute erosion correlation loss: high z_progression should correlate with high erosion index.

    This loss ensures that the learned NE progression axis correlates with
    phenotypic erosion (loss of lineage-specific markers), a hallmark of
    neuroendocrine transdifferentiation.

    Args:
        z_progression: Progression latent scores (batch,) or (batch, 1)
        erosion_index: Computed erosion index (batch,) - higher means more dedifferentiated

    Returns:
        Scalar erosion correlation loss (negative correlation to maximize)
    """
    # Flatten if needed
    z_progression = z_progression.flatten()
    erosion_index = erosion_index.flatten()

    # Center the vectors
    z_centered = z_progression - z_progression.mean()
    erosion_centered = erosion_index - erosion_index.mean()

    # Compute Pearson correlation
    z_std = z_centered.std() + 1e-8
    erosion_std = erosion_centered.std() + 1e-8
    correlation = (z_centered * erosion_centered).mean() / (z_std * erosion_std)

    # Return negative correlation (we want to maximize correlation)
    return -correlation


def get_default_lineage_gene_indices(gene_names: list[str]) -> list[int]:
    """
    Get indices of lineage-specific marker genes for erosion computation.

    These genes represent tissue-specific identity that is lost during
    neuroendocrine transdifferentiation.

    Args:
        gene_names: List of gene names in the dataset

    Returns:
        List of indices for lineage-specific genes found in the dataset
    """
    # Lineage-specific markers (primarily prostate, but includes others)
    lineage_genes = [
        # Prostate lineage
        'AR', 'NKX3-1', 'KLK3', 'KLK2', 'FKBP5', 'TMPRSS2',
        # General epithelial
        'EPCAM', 'KRT8', 'KRT18', 'KRT19',
        # Lung lineage (for lung cancer samples)
        'NKX2-1', 'SFTPB', 'SFTPC',
        # Other tissue-specific markers
        'CDX2',  # Intestinal
        'GATA3',  # Breast/urothelial
        'PAX8',  # Renal/ovarian
        'TTF1',  # Thyroid/lung
    ]

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    indices = [gene_to_idx[g] for g in lineage_genes if g in gene_to_idx]

    logger.info(f"Found {len(indices)} lineage-specific markers for erosion computation")
    return indices


def compute_erosion_index(
    expression: torch.Tensor,
    lineage_gene_indices: list[int],
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute erosion index: inverse of lineage-specific marker expression.

    The erosion index measures phenotypic dedifferentiation by tracking
    the loss of lineage-specific markers. Higher values indicate more
    dedifferentiated/eroded phenotype.

    Args:
        expression: Full expression matrix (batch, n_genes)
        lineage_gene_indices: Indices of lineage-specific marker genes
        temperature: Temperature for soft inversion

    Returns:
        Erosion index (batch,) - higher means more dedifferentiated
    """
    if len(lineage_gene_indices) == 0:
        return torch.zeros(expression.shape[0], device=expression.device)

    # Get lineage marker expression
    lineage_expr = expression[:, lineage_gene_indices]  # (batch, n_lineage)

    # Mean lineage expression (higher = more differentiated)
    lineage_mean = lineage_expr.mean(dim=1)  # (batch,)

    # Erosion index is inverse (higher = more dedifferentiated)
    # Use soft inverse to avoid division by zero
    erosion = torch.exp(-lineage_mean / temperature)

    return erosion


def identify_amphicrine_samples(
    adata: ad.AnnData,
    ar_threshold: float = 1.0,
    syp_threshold: float = 0.5,
    min_samples: int = 20
) -> tuple[list[int], list[int]]:
    """
    Identify Amphicrine-like samples (high AR + high SYP) for rare state protection.

    Amphicrine tumors are a rare subtype that co-express adenocarcinoma (AR)
    and neuroendocrine (SYP) markers. Without protection, these rare samples
    may be pruned by sparsity penalties.

    Args:
        adata: AnnData with expression data
        ar_threshold: Z-score threshold for AR expression
        syp_threshold: Z-score threshold for SYP expression
        min_samples: Minimum samples required for valid rare population

    Returns:
        Tuple of (rare_indices, anchor_indices)
    """
    # Get expression matrix
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X

    # Find AR and SYP gene indices
    gene_names = adata.var_names.tolist()

    ar_idx = None
    syp_idx = None

    for i, gene in enumerate(gene_names):
        if gene == 'AR':
            ar_idx = i
        elif gene == 'SYP':
            syp_idx = i

    if ar_idx is None or syp_idx is None:
        logger.warning("AR or SYP not found in gene list. Rare state protection disabled.")
        return [], []

    # Identify Amphicrine-like samples
    ar_expr = X[:, ar_idx]
    syp_expr = X[:, syp_idx]

    rare_mask = (ar_expr > ar_threshold) & (syp_expr > syp_threshold)
    rare_indices = np.where(rare_mask)[0].tolist()

    # Identify anchor samples (classical NE: high CHGA, low AR)
    chga_idx = gene_names.index('CHGA') if 'CHGA' in gene_names else None

    if chga_idx is not None:
        chga_expr = X[:, chga_idx]
        anchor_mask = (chga_expr > 1.0) & (ar_expr < 0)
        anchor_indices = np.where(anchor_mask)[0].tolist()
    else:
        # Fallback: use all non-rare samples as anchors
        anchor_indices = [i for i in range(len(rare_mask)) if not rare_mask[i]]

    if len(rare_indices) < min_samples:
        logger.warning(
            f"Only {len(rare_indices)} Amphicrine-like samples found. "
            f"Need at least {min_samples} for rare state protection."
        )
        return [], []

    logger.info(
        f"Identified {len(rare_indices)} Amphicrine-like samples "
        f"(AR>{ar_threshold}, SYP>{syp_threshold}) and {len(anchor_indices)} anchor samples"
    )

    return rare_indices, anchor_indices


def get_phase_lambdas(epoch: int, total_epochs: int = 500) -> dict:
    """
    NESTOR training schedule with three phases.

    Phase A (0-100): Foundation - NO sparsity penalty
    Phase B (100-400): Transition - ramp up sparsity and KL
    Phase C (400+): Full training - strong sparsity
    """
    if epoch < 100:
        # Phase A: Foundation - NO sparsity penalty, let model learn important genes
        return {
            'phase': 'A',
            'lambda_progression': 4.0,
            'lambda_distribution': 1.0,
            'lambda_diversity': 0.0,
            'lambda_pathway_ortho': 0.0,
            'lambda_rare': 0.0,
            'lambda_pathway_gate': 0.0,
            'lambda_pathway_util': 0.0,
            'lambda_erosion': 1.0,
            'lambda_sparse': 0.0,  # CRITICAL FIX: No pruning in Phase A
            'lambda_proliferation': 5.0,
            'lambda_kl': 0.0,
            'lambda_domain': 0.0
        }
    elif epoch < 400:
        # Phase B: Transition - gradually ramp up sparsity
        progress = (epoch - 100) / 300  # 0 -> 1
        return {
            'phase': 'B',
            'lambda_progression': 4.0,
            'lambda_distribution': 1.0 - 0.5 * progress,
            'lambda_diversity': 0.5 * progress,
            'lambda_pathway_ortho': 0.2 * progress,
            'lambda_rare': 0.2 * progress,
            'lambda_pathway_gate': 0.2 * progress,
            'lambda_pathway_util': 0.0,
            'lambda_erosion': 1.0,
            'lambda_sparse': 2.0 + 8.0 * progress,   # 2.0 -> 10.0
            'lambda_proliferation': 5.0,
            'lambda_kl': 0.1 * progress,
            'lambda_domain': 0.1 * progress
        }
    else:
        # Phase C: Full training - consolidation with strong sparsity
        return {
            'phase': 'C',
            'lambda_progression': 5.0,
            'lambda_distribution': 0.5,
            'lambda_diversity': 0.5,
            'lambda_pathway_ortho': 0.2,
            'lambda_rare': 0.2,
            'lambda_pathway_gate': 0.2,
            'lambda_pathway_util': 0.0,
            'lambda_erosion': 1.0,
            'lambda_sparse': 20.0,  # Strong pruning
            'lambda_proliferation': 5.0,
            'lambda_kl': 0.1,
            'lambda_domain': 0.1
        }


def get_gate_temperature(epoch: int, total_epochs: int = 180) -> float:
    """
    Get temperature for L0 gates with exponential decay.

    FIX: Use constant high temperature (1.0) during Phase A.
    """
    if epoch < 40:
        return 1.0

    start_temp = 1.0
    min_temp = 0.1
    decay_rate = 0.96
    adjusted_epoch = epoch - 40
    return max(min_temp, start_temp * (decay_rate ** adjusted_epoch))


def compute_pathway_utilization_bonus(path_mask: torch.Tensor, target_active: float = 4.0) -> torch.Tensor:
    """
    Compute bonus for pathway utilization to prevent collapse.

    This provides POSITIVE incentive for the model to use pathways,
    counteracting the L0 sparsity penalty. Without this, the model
    can achieve good reconstruction without using pathways at all.

    Args:
        path_mask: Pathway mask tensor (batch, n_pathways)
        target_active: Target number of active pathways per sample

    Returns:
        Negative loss (bonus) - higher is better for the model
    """
    if torch.isnan(path_mask).any():
        return torch.tensor(0.0, device=path_mask.device)

    # Count active pathways per sample (mask > 0.5)
    active_per_sample = (path_mask > 0.5).float().sum(dim=1)

    # Compute mean active pathways
    mean_active = active_per_sample.mean()

    # Bonus: reward being close to target (use negative squared distance)
    # This creates a "sweet spot" around target_active
    bonus = -((mean_active - target_active) ** 2)

    # Scale to reasonable magnitude
    return bonus * 0.1


def compute_seed_score_s0(
    expression: pd.DataFrame | ad.AnnData,
    seed_genes: list[str],
    method: str = "mean"
) -> pd.Series:
    """
    Compute initial seed score S0 from expression and seed genes.

    CRITICAL: This uses a GLOBAL threshold, NOT per-study. Many cancer types
    lack NE biology entirely. A per-study threshold would incorrectly label
    non-NE samples as "NE-high" in those studies.

    Args:
        expression: Expression matrix (samples x genes) as DataFrame or AnnData
        seed_genes: List of seed gene symbols
        method: Aggregation method ('mean', 'sum', 'max')

    Returns:
        Series of seed scores indexed by sample
    """
    # Handle both DataFrame and AnnData inputs
    if isinstance(expression, ad.AnnData):
        expr_df = pd.DataFrame(expression.X, index=expression.obs.index, columns=expression.var.index)
    else:
        expr_df = expression

    # Find available seed genes
    available = [g for g in seed_genes if g in expr_df.columns]
    if not available:
        logger.warning(f"No seed genes found in expression data. Looking for: {seed_genes[:5]}...")
        return pd.Series(0.0, index=expr_df.index)

    logger.info(f"Found {len(available)}/{len(seed_genes)} seed genes in expression data")

    # Compute mean expression of seed genes
    seed_expr = expr_df[available]

    if method == "mean":
        scores = seed_expr.mean(axis=1)
    elif method == "sum":
        scores = seed_expr.sum(axis=1)
    elif method == "max":
        scores = seed_expr.max(axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    return scores


def select_anchors_global_threshold(
    s0: pd.Series,
    quantile: float = 0.8
) -> list:
    """
    Select anchors using GLOBAL threshold (not per-study).

    CRITICAL: This uses a GLOBAL threshold across all samples, NOT per-study.
    Many cancer types lack NE biology entirely. A per-study threshold would
    incorrectly label non-NE samples as "NE-high" in those studies.

    Args:
        s0: Seed scores indexed by sample_id
        quantile: Quantile threshold (default 0.8 = top 20%)

    Returns:
        List of anchor sample IDs
    """
    if len(s0) == 0:
        return []

    threshold = s0.quantile(quantile)
    anchors = s0[s0 >= threshold].index.tolist()

    logger.info(f"Selected {len(anchors)} anchors at {quantile:.0%} quantile (threshold={threshold:.4f})")

    return anchors


def save_model_checkpoint(
    model: "NESTORModel | NESTORModel",
    output_dir: Path,
    config: dict[str, Any],
    history: dict[str, list[float]] | None = None
) -> None:
    """
    Save model checkpoint and configuration.

    Args:
        model: Trained model (v1 or v2)
        output_dir: Directory to save checkpoint
        config: Model configuration dictionary
        history: Optional training history
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model state - move to CPU for portability (BUGFIX: MPS tensors not portable)
    model_path = output_dir / "model.pt"
    state_dict = model.state_dict()
    # Move all tensors to CPU
    cpu_state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
    torch.save({
        "model_state_dict": cpu_state_dict,
        "config": config
    }, model_path)
    logger.info(f"Saved model to {model_path}")

    # Save config as JSON
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_path}")

    # Save training log
    if history is not None:
        log_path = output_dir / "training_log.csv"
        df = pd.DataFrame(history)
        df.index.name = "epoch"
        df.to_csv(log_path)
        logger.info(f"Saved training log to {log_path}")


def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between learned and prior distributions.

    KL(q(z|x) || p(z)) where p(z) = N(0, I)

    Args:
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution

    Returns:
        KL divergence (scalar)
    """
    # KL = -0.5 * sum(1 + log(var) - mu^2 - var)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl


def compute_kl_divergence_with_free_bits(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.5
) -> torch.Tensor:
    """
    FIX: Compute KL divergence with free bits to prevent posterior collapse.

    Free bits sets a minimum KL per dimension, preventing the model from
    completely ignoring latent dimensions.

    Args:
        mu: Mean of latent distribution (batch, latent_dim)
        logvar: Log variance of latent distribution (batch, latent_dim)
        free_bits: Minimum KL per dimension (default 0.5 nats)

    Returns:
        KL divergence with free bits (scalar)
    """
    # Guard against NaN inputs
    if torch.isnan(mu).any() or torch.isnan(logvar).any():
        return torch.tensor(0.0, device=mu.device)

    # Compute per-dimension KL
    # KL = -0.5 * (1 + logvar - mu^2 - exp(logvar))
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (batch, dim)

    # Apply free bits: max(free_bits, kl)
    # This means we don't penalize KL below free_bits threshold
    kl_with_free_bits = torch.clamp(kl_per_dim, min=free_bits)

    # Return mean over batch and dimensions (not sum, to normalize by batch size)
    return kl_with_free_bits.mean()


def compute_diversity_loss(
    z_state: torch.Tensor,
    z_potential: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    FIX: Compute diversity loss to encourage spread in latent space.

    This loss penalizes samples that cluster too closely together,
    encouraging the model to use the full latent space.

    Args:
        z_state: State latent (batch, dim)
        z_potential: Potential latent (batch, dim)
        temperature: Temperature for soft min

    Returns:
        Scalar diversity loss
    """
    batch_size = z_state.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=z_state.device)

    # Aggregate to 1D scores for diversity computation
    z_state_1d = z_state.mean(dim=1)  # (batch,)
    z_potential_1d = z_potential.mean(dim=1)  # (batch,)

    # Compute pairwise distances
    # We want samples to be spread out, so penalize small distances
    state_diff = z_state_1d.unsqueeze(1) - z_state_1d.unsqueeze(0)  # (batch, batch)
    potential_diff = z_potential_1d.unsqueeze(1) - z_potential_1d.unsqueeze(0)  # (batch, batch)

    # Compute squared distances
    state_dist = state_diff.pow(2)
    potential_dist = potential_diff.pow(2)

    # Mask out diagonal (self-distances)
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=z_state.device)

    # Compute soft minimum distance (we want this to be large)
    # Use negative log of mean exp(-dist/temperature) to encourage spread
    state_diversity = -torch.log(torch.mean(torch.exp(-state_dist[mask] / temperature)) + 1e-8)
    potential_diversity = -torch.log(torch.mean(torch.exp(-potential_dist[mask] / temperature)) + 1e-8)

    return state_diversity + potential_diversity


def get_kl_weight_with_annealing(
    epoch: int,
    anneal_epochs: int = 50,
    max_weight: float = 1.0
) -> float:
    """
    FIX: Compute KL weight with linear annealing.

    Gradually increases KL weight from 0 to max_weight over anneal_epochs.
    This prevents posterior collapse in early training.

    Args:
        epoch: Current epoch
        anneal_epochs: Number of epochs to anneal over
        max_weight: Maximum KL weight to reach

    Returns:
        Current KL weight (beta)
    """
    if epoch >= anneal_epochs:
        return max_weight

    # Linear annealing
    progress = epoch / anneal_epochs
    return progress * max_weight


def compute_gate_penalty(gate: HardConcreteGate) -> torch.Tensor:
    """
    Compute gate sparsity penalty.

    Args:
        gate: HardConcreteGate module

    Returns:
        L0 penalty (scalar)
    """
    return gate.l0_penalty()


def compute_per_sample_l0_penalty(gate_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute L0 penalty for per-sample gate logits.
    """
    # Guard against NaN inputs
    if torch.isnan(gate_logits).any():
        return torch.tensor(0.0, device=gate_logits.device)

    log_ratio = math.log(-(-0.1) / 1.1)
    probs = torch.sigmoid(gate_logits - log_ratio)
    return probs.mean()



def compute_decorrelation_loss(
    z_state: torch.Tensor,
    z_potential: torch.Tensor
) -> torch.Tensor:
    """
    Compute decorrelation loss between z_state and z_potential.

    Uses squared covariance as the loss term.

    FIX: Now handles multi-dimensional latents by aggregating to 1D first.

    Args:
        z_state: State head output (batch, dim_state)
        z_potential: Potential head output (batch, dim_potential)

    Returns:
        Scalar decorrelation loss
    """
    # FIX: Aggregate multi-dimensional latents to 1D scores
    z_state_1d = z_state.mean(dim=1)  # (batch,)
    z_potential_1d = z_potential.mean(dim=1)  # (batch,)

    # Center the vectors
    z_state_centered = z_state_1d - z_state_1d.mean()
    z_potential_centered = z_potential_1d - z_potential_1d.mean()

    # Compute covariance
    cov = (z_state_centered * z_potential_centered).mean()

    # Return squared covariance (penalize correlation)
    return cov ** 2


def get_phase_loss_weights(epoch: int) -> dict[str, float]:
    """
    Get loss weights for 3-phase training schedule.

    Phase 1 (Warm-up, epochs 0-30): Only recon + gate
    Phase 2 (Elasticity, epochs 30-90): Gradually increase manifold
    Phase 3 (Smoothness, epochs 90-120): Enable jacobian

    Args:
        epoch: Current training epoch

    Returns:
        Dictionary of loss weights for each component
    """
    if epoch < 30:
        return {
            'recon': 1.0,
            'gate': 1.0,
            'manifold': 0.0,
            'jacobian': 0.0,
            'kl': 0.5
        }
    elif epoch < 90:
        # Gradually increase manifold weight
        progress = (epoch - 30) / 60  # 0 to 1 over elasticity phase
        return {
            'recon': 1.0,
            'gate': 1.0,
            'manifold': progress * 1.0,  # 0 to 1.0
            'jacobian': 0.0,
            'kl': 0.5
        }
    else:
        return {
            'recon': 1.0,
            'gate': 1.0,
            'manifold': 1.0,
            'jacobian': 0.1,
            'kl': 0.5
        }


def load_manifold_neighbors(processed_dir: Path) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Load pre-computed k-NN neighbor indices and weights.

    Args:
        processed_dir: Path to processed data directory

    Returns:
        Tuple of (neighbor_indices, neighbor_weights) tensors, or (None, None) if not found
    """
    neighbors_path = processed_dir / 'manifold_neighbors.parquet'
    if not neighbors_path.exists():
        logger.warning(f"Manifold neighbors not found at {neighbors_path}")
        logger.warning("Run scripts/build_manifold.py first")
        return None, None

    df = pd.read_parquet(neighbors_path)
    
    # IMPROVE: Use contiguous NumPy arrays to avoid "extremely slow list-to-tensor" warning
    indices_np = np.stack(df['neighbor_indices'].values).astype(np.int64)
    weights_np = np.stack(df['neighbor_weights'].values).astype(np.float32)
    
    neighbor_indices = torch.from_numpy(indices_np)
    neighbor_weights = torch.from_numpy(weights_np)

    logger.info(f"Loaded manifold neighbors: {neighbor_indices.shape}")
    return neighbor_indices, neighbor_weights


def train_model(
    model: NESTORModel,
    adata: ad.AnnData,
    seed_gene_indices: list[int],
    s_scores: np.ndarray | None = None,
    g2m_scores: np.ndarray | None = None,
    precomputed_erosion_indices: np.ndarray | None = None,
    rare_sample_indices: list[int] | None = None,
    anchor_sample_indices: list[int] | None = None,
    lineage_gene_indices: list[int] | None = None,
    n_epochs: int = 180,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    study_ids: np.ndarray | None = None,
    manifold_neighbors_path: Path | None = None
) -> tuple[NESTORModel, dict]:
    """
    Train NESTOR model with L0-gated pathway dimensions.

    This function implements the full training pipeline with:
    1. L0-gated pathway space (learns optimal D*) from D_max=16)
    2. Decoder weight orthogonality (prevents rotational invariance)
    3. Rare state protection (contrastive margins for Amphicrine-like samples)
    4. Tunnel gate architecture (pathway contribution scaled by progression)

    Args:
        model: NESTORModel to train
        adata: AnnData with expression data
        seed_gene_indices: Indices of NE seed genes
        s_scores: S-phase scores for each sample
        g2m_scores: G2M-phase scores for each sample
        precomputed_erosion_indices: Pre-computed erosion indices per sample
        rare_sample_indices: Indices of rare (Amphicrine-like) samples
        anchor_sample_indices: Indices of anchor (classical NE) samples
        lineage_gene_indices: Indices of lineage-specific genes
        n_epochs: Number of training epochs (default 180)
        batch_size: Batch size
        learning_rate: Learning rate
        study_ids: Study IDs for domain adversarial training

    Returns:
        Tuple of (trained model, training history)
    """
    device = get_device()
    model = model.to(device)

    logger.info(f"Training NESTOR model on {device}")
    logger.info(f"  Samples: {adata.n_obs}, Genes: {adata.n_vars}")
    logger.info(f"  Max pathway dimensions: {model.max_pathways}")
    logger.info(f"  Epochs: {n_epochs}, Batch size: {batch_size}")

    # Prepare data
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X

    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    # HARDWARE OPTIMIZATION: Move expression matrix to device once
    X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)

    # Create dataloader with STUDY STRATIFICATION for improved generalization
    # This was identified as the critical fix in Phase 3 ablation experiments:
    # - A3 (Study Stratification) achieved 0.79 correlation vs SOTA's 0.11
    # - Prevents model from learning study-specific patterns
    dataset = torch.utils.data.TensorDataset(X_tensor, torch.arange(len(X_tensor), device=device))

    if study_ids is not None:
        # Use StratifiedBatchSampler for study diversity in each batch
        stratified_sampler = StratifiedBatchSampler(
            study_ids=study_ids,
            batch_size=batch_size,
            studies_per_batch=5,  # Each batch contains samples from 5+ studies
            shuffle=True,
            seed=SEED
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_sampler=stratified_sampler
        )
        logger.info("Using STRATIFIED batch sampling for improved generalization")
    else:
        # Fallback to standard random sampling
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        logger.info("Using standard random batch sampling (no study IDs available)")

    # Prepare seed gene expression
    seed_gene_expr = X[:, seed_gene_indices].mean(axis=1).astype(np.float32)
    # HARDWARE OPTIMIZATION: Move to device once
    seed_gene_expr_tensor = torch.from_numpy(seed_gene_expr).to(device)

    # Setup cell-cycle scores for decorrelation
    if s_scores is not None and g2m_scores is not None:
        # HARDWARE OPTIMIZATION: Move to device once
        s_scores_tensor = torch.from_numpy(s_scores.astype(np.float32)).to(device)
        g2m_scores_tensor = torch.from_numpy(g2m_scores.astype(np.float32)).to(device)
        prolif_decorr_fn = ProliferationDecorrelationLoss(weight=1.0)
    else:
        s_scores_tensor = None
        g2m_scores_tensor = None
        prolif_decorr_fn = None

    # Setup erosion indices
    if precomputed_erosion_indices is not None:
        # HARDWARE OPTIMIZATION: Move to device once
        erosion_indices_tensor = torch.from_numpy(precomputed_erosion_indices.astype(np.float32)).to(device)
    else:
        erosion_indices_tensor = None

    # Setup lineage gene indices for erosion computation
    if lineage_gene_indices is None:
        lineage_gene_indices = get_default_lineage_gene_indices(adata.var_names.tolist())

    # Setup rare state protection
    contrastive_loss = ContrastiveMarginLoss(
        margin=1.0,
        rare_sample_indices=rare_sample_indices or [],
        anchor_sample_indices=anchor_sample_indices or []
    )

    # Load manifold neighbors and initialize Vectorized Manifold Loss
    _manifold_dir = manifold_neighbors_path.parent if manifold_neighbors_path else PROCESSED_DIR
    neighbor_indices, neighbor_weights = load_manifold_neighbors(_manifold_dir)
    if neighbor_indices is not None:
        # VECTORIZED MANIFOLD LOSS: Build sparse Laplacian L = D - W once
        n_samples = neighbor_indices.shape[0]
        k = neighbor_indices.shape[1]

        # Validate neighbor indices are within bounds
        max_idx = neighbor_indices.max().item()
        if max_idx >= n_samples:
            raise ValueError(
                f"Invalid neighbor index {max_idx} >= {n_samples} samples. "
                "Manifold neighbor file may be corrupted or mismatched with data."
            )
        
        # Create row indices: [0,0,..1,1,..]
        row_indices = torch.arange(n_samples).view(-1, 1).repeat(1, k).view(-1)
        col_indices = neighbor_indices.view(-1)
        weights = neighbor_weights.view(-1)
        
        # Build Adjacency matrix W
        indices = torch.stack([row_indices, col_indices])
        W = torch.sparse_coo_tensor(indices, weights, (n_samples, n_samples), device=device).coalesce()
        
        # Compute Degree matrix D (diagonal)
        d = torch.sparse.sum(W, dim=1).to_dense()
        diag_indices = torch.arange(n_samples, device=device).view(1, -1).repeat(2, 1)
        D = torch.sparse_coo_tensor(diag_indices, d, (n_samples, n_samples), device=device).coalesce()
        
        # Build Laplacian L = D - W
        L = (D - W).coalesce()
        
        manifold_loss_fn = ManifoldLoss(laplacian=L, weight=1.0)
        logger.info(f"Vectorized Manifold loss initialized with sparse Laplacian ({L._nnz()} non-zeros)")
    else:
        manifold_loss_fn = None
        logger.warning("Manifold loss disabled (neighbors not found)")

    # Jacobian regularization for decoder smoothness
    jacobian_reg_fn = JacobianRegularization(n_probes=2, weight=0.1)
    logger.info("Jacobian regularization initialized with 2 Rademacher probes")

    # Setup domain classifier for adversarial training
    if study_ids is not None:
        unique_studies = np.unique(study_ids)
        study_to_idx = {s: i for i, s in enumerate(unique_studies)}
        # HARDWARE OPTIMIZATION: Move to device once
        study_indices = torch.tensor([study_to_idx[s] for s in study_ids], device=device)
        n_studies = len(unique_studies)
        domain_classifier = DomainClassifier(
            latent_dim=1 + model.max_pathways + model.latent_other_dim,
            hidden_dim=256,
            n_studies=n_studies
        ).to(device)
        domain_optimizer = torch.optim.Adam(domain_classifier.parameters(), lr=0.001)
    else:
        domain_classifier = None
        study_indices = None

    # Optimizer - Use higher learning rate for gate parameters to speed up pruning
    gate_params = [p for n, p in model.named_parameters() if "gate.log_alpha" in n]
    other_params = [p for n, p in model.named_parameters() if "gate.log_alpha" not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': learning_rate},
        {'params': gate_params, 'lr': learning_rate * 10.0}  # 10x faster learning for gates
    ], weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )

    # Training history
    history = {
        "train_loss": [],
        "reconstruction_loss": [],
        "kl_loss": [],
        "progression_loss": [],
        "pathway_diversity_loss": [],
        "decoder_orthogonality_loss": [],
        "rare_state_loss": [],
        "pathway_gate_loss": [],
        "pathway_utilization_loss": [],  # NEW
        "erosion_loss": [],
        "gate_penalty": [],
        "proliferation_decorr_loss": [],  # NEW
        "domain_loss": [],
        "progression_correlation": [],
        "active_pathways": [],
        "active_genes": []
    }

    best_loss = float('inf')
    best_model_state = None
    best_epoch = -1

    # Instantiate loss functions once (not in loop)
    dist_loss_fn = ProgressionDistributionLoss(min_variance=0.08, weight=1.0)
    erosion_loss_fn = PrecomputedErosionLoss(weight=1.0)

    # Dynamic Sparsity Controller state
    current_lambda_sparse_multiplier = 1.0
    target_active_genes = 600

    # Training loop
    for epoch in range(n_epochs):
        # Get phase lambdas
        lambdas = get_phase_lambdas(epoch, total_epochs=n_epochs)
        # Apply dynamic multiplier to sparsity
        lambdas["lambda_sparse"] *= current_lambda_sparse_multiplier
        phase_name = lambdas["phase"]

        model.train()
        epoch_losses = {
            "total": 0.0,
            "reconstruction": 0.0,
            "kl": 0.0,
            "progression": 0.0,
            "pathway_diversity": 0.0,
            "decoder_orthogonality": 0.0,
            "rare_state": 0.0,
            "pathway_gate": 0.0,
            "pathway_utilization": 0.0,
            "erosion": 0.0,
            "gate_penalty": 0.0,
            "proliferation_decorr": 0.0,
            "domain": 0.0
        }
        epoch_prog_corr = []
        epoch_active_pathways = []
        epoch_active_genes = []
        n_batches = 0

        for batch_x, batch_indices in dataloader:
            optimizer.zero_grad()

            # Forward pass with temperature annealing
            gate_temperature = get_gate_temperature(epoch, n_epochs)

            # GRADUAL K-ANNEALING: Smoothly reduce active genes in final 30 epochs
            # instead of abrupt Top-500 switch that caused loss discontinuity
            anneal_start = n_epochs - 30
            if epoch >= anneal_start:
                # Linearly anneal k from current active count to 500
                progress = (epoch - anneal_start) / (n_epochs - anneal_start - 1)
                current_active = max(avg_active_genes if epoch > anneal_start else 1012, 500)
                target_k = 500
                k = int(current_active + (target_k - current_active) * progress)
                k = max(k, target_k)
                outputs = model(batch_x, k=k)
            else:
                outputs = model(batch_x, temperature=gate_temperature)

            # GATED RECONSTRUCTION LOSS: Only penalize reconstruction error on active genes
            # This prevents noise genes from forcing the gate to stay open.
            sq_error = (outputs["reconstruction"] - batch_x) ** 2
            # Apply gene mask (batch_size, n_genes)
            gated_sq_error = sq_error * outputs["gene_gate_mask"].unsqueeze(0)
            recon_loss = gated_sq_error.mean()

            # KL divergence for z_other
            kl_loss = compute_kl_divergence_with_free_bits(
                outputs["mu_other"],
                outputs["logvar_other"],
                free_bits=DEFAULT_FREE_BITS
            )

            # Gene gate penalty
            gate_p = compute_gate_penalty(model.gate)

            # Pathway gate penalty (L0 regularization)
            pathway_gate_p = compute_per_sample_l0_penalty(outputs["gate_logits"])

            # Progression loss (seed gene correlation)
            # HARDWARE OPTIMIZATION: Use pre-moved tensor on device
            batch_seed_expr = seed_gene_expr_tensor[batch_indices]
            prog_loss = compute_progression_loss(
                outputs["z_progression"].squeeze(),
                batch_seed_expr
            )
            # Distribution loss (using pre-instantiated loss function)
            dist_loss = dist_loss_fn(outputs["z_progression"])

            # Pathway diversity loss
            path_div_loss = compute_pathway_diversity_loss(
                outputs["z_pathway"],
                outputs["z_progression"],
                temperature=1.0,
                progression_threshold=0.3
            )

            # Pathway orthogonality loss (latent space)
            path_orth_loss = compute_pathway_orthogonality_loss(outputs["z_pathway"])

            # Decoder orthogonality loss (weight space)
            decoder_ortho_loss = compute_decoder_orthogonality_loss(model.decoder_pathway)

            # Erosion loss (using pre-instantiated loss function)
            if erosion_indices_tensor is not None:
                # HARDWARE OPTIMIZATION: Use pre-moved tensor on device
                batch_erosion = erosion_indices_tensor[batch_indices]
                eros_loss = erosion_loss_fn(outputs["z_progression"].squeeze(), batch_erosion)
            else:
                raise ValueError("Model requires precomputed GMM erosion indices. Run identity_erosion.py first.")

            # Rare state protection loss
            rare_loss = contrastive_loss(outputs["z_pathway"], batch_indices)

            # Proliferation decorrelation loss
            if prolif_decorr_fn is not None and lambdas.get("lambda_proliferation", 0) > 0:
                # HARDWARE OPTIMIZATION: Use pre-moved tensor on device
                batch_s = s_scores_tensor[batch_indices]
                batch_g2m = g2m_scores_tensor[batch_indices]
                prolif_loss = prolif_decorr_fn(outputs["z_progression"], batch_s, batch_g2m)
            else:
                prolif_loss = torch.tensor(0.0, device=device)

            # Domain adversarial loss
            if lambdas["lambda_domain"] > 0 and domain_classifier is not None and study_indices is not None:
                batch_study_indices = study_indices[batch_indices.cpu()].to(device)
                z_combined = torch.cat([
                    outputs["z_progression"],
                    outputs["z_pathway"],
                    outputs["z_other"]
                ], dim=-1)

                # Train domain classifier
                domain_logits = domain_classifier(z_combined.detach())
                dom_loss_classifier = F.cross_entropy(domain_logits, batch_study_indices)
                domain_optimizer.zero_grad()
                dom_loss_classifier.backward()
                domain_optimizer.step()

                # Encoder adversarial loss
                domain_logits_enc = domain_classifier(z_combined)
                dom_loss = F.cross_entropy(domain_logits_enc, batch_study_indices)
            else:
                dom_loss = torch.tensor(0.0, device=device)

            # Manifold loss (identity continuity)
            phase_weights = get_phase_loss_weights(epoch)
            if phase_weights["manifold"] > 0 and manifold_loss_fn is not None:
                manifold_loss = manifold_loss_fn(outputs["z_progression"], batch_indices)
            else:
                manifold_loss = torch.tensor(0.0, device=device)

            # Jacobian regularization (decoder smoothness)
            # LOSS IMPROVEMENTS: Compute jacobian_loss only every 10 batches
            if phase_weights["jacobian"] > 0 and n_batches % 10 == 0:
                # Use base decoder output for smoothness (depends only on z_progression)
                base_recon = model.decoder_base(outputs["z_progression"])
                # Restrict to gate-active genes for efficiency
                gene_mask = outputs["gene_gate_mask"]
                jacobian_loss = jacobian_reg_fn(base_recon, outputs["z_progression"], gene_mask)
            else:
                jacobian_loss = torch.tensor(0.0, device=device)

            # Adaptive width entropy regularization (prevent collapse)
            if "adaptive_width_tau" in outputs:
                tau = outputs["adaptive_width_tau"]
                # Entropy bonus: encourage distribution of tau values
                tau_normalized = (tau - 0.1) / (2.0 - 0.1)  # Normalize to [0, 1]
                tau_entropy = -(tau_normalized * torch.log(tau_normalized + 1e-8) +
                               (1 - tau_normalized) * torch.log(1 - tau_normalized + 1e-8)).mean()
                tau_entropy_bonus = -0.01 * tau_entropy  # Negative because we want to maximize entropy
            else:
                tau_entropy_bonus = torch.tensor(0.0, device=device)

            # Combined pathway loss
            combined_pathway_loss = path_div_loss + 0.5 * path_orth_loss

            # Pathway utilization bonus (NEW - prevents collapse)
            pathway_util_bonus = compute_pathway_utilization_bonus(outputs["path_mask"])

            # Total loss with stability guards
            # We add all terms. Bonus terms like pathway_util_bonus are negative (higher is better),
            # so adding them reduces total_loss.
            total_loss = (
                recon_loss +
                lambdas["lambda_kl"] * kl_loss +
                lambdas["lambda_progression"] * prog_loss +
                lambdas["lambda_distribution"] * dist_loss +
                lambdas["lambda_diversity"] * combined_pathway_loss +
                lambdas["lambda_pathway_ortho"] * decoder_ortho_loss +
                lambdas["lambda_rare"] * rare_loss +
                lambdas["lambda_pathway_gate"] * pathway_gate_p +
                lambdas.get("lambda_pathway_util", 0.0) * pathway_util_bonus +  # Bonus is negative, so we ADD it to reduce loss
                lambdas["lambda_erosion"] * eros_loss +
                lambdas["lambda_sparse"] * gate_p +
                lambdas.get("lambda_proliferation", 0.0) * prolif_loss +  # NEW
                phase_weights["manifold"] * manifold_loss +  # Manifold smoothness
                phase_weights["jacobian"] * jacobian_loss +  # Decoder smoothness
                tau_entropy_bonus  # Adaptive width entropy
            )

            # Domain adversarial loss (subtracted to maximize it for the encoder)
            # We clamp it to prevent it from dominating the loss and going to -infinity
            if lambdas["lambda_domain"] > 0:
                total_loss = total_loss - lambdas["lambda_domain"] * torch.clamp(dom_loss, max=10.0)


            # Debug: Check each loss component for NaN (first batch of first epoch only)
            if n_batches == 0 and epoch == 0:
                loss_components = {
                    "recon_loss": recon_loss,
                    "kl_loss": kl_loss,
                    "prog_loss": prog_loss,
                    "dist_loss": dist_loss,
                    "combined_pathway_loss": combined_pathway_loss,
                    "decoder_ortho_loss": decoder_ortho_loss,
                    "rare_loss": rare_loss,
                    "pathway_gate_p": pathway_gate_p,
                    "pathway_util_bonus": pathway_util_bonus,
                    "eros_loss": eros_loss,
                    "gate_p": gate_p,
                    "dom_loss": dom_loss
                }
                logger.info("=== Loss Component Values (First Batch) ===")
                for name, loss_val in loss_components.items():
                    val = loss_val.item() if hasattr(loss_val, 'item') else loss_val
                    is_nan = math.isnan(val) if isinstance(val, float) else torch.isnan(loss_val).item()
                    logger.info(f"  {name}: {val:.6f} (NaN: {is_nan})")
                logger.info("=== Lambda Values ===")
                for key, val in lambdas.items():
                    if key != 'phase':
                        logger.info(f"  {key}: {val}")

            # Guard against NaN in total loss
            if torch.isnan(total_loss):
                logger.warning(f"NaN in total loss, skipping batch")
                continue

            # Backward pass
            total_loss.backward()

            # Check gradients for NaN
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    logger.warning(f"NaN gradient in: {name}")
                    has_nan_grad = True
                    break

            if has_nan_grad:
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate losses
            epoch_losses["total"] += total_loss.item()
            epoch_losses["reconstruction"] += recon_loss.item()
            epoch_losses["kl"] += kl_loss.item()
            epoch_losses["progression"] += prog_loss.item()
            epoch_losses["pathway_diversity"] += path_div_loss.item()
            epoch_losses["decoder_orthogonality"] += decoder_ortho_loss.item()
            epoch_losses["rare_state"] += rare_loss.item()
            epoch_losses["pathway_gate"] += pathway_gate_p.item()
            epoch_losses["pathway_utilization"] += pathway_util_bonus.item()  # NEW
            epoch_losses["erosion"] += eros_loss.item()
            epoch_losses["gate_penalty"] += gate_p.item()
            epoch_losses["proliferation_decorr"] += prolif_loss.item()
            epoch_losses["domain"] += dom_loss.item()
            n_batches += 1

            # Track metrics
            with torch.no_grad():
                # Progression correlation (with NaN guard)
                z_prog_1d = outputs["z_progression"].squeeze()

                # Skip correlation computation if there's NaN in inputs
                if torch.isnan(z_prog_1d).any() or torch.isnan(batch_seed_expr).any():
                    prog_corr = torch.tensor(0.0)
                else:
                    z_centered = z_prog_1d - z_prog_1d.mean()
                    seed_centered = batch_seed_expr - batch_seed_expr.mean()
                    z_std = z_centered.std()
                    seed_std = seed_centered.std()
                    if z_std < 1e-8 or seed_std < 1e-8:
                        prog_corr = torch.tensor(0.0)
                    else:
                        prog_corr = (z_centered * seed_centered).mean() / (z_std * seed_std)
                        prog_corr = torch.clamp(prog_corr, min=-1.0, max=1.0)
                epoch_prog_corr.append(prog_corr.item())

                # Active pathways
                active_pathways = model.get_active_pathway_count()
                epoch_active_pathways.append(active_pathways)

                # Active genes
                gene_probs = model.gate.get_gates()
                active_genes = (gene_probs > 0.1).sum().item()
                epoch_active_genes.append(active_genes)

        # Average losses (guard against empty dataloader)
        if n_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= n_batches

        avg_prog_corr = np.mean(epoch_prog_corr)
        avg_active_pathways = np.mean(epoch_active_pathways)
        avg_active_genes = np.mean(epoch_active_genes)

        # Record history
        history["train_loss"].append(epoch_losses["total"])
        history["reconstruction_loss"].append(epoch_losses["reconstruction"])
        history["kl_loss"].append(epoch_losses["kl"])
        history["progression_loss"].append(epoch_losses["progression"])
        history["pathway_diversity_loss"].append(epoch_losses["pathway_diversity"])
        history["decoder_orthogonality_loss"].append(epoch_losses["decoder_orthogonality"])
        history["rare_state_loss"].append(epoch_losses["rare_state"])
        history["pathway_gate_loss"].append(epoch_losses["pathway_gate"])
        history["pathway_utilization_loss"].append(epoch_losses["pathway_utilization"])  # NEW
        history["erosion_loss"].append(epoch_losses["erosion"])
        history["gate_penalty"].append(epoch_losses["gate_penalty"])
        history["proliferation_decorr_loss"].append(epoch_losses["proliferation_decorr"])
        history["domain_loss"].append(epoch_losses["domain"])
        history["progression_correlation"].append(avg_prog_corr)
        history["active_pathways"].append(avg_active_pathways)
        history["active_genes"].append(avg_active_genes)

        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history.setdefault("learning_rate", []).append(current_lr)

        # Track best loss for convergence diagnostics
        history.setdefault("best_loss", []).append(best_loss)

        # Learning rate scheduling
        scheduler.step(epoch_losses["reconstruction"])

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            gene_probs = model.gate.get_gates()
            active_genes = (gene_probs > 0.1).sum().item()
            pathway_probs = model.pathway_gate.get_gates()
            active_pathways = (pathway_probs > 0.5).sum().item()

            logger.info(
                f"Epoch {epoch + 1}/{n_epochs} [{phase_name}] - "
                f"Loss: {epoch_losses['total']:.4f} - "
                f"Recon: {epoch_losses['reconstruction']:.4f} - "
                f"KL: {epoch_losses['kl']:.4f} - "
                f"ProgCorr: {avg_prog_corr:.4f} - "
                f"ActivePathways: {active_pathways}/{model.max_pathways} - "
                f"ActiveGenes: {active_genes}/{model.n_genes} - "
                f"LR: {current_lr:.6f}"
            )

        # DYNAMIC SPARSITY CONTROLLER: Increase penalty if target not met
        # CAP the multiplier to prevent gradient explosion (max 10x)
        if phase_name in ['B', 'C'] and (epoch + 1) % 10 == 0:
            if avg_active_genes > target_active_genes and current_lambda_sparse_multiplier < 4.0:
                current_lambda_sparse_multiplier = min(4.0, current_lambda_sparse_multiplier * 2.0)
                logger.info(f"Dynamic Controller: Active genes ({avg_active_genes:.0f}) > Target ({target_active_genes}). "
                            f"Increasing sparsity multiplier to {current_lambda_sparse_multiplier:.1f}")
            elif avg_active_genes < target_active_genes * 0.5:
                current_lambda_sparse_multiplier = max(1.0, current_lambda_sparse_multiplier * 0.5)

        # Save best model
        stable_metric = (
            epoch_losses["reconstruction"] +
            lambdas["lambda_kl"] * epoch_losses["kl"]
        )
        if not (math.isnan(stable_metric) or math.isinf(stable_metric)) and stable_metric < best_loss:
            best_loss = stable_metric
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model from epoch {best_epoch + 1} (metric={best_loss:.6f})")

    # Log final pathway gate probabilities
    pathway_probs = model.pathway_gate.get_gates().detach().cpu().numpy()
    logger.info(f"Final pathway gate probabilities: {pathway_probs}")
    logger.info(f"Active pathways (prob > 0.5): {(pathway_probs > 0.5).sum()}")

    history["best_epoch"] = best_epoch + 1 if best_epoch >= 0 else -1

    return model, history


def extract_ne_scores(
    model: NESTORModel,
    adata: ad.AnnData,
    output_path: Path | None = None
) -> pd.DataFrame:
    """
    Extract NE scores from trained model.

    Extracts NE progression and pathway scores from trained model.

    Args:
        model: Trained NESTORModel
        adata: AnnData with expression data
        output_path: Optional path to save scores as parquet

    Returns:
        DataFrame with NE scores indexed by sample ID
        For v1: 'ne_score' column
        For v2: 'NE_state_raw', 'NE_potential_raw' columns
        Returns: NE_progression, z_pathway_*, z_other_* columns
    """
    device = get_device()
    model = model.to(device)
    model.eval()

    # Prepare data - handle NaN values
    X_np = np.asarray(adata.X)
    X_np = np.nan_to_num(X_np, nan=0.0)
    X = torch.tensor(X_np, dtype=torch.float32).to(device)

    # Check model type and extract accordingly
    if isinstance(model, NESTORModel):
        # NESTORModel
        return extract_scores(model, X, adata.obs.index, output_path)
    elif hasattr(model, 'get_ne_scores'):
        # V1 or V2 model
        with torch.no_grad():
            ne_scores = model.get_ne_scores(X)

        # Handle v1 vs v2 output format
        if isinstance(ne_scores, dict) and "NE_state" in ne_scores:
            # v2 model - dual axis
            scores_df = pd.DataFrame({
                "NE_state_raw": ne_scores["NE_state"].cpu().numpy().flatten(),
                "NE_potential_raw": ne_scores["NE_potential"].cpu().numpy().flatten()
            }, index=adata.obs.index)
            logger.info(f"Extracted v2 dual-axis scores: {len(scores_df)} samples")
        else:
            # v1 model - single axis
            scores_df = pd.DataFrame({
                "ne_score": ne_scores.cpu().numpy().flatten()
            }, index=adata.obs.index)
            logger.info(f"Extracted v1 single-axis scores: {len(scores_df)} samples")
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    # Save if path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_parquet(output_path)
        logger.info(f"Saved NE scores to {output_path}")

    return scores_df


def extract_scores(
    model: NESTORModel,
    X: torch.Tensor,
    sample_index: pd.Index,
    output_path: Path | None = None
) -> pd.DataFrame:
    """
    Extract scores from NESTORModel.

    Args:
        model: Trained NESTORModel
        X: Expression tensor on device
        sample_index: Sample index for the output DataFrame
        output_path: Optional path to save scores as parquet

    Returns:
        DataFrame with:
            - NE_progression: Progression score [0, 1]
            - z_pathway_0, z_pathway_1, ...: Pathway latent dimensions
            - z_other_0, z_other_1, ...: Other latent dimensions
    """
    n_samples = X.shape[0]
    # Handle different model versions with different pathway dimension attributes
    pathway_dim = (
        getattr(model, 'latent_pathway_dim', None) or
        getattr(model, 'latent_pathway_dim_max', None) or
        getattr(model, 'max_pathways', 4)
    )
    other_dim = model.latent_other_dim

    # Initialize arrays
    progression_scores = np.zeros(n_samples)
    pathway_scores = np.zeros((n_samples, pathway_dim))
    other_scores = np.zeros((n_samples, other_dim))

    # Extract in batches to avoid memory issues
    batch_size = 512
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_x = X[i:i + batch_size]
            outputs = model(batch_x)

            progression_scores[i:i + batch_size] = outputs["z_progression"].cpu().numpy().flatten()
            pathway_scores[i:i + batch_size] = outputs["z_pathway"].cpu().numpy()
            other_scores[i:i + batch_size] = outputs["z_other"].cpu().numpy()

    # Build DataFrame
    scores_dict = {
        "NE_progression": progression_scores
    }

    # Add pathway dimensions
    for j in range(pathway_dim):
        scores_dict[f"z_pathway_{j}"] = pathway_scores[:, j]

    # Add other latent dimensions
    for j in range(other_dim):
        scores_dict[f"z_other_{j}"] = other_scores[:, j]

    scores_df = pd.DataFrame(scores_dict, index=sample_index)

    logger.info(f"Extracted scores: {len(scores_df)} samples")
    logger.info(f"  NE_progression range: [{progression_scores.min():.4f}, {progression_scores.max():.4f}]")

    # Save if path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_parquet(output_path)
        logger.info(f"Saved scores to {output_path}")

    return scores_df


def main() -> int:
    """Main entry point for NESTOR training."""
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train NESTOR model (dual-axis with weak supervision)")
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=500,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate"
    )
    parser.add_argument(
        "--anchor-quantile",
        type=float,
        default=0.8,
        help="Quantile threshold for anchor selection (default 0.8 = top 20%%)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to per-sample z-scored expression AnnData (.h5ad)"
    )
    parser.add_argument(
        "--seed-genes",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "ne_seed_genes.txt",
        help="Path to NE seed-gene list (one gene per line)"
    )
    parser.add_argument(
        "--erosion",
        type=Path,
        default=None,
        help="Path to precomputed erosion indices (.parquet). Omit to skip erosion loss."
    )
    parser.add_argument(
        "--cellcycle-scores",
        type=Path,
        default=None,
        help="Path to cell-cycle scores (.parquet). Omit to skip proliferation decorrelation."
    )
    parser.add_argument(
        "--manifold-neighbors",
        type=Path,
        default=None,
        help="Path to manifold neighbour graph (.parquet). Omit to skip manifold loss."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "models" / "nestor",
        help="Output directory for model checkpoint and config"
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=Path.cwd() / "outputs" / "nestor_scores_raw.parquet",
        help="Output path for raw per-sample NE scores (.parquet)"
    )
    args = parser.parse_args()

    # Set reproducibility
    set_reproducibility(SEED)

    logger.info("Starting NESTOR model training")

    # Load input data - use per-sample z-scored data (removes sample mean confounding)
    input_path = args.input
    logger.info("Using per-sample z-scored data (each sample mean=0, std=1 - removes global transcriptional confounding)")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Please run the data preprocessing pipeline first to generate the input file.")
        return 1

    logger.info(f"Loading data from {input_path}")
    adata = ad.read_h5ad(input_path)

    # Check for empty data
    if adata.n_obs == 0 or adata.n_vars == 0:
        logger.error("Input AnnData is empty")
        return 1

    logger.info(f"Loaded AnnData: {adata.n_obs} samples, {adata.n_vars} genes")

    # Load seed genes
    seed_genes = load_seed_genes(args.seed_genes)

    # Map seed genes to indices in the data
    gene_to_idx = {g: i for i, g in enumerate(adata.var.index)}
    seed_gene_indices = [gene_to_idx[g] for g in seed_genes if g in gene_to_idx]
    logger.info(f"Found {len(seed_gene_indices)} seed genes in data")

    # Get study IDs if available
    study_ids = None
    if "study_id" in adata.obs.columns:
        study_ids = adata.obs["study_id"].values
        logger.info(f"Found {len(np.unique(study_ids))} unique studies for domain adversarial training")

    # Load precomputed erosion indices
    erosion_indices = None
    if args.erosion is not None:
        erosion_path = args.erosion
        if erosion_path.exists():
            erosion_data = pd.read_parquet(erosion_path)
            # Map erosion indices to samples in adata
            sample_erosion_map = erosion_data.set_index('sample_id')['erosion_index'].to_dict()
            erosion_indices = np.array([
                sample_erosion_map.get(sid, 0.0) for sid in adata.obs.index
            ])
            # Handle NaN values (samples without tissue-specific markers)
            nan_count = np.isnan(erosion_indices).sum()
            if nan_count > 0:
                logger.info(f"Filling {nan_count} NaN erosion indices with 0.0")
                erosion_indices = np.nan_to_num(erosion_indices, nan=0.0)
            n_with_erosion = (erosion_indices > 0).sum()
            logger.info(f"Loaded erosion indices: {n_with_erosion}/{len(erosion_indices)} samples with tissue-specific markers")
            logger.info(f"Erosion index range: [{erosion_indices.min():.4f}, {erosion_indices.max():.4f}]")
        else:
            logger.warning(f"Erosion indices file not found: {erosion_path}")
            logger.warning("Training will proceed without erosion loss. Run scripts/identity_erosion.py first.")

    # Load cell-cycle scores for decorrelation
    s_scores = None
    g2m_scores = None
    cc_path = args.cellcycle_scores
    if cc_path is not None and cc_path.exists():
        cc_data = pd.read_parquet(cc_path)
        # Ensure alignment with adata using unique_sample_id
        sample_to_s = cc_data.set_index('unique_sample_id')['S_score'].to_dict()
        sample_to_g2m = cc_data.set_index('unique_sample_id')['G2M_score'].to_dict()
        
        # adata index should match unique_sample_id if processed correctly
        s_scores = np.array([sample_to_s.get(sid, 0.0) for sid in adata.obs.index])
        g2m_scores = np.array([sample_to_g2m.get(sid, 0.0) for sid in adata.obs.index])
        
        # LOSS IMPROVEMENTS: Pre-standardize scores to Mean=0, Std=1
        s_scores = (s_scores - np.mean(s_scores)) / (np.std(s_scores) + 1e-8)
        g2m_scores = (g2m_scores - np.mean(g2m_scores)) / (np.std(g2m_scores) + 1e-8)
        
        logger.info(f"Loaded and standardized cell-cycle scores for {len(s_scores)} samples")
    else:
        logger.warning(f"Cell-cycle scores not found at {cc_path}. Decorrelation disabled.")

    # Model configuration
    n_genes = adata.n_vars
    hidden_dim = DEFAULT_HIDDEN_DIM
    latent_other_dim = DEFAULT_LATENT_OTHER_DIM  # IMPROVE: Use constant (16)
    latent_state_dim = DEFAULT_LATENT_STATE_DIM
    latent_potential_dim = DEFAULT_LATENT_POTENTIAL_DIM
    gate_temperature = DEFAULT_GATE_TEMPERATURE
    gate_init_mean = DEFAULT_GATE_INIT_MEAN

    # Build config
    config = {
        "n_genes": n_genes,
        "hidden_dim": hidden_dim,
        "latent_other_dim": latent_other_dim,
        "latent_state_dim": latent_state_dim,
        "latent_potential_dim": latent_potential_dim,
        "kl_weight": DEFAULT_KL_WEIGHT,
        "kl_anneal_epochs": DEFAULT_KL_ANNEAL_EPOCHS,
        "free_bits": DEFAULT_FREE_BITS,
        "diversity_weight": DEFAULT_DIVERSITY_WEIGHT,
        "warmup_epochs": DEFAULT_WARMUP_EPOCHS,  # IMPROVE: Track warmup
        "gate_temperature": gate_temperature,
        "gate_init_mean": gate_init_mean,
        "n_samples": adata.n_obs,
        "n_seed_genes": len(seed_gene_indices),
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "anchor_quantile": args.anchor_quantile,
        "training_timestamp": datetime.now().isoformat(),
        # Model architecture config keys
        "adaptive_width": True,
        "tau_min": 0.1,
        "tau_max": 2.0,
        "manifold_neighbors_path": str(args.manifold_neighbors) if args.manifold_neighbors else str(PROCESSED_DIR / "manifold_neighbors.parquet"),
        "manifold_loss_weight": 1.0,
        "jacobian_loss_weight": 0.1,
        "jacobian_n_probes": 2,
    }

    # Compute seed scores S0 for anchor selection
    logger.info("Computing seed scores for anchor selection...")
    seed_scores = compute_seed_score_s0(adata, seed_genes)
    logger.info(f"Seed score range: [{seed_scores.min():.4f}, {seed_scores.max():.4f}]")

    # Select anchors using global threshold (returns list of anchor sample IDs)
    anchor_ids = select_anchors_global_threshold(
        seed_scores,
        quantile=args.anchor_quantile
    )

    # Convert to boolean mask
    anchor_mask = pd.Series(False, index=adata.obs.index)
    anchor_mask.loc[anchor_ids] = True
    anchor_mask = anchor_mask.values  # Convert to numpy array

    # Compute anchor weights (confidence weights based on distance from threshold)
    threshold = seed_scores.quantile(args.anchor_quantile)
    anchor_weights = np.zeros(len(adata))
    if seed_scores.max() > threshold:
        anchor_weights[anchor_mask] = (seed_scores.values[anchor_mask] - threshold) / (seed_scores.max() - threshold)
    anchor_weights = np.clip(anchor_weights, 0, 1)  # Normalize to [0, 1]

    logger.info(f"Selected {anchor_mask.sum()} anchors at threshold {threshold:.4f}")
    logger.info(f"Anchor quantile: {args.anchor_quantile} (top {(1-args.anchor_quantile)*100:.0f}%)")

    # Store anchor info in adata
    adata.obs["anchor_status"] = "UNLABELED"
    adata.obs.loc[anchor_mask, "anchor_status"] = "ANCHOR"
    adata.obs["seed_score_s0"] = seed_scores

    # Initialize model
    latent_pathway_dim_max = DEFAULT_LATENT_PATHWAY_DIM_MAX
    logger.info("Initializing NESTORModel (L0-gated pathway space)")
    logger.info(f"  Architecture: hidden_dim={hidden_dim}, total_latent={1 + latent_pathway_dim_max + latent_other_dim}D (max)")
    logger.info(f"    z_progression=1D [0,1], z_pathway={latent_pathway_dim_max}D (learns D*), z_other={latent_other_dim}D")
    logger.info(f"  Tunnel Gate: Output = f(z_prog) + [sigmoid(z_prog) * g(z_path * Gate_L0)] + h(z_other)")
    logger.info(f"  L0 Pathway Gate: temperature={DEFAULT_PATHWAY_GATE_TEMPERATURE}, init_mean={DEFAULT_PATHWAY_GATE_INIT_MEAN}")
    logger.info(f"  Regularization: kl_weight={DEFAULT_KL_WEIGHT}, free_bits={DEFAULT_FREE_BITS}")
    logger.info(f"  Gates: temperature={gate_temperature}, init_mean={gate_init_mean}")

    model = NESTORModel(
        n_genes=n_genes,
        hidden_dim=hidden_dim,
        max_pathways=latent_pathway_dim_max,
        latent_other_dim=latent_other_dim,
        gate_temperature=gate_temperature,
        gate_init_mean=gate_init_mean
    )

    # Identify Amphicrine-like samples for rare state protection
    rare_indices, anchor_indices = identify_amphicrine_samples(adata)
    if rare_indices:
        logger.info(f"  Rare state protection: {len(rare_indices)} Amphicrine-like samples identified")

    # Update config
    config["max_pathways"] = latent_pathway_dim_max
    config["total_latent_dim"] = 1 + latent_pathway_dim_max + latent_other_dim
    config["tunnel_gate"] = True
    config["l0_pathway_gate"] = True
    config["decoder_architecture"] = "l0_gated_multiplicative"
    config["pathway_gate_temperature"] = DEFAULT_PATHWAY_GATE_TEMPERATURE
    config["pathway_gate_init_mean"] = DEFAULT_PATHWAY_GATE_INIT_MEAN
    config["rare_sample_count"] = len(rare_indices)

    # Train model
    logger.info("Starting training with L0-gated pathway discovery")
    trained_model, history = train_model(
        model,
        adata,
        seed_gene_indices,
        s_scores=s_scores,
        g2m_scores=g2m_scores,
        precomputed_erosion_indices=erosion_indices,
        rare_sample_indices=rare_indices,
        anchor_sample_indices=anchor_indices,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        study_ids=study_ids,
        manifold_neighbors_path=args.manifold_neighbors
    )

    # Save model
    output_dir = args.output_dir

    # Save model checkpoint
    save_model_checkpoint(trained_model, output_dir, config, history)

    # Extract and save NE scores
    raw_scores_path = args.scores_output
    scores_df = extract_ne_scores(trained_model, adata, raw_scores_path)

    logger.info(f"Training complete. Model saved to {output_dir}")
    logger.info(f"Raw NE scores saved to {raw_scores_path}")
    logger.info(f"NE_progression range: [{scores_df['NE_progression'].min():.4f}, {scores_df['NE_progression'].max():.4f}]")
    logger.info("Run 08_export_scores_signature.py to generate finalized scores with metadata")

    return 0


if __name__ == "__main__":
    sys.exit(main())


