"""NESTOR inference module.

Provides NESTORInference class for loading trained NESTOR models and
running inference to obtain NE progression scores.

Usage:
    from scripts.nestor_inference import NESTORInference

    inference = NESTORInference(model_dir, gene_file)
    result = inference.predict(expr_matrix, expr_genes=gene_names)
    z_progression = result["z_progression"]
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def get_device() -> torch.device:
    """Get the best available device: MPS (Mac), CUDA (NVIDIA), or CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model_class(script_path: Path):
    """Dynamically load NESTORModel class from training script.

    Args:
        script_path: Path to train_nestor.py

    Returns:
        NESTORModel class
    """
    spec = importlib.util.spec_from_file_location("train_nestor", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_nestor"] = module
    spec.loader.exec_module(module)
    return module.NESTORModel


class NESTORInference:
    """NESTOR model inference class.

    Loads trained NESTOR model and provides methods for running inference
    on gene expression data to obtain NE progression scores.

    Attributes:
        model_dir: Directory containing model.pt and config.json
        gene_file: Path to file containing model gene order
        config: Model configuration dictionary
        model_genes: List of gene symbols in model order
        model: Loaded NESTORModel instance
        device: Torch device for inference
    """

    def __init__(self, model_dir: Path | str, gene_file: Path | str):
        """Initialize NESTOR inference.

        Args:
            model_dir: Directory containing model.pt and config.json
            gene_file: Path to file with gene symbols (one per line)

        Raises:
            FileNotFoundError: If model_dir or gene_file doesn't exist
            ValueError: If config or model loading fails
        """
        self.model_dir = Path(model_dir)
        self.gene_file = Path(gene_file)

        # Validate paths exist
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        if not self.gene_file.exists():
            raise FileNotFoundError(f"Gene file not found: {self.gene_file}")

        # Load config
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            self.config = json.load(f)

        # Load gene list
        with open(self.gene_file) as f:
            self.model_genes = [line.strip() for line in f if line.strip()]

        # Validate gene count matches config
        if len(self.model_genes) != self.config["n_genes"]:
            raise ValueError(
                f"Gene count mismatch: {len(self.model_genes)} genes in file, "
                f"but config specifies {self.config['n_genes']}"
            )

        # Get device
        self.device = get_device()

        # Load model class dynamically from training script
        script_path = Path(__file__).resolve().parent / "train_nestor.py"
        NESTORModel = load_model_class(script_path)

        # Initialize model
        self.model = NESTORModel(
            n_genes=self.config["n_genes"],
            hidden_dim=self.config["hidden_dim"],
            max_pathways=self.config["max_pathways"],
            latent_other_dim=self.config["latent_other_dim"],
            gate_temperature=self.config.get("gate_temperature", 0.5),
            gate_init_mean=self.config.get("gate_init_mean", 0.3)
        )

        # Load weights
        model_path = self.model_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Handle both direct state_dict and checkpoint format
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

    def align_to_model_genes(
        self,
        expr_matrix: np.ndarray,
        expr_genes: list[str]
    ) -> np.ndarray:
        """Align expression matrix to model gene order.

        Takes expression data with arbitrary gene order and realigns it to
        match the gene order expected by the model. Missing genes are filled
        with zeros. Extra genes not in the model are ignored.

        Args:
            expr_matrix: Expression matrix of shape (n_input_genes, n_samples)
            expr_genes: List of gene symbols corresponding to expr_matrix rows

        Returns:
            Aligned matrix of shape (n_samples, n_model_genes)
        """
        n_samples = expr_matrix.shape[1]
        n_model_genes = len(self.model_genes)

        # Create output matrix initialized with zeros
        aligned = np.zeros((n_samples, n_model_genes), dtype=np.float32)

        # Build mapping from model gene to input index
        expr_gene_to_idx = {gene: idx for idx, gene in enumerate(expr_genes)}

        # Fill in values for genes that exist in both
        for model_idx, model_gene in enumerate(self.model_genes):
            if model_gene in expr_gene_to_idx:
                input_idx = expr_gene_to_idx[model_gene]
                aligned[:, model_idx] = expr_matrix[input_idx, :]

        return aligned

    def predict(
        self,
        expr_matrix: np.ndarray,
        expr_genes: Optional[list[str]] = None,
        batch_size: int = 1024
    ) -> dict[str, np.ndarray]:
        """Run NESTOR inference on expression data.

        Args:
            expr_matrix: Expression matrix of shape (n_genes, n_samples)
                If expr_genes is None, assumes genes are in model order
            expr_genes: Optional list of gene symbols. If provided, will
                align expression to model gene order
            batch_size: Number of samples to process per batch

        Returns:
            Dictionary containing:
                - z_progression: NE progression scores [0, 1], shape (n_samples,)
                - z_pathway: Pathway coordinates, shape (n_samples, max_pathways)
        """
        # Align genes if needed
        if expr_genes is not None:
            aligned = self.align_to_model_genes(expr_matrix, expr_genes)
        else:
            # Assume genes are in correct order, just transpose
            aligned = expr_matrix.T.astype(np.float32)

        n_samples = aligned.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Collect results
        z_progression_list = []
        z_pathway_list = []

        with torch.no_grad():
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                batch = torch.tensor(
                    aligned[start_idx:end_idx],
                    dtype=torch.float32,
                    device=self.device
                )

                # Run forward pass
                output = self.model(batch)

                # Collect outputs
                z_progression_list.append(
                    output["z_progression"].cpu().numpy().flatten()
                )
                z_pathway_list.append(
                    output["z_pathway"].cpu().numpy()
                )

        # Concatenate batches
        z_progression = np.concatenate(z_progression_list)
        z_pathway = np.concatenate(z_pathway_list)

        return {
            "z_progression": z_progression,
            "z_pathway": z_pathway
        }


def main():
    """Command-line interface for NESTOR inference."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run NESTOR inference on gene expression data"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/nestor"),
        help="Directory containing model.pt and config.json"
    )
    parser.add_argument(
        "--gene-file",
        type=Path,
        default=Path("model_genes.txt"),
        help="File containing model gene order"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input expression file (NPZ or NPY format)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file for NE scores (NPZ format)"
    )

    args = parser.parse_args()

    # Load input
    if args.input.suffix == ".npz":
        data = np.load(args.input)
        expr_matrix = data["expr_matrix"]
        expr_genes = data["genes"].tolist() if "genes" in data else None
    else:
        expr_matrix = np.load(args.input)
        expr_genes = None

    # Initialize inference
    print(f"Loading model from {args.model_dir}")
    inference = NESTORInference(args.model_dir, args.gene_file)
    print(f"Model loaded with {len(inference.model_genes)} genes on {inference.device}")

    # Run inference
    print(f"Running inference on {expr_matrix.shape[1]} samples")
    result = inference.predict(expr_matrix, expr_genes=expr_genes)

    # Save output
    np.savez(
        args.output,
        z_progression=result["z_progression"],
        z_pathway=result["z_pathway"]
    )
    print(f"Saved results to {args.output}")

    # Print summary statistics
    print(f"\nz_progression statistics:")
    print(f"  Min: {result['z_progression'].min():.4f}")
    print(f"  Max: {result['z_progression'].max():.4f}")
    print(f"  Mean: {result['z_progression'].mean():.4f}")
    print(f"  Std: {result['z_progression'].std():.4f}")


if __name__ == "__main__":
    main()
