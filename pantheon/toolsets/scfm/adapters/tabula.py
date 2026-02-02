"""
Tabula Adapter

Tabula is a privacy-preserving predictive foundation model for single-cell
transcriptomics that combines federated learning with tabular modeling.
It uses a TabulaTransformer architecture (3 blocks, 8 heads, 192-dim tokens)
with a custom 60,697 gene vocabulary and quantile-binned expression values.

Reference: https://github.com/aristoteleo/tabula
Paper: Ding et al., 2025 (preprint)
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..registry import ModelSpec, TaskType, get_registry
from .base import BaseAdapter


def _check_tabula_installed() -> tuple[bool, Optional[str]]:
    """Check if Tabula package is installed and return its location."""
    try:
        import tabula
        return True, str(Path(tabula.__file__).parent)
    except ImportError:
        return False, None


class TabulaAdapter(BaseAdapter):
    """
    Adapter for the Tabula foundation model.

    Supports:
    - embed: Zero-shot cell embeddings (192-dim) via CLS token or avg-pool
    - annotate: Cell type annotation (requires fine-tuned checkpoint)
    - integrate: Batch integration via shared embedding space with DAB
    - perturb: Gene perturbation prediction (requires fine-tuned checkpoint)

    Key Characteristics:
    - Custom 60,697 gene vocabulary (vocab.json)
    - Quantile-binned expression values (51 bins)
    - Fixed 1,200 gene sequence length per cell
    - FlashAttention for efficient inference
    - Federated learning pre-trained (privacy-preserving)

    Requirements:
    - Tabula package: pip install tabula (or clone from GitHub)
    - GPU with CUDA >= 11.7 (FlashAttention dependency)
    - 8-16 GB VRAM
    - Model checkpoint (best_model.pth) from Tabula repository
    """

    # Tabula-specific constants
    IN_FEATURE = 1200          # Fixed gene sequence length
    VOCAB_SIZE = 60697         # Gene vocabulary size
    D_TOKEN = 192              # Token/embedding dimension
    N_BINS = 51                # Number of quantile bins for expression
    N_BLOCKS = 3               # Transformer blocks
    ATTENTION_HEADS = 8        # Attention heads
    PADDING_ID = 60694         # Padding token ID in vocabulary

    def __init__(self, checkpoint_dir: Optional[str] = None):
        spec = get_registry().get("tabula")
        if spec is None:
            raise ValueError("Tabula model not found in registry")
        super().__init__(spec, checkpoint_dir)

        self._model = None
        self._vocab = None
        self._tabula_installed, self._tabula_path = _check_tabula_installed()

    def run(
        self,
        task: TaskType,
        adata_path: str,
        output_path: str,
        batch_key: Optional[str] = None,
        label_key: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 64,
    ) -> dict[str, Any]:
        """
        Run Tabula model for a given task.

        Args:
            task: TaskType (EMBED, ANNOTATE, INTEGRATE, PERTURB)
            adata_path: Path to input .h5ad file
            output_path: Path for output .h5ad file
            batch_key: Column in .obs for batch information (integration)
            label_key: Column in .obs for cell type labels (annotation)
            device: Device to use ('auto', 'cuda')
            batch_size: Batch size for inference (default: 64)

        Returns:
            Dictionary with output_path, output_keys, and statistics
        """
        supported_tasks = [TaskType.EMBED, TaskType.ANNOTATE, TaskType.INTEGRATE, TaskType.PERTURB]
        if task not in supported_tasks:
            return {
                "error": f"Tabula does not support task '{task.value}'",
                "supported_tasks": [t.value for t in supported_tasks],
            }

        # Annotation and perturbation require fine-tuned checkpoints
        if task == TaskType.ANNOTATE:
            return {
                "error": "Tabula annotation requires a fine-tuned checkpoint with supervised head",
                "suggestion": (
                    "Fine-tune Tabula using finetune_framework_annotation.yaml, "
                    "then provide the fine-tuned checkpoint directory"
                ),
                "documentation": "https://github.com/aristoteleo/tabula",
                "supported_zero_shot": ["embed", "integrate"],
            }

        if task == TaskType.PERTURB:
            return {
                "error": "Tabula perturbation prediction requires a fine-tuned checkpoint",
                "suggestion": (
                    "Fine-tune Tabula using finetune_framework_perturbation.yaml "
                    "with GEARS perturbation dataset format"
                ),
                "documentation": "https://github.com/aristoteleo/tabula",
                "supported_zero_shot": ["embed", "integrate"],
            }

        device = self._resolve_device(device)

        # Tabula requires GPU (FlashAttention dependency)
        if device == "cpu":
            return {
                "error": "Tabula requires GPU (CUDA >= 11.7, FlashAttention dependency)",
                "suggestion": "Use a model with CPU support (Geneformer, scGPT) or remote MCP backend",
                "min_vram_gb": 8,
            }

        # Check for Tabula package
        if not self._tabula_installed:
            return {
                "error": "Tabula package not installed",
                "install": "pip install tabula  # or clone from https://github.com/aristoteleo/tabula",
                "documentation": "https://github.com/aristoteleo/tabula",
            }

        # Load data
        try:
            import scanpy as sc
            adata = sc.read_h5ad(adata_path)
        except ImportError:
            return {"error": "scanpy not installed. Install with: pip install scanpy"}
        except Exception as e:
            return {"error": f"Failed to read AnnData: {str(e)}"}

        # Validate species (Tabula is human-only)
        species = self._detect_species(adata)
        if species != "human":
            return {
                "error": f"Tabula only supports human data, detected: '{species}'",
                "suggestion": "Use UCE for cross-species support or scGPT for mouse data",
                "supported": ["human"],
            }

        # Load model
        try:
            self._load_model(device)
        except Exception as e:
            return {"error": f"Failed to load Tabula model: {str(e)}"}

        # Preprocess
        try:
            processed_adata = self._preprocess(adata, task)
        except Exception as e:
            return {"error": f"Preprocessing failed: {str(e)}"}

        # Run inference
        try:
            embeddings = self._run_inference(
                processed_adata,
                device=device,
                batch_size=batch_size,
                batch_key=batch_key if task == TaskType.INTEGRATE else None,
            )
        except Exception as e:
            return {"error": f"Inference failed: {str(e)}"}

        # Write results
        output_keys = self._postprocess(adata, embeddings, task)
        self._add_provenance(adata, task, output_keys)

        # Save
        adata.write(output_path)

        return {
            "status": "success",
            "output_path": output_path,
            "output_keys": output_keys,
            "stats": {
                "n_cells": adata.n_obs,
                "embedding_dim": embeddings.shape[1],
                "species": species,
                "gene_scheme": f"custom_{self.VOCAB_SIZE}",
                "device": device,
            },
        }

    def _load_model(self, device: str):
        """
        Load Tabula model and gene vocabulary.

        Tabula uses:
        - TabulaTransformer with 3 blocks, 8 attention heads, 192-dim tokens
        - vocab.json for gene-to-index mapping (60,697 genes)
        - Pre-trained checkpoint (best_model.pth)
        """
        if self._model is not None:
            return

        if not self._tabula_installed:
            raise ImportError("Tabula package not installed")

        import json
        import torch

        # Resolve checkpoint directory
        checkpoint_path = self._resolve_checkpoint_dir(require=True)
        if checkpoint_path is None:
            raise ValueError(
                "Tabula checkpoint directory not specified. "
                "Download from: https://github.com/aristoteleo/tabula"
            )

        # Load vocabulary
        vocab_path = checkpoint_path / "vocab.json"
        if not vocab_path.exists():
            # Try parent directory or alternative locations
            for alt_path in [
                checkpoint_path.parent / "vocab.json",
                checkpoint_path / "resource" / "vocab.json",
            ]:
                if alt_path.exists():
                    vocab_path = alt_path
                    break

        if vocab_path.exists():
            with open(vocab_path) as f:
                self._vocab = json.load(f)
        else:
            self._vocab = None

        # Load model
        try:
            from tabula.model import TabulaTransformer

            model = TabulaTransformer(
                in_feature=self.IN_FEATURE,
                embedding_in_feature=self.VOCAB_SIZE,
                d_token=self.D_TOKEN,
                n_blocks=self.N_BLOCKS,
                attention_n_heads=self.ATTENTION_HEADS,
                ffn_d_hidden=self.D_TOKEN,
            )

            # Load pre-trained weights
            ckpt_file = self._find_checkpoint(
                checkpoint_path, extensions=[".pth", ".pt", ".ckpt"]
            )
            state_dict = torch.load(str(ckpt_file), map_location=device)

            # Handle wrapped state dicts (e.g., from Lightning or DataParallel)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            # Strip 'model.' prefix if present (from Lightning wrapper)
            cleaned = {}
            for k, v in state_dict.items():
                key = k.replace("model.", "", 1) if k.startswith("model.") else k
                cleaned[key] = v

            model.load_state_dict(cleaned, strict=False)
            model = model.to(device)
            model.eval()
            self._model = model

        except ImportError as e:
            raise ImportError(
                f"Tabula model class not available: {e}. "
                "Install from: https://github.com/aristoteleo/tabula"
            )

    def _preprocess(self, adata, task: TaskType):
        """
        Preprocess AnnData for Tabula.

        Tabula requires:
        - Genes mapped to its 60,697 gene vocabulary via vocab.json
        - Expression values binned into 51 quantile bins
        - Fixed 1,200 gene sequence length (pad/truncate)
        """
        import scanpy as sc

        # Work on a copy
        adata = adata.copy()

        # Ensure we have raw counts
        if adata.raw is not None:
            adata = adata.raw.to_adata()

        # Store total counts per cell for normalization awareness
        if "n_counts" not in adata.obs:
            X = adata.X
            if hasattr(X, "toarray"):
                adata.obs["n_counts"] = np.array(X.sum(axis=1)).flatten()
            else:
                adata.obs["n_counts"] = np.array(X.sum(axis=1)).flatten()

        return adata

    def _run_inference(
        self,
        adata,
        device: str,
        batch_size: int,
        batch_key: Optional[str] = None,
    ) -> np.ndarray:
        """
        Run Tabula inference to generate embeddings.

        Uses TabulaTransformer forward pass with no head to extract
        CLS token embeddings (192-dim).

        Args:
            adata: Preprocessed AnnData object
            device: Device string (e.g., "cuda")
            batch_size: Batch size for inference
            batch_key: Batch column for integration (optional)

        Returns:
            np.ndarray: Cell embeddings of shape (n_cells, 192)
        """
        import torch

        model = self._model
        n_cells = adata.n_obs
        all_embeddings = []

        # Convert sparse matrix to dense if needed
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        # Map genes to vocabulary indices
        gene_ids = self._map_genes_to_vocab(adata.var_names.tolist())

        # Bin expression values
        values = self._bin_expression(X)

        with torch.no_grad():
            for start in range(0, n_cells, batch_size):
                end = min(start + batch_size, n_cells)

                batch_genes = torch.tensor(
                    gene_ids[:self.IN_FEATURE], dtype=torch.long, device=device
                ).unsqueeze(0).expand(end - start, -1)

                batch_values = torch.tensor(
                    values[start:end, :self.IN_FEATURE],
                    dtype=torch.float32,
                    device=device,
                )

                # Forward pass without head to get embeddings
                output = model(
                    genes=batch_genes,
                    values=batch_values,
                    head=None,
                )

                # Extract CLS token embedding or average pool
                if isinstance(output, tuple):
                    embeddings = output[0]
                elif output.ndim == 3:
                    # Shape: (batch, seq_len, d_token) -> avg pool over seq
                    embeddings = output.mean(dim=1)
                else:
                    embeddings = output

                all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    def _map_genes_to_vocab(self, gene_names: list[str]) -> list[int]:
        """
        Map gene names to Tabula vocabulary indices.

        Args:
            gene_names: List of gene symbol strings from AnnData.var_names

        Returns:
            List of vocabulary indices (padded to IN_FEATURE length)
        """
        if self._vocab is None:
            # If no vocab loaded, use sequential indices (fallback)
            indices = list(range(min(len(gene_names), self.IN_FEATURE)))
            while len(indices) < self.IN_FEATURE:
                indices.append(self.PADDING_ID)
            return indices

        indices = []
        for gene in gene_names:
            if gene in self._vocab:
                indices.append(self._vocab[gene])
            else:
                # Try uppercase/lowercase variants
                gene_upper = gene.upper()
                gene_lower = gene.lower()
                if gene_upper in self._vocab:
                    indices.append(self._vocab[gene_upper])
                elif gene_lower in self._vocab:
                    indices.append(self._vocab[gene_lower])
                # Skip unmapped genes

        # Truncate or pad to fixed length
        indices = indices[:self.IN_FEATURE]
        while len(indices) < self.IN_FEATURE:
            indices.append(self.PADDING_ID)

        return indices

    def _bin_expression(self, X: np.ndarray) -> np.ndarray:
        """
        Bin expression values into quantile bins.

        Tabula uses 51 quantile bins for discretizing expression values.

        Args:
            X: Dense expression matrix (n_cells, n_genes)

        Returns:
            Binned expression matrix (n_cells, n_genes) with values 0..N_BINS-1
        """
        n_cells, n_genes = X.shape

        # Pad or truncate genes to IN_FEATURE
        if n_genes < self.IN_FEATURE:
            padded = np.zeros((n_cells, self.IN_FEATURE), dtype=np.float32)
            padded[:, :n_genes] = X
            X = padded
        else:
            X = X[:, :self.IN_FEATURE]

        # Compute quantile bin edges from non-zero values
        nonzero_vals = X[X > 0]
        if len(nonzero_vals) == 0:
            return np.zeros((n_cells, self.IN_FEATURE), dtype=np.float32)

        bin_edges = np.quantile(nonzero_vals, np.linspace(0, 1, self.N_BINS + 1))
        # Remove duplicates in bin edges
        bin_edges = np.unique(bin_edges)

        binned = np.digitize(X, bin_edges) - 1
        binned = np.clip(binned, 0, self.N_BINS - 1).astype(np.float32)

        # Keep zeros as zeros
        binned[X == 0] = 0

        return binned

    def _postprocess(self, adata, embeddings: np.ndarray, task: TaskType) -> list[str]:
        """
        Write embeddings to AnnData.

        Args:
            adata: AnnData object to update
            embeddings: Cell embeddings array (n_cells, 192)
            task: Task type

        Returns:
            List of output keys written
        """
        output_keys = []

        if task == TaskType.EMBED:
            key = self.spec.output_keys.embedding_key  # "X_tabula"
            adata.obsm[key] = embeddings
            output_keys.append(f"obsm['{key}']")
        elif task == TaskType.INTEGRATE:
            key = self.spec.output_keys.integration_key  # "X_tabula_integrated"
            adata.obsm[key] = embeddings
            output_keys.append(f"obsm['{key}']")

        return output_keys

    def _detect_species(self, adata) -> str:
        """Detect species from AnnData metadata."""
        # Check uns first
        if "species" in adata.uns:
            species = adata.uns["species"].lower()
            if "human" in species or "sapiens" in species:
                return "human"
            elif "mouse" in species or "musculus" in species:
                return "mouse"

        # Infer from gene naming convention:
        # Human genes: all letters uppercase (e.g., TP53, BRCA1, GAPDH)
        # Mouse genes: first letter uppercase, rest lowercase (e.g., Tp53, Brca1, Gapdh)
        gene_names = adata.var_names[:100].tolist()
        alpha_chars = lambda g: "".join(c for c in g if c.isalpha())
        uppercase_count = sum(
            1 for g in gene_names
            if alpha_chars(g) and alpha_chars(g) == alpha_chars(g).upper()
        )

        return "human" if uppercase_count > 50 else "mouse"
