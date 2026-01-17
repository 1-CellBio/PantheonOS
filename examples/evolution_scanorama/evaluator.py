"""
Evaluator for Scanorama Algorithm Evolution.

This evaluator measures:
1. Integration quality (how well batches are mixed)
2. Biological variance preservation (how well structure is preserved)
3. Execution speed

The combined score balances these metrics for evolution.
"""

import numpy as np
import pandas as pd
import time
import sys
import importlib.util
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from typing import Dict, Any, Tuple, List


def load_tma_data(
    data_dir: Path,
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load TMA data with real cell type labels.

    Args:
        data_dir: Directory containing the data files
        split: Which split to load ("train", "val", or "test")

    Returns:
        X: Features (n_cells x 30)
        batch_labels: Batch labels (donor)
        celltype_labels: True cell type labels
    """
    df = pd.read_csv(data_dir / f"tma_8000_{split}.csv")
    X = df.iloc[:, :30].values  # PC1-PC30
    batch_labels = df["donor"].values
    celltype_labels = df["celltype"].values

    return X, batch_labels, celltype_labels


def split_by_batch(
    X: np.ndarray,
    batch_labels: np.ndarray,
) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
    """
    Split data matrix by batch labels for scanorama input.

    Args:
        X: Data matrix (n_cells x n_features)
        batch_labels: Batch assignments

    Returns:
        datasets: List of data matrices, one per batch
        batch_names: List of batch names
        original_order: Indices to reconstruct original order
    """
    unique_batches = np.unique(batch_labels)
    datasets = []
    batch_names = []
    indices_list = []

    for batch in unique_batches:
        mask = batch_labels == batch
        indices = np.where(mask)[0]
        datasets.append(X[mask].copy())
        batch_names.append(str(batch))
        indices_list.append(indices)

    # Create mapping to reconstruct original order
    original_order = np.concatenate(indices_list)

    return datasets, batch_names, original_order


def reconstruct_from_batches(
    datasets: List[np.ndarray],
    original_order: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct full matrix from batch-split datasets.

    Args:
        datasets: List of corrected data matrices
        original_order: Indices to reconstruct original order

    Returns:
        Full corrected matrix in original cell order
    """
    # Concatenate all datasets
    X_concat = np.vstack(datasets)

    # Reconstruct original order
    inverse_order = np.argsort(original_order)
    X_corrected = X_concat[inverse_order]

    return X_corrected


def compute_batch_mixing_score(
    X: np.ndarray,
    batch_labels: np.ndarray,
    k: int = 50,
) -> float:
    """
    Compute batch mixing score using k-nearest neighbors.

    Measures how well different batches are mixed in the embedding.
    Higher score = better mixing.

    Args:
        X: Embedding (n_cells x n_features)
        batch_labels: Batch assignments
        k: Number of neighbors

    Returns:
        Mixing score in [0, 1]
    """
    n_cells = X.shape[0]
    unique_batches = np.unique(batch_labels)

    # Expected proportion of each batch
    expected_props = np.array([
        np.sum(batch_labels == b) / n_cells
        for b in unique_batches
    ])

    # Find k nearest neighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, n_cells), algorithm="auto")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    # For each cell, compute batch proportions in neighborhood
    mixing_scores = []
    for i in range(n_cells):
        neighbor_batches = batch_labels[indices[i, 1:]]  # Exclude self
        observed_props = np.array([
            np.sum(neighbor_batches == b) / k
            for b in unique_batches
        ])

        # Compare to expected (lower deviation = better mixing)
        score = 1 - np.sqrt(np.mean((observed_props - expected_props) ** 2))
        mixing_scores.append(max(0, score))

    return np.mean(mixing_scores)


def compute_bio_conservation_score(
    X_corrected: np.ndarray,
    X_original: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """
    Compute biological structure conservation score using silhouette score.

    Measures how well the biological clusters are separated after correction.
    Higher silhouette = better separation of cell types.

    Args:
        X_corrected: Corrected embedding
        X_original: Original embedding (unused, kept for API compatibility)
        true_labels: True biological labels (cell types)

    Returns:
        Silhouette score normalized to [0, 1]
    """
    try:
        if len(np.unique(true_labels)) > 1:
            # Compute silhouette score on corrected data
            silhouette = silhouette_score(X_corrected, true_labels)
            # Normalize from [-1, 1] to [0, 1]
            return (silhouette + 1) / 2
        else:
            return 0.5
    except Exception:
        return 0.5


def evaluate(workspace_path: str) -> Dict[str, Any]:
    """
    Evaluate the Scanorama implementation on TMA training data.

    This is the main evaluation function called by Pantheon Evolution.
    Uses real cell type labels for accurate biological conservation scoring.

    Args:
        workspace_path: Path to the workspace containing scanorama package

    Returns:
        Dictionary with metrics including 'combined_score'
    """
    workspace = Path(workspace_path)

    # Add workspace to path so we can import scanorama
    scanorama_path = workspace / "scanorama"
    if not scanorama_path.exists():
        return {
            "combined_score": 0.0,
            "error": "scanorama package not found",
        }

    # Load the scanorama module from workspace
    try:
        # Import utils first
        utils_path = scanorama_path / "utils.py"
        spec_utils = importlib.util.spec_from_file_location("scanorama.utils", utils_path)
        utils_module = importlib.util.module_from_spec(spec_utils)
        sys.modules["scanorama.utils"] = utils_module
        spec_utils.loader.exec_module(utils_module)

        # Import main scanorama module
        main_path = scanorama_path / "scanorama.py"
        spec = importlib.util.spec_from_file_location("scanorama.scanorama", main_path)
        scanorama_module = importlib.util.module_from_spec(spec)
        sys.modules["scanorama.scanorama"] = scanorama_module
        spec.loader.exec_module(scanorama_module)

    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Failed to load scanorama: {e}",
        }

    # Load TMA training data with real cell type labels
    import os
    _default_data_dir = r"C:\Users\wzxu\Desktop\Pantheon\pantheon-agents-2\examples\evolution_harmonypy\data"
    data_dir = Path(os.environ.get("SCANORAMA_DATA_DIR", _default_data_dir))

    try:
        X_train, batch_train, celltype_train = load_tma_data(data_dir, split="train")
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Failed to load TMA data: {e}",
        }

    n_cells = len(X_train)

    # Split data by batch for scanorama input
    datasets, batch_names, original_order = split_by_batch(X_train, batch_train)

    # Create fake gene names (scanorama expects them, but we use PCA features)
    n_features = X_train.shape[1]
    genes_list = [[f"PC{i+1}" for i in range(n_features)] for _ in datasets]

    # Run scanorama and measure time
    try:
        start_time = time.time()

        # Use integrate function which returns low-dimensional embeddings
        # Use dimred=30 (same as input) to ensure dense arrays are returned
        # (numpy 2.x can't concatenate sparse matrices with np.concatenate)
        # Use approx=False to avoid annoy dependency (uses sklearn NearestNeighbors)
        datasets_corrected, _ = scanorama_module.integrate(
            datasets,
            genes_list,
            dimred=30,  # Same as input dims; ensures dense arrays are used
            verbose=0,
            knn=20,
            sigma=15,
            alpha=0.10,
            approx=False,  # Use exact NN (no annoy dependency)
            seed=42,
        )

        execution_time = time.time() - start_time

        # Reconstruct full matrix in original order
        X_corrected = reconstruct_from_batches(datasets_corrected, original_order)

    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Scanorama execution failed: {e}",
        }

    # Correctness check: verify that the algorithm actually corrected the data
    correction_magnitude = np.abs(X_train - X_corrected).mean()
    if correction_magnitude < 0.01:
        return {
            "combined_score": 0.0,
            "correction_magnitude": correction_magnitude,
            "error": "No meaningful correction applied (data unchanged)",
        }

    # Compute metrics on training data using real cell type labels
    try:
        # Batch mixing (higher = better, weight: 0.45)
        mixing_score = compute_batch_mixing_score(X_corrected, batch_train)

        # Biological conservation using REAL cell type labels (weight: 0.45)
        bio_score = compute_bio_conservation_score(X_corrected, X_train, celltype_train)

        # Speed score (faster = better, weight: 0.1)
        speed_score = 1.0 / (1 + execution_time)

        # Combined score
        combined_score = (
            0.45 * mixing_score +
            0.45 * bio_score +
            0.10 * speed_score
        )

        return {
            "combined_score": combined_score,
            "mixing_score": mixing_score,
            "bio_conservation_score": bio_score,
            "speed_score": speed_score,
            "correction_magnitude": correction_magnitude,
            "execution_time": execution_time,
            "n_cells": n_cells,
            "n_batches": len(datasets),
            "dataset": "tma_train",
        }

    except Exception as e:
        return {
            "combined_score": 0.1,  # Partial credit for running
            "error": f"Metric computation failed: {e}",
        }


def _evaluate_on_split(workspace_path: str, split: str) -> Dict[str, Any]:
    """
    Evaluate the Scanorama implementation on a specific TMA data split.

    Args:
        workspace_path: Path to the workspace containing scanorama package
        split: Which split to evaluate on ("train", "val", or "test")

    Returns:
        Dictionary with metrics
    """
    workspace = Path(workspace_path)

    # Add workspace to path so we can import scanorama
    scanorama_path = workspace / "scanorama"
    if not scanorama_path.exists():
        return {
            "combined_score": 0.0,
            "error": "scanorama package not found",
        }

    # Load the scanorama module from workspace
    try:
        # Import utils first
        utils_path = scanorama_path / "utils.py"
        spec_utils = importlib.util.spec_from_file_location("scanorama.utils", utils_path)
        utils_module = importlib.util.module_from_spec(spec_utils)
        sys.modules["scanorama.utils"] = utils_module
        spec_utils.loader.exec_module(utils_module)

        # Import main scanorama module
        main_path = scanorama_path / "scanorama.py"
        spec = importlib.util.spec_from_file_location("scanorama.scanorama", main_path)
        scanorama_module = importlib.util.module_from_spec(spec)
        sys.modules["scanorama.scanorama"] = scanorama_module
        spec.loader.exec_module(scanorama_module)

    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Failed to load scanorama: {e}",
        }

    # Load TMA data
    import os
    _default_data_dir = r"C:\Users\wzxu\Desktop\Pantheon\pantheon-agents-2\examples\evolution_harmonypy\data"
    data_dir = Path(os.environ.get("SCANORAMA_DATA_DIR", _default_data_dir))

    try:
        X, batch_labels, celltype_labels = load_tma_data(data_dir, split=split)
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Failed to load TMA {split} data: {e}",
        }

    n_cells = len(X)

    # Split data by batch
    datasets, batch_names, original_order = split_by_batch(X, batch_labels)
    n_features = X.shape[1]
    genes_list = [[f"PC{i+1}" for i in range(n_features)] for _ in datasets]

    # Run scanorama
    try:
        start_time = time.time()

        datasets_corrected, _ = scanorama_module.integrate(
            datasets,
            genes_list,
            dimred=30,  # Same as input dims; ensures dense arrays are used
            verbose=0,
            knn=20,
            sigma=15,
            alpha=0.10,
            approx=False,  # Use exact NN (no annoy dependency)
            seed=42,
        )

        execution_time = time.time() - start_time
        X_corrected = reconstruct_from_batches(datasets_corrected, original_order)

    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Scanorama execution failed: {e}",
        }

    # Compute metrics
    try:
        mixing_score = compute_batch_mixing_score(X_corrected, batch_labels)
        bio_score = compute_bio_conservation_score(X_corrected, X, celltype_labels)
        speed_score = 1.0 / (1 + execution_time)

        combined_score = (
            0.45 * mixing_score +
            0.45 * bio_score +
            0.10 * speed_score
        )

        return {
            "combined_score": combined_score,
            "mixing_score": mixing_score,
            "bio_conservation_score": bio_score,
            "speed_score": speed_score,
            "execution_time": execution_time,
            "n_cells": n_cells,
            "n_batches": len(datasets),
            "dataset": f"tma_{split}",
        }

    except Exception as e:
        return {
            "combined_score": 0.1,
            "error": f"Metric computation failed: {e}",
        }


def evaluate_on_validation(workspace_path: str) -> Dict[str, Any]:
    """Evaluate on TMA VALIDATION set."""
    return _evaluate_on_split(workspace_path, "val")


def evaluate_on_test(workspace_path: str) -> Dict[str, Any]:
    """Evaluate on TMA TEST set."""
    return _evaluate_on_split(workspace_path, "test")


if __name__ == "__main__":
    import os

    # Use current directory as workspace
    workspace = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("Scanorama Evaluator Test (TMA Data)")
    print("=" * 60)
    print(f"Workspace: {workspace}")

    # Training set evaluation
    print("\n" + "-" * 60)
    print("TRAINING SET Evaluation:")
    print("-" * 60)
    result = evaluate(workspace)
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, int):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
