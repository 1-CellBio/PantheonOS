"""
Evaluator for Harmony Algorithm Evolution.

This evaluator measures:
1. Integration quality (how well batches are mixed)
2. Biological variance preservation (how well structure is preserved)
3. Execution speed
4. Convergence behavior

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
from sklearn.cluster import KMeans
from typing import Dict, Any, Tuple


def generate_test_data(
    n_cells: int = 2000,
    n_features: int = 50,
    n_batches: int = 3,
    n_clusters: int = 5,
    batch_effect_strength: float = 2.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic single-cell data with batch effects.

    Args:
        n_cells: Total number of cells
        n_features: Number of features (PCs)
        n_batches: Number of batches
        n_clusters: Number of biological clusters
        batch_effect_strength: Strength of batch effects
        random_state: Random seed

    Returns:
        X: Data matrix (n_cells x n_features)
        batch_labels: Batch assignments
        true_labels: True biological cluster labels
    """
    np.random.seed(random_state)

    cells_per_batch = n_cells // n_batches

    X_list = []
    batch_list = []
    label_list = []

    # Generate cluster centers
    cluster_centers = np.random.randn(n_clusters, n_features) * 3

    for batch_idx in range(n_batches):
        # Batch-specific offset
        batch_offset = np.random.randn(n_features) * batch_effect_strength

        for cluster_idx in range(n_clusters):
            # Cells per cluster per batch
            n = cells_per_batch // n_clusters

            # Generate cells around cluster center with batch effect
            cells = (
                cluster_centers[cluster_idx]
                + np.random.randn(n, n_features) * 0.5
                + batch_offset
            )

            X_list.append(cells)
            batch_list.extend([batch_idx] * n)
            label_list.extend([cluster_idx] * n)

    X = np.vstack(X_list)
    batch_labels = np.array(batch_list)
    true_labels = np.array(label_list)

    return X, batch_labels, true_labels


def load_pbmc_data(
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load real PBMC data from CSV files.

    Args:
        data_dir: Directory containing the data files

    Returns:
        X_train: Training features (n_train x 30)
        batch_train: Training batch labels
        X_val: Validation features (n_val x 30)
        batch_val: Validation batch labels
    """
    # Read train data
    train_df = pd.read_csv(data_dir / "pbmc_3500_full_train.csv")
    X_train = train_df.iloc[:, :30].values  # PC1-PC30
    batch_train = train_df["donor"].values

    # Read val data
    val_df = pd.read_csv(data_dir / "pbmc_3500_full_val.csv")
    X_val = val_df.iloc[:, :30].values
    batch_val = val_df["donor"].values

    return X_train, batch_train, X_val, batch_val


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


def generate_pseudo_labels(X: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """
    Generate pseudo cell type labels using K-means clustering.

    Clusters the original (uncorrected) data to create reference labels
    for evaluating biological structure preservation.

    Args:
        X: Data matrix (n_cells x n_features)
        n_clusters: Number of clusters

    Returns:
        Cluster labels for each cell
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(X)


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
    n_batches = len(unique_batches)

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

        # Compare to expected (lower KL divergence = better mixing)
        # Use simple correlation instead for stability
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


def compute_convergence_score(objectives: list) -> float:
    """
    Compute convergence behavior score.

    Rewards fast and stable convergence.

    Args:
        objectives: List of objective values over iterations

    Returns:
        Convergence score in [0, 1]
    """
    if len(objectives) < 2:
        return 0.5

    # Check if converged (objective stabilized)
    final_change = abs(objectives[-1] - objectives[-2]) / (abs(objectives[-2]) + 1e-8)

    # Reward small final change
    convergence_quality = np.exp(-final_change * 10)

    # Reward fewer iterations (faster convergence)
    speed_bonus = 1.0 / (1 + len(objectives) / 10)

    return 0.7 * convergence_quality + 0.3 * speed_bonus


def evaluate(workspace_path: str) -> Dict[str, Any]:
    """
    Evaluate the Harmony implementation on TMA training data.

    This is the main evaluation function called by Pantheon Evolution.
    Uses real cell type labels for accurate biological conservation scoring.

    Args:
        workspace_path: Path to the workspace containing harmony.py

    Returns:
        Dictionary with metrics including 'combined_score'
    """
    workspace = Path(workspace_path)

    # Load the harmony module from workspace
    harmony_path = workspace / "harmony.py"
    if not harmony_path.exists():
        return {
            "combined_score": 0.0,
            "error": "harmony.py not found",
        }

    try:
        spec = importlib.util.spec_from_file_location("harmony", harmony_path)
        harmony_module = importlib.util.module_from_spec(spec)
        sys.modules["harmony"] = harmony_module
        spec.loader.exec_module(harmony_module)
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Failed to load harmony.py: {e}",
        }

    # Load TMA training data with real cell type labels
    # HARMONY_DATA_DIR must be set by the caller (run_evolution.py) since
    # this evaluator runs in a temp workspace where __file__ is not reliable
    import os
    data_dir = Path(os.environ.get("HARMONY_DATA_DIR", "data"))

    try:
        X_train, batch_train, celltype_train = load_tma_data(data_dir, split="train")
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Failed to load TMA data: {e}",
        }

    n_cells = len(X_train)

    # Run harmony on training data and measure time
    try:
        start_time = time.time()
        hm = harmony_module.run_harmony(
            X_train,
            batch_train,
            n_clusters=50,
            max_iter=10,
            random_state=42,
        )
        execution_time = time.time() - start_time

        X_corrected = hm.Z_corr
        objectives = hm.objectives

    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Harmony execution failed: {e}",
        }

    # Correctness check: verify that the algorithm actually corrected the data
    # This prevents "metric hacking" where the algorithm does nothing but scores well
    correction_magnitude = np.abs(X_train - X_corrected).mean()
    if correction_magnitude < 0.01:
        return {
            "combined_score": 0.0,
            "correction_magnitude": correction_magnitude,
            "error": "No meaningful correction applied (data unchanged)",
        }

    # Compute metrics on training data using real cell type labels
    try:
        # Batch mixing (higher = better, weight: 0.4)
        mixing_score = compute_batch_mixing_score(X_corrected, batch_train)

        # Biological conservation using REAL cell type labels (weight: 0.3)
        bio_score = compute_bio_conservation_score(X_corrected, X_train, celltype_train)

        # Speed score (faster = better, weight: 0.2)
        speed_score = 1.0 / (1 + execution_time)

        # Convergence score (weight: 0.1)
        conv_score = compute_convergence_score(objectives)

        # Combined score
        combined_score = (
            0.45 * mixing_score +
            0.45 * bio_score +
            0.05 * speed_score +
            0.05 * conv_score
        )

        return {
            "combined_score": combined_score,
            "mixing_score": mixing_score,
            "bio_conservation_score": bio_score,
            "speed_score": speed_score,
            "convergence_score": conv_score,
            "correction_magnitude": correction_magnitude,
            "execution_time": execution_time,
            "iterations": len(objectives),
            "n_cells": n_cells,
            "dataset": "tma_train",
        }

    except Exception as e:
        return {
            "combined_score": 0.1,  # Partial credit for running
            "error": f"Metric computation failed: {e}",
        }


def _evaluate_on_split(workspace_path: str, split: str) -> Dict[str, Any]:
    """
    Evaluate the Harmony implementation on a specific TMA data split.

    Args:
        workspace_path: Path to the workspace containing harmony.py
        split: Which split to evaluate on ("train", "val", or "test")

    Returns:
        Dictionary with metrics
    """
    workspace = Path(workspace_path)

    # Load the harmony module from workspace
    harmony_path = workspace / "harmony.py"
    if not harmony_path.exists():
        return {
            "combined_score": 0.0,
            "error": "harmony.py not found",
        }

    try:
        spec = importlib.util.spec_from_file_location("harmony", harmony_path)
        harmony_module = importlib.util.module_from_spec(spec)
        sys.modules["harmony"] = harmony_module
        spec.loader.exec_module(harmony_module)
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Failed to load harmony.py: {e}",
        }

    # Load TMA data
    import os
    data_dir = Path(os.environ.get("HARMONY_DATA_DIR", "data"))

    try:
        X, batch_labels, celltype_labels = load_tma_data(data_dir, split=split)
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Failed to load TMA {split} data: {e}",
        }

    n_cells = len(X)

    # Run harmony
    try:
        start_time = time.time()
        hm = harmony_module.run_harmony(
            X,
            batch_labels,
            n_clusters=50,
            max_iter=10,
            random_state=42,
        )
        execution_time = time.time() - start_time

        X_corrected = hm.Z_corr
        objectives = hm.objectives

    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Harmony execution failed: {e}",
        }

    # Compute metrics using real cell type labels
    try:
        mixing_score = compute_batch_mixing_score(X_corrected, batch_labels)
        bio_score = compute_bio_conservation_score(X_corrected, X, celltype_labels)
        speed_score = 1.0 / (1 + execution_time)
        conv_score = compute_convergence_score(objectives)

        combined_score = (
            0.45 * mixing_score +
            0.45 * bio_score +
            0.05 * speed_score +
            0.05 * conv_score
        )

        return {
            "combined_score": combined_score,
            "mixing_score": mixing_score,
            "bio_conservation_score": bio_score,
            "speed_score": speed_score,
            "convergence_score": conv_score,
            "execution_time": execution_time,
            "iterations": len(objectives),
            "n_cells": n_cells,
            "dataset": f"tma_{split}",
        }

    except Exception as e:
        return {
            "combined_score": 0.1,
            "error": f"Metric computation failed: {e}",
        }


def evaluate_on_validation(workspace_path: str) -> Dict[str, Any]:
    """
    Evaluate the Harmony implementation on the TMA VALIDATION set.

    This function should be used AFTER evolution is complete to evaluate
    the final selected program on held-out data.

    Args:
        workspace_path: Path to the workspace containing harmony.py

    Returns:
        Dictionary with validation metrics
    """
    return _evaluate_on_split(workspace_path, "val")


def evaluate_on_test(workspace_path: str) -> Dict[str, Any]:
    """
    Evaluate the Harmony implementation on the TMA TEST set.

    This function should be used for final evaluation after all
    hyperparameter tuning is complete.

    Args:
        workspace_path: Path to the workspace containing harmony.py

    Returns:
        Dictionary with test metrics
    """
    return _evaluate_on_split(workspace_path, "test")


if __name__ == "__main__":
    # Test the evaluator locally
    import os

    # Use current directory as workspace
    workspace = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("Harmony Evaluator Test (TMA Data with Real Cell Types)")
    print("=" * 60)
    print(f"Workspace: {workspace}")

    # Training set evaluation (used during evolution)
    print("\n" + "-" * 60)
    print("TRAINING SET Evaluation (used during evolution):")
    print("-" * 60)
    result = evaluate(workspace)
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, int):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    # Validation set evaluation (used after evolution)
    print("\n" + "-" * 60)
    print("VALIDATION SET Evaluation (used after evolution):")
    print("-" * 60)
    val_result = evaluate_on_validation(workspace)
    for key, value in val_result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, int):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    # Test set evaluation (final evaluation)
    print("\n" + "-" * 60)
    print("TEST SET Evaluation (final evaluation):")
    print("-" * 60)
    test_result = evaluate_on_test(workspace)
    for key, value in test_result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, int):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
