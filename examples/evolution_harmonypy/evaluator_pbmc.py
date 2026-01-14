"""
Evaluator for Harmony Algorithm Evolution using PBMC Dataset.

This evaluator uses PBMC 3500 dataset for training and validation.
Since PBMC data doesn't have ground-truth cell type labels, we use
K-means clustering on the uncorrected data to generate pseudo-labels
for biological conservation scoring.

Metrics:
1. Integration quality (batch mixing score)
2. Biological variance preservation (silhouette on pseudo-labels)
3. Execution speed
4. Convergence behavior
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


def load_pbmc_data(
    data_dir: Path,
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PBMC data from CSV files.

    Args:
        data_dir: Directory containing the data files
        split: Which split to load ("train" or "val")

    Returns:
        X: Features (n_cells x 30)
        batch_labels: Batch labels (donor)
    """
    filename = f"pbmc_3500_full_{split}.csv"
    df = pd.read_csv(data_dir / filename)
    X = df.iloc[:, :30].values  # PC1-PC30
    batch_labels = df["donor"].values
    return X, batch_labels


def generate_pseudo_labels(X: np.ndarray, n_clusters: int = 8) -> np.ndarray:
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

        # Compare to expected (lower RMSE = better mixing)
        score = 1 - np.sqrt(np.mean((observed_props - expected_props) ** 2))
        mixing_scores.append(max(0, score))

    return np.mean(mixing_scores)


def compute_bio_conservation_score(
    X_corrected: np.ndarray,
    pseudo_labels: np.ndarray,
) -> float:
    """
    Compute biological structure conservation score using silhouette score.

    Uses pseudo-labels generated from K-means on uncorrected data.

    Args:
        X_corrected: Corrected embedding
        pseudo_labels: Pseudo cell type labels from K-means

    Returns:
        Silhouette score normalized to [0, 1]
    """
    try:
        if len(np.unique(pseudo_labels)) > 1:
            silhouette = silhouette_score(X_corrected, pseudo_labels)
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
    Evaluate the Harmony implementation on PBMC training data.

    This is the main evaluation function called by Pantheon Evolution.
    Uses pseudo-labels from K-means for biological conservation scoring.

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

    # Load PBMC training data
    import os
    _script_dir = Path(__file__).parent.resolve() if "__file__" in dir() else Path.cwd()
    _default_data_dir = _script_dir / "data"
    data_dir = Path(os.environ.get("HARMONY_DATA_DIR", str(_default_data_dir)))

    try:
        X_train, batch_train = load_pbmc_data(data_dir, split="train")
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Failed to load PBMC data: {e}",
        }

    n_cells = len(X_train)

    # Generate pseudo-labels from uncorrected data
    pseudo_labels = generate_pseudo_labels(X_train, n_clusters=8)

    # Run harmony on training data and measure time
    try:
        # Convert batch labels to DataFrame for official harmonypy API
        meta_data = pd.DataFrame({"batch": batch_train})

        start_time = time.time()
        hm = harmony_module.run_harmony(
            X_train,
            meta_data,
            vars_use="batch",
            nclust=50,
            max_iter_harmony=10,
            random_state=42,
            verbose=False,
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
    correction_magnitude = np.abs(X_train - X_corrected).mean()
    if correction_magnitude < 0.01:
        return {
            "combined_score": 0.0,
            "correction_magnitude": correction_magnitude,
            "error": "No meaningful correction applied (data unchanged)",
        }

    # Compute metrics on training data
    try:
        # Batch mixing (higher = better, weight: 0.4)
        mixing_score = compute_batch_mixing_score(X_corrected, batch_train)

        # Biological conservation using pseudo-labels (weight: 0.3)
        bio_score = compute_bio_conservation_score(X_corrected, pseudo_labels)

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
            "dataset": "pbmc_train",
        }

    except Exception as e:
        return {
            "combined_score": 0.1,  # Partial credit for running
            "error": f"Metric computation failed: {e}",
        }


def evaluate_on_validation(workspace_path: str) -> Dict[str, Any]:
    """
    Evaluate the Harmony implementation on PBMC validation data.

    Args:
        workspace_path: Path to the workspace containing harmony.py

    Returns:
        Dictionary with validation metrics
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

    # Load PBMC validation data
    import os
    _script_dir = Path(__file__).parent.resolve() if "__file__" in dir() else Path.cwd()
    _default_data_dir = _script_dir / "data"
    data_dir = Path(os.environ.get("HARMONY_DATA_DIR", str(_default_data_dir)))

    try:
        X_val, batch_val = load_pbmc_data(data_dir, split="val")
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Failed to load PBMC validation data: {e}",
        }

    n_cells = len(X_val)

    # Generate pseudo-labels from uncorrected validation data
    pseudo_labels = generate_pseudo_labels(X_val, n_clusters=8)

    # Run harmony
    try:
        meta_data = pd.DataFrame({"batch": batch_val})

        start_time = time.time()
        hm = harmony_module.run_harmony(
            X_val,
            meta_data,
            vars_use="batch",
            nclust=50,
            max_iter_harmony=10,
            random_state=42,
            verbose=False,
        )
        execution_time = time.time() - start_time

        X_corrected = hm.Z_corr
        objectives = hm.objectives

    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Harmony execution failed: {e}",
        }

    # Compute metrics
    try:
        mixing_score = compute_batch_mixing_score(X_corrected, batch_val)
        bio_score = compute_bio_conservation_score(X_corrected, pseudo_labels)
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
            "dataset": "pbmc_val",
        }

    except Exception as e:
        return {
            "combined_score": 0.1,
            "error": f"Metric computation failed: {e}",
        }


if __name__ == "__main__":
    import os

    # Use current directory as workspace
    workspace = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("Harmony Evaluator Test (PBMC Dataset)")
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
        else:
            print(f"  {key}: {value}")

    # Validation set evaluation
    print("\n" + "-" * 60)
    print("VALIDATION SET Evaluation:")
    print("-" * 60)
    val_result = evaluate_on_validation(workspace)
    for key, value in val_result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
