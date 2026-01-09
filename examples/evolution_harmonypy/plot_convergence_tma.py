#!/usr/bin/env python
"""
Plot convergence curves comparing original Harmony vs Harmony #52 on TMA data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util
import sys
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# Setup paths
example_dir = Path(__file__).parent
data_dir = example_dir / "data"


def load_module(name: str, path: Path):
    """Load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_tma_data(split: str = "train", n_samples: int = 2000):
    """Load TMA data with real cell type labels."""
    df = pd.read_csv(data_dir / f"tma_8000_{split}.csv")
    # Sample for faster computation
    if n_samples and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
    X = df.iloc[:, :30].values  # PC1-PC30
    batch_labels = df["donor"].values
    cell_types = df["celltype"].values
    return X, batch_labels, cell_types


def compute_batch_mixing_score(X: np.ndarray, batch_labels: np.ndarray, k: int = 50) -> float:
    """Compute batch mixing score using k-nearest neighbors."""
    n_cells = X.shape[0]
    unique_batches = np.unique(batch_labels)
    expected_props = np.array([np.sum(batch_labels == b) / n_cells for b in unique_batches])

    nn = NearestNeighbors(n_neighbors=min(k + 1, n_cells), algorithm="auto")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    mixing_scores = []
    for i in range(n_cells):
        neighbor_batches = batch_labels[indices[i, 1:]]
        observed_props = np.array([np.sum(neighbor_batches == b) / k for b in unique_batches])
        score = 1 - np.sqrt(np.mean((observed_props - expected_props) ** 2))
        mixing_scores.append(max(0, score))

    return np.mean(mixing_scores)


def compute_bio_conservation_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute biological structure conservation using silhouette score."""
    try:
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            return (silhouette + 1) / 2
        return 0.5
    except Exception:
        return 0.5


def run_harmony_with_tracking(harmony_module, X, batch_labels, cell_types, n_clusters=50, max_iter=10):
    """Run Harmony and track metrics at each iteration."""
    metrics_history = []

    # Create harmony instance
    hm = harmony_module.Harmony(
        n_clusters=n_clusters,
        max_iter=1,  # Run one iteration at a time
        random_state=42
    )

    # Initialize manually
    n_cells, n_features = X.shape
    hm.Z_orig = X.copy()
    hm.Z_corr = X.copy()

    # Create batch membership matrix
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)
    hm.Phi = np.zeros((n_batches, n_cells))
    for i, batch in enumerate(unique_batches):
        hm.Phi[i, batch_labels == batch] = 1
    hm.batch_props = hm.Phi.sum(axis=1) / n_cells

    # Handle optimized version's additional attributes
    if hasattr(hm, 'n_cells'):
        hm.n_cells = n_cells
        hm.n_features = n_features
        hm.n_batches = n_batches
    if hasattr(hm, 'batch_id'):
        batch_map = {b: i for i, b in enumerate(unique_batches)}
        hm.batch_id = np.array([batch_map[b] for b in batch_labels])
    if hasattr(hm, 'batch_indices'):
        hm.batch_indices = [np.where(batch_labels == b)[0] for b in unique_batches]
    if hasattr(hm, 'Phi_reduced'):
        hm.Phi_reduced = hm.Phi[1:, :]
        hm.design = np.column_stack([np.ones(n_cells), hm.Phi_reduced.T])

    # Initial metrics (iteration 0)
    mixing = compute_batch_mixing_score(hm.Z_corr, batch_labels)
    bio = compute_bio_conservation_score(hm.Z_corr, cell_types)
    metrics_history.append({
        'iteration': 0,
        'mixing_score': mixing,
        'bio_conservation': bio
    })

    # Initialize clusters
    hm._init_clusters()

    # Run iterations
    hm.objectives = []
    for iteration in range(1, max_iter + 1):
        # Clustering step
        hm._cluster()

        # Correction step
        hm._correct()

        # Compute metrics
        mixing = compute_batch_mixing_score(hm.Z_corr, batch_labels)
        bio = compute_bio_conservation_score(hm.Z_corr, cell_types)
        metrics_history.append({
            'iteration': iteration,
            'mixing_score': mixing,
            'bio_conservation': bio
        })

        # Track objective
        obj = hm._compute_objective()
        hm.objectives.append(obj)

        print(f"  Iter {iteration}: mixing={mixing:.4f}, bio={bio:.4f}")

    return hm, metrics_history


def plot_convergence():
    """Create convergence comparison plot."""
    print("Loading TMA data...")
    X, batch_labels, cell_types = load_tma_data("train")
    print(f"  Data shape: {X.shape}")
    print(f"  Batches: {np.unique(batch_labels)}")
    print(f"  Cell types: {len(np.unique(cell_types))}")

    # Load both harmony implementations
    print("\nLoading Harmony implementations...")
    harmony_original = load_module("harmony_original", example_dir / "harmony.py")
    harmony_52 = load_module("harmony_52", example_dir / "results_pbmc" / "harmony_52.py")

    # Run original harmony with tracking
    print("\nRunning original Harmony with iteration tracking...")
    _, metrics_original = run_harmony_with_tracking(
        harmony_original, X, batch_labels, cell_types, n_clusters=50, max_iter=10
    )

    # Run optimized harmony with tracking
    print("\nRunning Harmony #52 with iteration tracking...")
    _, metrics_52 = run_harmony_with_tracking(
        harmony_52, X, batch_labels, cell_types, n_clusters=50, max_iter=10
    )

    # Convert to arrays
    iters_orig = [m['iteration'] for m in metrics_original]
    mixing_orig = [m['mixing_score'] for m in metrics_original]
    bio_orig = [m['bio_conservation'] for m in metrics_original]

    iters_52 = [m['iteration'] for m in metrics_52]
    mixing_52 = [m['mixing_score'] for m in metrics_52]
    bio_52 = [m['bio_conservation'] for m in metrics_52]

    # Create figure
    print("\nCreating convergence plot...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Mixing score convergence
    ax = axes[0]
    ax.plot(iters_orig, mixing_orig, 'o-', color='#58a6ff', linewidth=2, markersize=8, label='Original Harmony')
    ax.plot(iters_52, mixing_52, 's-', color='#3fb950', linewidth=2, markersize=8, label='Harmony #52')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Batch Mixing Score', fontsize=12)
    ax.set_title('Batch Mixing Convergence', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 10.5)

    # Plot 2: Bio conservation convergence
    ax = axes[1]
    ax.plot(iters_orig, bio_orig, 'o-', color='#58a6ff', linewidth=2, markersize=8, label='Original Harmony')
    ax.plot(iters_52, bio_52, 's-', color='#3fb950', linewidth=2, markersize=8, label='Harmony #52')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Bio Conservation Score', fontsize=12)
    ax.set_title('Bio Conservation Convergence', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 10.5)

    # Plot 3: Trajectory in Mixing vs Bio space
    ax = axes[2]
    # Original Harmony trajectory
    ax.plot(mixing_orig, bio_orig, 'o-', color='#58a6ff', linewidth=2, markersize=8, label='Original Harmony')
    ax.scatter(mixing_orig[0], bio_orig[0], c='#58a6ff', s=150, marker='*', zorder=5, edgecolors='white', linewidths=1.5)
    ax.scatter(mixing_orig[-1], bio_orig[-1], c='#58a6ff', s=150, marker='D', zorder=5, edgecolors='white', linewidths=1.5)

    # Harmony #52 trajectory
    ax.plot(mixing_52, bio_52, 's-', color='#3fb950', linewidth=2, markersize=8, label='Harmony #52')
    ax.scatter(mixing_52[0], bio_52[0], c='#3fb950', s=150, marker='*', zorder=5, edgecolors='white', linewidths=1.5)
    ax.scatter(mixing_52[-1], bio_52[-1], c='#3fb950', s=150, marker='D', zorder=5, edgecolors='white', linewidths=1.5)

    ax.set_xlabel('Batch Mixing Score', fontsize=12)
    ax.set_ylabel('Bio Conservation Score', fontsize=12)
    ax.set_title('Optimization Trajectory\n(★=Start, ◆=End)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Add iteration labels on trajectory
    for i, (mx, by) in enumerate(zip(mixing_orig, bio_orig)):
        if i in [0, 5, 10]:
            ax.annotate(f'{i}', (mx, by), textcoords="offset points", xytext=(5, 5), fontsize=8, color='#58a6ff')
    for i, (mx, by) in enumerate(zip(mixing_52, bio_52)):
        if i in [0, 5, 10]:
            ax.annotate(f'{i}', (mx, by), textcoords="offset points", xytext=(5, -10), fontsize=8, color='#3fb950')

    plt.suptitle('TMA Data: Harmony Convergence Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    output_dir = example_dir / "results_pbmc"
    fig.savefig(output_dir / "tma_convergence_comparison.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: results_pbmc/tma_convergence_comparison.png")

    # Print summary
    print("\n" + "=" * 60)
    print("CONVERGENCE SUMMARY (TMA Data)")
    print("=" * 60)

    print(f"\nOriginal Harmony:")
    print(f"  Start (iter 0): mixing={mixing_orig[0]:.4f}, bio={bio_orig[0]:.4f}")
    print(f"  End (iter 10):  mixing={mixing_orig[-1]:.4f}, bio={bio_orig[-1]:.4f}")
    print(f"  Improvement:    mixing={mixing_orig[-1]-mixing_orig[0]:+.4f}, bio={bio_orig[-1]-bio_orig[0]:+.4f}")

    print(f"\nHarmony #52:")
    print(f"  Start (iter 0): mixing={mixing_52[0]:.4f}, bio={bio_52[0]:.4f}")
    print(f"  End (iter 10):  mixing={mixing_52[-1]:.4f}, bio={bio_52[-1]:.4f}")
    print(f"  Improvement:    mixing={mixing_52[-1]-mixing_52[0]:+.4f}, bio={bio_52[-1]-bio_52[0]:+.4f}")

    return metrics_original, metrics_52


if __name__ == "__main__":
    plot_convergence()
