#!/usr/bin/env python
"""
Compare original Harmony vs best evolved Harmony (#209) on TMA data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import importlib.util
import sys
from pathlib import Path
from umap import UMAP
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


def load_tma_data(split: str = "train"):
    """Load TMA data with real cell type labels."""
    df = pd.read_csv(data_dir / f"tma_8000_{split}.csv")
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


def run_comparison():
    """Run the comparison and create plots."""
    print("Loading TMA data...")
    X, batch_labels, cell_types = load_tma_data("train")
    print(f"  Data shape: {X.shape}")
    print(f"  Batches: {np.unique(batch_labels)}")
    print(f"  Cell types: {len(np.unique(cell_types))}")

    # Load both harmony implementations
    print("\nLoading Harmony implementations...")
    harmony_original = load_module("harmony_original", example_dir / "harmony.py")
    harmony_best = load_module("harmony_best", example_dir / "results_pbmc" / "harmony_best.py")

    # Run original harmony
    print("\nRunning original Harmony...")
    start_time = time.time()
    hm_original = harmony_original.run_harmony(
        X, batch_labels, n_clusters=50, max_iter=10, random_state=42
    )
    time_original = time.time() - start_time
    X_corrected_original = hm_original.Z_corr
    print(f"  Time: {time_original:.2f}s")

    # Run best harmony
    print("\nRunning best evolved Harmony (#209)...")
    start_time = time.time()
    hm_best = harmony_best.run_harmony(
        X, batch_labels, n_clusters=50, max_iter=10, random_state=42
    )
    time_best = time.time() - start_time
    X_corrected_best = hm_best.Z_corr
    print(f"  Time: {time_best:.2f}s")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = {
        "Original": {
            "mixing_score": compute_batch_mixing_score(X, batch_labels),
            "bio_score": compute_bio_conservation_score(X, cell_types),
            "time": 0,
        },
        "Harmony": {
            "mixing_score": compute_batch_mixing_score(X_corrected_original, batch_labels),
            "bio_score": compute_bio_conservation_score(X_corrected_original, cell_types),
            "time": time_original,
        },
        "Harmony Best": {
            "mixing_score": compute_batch_mixing_score(X_corrected_best, batch_labels),
            "bio_score": compute_bio_conservation_score(X_corrected_best, cell_types),
            "time": time_best,
        },
    }

    for name, m in metrics.items():
        print(f"  {name}: mixing={m['mixing_score']:.4f}, bio={m['bio_score']:.4f}, time={m['time']:.2f}s")

    # Compute UMAP embeddings
    print("\nComputing UMAP embeddings...")
    umap = UMAP(n_neighbors=30, min_dist=0.3, random_state=42)

    print("  Original...")
    umap_original = umap.fit_transform(X)

    print("  Harmony corrected...")
    umap_harmony = umap.fit_transform(X_corrected_original)

    print("  Best Harmony corrected...")
    umap_best = umap.fit_transform(X_corrected_best)

    # Create output directory
    output_dir = example_dir / "results_pbmc"

    # Create figure 1: UMAP comparisons
    print("\nCreating UMAP comparison figure...")
    fig1, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Color maps
    batch_cmap = plt.cm.Set1
    cluster_cmap = plt.cm.tab10

    unique_batches = np.unique(batch_labels)
    unique_clusters = np.unique(cell_types)
    batch_colors = {b: batch_cmap(i / len(unique_batches)) for i, b in enumerate(unique_batches)}
    cluster_colors = {c: cluster_cmap(i % 10) for i, c in enumerate(unique_clusters)}

    # Row 1: Color by batch
    datasets = [
        (umap_original, "Original", metrics["Original"]),
        (umap_harmony, "Harmony", metrics["Harmony"]),
        (umap_best, "Harmony Best (#209)", metrics["Harmony Best"]),
    ]

    for idx, (umap_emb, title, m) in enumerate(datasets):
        ax = axes[0, idx]
        for batch in unique_batches:
            mask = batch_labels == batch
            ax.scatter(umap_emb[mask, 0], umap_emb[mask, 1],
                      c=[batch_colors[batch]], label=batch, s=5, alpha=0.6)
        ax.set_title(f"{title}\nMixing: {m['mixing_score']:.3f}", fontsize=11)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        if idx == 2:
            ax.legend(title="Batch", markerscale=3, loc='upper right', fontsize=9)

    # Row 2: Color by cell type
    for idx, (umap_emb, title, m) in enumerate(datasets):
        ax = axes[1, idx]
        for ct in unique_clusters:
            mask = cell_types == ct
            ax.scatter(umap_emb[mask, 0], umap_emb[mask, 1],
                      c=[cluster_colors[ct]], label=ct, s=5, alpha=0.6)
        ax.set_title(f"{title}\nBio Conservation: {m['bio_score']:.3f}", fontsize=11)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        if idx == 2:
            ax.legend(title="Cell Type", markerscale=3, loc='upper right', fontsize=7, ncol=2)

    axes[0, 0].set_ylabel("Color by Batch\nUMAP2")
    axes[1, 0].set_ylabel("Color by Cell Type\nUMAP2")

    plt.suptitle("TMA Data: Batch Effect Correction Comparison (Best Evolved Program)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig1.savefig(output_dir / "tma_umap_comparison_best.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: results_pbmc/tma_umap_comparison_best.png")

    # Create figure 2: Performance comparison
    print("\nCreating performance comparison figure...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))

    methods = ["Original", "Harmony", "Harmony\nBest (#209)"]
    colors = ["#8b949e", "#58a6ff", "#f85149"]

    # Mixing score
    ax = axes2[0]
    mixing_scores = [metrics["Original"]["mixing_score"],
                     metrics["Harmony"]["mixing_score"],
                     metrics["Harmony Best"]["mixing_score"]]
    bars = ax.bar(methods, mixing_scores, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel("Batch Mixing Score")
    ax.set_title("Batch Integration Quality\n(Higher = Better)")
    ax.set_ylim(0, 1)
    for bar, score in zip(bars, mixing_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Bio conservation score
    ax = axes2[1]
    bio_scores = [metrics["Original"]["bio_score"],
                  metrics["Harmony"]["bio_score"],
                  metrics["Harmony Best"]["bio_score"]]
    bars = ax.bar(methods, bio_scores, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel("Bio Conservation Score")
    ax.set_title("Biological Structure Preservation\n(Higher = Better)")
    ax.set_ylim(0, 1)
    for bar, score in zip(bars, bio_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Execution time (only for Harmony methods)
    ax = axes2[2]
    time_methods = ["Harmony", "Harmony\nBest (#209)"]
    times = [metrics["Harmony"]["time"], metrics["Harmony Best"]["time"]]
    bars = ax.bar(time_methods, times, color=["#58a6ff", "#f85149"], edgecolor='white', linewidth=1.5)
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Computational Performance\n(Lower = Better)")
    speedup = time_original / time_best if time_best > 0 else 1
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{t:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.95, f"Speedup: {speedup:.2f}x", transform=ax.transAxes,
           ha='center', va='top', fontsize=11, color='#f85149', fontweight='bold')

    plt.suptitle("TMA Data: Harmony vs Best Evolved Harmony Performance", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig2.savefig(output_dir / "tma_performance_comparison_best.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: results_pbmc/tma_performance_comparison_best.png")

    # Create figure 3: Combined score comparison
    print("\nCreating combined score figure...")
    fig3, ax3 = plt.subplots(figsize=(8, 6))

    # Compute combined scores (same weights as evaluator)
    combined_original = 0.45 * metrics["Original"]["mixing_score"] + 0.45 * metrics["Original"]["bio_score"]
    combined_harmony = 0.45 * metrics["Harmony"]["mixing_score"] + 0.45 * metrics["Harmony"]["bio_score"]
    combined_best = 0.45 * metrics["Harmony Best"]["mixing_score"] + 0.45 * metrics["Harmony Best"]["bio_score"]

    categories = ["Mixing\nScore", "Bio\nConservation", "Combined\nScore"]
    original_vals = [metrics["Original"]["mixing_score"], metrics["Original"]["bio_score"], combined_original]
    harmony_vals = [metrics["Harmony"]["mixing_score"], metrics["Harmony"]["bio_score"], combined_harmony]
    best_vals = [metrics["Harmony Best"]["mixing_score"], metrics["Harmony Best"]["bio_score"], combined_best]

    x = np.arange(len(categories))
    width = 0.25

    bars1 = ax3.bar(x - width, original_vals, width, label='Original', color='#8b949e', edgecolor='white')
    bars2 = ax3.bar(x, harmony_vals, width, label='Harmony', color='#58a6ff', edgecolor='white')
    bars3 = ax3.bar(x + width, best_vals, width, label='Harmony Best (#209)', color='#f85149', edgecolor='white')

    ax3.set_ylabel('Score')
    ax3.set_title('TMA Data: Comprehensive Performance Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend(loc='upper left')
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig3.savefig(output_dir / "tma_combined_comparison_best.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: results_pbmc/tma_combined_comparison_best.png")

    print("\nDone! Generated 3 figures in results_pbmc/")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (TMA Data - Best Evolved Program)")
    print("=" * 60)
    print(f"\nMixing Score:")
    print(f"  Original → Harmony:      {metrics['Original']['mixing_score']:.3f} → {metrics['Harmony']['mixing_score']:.3f} ({(metrics['Harmony']['mixing_score'] - metrics['Original']['mixing_score']) / metrics['Original']['mixing_score'] * 100:+.1f}%)")
    print(f"  Original → Harmony Best: {metrics['Original']['mixing_score']:.3f} → {metrics['Harmony Best']['mixing_score']:.3f} ({(metrics['Harmony Best']['mixing_score'] - metrics['Original']['mixing_score']) / metrics['Original']['mixing_score'] * 100:+.1f}%)")

    print(f"\nBio Conservation:")
    print(f"  Original → Harmony:      {metrics['Original']['bio_score']:.3f} → {metrics['Harmony']['bio_score']:.3f} ({(metrics['Harmony']['bio_score'] - metrics['Original']['bio_score']) / metrics['Original']['bio_score'] * 100:+.1f}%)")
    print(f"  Original → Harmony Best: {metrics['Original']['bio_score']:.3f} → {metrics['Harmony Best']['bio_score']:.3f} ({(metrics['Harmony Best']['bio_score'] - metrics['Original']['bio_score']) / metrics['Original']['bio_score'] * 100:+.1f}%)")

    print(f"\nExecution Time:")
    print(f"  Harmony:      {time_original:.2f}s")
    print(f"  Harmony Best: {time_best:.2f}s")
    print(f"  Speedup:      {speedup:.2f}x")

    return metrics


if __name__ == "__main__":
    run_comparison()
