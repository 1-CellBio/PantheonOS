"""
Evaluator for Code Distillation.

This evaluator measures how well the distilled code matches the original model.
"""

import numpy as np


def evaluate(workspace_path: str) -> dict:
    """
    Evaluate distilled code against CellTypist model.

    Args:
        workspace_path: Path containing distilled_code.py

    Returns:
        Dict with metrics:
        - fidelity: Agreement rate with original model (0-1)
        - complexity: Number of lines of code (lower is better)
        - coverage: Fraction of samples not returning "Unknown"
    """
    import sys
    import os
    from pathlib import Path

    # Add workspace to path
    sys.path.insert(0, workspace_path)

    # Import the distilled function
    try:
        # Clear cached module if exists
        if 'distilled_code' in sys.modules:
            del sys.modules['distilled_code']

        from distilled_code import predict_cell_type
    except ImportError as e:
        return {
            "fidelity": 0.0,
            "coverage": 0.0,
            "complexity": 1000,
            "error": f"Import error: {e}",
            "fitness_weights": {"fidelity": 1.0},
        }
    except Exception as e:
        return {
            "fidelity": 0.0,
            "coverage": 0.0,
            "complexity": 1000,
            "error": f"Load error: {e}",
            "fitness_weights": {"fidelity": 1.0},
        }

    # Load test data and model (cached globally for speed)
    X_test, features, original_labels = _get_test_data()

    # Run distilled predictions
    n_samples = len(X_test)
    correct = 0
    unknown = 0

    for i in range(n_samples):
        expr = dict(zip(features, X_test[i]))
        try:
            pred = predict_cell_type(expr)
        except Exception:
            pred = "Error"

        if pred == original_labels[i]:
            correct += 1
        if pred == "Unknown":
            unknown += 1

    fidelity = correct / n_samples
    coverage = 1.0 - (unknown / n_samples)

    # Measure code complexity (lines of code)
    code_path = Path(workspace_path) / "distilled_code.py"
    if code_path.exists():
        code = code_path.read_text()
        # Count non-empty, non-comment lines
        lines = [l for l in code.split('\n')
                 if l.strip() and not l.strip().startswith('#')]
        complexity = len(lines)
    else:
        complexity = 1000

    # Combined fitness score
    # Prioritize fidelity, but reward simplicity
    # fitness = fidelity * 0.8 + (1 - complexity/200) * 0.2
    fitness_score = fidelity  # Keep it simple: just fidelity

    return {
        "fidelity": fidelity,
        "coverage": coverage,
        "complexity": complexity,
        "fitness_score": fitness_score,
        "n_correct": correct,
        "n_samples": n_samples,
        # Required for evolution framework: specify which metrics to use for fitness
        "fitness_weights": {"fidelity": 1.0},
    }


# Global cache for test data (in-memory)
_cached_data = None

# Disk cache path for pre-computed test data (avoids loading model in each subprocess)
_CACHE_FILE = "/Users/wzxu/Projects/Pantheon/pantheon-agents/examples/code_distillation/data/test_data_cache.npz"


def _get_test_data():
    """Load test data from disk cache or compute it once."""
    global _cached_data

    if _cached_data is not None:
        return _cached_data

    from pathlib import Path
    import os

    cache_path = Path(_CACHE_FILE)

    # Try to load from disk cache first (fast, no model loading needed)
    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=True)
            X_test = cached['X_test']
            features = cached['features'].tolist()
            labels = cached['labels']
            _cached_data = (X_test, features, labels)
            return _cached_data
        except Exception as e:
            print(f"Warning: Failed to load cache, regenerating: {e}")

    # Cache miss - compute and save (this loads the model, but only once)
    import scanpy as sc
    import celltypist
    from celltypist import models

    # Data directory
    data_dir = cache_path.parent
    data_dir.mkdir(exist_ok=True)

    # Download demo data
    demo_file = data_dir / "demo_2000_cells.h5ad"
    adata = sc.read(
        demo_file,
        backup_url='https://celltypist.cog.sanger.ac.uk/Notebook_demo_data/demo_2000_cells.h5ad'
    )

    # Load model
    model_name = "Immune_All_Low.pkl"
    models.download_models(model=model_name, force_update=False)
    model = models.Model.load(model=model_name)

    # Get model features
    model_features = list(model.features)

    # Align expression matrix to model features
    if hasattr(adata, 'raw') and adata.raw is not None:
        X_full = adata.raw.X
        var_names = list(adata.raw.var_names)
    else:
        X_full = adata.X
        var_names = list(adata.var_names)

    if hasattr(X_full, 'toarray'):
        X_full = X_full.toarray()

    X_test = np.zeros((X_full.shape[0], len(model_features)))
    for i, gene in enumerate(model_features):
        if gene in var_names:
            idx = var_names.index(gene)
            X_test[:, i] = X_full[:, idx]

    # Get original predictions
    predictions = celltypist.annotate(adata, model=model_name)
    original_labels = predictions.predicted_labels.predicted_labels.values

    # Save to disk cache for future runs (silent in subprocess mode)
    try:
        np.savez(cache_path, X_test=X_test, features=np.array(model_features), labels=original_labels)
        # Only print if running as main script, not in subprocess
        if __name__ == "__main__":
            print(f"Test data cached to {cache_path}")
    except Exception:
        pass  # Silently ignore cache save failures

    _cached_data = (X_test, model_features, original_labels)
    return _cached_data


# Only run test code when executed directly as a script (not when inlined in subprocess)
# The subprocess execution sets __name__ = "__main__" but also defines evaluate() after this code
# So we need to check if we're being run as a standalone file
if __name__ == "__main__" and len(__import__('sys').argv) > 0 and __import__('sys').argv[0].endswith('evaluator.py'):
    # Test evaluator (only when running: python evaluator.py [workspace])
    import sys
    workspace = sys.argv[1] if len(sys.argv) > 1 else "."
    result = evaluate(workspace)
    print(f"Fidelity: {result['fidelity']:.1%}")
    print(f"Coverage: {result['coverage']:.1%}")
    print(f"Complexity: {result['complexity']} lines")
