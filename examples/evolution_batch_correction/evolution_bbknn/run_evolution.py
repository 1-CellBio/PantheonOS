#!/usr/bin/env python
"""
Run Pantheon Evolution on the BBKNN Algorithm.

This script demonstrates how to use Pantheon Evolution to optimize
a multi-file batch correction algorithm. BBKNN (Batch Balanced KNN)
modifies the k-nearest neighbor graph to account for batch effects.

Usage:
    python run_evolution.py [--iterations N] [--output DIR]

Example:
    python run_evolution.py --iterations 50 --output results/
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Load .env file from the example directory
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Set environment variables so evaluator can find shared modules and data
_example_dir = Path(__file__).parent.resolve()
os.environ.setdefault("BBKNN_DATA_DIR", str(_example_dir.parent / "data"))
os.environ.setdefault("METRICS_MODULE_PATH", str(_example_dir.parent / "metrics.py"))


async def run_evolution(
    iterations: int = 50,
    output_dir: str = None,
    verbose: bool = False,
    resume: str = None,
):
    """
    Run the evolution process.

    Args:
        iterations: Maximum number of evolution iterations
        output_dir: Directory to save results
        verbose: Enable verbose logging
        resume: Path to resume evolution from
    """
    from pantheon.evolution import EvolutionTeam, EvolutionConfig
    from pantheon.evolution.program import CodebaseSnapshot

    # Get paths
    example_dir = Path(__file__).parent
    bbknn_dir = example_dir / "bbknn"
    evaluator_path = example_dir / "evaluator.py"

    # Load initial code as multi-file CodebaseSnapshot
    initial_code = CodebaseSnapshot.from_directory(
        str(bbknn_dir),
        include_patterns=["*.py"],
    )
    evaluator_code = evaluator_path.read_text()

    print(f"Loaded {initial_code.file_count()} files, {initial_code.total_lines()} lines")
    for path in sorted(initial_code.files.keys()):
        lines = len(initial_code.files[path].split('\n'))
        print(f"  - {path}: {lines} lines")

    # Load configuration from file if exists, otherwise create default
    config_path = Path(output_dir) / "config.yaml" if output_dir else None
    if config_path and config_path.exists():
        config = EvolutionConfig.from_yaml(str(config_path))
        config.max_iterations = iterations
        config.log_level = "DEBUG" if verbose else "INFO"
        print(f"Loaded config from: {config_path}")
    else:
        # Create configuration optimized for multi-file evolution
        config = EvolutionConfig(
            max_iterations=iterations,
            num_workers=8,
            num_islands=8,
            num_inspirations=1,
            num_top_programs=2,
            max_parallel_evaluations=2,
            evaluation_timeout=180,
            analyzer_timeout=120,
            feature_dimensions=["mixing_score", "speed_score", "bio_conservation_score"],
            early_stop_generations=100,
            function_weight=1.0,
            llm_weight=0.0,
            log_level="DEBUG" if verbose else "INFO",
            log_iterations=True,
            checkpoint_interval=10,
            db_path=output_dir,
            max_code_length=80000,
        )

    # Define optimization objective
    objective = """Optimize the BBKNN (Batch Balanced KNN) algorithm for:

1. **Integration Quality** (45% weight): Improve batch mixing while preserving biological structure.
   - BBKNN identifies each cell's top neighbors in each batch separately
   - The `get_graph()` function constructs the batch-balanced KNN graph
   - The `compute_connectivities_umap()` function computes the fuzzy simplicial set

2. **Biological Conservation** (45% weight): Preserve biological variance.
   - Don't over-correct and remove biological signal
   - Maintain cell type separation in the resulting graph
   - The `trimming()` function affects connectivity strength

3. **Performance** (10% weight): Reduce execution time.
   - The `create_tree()` function builds KNN indices (uses cKDTree by default)
   - The `query_tree()` function performs neighbor queries
   - Consider algorithmic improvements to `get_graph()`

## File Structure (2 files):

### bbknn/__init__.py (High-level API)
- `bbknn()`: Main AnnData-based entry point (wraps matrix.bbknn)
- `ridge_regression()`: Optional preprocessing
- `extract_cell_connectivity()`: Helper for visualization

### bbknn/matrix.py (Core Algorithm)
Key functions to optimize:
- `bbknn()`: Scanpy-independent entry point (takes pca and batch_list)
- `get_graph()`: Constructs the batch-balanced KNN graph
- `create_tree()`: Creates KNN index for each batch
- `query_tree()`: Queries neighbors from each batch
- `compute_connectivities_umap()`: Computes fuzzy simplicial set
- `trimming()`: Trims connectivities to top values

## Key Parameters:
- neighbors_within_batch (3): Top neighbors per batch
- n_pcs (50): PCA dimensions to use
- trim: Connectivity trimming threshold
- metric: Distance metric (euclidean, manhattan, etc.)
- set_op_mix_ratio: UMAP fuzzy set mixing parameter
- local_connectivity: UMAP local connectivity parameter

## Constraints:
- Keep the public API (bbknn, matrix.bbknn function signatures)
- Maintain compatibility with sparse matrix outputs
- Don't break imports between files
- Ensure numerical stability

## Areas for Algorithm-Level Improvement:
- The neighbor selection strategy in get_graph() could be improved
- The connectivity computation could use different kernels
- Adaptive neighbors_within_batch based on batch sizes
- Different distance metrics or weighted combinations
- Improved trimming strategies
"""

    print("=" * 60)
    print("Pantheon Evolution: BBKNN Algorithm Optimization")
    print("=" * 60)
    print(f"\nInitial code: {bbknn_dir} ({initial_code.file_count()} files)")
    print(f"Evaluator: {evaluator_path}")
    print(f"Iterations: {iterations}")
    print(f"Output: {output_dir or 'None (results not saved)'}")
    if resume:
        print(f"Resume from: {resume}")
    print("\n" + "-" * 60)
    print("Starting evolution...\n")

    # Create and run evolution team
    team = EvolutionTeam(config=config)
    result = await team.evolve(
        initial_code=initial_code,
        evaluator_code=evaluator_code,
        objective=objective,
        resume_from=resume,
    )

    # Print results
    print("\n" + "=" * 60)
    print(result.get_summary())

    # Save results if output specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save best code (all files)
        best_program = result.best_program
        if best_program:
            optimized_dir = output_path / "bbknn_optimized"
            optimized_dir.mkdir(exist_ok=True)
            best_program.snapshot.to_workspace(str(optimized_dir))
            print(f"\nBest code saved to: {optimized_dir}")

        # Save report
        report_path = output_path / "evolution_report.json"
        result.save_report(str(report_path))
        print(f"Report saved to: {report_path}")

        # Save configuration
        config.to_yaml(str(output_path / "config.yaml"))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evolve the BBKNN algorithm using Pantheon Evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  python run_evolution.py --iterations 10

  # Full evolution with output
  python run_evolution.py --iterations 100 --output results/

  # Verbose mode
  python run_evolution.py --iterations 50 --verbose

  # Resume from checkpoint
  python run_evolution.py --iterations 100 --output results/ --resume results/
        """,
    )

    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=50,
        help="Maximum number of evolution iterations (default: 50)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Resume evolution from checkpoint directory",
    )

    args = parser.parse_args()

    try:
        result = asyncio.run(run_evolution(
            iterations=args.iterations,
            output_dir=args.output,
            verbose=args.verbose,
            resume=args.resume,
        ))
        print(f"\nFinal best score: {result.best_score:.4f}")
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
