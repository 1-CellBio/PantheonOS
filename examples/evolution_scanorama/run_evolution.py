#!/usr/bin/env python
"""
Run Pantheon Evolution on the Scanorama Algorithm.

This script demonstrates how to use Pantheon Evolution to optimize
a multi-file batch correction algorithm. The evolution process will:

1. Generate mutations of the scanorama package (3 files)
2. Evaluate each mutation on TMA single-cell data
3. Select the best-performing variants
4. Iterate until convergence or max iterations

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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env file from the example directory
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Set data dir so evaluator can find data when running in temp workspace
_example_dir = Path(__file__).parent.resolve()
os.environ.setdefault("SCANORAMA_DATA_DIR", str(_example_dir.parent / "evolution_harmonypy" / "data"))


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
    scanorama_dir = example_dir / "scanorama"
    evaluator_path = example_dir / "evaluator.py"

    # Load initial code as multi-file CodebaseSnapshot
    # This is the key difference from single-file evolution
    initial_code = CodebaseSnapshot.from_directory(
        str(scanorama_dir),
        include_patterns=["*.py"],
    )
    evaluator_code = evaluator_path.read_text()

    print(f"Loaded {initial_code.file_count()} files, {initial_code.total_lines()} total lines")
    for path in sorted(initial_code.files.keys()):
        lines = len(initial_code.files[path].split('\n'))
        print(f"  - {path}: {lines} lines")

    # Load configuration from file if output_dir has config.yaml, otherwise create default
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
            num_workers=2,  # Fewer workers for multi-file (more context per mutation)
            num_islands=2,  # Fewer islands
            num_inspirations=1,  # Fewer inspirations to save context
            num_top_programs=2,  # Fewer top programs
            max_parallel_evaluations=2,
            evaluation_timeout=180,  # Longer timeout for more complex code
            analyzer_timeout=120,
            feature_dimensions=["mixing_score", "speed_score", "bio_conservation_score"],
            early_stop_generations=100,
            function_weight=1.0,
            llm_weight=0.0,
            log_level="DEBUG" if verbose else "INFO",
            log_iterations=True,
            checkpoint_interval=10,
            db_path=output_dir,
            # Multi-file specific settings
            max_code_length=80000,  # Allow more code for 3 files
        )

    # Define optimization objective - specific to scanorama's structure
    objective = """Optimize the Scanorama batch correction algorithm for:

1. **Integration Quality** (45% weight): Improve batch mixing while preserving biological structure.
   - The algorithm uses mutual nearest neighbors (MNN) to find correspondences
   - The `assemble()` function applies nonlinear corrections using RBF kernels
   - The `transform()` function computes bias vectors for correction

2. **Biological Conservation** (45% weight): Preserve biological variance.
   - Don't over-correct and remove biological signal
   - Maintain cell type separation after correction

3. **Performance** (10% weight): Reduce execution time.
   - The `nn_approx()` function uses Annoy for approximate nearest neighbors
   - The `batch_bias()` function can be a bottleneck for large datasets
   - Consider vectorization opportunities in `mnn()` and `transform()`

## File Structure (3 files):

### scanorama/scanorama.py (Main Algorithm)
Key functions to consider optimizing:
- `integrate()` / `correct()`: Main entry points
- `assemble()`: Core panorama assembly (orchestrates alignment and correction)
- `find_alignments()`: Finds matching cells between datasets
- `transform()`: Computes nonlinear bias vectors using RBF kernel
- `mnn()`: Mutual nearest neighbor detection
- `nn_approx()`: Approximate nearest neighbor search
- `batch_bias()`: Computes smoothed bias vectors (potential bottleneck)

### scanorama/utils.py (Utilities)
- `reduce_dimensionality()`: PCA-based dimensionality reduction
- `handle_zeros_in_scale()`: Numerical stability helper

### scanorama/__init__.py (Package Init)
- Just exports from scanorama.py

## Key Parameters (in scanorama.py):
- ALPHA (0.10): Alignment score minimum cutoff
- KNN (20): Number of nearest neighbors
- SIGMA (15): RBF kernel smoothing parameter
- DIMRED (100): Dimensionality for integration

## Constraints:
- Keep the public API (integrate, correct function signatures)
- Maintain numerical stability (handle edge cases)
- Don't break imports between files
- Keep the package structure intact

## Areas for Algorithm-Level Improvement:
- The RBF kernel in transform() could use different kernel functions
- The alignment scoring in find_alignments() could be improved
- The panorama merging strategy in assemble() could be optimized
- Consider adaptive sigma based on local density
"""

    print("=" * 60)
    print("Pantheon Evolution: Scanorama Algorithm Optimization")
    print("=" * 60)
    print(f"\nInitial code: {scanorama_dir} ({initial_code.file_count()} files)")
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
            optimized_dir = output_path / "scanorama_optimized"
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
        description="Evolve the Scanorama algorithm using Pantheon Evolution",
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
