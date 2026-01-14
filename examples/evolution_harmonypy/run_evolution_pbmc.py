#!/usr/bin/env python
"""
Run Pantheon Evolution on the Harmony Algorithm using PBMC dataset.

This script runs evolution with PBMC 3500 dataset for training.
Results will be saved to results_pbmc/ directory.

Usage:
    python run_evolution_pbmc.py [--iterations N] [--resume]

Example:
    python run_evolution_pbmc.py --iterations 500
    python run_evolution_pbmc.py --iterations 500 --resume
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

# Set HARMONY_DATA_DIR so evaluator can find data when running in temp workspace
_example_dir = Path(__file__).parent.resolve()
os.environ.setdefault("HARMONY_DATA_DIR", str(_example_dir / "data"))


async def run_evolution(
    iterations: int = 500,
    verbose: bool = False,
    resume: bool = False,
):
    """
    Run the evolution process with PBMC dataset.

    Args:
        iterations: Maximum number of evolution iterations
        verbose: Enable verbose logging
        resume: Resume from existing checkpoint
    """
    from pantheon.evolution import EvolutionTeam, EvolutionConfig
    from pantheon.evolution.program import CodebaseSnapshot

    # Get paths
    example_dir = Path(__file__).parent
    harmony_path = example_dir / "harmony.py"
    evaluator_path = example_dir / "evaluator_pbmc.py"  # Use PBMC evaluator
    output_dir = example_dir / "results_pbmc"

    # Load initial code and evaluator
    initial_code = CodebaseSnapshot.from_single_file("harmony.py", harmony_path.read_text())
    evaluator_code = evaluator_path.read_text()

    # Load configuration from file if exists, otherwise create default
    config_path = output_dir / "config.yaml"
    if resume and config_path.exists():
        config = EvolutionConfig.from_yaml(str(config_path))
        config.max_iterations = iterations
        config.log_level = "DEBUG" if verbose else "INFO"
        print(f"Resuming from: {output_dir}")
    else:
        # Create configuration optimized for PBMC dataset
        config = EvolutionConfig(
            max_iterations=iterations,
            checkpoint_interval=10,
            early_stop_generations=200,
            num_workers=4,
            num_islands=3,
            migration_interval=20,
            migration_rate=0.1,
            feature_dimensions=["mixing_score", "speed_score", "bio_conservation_score"],
            feature_bins=10,
            feature_range_padding=0.1,
            feature_range_adaptive=True,
            archive_ratio=0.25,
            population_size=500,
            num_inspirations=2,
            num_top_programs=3,
            exploration_ratio=0.2,
            exploitation_ratio=0.7,
            evaluation_timeout=120,
            max_parallel_evaluations=2,
            function_weight=1.0,
            llm_weight=0.0,
            cascade_evaluation=False,
            diff_based_evolution=True,
            max_code_length=50000,
            max_diff_size=5000,
            temperature=0.7,
            max_retries=3,
            mutation_timeout=120,
            use_analyzer=True,
            analyzer_model="normal",
            analyzer_timeout=180,
            top_programs_probability=0.5,
            inspirations_probability=0.2,
            mutator_model="normal",
            feedback_model="normal",
            db_path=str(output_dir),
            save_prompts=True,
            save_all_programs=False,
            log_level="DEBUG" if verbose else "INFO",
            log_iterations=True,
            log_improvements=True,
        )

    # Define optimization objective
    objective = """Optimize the Harmony algorithm implementation for:

1. **Integration Quality** (45% weight): Improve batch mixing while preserving biological structure.
   - The algorithm should effectively remove batch effects
   - Biological clusters should remain distinct after correction

2. **Biological Conservation** (45% weight): Preserve biological variance.
   - Don't over-correct and remove biological signal
   - Maintain cluster separation (measured via silhouette score on pseudo-labels)

3. **Performance** (5% weight): Reduce execution time.
   - Optimize hot loops and matrix operations
   - Consider vectorization opportunities

4. **Convergence** (5% weight): Improve convergence behavior.
   - Reduce number of iterations needed
   - Ensure stable convergence

Key areas to consider:
- The _update_R() method computes soft cluster assignments
- The _correct() method applies linear corrections
- Ridge regression in _correct() could be optimized
- The diversity penalty in _update_R() balances batch mixing

Constraints:
- Keep the public API (run_harmony function signature)
- Maintain numerical stability
- Don't remove essential functionality
"""

    print("=" * 60)
    print("Pantheon Evolution: Harmony Algorithm (PBMC Dataset)")
    print("=" * 60)
    print(f"\nInitial code: {harmony_path}")
    print(f"Evaluator: {evaluator_path}")
    print(f"Iterations: {iterations}")
    print(f"Output: {output_dir}")
    if resume:
        print("Mode: RESUME")
    print("\n" + "-" * 60)
    print("Starting evolution...\n")

    # Create and run evolution team
    team = EvolutionTeam(config=config)
    result = await team.evolve(
        initial_code=initial_code,
        evaluator_code=evaluator_code,
        objective=objective,
        resume_from=str(output_dir) if resume else None,
    )

    # Print results
    print("\n" + "=" * 60)
    print(result.get_summary())

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best code
    best_code_path = output_dir / "harmony_optimized.py"
    best_code_path.write_text(result.best_code)
    print(f"\nBest code saved to: {best_code_path}")

    # Save report
    report_path = output_dir / "evolution_report.json"
    result.save_report(str(report_path))
    print(f"Report saved to: {report_path}")

    # Save configuration
    config.to_yaml(str(output_dir / "config.yaml"))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evolve Harmony using PBMC dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=500,
        help="Maximum iterations (default: 500)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from existing results_pbmc/ checkpoint",
    )

    args = parser.parse_args()

    try:
        result = asyncio.run(run_evolution(
            iterations=args.iterations,
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
