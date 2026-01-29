#!/usr/bin/env python
"""
Code Distillation via Evolution.

Evolve Python code to match a black-box ML model's predictions.
Uses Pantheon Evolution framework with MAP-Elites.

Usage:
    python run.py [--iterations N] [--output DIR]
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Set data directory for evaluator (must be absolute path)
example_dir = Path(__file__).resolve().parent
os.environ["CODE_DISTILLATION_DATA_DIR"] = str(example_dir / "data")


async def run_evolution(
    iterations: int = 100,
    output_dir: str = None,
    resume: str = None,
):
    """Run code distillation evolution."""
    from pantheon.evolution import EvolutionTeam, EvolutionConfig
    from pantheon.evolution.program import CodebaseSnapshot

    example_dir = Path(__file__).parent

    # Load initial code and evaluator
    initial_code = CodebaseSnapshot.from_single_file(
        "distilled_code.py",
        (example_dir / "distilled_code.py").read_text()
    )
    evaluator_code = (example_dir / "evaluator.py").read_text()

    # Configuration
    output_path = example_dir / "results" if output_dir is None else Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = EvolutionConfig(
        max_iterations=iterations,
        num_workers=4,
        num_islands=2,
        num_inspirations=2,
        num_top_programs=3,
        max_parallel_evaluations=2,
        evaluation_timeout=120,
        analyzer_timeout=120,
        feature_dimensions=["fidelity"],
        early_stop_generations=50,
        function_weight=1.0,
        llm_weight=0.0,
        log_level="INFO",
        checkpoint_interval=10,
        db_path=str(output_path),
    )

    # Optimization objective
    objective = """Improve the distilled classifier to match CellTypist predictions.

## Current Status
- The code already has scoring formulas for 10 cell types
- Current fidelity is shown in evaluation metrics
- Goal: increase fidelity to >= 95%

## Cell Types in Test Data (EXACT NAMES - must match exactly):
1. Plasma cells (202) - markers: JCHAIN, MZB1, XBP1
2. Mast cells (200) - markers: TPSAB1, CPA3, TPSB2
3. DC1 (200) - markers: CLEC9A, XCR1, CADM1
4. Kupffer cells (199) - markers: TIMD4, MARCO, CD163
5. pDC (198) - markers: LILRA4, IL3RA, CLEC4C
6. gamma-delta T cells (197) - markers: TRDC, TRGC1
7. Endothelial cells (197) - markers: VWF, CDH5, PECAM1
8. Follicular B cells (196) - markers: MS4A1, CD79A
9. Alveolar macrophages (184) - markers: FABP4, MRC1
10. Neutrophil-myeloid progenitor (182) - markers: MPO, ELANE

## Improvement Strategies
1. Adjust weights - the current weights (0.5) are guesses, tune them
2. Add more marker genes per cell type
3. Add negative markers (genes that should be LOW for a cell type)
4. Add bias terms to shift decision boundaries

## Code Pattern
```python
scores["Cell Type"] = bias  # e.g., -1.0
scores["Cell Type"] += w1 * expression.get("GENE1", 0)  # positive marker
scores["Cell Type"] -= w2 * expression.get("GENE2", 0)  # negative marker
```

## Constraints
- Use EXACT cell type names as shown above
- Keep the scoring pattern (no if-else chains)
- Return max(scores, key=scores.get)
"""

    print("=" * 60)
    print("Code Distillation via Evolution")
    print("=" * 60)
    print(f"Model: CellTypist Immune_All_Low.pkl")
    print(f"Iterations: {iterations}")
    print(f"Output: {output_path}")
    print()

    # Run evolution
    team = EvolutionTeam(config=config)
    result = await team.evolve(
        initial_code=initial_code,
        evaluator_code=evaluator_code,
        objective=objective,
        resume_from=resume,
    )

    # Save results
    print("\n" + "=" * 60)
    print(result.get_summary())

    # Save best code
    best_code_path = output_path / "distilled_code_best.py"
    best_code_path.write_text(result.best_code)
    print(f"\nBest code saved to: {best_code_path}")

    # Also update the main distilled_code.py
    (example_dir / "distilled_code.py").write_text(result.best_code)

    return result


def main():
    parser = argparse.ArgumentParser(description="Code Distillation via Evolution")
    parser.add_argument("--iterations", "-n", type=int, default=100)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--resume", "-r", type=str, default=None)

    args = parser.parse_args()

    try:
        result = asyncio.run(run_evolution(
            iterations=args.iterations,
            output_dir=args.output,
            resume=args.resume,
        ))
        print(f"\nFinal fidelity: {result.best_score:.1%}")
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
