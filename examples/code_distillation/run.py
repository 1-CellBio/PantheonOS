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
        num_workers=4,  # Reduced to avoid Python process contention
        num_islands=2,
        num_inspirations=2,
        num_top_programs=3,
        max_parallel_evaluations=2,
        evaluation_timeout=120,
        analyzer_timeout=600,  # 10 minutes for Python analysis (model loading is slow)
        feature_dimensions=["fidelity"],
        early_stop_generations=50,
        function_weight=1.0,
        llm_weight=0.0,
        log_level="INFO",
        checkpoint_interval=10,
        db_path=str(output_path),
        # Enable Python interpreter for analyzer to inspect model weights
        analyzer_use_python=True,
        analyzer_python_workdir=str(example_dir),
    )

    # Optimization objective - HARD MODE with 26 cell types
    objective = """Distill the CellTypist classifier into interpretable Python code.

## Goal
Maximize fidelity (agreement rate with CellTypist model). Target: >= 90%

## HARD MODE: 26 Cell Types (EXACT NAMES required)
Must distinguish between similar subtypes - this is the real challenge!

Major types (~1955 cells):
- Plasma cells, DC1, Mast cells, Kupffer cells, pDC
- Endothelial cells, gamma-delta T cells, Follicular B cells
- Alveolar macrophages, Neutrophil-myeloid progenitor

Minor/rare types (~45 cells) - CRITICAL for true fidelity:
- Intermediate macrophages, HSC/MPP, Double-negative thymocytes
- Late erythroid, Macrophages, CD16- NK cells, Classical monocytes
- DC2, Monocyte precursor, CD16+ NK cells, Double-positive thymocytes
- ETP, Early erythroid, Mono-mac, Naive B cells, Plasmablasts

## Key Challenges
1. Distinguish Alveolar macrophages vs Kupffer cells vs Intermediate macrophages vs generic Macrophages
2. Distinguish DC1 vs DC2 vs pDC
3. Distinguish Follicular B cells vs Naive B cells vs Plasma cells vs Plasmablasts
4. Distinguish Classical monocytes vs Mono-mac vs Monocyte precursor
5. Identify rare progenitor types (HSC/MPP, ETP)
6. Identify erythroid stages (Early erythroid, Late erythroid)

## STRICT CONSTRAINTS (IMPORTANT!)
The distilled code must be SELF-CONTAINED and INDEPENDENT:
- DO NOT import celltypist or load any .pkl model files
- DO NOT load external weight files at runtime
- All decision logic must be hardcoded in the Python code itself
- The code should work without any external model files

## CRITICAL: Use Python Analysis to Discover Model Logic
The analyzer MUST use Python (`run_python_code` tool) to experimentally discover the model's decision logic.

**MODEL FILE**: `./Immune_All_Low.pkl` (CellTypist model, can be loaded with `celltypist.models.Model.load()`)

### Experimental Approaches (choose what works best)

1. **Perturbation Analysis**: Create synthetic samples, perturb gene values, observe how predictions change
   - Which genes cause prediction flips between similar cell types?
   - What are the threshold values where decisions change?

2. **Feature Importance**: Analyze which genes most strongly influence each cell type prediction
   - Don't just copy weights - understand the decision logic
   - Find sparse, interpretable rules

3. **Confusion Analysis**: For cell types that are often confused (e.g., DC1 vs DC2):
   - What distinguishes them in the model's view?
   - What gene combinations define the boundary?

4. **Decision Boundary Probing**: For ambiguous samples, what tips the balance?

### Goal
Discover **interpretable rules** that capture the model's behavior, not just copy raw weights.
The distilled code should embody the model's logic in human-readable form.

## Requirements
1. Function signature: def predict_cell_type(expression: dict) -> str
2. Return one of the exact 26 cell type names
3. Maximize fidelity with the original model
4. Code must be self-contained (no external model dependencies)
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
