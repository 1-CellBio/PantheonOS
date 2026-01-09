"""
Compare benchmark results between baseline and learning runs.

Loads result files and calculates accuracy differences.

Usage:
    # Auto mode (finds latest runs)
    python -m benchmarks.bixbench.compare
    
    # Manual mode (specify runs)
    python -m benchmarks.bixbench.compare \
        --baseline baseline_20260109_073000 \
        --learning with_learning_20260109_074500
    
    # Compare regrade results
    python -m benchmarks.bixbench.compare \
        --baseline baseline_20260109_073000 \
        --learning with_learning_20260109_074500 \
        --use-regrade
"""
import argparse
import json
from pathlib import Path
from typing import Optional


def find_result_file(results_dir: Path, run_identifier: str) -> Optional[Path]:
    """Find result file for a given run identifier.
    
    Args:
        results_dir: Results directory
        run_identifier: Can be:
            - Full path to JSON file (e.g., "results_baseline_xxx.json" or "results_baseline_xxx_regrade.json")
            - Directory name (e.g., "baseline_20260109_073000")
            - Run name prefix (e.g., "baseline" - finds latest, prefers regrade if exists)
    
    Returns:
        Path to result file or None
    """
    # Case 1: Direct JSON file path
    if run_identifier.endswith('.json'):
        path = Path(run_identifier)
        if path.exists():
            return path
        # Try relative to results_dir
        path = results_dir / run_identifier
        if path.exists():
            return path
        return None
    
    # Case 2: Directory name (e.g., "baseline_20260109_073000")
    run_dir = results_dir / run_identifier
    if run_dir.exists() and run_dir.is_dir():
        # Look for summary.json in the directory
        summary_file = run_dir / "summary.json"
        if summary_file.exists():
            return summary_file
        return None
    
    # Case 3: Run name prefix (e.g., "baseline" or "with_learning")
    # Find latest matching run, prefer regrade version if exists
    
    # First try regrade files (higher priority)
    regrade_pattern = f"results_{run_identifier}_*_regrade.json"
    regrade_files = sorted(results_dir.glob(regrade_pattern))
    if regrade_files:
        return regrade_files[-1]
    
    # Fallback to regular results
    regular_pattern = f"results_{run_identifier}_*.json"
    regular_files = sorted(results_dir.glob(regular_pattern))
    # Filter out regrade files (already checked above)
    regular_files = [f for f in regular_files if not f.name.endswith('_regrade.json')]
    if regular_files:
        return regular_files[-1]
    
    return None


def load_result_file(file_path: Path) -> Optional[dict]:
    """Load and parse result file."""
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"⚠️  Failed to load {file_path}: {e}")
        return None


def compare_results(
    results_dir: str = "benchmarks/bixbench/results",
    baseline_id: Optional[str] = None,
    learning_id: Optional[str] = None,
):
    """Compare baseline vs learning results.
    
    Auto-detects result format (original or regrade) and handles both.
    
    Args:
        results_dir: Results directory
        baseline_id: Baseline run identifier (auto-detect if None)
        learning_id: Learning run identifier (auto-detect if None)
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print("❌ Results directory not found")
        return
    
    # Find result files
    if baseline_id:
        baseline_file = find_result_file(results_path, baseline_id)
        if not baseline_file:
            print(f"❌ Baseline run not found: {baseline_id}")
            return
    else:
        # Auto-detect latest baseline
        baseline_id = "baseline"
        baseline_file = find_result_file(results_path, baseline_id)
    
    if learning_id:
        learning_file = find_result_file(results_path, learning_id)
        if not learning_file:
            print(f"❌ Learning run not found: {learning_id}")
            return
    else:
        # Auto-detect latest learning
        learning_id = "with_learning"
        learning_file = find_result_file(results_path, learning_id)
    
    # Load results
    baseline = load_result_file(baseline_file) if baseline_file else None
    learning = load_result_file(learning_file) if learning_file else None
    
    # Detect format type
    is_regrade = (baseline and "llm_regrade" in baseline) or (learning and "llm_regrade" in learning)
    
    print("="*70)
    if is_regrade:
        print("📊 BixBench Evaluation Comparison (Regrade Results)")
    else:
        print("📊 BixBench Evaluation Comparison")
    print("="*70)
    
    if not baseline and not learning:
        print("\n❌ No results found.")
        print("   Run benchmark first:")
        print("   python -m benchmarks.bixbench.run")
        print("   python -m benchmarks.bixbench.run --enable-learning")
        return
    
    # Print file sources
    print(f"\n📁 Comparing:")
    if baseline_file:
        print(f"   Baseline: {baseline_file.name}")
    if learning_file:
        print(f"   Learning: {learning_file.name}")
    
    # Print individual results
    for name, data in [("Baseline", baseline), ("With Learning", learning)]:
        print(f"\n📋 {name}:")
        if data:
            # Handle both summary.json and results_*.json formats
            if "llm_regrade" in data:
                # Regrade format
                grade_data = data["llm_regrade"]
                print(f"   Questions: {grade_data.get('total_questions', 0)}")
                print(f"   Correct: {grade_data.get('total_correct', 0)}")
                accuracy = grade_data.get('accuracy', 0)
                print(f"   Accuracy: {accuracy:.1%}")
                print(f"   Model: {data.get('grading_model', 'N/A')}")
            else:
                # Original format
                print(f"   Timestamp: {data.get('timestamp', 'N/A')}")
                print(f"   Capsules: {data.get('capsule_count', 0)}")
                print(f"   Questions: {data.get('total_questions', 0)}")
                print(f"   Correct: {data.get('total_correct', 0)}")
                accuracy = data.get('overall_accuracy', 0)
                print(f"   Accuracy: {accuracy:.1%}")
                
                # Per-capsule breakdown (only for original results)
                for result in data.get("results", []):
                    if result.get("status") == "completed":
                        grading = result.get("grading", {})
                        cap_acc = grading.get("accuracy", 0)
                        cap_correct = grading.get("correct", 0)
                        cap_total = grading.get("total", 0)
                        print(f"     - {result['capsule_id']}: {cap_correct}/{cap_total} ({cap_acc:.1%})")
        else:
            print("   ❌ No results available")
    
    # Calculate improvement if both exist
    if baseline and learning:
        print(f"\n{'='*70}")
        print("📈 Comparison")
        print("="*70)
        
        # Extract accuracy based on format
        if "llm_regrade" in baseline:
            base_acc = baseline["llm_regrade"].get('accuracy', 0)
        else:
            base_acc = baseline.get('overall_accuracy', 0)
        
        if "llm_regrade" in learning:
            learn_acc = learning["llm_regrade"].get('accuracy', 0)
        else:
            learn_acc = learning.get('overall_accuracy', 0)
        
        diff = learn_acc - base_acc
        if base_acc > 0:
            improvement_pct = (diff / base_acc) * 100
        else:
            improvement_pct = float('inf') if learn_acc > 0 else 0
        
        print(f"\n   Baseline Accuracy:      {base_acc:.1%}")
        print(f"   With Learning Accuracy: {learn_acc:.1%}")
        print(f"   Difference:             {diff:+.1%}")
        
        if diff > 0:
            print(f"\n   ✅ Learning improved accuracy by {improvement_pct:.1f}%")
        elif diff < 0:
            print(f"\n   ⚠️  Learning decreased accuracy by {abs(improvement_pct):.1f}%")
        else:
            print(f"\n   ➖ No change in accuracy")
        
        # Per-capsule comparison (only for original results)
        if "results" in baseline and "results" in learning:
            print(f"\n{'='*70}")
            print("📊 Per-Capsule Comparison")
            print("="*70)
            print(f"\n   {'Capsule':<15} {'Baseline':<15} {'Learning':<15} {'Diff':<10}")
            print(f"   {'-'*55}")
            
            base_results = {r['capsule_id']: r for r in baseline.get('results', []) if r.get('status') == 'completed'}
            learn_results = {r['capsule_id']: r for r in learning.get('results', []) if r.get('status') == 'completed'}
            
            all_capsules = set(base_results.keys()) | set(learn_results.keys())
            
            for capsule_id in sorted(all_capsules):
                base_res = base_results.get(capsule_id, {}).get('grading', {})
                learn_res = learn_results.get(capsule_id, {}).get('grading', {})
                
                b_acc = base_res.get('accuracy', 0)
                l_acc = learn_res.get('accuracy', 0)
                d = l_acc - b_acc
                
                b_str = f"{base_res.get('correct', 0)}/{base_res.get('total', 0)}" if base_res else "N/A"
                l_str = f"{learn_res.get('correct', 0)}/{learn_res.get('total', 0)}" if learn_res else "N/A"
                d_str = f"{d:+.1%}" if base_res and learn_res else "N/A"
                
                print(f"   {capsule_id:<15} {b_str:<15} {l_str:<15} {d_str:<10}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare BixBench results (auto-detects regrade vs original)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect latest runs (prefers regrade if available)
  python -m benchmarks.bixbench.compare

  # Specify runs manually (auto-detects format)
  python -m benchmarks.bixbench.compare \\
      --baseline baseline_20260109_073000 \\
      --learning with_learning_20260109_074500

  # Use result file directly
  python -m benchmarks.bixbench.compare \\
      --baseline results/results_baseline_20260109_073000.json \\
      --learning results/results_with_learning_20260109_074500.json
  
  # Compare specific regrade files
  python -m benchmarks.bixbench.compare \\
      --baseline results/results_baseline_20260109_073000_regrade.json \\
      --learning results/results_with_learning_20260109_074500_regrade.json
        """
    )
    parser.add_argument("--results-dir", default="benchmarks/bixbench/results",
                        help="Results directory")
    parser.add_argument("--baseline", default=None,
                        help="Baseline run identifier (dir name, file, or prefix)")
    parser.add_argument("--learning", default=None,
                        help="Learning run identifier (dir name, file, or prefix)")
    
    args = parser.parse_args()
    compare_results(
        results_dir=args.results_dir,
        baseline_id=args.baseline,
        learning_id=args.learning,
    )


if __name__ == "__main__":
    main()
