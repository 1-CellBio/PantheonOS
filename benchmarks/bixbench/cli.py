"""
BixBench utility commands.

Usage:
    python -m benchmarks.bixbench.cli clean
    python -m benchmarks.bixbench.cli status
    python -m benchmarks.bixbench.cli regrade <run_dir> [--model MODEL]
"""
import argparse
import asyncio
import json
import shutil
from pathlib import Path


BIXBENCH_DIR = Path(__file__).parent
WORKSPACES_DIR = BIXBENCH_DIR / "workspaces"
RESULTS_DIR = BIXBENCH_DIR / "results"
DATA_DIR = BIXBENCH_DIR / "data"
TEST_CAPSULES_DIR = BIXBENCH_DIR / "test_capsules"


def clean():
    """Clean up BixBench workspace directories.
    
    Removes temporary files (notebooks, data) from workspace directories
    while preserving benchmark results in the results/ directory.
    
    Handles both:
    - Timestamped workspace runs (e.g., baseline_20260108_130351/) - deletes entire directory
    - Legacy capsule directories (e.g., bix-1/) - cleans contents only
    """
    print("🧹 BixBench Cleanup")
    print("="*50)
    
    cleaned_count = 0
    
    # Clean workspaces (all files inside run directories or capsule directories)
    if WORKSPACES_DIR.exists():
        print(f"\n📁 Cleaning workspaces: {WORKSPACES_DIR}")
        for workspace_dir in WORKSPACES_DIR.iterdir():
            if workspace_dir.is_dir():
                # Check if it's a timestamped run directory (e.g., baseline_20260108_130351)
                # or a legacy capsule directory (e.g., bix-1)
                is_run_dir = "_" in workspace_dir.name and len(workspace_dir.name.split("_")) >= 2
                
                if is_run_dir:
                    # Timestamped run directory - clean whole directory
                    print(f"   📂 Cleaning run: {workspace_dir.name}")
                    try:
                        shutil.rmtree(workspace_dir)
                        print(f"      ✓ Deleted run directory: {workspace_dir.name}/")
                        cleaned_count += 1
                    except Exception as e:
                        print(f"      × Failed to delete {workspace_dir.name}: {e}")
                else:
                    # Legacy capsule directory - clean contents only
                    print(f"   📂 Cleaning capsule: {workspace_dir.name}")
                    for item in workspace_dir.iterdir():
                        try:
                            if item.is_dir():
                                shutil.rmtree(item)
                                print(f"      ✓ Deleted directory: {item.name}/")
                            else:
                                item.unlink()
                                print(f"      ✓ Deleted file: {item.name}")
                            cleaned_count += 1
                        except Exception as e:
                            print(f"      × Failed to delete {item.name}: {e}")
    else:
        print(f"\n⚠️  Workspaces directory not found: {WORKSPACES_DIR}")
    
    print(f"\n✅ Cleaned {cleaned_count} items")
    print(f"\n💡 Note: Results are preserved in {RESULTS_DIR}")
    print(f"   To remove results, manually delete specific run directories")


def status():
    """Show BixBench status."""
    print("📊 BixBench Status")
    print("="*50)
    
    # Workspaces
    if WORKSPACES_DIR.exists():
        workspace_dirs = [d for d in WORKSPACES_DIR.iterdir() if d.is_dir()]
        run_dirs = [d for d in workspace_dirs if "_" in d.name and len(d.name.split("_")) >= 2]
        legacy_dirs = [d for d in workspace_dirs if d not in run_dirs]
        
        print(f"\n📁 Workspaces: {WORKSPACES_DIR}")
        if run_dirs:
            print(f"   Timestamped runs: {len(run_dirs)}")
            for d in sorted(run_dirs, reverse=True)[:5]:  # Show latest 5
                capsule_count = len([c for c in d.iterdir() if c.is_dir()])
                notebook_count = len(list(d.glob("*/*.ipynb")))
                print(f"     - {d.name}: {capsule_count} capsules, {notebook_count} notebooks")
        if legacy_dirs:
            notebook_count = len(list(WORKSPACES_DIR.glob("*/*.ipynb")))
            print(f"   Legacy capsules: {len(legacy_dirs)}")
            print(f"   Legacy notebooks: {notebook_count}")
    else:
        print(f"\n📁 Workspaces: Not created")
    
    # Results
    if RESULTS_DIR.exists():
        result_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]
        print(f"\n📁 Results: {RESULTS_DIR}")
        for d in result_dirs:
            summary_file = d / "summary.json"
            if summary_file.exists():
                import json
                with open(summary_file) as f:
                    summary = json.load(f)
                acc = summary.get("overall_accuracy", 0)
                total = summary.get("total_questions", 0)
                correct = summary.get("total_correct", 0)
                print(f"   - {d.name}: {correct}/{total} ({acc:.1%})")
            else:
                print(f"   - {d.name}: (incomplete)")
    else:
        print(f"\n📁 Results: None")
    
    # Data
    if DATA_DIR.exists():
        data_count = len([d for d in DATA_DIR.iterdir() if d.is_dir()])
        print(f"\n📁 Data: {DATA_DIR}")
        print(f"   Capsules downloaded: {data_count}")
    else:
        print(f"\n📁 Data: Not downloaded")


async def regrade(run_dir: str, model: str = "gemini/gemini-3-flash-preview"):
    """Re-grade answers using LLM semantic verification.
    
    This tool aggregates all questions and answers from a benchmark run
    and uses an LLM to determine semantic equivalence for better accuracy.
    """
    from .grader import regrade_run
    
    print(f"🔄 Re-grading run: {run_dir}")
    print(f"🤖 Model: {model}")
    print("=" * 50)
    
    try:
        result = await regrade_run(run_dir=run_dir, model=model)
        
        # Extract results
        regrade_result = result['regrade_result']
        regrade_file = result['regrade_file']
        total_questions = result['total_questions']
        trajectory_count = result['trajectory_count']
        
        print(f"📋 Found {trajectory_count} question trajectories")
        print(f"\n🧠 Calling LLM for batch grading...")
        print(f"✅ Received grades for {total_questions} questions")
        
        # Print results
        original = regrade_result['original_run']
        llm_regrade = regrade_result['llm_regrade']
        changes = llm_regrade['changes']
        
        print(f"\n{'=' * 50}")
        print(f"📊 Re-grading Results")
        print(f"{'=' * 50}")
        print(f"Original accuracy: {original['total_correct']}/{total_questions} ({original['overall_accuracy']:.1%})")
        print(f"LLM regrade accuracy: {llm_regrade['total_correct']}/{total_questions} ({llm_regrade['accuracy']:.1%})")
        print(f"Changes: {len(changes)}")
        
        if changes:
            print(f"\n📝 Grade changes:")
            for c in changes:
                arrow = "⬆️" if c['new'] == 'correct' else "⬇️"
                print(f"   {arrow} {c['question_id']}: {c['old']} → {c['new']}")
        
        print(f"\n✅ Regrade saved to: {regrade_file}")
        
    except Exception as e:
        print(f"❌ Re-grading failed: {e}")
        import traceback
        traceback.print_exc()




def main():
    parser = argparse.ArgumentParser(description="BixBench utility commands")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Clean command
    subparsers.add_parser("clean", help="Clean up workspace directories")
    
    # Status command
    subparsers.add_parser("status", help="Show BixBench status")
    
    # Regrade command
    regrade_parser = subparsers.add_parser("regrade", help="Re-grade answers using LLM")
    regrade_parser.add_argument("run_dir", help="Path to run directory (or name in results/)")
    regrade_parser.add_argument("--model", "-m", default="gemini/gemini-3-flash-preview",
                                help="LLM model for grading (default: gemini/gemini-3-flash-preview)")
    
    args = parser.parse_args()
    
    if args.command == "clean":
        clean()
    elif args.command == "status":
        status()
    elif args.command == "regrade":
        asyncio.run(regrade(run_dir=args.run_dir, model=args.model))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
