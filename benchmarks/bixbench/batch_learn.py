"""
Batch learning script for Pantheon.

Processes all memory files in a directory and learns from each one,
outputting to a dedicated skillbook.json.

Supports two learning modes:
- pipeline: Local processing via Reflector + SkillManager (faster, no endpoint)
- team: Processing via learning agents (requires endpoint for tools)

Usage:
    # Pipeline mode (default, no endpoint needed)
    python -m benchmarks.bixbench.batch_learn --memory-dir results/baseline_xxx

    # Team mode (requires endpoint)
    python -m benchmarks.bixbench.batch_learn --memory-dir results/xxx --mode team

    # Custom skillbook and team ID
    python -m benchmarks.bixbench.batch_learn \
        --memory-dir results/xxx \
        --output skillbook_bixbench.json \
        --mode team \
        --team-id bixbench_learning_team
"""
import argparse
import asyncio
import json
import uuid
from pathlib import Path
from typing import List, Optional

from pantheon.utils.log import setup_file_logging, logger


def setup_logging(log_dir: str = ".pantheon/logs/benchmark", log_level: str = "INFO"):
    """Setup file logging for batch learning.
    
    Args:
        log_dir: Directory for log files
        log_level: Log level (INFO, DEBUG, WARNING)
        
    Returns:
        Path to log file
    """
    log_path = Path(log_dir)
    log_file = setup_file_logging(
        log_dir=log_path,
        level=log_level,
        session_name="batch_learning",
    )
    
    logger.info(f"Batch learning logging started: {log_file}")
    return log_file


async def batch_learn(
    memory_dir: str,
    output_skillbook: str | None = None,
    learning_mode: str = "pipeline",
    learning_model: str | None = None,
    log_level: str = "INFO",
    cooldown_seconds: int = 10,
    filter_ids: List[str] | None = None,
    **config_overrides,
):
    """
    Batch learn from all memory files in a directory.
    
    Args:
        memory_dir: Directory containing memory JSON files
        output_skillbook: Path to output skillbook JSON file (default: {memory_dir}/skillbook_batch.json)
        learning_mode: Learning mode ("pipeline" or "team")
        learning_model: Model for learning (overrides settings)
        log_level: Log level (INFO, DEBUG, WARNING)
        cooldown_seconds: Seconds to wait between learning tasks (default: 10, to avoid API quota limits)
        filter_ids: Optional list of capsule IDs to process (e.g., ["bix-1", "bix-2"]). 
                   If provided, only memory files matching these IDs will be processed.
                   Can be full filenames (e.g., "bix-1_memory.json") or just IDs (e.g., "bix-1").
        **config_overrides: Additional learning config overrides
            - max_tool_arg_length: Max chars for tool args in compression
            - max_tool_output_length: Max chars for tool output in compression
            - min_confidence_threshold: Min confidence for reflection
            - min_atomicity_score: Min atomicity score for skills
            - team_id: Team ID for team mode (default: "skill_learning_team")
            - workspace_path: Workspace path for endpoint (team mode only)
            - cleanup_after_learning: Whether to cleanup learning files
            - Any other learning config parameter from settings
    
    Example:
        await batch_learn(
            memory_dir="results/run1",
            learning_mode="pipeline",
            cooldown_seconds=15,
            max_tool_arg_length=300,
            min_confidence_threshold=0.7,
            filter_ids=["bix-1", "bix-5", "bix-10"],  # Only process these capsules
        )
    """
    from pantheon.internal.learning.skillbook import Skillbook
    from pantheon.internal.learning.reflector import Reflector
    from pantheon.internal.learning.skill_manager import SkillManager
    from pantheon.internal.learning.pipeline import LearningPipeline, LearningInput
    from pantheon.settings import get_settings
    
    # Setup logging
    log_file = setup_logging(log_level=log_level)
    print(f"📝 Logs: {log_file}")
    logger.info(f"Starting batch learning from: {memory_dir}")
    
    memory_path = Path(memory_dir)
    if not memory_path.exists():
        print(f"❌ Memory directory not found: {memory_dir}")
        logger.error(f"Memory directory not found: {memory_dir}")
        return
    
    # Default output to memory_dir/skillbook_batch.json
    if output_skillbook is None:
        output_skillbook = str(memory_path / "skillbook_batch.json")
    
    # Find all memory JSON files
    memory_files = list(memory_path.glob("**/*_memory.json"))
    memory_files.extend(memory_path.glob("**/round_*.json"))
    
    # Also check for direct JSON files (but filter out metadata)
    for f in memory_path.glob("**/*.json"):
        if f.name not in [p.name for p in memory_files]:
            # Skip non-memory files
            if any(skip in f.name.lower() for skip in ["summary", "index", "info", "config", "capsule"]):
                continue
            memory_files.append(f)
    
    # Deduplicate
    seen = set()
    unique_files = []
    for f in memory_files:
        if str(f) not in seen:
            seen.add(str(f))
            unique_files.append(f)
    
    if not unique_files:
        print(f"❌ No memory files found in: {memory_dir}")
        return
    
    # Filter by IDs if provided
    if filter_ids:
        # Normalize filter IDs to handle both "bix-1" and "bix-1_memory.json" formats
        normalized_ids = set()
        for fid in filter_ids:
            # Remove .json extension if present
            if fid.endswith(".json"):
                fid = fid[:-5]
            # Remove _memory suffix if present
            if fid.endswith("_memory"):
                fid = fid[:-7]
            normalized_ids.add(fid)
        
        # Filter files based on normalized IDs
        filtered_files = []
        for f in unique_files:
            fname = f.stem  # Get filename without extension
            # Remove _memory suffix if present
            if fname.endswith("_memory"):
                fname = fname[:-7]
            # Check if this file matches any of the filter IDs
            if fname in normalized_ids:
                filtered_files.append(f)
        
        if not filtered_files:
            print(f"❌ No memory files matched the provided IDs: {filter_ids}")
            print(f"   Available files: {[f.name for f in unique_files[:10]]}...")
            return
        
        print(f"🔍 Filtered to {len(filtered_files)} memory files matching IDs: {filter_ids}")
        unique_files = filtered_files
    else:
        print(f"📂 Found {len(unique_files)} memory files in {memory_dir}")
    print(f"🔧 Learning mode: {learning_mode}")
    if learning_mode == "team":
        print(f"🏷️  Team ID: {team_id}")
    
    # Initialize skillbook with dedicated output path
    output_path = Path(output_skillbook)
    skillbook_dir = output_path.parent
    if skillbook_dir != Path("."):
        skillbook_dir.mkdir(parents=True, exist_ok=True)
    
    # Create fresh skillbook for this batch
    skillbook = Skillbook(skillbook_path=str(output_path))
    
    # Build learning_config from settings + overrides
    settings = get_settings()
    learning_config = settings.get_learning_config().copy()
    
    # Apply explicit parameter overrides
    if learning_model is not None:
        learning_config["learning_model"] = learning_model
    
    # Apply all kwargs as config overrides
    learning_config.update(config_overrides)
    
    # Initialize learning components with config
    reflector = Reflector(
        model=learning_config.get("learning_model", "gemini/gemini-3-flash-preview"),
        learning_config=learning_config,
    )
    skill_manager = SkillManager(
        model=learning_config.get("learning_model", "gemini/gemini-3-flash-preview"),
        learning_config=learning_config,
    )
    
    # Create pipeline
    learning_dir = str(Path(memory_dir) / ".batch_learning")
    pipeline = LearningPipeline(
        skillbook=skillbook,
        reflector=reflector,
        skill_manager=skill_manager,
        learning_dir=learning_dir,
        cleanup_after_learning=learning_config.get("cleanup_after_learning", False),
        min_confidence_threshold=learning_config.get("min_confidence_threshold", 0.5),
        min_atomicity_score=learning_config.get("min_atomicity_score", 0.85),
        mode=learning_mode,
        team_id=learning_config.get("team_id", "skill_learning_team"),
        # Config not needed - already passed to Reflector and SkillManager above
    )
    
    # For team mode, initialize endpoint
    endpoint = None
    if learning_mode == "team":
        print(f"\n🚀 Starting endpoint for team mode...")
        from pantheon.chatroom.start import _start_endpoint_embedded
        
        endpoint_id_hash = str(uuid.uuid4())
        workspace = learning_config.get("workspace_path", str(Path.cwd()))
        
        endpoint = await _start_endpoint_embedded(
            endpoint_id_hash=endpoint_id_hash,
            workspace_path=workspace,
            log_level="WARNING",
        )
        
        # Initialize learning team with endpoint
        await pipeline.initialize_team(endpoint)
        print(f"✓ Endpoint ready, learning team initialized")
    
    # Start pipeline
    await pipeline.start()
    
    print(f"\n🎓 Starting batch learning...")
    print(f"📝 Output skillbook: {output_path}")
    
    # Process each memory file sequentially
    processed = 0
    errors = 0
    
    for i, memory_file in enumerate(unique_files, 1):
        try:
            print(f"\n[{i}/{len(unique_files)}] 📄 Processing: {memory_file.name}")
            
            # Load messages from file
            with open(memory_file) as f:
                data = json.load(f)
            
            # Extract messages (handle different formats)
            if isinstance(data, dict):
                messages = data.get("messages", [])
                agent_name = data.get("agent_name", "global")
                capsule_id = data.get("capsule_id", "")
                if not messages and "content" in data:
                    # Single message format
                    messages = [data]
            elif isinstance(data, list):
                messages = data
                agent_name = "global"
                capsule_id = ""
            else:
                print(f"  ⚠️ Unknown format, skipping")
                continue
            
            if not messages:
                print(f"  ⚠️ No messages found, skipping")
                continue
            
            # Create learning input
            turn_id = memory_file.stem
            learning_input = LearningInput(
                turn_id=turn_id,
                agent_name=agent_name,
                details_path=str(memory_file),
                chat_id=capsule_id or memory_file.parent.name,
            )
            
            # Submit to pipeline
            pipeline.submit(learning_input)
            print(f"  ✓ Submitted for learning ({len(messages)} messages)")
            
            # Wait for this specific task to be processed (queue to drain)
            print(f"  ⏳ Waiting for queue to drain...")
            wait_count = 0
            max_wait = 60  # 1 minute max per task
            while not pipeline._queue.empty() and wait_count < max_wait:
                await asyncio.sleep(0.5)
                wait_count += 1
            
            if pipeline._queue.empty():
                print(f"  ✅ Learning complete")
                processed += 1
                # Cooldown period between tasks to avoid API quota limits
                print(f"  ⏸️  Cooling down for {cooldown_seconds}s...")
                await asyncio.sleep(cooldown_seconds)
            else:
                print(f"  ⚠️ Timeout waiting for queue to drain")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            logger.exception(f"Error processing {memory_file}")
            errors += 1
    
    # Final wait to ensure all processing is complete
    print(f"\n⏳ Final check - ensuring all tasks are complete...")
    wait_count = 0
    max_wait = 30
    while not pipeline._queue.empty() and wait_count < max_wait:
        await asyncio.sleep(1)
        wait_count += 1
        if wait_count % 5 == 0:
            print(f"  Queue size: {pipeline._queue.qsize()}, waited {wait_count}s")
    
    # Give some extra time for final processing
    await asyncio.sleep(5)
    
    # Stop pipeline and save
    await pipeline.stop()
    
    # Cleanup endpoint if we started one
    if endpoint is not None:
        try:
            await endpoint.shutdown()
        except Exception as e:
            print(f"⚠️ Endpoint cleanup warning: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ Batch learning complete!")
    print(f"   Processed: {processed}/{len(unique_files)}")
    print(f"   Errors: {errors}")
    print(f"   Skillbook: {output_path}")
    
    # Print skillbook stats
    summary = skillbook.stats()
    print(f"\n📊 Skillbook Summary:")
    print(f"   Total skills: {summary.get('total_skills', 0)}")
    print(f"   Active skills: {summary.get('active_skills', 0)}")
    section_breakdown = summary.get('section_breakdown', {})
    if section_breakdown:
        print(f"   By section:")
        for section, count in section_breakdown.items():
            if count > 0:  # Only show sections with skills
                print(f"     - {section}: {count} skills")
    
    return {
        "processed": processed,
        "errors": errors,
        "output_skillbook": str(output_path),
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch learn from memory files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m benchmarks.bixbench.batch_learn --memory-dir results/run1

  # Continue learning for specific capsules that failed due to quota limits
  python -m benchmarks.bixbench.batch_learn --memory-dir results/run1 \\
      --ids bix-1 bix-5 bix-10

  # With custom compression limits
  python -m benchmarks.bixbench.batch_learn --memory-dir results/run1 \\
      --config max_tool_arg_length=300 max_tool_output_length=600

  # With model and confidence threshold
  python -m benchmarks.bixbench.batch_learn --memory-dir results/run1 \\
      --learning-model gemini-2.0-flash-exp \\
      --config min_confidence_threshold=0.7 min_atomicity_score=0.9

  # Team mode with custom team
  python -m benchmarks.bixbench.batch_learn --memory-dir results/run1 \\
      --mode team --config team_id=my_custom_team
        """
    )
    
    # Core parameters
    parser.add_argument(
        "--memory-dir",
        required=True,
        help="Directory containing memory JSON files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output skillbook JSON file path (default: {memory_dir}/skillbook_batch.json)",
    )
    parser.add_argument(
        "--mode",
        default="pipeline",
        choices=["pipeline", "team"],
        help="Learning mode: pipeline (local) or team (via agents, requires endpoint)",
    )
    parser.add_argument(
        "--learning-model",
        type=str,
        default=None,
        help="Model for learning (overrides settings)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Log level for batch learning",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=10,
        help="Seconds to wait between learning tasks (default: 10, to avoid API quota limits)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Optional list of capsule IDs to process (e.g., bix-1 bix-2 bix-3). " 
             "Only memory files matching these IDs will be processed. "
             "Useful for continuing failed batch learning runs.",
    )
    
    # Optional config overrides via key=value pairs
    parser.add_argument(
        "--config",
        nargs="*",
        metavar="KEY=VALUE",
        help="""Additional learning config overrides as KEY=VALUE pairs.
        Available keys: max_tool_arg_length, max_tool_output_length,
        min_confidence_threshold, min_atomicity_score, team_id,
        workspace_path, cleanup_after_learning, etc.""",
    )
    
    args = parser.parse_args()
    
    # Parse config overrides from --config key=value pairs
    config_overrides = {}
    if args.config:
        for item in args.config:
            if "=" not in item:
                print(f"⚠️  Warning: Ignoring invalid config format '{item}' (expected KEY=VALUE)")
                continue
            key, value = item.split("=", 1)
            # Try to parse value as int, float, or bool
            try:
                if value.lower() in ("true", "false"):
                    config_overrides[key] = value.lower() == "true"
                elif value.isdigit():
                    config_overrides[key] = int(value)
                elif value.replace(".", "", 1).isdigit():
                    config_overrides[key] = float(value)
                else:
                    config_overrides[key] = value
            except:
                config_overrides[key] = value
    
    asyncio.run(batch_learn(
        memory_dir=args.memory_dir,
        output_skillbook=args.output,
        learning_mode=args.mode,
        learning_model=args.learning_model,
        log_level=args.log_level,
        cooldown_seconds=args.cooldown,
        filter_ids=args.ids,
        **config_overrides,
    ))


if __name__ == "__main__":
    main()
