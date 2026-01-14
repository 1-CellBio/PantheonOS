"""
Evolution team for coordinating multi-agent code evolution.

EvolutionTeam orchestrates:
- Mutator agent for generating code changes
- Evaluator for scoring programs
- Optional critic agent for analysis
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pantheon.utils.log import logger

from .config import EvolutionConfig
from .database import EvolutionDatabase
from .evaluator import EvaluationResult, HybridEvaluator
from .program import CodebaseSnapshot, Program
from .prompt_builder import (
    EvolutionPromptBuilder,
    MUTATION_SYSTEM_PROMPT_CODEBASE,
    MUTATION_SYSTEM_PROMPT_SIMPLE,
    ANALYZER_SYSTEM_PROMPT,
)
from .result import EvolutionResult, IterationResult
from .utils.diff import parse_diff, apply_diff


def think(thought: str) -> str:
    """
    Use this tool to think through problems step by step.

    Args:
        thought: Your reasoning, analysis, or intermediate thoughts

    Returns:
        Acknowledgment that thinking was recorded
    """
    return "Thought recorded. Continue your analysis."


def extract_cost_from_response(response) -> float:
    """
    Extract LLM cost from an agent response.

    Args:
        response: AgentResponse from agent.run()

    Returns:
        Cost in USD, or 0.0 if not available
    """
    try:
        if response and response.details and response.details.messages:
            # Find the last assistant message which contains cost info
            for msg in reversed(response.details.messages):
                if msg.get("role") == "assistant" and "_metadata" in msg:
                    return msg.get("_metadata", {}).get("current_cost", 0.0)
    except Exception:
        pass
    return 0.0


class EvolutionTeam:
    """
    Evolution team coordinating mutation, evaluation, and selection.

    Implements the core evolution loop:
    1. Sample parent and inspirations from database
    2. Build mutation prompt
    3. Generate mutation via LLM
    4. Apply diff to create child program
    5. Evaluate child
    6. Add to database (MAP-Elites decides if kept)
    7. Repeat
    """

    def __init__(
        self,
        mutator: Optional[Any] = None,  # Agent
        evaluator: Optional[Union[HybridEvaluator, Any]] = None,  # HybridEvaluator or Agent
        analyzer: Optional[Any] = None,  # Agent for code analysis
        critic: Optional[Any] = None,  # Agent
        database: Optional[EvolutionDatabase] = None,
        config: Optional[EvolutionConfig] = None,
    ):
        """
        Initialize evolution team.

        Args:
            mutator: Agent for generating mutations (created if None)
            evaluator: HybridEvaluator or Agent for evaluation
            analyzer: Agent for code analysis (created if None when use_analyzer=True)
            critic: Optional critic agent for failure analysis
            database: Program database (created if None)
            config: Evolution configuration (created if None)
        """
        self.config = config or EvolutionConfig()

        # Configure log level from config
        if self.config.log_level:
            from pantheon.utils.log import set_level
            set_level(self.config.log_level)

        self.database = database or EvolutionDatabase(config=self.config)
        self.prompt_builder = EvolutionPromptBuilder(
            max_code_length=self.config.max_code_length,
            max_top_programs=self.config.num_top_programs,
            max_inspirations=self.config.num_inspirations,
        )

        # Agents (lazy-initialized)
        self._mutator = mutator
        self._evaluator = evaluator
        self._analyzer = analyzer
        self._critic = critic

        # State
        self.objective: str = ""
        self.evaluator_code: str = ""
        self._initialized = False

    async def _ensure_mutator(self):
        """Ensure mutator agent is initialized."""
        if self._mutator is None:
            try:
                from pantheon.agent import Agent
                # Use simplified prompt when analyzer is enabled
                system_prompt = (
                    MUTATION_SYSTEM_PROMPT_SIMPLE
                    if self.config.use_analyzer
                    else MUTATION_SYSTEM_PROMPT_CODEBASE
                )
                self._mutator = Agent(
                    name="code-mutator",
                    instructions=system_prompt,
                    model=self.config.mutator_model,
                    use_memory=False,  # Prevent context accumulation across iterations
                )
            except ImportError:
                raise RuntimeError("Pantheon Agent not available for mutation")
        return self._mutator

    async def _ensure_analyzer(self):
        """Ensure analyzer agent is initialized."""
        if self._analyzer is None:
            try:
                from pantheon.agent import Agent
                self._analyzer = Agent(
                    name="code-analyzer",
                    instructions=ANALYZER_SYSTEM_PROMPT,
                    model=self.config.analyzer_model,
                    tools=[think],  # Add thinking tool for deeper reasoning
                    use_memory=False,  # Prevent context accumulation across iterations
                )
            except ImportError:
                raise RuntimeError("Pantheon Agent not available for analysis")
        return self._analyzer

    async def _ensure_evaluator(self):
        """Ensure evaluator is initialized."""
        if self._evaluator is None:
            self._evaluator = HybridEvaluator(
                evaluator_code=self.evaluator_code,
                function_weight=self.config.function_weight,
                llm_weight=self.config.llm_weight,
                max_parallel=self.config.max_parallel_evaluations,
                timeout=self.config.evaluation_timeout,
                workspace_base=self.config.workspace_path,
                feedback_max_lines_per_file=self.config.feedback_max_lines_per_file,
            )
        return self._evaluator

    async def evolve(
        self,
        initial_code: Union[str, CodebaseSnapshot],
        evaluator_code: str,
        objective: str,
        max_iterations: Optional[int] = None,
        initial_path: Optional[str] = None,
        resume_from: Optional[str] = None,
        **kwargs,
    ) -> EvolutionResult:
        """
        Run the evolution loop.

        Args:
            initial_code: Initial code string or CodebaseSnapshot
            evaluator_code: Python code defining evaluate(workspace_path) function
            objective: Natural language optimization objective
            max_iterations: Override config max_iterations
            initial_path: Path for loading initial codebase (if initial_code is path)
            resume_from: Path to resume from (directory with evolution_state.json)
            **kwargs: Additional arguments

        Returns:
            EvolutionResult with best program and history
        """
        max_iterations = max_iterations or self.config.max_iterations
        self.objective = objective
        self.evaluator_code = evaluator_code

        # Initialize result
        result = EvolutionResult(
            config_used=self.config.to_dict(),
        )

        # Check if resuming from checkpoint
        start_iteration = 0
        best_score = 0
        generations_without_improvement = 0

        if resume_from:
            resume_path = Path(resume_from)
            state_file = resume_path / "evolution_state.json"

            if state_file.exists():
                logger.info(f"Resuming evolution from {resume_from}")

                # Load database
                self.database = EvolutionDatabase.load(str(resume_path))

                # Load evolution state
                with open(state_file, "r") as f:
                    state = json.load(f)

                start_iteration = state.get("current_iteration", 0) + 1
                best_score = state.get("best_score", 0)
                result.score_history = state.get("score_history", [])
                result.best_score_history = state.get("best_score_history", [])
                generations_without_improvement = state.get("generations_without_improvement", 0)

                # Restore objective and evaluator if not provided
                if not objective:
                    self.objective = state.get("objective", "")
                if not evaluator_code:
                    self.evaluator_code = state.get("evaluator_code", "")

                logger.info(f"Resumed from iteration {start_iteration}, best_score={best_score:.4f}")
                logger.info(f"Database has {len(self.database.programs)} programs")
            else:
                logger.warning(f"No evolution_state.json found in {resume_from}, starting fresh")
                resume_from = None

        if not resume_from:
            # Create initial snapshot
            if isinstance(initial_code, CodebaseSnapshot):
                initial_snapshot = initial_code
            elif initial_path:
                initial_snapshot = CodebaseSnapshot.from_directory(initial_path)
            else:
                # Single file code
                initial_snapshot = CodebaseSnapshot.from_single_file("main.py", initial_code)

            # Create and evaluate initial program
            initial_program = Program(
                id=str(uuid.uuid4())[:8],
                snapshot=initial_snapshot,
                generation=0,
            )

            logger.info(f"Starting evolution with {initial_program.file_count()} files, "
                       f"{initial_program.total_lines()} lines")

            # Evaluate initial program
            evaluator = await self._ensure_evaluator()
            eval_result = await evaluator.evaluate(initial_program)

            initial_program.metrics = eval_result.metrics
            initial_program.artifacts = eval_result.artifacts
            initial_program.llm_feedback = eval_result.llm_feedback

            self.database.add(initial_program)

            initial_score = initial_program.fitness_score(self.config.feature_dimensions)
            result.score_history.append(initial_score)
            result.best_score_history.append(initial_score)
            best_score = initial_score

            logger.info(f"Initial program score: {initial_score:.4f}")

        # Evolution loop - use workers if num_workers > 1
        if self.config.num_workers > 1:
            # Parallel worker-based evolution
            logger.info(f"Starting parallel evolution with {self.config.num_workers} workers")

            # Shared atomic counter for iteration numbers
            iteration_lock = asyncio.Lock()
            iteration_state = {"next": start_iteration}

            async def get_next_iteration():
                """Atomically get and increment iteration counter."""
                async with iteration_lock:
                    val = iteration_state["next"]
                    iteration_state["next"] += 1
                    return val

            result_queue = asyncio.Queue()

            # Start workers
            workers = [
                asyncio.create_task(
                    self._worker(i, get_next_iteration, max_iterations, result_queue)
                )
                for i in range(self.config.num_workers)
            ]

            # Collect results as they come in
            completed_iterations = 0
            target_iterations = max_iterations - start_iteration

            while completed_iterations < target_iterations:
                try:
                    iter_result = await asyncio.wait_for(result_queue.get(), timeout=300)
                    result.iteration_results.append(iter_result)
                    completed_iterations += 1

                    # Track scores
                    result.score_history.append(iter_result.child_score)

                    is_new_best = False
                    if iter_result.child_score > best_score:
                        best_score = iter_result.child_score
                        generations_without_improvement = 0
                        is_new_best = True
                    else:
                        generations_without_improvement += 1

                    result.best_score_history.append(best_score)

                    # Log every iteration with clear progress
                    progress_pct = completed_iterations / target_iterations * 100
                    status = "★ NEW BEST" if is_new_best else ("✓ accepted" if iter_result.accepted else "✗ rejected")
                    logger.info(
                        f"[{completed_iterations}/{target_iterations}] ({progress_pct:.0f}%) "
                        f"iter={iter_result.iteration} score={iter_result.child_score:.4f} "
                        f"best={best_score:.4f} {status}"
                    )

                    # Periodic summary (every 10 iterations)
                    if completed_iterations % 10 == 0:
                        stats = self.database.get_statistics()
                        logger.info(
                            f"=== Summary: {completed_iterations}/{target_iterations} complete, "
                            f"best={best_score:.4f}, avg={stats['avg_fitness']:.4f}, "
                            f"programs={stats['total_programs']} ==="
                        )

                    # Periodic checkpoint
                    if self.config.db_path and completed_iterations % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(
                            self.config.db_path,
                            start_iteration + completed_iterations,
                            best_score,
                            result.score_history,
                            result.best_score_history,
                            generations_without_improvement,
                        )

                    # Early stopping check
                    if generations_without_improvement >= self.config.early_stop_generations:
                        logger.info(
                            f"Early stopping: no improvement for "
                            f"{generations_without_improvement} iterations"
                        )
                        break

                except asyncio.TimeoutError:
                    logger.warning("Waiting for worker results...")
                    continue

            # Cancel remaining workers
            for worker in workers:
                worker.cancel()

            # Wait for workers to finish
            await asyncio.gather(*workers, return_exceptions=True)

        else:
            # Sequential evolution (original behavior)
            for iteration in range(start_iteration, max_iterations):
                try:
                    iter_result = await self._run_iteration(iteration, max_iterations)
                    result.iteration_results.append(iter_result)

                    # Track scores
                    result.score_history.append(iter_result.child_score)

                    if iter_result.child_score > best_score:
                        best_score = iter_result.child_score
                        generations_without_improvement = 0
                        logger.info(
                            f"  ★ New best score: {best_score:.4f} "
                            f"(+{iter_result.improvement:.4f})"
                        )
                    else:
                        generations_without_improvement += 1

                    result.best_score_history.append(best_score)

                    # Periodic logging
                    if self.config.log_iterations and iteration % 10 == 0 and iteration > 0:
                        stats = self.database.get_statistics()
                        logger.info(
                            f"--- Progress: {iteration}/{max_iterations}, best={best_score:.4f}, "
                            f"avg={stats['avg_fitness']:.4f}, programs={stats['total_programs']} ---"
                        )

                    # Periodic migration
                    if iteration > 0 and iteration % self.config.migration_interval == 0:
                        self.database.migrate()

                    # Periodic checkpoint
                    if self.config.db_path and iteration % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(
                            self.config.db_path,
                            iteration,
                            best_score,
                            result.score_history,
                            result.best_score_history,
                            generations_without_improvement,
                        )

                    # Early stopping
                    if generations_without_improvement >= self.config.early_stop_generations:
                        logger.info(
                            f"Early stopping: no improvement for "
                            f"{generations_without_improvement} generations"
                        )
                        break

                except Exception as e:
                    logger.error(f"Iteration {iteration} failed: {e}")
                    result.errors.append(f"Iteration {iteration}: {e}")
                    result.iteration_results.append(
                        IterationResult(
                            iteration=iteration,
                            parent_id="",
                            child_id="",
                            parent_score=0,
                            child_score=0,
                            improvement=0,
                            accepted=False,
                            error=str(e),
                        )
                    )

        # Finalize result
        result.total_iterations = len(result.iteration_results)
        result.best_program = self.database.get_best_program()
        result.best_score = best_score
        result.database = self.database
        result.finalize()

        # Save final checkpoint
        if self.config.db_path:
            # Compute final iteration number
            final_iteration = start_iteration + len(result.iteration_results) - 1
            self._save_checkpoint(
                self.config.db_path,
                final_iteration,
                best_score,
                result.score_history,
                result.best_score_history,
                generations_without_improvement,
            )

        logger.info(result.get_summary())

        return result

    def _save_checkpoint(
        self,
        path: str,
        iteration: int,
        best_score: float,
        score_history: List[float],
        best_score_history: List[float],
        generations_without_improvement: int,
    ) -> None:
        """Save evolution checkpoint including state for resume."""
        # Save database
        self.database.save(path)

        # Save evolution state
        state = {
            "current_iteration": iteration,
            "best_score": best_score,
            "score_history": score_history,
            "best_score_history": best_score_history,
            "generations_without_improvement": generations_without_improvement,
            "objective": self.objective,
            "evaluator_code": self.evaluator_code,
        }

        state_path = Path(path) / "evolution_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Checkpoint saved: iteration {iteration}, {len(self.database.programs)} programs")

    async def _run_iteration(
        self,
        iteration: int,
        max_iterations: int = 0,
        worker_id: Optional[int] = None,
    ) -> IterationResult:
        """
        Run a single evolution iteration.

        Args:
            iteration: Current iteration number
            max_iterations: Total iterations for logging progress
            worker_id: Worker ID for parallel execution (None for sequential)

        Returns:
            IterationResult with details
        """
        iter_start = time.time()
        log_prefix = f"[Worker {worker_id}]" if worker_id is not None else f"[{iteration + 1}/{max_iterations}]"
        logger.info(f"{log_prefix} Starting iteration...")

        # Sample parent and inspirations (thread-safe)
        parent, inspirations = await self.database.sample_async(
            num_inspirations=self.config.num_inspirations,
        )

        parent_score = parent.fitness_score(self.config.feature_dimensions)

        # Get top programs for reference (thread-safe)
        async with self.database._lock:
            top_programs = self.database.get_top_programs(
                n=self.config.num_top_programs,
            )

        # Apply probability filtering for context sections
        use_top_programs = random.random() < self.config.top_programs_probability
        use_inspirations = random.random() < self.config.inspirations_probability

        effective_top_programs = top_programs if use_top_programs else None
        effective_inspirations = inspirations if use_inspirations else None

        # Build prompt (with or without analyzer)
        analysis_text = ""  # Store analyzer output for program record
        iteration_cost = 0.0  # Track LLM cost for this iteration
        if self.config.use_analyzer:
            # === Analyzer Phase (full context) ===
            analysis_start = time.time()
            try:
                analyzer = await self._ensure_analyzer()
                analysis_prompt = self.prompt_builder.build_analysis_prompt(
                    parent=parent,
                    objective=self.objective,
                    top_programs=effective_top_programs,
                    inspirations=effective_inspirations,
                    artifacts=parent.artifacts,
                    iteration=iteration,
                )
                analysis_response = await asyncio.wait_for(
                    analyzer.run(analysis_prompt, update_memory=False),
                    timeout=self.config.analyzer_timeout
                )
                analysis_text = analysis_response.content
                iteration_cost += extract_cost_from_response(analysis_response)
                analysis_time = time.time() - analysis_start
                logger.info(f"{log_prefix} Analysis: {analysis_time:.1f}s (${iteration_cost:.4f})")
            except asyncio.TimeoutError:
                analysis_time = time.time() - analysis_start
                logger.warning(f"{log_prefix} Analyzer timeout after {analysis_time:.1f}s, skipping iteration")
                return IterationResult(
                    iteration=iteration,
                    parent_id=parent.id,
                    child_id=parent.id,
                    parent_score=parent_score,
                    child_score=parent_score,
                    improvement=0,
                    accepted=False,
                    mutation_time=0,
                    evaluation_time=0,
                    total_time=time.time() - iter_start,
                    error="analyzer_timeout",
                )
            except Exception as e:
                logger.warning(f"{log_prefix} Analyzer failed: {e}, skipping iteration")
                return IterationResult(
                    iteration=iteration,
                    parent_id=parent.id,
                    child_id=parent.id,
                    parent_score=parent_score,
                    child_score=parent_score,
                    improvement=0,
                    accepted=False,
                    mutation_time=0,
                    evaluation_time=0,
                    total_time=time.time() - iter_start,
                    error=f"analyzer_failed: {e}",
                )

            # === Mutator Phase (code + instructions only) ===
            prompt = self.prompt_builder.build_simple_mutation_prompt(
                parent=parent,
                analysis=analysis_text,
            )
        else:
            # Original behavior: mutator gets full context
            prompt = self.prompt_builder.build_mutation_prompt(
                parent=parent,
                objective=self.objective,
                top_programs=effective_top_programs,
                inspirations=effective_inspirations,
                artifacts=parent.artifacts,
                iteration=iteration,
            )

        # Generate mutation with timeout
        mutation_start = time.time()
        mutator = await self._ensure_mutator()

        try:
            response = await asyncio.wait_for(
                mutator.run(prompt, update_memory=False),
                timeout=self.config.mutation_timeout
            )
            mutation_cost = extract_cost_from_response(response)
            iteration_cost += mutation_cost
            mutation_time = time.time() - mutation_start
            logger.info(f"{log_prefix} Mutation: {mutation_time:.1f}s (${mutation_cost:.4f})")
        except asyncio.TimeoutError:
            mutation_time = time.time() - mutation_start
            logger.warning(f"{log_prefix} Mutation timeout after {mutation_time:.1f}s")
            return IterationResult(
                iteration=iteration,
                parent_id=parent.id,
                child_id=parent.id,
                parent_score=parent_score,
                child_score=parent_score,
                improvement=0,
                accepted=False,
                mutation_time=mutation_time,
                evaluation_time=0,
                total_time=time.time() - iter_start,
                error="mutation_timeout",
                llm_cost=iteration_cost,
            )

        # Apply mutation
        try:
            child_snapshot = self._apply_mutation(parent.snapshot, response.content)
            default_file = next(iter(parent.snapshot.files.keys()), "main.py")
            changes = parse_diff(response.content, default_file)
            logger.info(f"{log_prefix} Applied {len(changes)} change(s)")
        except Exception as e:
            logger.warning(f"{log_prefix} Failed to apply mutation: {e}")
            child_snapshot = parent.snapshot

        # Create child program
        child = Program(
            id=str(uuid.uuid4())[:8],
            snapshot=child_snapshot,
            diff_from_parent=child_snapshot.diff_from(parent.snapshot),
            parent_id=parent.id,
            generation=parent.generation + 1,
            prompt_used=prompt if self.config.save_prompts else "",
            analysis_used=analysis_text if self.config.save_prompts else "",
        )

        # Evaluate child
        eval_start = time.time()
        evaluator = await self._ensure_evaluator()
        eval_result = await evaluator.evaluate(child)
        eval_time = time.time() - eval_start

        child.metrics = eval_result.metrics
        child.artifacts = eval_result.artifacts
        child.llm_feedback = eval_result.llm_feedback

        child_score = child.fitness_score(self.config.feature_dimensions)
        improvement = child_score - parent_score

        # Log evaluation results
        logger.info(f"{log_prefix} Evaluation: {eval_time:.1f}s, score: {child_score:.4f}")

        # Log key metrics if available
        metric_parts = []
        for key in ['mixing_score', 'speed_score', 'combined_score']:
            if key in child.metrics:
                metric_parts.append(f"{key.replace('_score', '')}={child.metrics[key]:.3f}")
        if metric_parts:
            logger.info(f"{log_prefix} Metrics: {', '.join(metric_parts)}")

        # Add to database (thread-safe)
        accepted = await self.database.add_async(child)

        total_time = time.time() - iter_start

        # Log result
        result_str = "✓ Improved" if improvement > 0 else "✗ No improvement"
        accepted_str = "(accepted)" if accepted else "(rejected)"
        cost_str = f"${iteration_cost:.4f}" if iteration_cost > 0 else ""
        logger.info(f"{log_prefix} Result: {result_str} ({improvement:+.4f}) {accepted_str} [{total_time:.1f}s] {cost_str}")

        return IterationResult(
            iteration=iteration,
            parent_id=parent.id,
            child_id=child.id,
            parent_score=parent_score,
            child_score=child_score,
            improvement=improvement,
            accepted=accepted,
            mutation_time=mutation_time,
            evaluation_time=eval_time,
            total_time=total_time,
            llm_cost=iteration_cost,
        )

    async def _worker(
        self,
        worker_id: int,
        get_next_iteration,
        max_iterations: int,
        result_queue: asyncio.Queue,
    ) -> None:
        """
        Single worker that runs evolution iterations independently.

        Args:
            worker_id: Identifier for this worker
            get_next_iteration: Async function to get next iteration number
            max_iterations: Stop when counter reaches this value
            result_queue: Queue to put iteration results
        """
        while True:
            # Get next iteration number atomically
            iteration = await get_next_iteration()

            if iteration >= max_iterations:
                break

            logger.info(f"[Worker {worker_id}] Starting iteration {iteration + 1}/{max_iterations}")

            try:
                iter_result = await self._run_iteration(iteration, max_iterations, worker_id=worker_id)
                await result_queue.put(iter_result)
            except Exception as e:
                logger.error(f"[Worker {worker_id}] Iteration {iteration} failed: {e}")
                await result_queue.put(IterationResult(
                    iteration=iteration,
                    parent_id="",
                    child_id="",
                    parent_score=0,
                    child_score=0,
                    improvement=0,
                    accepted=False,
                    error=str(e),
                ))

    def _apply_mutation(
        self,
        parent_snapshot: CodebaseSnapshot,
        mutation_response: str,
    ) -> CodebaseSnapshot:
        """
        Apply LLM mutation response to parent snapshot.

        Args:
            parent_snapshot: Parent codebase snapshot
            mutation_response: LLM response with SEARCH/REPLACE blocks

        Returns:
            New CodebaseSnapshot with mutations applied
        """
        # Parse changes from response
        default_file = next(iter(parent_snapshot.files.keys()), "main.py")
        changes = parse_diff(mutation_response, default_file)

        if not changes:
            logger.warning("No valid changes parsed from mutation response")
            return parent_snapshot

        # Apply changes
        new_files = apply_diff(parent_snapshot.files, changes)

        return CodebaseSnapshot(
            files=new_files,
            base_path=parent_snapshot.base_path,
        )


async def evolve(
    initial_code: Union[str, CodebaseSnapshot],
    evaluator_code: str,
    objective: str,
    config: Optional[EvolutionConfig] = None,
    **kwargs,
) -> EvolutionResult:
    """
    Convenience function to run evolution.

    Args:
        initial_code: Initial code or CodebaseSnapshot
        evaluator_code: Evaluation function code
        objective: Optimization objective
        config: Evolution configuration
        **kwargs: Additional arguments

    Returns:
        EvolutionResult
    """
    team = EvolutionTeam(config=config)
    return await team.evolve(
        initial_code=initial_code,
        evaluator_code=evaluator_code,
        objective=objective,
        **kwargs,
    )
