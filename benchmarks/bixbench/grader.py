"""
BixBench Grader - Uses official BixBench grading logic.

This module wraps the official BixBench graders to evaluate agent answers
against ground truth answers.
"""
import re
import sys
from pathlib import Path
from typing import Literal

# Add BixBench to path for importing official graders
BIXBENCH_PATH = Path(__file__).parent.parent.parent / "tmp" / "BixBench"
if BIXBENCH_PATH.exists():
    sys.path.insert(0, str(BIXBENCH_PATH))


class BixBenchGrader:
    """
    Grader for BixBench answers.
    
    Uses official BixBench grading logic when available, with fallback
    to simple string matching.
    """
    
    def __init__(self, use_official: bool = True):
        """
        Initialize grader.
        
        Args:
            use_official: Whether to try using official BixBench graders
        """
        self.use_official = use_official
        self._official_available = False
        
        if use_official:
            try:
                from bixbench.graders import GradeAnswer
                from bixbench.utils import AnswerMode
                self._GradeAnswer = GradeAnswer
                self._AnswerMode = AnswerMode
                self._official_available = True
            except ImportError:
                print("⚠️  Official BixBench graders not available, using fallback")
    
    async def grade(
        self,
        predicted: str,
        target: str,
        question: str = "",
        eval_mode: Literal["str_verifier", "range_verifier", "llm_verifier"] = "str_verifier",
    ) -> dict:
        """
        Grade an answer.
        
        Args:
            predicted: Agent's predicted answer
            target: Ground truth answer (ideal)
            question: The question text (for llm_verifier mode)
            eval_mode: Evaluation mode from question metadata
            
        Returns:
            Dict with 'correct', 'score', 'grade_type'
        """
        if not predicted:
            return {
                "correct": False,
                "score": 0,
                "grade_type": "no_answer",
            }
        
        # Try official grader first
        if self._official_available and eval_mode != "llm_verifier":
            try:
                grader = self._GradeAnswer(
                    answer_mode=self._AnswerMode.openanswer,
                    llm_client=None,
                )
                score, correct, refusal = await grader.grade(
                    target=target,
                    predicted=predicted,
                    question=question,
                    evaluation_mode=eval_mode,
                )
                return {
                    "correct": correct,
                    "score": score,
                    "grade_type": "correct" if correct else ("refused" if refusal else "incorrect"),
                }
            except Exception as e:
                print(f"Official grader failed: {e}, using fallback")
        
        # Fallback: simple string matching
        return self._fallback_grade(predicted, target, eval_mode)
    
    def _fallback_grade(
        self,
        predicted: str,
        target: str,
        eval_mode: str,
    ) -> dict:
        """
        Fallback grading when official graders not available.
        """
        if eval_mode == "range_verifier":
            correct = self._range_match(predicted, target)
        else:
            # Normalize strings (same as official grader)
            cleaned_predicted = re.sub(r"[^a-zA-Z0-9]", "", str(predicted)).lower()
            cleaned_target = re.sub(r"[^a-zA-Z0-9]", "", str(target)).lower()
            
            correct = cleaned_predicted == cleaned_target
            
            # Also try numeric comparison for scientific notation
            if not correct:
                correct = self._numeric_match(predicted, target)
        
        return {
            "correct": correct,
            "score": 1 if correct else 0,
            "grade_type": "correct" if correct else "incorrect",
        }
    
    def _numeric_match(self, predicted: str, target: str) -> bool:
        """
        Try to match answers as numbers (handles scientific notation and percentages).
        
        E.g., "0.0002" should match "2E-04" or "2e-4"
        "35.3414" should match "35%" (with 1% tolerance)
        """
        try:
            # Strip % and whitespace
            clean_pred = str(predicted).replace("%", "").strip()
            clean_target = str(target).replace("%", "").strip()
            
            pred_val = float(clean_pred)
            target_val = float(clean_target)
            
            # Allow 1% tolerance for floating point
            if target_val == 0:
                return pred_val == 0
            return abs(pred_val - target_val) / abs(target_val) < 0.01
        except (ValueError, TypeError):
            return False

    def _range_match(self, predicted: str, target: str) -> bool:
        """
        Check if predicted value falls within a range target.
        
        E.g., predicted="21.8036%", target="(20,25)" -> True
        """
        try:
            # Extract number from predicted (handle %)
            clean_pred = str(predicted).replace("%", "").strip()
            pred_val = float(clean_pred)
            
            # Parse range from target
            # Supports (min, max), [min, max], min-max
            range_match = re.search(r'[\(\[]\s*([0-9\.\-eE]+)\s*,\s*([0-9\.\-eE]+)\s*[\)\]]', str(target))
            if range_match:
                low = float(range_match.group(1))
                high = float(range_match.group(2))
                # Check raw value and scaled value (0.29 vs 29)
                return (low <= pred_val <= high) or (low <= pred_val * 100 <= high)
            
            # Try simple hyphenated range: 20-25
            hyphen_match = re.search(r'([0-9\.\-eE]+)\s*-\s*([0-9\.\-eE]+)', str(target))
            if hyphen_match:
                low = float(hyphen_match.group(1))
                high = float(hyphen_match.group(2))
                return (low <= pred_val <= high) or (low <= pred_val * 100 <= high)
                
            return False
        except (ValueError, TypeError):
            return False


async def grade_capsule_answers(
    answers: dict,
    questions: list,
    grader: BixBenchGrader = None,
) -> dict:
    """
    Grade all answers for a capsule.
    
    Args:
        answers: Dict of question_id -> agent_answer
        questions: List of question dicts with 'id', 'ideal', 'question', 'eval_mode'
        grader: Optional grader instance (creates one if not provided)
        
    Returns:
        Dict with per-question results and summary stats
    """
    if grader is None:
        grader = BixBenchGrader()
    
    results = {}
    correct_count = 0
    
    for q in questions:
        qid = q["id"]
        predicted = answers.get(qid, "")
        target = q.get("ideal", "")
        eval_mode = q.get("eval_mode", "str_verifier")
        
        grade_result = await grader.grade(
            predicted=str(predicted) if predicted else "",
            target=str(target),
            question=q.get("question", ""),
            eval_mode=eval_mode,
        )
        
        results[qid] = {
            "predicted": predicted,
            "target": target,
            "correct": grade_result["correct"],
            "score": grade_result["score"],
            "grade_type": grade_result["grade_type"],
        }
        
        if grade_result["correct"]:
            correct_count += 1
    
    return {
        "questions": results,
        "total": len(questions),
        "correct": correct_count,
        "accuracy": correct_count / len(questions) if questions else 0,
    }


async def regrade_run(run_dir: str, model: str = "gemini/gemini-3-flash-preview") -> dict:
    """Re-grade a benchmark run using LLM semantic verification.
    
    This function aggregates all questions and answers from a benchmark run
    and uses an LLM to determine semantic equivalence for better accuracy.
    
    Args:
        run_dir: Path to run directory (or name in results/)
        model: LLM model for grading
        
    Returns:
        Dict with regrade results or None if failed
    """
    import re
    import json
    from pathlib import Path
    from pantheon.agent import Agent
    from pantheon.utils.log import temporary_log_level
    
    # Resolve run path
    run_path = Path(run_dir)
    if not run_path.exists():
        # Try relative to results dir
        bixbench_dir = Path(__file__).parent
        results_dir = bixbench_dir / "results"
        run_path = results_dir / run_dir
    
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Load summary
    summary_file = run_path / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"No summary.json found in {run_path}")
    
    with open(summary_file) as f:
        summary = json.load(f)
    
    # Collect all trajectory files
    trajectory_files = sorted(run_path.glob("bix-*-q*.json"))
    if not trajectory_files:
        raise ValueError("No trajectory files found")
    
    # Group by capsule for batch processing
    capsule_questions = {}
    for traj_file in trajectory_files:
        with open(traj_file) as f:
            traj = json.load(f)
        
        capsule_id = traj.get("capsule_id", traj_file.stem.rsplit("-q", 1)[0])
        if capsule_id not in capsule_questions:
            capsule_questions[capsule_id] = []
        
        capsule_questions[capsule_id].append({
            "file": traj_file,
            "question_id": traj.get("question_id", traj_file.stem),
            "question": traj.get("question", ""),
            "target": traj.get("ideal", traj.get("target", "")),
            "predicted": traj.get("agent_answer", traj.get("predicted", "")),
            "original_correct": traj.get("correct", False),
        })
    
    # Process all questions in one batch for efficiency
    all_questions = []
    for capsule_id, questions in capsule_questions.items():
        for q in questions:
            all_questions.append(q)
    
    # Build batch grading prompt with explicit numerical rules
    grading_prompt = """You are a precise grader for a bioinformatics benchmark. Your task is to compare predicted answers with target answers and determine if they are semantically equivalent.

## GRADING RULES:

### For NUMERICAL answers:
1. **Percentage vs Decimal**: 35% equals 0.35 equals 35.0
   - Example: Predicted "35.3414" vs Target "35%" → CORRECT (35.3414 ≈ 35)
   - Example: Predicted "0.35" vs Target "35%" → CORRECT (0.35 = 35%)
   - Example: Predicted "0.3534" vs Target "35%" → CORRECT (0.3534 = 35.34%)
   - Example: Predicted "35.0" vs Target "0.35" → CORRECT (35% = 0.35)

2. **Decimal Precision**: Ignore trailing zeros and minor precision differences
   - Example: Predicted "1.670" vs Target "1.67" → CORRECT
   - Example: Predicted "4.0017" vs Target "4.0" → CORRECT (0.04% difference)

3. **Scientific Notation**: Different formats of same value are equivalent
   - Example: Predicted "0.000019" vs Target "1.9E-5" → CORRECT
   - Example: Predicted "2E-04" vs Target "0.0002" → CORRECT

4. **Tolerance**: Accept answers within 5% relative error
   - Formula: |predicted - target| / |target| < 0.05
   - Example: Predicted "0.0501" vs Target "0.05" → CORRECT (2% error)
   - Example: Predicted "2.2638" vs Target "(1.50,1.54)" → INCORRECT (outside range)

### For TEXT/STRING answers:
- Must match exactly (case-insensitive, ignoring whitespace)
- Example: Predicted "1-50, >100" vs Target "1-50" → INCORRECT (extra content)

### For EMPTY answers:
- Predicted "" or "NO ANSWER" → INCORRECT

## EXAMPLES:
✓ CORRECT: Predicted="35.3414", Target="35%" (35.3414 ≈ 35)
✓ CORRECT: Predicted="4.0017", Target="4.0" (0.04% error)
✓ CORRECT: Predicted="0.0501", Target="0.05" (2% error)
✗ INCORRECT: Predicted="0.0045", Target="(0.43,0.45)" (0.0045 not in range, likely scale mismatch)
✗ INCORRECT: Predicted="1-50, >100", Target="1-50" (extra content)

Questions to grade:
"""
    
    for i, q in enumerate(all_questions, 1):
        grading_prompt += f"""
---
[{i}] Question ID: {q['question_id']}
Question: {q['question'][:200]}...
Target Answer: {q['target']}
Predicted Answer: {q['predicted'] if q['predicted'] else 'NO ANSWER'}
"""
    
    grading_prompt += """

IMPORTANT: Output ONLY a valid JSON object mapping question_id to grade.
Use "correct" or "incorrect" as values.
Example: {"bix-1-q1": "correct", "bix-1-q2": "incorrect"}

Output:"""

    # Create grading agent using Pantheon Agent
    grading_agent = Agent(
        name="BixBenchGrader",
        model=model,
        instructions="You are a precise answer grader. Compare predicted and target answers for semantic equivalence.",
    )
    
    # Run agent with suppressed logs
    with temporary_log_level("WARNING"):
        response = await grading_agent.run(grading_prompt)
    
    response_text = response.content if hasattr(response, "content") else str(response)
    response_text = response_text.strip()
    
    # Extract JSON from response
    json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
    if json_match:
        grades = json.loads(json_match.group())
    else:
        # Try parsing the whole response
        grades = json.loads(response_text)
    
    # Count results
    total_correct = 0
    total_questions = len(all_questions)
    changes = []
    question_grades = {}
    
    for q in all_questions:
        qid = q['question_id']
        new_grade = grades.get(qid, "").lower() == "correct"
        old_grade = q['original_correct']
        
        question_grades[qid] = {
            'llm_correct': new_grade,
            'original_correct': old_grade,
            'target': q['target'],
            'predicted': q['predicted'],
        }
        
        if new_grade != old_grade:
            changes.append({
                "question_id": qid,
                "question": q['question'],
                "target": q['target'],
                "predicted": q['predicted'],
                "old": "correct" if old_grade else "incorrect",
                "new": "correct" if new_grade else "incorrect",
            })
        
        if new_grade:
            total_correct += 1
    
    # Generate regrade summary
    old_accuracy = summary.get('overall_accuracy', 0)
    new_accuracy = total_correct / total_questions if total_questions else 0
    
    regrade_result = {
        'original_run': {
            'timestamp': summary.get('timestamp'),
            'run_name': summary.get('run_name'),
            'capsule_count': summary.get('capsule_count'),
            'total_questions': summary.get('total_questions'),
            'total_correct': summary.get('total_correct'),
            'overall_accuracy': old_accuracy,
        },
        'llm_regrade': {
            'total_correct': total_correct,
            'total_questions': total_questions,
            'accuracy': new_accuracy,
            'changes_count': len(changes),
            'changes': changes,
        },
        'grading_model': model,
        'question_grades': question_grades,
    }
    
    # Save regrade result to separate file
    regrade_file = run_path.parent / f"results_{run_path.name}_regrade.json"
    with open(regrade_file, 'w') as f:
        json.dump(regrade_result, f, indent=2, ensure_ascii=False)
    
    return {
        'regrade_result': regrade_result,
        'regrade_file': regrade_file,
        'run_path': run_path,
        'total_questions': total_questions,
        'trajectory_count': len(trajectory_files),
    }

