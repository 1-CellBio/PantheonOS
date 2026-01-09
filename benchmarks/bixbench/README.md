# BixBench Evaluation Pipeline

BixBench benchmark integration for Pantheon Agents. This module provides tools to evaluate agent performance on the BixBench bioinformatics benchmark.

## Quick Start

```bash
# 1. Prepare capsules (download data)
python -m benchmarks.bixbench.prepare_capsules -n 10

# 2. Run benchmark
python -m benchmarks.bixbench.run --capsule-limit 10

# 3. Compare results (baseline vs learning)
python -m benchmarks.bixbench.compare
```

## CLI Commands

### 1. `prepare_capsules` - Download and Prepare Data

Downloads BixBench capsules from HuggingFace and prepares test prompts.

```bash
python -m benchmarks.bixbench.prepare_capsules -n 10
```

| Argument | Description | Default |
|----------|-------------|---------|
| `-n` | Number of capsules to prepare | 3 |

**Output:**
- `data/` - Raw capsule data downloaded from HuggingFace
- `test_capsules/` - Prepared prompts with `{WORKSPACE_PATH}` placeholder

**Note:** Workspaces are created at runtime during benchmark runs, not during preparation.

---

### 2. `run` - Execute Benchmark

Runs the benchmark against prepared capsules.

```bash
python -m benchmarks.bixbench.run --capsule-limit 10
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--capsule-limit` | Number of capsules to evaluate | 3 |
| `--enable-learning` | Enable ACE Learning module | False |
| `--output-dir` | Output directory for results | `benchmarks/bixbench/results` |
| `--log-level` | Log level (DEBUG, INFO, WARNING) | INFO |
| `--continue` | Path to previous run to continue from | None |

**Examples:**
```bash
# Basic run
python -m benchmarks.bixbench.run --capsule-limit 10

# With learning enabled
python -m benchmarks.bixbench.run --capsule-limit 10 --enable-learning

# Continue interrupted run (resumes from where it stopped)
python -m benchmarks.bixbench.run --capsule-limit 10 \
    --continue benchmarks/bixbench/results/baseline_20260108_120000

# Continue with short name (auto-resolves to results directory)
python -m benchmarks.bixbench.run --capsule-limit 10 \
    --continue baseline_20260108_120000

# Debug mode
python -m benchmarks.bixbench.run --capsule-limit 3 --log-level DEBUG
```

**Continue Mode Behavior:**
- ✅ **Skips completed capsules** - Any capsule with trajectory files (`bix-X-qY.json`) is skipped
- ✅ **Automatically retries error capsules** - Capsules with `status: "error"` have no trajectory files, so they will be re-run
- 🎯 **Smart prioritization** - Runs untried capsules first, then retries failed ones (avoids getting stuck on same failures)
- 📊 **Preserves previous results** - Appends new results to existing `summary.json`
- 📁 **Reuses workspace** - Workspace directory matches the results directory timestamp (e.g., `workspaces/baseline_20260108_130351/`)
- 🔄 **Error capsules are retried automatically** - No manual intervention needed

**What are trajectory files?**
- `bix-X-qY.json` - Individual question results (e.g., `bix-10-q1.json`, `bix-11-q2.json`)
- Only created for **successfully completed** capsules
- Error capsules have **no trajectory files**, so continue mode will retry them

**Example:**
```bash
# Run interrupted at capsule 25/50
python -m benchmarks.bixbench.run --capsule-limit 50

# Continue from where it stopped (retries any errors automatically)
python -m benchmarks.bixbench.run --capsule-limit 50 \
    --continue baseline_20260108_130351
```



**Output:**
- `results/{run_name}_{timestamp}/` - Run results directory
- `results/{...}/summary.json` - Overall results summary
- `results/{...}/{short_id}_memory.json` - Agent conversation history
- `results/{...}/bix-X-qY.json` - Individual trajectory files
- `workspaces/{run_name}_{timestamp}/` - Timestamped workspace directory
- `workspaces/{...}/{capsule_id}/` - Per-capsule workspace (notebooks, data files)
- `.pantheon/logs/benchmark/benchmark_{timestamp}.log` - Detailed logs

---

### 3. `cli` - Utility Commands

General utility commands for cleanup and status.

```bash
python -m benchmarks.bixbench.cli <command>
```

#### `status` - Show BixBench Status
```bash
python -m benchmarks.bixbench.cli status
```

#### `clean` - Clean Up Workspace Files

Removes temporary files (notebooks, data) from workspace directories while preserving benchmark results.

```bash
python -m benchmarks.bixbench.cli clean
```

**What gets cleaned:**
- ✅ **Timestamped workspace runs** in `workspaces/` (e.g., `workspaces/baseline_20260108_130351/`)
- ✅ **Legacy capsule directories** - Cleans contents only, preserves directory structure
- ❌ **Results are preserved** in `results/`

**To remove results:**
Manually delete specific run directories:
```bash
# Remove a specific run
rm -rf benchmarks/bixbench/results/baseline_20260108_111336

# Remove all old runs (be careful!)
rm -rf benchmarks/bixbench/results/baseline_202601*
```


#### `regrade` - Re-grade Answers Using LLM

Re-evaluate benchmark answers using LLM semantic verification for better accuracy.

```bash
# Use default model (gemini-3-flash)
python -m benchmarks.bixbench.cli regrade baseline_20260108_111336

# Specify custom model
python -m benchmarks.bixbench.cli regrade baseline_20260108_111336 --model openai/gpt-4o
python -m benchmarks.bixbench.cli regrade baseline_20260108_111336 -m claude-3-5-sonnet
```

| Argument | Description | Default |
|----------|-------------|---------|
| `run_dir` | Path to run directory (or name in results/) | **Required** |
| `--model`, `-m` | LLM model for grading | `gemini-3-flash` |

**Features:**
- Batch LLM grading for semantic equivalence
- Handles percentage/decimal conversions (35% ≈ 0.35 ≈ 35.0)
- 5% relative error tolerance
- Non-destructive: creates separate `results_<run_name>_regrade.json`

**Output:**
- `results/results_<run_name>_regrade.json` - Regrade results with:
  - `original_run`: Original accuracy statistics
  - `llm_regrade`: New accuracy and changes
  - `question_grades`: Detailed per-question comparison


---

### 4. `compare` - Compare Results

Compare baseline and learning-enabled runs.

```bash
python -m benchmarks.bixbench.compare
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--results-dir` | Results directory | `benchmarks/bixbench/results` |

**Output:**
```
📊 Comparison Results
==================================================
Baseline:      15/43 (34.9%)
With Learning: 20/43 (46.5%)
Improvement:   +5 questions (+11.6%)

Per-Capsule Comparison:
  bix-1:  baseline 1/2, learning 2/2 (+1)
  bix-10: baseline 3/7, learning 4/7 (+1)
  ...
```

---

### 5. `batch_learn` - Batch Learning from Memory Files

Process memory files from a completed benchmark run to build a skillbook.

```bash
python -m benchmarks.bixbench.batch_learn --memory-dir <path>
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--memory-dir` | Directory containing memory JSON files | **Required** |
| `--output` | Output skillbook JSON file | `{memory_dir}/skillbook_batch.json` |
| `--mode` | Learning mode (`pipeline` or `team`) | `pipeline` |
| `--learning-model` | Model for learning | From settings |
| `--config` | Additional config overrides as `KEY=VALUE` pairs | None |

**Available Config Keys** (via `--config`):
- `max_tool_arg_length` - Max chars for tool arguments in compression
- `max_tool_output_length` - Max chars for tool output in compression
- `min_confidence_threshold` - Min confidence for reflection (0.0-1.0)
- `min_atomicity_score` - Min atomicity score for skills (0.0-1.0)
- `team_id` - Team ID for team mode
- `workspace_path` - Workspace path for endpoint (team mode)
- `cleanup_after_learning` - Whether to cleanup learning files (true/false)

**Examples:**
```bash
# Basic usage (skillbook saved to memory_dir/skillbook_batch.json)
python -m benchmarks.bixbench.batch_learn \
    --memory-dir benchmarks/bixbench/results/baseline_20260108_120000

# Custom output location
python -m benchmarks.bixbench.batch_learn \
    --memory-dir benchmarks/bixbench/results/baseline_20260108_120000 \
    --output .pantheon/ace/skillbook_bixbench.json

# With compression tuning
python -m benchmarks.bixbench.batch_learn \
    --memory-dir benchmarks/bixbench/results/baseline_20260108_120000 \
    --config max_tool_arg_length=500 max_tool_output_length=1000

# With quality filtering
python -m benchmarks.bixbench.batch_learn \
    --memory-dir benchmarks/bixbench/results/baseline_20260108_120000 \
    --learning-model gemini-2.0-flash-exp \
    --config min_confidence_threshold=0.8 min_atomicity_score=0.95

# Team mode with custom settings
python -m benchmarks.bixbench.batch_learn \
    --memory-dir benchmarks/bixbench/results/baseline_xxx \
    --mode team \
    --config team_id=bixbench_learning workspace_path=/workspace
```

**Note**: By default, skillbook is saved inside the memory directory to avoid conflicts when running multiple batch learning sessions.

---

## Evaluation Workflow

### Standard Benchmark

```bash
# Step 1: Prepare data
python -m benchmarks.bixbench.prepare_capsules -n 10

# Step 2: Run baseline (no learning)
python -m benchmarks.bixbench.run --capsule-limit 10

# Step 3: Run with learning
python -m benchmarks.bixbench.run --capsule-limit 10 --enable-learning

# Step 4: Compare
python -m benchmarks.bixbench.compare
```

### Offline Learning Workflow

```bash
# Step 1: Run baseline and collect memory files
python -m benchmarks.bixbench.run --capsule-limit 10

# Step 2: Batch learn from memory files
python -m benchmarks.bixbench.batch_learn \
    --memory-dir benchmarks/bixbench/results/baseline_xxx \
    --output .pantheon/ace/skillbook.json

# Step 3: Run with injection-only (uses learned skills)
# TODO: injection-only mode
```

---

## Directory Structure

```
benchmarks/bixbench/
├── README.md           # This file
├── adapter.py          # Pantheon team adapter for BixBench
├── batch_learn.py      # Batch learning script
├── cli.py              # Utility CLI (clean, status, regrade)
├── compare.py          # Results comparison script
├── grader.py           # Answer grading module
├── prepare_capsules.py # Data preparation script
├── run.py              # Main benchmark runner
├── configs/            # Configuration files
├── data/               # Downloaded capsule data
├── groundtruth/        # Ground truth answers (separated)
├── results/            # Benchmark results (timestamped)
│   ├── baseline_20260108_130351/
│   │   ├── summary.json
│   │   ├── bix-1-q1.json
│   │   └── bix-1_memory.json
│   └── results_baseline_20260108_130351_regrade.json
├── test_capsules/      # Prepared test prompts with placeholders
└── workspaces/         # Agent workspaces (timestamped per run)
    ├── baseline_20260108_130351/
    │   ├── bix-1/      # Capsule workspace
    │   └── bix-2/
    └── bix-*/          # Legacy capsule directories (if any)
```

---

## Notes

- **Agentic Evaluation**: This uses agentic evaluation where agents execute code and analyze data to answer questions.
- **Grading**: Uses official BixBench `str_verifier` grading (case-insensitive string matching).
- **Logs**: All logs are saved to `.pantheon/logs/benchmark/`.
