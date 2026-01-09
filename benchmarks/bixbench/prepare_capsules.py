"""
Prepare BixBench capsule data for REPL manual testing.

Downloads capsules from HuggingFace and generates test prompts with absolute paths.
"""
import json
import shutil
import zipfile
from pathlib import Path
from collections import defaultdict

import datasets
from huggingface_hub import hf_hub_download


# Base paths (absolute)
BASE_DIR = Path("/home/bakezq/pantheon/pantheon-agents")
WORKSPACES_DIR = BASE_DIR / "benchmarks/bixbench/workspaces"
DATA_DIR = BASE_DIR / "benchmarks/bixbench/data"
HF_REPO_ID = "futurehouse/bixbench"
GROUNDTRUTH_DIR = BASE_DIR / "benchmarks/bixbench/groundtruth"


def download_capsule_data(zip_filename: str) -> Path:
    """Download and extract capsule data from HuggingFace.
    
    Separates ground truth notebooks from actual data files.
    
    Args:
        zip_filename: Name of the zip file (e.g., 'CapsuleFolder-xxx.zip')
        
    Returns:
        Path to the extracted data directory (without ground truth)
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GROUNDTRUTH_DIR.mkdir(parents=True, exist_ok=True)
    
    extract_dir = DATA_DIR / zip_filename.replace(".zip", "")
    capsule_id = zip_filename.replace("CapsuleFolder-", "").replace(".zip", "")
    groundtruth_dest = GROUNDTRUTH_DIR / capsule_id
    
    # Skip if already extracted
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"  📁 Data already exists: {extract_dir}")
        return extract_dir
    
    print(f"  ⬇️  Downloading {zip_filename}...")
    zip_path = Path(hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=zip_filename,
        repo_type="dataset",
        local_dir=DATA_DIR,
    ))
    
    print(f"  📦 Extracting to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    
    # Move contents from nested Data folder if exists
    data_subfolder = None
    for p in extract_dir.rglob("*"):
        if p.is_dir() and "Data" in p.name:
            data_subfolder = p
            break
    
    if data_subfolder:
        for item in data_subfolder.iterdir():
            shutil.move(str(item), str(extract_dir / item.name))
        shutil.rmtree(data_subfolder)
    
    # IMPORTANT: Move ground truth notebooks to separate directory
    notebook_dirs = list(extract_dir.glob("CapsuleNotebook*"))
    if notebook_dirs:
        groundtruth_dest.mkdir(parents=True, exist_ok=True)
        for nb_dir in notebook_dirs:
            dest = groundtruth_dest / nb_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(nb_dir), str(dest))
            print(f"  🔒 Moved ground truth to: {dest}")
    
    # Remove zip file to save space
    zip_path.unlink(missing_ok=True)
    
    return extract_dir


def prepare_capsules(
    n: int = 3,
    output_dir: str = "benchmarks/bixbench/test_capsules",
) -> list[dict]:
    """Download and prepare n capsules for testing.
    
    Args:
        n: Number of capsules to prepare
        output_dir: Directory to save test prompts
        
    Returns:
        List of prepared capsule info dicts
    """
    output_path = BASE_DIR / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load BixBench dataset
    print("Loading BixBench dataset from HuggingFace...")
    ds = datasets.load_dataset(HF_REPO_ID, split="train")
    print(f"Total questions: {len(ds)}")
    
    # Group questions by capsule_uuid
    capsules = defaultdict(list)
    for item in ds:
        capsule_uuid = item["capsule_uuid"]
        capsules[capsule_uuid].append(item)
    
    print(f"Total unique capsules: {len(capsules)}")
    
    prepared = []
    capsule_uuids = list(capsules.keys())[:n]
    
    for i, capsule_uuid in enumerate(capsule_uuids):
        questions = capsules[capsule_uuid]
        first_q = questions[0]
        
        short_id = first_q.get("short_id", capsule_uuid[:8])
        zip_filename = first_q.get("data_folder", "")
        
        print(f"\n{'='*60}")
        print(f"Preparing capsule {i+1}/{n}: {short_id}")
        
        # Download data
        if zip_filename:
            data_path = download_capsule_data(zip_filename)
        else:
            data_path = None
            print("  ⚠️  No data folder specified")
        
        # Create output directory for prompts
        capsule_dir = output_path / short_id
        capsule_dir.mkdir(exist_ok=True)
        
        # Build capsule info with absolute paths
        info = {
            "capsule_uuid": capsule_uuid,
            "short_id": short_id,
            "hypothesis": first_q.get("hypothesis", ""),
            "result": first_q.get("result", ""),
            "data_folder": zip_filename,
            "data_path": str(data_path) if data_path else None,
            "paper": first_q.get("paper", ""),
            "questions": [
                {
                    "id": q.get("question_id", f"q{j+1}"),
                    "question": q.get("question", ""),
                    "ideal": q.get("ideal"),
                    "distractors": q.get("distractors", []),
                    "eval_mode": q.get("eval_mode", ""),
                }
                for j, q in enumerate(questions)
            ],
        }
        
        # Save capsule info to test_capsules dir
        with open(capsule_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        # Also save capsule metadata to data directory (without ground truth answers)
        # This gives the agent context about the task
        if data_path:
            capsule_metadata = {
                "capsule_uuid": capsule_uuid,
                "short_id": short_id,
                "hypothesis": first_q.get("hypothesis", ""),
                "paper": first_q.get("paper", ""),
                "questions": [
                    {
                        "id": q.get("question_id", f"q{j+1}"),
                        "question": q.get("question", ""),
                        # NOTE: Do NOT include ideal/distractors here - that's the answer!
                    }
                    for j, q in enumerate(questions)
                ],
            }
            with open(data_path / "capsule_info.json", "w") as f:
                json.dump(capsule_metadata, f, indent=2, ensure_ascii=False)
            print(f"  📄 Saved capsule_info.json to data directory")
        
        # Generate REPL test prompt with absolute paths
        prompt = generate_test_prompt(info)
        with open(capsule_dir / "test_prompt.txt", "w") as f:
            f.write(prompt)
        
        # Generate summary README
        summary = f"""# Capsule: {short_id}

## Hypothesis
{info['hypothesis'][:300]}{'...' if len(info['hypothesis']) > 300 else ''}

## Data
- Source: `{info['data_folder']}`
- Data Path: `{info['data_path']}`

## Questions ({len(info['questions'])})
"""
        for q in info['questions']:
            question_text = q['question'][:150] + '...' if len(q['question']) > 150 else q['question']
            summary += f"- **{q['id']}**: {question_text}\n"
        
        with open(capsule_dir / "README.md", "w") as f:
            f.write(summary)
        
        prepared.append(info)
        print(f"  ✓ {len(info['questions'])} questions saved")
        print(f"  📁 Data: {data_path}")
    
    # Create master index by scanning actual prepared capsule directories
    # This ensures we capture ALL prepared capsules, even if they were skipped during this run
    actual_capsules = []
    for capsule_dir in sorted(output_path.iterdir()):
        if capsule_dir.is_dir() and capsule_dir.name.startswith("bix-"):
            info_file = capsule_dir / "info.json"
            if info_file.exists():
                try:
                    with open(info_file) as f:
                        info = json.load(f)
                    actual_capsules.append({
                        "short_id": info["short_id"],
                        "questions": len(info.get("questions", [])),
                        "data_path": info.get("data_path"),
                    })
                except Exception as e:
                    print(f"⚠️  Warning: Failed to read {info_file}: {e}")
    
    index = {
        "prepared_count": len(actual_capsules),
        "capsules": actual_capsules,
    }
    with open(output_path / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ {len(actual_capsules)} total capsules ready ({len(prepared)} processed in this run)")
    print(f"📂 Capsules: {output_path}")
    print(f"📂 Data: {DATA_DIR}")
    print(f"📋 Index: {output_path / 'index.json'}")
    print(f"\n📋 Next: Workspaces will be created at runtime during benchmark runs")
    
    return prepared


def generate_test_prompt(info: dict) -> str:
    """Generate a test prompt for REPL manual testing with absolute paths."""
    short_id = info.get("short_id", "unknown")
    questions_text = "\n".join([
        f"{q['id']}: {q['question']}" 
        for q in info.get("questions", [])
    ])
    
    hypothesis = info.get("hypothesis", "N/A")
    data_path = info.get("data_path", "")
    workspace_path = info.get("workspace_path", "")
    
    return f'''You are an expert bioinformatician. Analyze the data and answer the research questions.

## IMPORTANT: Execution Rules

1. **NON-INTERACTIVE**: Do NOT ask the user any questions or request approval/feedback. Work autonomously until completion.
2. **NO USER INTERACTION**: Complete the entire analysis independently without requesting feedback.
3. **FINAL ANSWER ONLY**: Only communicate with the user when providing your final answers.

## Environment (PRE-INSTALLED - DO NOT REINSTALL)

The following R packages are ALREADY INSTALLED. Use `library()` directly, do NOT run install.packages() or BiocManager::install() for these:
- **DESeq2** (v1.42.1) - Differential expression analysis
- **clusterProfiler** (v4.10.1) - GO/KEGG enrichment with simplify()
- **org.Hs.eg.db** (v3.18.0) - Human gene annotation
- **AnnotationDbi** (v1.64.1) - Database interface
- **enrichplot** (v1.22.0) - Visualization

IMPORTANT: Do NOT try to install fgsea, ggtree, or DOSE - they are not needed for this analysis and will fail to compile.

## R Best Practices (CRITICAL FOR PERFORMANCE)

**FORBIDDEN - These will HANG the Jupyter kernel:**
- Do NOT use `parallel::mclapply()`, `parallel::parLapply()`, or any fork-based parallelism - they WILL hang in Jupyter and cause timeout.
- Do NOT use `foreach` with parallel backends.
- Use sequential `lapply()` or `sapply()` instead.

**For Gene Ontology Enrichment with enrichGO() and simplify():**
- Use `keyType = "ENSEMBL"` directly with ENSEMBL gene IDs (strip version with gsub("\\..*","",gene_id))
- Do NOT use `bitr()` - it is too slow for this dataset.
- **ABSOLUTE RULE**: NEVER run `simplify()` on an `enrichResult` object that was created with `pvalueCutoff > 0.1` or `qvalueCutoff > 0.1`. Doing so will try to compute similarity for 10,000+ terms, which takes hours and will result in a hard FAILURE of this task.
- If a GO term is "missing" after `simplify()`, assume it was successfully merged into a more significant redundant term, as per the `cutoff = 0.7` requirement. Do NOT try to retrieve it by increasing the `pvalueCutoff` beyond 0.1.
- Example (WILL CAUSE IMMEDIATE TIMEOUT):
  ```r
  # DANGEROUS - SHUTDOWN LIKELY
  ego <- enrichGO(..., pvalueCutoff = 1) 
  ego_simp <- simplify(ego, ...) 
  ```

## Paths (ABSOLUTE - USE THESE EXACT PATHS)

**Data files location (READ ONLY):**
```
{data_path}
```

**Your workspace (ALL output files go here):**
```
{{WORKSPACE_PATH}}
```

CRITICAL: 
- Read data from the data path above
- Write ALL output files to the workspace path
- Do NOT create files anywhere else

## Hypothesis
{hypothesis}

## Questions to Answer
{questions_text}

## Instructions
1. List files in the data directory: `{data_path}`
2. Create a Jupyter Notebook in your workspace for analysis
3. Use R code via %%R magic for DESeq2/clusterProfiler analysis (packages are pre-installed)
4. Answer each question precisely based on your analysis
5. For numeric answers, be precise to 4 decimal places
6. After completing analysis, use the `submit_answer` tool to submit all answers:
   ```
   submit_answer(answers={{"q1": "answer1", "q2": "answer2"}})
   ```

**Analysis Method**: Prefer using Jupyter Notebooks with %%R magic for R code execution.

Begin now. Work autonomously without asking questions.
'''


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare BixBench capsules for testing")
    parser.add_argument("-n", type=int, default=3, help="Number of capsules to prepare")
    args = parser.parse_args()
    
    prepare_capsules(n=args.n)
