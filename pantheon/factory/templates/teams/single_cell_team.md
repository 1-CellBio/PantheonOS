---
category: bioinformatics
description: |
  AI team for autonomous exploratory analysis of single-cell and spatial omics data.
  Features 6 specialized agents for data analysis, hypothesis generation, and publication-quality reporting.
icon: 🧬
id: single_cell_team
name: Single Cell Analysis Team
type: team
version: 1.0.0
agents:
  - single_cell/leader
  - single_cell/analysis_expert
  - single_cell/biologist
  - single_cell/reporter
  - single_cell/system_manager
  - single_cell/browser_use
---

# Single Cell Analysis Team

A specialized AI team for autonomous exploratory analysis of single-cell and spatial omics data (e.g., scRNA-seq, MERFISH, Visium).

## Team Structure

| Agent | Role | Key Capabilities |
|-------|------|------------------|
| **leader** | Orchestrator | Task delegation, workflow management, workdir organization |
| **analysis_expert** | Data Analyst | Python/notebook analysis, visualization, skills system |
| **biologist** | Domain Expert | Hypothesis generation, biological interpretation |
| **reporter** | Report Writer | LaTeX/PDF report generation, figure organization |
| **system_manager** | DevOps | Environment investigation, package installation |
| **browser_use** | Researcher | Web search, literature collection, reference management |

## Core Workflow

1. **Environment Understanding**: Check computational environment, record in `environment.md`
2. **Dataset Analysis**: Delegate to `analysis_expert` for data understanding and QC
3. **Hypothesis Generation**: Delegate to `biologist` for generating research directions
4. **Planning**: Create `todolist.md` with analysis plan
5. **Execution Loop**: Iteratively run analysis → biological interpretation
6. **Reporting**: Delegate to `reporter` for publication-quality PDF report

## Work Intensity Levels

| Level | Keyword | Analysis Loops |
|-------|---------|----------------|
| Low | "basic" | 1 loop |
| Medium | (default) | 3 loops |
| High | "deep", "hard" | ≥5 loops |

## Skills System

Agents can access domain-specific best practices from `analysis-skills/SKILL.md`.
