---
category: learning
description: |
  Team V3 for skill learning - 3-agent architecture with Pipeline prompts.
  Coordinator orchestrates workflow, Reflector/SkillManager use exact Pipeline prompts.
icon: 🧠
id: skill_learning_team_v3
name: Skill Learning Team V3
type: team
version: 1.0.0
agents:
  - skill_learning_v3/coordinator
  - skill_learning_v3/reflector
  - skill_learning_v3/skill_manager
---

# Skill Learning Team V3

A 3-agent team that uses exact Pipeline prompts for consistency.

## Architecture

```
              ┌─────────────────────┐
              │     Coordinator     │  ← Has tools (skillbook)
              │  (workflow + tools) │
              └──────────┬──────────┘
                         │
          ┌──────────────┴──────────────┐
          ↓                             ↓
┌───────────────────┐         ┌───────────────────┐
│     Reflector     │         │   Skill Manager   │
│ (Pipeline prompt) │         │ (Pipeline prompt) │
│    No tools       │         │    No tools       │
│    Returns JSON   │         │    Returns JSON   │
└───────────────────┘         └───────────────────┘
```

## Agent Roles

| Agent | Role | Tools |
|-------|------|-------|
| **coordinator** | Orchestrate workflow, call sub-agents, apply operations | skillbook |
| **reflector** | Analyze trajectory, extract learnings | None |
| **skill_manager** | Decide update operations | None |

## Workflow

```
1. Coordinator: compress_trajectory(memory_path)
2. Coordinator: call_agent("reflector", ...)
3. Coordinator: apply skill_tags via tag_skill()
4. Coordinator: call_agent("skill_manager", ...)
5. Coordinator: apply operations via add_skill/update_skill/...
6. Coordinator: report summary
```

## Key Features

- **Same prompts as Pipeline** - Reflector/SkillManager prompts identical
- **Coordinator handles flow** - Only orchestration, no analysis
- **Auto-save** - Each skillbook operation saves automatically
- **User-defined protection** - Cannot modify user-defined skills
