---
category: learning
description: |
  Team V2 for skill learning - simplified architecture.
  Agents do NOT handle sources - SkillbookToolSet auto-converts long content internally.
icon: 🧠
id: skill_learning_team_v2
name: Skill Learning Team V2
type: team
version: 1.0.0
agents:
  - skill_learning_v2/skill_manager
  - skill_learning_v2/reflector
---

# Skill Learning Team V2

A 2-agent team for extracting knowledge into a skillbook. **Simplified architecture** - agents don't handle sources.

## Key Difference from V1

| Aspect | V1 | V2 |
|--------|----|----|
| **sources handling** | Agent creates /tmp files | SkillbookToolSet auto-converts |
| **Prompt complexity** | High (file formats, front matter) | Low |
| **Agent count** | 3 (manager, reflector, developer) | 2 (coordinator, analyst) |

---

## Agent Roles

| Agent | Icon | Role | Tools |
|-------|------|------|-------|
| **skill_coordinator** | 📚 | Orchestrate, decide, update | skillbook |
| **skill_analyst** | 🔍 | Analyze sources, extract learnings | file_manager |

---

## Architecture

```
              ┌─────────────────────┐
              │  Skill Coordinator  │
              │      (Leader)       │
              │                     │
              │ compress_trajectory()
              │ add_skill(content, description)
              │ → auto-converts internally!
              └──────────┬──────────┘
                         │
                         ▼
               ┌─────────────────┐
               │  Skill Analyst  │
               │                 │
               │ read trajectory │
               │ extract content │
               │ + description   │
               │ (NO sources!)   │
               └─────────────────┘
```

---

## Workflow

### Memory/Conversation → Skills

```
1. coordinator: compress_trajectory(memory_path)
   → trajectory_path + details_path

2. coordinator: call skill_analyst("Analyze: ...")
   
3. analyst: read trajectory
   → extract learnings: {content, description, confidence}
   → NO /tmp files created!
   → return JSON

4. coordinator: process learnings
   → list_skills() check duplicates
   → add_skill(content, description)
      → SkillbookToolSet auto-converts long content to sources!
   → tag_skill() for skill_tags
```

---

## Auto-Conversion (Internal)

When `add_skill(content, description)` is called:

```
IF len(content) > 500 chars:
  1. Create {skill_id}.md with front matter
  2. Store full content in file
  3. skill.content = description (short)
  4. skill.sources = ["{skill_id}.md"]
ELSE:
  skill.content = content (as-is)
```

**Agents never need to know about this!**

---

## Shared Resources

| Resource | Managed By | Notes |
|----------|------------|-------|
| `skillbook.json` | SkillbookToolSet | Skills + ratings |
| `skills/*.md` | SkillbookToolSet | Auto-generated for long content |
| `SKILLS.md` | User | User-defined, NOT modified |

---

## Key Points

1. **coordinator** only passes `content` + `description`
2. **analyst** does NOT create files
3. **Long content → sources** happens inside SkillbookToolSet
4. **Simpler prompts** = fewer Agent errors
