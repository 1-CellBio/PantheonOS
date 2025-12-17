"""
ACE Skillbook module for Pantheon.

This module provides the core Skillbook and Skill classes for the
Agentic Context Engineering (ACE) long-term memory system.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pantheon.utils.log import logger


# ===========================================================================
# Skillbook Injection Prompt Constants
# ===========================================================================

SKILLBOOK_USAGE_INSTRUCTIONS = """\
**How to use these strategies:**
- Review skills relevant to your current task
- **When applying a strategy, cite its ID in your reasoning** (e.g., "Following [content_extraction-00001], I will extract the title...")
  - Citations enable precise tracking of strategy effectiveness
  - Makes reasoning transparent and auditable
  - Improves learning quality through accurate attribution
- Prioritize strategies with high success rates (helpful > harmful)
- Apply strategies when they match your context
- Adapt general strategies to your specific situation
- Learn from both successful patterns and failure avoidance
**Important:** These are learned patterns, not rigid rules. Use judgment.
"""

SKILL_LOADING_GUIDANCE = """\
## 🔧 Skill Loading

When a user message starts with `/` followed by a skill ID (e.g., `/scrna-workflow`):

1. **Confirm loading**: Tell the user you are loading that skill
2. **Read details**: If the skill content contains a file path reference (e.g., `see skills/xxx.md`), use tools to read the full content
3. **Execute**: Follow the skill's guidance for the subsequent task

**Example**:
- User input: `/scrna-workflow`
- Match: `[scrna-workflow] Standard scRNA-seq analysis workflow (see skills/omics/scrna.md)`
- Action: Read `.pantheon/skills/omics/scrna.md` and follow its instructions
"""

SKILLBOOK_HEADER = """\
## 📚 Available Strategic Knowledge (Learned from Experience)
The following strategies have been learned from previous interactions.
Each skill shows its success rate: (stats: +helpful / -harmful / ~neutral)
"""

USER_RULES_HEADER = """\
## 📌 User Rules (MUST FOLLOW)

These are explicit preferences set by the user. Apply them unless there's a strong reason not to.
"""


def _format_skillbook_for_injection(
    user_rules_text: str,
    strategies_text: str,
    include_loading_guidance: bool = True,
) -> str:
    """Format skillbook content for injection into agent system prompt."""
    parts = []

    # User rules section (highest priority, MUST follow)
    if user_rules_text:
        parts.append(USER_RULES_HEADER)
        parts.append(user_rules_text)
        parts.append("")

    # Other learned skills
    if strategies_text:
        parts.append(SKILLBOOK_HEADER)
        parts.append(strategies_text)
        parts.append("")
        parts.append(SKILLBOOK_USAGE_INSTRUCTIONS)

    # Skill loading guidance (always include if we have any skills)
    if include_loading_guidance and (user_rules_text or strategies_text):
        parts.append("")
        parts.append(SKILL_LOADING_GUIDANCE)

    return "\n".join(parts).strip()


@dataclass
class Skill:
    """Single skillbook entry representing a learned strategy or insight."""

    id: str  # Unique identifier, also used for /xxx trigger
    section: str  # user_rules | strategies | patterns | workflows
    content: str  # Skill content (max 500 chars), may contain file path reference
    helpful: int = 0
    harmful: int = 0
    neutral: int = 0
    agent_scope: str = "global"  # "global" | specific agent name

    # Source identification
    type: Optional[str] = None  # "system" = auto-learned, None/other = user-defined
    source_path: Optional[str] = None  # Source file path (relative to skills dir)

    # Optional metadata
    tags: List[str] = field(default_factory=list)
    learned_from: Optional[str] = None  # Learning source (chat ID, trajectory ID)

    # Timestamps
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: Literal["active", "invalid"] = "active"

    def tag(self, tag: str, increment: int = 1) -> None:
        """Apply a tag (helpful/harmful/neutral) to this skill."""
        if tag not in ("helpful", "harmful", "neutral"):
            raise ValueError(f"Unsupported tag: {tag}")
        current = getattr(self, tag)
        setattr(self, tag, current + increment)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def is_system(self) -> bool:
        """Check if this skill was auto-learned by the system."""
        return self.type == "system"

    def is_user_defined(self) -> bool:
        """Check if this skill was defined by the user (from files)."""
        return self.type == "user"

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Return dict with only LLM-relevant fields."""
        return {
            "id": self.id,
            "section": self.section,
            "content": self.content,
            "helpful": self.helpful,
            "harmful": self.harmful,
        }


class Skillbook:
    """
    Structured context store for ACE long-term memory.

    Manages a collection of learned skills that can be injected into
    agent prompts and updated based on agent performance.
    """

    def __init__(
        self,
        max_skills_per_section: int = 30,
        max_content_length: int = 500,
        enable_agent_scope: bool = False,
    ):
        self._skills: Dict[str, Skill] = {}
        self._sections: Dict[str, List[str]] = {}
        self._next_id = 0
        self._path: Optional[Path] = None
        self.max_skills_per_section = max_skills_per_section
        self.max_content_length = max_content_length
        self.enable_agent_scope = enable_agent_scope

    def __repr__(self) -> str:
        return f"Skillbook(skills={len(self._skills)}, sections={list(self._sections.keys())})"

    # ------------------------------------------------------------------ #
    # CRUD Operations
    # ------------------------------------------------------------------ #

    def add_skill(
        self,
        section: str,
        content: str,
        agent_scope: str = "global",
        skill_id: Optional[str] = None,
    ) -> Optional[Skill]:
        """
        Add a new skill to the skillbook (used by ACE learning pipeline).

        Returns None if section is full, otherwise returns the created skill.
        Note: Skills added through this method are marked as type='system'
        (auto-learned), distinguishing them from user-defined file skills.
        """
        # Enforce content length
        if len(content) > self.max_content_length:
            content = content[: self.max_content_length]
            logger.warning(
                f"Skill content truncated to {self.max_content_length} chars"
            )

        # Check section limit
        section_skills = self._sections.get(section, [])
        if len(section_skills) >= self.max_skills_per_section:
            # Try to evict worst skill
            evicted = self._evict_worst_skill(section)
            if not evicted:
                logger.warning(f"Section '{section}' is full, cannot add skill")
                return None

        skill_id = skill_id or self._generate_id(section)
        skill = Skill(
            id=skill_id,
            section=section,
            content=content,
            agent_scope=agent_scope,
            type="system",  # Mark as auto-learned by ACE pipeline
        )
        self._skills[skill_id] = skill
        self._sections.setdefault(section, []).append(skill_id)
        return skill

    def update_skill(
        self,
        skill_id: str,
        content: Optional[str] = None,
    ) -> Optional[Skill]:
        """Update an existing skill's content."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return None
        if content is not None:
            if len(content) > self.max_content_length:
                content = content[: self.max_content_length]
            skill.content = content
        skill.updated_at = datetime.now(timezone.utc).isoformat()
        return skill

    def tag_skill(self, skill_id: str, tag: str, increment: int = 1) -> Optional[Skill]:
        """Apply a tag to a skill."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return None
        skill.tag(tag, increment)
        return skill

    def remove_skill(self, skill_id: str, soft: bool = True) -> None:
        """
        Remove a skill from the skillbook.

        Args:
            skill_id: ID of the skill to remove
            soft: If True, mark as invalid; if False, delete entirely
        """
        skill = self._skills.get(skill_id)
        if skill is None:
            return

        if soft:
            skill.status = "invalid"
            skill.updated_at = datetime.now(timezone.utc).isoformat()
        else:
            self._skills.pop(skill_id, None)
            section_list = self._sections.get(skill.section, [])
            self._sections[skill.section] = [
                sid for sid in section_list if sid != skill_id
            ]
            if not self._sections[skill.section]:
                del self._sections[skill.section]

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        return self._skills.get(skill_id)

    def skills(self, include_invalid: bool = False) -> List[Skill]:
        """Get all skills (active only by default)."""
        if include_invalid:
            return list(self._skills.values())
        return [s for s in self._skills.values() if s.status == "active"]

    # ------------------------------------------------------------------ #
    # Query Methods
    # ------------------------------------------------------------------ #

    def get_skills_for_agent(self, agent_name: str) -> List[Skill]:
        """
        Get skills applicable to a specific agent.

        If enable_agent_scope is False, returns all skills.
        If True, returns global skills + agent-specific skills.
        Skills are sorted by helpfulness.
        """
        if not self.enable_agent_scope:
            # Return all skills when scope filtering is disabled
            return self._sort_skills_by_helpfulness(self.skills())

        # Filter by scope
        applicable = [
            s
            for s in self.skills()
            if s.agent_scope == "global" or s.agent_scope == agent_name
        ]
        return self._sort_skills_by_helpfulness(applicable)

    def get_skills_by_section(self, section: str) -> List[Skill]:
        """Get all active skills in a section."""
        skill_ids = self._sections.get(section, [])
        return [
            self._skills[sid]
            for sid in skill_ids
            if sid in self._skills and self._skills[sid].status == "active"
        ]

    # ------------------------------------------------------------------ #
    # Presentation
    # ------------------------------------------------------------------ #

    def _format_skill_content(self, skill: Skill) -> str:
        """
        Format skill content for display, adding file reference if applicable.

        Only adds file reference for file-based skills (not SKILLS.md rules).
        Converts relative source_path to absolute path for agent usability.
        """
        content = skill.content

        # Include stats: (stats: +5/-0/~2)
        stats = f"(stats: +{skill.helpful}/-{skill.harmful}/~{skill.neutral})"
        content = f"{stats} {skill.content}"

        # Add file reference for file-based skills (exclude SKILLS.md)
        if skill.source_path and not skill.source_path.endswith("SKILLS.md"):
            # Convert relative path (relative to skills_dir) to absolute
            from pantheon.settings import get_settings
            abs_path = get_settings().skills_dir / skill.source_path
            content = f"{content} (see `{abs_path}`)"

        return content

    def as_prompt(self, agent_name: str) -> str:
        """
        Format skillbook as a prompt section for LLM injection.

        User rules (user_rules section) are presented as MUST FOLLOW rules.
        Other skills are presented as learned strategies with usage instructions.
        Skills are sorted by helpfulness (helpful - harmful), highest first.
        Returns empty string if no applicable skills.
        """

        skills = self.get_skills_for_agent(agent_name)
        if not skills:
            return ""

        # Separate user_rules from other sections
        user_rules = [s for s in skills if s.section == "user_rules"]
        other_skills = [s for s in skills if s.section != "user_rules"]

        # Format user_rules
        user_rules_text = ""
        if user_rules:
            user_rules_text = "\n".join(
                f"[{s.id}] {self._format_skill_content(s)}" for s in user_rules
            )

        # Format other skills by section
        strategies_text = ""
        if other_skills:
            sections: Dict[str, List[Skill]] = {}
            for skill in other_skills:
                sections.setdefault(skill.section, []).append(skill)

            parts = []
            for section_name in sorted(sections.keys()):
                section_skills = sections[section_name]
                parts.append(f"### {section_name.upper()}")
                for skill in section_skills:
                    parts.append(f"[{skill.id}] {self._format_skill_content(skill)}")
                parts.append("")
            strategies_text = "\n".join(parts).strip()

        return _format_skillbook_for_injection(user_rules_text, strategies_text)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: Optional[str] = None) -> None:
        """Save skillbook to JSON file."""
        if path:
            save_path = Path(path)
        elif self._path:
            save_path = Path(self._path) if isinstance(self._path, str) else self._path
        else:
            logger.warning("No path specified for skillbook save")
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(self._to_dict(), f, ensure_ascii=False, indent=2)
        logger.debug(f"Skillbook saved to {save_path}")

    def load(self, path: str) -> None:
        """Load skillbook from JSON file."""
        self._path = Path(path)
        if not self._path.exists():
            logger.info(f"Skillbook file not found: {path}, starting fresh")
            return

        try:
            with self._path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self._from_dict(data)
            logger.info(f"Loaded skillbook with {len(self._skills)} skills")
        except Exception as e:
            logger.error(f"Failed to load skillbook: {e}")

    def _to_dict(self) -> Dict[str, Any]:
        """Serialize skillbook to dictionary."""
        return {
            "skills": {
                skill_id: asdict(skill) for skill_id, skill in self._skills.items()
            },
            "sections": self._sections,
            "next_id": self._next_id,
        }

    def _from_dict(self, data: Dict[str, Any]) -> None:
        """Deserialize skillbook from dictionary."""
        skills_data = data.get("skills", {})
        for skill_id, skill_dict in skills_data.items():
            # Handle backwards compatibility for new fields
            if "status" not in skill_dict:
                skill_dict["status"] = "active"
            if "type" not in skill_dict:
                skill_dict["type"] = None
            if "source_path" not in skill_dict:
                skill_dict["source_path"] = None
            if "tags" not in skill_dict:
                skill_dict["tags"] = []
            if "learned_from" not in skill_dict:
                skill_dict["learned_from"] = None

            self._skills[skill_id] = Skill(**skill_dict)

        sections_data = data.get("sections", {})
        self._sections = {section: list(ids) for section, ids in sections_data.items()}
        self._next_id = data.get("next_id", 0)

    # ------------------------------------------------------------------ #
    # Internal Helpers
    # ------------------------------------------------------------------ #

    def _generate_id(self, section: str) -> str:
        """Generate unique skill ID."""
        self._next_id += 1
        # Use first word of section as prefix
        prefix = re.sub(r"[^a-z]", "", section.split()[0].lower())[:3]
        return f"{prefix}-{self._next_id:05d}"

    def _sort_skills_by_helpfulness(self, skills: List[Skill]) -> List[Skill]:
        """Sort skills by (helpful - harmful), highest first."""
        return sorted(skills, key=lambda s: (s.helpful - s.harmful), reverse=True)

    def _evict_worst_skill(self, section: str) -> bool:
        """
        Evict the worst skill from a section if it has negative score.

        Returns True if a skill was evicted, False otherwise.
        """
        section_skills = self.get_skills_by_section(section)
        if not section_skills:
            return False

        worst = min(section_skills, key=lambda s: s.helpful - s.harmful)
        if worst.harmful > worst.helpful:
            self.remove_skill(worst.id, soft=False)
            logger.info(f"Evicted low-quality skill: {worst.id}")
            return True
        return False

    def stats(self) -> Dict[str, Any]:
        """Get detailed skillbook statistics."""
        active_skills = self.skills()

        # Section breakdown
        section_stats = {}
        for section, skill_ids in self._sections.items():
            active_in_section = [
                sid
                for sid in skill_ids
                if sid in self._skills and self._skills[sid].status == "active"
            ]
            section_stats[section] = len(active_in_section)

        # Calculate net score
        total_helpful = sum(s.helpful for s in self._skills.values())
        total_harmful = sum(s.harmful for s in self._skills.values())

        return {
            "total_skills": len(self._skills),
            "active_skills": len(active_skills),
            "sections": len(self._sections),
            "section_breakdown": section_stats,
            "tags": {
                "helpful": total_helpful,
                "harmful": total_harmful,
                "neutral": sum(s.neutral for s in self._skills.values()),
            },
            "net_score": total_helpful - total_harmful,
        }

    def summary_line(self) -> str:
        """Get a one-line summary of the skillbook for logging."""
        s = self.stats()
        sections_str = (
            ", ".join(f"{k}:{v}" for k, v in s["section_breakdown"].items()) or "empty"
        )
        return (
            f"Skillbook: {s['active_skills']} skills | "
            f"Sections: [{sections_str}] | "
            f"Score: +{s['tags']['helpful']}/-{s['tags']['harmful']}"
        )
