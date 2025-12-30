"""
SkillbookToolSet for Skill Learning Team.

This module provides LLM tools for the Skill Manager agent to manage the skillbook.
Implements confidence filtering and delegates to Skillbook for persistence.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pantheon.toolset import ToolSet, tool
from pantheon.utils.log import logger

if TYPE_CHECKING:
    from pantheon.internal.learning.skillbook import Skill, Skillbook


# ===========================================================================
# SkillbookToolSet
# ===========================================================================


class SkillbookToolSet(ToolSet):
    """
    Skillbook management tools for Skill Manager agent.
    
    LLM Tools (@tool decorated):
    - add_skill: Add a new skill
    - update_skill: Update existing skill
    - remove_skill: Remove a skill
    - tag_skill: Tag a skill as helpful/harmful/neutral
    - list_skills: List all skills
    - get_skill: Get skill details
    
    Internal Methods (not exposed to LLM):
    - as_prompt: Generate prompt for injection
    - load/save: Load/save skillbook
    - merge_from_files: Merge user-defined skills from files
    """

    def __init__(
        self,
        skillbook: Optional["Skillbook"] = None,
        min_confidence: float = 0.7,
        name: str = "skillbook",  # For Endpoint compatibility
        **kwargs,
    ):
        """
        Initialize SkillbookToolSet.
        
        Args:
            skillbook: The Skillbook instance to manage. If None, auto-creates from settings.
            min_confidence: Minimum confidence threshold for adding skills (default 0.7)
            name: Service name for Endpoint registration (default "skillbook")
        
        If skillbook is not provided, a new Skillbook is created with default settings
        (paths from settings.json, auto-load enabled). This allows Endpoint to auto-start
        this toolset via start_services(["skillbook"]).
        """
        super().__init__(name, **kwargs)
        if skillbook is None:
            from pantheon.internal.learning.skillbook import Skillbook
            skillbook = Skillbook()  # Auto-creates with settings, auto-loads
        
        self.skillbook = skillbook
        self.min_confidence = min_confidence
    
    @property
    def skills_dir(self) -> Path:
        """Delegate to Skillbook.skills_dir."""
        return self.skillbook.skills_dir

    # ======================================================================= #
    # LLM Tool Functions
    # ======================================================================= #

    @tool
    def add_skill(
        self,
        section: str,
        content: str,
        description: Optional[str] = None,
        agent_name: str = "global",
        sources: Optional[List[str]] = None,
        skill_id: Optional[str] = None,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Add a new skill to the skillbook.
        
        Use this to record a new learning, pattern, or rule extracted from analysis.
        Always call list_skills() first to check for duplicates - prefer update_skill()
        for existing similar skills.

        Args:
            section: Category for the skill:
                - "user_rules": User preferences and explicit instructions
                - "strategies": Problem-solving approaches and methodologies  
                - "patterns": Reusable code patterns and templates
                - "workflows": Multi-step procedures and pipelines
            content: The skill content (no length limit). Should be specific and actionable.
                Good: "Use pl.scan_csv() for streaming CSV files > 1GB"
                Bad: "Use appropriate tools for large files"
            description: Short summary (max 15 words) for prompt display.
                Required if content > 100 chars.
            agent_name: Agent this skill applies to. Use "global" for all agents,
                or specific agent name for agent-scoped skills.
            sources: Optional list of temp file paths containing detailed examples,
                code, or documentation to attach to this skill
            skill_id: Custom ID (auto-generated as "<section[:3]>-<uuid>" if omitted)
            confidence: How confident you are this skill is correct (0.0-1.0).
                Skills below threshold (default 0.7) are rejected.

        Returns:
            {"success": True, "skill_id": "...", "section": "...", ...} on success
            {"success": False, "error": "..."} on failure

        Example:
            add_skill(
                section="strategies",
                content="Use polars lazy evaluation for memory-efficient large file processing",
                agent_name="data_analyst",
                confidence=0.9
            )
        """
        # Check confidence threshold
        if confidence < self.min_confidence:
            return {
                "success": False,
                "error": f"Confidence {confidence:.2f} below threshold {self.min_confidence}",
            }
        # Generate skill ID if not provided
        if not skill_id:
            prefix = section[:3].lower()
            skill_id = f"{prefix}-{uuid.uuid4().hex[:8]}"

        # Handle sources
        final_sources: List[str] = []
        if sources:
            final_sources = self.skillbook.copy_sources(skill_id, sources)

        # Add to skillbook with description, sources, and agent_scope
        skill = self.skillbook.add_skill(
            section=section,
            content=content,
            skill_id=skill_id,
            sources=final_sources,
            agent_scope=agent_name,  # Map agent_name to agent_scope
            description=description,
        )

        if skill is None:
            return {
                "success": False,
                "error": f"Failed to add skill: section '{section}' may be full",
            }

        # Sync front matter in first source file only (must be markdown)
        # This adds skill metadata (id, description, section, type) to the main file
        if final_sources:
            primary_source = self.skills_dir / final_sources[0]
            if primary_source.suffix == '.md' and primary_source.exists():
                self.skillbook.sync_front_matter(
                    primary_source,
                    skill_id=skill_id,
                    description=content,
                    section=section,
                )

        # Save skillbook
        self.skillbook.save()

        return {
            "success": True,
            "skill_id": skill_id,
            "message": f"Added skill '{skill_id}' to {section}",
        }

    @tool
    def update_skill(
        self,
        skill_id: str,
        content: Optional[str] = None,
        description: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Update an existing skill's content or attached files.
        
        Prefer this over add_skill() when a similar skill already exists.
        Updates are automatically timestamped.

        Args:
            skill_id: ID of the skill to update (e.g., "str-a1b2c3d4")
            content: New content text (no length limit). Pass None to keep unchanged.
            description: Short summary for long content (max 15 words).
            sources: New source file paths. Replaces all existing files.

        Returns:
            {"success": True, "skill_id": "...", ...} on success
            {"success": False, "error": "..."} on failure
        """
        skill = self.skillbook.get_skill(skill_id)
        if skill is None:
            return {"success": False, "error": f"Skill '{skill_id}' not found"}

        # Handle sources update first (before content update)
        if sources is not None:
            # Check user-defined protection before modifying
            if skill.is_user_defined():
                return {
                    "success": False,
                    "error": f"Cannot modify user-defined skill '{skill_id}'. Edit the source file directly.",
                }
            # Delete old sources
            if skill.sources:
                self.skillbook.delete_sources(skill.sources)
            # Copy new sources
            skill.sources = self.skillbook.copy_sources(skill_id, sources)
            # Sync front matter in first source file only (must be markdown)
            if skill.sources:
                primary_source = self.skills_dir / skill.sources[0]
                if primary_source.suffix == '.md' and primary_source.exists():
                    self.skillbook.sync_front_matter(
                        primary_source,
                        skill_id=skill_id,
                        description=skill.content,
                        section=skill.section,
                    )

        # Update content using Skillbook.update_skill (handles auto-conversion + user-defined protection)
        if content is not None:
            result = self.skillbook.update_skill(skill_id, content=content, description=description)
            if result is None:
                return {
                    "success": False,
                    "error": f"Cannot modify user-defined skill '{skill_id}'. Edit the source file directly.",
                }

        self.skillbook.save()

        return {
            "success": True,
            "skill_id": skill_id,
            "message": f"Updated skill '{skill_id}'",
        }

    @tool
    def remove_skill(self, skill_id: str) -> Dict[str, Any]:
        """
        Permanently remove a skill and its attached files.
        
        Use for skills that are incorrect, outdated, or harmful.
        This is a hard delete - the skill cannot be recovered.

        Args:
            skill_id: ID of the skill to remove (e.g., "str-a1b2c3d4")

        Returns:
            {"success": True, "skill_id": "...", "message": "..."} on success
            {"success": False, "error": "Skill not found"} if skill_id invalid
        """
        skill = self.skillbook.get_skill(skill_id)
        if skill is None:
            return {"success": False, "error": f"Skill '{skill_id}' not found"}
        
        # Protect user-defined skills from deletion
        if skill.is_user_defined():
            return {
                "success": False,
                "error": f"Cannot remove user-defined skill '{skill_id}'. Delete the source file directly.",
            }

        # Delete source files
        if skill.sources:
            self.skillbook.delete_sources(skill.sources)

        # Remove from skillbook (hard delete)
        self.skillbook.remove_skill(skill_id, soft=False)
        self.skillbook.save()

        return {
            "success": True,
            "skill_id": skill_id,
            "message": f"Removed skill '{skill_id}'",
        }

    @tool
    def tag_skill(
        self,
        skill_id: str,
        tag: Literal["helpful", "harmful", "neutral"],
    ) -> Dict[str, Any]:
        """
        Record feedback on whether a skill was helpful or harmful.
        
        Call this after observing how a skill performed in practice.
        Tags accumulate over time to build skill quality metrics.

        Args:
            skill_id: ID of the skill to tag (e.g., "str-a1b2c3d4")
            tag: Feedback type:
                - "helpful": Skill contributed to task success
                - "harmful": Skill led to errors or wrong approach
                - "neutral": Skill was referenced but had no clear impact

        Returns:
            {"success": True, "stats": "+3/-1/~0", ...} showing cumulative counts
            {"success": False, "error": "Skill not found"} if skill_id invalid

        Example:
            tag_skill(skill_id="str-a1b2c3d4", tag="helpful")
        """
        skill = self.skillbook.tag_skill(skill_id, tag)
        if skill is None:
            return {"success": False, "error": f"Skill '{skill_id}' not found"}

        self.skillbook.save()

        return {
            "success": True,
            "skill_id": skill_id,
            "stats": f"+{skill.helpful}/-{skill.harmful}/~{skill.neutral}",
            "message": f"Tagged skill '{skill_id}' as {tag}",
        }

    @tool
    def list_skills(
        self,
        section: Optional[str] = None,
        tag: Optional[str] = None,
        keyword: Optional[str] = None,
        include_full_content: bool = False,
    ) -> Dict[str, Any]:
        """
        List and filter skills in the skillbook.
        
        Always call this before add_skill() to check for existing similar skills.
        Use filters to narrow down results.

        Args:
            section: Filter by category:
                - "user_rules", "strategies", "patterns", "workflows"
                - None (default) returns all sections
            tag: Filter by dominant feedback:
                - "helpful": Skills with more helpful than harmful tags
                - "harmful": Skills with more harmful than helpful tags  
                - "neutral": Skills with only neutral feedback
            keyword: Case-insensitive search in skill content
            include_full_content: If True, return complete content.
                If False (default), truncate to 100 chars for overview.

        Returns:
            {
                "total": 5,
                "filters": {"section": "...", "tag": "...", "keyword": "..."},
                "skills": [
                    {"id": "str-xxx", "section": "...", "content": "...", 
                     "sources": [...], "stats": "+3/-0/~1"}
                ]
            }

        Example:
            list_skills(section="strategies", keyword="polars")
        """
        skills = self.skillbook.skills()
        
        # Apply filters
        if section:
            skills = [s for s in skills if s.section == section]
        
        if tag:
            # Filter by most common feedback tag
            def get_dominant_tag(s):
                if s.harmful > s.helpful and s.harmful > s.neutral:
                    return "harmful"
                elif s.helpful > s.harmful and s.helpful > s.neutral:
                    return "helpful"
                elif s.neutral > 0:
                    return "neutral"
                return None
            skills = [s for s in skills if get_dominant_tag(s) == tag]
        
        if keyword:
            keyword_lower = keyword.lower()
            skills = [s for s in skills if keyword_lower in s.content.lower()]

        skill_list = []
        for s in skills:
            content = s.content if include_full_content else (
                s.content[:100] + "..." if len(s.content) > 100 else s.content
            )
            skill_list.append({
                "id": s.id,
                "section": s.section,
                "content": content,
                "sources": s.sources,
                "stats": f"+{s.helpful}/-{s.harmful}/~{s.neutral}",
            })

        return {
            "total": len(skill_list),
            "filters": {"section": section, "tag": tag, "keyword": keyword},
            "skills": skill_list,
        }

    @tool
    def compress_trajectory(
        self,
        memory_path: str,
        output_dir: Optional[str] = None,
        max_arg_length: Optional[int] = None,
        max_output_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compress a conversation memory file into trajectory for analysis.
        
        Call this first when analyzing a conversation. Produces a truncated summary
        that's easier for LLMs to analyze, while preserving full details for reference.
        Also extracts which skills were referenced in the conversation.

        Args:
            memory_path: Path to memory.json file containing conversation messages
            output_dir: Where to save trajectory file (default: same directory as memory_path)
            max_arg_length: Max characters for tool argument values in output (default: from settings)
            max_output_length: Max characters for tool output values (default: from settings)

        Returns:
            {
                "success": True,
                "trajectory_path": "/path/to/trajectory_xxx.txt",  # Read this for analysis
                "details_path": "/path/to/memory.json",            # Original full content
                "skill_ids_cited": ["str-xxx", "pat-yyy"],         # Skills referenced
                "message": "Compressed to ..."
            }

        Example:
            result = compress_trajectory("/tmp/memory.json")
            # Then read trajectory_path with file_manager
            # Pass skill_ids_cited to reflector for tagging
        """
        from pantheon.utils.memory_compress import compress_memory
        from pantheon.settings import get_settings

        try:
            # Get settings defaults if not provided
            if max_arg_length is None or max_output_length is None:
                settings = get_settings()
                learning_config = settings.get_learning_config()
                if max_arg_length is None:
                    max_arg_length = learning_config.get("max_tool_arg_length", 200)
                if max_output_length is None:
                    max_output_length = learning_config.get("max_tool_output_length", 500)

            # Default output_dir to same directory as memory_path
            if output_dir is None:
                output_dir = str(Path(memory_path).parent)

            compressed = compress_memory(
                memory_path=memory_path,
                output_dir=output_dir,
                max_arg_length=max_arg_length,
                max_output_length=max_output_length,
            )
            return {
                "success": True,
                "trajectory_path": compressed.trajectory_path,
                "details_path": compressed.details_path,
                "skill_ids_cited": compressed.skill_ids_cited,
                "message": f"Compressed to {compressed.trajectory_path}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ======================================================================= #
    # Internal Methods (not exposed to LLM)
    # ======================================================================= #

    @tool
    def get_skillbook_content(self, agent_name: str = "global") -> Dict[str, Any]:
        """
        Get formatted skillbook content for analysis.
        
        Call this to get all skills formatted as text, ready to pass to sub-agents
        like reflector or skill_manager for context.

        Args:
            agent_name: Agent name for scoped skills (default "global")

        Returns:
            {
                "success": True,
                "content": "<formatted skillbook text>",
                "skill_count": 42
            }

        Example:
            result = get_skillbook_content("my_agent")
            # Pass result["content"] to reflector/skill_manager
        """
        # Reload from file to get latest state
        self.skillbook.load()
        
        return {
            "success": True,
            "content": self.skillbook.as_prompt(agent_name),
            "skill_count": len(self.skillbook.skills()),
        }

    def as_prompt(self, agent_name: Optional[str] = None) -> str:
        """Generate skillbook prompt for LLM injection."""
        return self.skillbook.as_prompt(agent_name or "global")

    @tool(exclude=True)
    def get_prompt(self, agent_name: str = "global") -> Dict[str, Any]:
        """Get skillbook prompt for agent injection.
        
        Loads latest from file and generates prompt.
        Not exposed to LLM, only for internal use via Endpoint RPC.
        
        Args:
            agent_name: Agent name for scoped skills (default "global")
            
        Returns:
            Dict with success, prompt, summary, and stats
        """
        # Reload from file to get latest state
        self.skillbook.load()
        
        return {
            "success": True,
            "prompt": self.skillbook.as_prompt(agent_name),
            "summary": self.skillbook.summary_line(),
            "stats": self.skillbook.stats(),
        }


