"""
ACE (Agentic Context Engineering) module for Pantheon.

This module provides a long-term memory system that allows agents to learn
from their interactions and improve over time.

Core Components:
- Skillbook: Structured knowledge store with skills organized by section
- SkillLoader: Loads and merges skills from multiple sources
- Reflector: Analyzes agent trajectories to extract learnings
- SkillManager: Decides what updates to apply to the Skillbook
- LearningPipeline: Async pipeline coordinating the learning process
"""

from typing import Optional, Tuple

from .pipeline import LearningPipeline, LearningInput, build_learning_input
from .reflector import (
    ExtractedLearning,
    Reflector,
    ReflectorOutput,
    SkillTag,
)
from .skill_loader import SkillLoader, load_skills_into_skillbook
from .skill_manager import SkillManager, SkillManagerOutput, UpdateOperation
from .skillbook import Skill, Skillbook


def create_learning_resources(
    enable: Optional[bool] = None,
    config: Optional[dict] = None,
) -> Tuple[Optional[Skillbook], Optional[LearningPipeline]]:
    """
    Factory function to create ACE resources.
    
    Reads configuration from settings if not provided.
    Returns (None, None) if ACE is disabled.
    
    Loading order:
    1. Load skillbook.json (base data + ratings)
    2. Scan skills/*.md files and merge (user skills have priority)
    3. Save updated skillbook.json
    
    Args:
        enable: Override for enable flag (None = use config)
        config: Override for ACE config (None = read from settings)
    
    Returns:
        Tuple of (Skillbook, LearningPipeline) or (None, None)
    """
    from pantheon.settings import get_settings
    from pantheon.utils.log import logger
    
    settings = get_settings()
    _config = config or settings.get_learning_config()
    _enable = enable if enable is not None else _config["enable"]
    
    if not _enable:
        logger.info("Learning disabled")
        return None, None
    
    # Create skillbook and load from JSON
    skillbook = Skillbook(
        max_skills_per_section=_config["max_skills_per_section"],
        max_content_length=_config["max_content_length"],
        enable_agent_scope=_config.get("enable_agent_scope", False),
    )
    skillbook.load(_config["skillbook_path"])
    
    # Load and merge skills from files
    skills_dir = settings.skills_dir
    if skills_dir.exists():
        loaded = load_skills_into_skillbook(skills_dir, skillbook)
        if loaded > 0:
            skillbook.save()  # Save merged skills
            logger.info(f"Merged {loaded} skills from {skills_dir}")
    
    # Create learning pipeline (with skills_dir for async reload)
    pipeline = LearningPipeline(
        skillbook=skillbook,
        reflector=Reflector(model=_config["learning_model"]),
        skill_manager=SkillManager(model=_config["learning_model"]),
        learning_dir=_config["learning_dir"],
        cleanup_after_learning=_config.get("cleanup_after_learning", False),
        skills_dir=skills_dir if skills_dir.exists() else None,
    )
    
    logger.info(f"Learning enabled: {len(skillbook.skills())} skills loaded")
    return skillbook, pipeline


__all__ = [
    # Factory
    "create_learning_resources",
    # Skillbook
    "Skill",
    "Skillbook",
    # SkillLoader
    "SkillLoader",
    "load_skills_into_skillbook",
    # Learning
    "LearningInput",
    "build_learning_input",
    # Reflector
    "Reflector",
    "ReflectorOutput",
    "SkillTag",
    "ExtractedLearning",
    # SkillManager
    "SkillManager",
    "SkillManagerOutput",
    "UpdateOperation",
    # Pipeline
    "LearningPipeline",
    # Prompt Constants
    "SKILLBOOK_USAGE_INSTRUCTIONS",
    "SKILLBOOK_HEADER",
    "USER_RULES_HEADER",
    "SKILL_LOADING_GUIDANCE",
]

# Import prompt constants from skillbook
from .skillbook import (
    SKILLBOOK_USAGE_INSTRUCTIONS,
    SKILLBOOK_HEADER,
    USER_RULES_HEADER,
    SKILL_LOADING_GUIDANCE,
)
