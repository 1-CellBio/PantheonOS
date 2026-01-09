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
from .skill_injector import inject_skills_to_team, load_static_skills, load_dynamic_skills
from pantheon.toolsets.skillbook import SkillbookToolSet


def create_learning_resources(
    config: Optional[dict] = None,
) -> Tuple[Optional[Skillbook], Optional[LearningPipeline]]:
    """
    Factory function to create ACE resources.
    
    Reads configuration from settings, optionally merged with provided config.
    Returns (None, None) if both learning and injection are disabled.
    
    Config keys:
    - enable_learning: Controls trajectory learning (LearningPipeline creation)
    - enable_injection: Controls skill injection (Skillbook creation)
    
    Note: Injection is handled externally via inject_skills_to_team().
    If enable_injection=True, a Skillbook is returned for use with
    inject_skills_to_team(team, skillbook).
    
    Args:
        config: Override for ACE config (merged with settings.get_learning_config())
    
    Returns:
        Tuple of (Skillbook, LearningPipeline)
        - Skillbook: Created if either feature is enabled
        - LearningPipeline: Created only if enable_learning=True
    """
    from pantheon.settings import get_settings
    from pantheon.utils.log import logger
    
    settings = get_settings()
    # Start with defaults from settings
    _config = settings.get_learning_config().copy()
    
    # Merge provided config override if present
    if config:
        _config.update(config)
    
    # Determine feature flags from config
    _enable_learning = _config.get("enable_learning", False)
    _enable_injection = _config.get("enable_injection", _enable_learning)
    
    # If both disabled, return early
    if not _enable_learning and not _enable_injection:
        logger.info("Learning and injection both disabled")
        return None, None
    
    # Create skillbook (shared by both features)
    skillbook = Skillbook(
        skills_dir=settings.skills_dir,
        skillbook_path=_config["skillbook_path"],
        max_skills_per_section=_config["max_skills_per_section"],
        max_content_length=_config["max_content_length"],
        enable_agent_scope=_config.get("enable_agent_scope", False),
        auto_load=True,  # Automatically loads JSON + merges files
    )
    
    # Create learning pipeline (only if learning enabled)
    pipeline = None
    if _enable_learning:
        pipeline = LearningPipeline(
            skillbook=skillbook,
            reflector=Reflector(
                model=_config["learning_model"],
                learning_config=_config,  # Config passed to Reflector
            ),
            skill_manager=SkillManager(
                model=_config["learning_model"],
                learning_config=_config,  # Config passed to SkillManager
            ),
            learning_dir=_config["learning_dir"],
            cleanup_after_learning=_config.get("cleanup_after_learning", False),
            min_confidence_threshold=_config.get("min_confidence_threshold", 0.5),
            min_atomicity_score=_config.get("min_atomicity_score", 0.85),
            mode=_config.get("mode", "pipeline"),  # "pipeline" or "team"
            team_id=_config.get("team_id", "skill_learning_team"),
            # Config not needed here - already in Reflector and SkillManager
        )
    
    # Logging
    features = []
    if _enable_learning:
        features.append(f"learning({_config.get('mode', 'pipeline')})")
    if _enable_injection:
        features.append("injection")
    logger.info(f"ACE enabled [{', '.join(features)}]: {len(skillbook.skills())} skills loaded")
    
    return skillbook, pipeline


__all__ = [
    # Factory
    "create_learning_resources",
    # Skillbook
    "Skill",
    "Skillbook",
    "SkillbookToolSet",
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
    # Injection utilities
    "inject_skills_to_team",
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

