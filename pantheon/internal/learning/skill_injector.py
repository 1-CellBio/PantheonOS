"""
Skill injection utilities for ACE.

Provides stateless functions to inject skills from Skillbook into Agents/Teams.
Called externally before team.run() to keep PantheonTeam pure.
"""

from typing import TYPE_CHECKING, List, Dict, Any

from pantheon.utils.log import logger

if TYPE_CHECKING:
    from pantheon.team import PantheonTeam
    from .skillbook import Skillbook


async def inject_skills_to_team(
    team: "PantheonTeam",
    skillbook: "Skillbook",
    config: dict | None = None,
) -> int:
    """
    Inject static skills into all agents in a team.
    
    Static skills filtering:
    - If config["static_injection_sections"] is set: use section-based filtering
    - Otherwise: default rules (user_rules + type="user" strategies)
    
    This is a one-time operation, typically called after team creation
    and before team.run(). Skips agents that already have skills injected.
    
    Reloads skillbook from file to get latest skills (e.g., from learning pipeline).
    
    Args:
        team: PantheonTeam instance to inject skills into
        skillbook: Skillbook instance containing skills to inject
        config: Learning config dict (optional, for section filtering)
        
    Returns:
        Number of agents that received skill injection
    """
    # Reload skillbook to get latest skills (once per team creation)
    skillbook.load()
    
    injected_count = 0
    logger.info(f"Injecting static skills into team: {team}")
    for agent in team.team_agents:
        # Skip if already injected (check flag for optimization)
        # Fallback to string check for backward compatibility with older agent instances (if any)
        if getattr(agent, "_static_skills_injected", False):
            continue
            
        # Backward compatibility check (in case flag wasn't set but content exists)
        if "📌 User Rules" in agent.instructions or "📚 Learned" in agent.instructions:
            # Mark as injected and skip
            agent._static_skills_injected = True
            continue
        
        # Load static skills with config-based filtering
        prompt = load_static_skills(skillbook, agent.name, config=config)
        if prompt:
            agent.instructions += f"\n\n{prompt}"
            injected_count += 1
            agent._static_skills_injected = True
            logger.debug(f"Injected static skills into agent: {agent.name}")
    
    if injected_count > 0:
        summary = skillbook.summary_line()
        logger.info(f"Static skills injected into {injected_count} agents: {summary}")
    
    return injected_count


# ===========================================================================
# Static Skill Injection (User Rules + User-Defined Strategies)
# ===========================================================================


def load_static_skills(
    skillbook: "Skillbook",
    agent_name: str = "global",
    config: dict | None = None,
) -> str:
    """
    Load static skill prompt for injection into agent instructions.
    
    Filtering logic:
    1. If config["static_injection_sections"] is provided:
       - ["*"]: Include all sections
       - [section1, section2, ...]: Include only specified sections
    2. If not provided: Use default rules (user_rules + type="user" strategies)
    
    These are EXCLUDED from dynamic injection to avoid duplication.
    
    Args:
        skillbook: Skillbook instance to load skills from
        agent_name: Agent name for scope filtering (default: "global")
        config: Learning config dict (optional, for section filtering)
        
    Returns:
        Formatted skill prompt string for static injection
    """
    from pantheon.internal.learning.skillbook import _format_skillbook_for_injection
    
    # Get all skills for agent
    skills = skillbook.get_skills_for_agent(agent_name)
    if not skills:
        return ""
    
    # Get section filter from config
    section_filter = (config or {}).get("static_injection_sections")
    
    # Apply filtering logic
    if section_filter is not None:
        # Custom section-based filtering
        static_skills = _filter_by_sections(skills, section_filter)
    else:
        # Default rule-based filtering (backward compatibility)
        static_skills = _filter_static_skills(skills)
    
    if not static_skills:
        return ""
    
    # Separate user_rules from other static skills
    user_rules = [s for s in static_skills if s.section == "user_rules"]
    user_strategies = [s for s in static_skills if s.section != "user_rules"]
    
    # Format user_rules using NEW content-first logic
    user_rules_text = ""
    if user_rules:
        user_rules_text = "\n".join(
            f"[{s.id}] {skillbook._format_skill_for_display(s)}" for s in user_rules
        )
    
    # Format user-defined strategies by section using NEW method
    strategies_text = ""
    if user_strategies:
        strategies_text = skillbook._format_skills_by_section(user_strategies)
    
    return _format_skillbook_for_injection(user_rules_text, strategies_text)


def _filter_static_skills(skills: List["Skill"]) -> List["Skill"]:
    """Filter for static skills (user_rules, user-defined strategies)."""
    return [s for s in skills if _is_static_skill(s)]


def _filter_by_sections(skills: List["Skill"], section_filter: List[str]) -> List["Skill"]:
    """
    Filter skills by section names.
    
    Args:
        skills: List of skills to filter
        section_filter: List of section names, or ["*"] for all sections
        
    Returns:
        Filtered skills
    """
    # "*" means include all sections
    if "*" in section_filter:
        return skills
    
    # Filter by section names
    return [s for s in skills if s.section in section_filter]


def _is_static_skill(skill: "Skill") -> bool:
    """
    Check if skill is static (user_rules or user-defined strategies).
    
    Static skills are:
    - user_rules section (any type)
    - strategies section AND type="user" (or type=None for backward compatibility)
    
    All other skills (including type="user" in other sections) are dynamic.
    """
    if skill.section == "user_rules":
        return True
    # Only strategies with type="user" are static
    if skill.section == "strategies" and (skill.type == "user" or skill.type is None):
        return True
    
    return False


# ===========================================================================
# Dynamic Skill Injection (System-Learned Skills)
# ===========================================================================


async def load_dynamic_skills(
    skillbook_toolset: "SkillbookToolSet",
    user_input: str,
    context: dict,
    top_k: int = 5,
) -> str:
    """
    Load dynamic skill prompt based on user input (context-relevant skills).
    
    Uses LLM-based semantic search to find skills relevant to the current task,
    filtering out skills that should only be in static injection.
    
    Dynamic skills exclude:
    1. user_rules (always in static injection)
    2. User-defined strategies (always in static injection)
    
    Args:
        skillbook_toolset: SkillbookToolSet instance for skill retrieval
        user_input: User's current input/query
        context: Context variables containing:
            - agent_name: Agent name
            - _call_agent: Agent._call_agent method (for LLM calls)
        top_k: Maximum number of skills to return (default: 5)
        
    Returns:
        Formatted skill prompt string wrapped in <EPHEMERAL_SKILLS> tags,
        or empty string if no relevant skills found
    """
    from pantheon.utils.log import logger
    from pantheon.internal.learning.skillbook import Skill
    
    logger.info(f"load_dynamic_skills: query='{user_input}', top_k={top_k}")
    logger.debug(f"load_dynamic_skills: context keys={list(context.keys())}")
    logger.debug(f"load_dynamic_skills: has _call_agent={('_call_agent' in context)}")
    
    # Call list_skills with context_variables (standard toolset calling pattern)
    # The @tool decorator will automatically handle context_variables
    logger.debug("load_dynamic_skills: Calling list_skills with semantic=True")
    result = await skillbook_toolset.list_skills(
        query=user_input,
        semantic=True,
        top_k=top_k * 2,  # Get more candidates, filter to top_k
        include_full_content=True,
        context_variables=context,  # Pass context as context_variables parameter
    )
    logger.info(f"load_dynamic_skills: list_skills returned {len(result.get('skills', []))} skills")

    if not result["success"] or not result["skills"]:
        return ""

    # Filter out skills already in static injection
    dynamic_skills_dicts = _filter_dynamic_skills(result["skills"])[:top_k]

    if not dynamic_skills_dicts:
        return ""


    # Convert skill dicts to Skill objects
    # Note: list_skills returns dicts with 'stats' field (string), but Skill uses helpful/harmful/neutral
    skill_objects = []
    for skill_dict in dynamic_skills_dicts:
        # Remove 'stats' field if present (it's a computed field, not part of Skill dataclass)
        skill_dict_clean = {k: v for k, v in skill_dict.items() if k != 'stats'}
        skill_objects.append(Skill(**skill_dict_clean))

    # Format skills by section using NEW content-first method
    skills_text = skillbook_toolset.skillbook._format_skills_by_section(skill_objects)

    # Use prompt constant
    return DYNAMIC_SKILLS_PROMPT.format(skills_text=skills_text)


def _filter_dynamic_skills(skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out core static skills (user_rules, user strategies)."""
    return [s for s in skills if not _is_core_static_skill(s)]


def _is_core_static_skill(skill: Any) -> bool:
    """
    Check if skill is core static (user_rules or user-defined strategies).
    
    Static skills are:
    - user_rules section (any type)
    - strategies section AND type="user" (or type=None for backward compatibility)
    
    All other skills (including type="user" in other sections) are dynamic.
    
    This function works with both Skill objects and dicts.
    """
    section = skill.section if hasattr(skill, 'section') else skill.get("section")
    skill_type = skill.type if hasattr(skill, 'type') else skill.get("type")
    
    if section == "user_rules":
        return True
    # Only strategies with type="user" are static
    if section == "strategies" and (skill_type == "user" or skill_type is None):
        return True
    
    return False


# Dynamic skills prompt template
DYNAMIC_SKILLS_PROMPT = """<EPHEMERAL_SKILLS>
## 📚 Task-Relevant Skills (Dynamically Retrieved)

The following skills are relevant to your current task:

{skills_text}

</EPHEMERAL_SKILLS>"""
