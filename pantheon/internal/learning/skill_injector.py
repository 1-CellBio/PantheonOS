"""
Skill injection utilities for ACE.

Provides stateless functions to inject skills from Skillbook into Agents/Teams.
Called externally before team.run() to keep PantheonTeam pure.
"""

from typing import TYPE_CHECKING

from pantheon.utils.log import logger

if TYPE_CHECKING:
    from pantheon.team import PantheonTeam
    from .skillbook import Skillbook


def load_skill_prompt(skillbook: "Skillbook", agent_name: str = "global", reload: bool = False) -> str:
    """
    Load skill prompt for a specific agent from skillbook.
    
    Args:
        skillbook: Skillbook instance to load skills from
        agent_name: Agent name for scope filtering (default: "global")
        reload: If True, reload skillbook from file (default: False)
        
    Returns:
        Formatted skill prompt string
    """
    if reload:
        skillbook.load()  # Reload from file only when requested
    return skillbook.as_prompt(agent_name)


async def inject_skills_to_team(
    team: "PantheonTeam",
    skillbook: "Skillbook",
) -> int:
    """
    Inject skills into all agents in a team.
    
    This is a one-time operation, typically called after team creation
    and before team.run(). Skips agents that already have skills injected.
    
    Reloads skillbook from file to get latest skills (e.g., from learning pipeline).
    
    Args:
        team: PantheonTeam instance to inject skills into
        skillbook: Skillbook instance containing skills to inject
        
    Returns:
        Number of agents that received skill injection
    """
    # Reload skillbook to get latest skills (once per team creation)
    skillbook.load()
    
    injected_count = 0
    
    for agent in team.team_agents:
        # Skip if already injected (check for skill markers)
        if "📌 User Rules" in agent.instructions or "📚 Learned" in agent.instructions:
            continue
        
        prompt = load_skill_prompt(skillbook, agent.name)
        if prompt:
            agent.instructions += f"\n\n{prompt}"
            injected_count += 1
            logger.debug(f"Injected skills into agent: {agent.name}")
    
    if injected_count > 0:
        summary = skillbook.summary_line()
        logger.info(f"Skills injected into {injected_count} agents: {summary}")
    
    return injected_count

