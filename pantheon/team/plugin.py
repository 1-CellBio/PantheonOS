"""
Team plugin system for extending PantheonTeam functionality.

Plugins provide a clean way to add optional features (learning, monitoring, etc.)
without coupling them directly to PantheonTeam.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pantheon.team.pantheon import PantheonTeam


class TeamPlugin(ABC):
    """
    Base class for PantheonTeam plugins.
    
    Plugins can hook into team lifecycle events to add functionality:
    - on_team_created: Initialize resources after team creation
    - on_run_start: Prepare before each run
    - on_run_end: Clean up or learn after each run
    
    Example:
        class MonitoringPlugin(TeamPlugin):
            async def on_run_start(self, team, user_input):
                self.start_time = time.time()
            
            async def on_run_end(self, team, result):
                duration = time.time() - self.start_time
                logger.info(f"Run took {duration}s")
    """
    
    @abstractmethod
    async def on_team_created(self, team: "PantheonTeam") -> None:
        """
        Called after team is created and agents are set up.
        
        Use this to:
        - Initialize plugin resources
        - Modify team configuration
        - Register context injectors
        
        Args:
            team: The PantheonTeam instance
        """
        pass
    
    async def on_run_start(self, team: "PantheonTeam", user_input: Any, context: dict) -> None:
        """
        Called before each run starts.
        
        Use this to:
        - Prepare resources for this run
        - Modify context (e.g., check cache, perform compression)
        - Log/monitor run start
        
        Args:
            team: The PantheonTeam instance
            user_input: User's input message (str or AgentInput)
            context: Run context dictionary containing:
                - memory: Memory instance
                - kwargs: Other run arguments
        """
        pass
    
    async def on_run_end(self, team: "PantheonTeam", result: dict) -> None:
        """
        Called after each run completes.
        
        Use this to:
        - Clean up resources
        - Learn from the run
        - Log/monitor run completion
        
        Args:
            team: The PantheonTeam instance
            result: Run result dictionary
        """
        pass
