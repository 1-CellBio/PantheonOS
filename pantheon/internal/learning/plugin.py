"""
Learning plugin for PantheonTeam.

Encapsulates all learning/skill injection functionality as a plugin,
decoupling it from PantheonTeam core logic.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict

from pantheon.team.plugin import TeamPlugin
from pantheon.utils.log import logger

if TYPE_CHECKING:
    from pantheon.team.pantheon import PantheonTeam
    from pantheon.internal.learning.skillbook import Skillbook
    from pantheon.internal.learning.pipeline import LearningPipeline


# Global singleton instance
_global_learning_plugin: "LearningPlugin | None" = None
_plugin_lock = asyncio.Lock()


async def get_global_learning_plugin(config: Dict[str, Any]) -> "LearningPlugin":
    """
    Get or create global learning plugin (singleton).
    
    The plugin is shared across all teams/chatrooms in the same process
    to ensure a single learning pipeline and avoid resource duplication.
    
    Args:
        config: Learning configuration dict
        
    Returns:
        Global LearningPlugin instance
    """
    global _global_learning_plugin
    
    async with _plugin_lock:
        if _global_learning_plugin is None:
            _global_learning_plugin = LearningPlugin(config)
            await _global_learning_plugin.initialize()
            logger.info("Global learning plugin created and initialized")
        
        return _global_learning_plugin


async def shutdown_global_learning_plugin():
    """Shutdown global learning plugin."""
    global _global_learning_plugin
    
    if _global_learning_plugin is not None:
        await _global_learning_plugin.cleanup()
        _global_learning_plugin = None
        logger.info("Global learning plugin shutdown")


class LearningPlugin(TeamPlugin):
    """
    Plugin that adds learning and skill injection capabilities to PantheonTeam.
    
    This plugin is designed to be used as a singleton (via get_global_learning_plugin)
    to share learning resources across all teams in the same process.
    
    Features:
    - Static skill injection (modifies agent instructions)
    - Dynamic skill injection (via context injectors)
    - Trajectory learning (extracts skills from runs)
    
    Configuration:
        enable_injection: Enable static skill injection (default: False)
        enable_dynamic_injection: Enable dynamic skill injection (default: False)
        enable_learning: Enable trajectory learning (default: False)
        dynamic_injection_top_k: Number of skills for dynamic injection (default: 5)
    
    Example:
        # Get global singleton
        plugin = await get_global_learning_plugin(config)
        team = PantheonTeam(agents=agents, plugins=[plugin])
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize learning plugin.
        
        Args:
            config: Learning configuration dict (from settings.get_learning_config())
        """
        self.config = config
        self.skillbook: "Skillbook | None" = None
        self.learning_pipeline: "LearningPipeline | None" = None
        self.skill_injector = None
        self._initialized = False
        
        # Extract config flags
        self.enable_injection = config.get("enable_injection", False)
        self.enable_dynamic_injection = config.get("enable_dynamic_injection", False)
        self.enable_learning = config.get("enable_learning", False)
        self.dynamic_injection_top_k = config.get("dynamic_injection_top_k", 10)
    
    async def initialize(self):
        """
        Initialize plugin resources (called once on first use).
        
        Creates skillbook and learning pipeline. Should be called before
        using the plugin with any team.
        """
        if self._initialized:
            return
        
        # Skip if no features enabled
        if not (self.enable_injection or self.enable_dynamic_injection or self.enable_learning):
            self._initialized = True
            return
        
        # Create learning resources
        await self._initialize_learning_resources()
        
        self._initialized = True
        logger.info("LearningPlugin initialized")
    
    async def cleanup(self):
        """
        Cleanup plugin resources (called on shutdown).
        
        Stops learning pipeline and releases resources.
        """
        if self.learning_pipeline:
            try:
                await self.learning_pipeline.stop()
                logger.info("Learning pipeline stopped")
            except Exception as e:
                logger.error(f"Error stopping learning pipeline: {e}")
        
        self._initialized = False
        logger.info("LearningPlugin cleaned up")
    
    async def on_team_created(self, team: "PantheonTeam") -> None:
        """
        Initialize learning resources and inject skills.
        
        Steps:
        1. Ensure plugin is initialized
        2. Perform static skill injection (if enabled)
        3. Create and register dynamic skill injector (if enabled)
        """
        if not self._initialized:
            raise RuntimeError(
                "LearningPlugin not initialized. Call initialize() first or use "
                "get_global_learning_plugin() to get initialized singleton."
            )
        
        # Static skill injection
        if self.skillbook and self.enable_injection:
            await self._inject_static_skills(team)
        
        # Dynamic skill injection
        if self.skillbook and self.enable_dynamic_injection:
            await self._register_dynamic_injector(team)
    
    async def on_run_end(self, team: "PantheonTeam", result: dict) -> None:
        """
        Trigger learning after run completes.
        
        Args:
            team: The PantheonTeam instance
            result: Run result with agent_name, messages, chat_id, question (optional)
        """
        if not self.learning_pipeline or not self.enable_learning:
            return
        
        agent_name = result.get("agent_name")
        messages = result.get("messages", [])
        chat_id = result.get("chat_id", "")
        question = result.get("question")  # Optional, for sub-agents
        
        if not agent_name or not messages:
            return
        
        # Submit learning
        self._submit_learning(
            agent_name=agent_name,
            messages=messages,
            chat_id=chat_id,
            parent_question=question,
        )
    
    def _submit_learning(
        self,
        agent_name: str,
        messages: list,
        chat_id: str = "",
        parent_question: str | None = None,
    ) -> None:
        """Submit learning data to learning pipeline.
        
        Args:
            agent_name: Name of the agent that produced the trajectory
            messages: List of messages from the conversation
            chat_id: Original chat/memory ID for grouping learning files
            parent_question: For sub_agent, the delegation instruction
        """
        import uuid
        from pantheon.internal.learning.pipeline import build_learning_input
        from pantheon.utils.log import logger
        
        turn_id = str(uuid.uuid4())
        learning_input = build_learning_input(
            turn_id=turn_id,
            agent_name=agent_name,
            messages=messages,
            learning_dir=self.config["learning_dir"],
            chat_id=chat_id,
        )
        
        # For sub_agent, use delegation instruction as question
        if parent_question:
            learning_input.question = parent_question
        
        if self.learning_pipeline:
            self.learning_pipeline.submit(learning_input)
            logger.debug(
                f"Submitted learning for {agent_name}, turn_id={turn_id}, "
                f"chat_id={chat_id[:8] if chat_id else 'N/A'}"
            )
    
    
    async def _initialize_learning_resources(self) -> None:
        """Create skillbook and learning pipeline."""
        from pantheon.internal.learning import create_learning_resources
        
        self.skillbook, self.learning_pipeline = create_learning_resources(
            config=self.config
        )
        
        # Start learning pipeline if created
        if self.learning_pipeline and self.enable_learning:
            await self.learning_pipeline.start()
            logger.info("Learning pipeline started")
        
        if self.skillbook:
            logger.info(f"Learning plugin initialized: {len(self.skillbook.skills())} skills loaded")
    
    async def _inject_static_skills(self, team: "PantheonTeam") -> None:
        """Inject static skills into agent instructions."""
        from pantheon.internal.learning import inject_skills_to_team
        
        await inject_skills_to_team(team, self.skillbook, config=self.config)
        logger.info("Static skill injection completed")
    
    async def _register_dynamic_injector(self, team: "PantheonTeam") -> None:
        """Create and register dynamic skill injector."""
        from pantheon.internal.injector import SkillInjector
        from pantheon.toolsets.skillbook import SkillbookToolSet
        
        # Create SkillInjector only once (reused across multiple calls/agents)
        if self.skill_injector is None:
            # Create SkillbookToolSet wrapper
            skillbook_toolset = SkillbookToolSet(skillbook=self.skillbook)
            
            self.skill_injector = SkillInjector(
                skillbook_toolset=skillbook_toolset,
                top_k=self.dynamic_injection_top_k,
            )
        
        # Register with all agents (avoiding duplicates)
        for agent in team.team_agents:
            if self.skill_injector not in agent.context_injectors:
                agent.context_injectors.append(self.skill_injector)
        
        logger.info(f"Dynamic skill injection enabled (top_k={self.dynamic_injection_top_k})")
    
    async def initialize_learning_team(self, endpoint_service: Any) -> None:
        """
        Initialize the learning team (for team-based learning).
        
        This is required when the pipeline is configured in 'team' mode.
        Since the plugin singleton is created before we have an endpoint connection,
        we must initialize the team later when an endpoint is available (from ChatRoom).
        
        Args:
            endpoint_service: Active endpoint service connection
        """
        if self.learning_pipeline:
            await self.learning_pipeline.initialize_team(endpoint_service)


    @property
    def learning_pipeline_instance(self) -> "LearningPipeline | None":
        """Get learning pipeline instance (for team to access)."""
        return self.learning_pipeline
