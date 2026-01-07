"""
Context Injection module for Pantheon.

Provides injectors that dynamically augment user messages with relevant context
(skills, tools, knowledge) via the _llm_content field.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

from pantheon.utils.log import logger

if TYPE_CHECKING:
    from pantheon.toolsets.skillbook import SkillbookToolSet


# ===========================================================================
# Base Class
# ===========================================================================


class ContextInjector(ABC):
    """Base class for context injectors.
    
    Injectors augment user messages by appending relevant context to the
    _llm_content field. This allows dynamic, task-specific knowledge injection
    without creating additional messages in the conversation history.
    """

    @abstractmethod
    async def inject(self, user_input: str, context: dict) -> str:
        """Generate content to append to _llm_content.
        
        Args:
            user_input: Original user message content
            context: Context variables, including:
                - agent_name: Name of the agent
                - _call_agent: Agent._call_agent method for LLM calls
        
        Returns:
            String to append to _llm_content, or empty string to skip injection
        """
        pass


# ===========================================================================
# Skill Injector
# ===========================================================================


class SkillInjector(ContextInjector):
    """Inject relevant skills based on user message.
    
    Uses LLM-based semantic search to find skills relevant to the current task,
    filtering out skills that are already in the initial (static) injection.
    """

    def __init__(self, skillbook_toolset: "SkillbookToolSet", top_k: int = 10):
        """Initialize SkillInjector.
        
        Args:
            skillbook_toolset: SkillbookToolSet instance for skill retrieval
            top_k: Maximum number of skills to inject
        """
        self.skillbook = skillbook_toolset
        self.top_k = top_k

    async def inject(self, user_input: str, context: dict) -> str:
        """Inject relevant skills.
        
        Args:
            user_input: User's original input
            context: Context variables containing:
                - agent_name: Agent name
                - _call_agent: Agent._call_agent method (for LLM calls)
        
        Returns:
            Formatted skills block or empty string
        """
        from pantheon.internal.learning.skill_injector import load_dynamic_skills
        
        return await load_dynamic_skills(
            skillbook_toolset=self.skillbook,
            user_input=user_input,
            context=context,
            top_k=self.top_k,
        )


# ===========================================================================
# Tool Injector (Future)
# ===========================================================================


class ToolInjector(ContextInjector):
    """Inject relevant tools (future implementation)."""

    async def inject(self, user_input: str, context: dict) -> str:
        # Future: Use similar mechanism to list_skills for tool retrieval
        return ""


# ===========================================================================
# Knowledge Injector (Future)
# ===========================================================================


class KnowledgeInjector(ContextInjector):
    """Inject relevant knowledge/documents (future RAG implementation)."""

    async def inject(self, user_input: str, context: dict) -> str:
        # Future: RAG retrieval of relevant document chunks
        return ""
