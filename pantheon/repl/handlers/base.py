from rich.console import Console
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..core import Repl


class CommandHandler(ABC):
    """Base class for command handlers."""
    def __init__(self, console: Console, parent: "Repl"):
        self.console = console
        self.parent = parent

    @property
    def team(self):
        """Get team from parent (supports both _team and team)."""
        return getattr(self.parent, '_team', None) or getattr(self.parent, 'team', None)

    @abstractmethod
    def match_command(self, command: str) -> bool:
        """Match a command.

        Args:
            command (str): The command to match.

        Returns:
            bool: True if the command was matched, False otherwise.
        """
        pass

    @abstractmethod
    async def handle_command(self, command: str) -> str | None:
        """Handle a command.

        Args:
            command (str): The command to handle.

        Returns:
            str | None:
            A message for the agent to respond with.
        """
        pass
