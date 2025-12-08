"""
REPL entry point - Launch Pantheon REPL from command line.

Usage:
    python -m pantheon.repl                              # Start with defaults
    python -m pantheon.repl --template team.md           # Use specific template
    python -m pantheon.repl --memory-dir ./chats         # Custom directory
    python -m pantheon.repl start --template team.md     # Explicit start command
"""

import asyncio
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def start(
    template: str = None,
    memory_dir: str = ".pantheon",
    workspace: str = None,
    chat_id: str = None,
    log_level: str = "WARNING",
):
    """Start Pantheon REPL.

    Args:
        template: Path to team template markdown file.
        memory_dir: Directory for chat persistence (default: .pantheon).
        workspace: Workspace directory for Endpoint.
        chat_id: Resume specific chat by ID.
        log_level: Log level (DEBUG, INFO, WARNING, ERROR).
    """
    asyncio.run(
        _start_async(
            template=template,
            memory_dir=memory_dir,
            workspace=workspace,
            chat_id=chat_id,
            log_level=log_level,
        )
    )


async def _start_async(
    template: str = None,
    memory_dir: str = ".pantheon",
    workspace: str = None,
    chat_id: str = None,
    log_level: str = "WARNING",
):
    """Async implementation of start."""
    # Setup logging
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=log_level)

    from .core import Repl
    from ..chatroom import ChatRoom

    if template:
        # Load team from template file
        template_path = Path(template)
        if not template_path.exists():
            print(f"Error: Template file not found: {template_path}")
            sys.exit(1)

        # Create ChatRoom
        chatroom = ChatRoom(
            endpoint=None,
            memory_dir=memory_dir,
            workspace_path=workspace,
            enable_nats_streaming=False,
        )

        # Setup ChatRoom (including auto-created Endpoint)
        await chatroom.run_setup()

        # Parse template and create team
        from ..factory import get_template_manager

        template_manager = get_template_manager()
        template_content = template_path.read_text(encoding="utf-8")
        team_config = template_manager.parse_template_content(template_content)

        # Create chat with template
        result = await chatroom.create_chat("repl-session")
        chat_id = result["chat_id"]
        await chatroom.setup_team_for_chat(chat_id, team_config)

        repl = Repl(
            chatroom=chatroom,
            chat_id=chat_id,
        )
    else:
        # Default: auto-create everything
        repl = Repl(
            memory_dir=memory_dir,
            chat_id=chat_id,
        )

    await repl.run(disable_logging=(log_level != "DEBUG"))


def main():
    """Entry point."""
    fire.Fire(
        {
            "start": start,
        },
        name="pantheon-repl",
    )


if __name__ == "__main__":
    # Support two call styles:
    # 1. python -m pantheon.repl start --template xxx
    # 2. python -m pantheon.repl --template xxx (implicit start)
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1].startswith("-")):
        sys.argv.insert(1, "start")
    fire.Fire(
        {
            "start": start,
        },
        name="pantheon-repl",
    )
