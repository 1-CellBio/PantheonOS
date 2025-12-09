"""REPL - Command line interface for Pantheon agents, based on ChatRoom."""

import asyncio
import sys
import time
import signal
import threading
from datetime import datetime
from pathlib import Path

from rich.text import Text
from rich.live import Live
from rich.markdown import Markdown

# Simple readline support for history
try:
    import readline
    import atexit

    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False

from ..agent import Agent
from ..team import Team
from ..team.pantheon import PantheonTeam
from ..chatroom import ChatRoom
from ..constant import CLI_HISTORY_FILE
from .ui import ReplUI
from .handlers.base import CommandHandler
from .handlers.template_handler import TemplateHandler, load_template
from .handlers.builtin.bash import BashCommandHandler


class Repl(ReplUI):
    """REPL for agent or team interaction, based on ChatRoom.

    Supports multiple initialization modes:
    - agent: Pass an Agent or Team directly (legacy mode, creates embedded ChatRoom)
    - chatroom: Pass an existing ChatRoom instance
    - endpoint: Pass an Endpoint instance (creates ChatRoom with it)
    - None: Auto-create ChatRoom with embedded Endpoint

    Args:
        agent: An Agent or Team instance (legacy mode).
        chatroom: An existing ChatRoom instance.
        endpoint: An Endpoint instance to create ChatRoom with.
        memory_dir: Directory for chat persistence.
        chat_id: Specific chat ID to use (creates new if None).
    """

    def __init__(
        self,
        agent: Agent | Team | None = None,
        chatroom: ChatRoom | None = None,
        endpoint: "Endpoint | None" = None,
        memory_dir: str = ".pantheon",
        chat_id: str | None = None,
    ):
        super().__init__()  # init UI

        # Determine ChatRoom source
        if chatroom is not None:
            # Mode 1: Use existing ChatRoom
            self._chatroom = chatroom
        elif agent is not None:
            # Mode 2: Legacy - create ChatRoom from Agent/Team
            self._chatroom = self._create_chatroom_from_agent(agent, memory_dir)
        elif endpoint is not None:
            # Mode 3: Create ChatRoom with provided Endpoint
            self._chatroom = ChatRoom(
                endpoint=endpoint,
                memory_dir=memory_dir,
                enable_nats_streaming=False,
            )
        else:
            # Mode 4: Auto-create everything
            self._chatroom = ChatRoom(
                endpoint=None,  # Auto-create Endpoint
                memory_dir=memory_dir,
                enable_nats_streaming=False,
            )

        # Current chat session
        self._chat_id = chat_id

        # Reference to team for UI display (will be set after first chat)
        self._team: PantheonTeam | None = None
        if agent is not None:
            if isinstance(agent, Team):
                self._team = agent
            else:
                self._team = PantheonTeam([agent])

        # Set multi-agent mode flag (for UI display)
        self._is_multi_agent = (
            self._team is not None and len(self._team.agents) > 1
        )

        self.current_task = None
        self.tool_calls_active = False
        self.session_start = datetime.now()
        self.message_count = 0

        # Token statistics
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.current_input_tokens = 0
        self.current_output_tokens = 0

        # Processing status tracking
        self._current_live_display = None
        self._tools_executing = False
        self._current_agent_task = None
        self._current_tool_name = None

        # Setup history file
        self.history_file = Path(CLI_HISTORY_FILE)
        if not self.history_file.exists():
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            self.history_file.touch()
        self.command_history = []
        self.history_index = -1

        # Setup input system
        self._setup_input_system()
        self._load_history()

        # Setup signal handlers for better interrupt handling
        self._setup_signal_handlers()

        # Command handlers
        self.handlers: list[CommandHandler] = [
            BashCommandHandler(self.console, self),
        ]

    def _create_chatroom_from_agent(
        self, agent: Agent | Team, memory_dir: str
    ) -> ChatRoom:
        """Create ChatRoom from Agent/Team (legacy compatibility)."""
        # Wrap single Agent in PantheonTeam
        if isinstance(agent, Team):
            team = agent
        else:
            team = PantheonTeam([agent])

        # Create ChatRoom with default_team (bypasses template system)
        return ChatRoom(
            endpoint=None,  # Auto-create Endpoint
            memory_dir=memory_dir,
            enable_nats_streaming=False,
            default_team=team,
        )

    def register_handler(self, handler: CommandHandler | str | Path):
        """Register a handler for processing commands."""
        if isinstance(handler, CommandHandler):
            self.handlers.append(handler)
        elif isinstance(handler, (str, Path)):
            p = Path(handler)
            if p.exists():
                template = load_template(p)
                template_handler = TemplateHandler(self.console, self, template)
                self.handlers.append(template_handler)
            else:
                self.console.print(f"[red]Template file not found: {p}[/red]")

    def _setup_signal_handlers(self):
        """Setup signal handlers for better interrupt management."""
        self._interrupt_count = 0
        self._last_interrupt_time = 0

        def signal_handler(signum, frame):
            current_time = time.time()

            if current_time - self._last_interrupt_time < 2.0:
                self._interrupt_count += 1
            else:
                self._interrupt_count = 1

            self._last_interrupt_time = current_time

            if self._interrupt_count == 1:
                self.console.print(
                    "\n[yellow]Operation interrupted - press Ctrl+C again within 2 seconds to force exit[/yellow]"
                )
                if (
                    hasattr(self, "_current_agent_task")
                    and self._current_agent_task
                    and not self._current_agent_task.done()
                ):
                    try:
                        self._current_agent_task.cancel()
                    except Exception:
                        pass
            elif self._interrupt_count >= 2:
                self.console.print("\n[red]Force exit requested[/red]")
                sys.exit(1)

        if hasattr(signal, "SIGINT"):
            signal.signal(signal.SIGINT, signal_handler)

    def _setup_input_system(self):
        """Setup simple input system with readline history."""
        if READLINE_AVAILABLE:
            if self.history_file.exists():
                readline.read_history_file(str(self.history_file))
            atexit.register(readline.write_history_file, str(self.history_file))
            readline.set_history_length(1000)
            readline.parse_and_bind("tab: complete")
            readline.parse_and_bind("set completion-ignore-case on")
            readline.set_startup_hook(None)
            readline.set_pre_input_hook(None)
            readline.parse_and_bind('"\\e[A": previous-history')
            readline.parse_and_bind('"\\e[B": next-history')

    def _load_history(self):
        """Load command history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.command_history = [
                        line.strip() for line in f.readlines()[-100:]
                    ]
            except Exception:
                self.command_history = []

    def _save_history(self):
        """Save command history to file."""
        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                if self.command_history:
                    f.write(self.command_history[-1] + "\n")
        except Exception:
            pass

    def _add_to_history(self, command: str):
        """Add command to history."""
        command = command.strip()
        if command and (
            not self.command_history or self.command_history[-1] != command
        ):
            self.command_history.append(command)
            self._save_history()
            self.history_index = len(self.command_history)

    def ask_user_input(self) -> str:
        """Get user input with multi-line support and readline history."""
        try:
            self.console.print(
                "[dim]Enter your message (press Enter twice to finish)[/dim]"
            )
            lines = []
            while True:
                prompt_text = "... " if lines else ">   "

                if READLINE_AVAILABLE:
                    line = input(prompt_text)
                else:
                    self.console.print(
                        f"[bright_blue]{prompt_text}[/bright_blue]", end=" "
                    )
                    line = input()

                if line.strip() == "":
                    break

                lines.append(line)

            return "\n".join(lines).strip()

        except KeyboardInterrupt:
            self.console.print("\n[dim]Ctrl+C pressed - operation cancelled[/dim]")
            return ""
        except EOFError:
            raise

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using rough approximation."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _format_token_count(self, count: int) -> str:
        """Format token count with appropriate units."""
        if count >= 1000000:
            return f"{count/1000000:.1f}M"
        elif count >= 10000:
            return f"{count/1000:.1f}K"
        elif count >= 1000:
            return f"{count:,}"
        else:
            return str(count)

    def _update_token_stats(self, input_tokens: int, output_tokens: int):
        """Update token statistics."""
        self.current_input_tokens = input_tokens
        self.current_output_tokens = output_tokens
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    async def _setup(self):
        """Initialize REPL and ChatRoom."""
        # Start ChatRoom setup (including auto-created Endpoint)
        await self._chatroom.run_setup()

        # Create or get chat session
        if self._chat_id is None:
            result = await self._chatroom.create_chat("repl-session")
            self._chat_id = result["chat_id"]

        # Get team reference for UI display
        if self._team is None:
            self._team = await self._chatroom.get_team_for_chat(self._chat_id)
            self._is_multi_agent = len(self._team.agents) > 1

    async def run(self, message: str | dict | None = None, disable_logging: bool = True):
        """Main REPL loop."""
        if disable_logging:
            from ..utils.log import disable_all
            disable_all()

        # Initialize
        await self._setup()

        # Print greeting
        await self.print_greeting()

        # Set up connection between UI and token tracking
        self._parent_repl = self

        # NOTE: print_message() is disabled for ChatRoom-based REPL
        # In ChatRoom mode, streaming is handled via process_chunk/process_step_message callbacks
        # The events_queue approach is only for legacy direct agent/team mode (not used here)
        print_task = None

        # Handle initial message if provided
        current_message = message
        if current_message is not None:
            self._add_to_history(current_message)

        # Main message processing loop
        while True:
            if current_message is None:
                try:
                    current_message = self.ask_user_input()
                    if not current_message.strip():
                        current_message = None  # Reset to request new input
                        continue
                    self._add_to_history(current_message)
                except (KeyboardInterrupt, EOFError):
                    self.console.print("\n[dim]Session interrupted[/dim]")
                    self._print_session_summary()
                    break

            # Handle commands
            cmd = current_message.strip()
            cmd_lower = cmd.lower()

            # Exit commands
            if cmd_lower in ["exit", "quit", "q", "/exit", "/quit", "/q"]:
                self._print_session_summary()
                break

            # Help command
            elif cmd_lower in ["help", "/help"]:
                self._print_help()
                current_message = None
                continue

            # Status command
            elif cmd_lower in ["status", "/status"]:
                self._print_status()
                current_message = None
                continue

            # Clear command
            elif cmd_lower in ["clear", "/clear"]:
                await self._handle_clear()
                current_message = None
                continue

            # History command
            elif cmd_lower in ["history", "/history"]:
                self._print_history()
                current_message = None
                continue

            # Tokens command
            elif cmd_lower in ["tokens", "/tokens"]:
                self._print_token_analysis()
                current_message = None
                continue

            # Save command
            elif cmd_lower == "/save" or cmd_lower.startswith("/save "):
                self._handle_save_command(cmd)
                current_message = None
                continue

            # Load command
            elif cmd_lower.startswith("/load "):
                self._handle_load_command(cmd)
                current_message = None
                continue

            # New chat command
            elif cmd_lower in ["/new", "/new-chat"]:
                await self._handle_new_chat()
                current_message = None
                continue

            # List chats command
            elif cmd_lower in ["/list", "/chats"]:
                await self._handle_list_chats()
                current_message = None
                continue

            # Switch chat command
            elif cmd_lower.startswith("/switch "):
                chat_id = cmd[8:].strip()
                await self._handle_switch_chat(chat_id)
                current_message = None
                continue

            # Agents command
            elif cmd_lower in ["/agents", "/team"]:
                await self._handle_show_agents()
                current_message = None
                continue

            # Custom command handlers
            continue_flag = False
            for handler in self.handlers:
                if handler.match_command(cmd):
                    current_message = await handler.handle_command(cmd)
                    if current_message is not None:
                        continue_flag = False
                    else:
                        continue_flag = True
                    break
            if continue_flag:
                continue

            # Process with ChatRoom
            await self._process_message(current_message)
            current_message = None

        if print_task:
            print_task.cancel()

    async def _process_message(self, message: str):
        """Process a message through ChatRoom."""
        start_time = time.time()

        # Estimate input tokens
        input_tokens = self._estimate_tokens(message)
        output_tokens = 0

        # Content buffers for streaming
        content_buffer = []
        tool_calls_content_buffer = []
        estimated_output_tokens = 0

        def process_chunk(chunk: dict):
            nonlocal estimated_output_tokens
            content = chunk.get("content")
            tool_calls = chunk.get("tool_calls")
            if content is None and tool_calls is None:
                return
            if tool_calls is not None:
                for tool_call in tool_calls:
                    if "function" in tool_call:
                        if "arguments" in tool_call["function"]:
                            t_content = (content or "") + tool_call["function"][
                                "arguments"
                            ]
                            tool_calls_content_buffer.append(t_content)
            if content is not None:
                content_buffer.append(content)
            estimated_output_tokens = self._estimate_tokens(
                "".join(content_buffer + tool_calls_content_buffer)
            )

        # Animation frames - try fancy Unicode, fallback to ASCII on Windows
        def get_animation_frames():
            # Braille spinner (commonly supported)
            fancy_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            ascii_frames = ["-", "\\", "|", "/"]

            # Test if console can handle Unicode
            try:
                import sys
                # Try to encode a test character
                test_char = fancy_frames[0]
                if sys.stdout.encoding:
                    test_char.encode(sys.stdout.encoding)
                return fancy_frames
            except (UnicodeEncodeError, LookupError):
                return ascii_frames

        animation_frames = get_animation_frames()

        # Separator - fancy or ASCII
        def get_separator():
            try:
                import sys
                sep = "•"
                if sys.stdout.encoding:
                    sep.encode(sys.stdout.encoding)
                return sep
            except (UnicodeEncodeError, LookupError):
                return "|"

        sep = get_separator()

        processing_live = Live(console=self.console, refresh_per_second=8, transient=True)
        processing_live.start()

        try:
            # Wave effect brightness levels (grey scale gradient)
            wave_colors = ["grey30", "grey42", "grey54", "grey66", "grey78", "grey89", "white", "grey89", "grey78", "grey66", "grey54", "grey42"]

            def create_wave_text(text: str, offset: int) -> str:
                """Create text with wave brightness effect."""
                result = []
                for i, char in enumerate(text):
                    # Calculate wave position for this character
                    wave_pos = (i + offset) % len(wave_colors)
                    color = wave_colors[wave_pos]
                    result.append(f"[{color}]{char}[/{color}]")
                return "".join(result)

            def format_tool_name_for_status(tool_name: str) -> str:
                """Format tool name for status display: 'toolset__function' → 'function'"""
                if "__" in tool_name:
                    _, function = tool_name.split("__", 1)
                    return function
                return tool_name

            def update_processing_status():
                current_output_tokens = estimated_output_tokens
                elapsed = time.time() - start_time

                # Time-based animation for consistent speed
                animation_fps = 8  # Spinner: 8 frames per second
                wave_fps = 4  # Wave: 4 steps per second (slower for visual effect)

                animation_index = int(elapsed * animation_fps) % len(animation_frames)
                wave_offset = int(elapsed * wave_fps)

                current_frame = animation_frames[animation_index]

                if self._current_tool_name and self._tools_executing:
                    display_name = format_tool_name_for_status(self._current_tool_name)
                    wave_text = create_wave_text(f"Running {display_name}...", wave_offset)
                    status_text = f"[dim]{current_frame}[/dim] {wave_text} [dim]{sep} {self._format_token_count(input_tokens)} in, {self._format_token_count(current_output_tokens)} out"
                else:
                    wave_text = create_wave_text("Processing...", wave_offset)
                    status_text = f"[dim]{current_frame}[/dim] {wave_text} [dim]{sep} {self._format_token_count(input_tokens)} in, {self._format_token_count(current_output_tokens)} out"

                if elapsed > 1:
                    status_text += f"[dim] {sep} {elapsed:.1f}s[/dim]"

                processing_live.update(Text.from_markup(status_text))

            try:
                update_processing_status()
                self._tools_executing = False

                # Thread-based animation events (defined early for use in callbacks)
                animation_stop_event = threading.Event()
                animation_pause_event = threading.Event()  # For pausing during prints

                def smart_process_chunk(chunk: dict):
                    process_chunk(chunk)
                    # Note: animation now handled by separate thread

                def process_step_message(step: dict):
                    """Handle step messages (tool calls, tool results, assistant content)."""
                    # Handle assistant content FIRST (before tool calls)
                    # This prints intermediate thoughts before showing tool usage
                    if step.get("role") == "assistant" and step.get("content"):
                        assistant_content = step.get("content")
                        if assistant_content.strip():
                            # Pause animation, print content, resume animation
                            animation_pause_event.set()
                            processing_live.stop()
                            self.console.print()
                            if "```" in assistant_content or "def " in assistant_content or "import " in assistant_content:
                                self.console.print(assistant_content)
                            else:
                                self.console.print(Markdown(assistant_content))
                            self.console.print()
                            processing_live.start()
                            animation_pause_event.clear()
                            # Clear buffer to avoid duplicate printing at the end
                            content_buffer.clear()

                    # Handle tool calls
                    if tool_calls := step.get("tool_calls"):
                        for call in tool_calls:
                            tool_name = call.get("function", {}).get("name")
                            if tool_name:
                                try:
                                    import json
                                    args = json.loads(
                                        call.get("function", {}).get("arguments", "{}")
                                    )
                                except Exception:
                                    args = {}
                                self.print_tool_call(tool_name, args)

                    # Handle tool results
                    elif step.get("role") == "tool":
                        tool_name = step.get("tool_name", "")
                        content = step.get("content", "")
                        try:
                            import json
                            result = json.loads(content)
                            self.print_tool_result(tool_name, result)
                        except Exception:
                            if content.strip():
                                self.print_tool_result(tool_name, {"output": content})

                self._current_live_display = processing_live

                def animation_thread_func():
                    """Run animation in separate thread to avoid event loop blocking."""
                    while not animation_stop_event.is_set():
                        # Check if paused (during content printing)
                        if not animation_pause_event.is_set():
                            try:
                                update_processing_status()
                            except Exception:
                                pass  # Ignore errors in animation thread
                        time.sleep(0.125)  # 8 fps

                animation_thread = threading.Thread(target=animation_thread_func, daemon=True)
                animation_thread.start()

                # Call ChatRoom.chat()
                chat_task = asyncio.create_task(
                    self._chatroom.chat(
                        chat_id=self._chat_id,
                        message=[{"role": "user", "content": message}],
                        process_chunk=smart_process_chunk,
                        process_step_message=process_step_message,
                    )
                )

                self._current_agent_task = chat_task

                try:
                    result = await chat_task
                except asyncio.CancelledError:
                    self.console.print("\n[yellow]Operation was cancelled[/yellow]")
                    raise KeyboardInterrupt
                finally:
                    self._current_agent_task = None
                    # Stop animation thread
                    animation_stop_event.set()
                    animation_thread.join(timeout=0.5)

                # Check for errors in result
                if result and not result.get("success", True):
                    processing_live.stop()  # Stop Live before printing error
                    error_msg = result.get("message", "Unknown error")
                    self.console.print(f"\n[red]Error:[/red] {error_msg}")
                    self.console.print(
                        "[dim]You can continue the conversation or type 'exit' to quit[/dim]"
                    )
                    return

                # Final output token calculation
                if content_buffer:
                    full_content = "".join(content_buffer)
                    if full_content.strip():
                        output_tokens = self._estimate_tokens(full_content)

                # Update token statistics
                self._update_token_stats(input_tokens, output_tokens)
                self.message_count += 1

            except KeyboardInterrupt:
                self._interrupt_count = 0
                if self._current_agent_task and not self._current_agent_task.done():
                    self._current_agent_task.cancel()
                    try:
                        await self._current_agent_task
                    except asyncio.CancelledError:
                        pass
                # Stop animation thread
                if "animation_stop_event" in locals():
                    animation_stop_event.set()
                if "animation_thread" in locals() and animation_thread.is_alive():
                    animation_thread.join(timeout=0.5)
                return
            except Exception as e:
                self.console.print(f"\n[red]Error:[/red] {str(e)}")
                self.console.print(
                    "[dim]You can continue the conversation or type 'exit' to quit[/dim]"
                )
            finally:
                # Stop animation thread first
                if "animation_stop_event" in locals():
                    animation_stop_event.set()
                if "animation_thread" in locals() and animation_thread.is_alive():
                    animation_thread.join(timeout=0.5)
                processing_live.stop()
                self._tools_executing = False
                self._current_live_display = None
                self._current_tool_name = None
                if self._current_agent_task and not self._current_agent_task.done():
                    self._current_agent_task.cancel()
                self._current_agent_task = None
        finally:
            if "processing_live" in locals():
                processing_live.stop()

        # Print final content
        self.console.print()
        if content_buffer:
            full_content = "".join(content_buffer)
            if full_content.strip():
                if (
                    "```" in full_content
                    or "def " in full_content
                    or "import " in full_content
                ):
                    self.console.print(full_content)
                else:
                    self.console.print(Markdown(full_content))

        self.console.print()

    # ===== Chat management commands =====

    async def _handle_new_chat(self):
        """Create a new chat session."""
        result = await self._chatroom.create_chat()
        self._chat_id = result["chat_id"]
        self.console.print(
            f"[green]✅ Created new chat:[/green] {result.get('chat_name', self._chat_id)}"
        )
        self.console.print()

    async def _handle_list_chats(self):
        """List all chat sessions."""
        result = await self._chatroom.list_chats()
        if result.get("success"):
            chats = result.get("chats", [])
            self.console.print()
            self.console.print(
                "[dim][bold blue]-- CHATS ------------------------------------------------------------[/bold blue][/dim]"
            )
            self.console.print()
            if not chats:
                self.console.print("[dim]No chats found[/dim]")
            else:
                for chat in chats[:10]:  # Show last 10
                    marker = "→ " if chat.get("chat_id") == self._chat_id else "  "
                    name = chat.get("chat_name", "Unnamed")
                    chat_id = chat.get("chat_id", "")[:8]
                    self.console.print(f"[dim]{marker}[/dim][bold]{name}[/bold] [dim]({chat_id})[/dim]")
            self.console.print()
        else:
            self.console.print(f"[red]Error listing chats: {result.get('message')}[/red]")

    async def _handle_switch_chat(self, chat_id: str):
        """Switch to a different chat session."""
        # Verify chat exists
        result = await self._chatroom.list_chats()
        if result.get("success"):
            chats = result.get("chats", [])
            found = None
            for chat in chats:
                if chat.get("chat_id", "").startswith(chat_id) or chat.get(
                    "chat_name", ""
                ).lower().startswith(chat_id.lower()):
                    found = chat
                    break

            if found:
                self._chat_id = found["chat_id"]
                self.console.print(
                    f"[green]✅ Switched to:[/green] {found.get('chat_name', self._chat_id)}"
                )
            else:
                self.console.print(f"[red]Chat not found: {chat_id}[/red]")
        self.console.print()

    async def _handle_show_agents(self):
        """Show agents in current team."""
        self.console.print()
        self.console.print(
            "[dim][bold blue]-- AGENTS -----------------------------------------------------------[/bold blue][/dim]"
        )
        self.console.print()

        if self._team:
            for agent in self._team.agents.values():
                agent_info = f"  • [bright_blue]{agent.name}[/bright_blue]"
                if hasattr(agent, "models") and agent.models:
                    model = (
                        agent.models[0]
                        if isinstance(agent.models, list)
                        else agent.models
                    )
                    agent_info += f" [dim]({model})[/dim]"
                self.console.print(agent_info)
        else:
            self.console.print("[dim]No team loaded[/dim]")
        self.console.print()

    async def _handle_clear(self):
        """Clear current chat and create new one."""
        self.console.clear()
        # Delete old chat and create new
        if self._chat_id:
            await self._chatroom.delete_chat(self._chat_id)
        result = await self._chatroom.create_chat("repl-session")
        self._chat_id = result["chat_id"]
        self._current_agent_name = None
        self._last_printed_agent = None
        await self.print_greeting()

    def _handle_save_command(self, command: str):
        """Handle /save command."""
        try:
            parts = command.split()
            if len(parts) > 1:
                filename = parts[1]
                if not filename.endswith(".json"):
                    filename += ".json"
            else:
                filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"

            # Save via ChatRoom's memory manager
            asyncio.create_task(self._save_chat(filename))
            self.console.print(f"[green]✅ Conversation saved to:[/green] {filename}")
        except Exception as e:
            self.console.print(f"[red]Error saving conversation: {str(e)}[/red]")
        self.console.print()

    async def _save_chat(self, filename: str):
        """Save current chat to file."""
        memory = await self._chatroom.memory_manager.get_memory(self._chat_id)
        memory.save(filename)

    def _handle_load_command(self, command: str):
        """Handle /load command."""
        try:
            parts = command.split()
            filename = parts[1]
            self.console.print(
                f"[yellow]Note: /load is not fully supported in ChatRoom mode. Use /switch instead.[/yellow]"
            )
        except Exception as e:
            self.console.print(f"[red]Error loading conversation: {str(e)}[/red]")
        self.console.print()


if __name__ == "__main__":
    agent = Agent("agent", "You are a helpful assistant.")
    repl = Repl(agent=agent)
    asyncio.run(repl.run())
