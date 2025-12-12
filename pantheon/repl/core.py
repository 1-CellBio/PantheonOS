"""REPL - Command line interface for Pantheon agents, based on ChatRoom."""

import asyncio
import sys
import time
import signal
import threading
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from rich.text import Text
from rich.live import Live
from rich.markdown import Markdown

# prompt_toolkit for enhanced input
from prompt_toolkit.patch_stdout import patch_stdout

# Simple readline support for history (fallback)
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
from .renderers import DisplayMode
from .handlers.base import CommandHandler
from .handlers.template_handler import TemplateHandler, load_template
from .handlers.builtin.bash import BashCommandHandler
from .prompt_app import PantheonInputApp, ReplCompleter
from .utils import get_animation_frames, get_separator, format_tool_name, format_relative_time


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
        memory_dir: str | None = None,
        chat_id: str | None = None,
    ):
        if memory_dir is None:
            from ..settings import get_settings
            memory_dir = str(get_settings().memory_dir)
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

        # prompt_toolkit application for enhanced input
        self.prompt_app: PantheonInputApp | None = None
        self._use_prompt_toolkit = True  # Enable new UI by default
        self.message_queue = None

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
        self._last_interrupt_time = 0.0

        def signal_handler(signum, frame):
            should_exit = self.handle_interrupt()
            if should_exit:
                sys.exit(1)

        if hasattr(signal, "SIGINT"):
            signal.signal(signal.SIGINT, signal_handler)

    def handle_interrupt(self) -> bool:
        """Handle Ctrl+C interrupt with double-press logic.
        
        Returns:
            True if should exit (double press), False otherwise.
        """
        current_time = time.time()

        if current_time - self._last_interrupt_time < 2.0:
            self._interrupt_count += 1
        else:
            self._interrupt_count = 1

        self._last_interrupt_time = current_time

        if self._interrupt_count == 1:
            self.output.console.print(
                "\n[yellow]Press Ctrl+C again within 2 seconds to exit[/yellow]"
            )
            # Cancel any running agent task
            if (
                hasattr(self, "_current_agent_task")
                and self._current_agent_task
                and not self._current_agent_task.done()
            ):
                try:
                    self._current_agent_task.cancel()
                except Exception:
                    pass
            return False
        elif self._interrupt_count >= 2:
            self._print_session_summary()
            return True
        elif self._interrupt_count >= 2:
            self._print_session_summary()
            return True
        return False

    async def _processing_loop(self):
        """Concurrent loop for processing messages from queue."""
        while True:
            try:
                # Get message from queue
                current_message = await self.message_queue.get()
                
                # Check for exit command first (handled here to stop processing)
                if current_message.strip().lower() in ["exit", "quit", "q", "/exit", "/quit", "/q"]:
                     self.message_queue.task_done()
                     self._print_session_summary()
                     # We can't easily break the main run wait, so we might need a signal or just let the input app exit
                     # Actually input app exit is handled by prompt_toolkit Exit exception usually.
                     # But here we are consuming from queue. 
                     # Let's handle commands same as before.
                     pass 

                # Process message
                if self.prompt_app:
                    self.prompt_app.start_processing(self._estimate_tokens(current_message))
                
                try:
                    # Reuse existing logic to process command or chat
                    # We need to adapt the command handling logic from run() to here
                    await self._handle_message_or_command(current_message)
                except Exception as e:
                    self.console.print(f"[red]Error processing message: {e}[/red]")
                finally:
                    if self.prompt_app:
                        self.prompt_app.stop_processing()
                        # Update token usage in status bar
                        self._update_status_bar_token_usage()
                    self.message_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.console.print(f"[red]Processing loop error: {e}[/red]")
                await asyncio.sleep(1)

    def _setup_input_system(self):
        """Setup simple input system with readline history."""
        if READLINE_AVAILABLE:
            if self.history_file.exists():
                try:
                    readline.read_history_file(str(self.history_file))
                except (OSError, FileNotFoundError):
                    # Ignore errors from empty/corrupted/missing history files
                    # (macOS libedit can throw EINVAL on empty files)
                    pass
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
        """Estimate token count with language-aware approximation.

        Different character types have different token ratios:
        - CJK characters (Chinese/Japanese/Korean): ~1.5 tokens per character
        - ASCII (English, code, punctuation): ~0.25 tokens per character (4 chars/token)
        - Other Unicode: ~0.5 tokens per character (2 chars/token)
        """
        if not text:
            return 0

        cjk_chars = 0
        ascii_chars = 0
        other_chars = 0

        for char in text:
            code = ord(char)
            # CJK Unified Ideographs and common CJK ranges
            if (0x4E00 <= code <= 0x9FFF or      # CJK Unified Ideographs
                0x3400 <= code <= 0x4DBF or      # CJK Extension A
                0x3000 <= code <= 0x303F or      # CJK Punctuation
                0xFF00 <= code <= 0xFFEF or      # Fullwidth Forms
                0xAC00 <= code <= 0xD7AF or      # Korean Hangul
                0x3040 <= code <= 0x309F or      # Japanese Hiragana
                0x30A0 <= code <= 0x30FF):       # Japanese Katakana
                cjk_chars += 1
            elif code < 128:  # ASCII
                ascii_chars += 1
            else:  # Other Unicode (emojis, symbols, etc.)
                other_chars += 1

        # Calculate estimated tokens
        tokens = (
            cjk_chars * 1.5 +       # CJK: ~1.5 tokens per char
            ascii_chars * 0.25 +    # ASCII: ~4 chars per token
            other_chars * 0.5       # Other: ~2 chars per token
        )

        return max(1, int(tokens))

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

    def _update_status_bar_token_usage(self):
        """Update token usage percentage in status bar."""
        if not self.prompt_app:
            return
        try:
            from .utils import get_token_stats
            fallback = {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "message_count": self.message_count,
            }
            token_info = get_token_stats(self._chatroom, self._chat_id, self._team, fallback)
            usage_pct = token_info.get("usage_percent", 0)
            self.prompt_app.update_token_usage(usage_pct)
        except Exception:
            pass  # Silently ignore errors


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
        self.message_queue = asyncio.Queue()

        # Print greeting (before patch_stdout context)
        await self.print_greeting()

        # Initialize prompt_toolkit app if enabled
        if self._use_prompt_toolkit:
            self.prompt_app = PantheonInputApp(
                str(self.history_file),
                ReplCompleter(self),
                self,
                self.message_queue
            )
            # Update model/agent info
            if self._team and self._team.agents:
                agent_names = list(self._team.agents.keys())
                if agent_names:
                    first_agent_name = agent_names[0]
                    agent = self._team.agents[first_agent_name]
                    
                    # Extract model name handling both scalar 'model' and list 'models'
                    model = getattr(agent, 'model', 'unknown')
                    if hasattr(agent, 'models'):
                         models = agent.models
                         if isinstance(models, list) and models:
                             model = models[0]
                         elif isinstance(models, str):
                             model = models
                             
                    self.prompt_app.update_model(model)
                    self.prompt_app.update_agent(first_agent_name)
            
            # Note: Renderers will be re-initialized inside patch_stdout context in loop

        self._parent_repl = self

        # Handle initial message if provided
        if message is not None:
            self._add_to_history(message)
            self.message_queue.put_nowait(message)

        # Main concurrency setup
        try:
             # Logic split: 
             # 1. Input App (run_async) - handles user typing
             # 2. Processing Loop - handles commands/agents
             
             if self._use_prompt_toolkit:
                 with patch_stdout(raw=True):
                     # Enter patch context for output adapter
                     self.output.enter_patch_context()
                     # Reinitialize renderers with patched console
                     self._init_renderers()

                     # Create background processing task
                     processing_task = asyncio.create_task(self._processing_loop())
                     
                     try:
                         try:
                             # Run input app (blocks until app exit)
                             await self.prompt_app.run_async()
                         except (EOFError, KeyboardInterrupt):
                             pass # Suppress stack traces on exit
                     finally:
                         # Cancel processing loop on exit
                         processing_task.cancel()
                         try:
                             await processing_task
                         except asyncio.CancelledError:
                             pass
             else:
                 # Fallback legacy loop (readline)
                 while True:
                     msg = self.ask_user_input()
                     if not msg.strip(): continue
                     await self._handle_message_or_command(msg)

        finally:
            if self._use_prompt_toolkit:
                self.output.exit_patch_context()

    async def _handle_message_or_command(self, current_message: str):
        """Process a single message or command (extracted from run loop)."""
        if not current_message:
            return

        self._add_to_history(current_message)

        # Handle commands
        cmd = current_message.strip()
        cmd_lower = cmd.lower()

        # Exit commands
        if cmd_lower in ["exit", "quit", "q", "/exit", "/quit", "/q"]:
            self._print_session_summary()
            # For prompt_toolkit app, we need to trigger app exit
            if self.prompt_app:
                self.prompt_app.app.exit()
            else:
                 sys.exit(0)
            return

        # Help command
        elif cmd_lower in ["help", "/help"]:
            self._print_help()
            return

        # Status command
        elif cmd_lower in ["status", "/status"]:
            self._print_status()
            return

        # Clear command
        elif cmd_lower in ["clear", "/clear"]:
            await self._handle_clear()
            return

        # History command
        elif cmd_lower in ["history", "/history"]:
            self._print_history()
            return

        # Tokens command
        elif cmd_lower in ["tokens", "/tokens"]:
            await self._print_token_analysis()
            return

        # Save command
        elif cmd_lower == "/save" or cmd_lower.startswith("/save "):
            self._handle_save_command(cmd)
            return

        # Load command
        elif cmd_lower.startswith("/load "):
            self._handle_load_command(cmd)
            return

        # New chat command
        elif cmd_lower in ["/new", "/new-chat"]:
            await self._handle_new_chat()
            return

        # List chats command
        elif cmd_lower in ["/list", "/chats"]:
            await self._handle_list_chats()
            return

        # Switch chat command
        elif cmd_lower.startswith("/switch "):
            chat_id = cmd[8:].strip()
            await self._handle_switch_chat(chat_id)
            return

        # Agents command
        elif cmd_lower in ["/agents", "/team"]:
            await self._handle_show_agents()
            return

        # Agent switch command: /agent <name> or /agent <number>
        elif cmd_lower.startswith("/agent "):
            agent_arg = cmd[7:].strip()
            await self._handle_switch_agent(agent_arg)
            return

        # Verbose mode command
        elif cmd_lower in ["/verbose", "/v"]:
            self.set_display_mode(DisplayMode.VERBOSE)
            self.output.console.print("[green]✓[/green] Switched to [bold]verbose[/bold] mode")
            self.output.console.print("[dim]  All code, file content, and tool details will be shown[/dim]")
            self.output.console.print()
            return

        # Compact mode command
        elif cmd_lower in ["/compact", "/c"]:
            self.set_display_mode(DisplayMode.COMPACT)
            self.output.console.print("[green]✓[/green] Switched to [bold]compact[/bold] mode")
            self.output.console.print("[dim]  Output will be truncated for readability[/dim]")
            self.output.console.print()
            return

        # Custom command handlers
        continue_flag = False
        for handler in self.handlers:
            if handler.match_command(cmd):
                result = await handler.handle_command(cmd)
                # If command returns a result, we treat it as next message? 
                # Original logic was complex loop. Simplified here: 
                # If handler returns None, we are done. If str, it's new input (uncommon).
                if result is not None:
                     # Recursive call for chained commands? 
                     # For now, just print or ignore to match previous flow mostly
                     pass
                return

        # Process with ChatRoom
        await self._process_message(current_message)

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

        # Animation frames and separator from utils
        animation_frames = get_animation_frames()
        sep = get_separator()


        processing_live = Live(console=self.console, refresh_per_second=8, transient=True)
        if not self.prompt_app:
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

            def update_processing_status():
                current_output_tokens = estimated_output_tokens
                elapsed = time.time() - start_time

                # Time-based animation for consistent speed
                animation_fps = 8  # Spinner: 8 frames per second
                wave_fps = 4  # Wave: 4 steps per second (slower for visual effect)

                animation_index = int(elapsed * animation_fps) % len(animation_frames)
                wave_offset = int(elapsed * wave_fps)

                current_frame = animation_frames[animation_index]
                
                if self.prompt_app:
                    # Update Prompt App Status Bar with animation state
                    status_str = f"{'Running ' + format_tool_name(self._current_tool_name) + '...' if (self._current_tool_name and self._tools_executing) else 'Processing...'}"
                    self.prompt_app.update_processing(
                        status=status_str,
                        output_tokens=current_output_tokens,
                        spinner=current_frame,
                        elapsed=elapsed,
                        wave_offset=wave_offset
                    )
                else:
                    # Update Rich Live Status
                    current_frame = animation_frames[animation_index]
    
                    # Build agent prefix for multi-agent mode
                    agent_prefix = ""
                    if self._is_multi_agent and self._current_agent_name:
                        agent_prefix = f"[cyan]{self._current_agent_name}[/cyan] "
    
                    # Only show token counts when we have output tokens
                    if current_output_tokens > 0:
                        token_info = f"[dim]{sep} {self._format_token_count(input_tokens)} in, {self._format_token_count(current_output_tokens)} out"
                    else:
                        token_info = ""

                    if self._current_tool_name and self._tools_executing:
                        display_name = format_tool_name(self._current_tool_name)
                        wave_text = create_wave_text(f"Running {display_name}...", wave_offset)
                        status_text = f"[dim]{current_frame}[/dim] {agent_prefix}{wave_text} {token_info}"
                    else:
                        wave_text = create_wave_text("Processing...", wave_offset)
                        status_text = f"[dim]{current_frame}[/dim] {agent_prefix}{wave_text} {token_info}"
    
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
                    # Track agent for status bar updates
                    agent_name = step.get("agent_name")
                    if agent_name:
                        # Update current agent name and status bar
                        if agent_name != self._current_agent_name:
                            self._current_agent_name = agent_name
                            if self.prompt_app:
                                self.prompt_app.update_agent(agent_name)

                    # Handle assistant content FIRST (before tool calls)
                    # This prints intermediate thoughts before showing tool usage
                    if step.get("role") == "assistant" and step.get("content"):
                        assistant_content = step.get("content")
                        if assistant_content.strip():
                            # Pause animation, print content, resume animation
                            animation_pause_event.set()
                            if not self.prompt_app:
                                processing_live.stop()
                            # Print agent name before each response (multi-agent mode)
                            if agent_name and self._is_multi_agent:
                                self.console.print(
                                    f"\n[dim]→[/dim] [bold cyan]{agent_name}[/bold cyan]"
                                )
                            self.console.print()
                            try:
                                self.console.print(Markdown(assistant_content))
                            except Exception:
                                self.console.print(assistant_content)
                            self.console.print()
                            if not self.prompt_app:
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

                        # Prefer raw_content if available (original dict)
                        raw_content = step.get("raw_content")
                        if raw_content is not None and isinstance(raw_content, dict):
                            self.print_tool_result(tool_name, raw_content)
                        else:
                            # Try to parse content
                            try:
                                import json
                                result = json.loads(content)
                                self.print_tool_result(tool_name, result)
                            except json.JSONDecodeError:
                                # Try ast.literal_eval for repr() output
                                try:
                                    import ast
                                    result = ast.literal_eval(content)
                                    if isinstance(result, dict):
                                        self.print_tool_result(tool_name, result)
                                    else:
                                        self.print_tool_result(tool_name, {"output": str(result)})
                                except Exception:
                                    if content.strip():
                                        self.print_tool_result(tool_name, {"output": content})
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

                # Yield to event loop to allow initial animation frame to render
                # This prevents "start freeze" if backend init is synchronous/heavy
                await asyncio.sleep(0.1)

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

        # Print final content with agent label (multi-agent mode)
        self.console.print()
        if content_buffer:
            full_content = "".join(content_buffer)
            if full_content.strip():
                # Show agent label before final response (multi-agent mode)
                if self._is_multi_agent and self._current_agent_name:
                    self.console.print(f"[dim]→[/dim] [bold cyan]{self._current_agent_name}[/bold cyan]")
                try:
                    self.console.print(Markdown(full_content))
                except Exception:
                    self.console.print(full_content)

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
                # Print header
                self.console.print(
                    f"[dim]  #   {'Name':<30} {'Last Activity':<18} ID[/dim]"
                )
                self.console.print(
                    "[dim]  ─────────────────────────────────────────────────────────────────[/dim]"
                )
                # Print chats
                for idx, chat in enumerate(chats[:10], 1):  # Show last 10
                    marker = "→" if chat.get("id") == self._chat_id else " "
                    name = chat.get("name", "Unnamed")
                    # Truncate long names
                    if len(name) > 28:
                        name = name[:25] + "..."
                    chat_id = chat.get("id", "")[:8]
                    last_activity = format_relative_time(
                        chat.get("last_activity_date")
                    )
                    self.console.print(
                        f"[dim]{marker}[/dim] [cyan]{idx:<3}[/cyan] [bold]{name:<30}[/bold] [dim]{last_activity:<18}[/dim] [dim]{chat_id}[/dim]"
                    )
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
                if chat.get("id", "").startswith(chat_id) or chat.get(
                    "name", ""
                ).lower().startswith(chat_id.lower()):
                    found = chat
                    break

            if found:
                self._chat_id = found["id"]
                self.console.print(
                    f"[green]✅ Switched to:[/green] {found.get('name', self._chat_id)}"
                )
            else:
                self.console.print(f"[red]Chat not found: {chat_id}[/red]")
        self.console.print()

    async def _handle_show_agents(self):
        """Show agents in current team with current agent indicator."""
        self.console.print()
        self.console.print(
            "[dim][bold blue]-- AGENTS -----------------------------------------------------------[/bold blue][/dim]"
        )
        self.console.print()

        if self._team:
            # Determine current active agent
            current_agent_name = self._current_agent_name
            if not current_agent_name:
                # Default to first agent if not set
                current_agent_name = list(self._team.agents.keys())[0] if self._team.agents else None

            # Print header
            self.console.print(
                f"[dim]  #   {'Name':<20} {'Model':<25} Description[/dim]"
            )
            self.console.print(
                "[dim]  ─────────────────────────────────────────────────────────────────[/dim]"
            )

            for idx, agent in enumerate(self._team.agents.values(), 1):
                # Check if this is the current agent
                is_current = agent.name == current_agent_name
                marker = "→" if is_current else " "

                # Get model info
                if hasattr(agent, "models") and agent.models:
                    model = (
                        agent.models[0]
                        if isinstance(agent.models, list)
                        else agent.models
                    )
                    # Truncate long model names
                    if len(model) > 23:
                        model = model[:20] + "..."
                else:
                    model = "-"

                # Get description
                description = ""
                if hasattr(agent, "description") and agent.description:
                    description = agent.description
                    if len(description) > 30:
                        description = description[:27] + "..."

                # Format agent name
                name = agent.name
                if len(name) > 18:
                    name = name[:15] + "..."

                if is_current:
                    self.console.print(
                        f"[dim]{marker}[/dim] [cyan]{idx:<3}[/cyan] [bold cyan]{name:<20}[/bold cyan] [dim]{model:<25}[/dim] [dim]{description}[/dim]"
                    )
                else:
                    self.console.print(
                        f"[dim]{marker}[/dim] [cyan]{idx:<3}[/cyan] [bold]{name:<20}[/bold] [dim]{model:<25}[/dim] [dim]{description}[/dim]"
                    )
        else:
            self.console.print("[dim]No team loaded[/dim]")
        self.console.print()

    async def _handle_switch_agent(self, agent_arg: str):
        """Switch to a different agent by name or number."""
        if not self._team:
            self.console.print("[red]No team loaded[/red]")
            return

        if not self._is_multi_agent:
            self.console.print("[yellow]Single agent mode - no switching needed[/yellow]")
            return

        agent_names = list(self._team.agents.keys())

        # Try to parse as number first
        target_agent_name = None
        try:
            idx = int(agent_arg)
            if 1 <= idx <= len(agent_names):
                target_agent_name = agent_names[idx - 1]
            else:
                self.console.print(f"[red]Invalid agent number: {idx}. Valid range: 1-{len(agent_names)}[/red]")
                return
        except ValueError:
            # Try to match by name (case-insensitive, partial match)
            agent_arg_lower = agent_arg.lower()
            for name in agent_names:
                if name.lower() == agent_arg_lower or name.lower().startswith(agent_arg_lower):
                    target_agent_name = name
                    break

            if not target_agent_name:
                self.console.print(f"[red]Agent not found: {agent_arg}[/red]")
                self.console.print(f"[dim]Available agents: {', '.join(agent_names)}[/dim]")
                return

        # Check if already on this agent
        if target_agent_name == self._current_agent_name:
            self.console.print(f"[dim]Already on agent: {target_agent_name}[/dim]")
            return

        # Switch agent via chatroom
        result = await self._chatroom.set_active_agent(self._chat_id, target_agent_name)
        if result.get("success"):
            self._current_agent_name = target_agent_name
            # Update status bar agent display
            if self.prompt_app:
                self.prompt_app.update_agent(target_agent_name)
            self.console.print(f"[green]✅ Switched to:[/green] [bold cyan]{target_agent_name}[/bold cyan]")
        else:
            self.console.print(f"[red]Failed to switch agent: {result.get('message', 'Unknown error')}[/red]")

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
        # Reset status bar to first agent
        if self.prompt_app and self._team and self._team.agents:
            first_agent_name = list(self._team.agents.keys())[0]
            self.prompt_app.update_agent(first_agent_name)
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
