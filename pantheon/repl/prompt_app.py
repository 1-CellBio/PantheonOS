"""Prompt application for REPL with prompt_toolkit integration.

This module provides a prompt_toolkit-based input session with:
- Fixed input box at bottom with horizontal line borders (top/bottom only)
- Dynamic status bar below input
- Command completion
- Async status bar refresh during processing
- Concurrent input processing via Application.run_async()
- Merged command completer logic
"""

import re
import sys
import time
import asyncio

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Callable, Awaitable

from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit, FloatContainer, Float, DynamicContainer, ConditionalContainer
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.filters import Condition
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML

from .utils import get_animation_frames, get_separator, get_wave_color
from ..constant import FILE_COMPLETION_IGNORED, PROJECT_ROOT

if TYPE_CHECKING:
    from .core import Repl


class ReplCompleter(Completer):
    """Completer that provides completions for REPL commands and file paths.

    Supports:
    - Built-in commands starting with /
    - Custom handler commands
    - File path completions starting with @
    """
    
    # Built-in commands (sync with core.py run() command handling)
    BUILTIN_COMMANDS = [
        # Basic commands
        ('/help', 'Show available commands'),
        ('/status', 'Session info'),
        ('/history', 'Command history'),
        ('/tokens', 'Token usage analysis'),
        ('/save', 'Save conversation'),
        ('/clear', 'Clear/new chat'),
        ('/exit', 'Exit REPL'),
        # Chat management
        ('/new', 'New chat session'),
        ('/list', 'List chat sessions'),
        ('/switch', 'Switch to another chat'),
        # Agent/Team
        ('/agents', 'Show agents in team'),
        ('/agent', 'Switch to specific agent'),
        ('/team', 'Switch team: /team list | /team <id>'),
        # Display modes
        ('/verbose', 'Verbose output mode'),
        ('/v', 'Verbose (short)'),
        ('/compact', 'Compact output mode'),
        ('/c', 'Compact (short)'),
    ]
    
    def __init__(self, repl: "Repl" = None):
        """Initialize completer.
        
        Args:
            repl: Reference to Repl instance for accessing custom handlers.
        """
        self.repl = repl
    
    def get_completions(self, document, complete_event):
        """Generate completions for the current input.

        Args:
            document: The current document being edited.
            complete_event: Information about how completion was triggered.

        Yields:
            Completion objects for matching commands or file paths.
        """
        text = document.text_before_cursor.lstrip()

        # File path completion: triggered by @
        if '@' in text:
            yield from self._get_file_completions(document)
            return

        # Command completion: starts with /
        if not text.startswith('/'):
            return

        # Built-in command completions
        for cmd, desc in self.BUILTIN_COMMANDS:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display_meta=desc
                )

        # Custom handler command completions
        if self.repl and hasattr(self.repl, 'handlers'):
            for handler in self.repl.handlers:
                if hasattr(handler, 'get_commands'):
                    for cmd, desc in handler.get_commands():
                        if cmd.startswith(text):
                            yield Completion(
                                cmd,
                                start_position=-len(text),
                                display_meta=desc
                            )

    def _get_file_completions(self, document):
        """Generate file path completions for @ mentions."""
        text = document.text_before_cursor

        # Find the last @ position
        at_pos = text.rfind('@')
        if at_pos == -1:
            return

        # Extract path prefix after @
        path_prefix = text[at_pos + 1:]
        start_position = -(len(path_prefix) + 1)  # Include @

        # Get workspace directory
        workspace = self._get_workspace()

        # Parse path: separate directory and filename prefix
        if '/' in path_prefix:
            dir_part, name_prefix = path_prefix.rsplit('/', 1)
            search_dir = workspace / dir_part
        else:
            dir_part, name_prefix = "", path_prefix
            search_dir = workspace

        if not search_dir.exists() or not search_dir.is_dir():
            return

        # List and filter files
        try:
            entries = sorted(
                search_dir.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except PermissionError:
            return

        for entry in entries:
            # Only filter ignored directories (keep hidden files like .env)
            if entry.name in FILE_COMPLETION_IGNORED:
                continue

            # Prefix match (case-insensitive)
            if not entry.name.lower().startswith(name_prefix.lower()):
                continue

            # Build completion path
            rel_path = f"{dir_part}/{entry.name}" if dir_part else entry.name

            if entry.is_dir():
                yield Completion(
                    f"@{rel_path}/",
                    start_position=start_position,
                    display=f"{entry.name}/",
                    display_meta="dir"
                )
            else:
                yield Completion(
                    f"@{rel_path}",
                    start_position=start_position,
                    display=entry.name,
                    display_meta=self._get_file_type(entry)
                )

    def _get_workspace(self) -> Path:
        """Get workspace path from repl or fallback to PROJECT_ROOT."""
        if self.repl:
            try:
                endpoint = self.repl._chatroom._endpoint
                if endpoint and hasattr(endpoint, 'path'):
                    return endpoint.path
            except AttributeError:
                pass
        return PROJECT_ROOT

    def _get_file_type(self, path: Path) -> str:
        """Get file type description based on extension."""
        ext_map = {
            '.py': 'py', '.js': 'js', '.ts': 'ts', '.tsx': 'tsx',
            '.md': 'md', '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml',
            '.toml': 'toml', '.txt': 'txt', '.sh': 'sh', '.css': 'css',
            '.html': 'html', '.sql': 'sql', '.rs': 'rs', '.go': 'go',
        }
        return ext_map.get(path.suffix.lower(), 'file')


def create_key_bindings(app_instance: "PantheonInputApp") -> KeyBindings:
    """Create key bindings for app input handling.
    
    Args:
        app_instance: The PantheonInputApp instance.
    """
    kb = KeyBindings()
    
    @kb.add('c-c')
    def _(event):
        """Ctrl+C to clear/cancel or exit."""
        # Calculate how many extra lines need to be cleared
        prev_lines = app_instance._prev_line_count

        # Reset state first
        app_instance._prev_line_count = 1

        # Clean current input buffer
        event.current_buffer.text = ""

        # Force complete re-render by resetting renderer state
        try:
            renderer = event.app.renderer
            # Reset the renderer's height tracking
            renderer.reset()
            event.app.invalidate()
        except Exception:
            pass

        # Check if we should exit (double press check via repl)
        repl = app_instance.repl
        if hasattr(repl, 'handle_interrupt'):
            should_exit = repl.handle_interrupt()
            if should_exit:
                # Signal app to exit
                event.app.exit(exception=EOFError())
        else:
            pass

    @kb.add('enter')
    def _(event):
        """Enter to submit or accept completion."""
        buffer = event.current_buffer

        # If completion menu is open, accept the selected completion
        if buffer.complete_state:
            completion = buffer.complete_state.current_completion
            if completion:
                buffer.apply_completion(completion)
            return

        # Otherwise submit the input
        text = buffer.text
        if text.strip():
            app_instance.accept_input(buffer)

    # Condition: check if processing
    is_processing = Condition(lambda: getattr(app_instance, '_is_processing', False))

    @kb.add('escape', 'enter')  # Alt+Enter to insert newline
    @kb.add('c-j')  # Ctrl+J as reliable fallback
    def _(event):
        """Insert newline."""
        event.current_buffer.insert_text('\n')

    @kb.add('c-d')
    def _(event):
        """Ctrl+D to exit (EOF)."""
        # Print session summary before exit
        repl = app_instance.repl
        if hasattr(repl, '_print_session_summary'):
            repl._print_session_summary()
        # Signal app to exit
        event.app.exit(exception=EOFError())

    @kb.add('escape', eager=True, filter=is_processing)
    def _(event):
        """Escape to cancel operation immediately (only when processing)."""
        # Cancel any running agent task
        repl = app_instance.repl
        if hasattr(repl, '_current_agent_task') and repl._current_agent_task:
            if not repl._current_agent_task.done():
                repl._current_agent_task.cancel()
                print("\n[Cancelled]")

    @kb.add('escape', filter=~is_processing)
    def _(event):
        """Escape to clear input (when idle)."""
        event.current_buffer.text = ""

    return kb


class PantheonInputApp:
    """prompt_toolkit Application based input session.
    
    Uses standard Application + Layout + Frame + TextArea to achieve:
    1. Bordered Input Box
    2. Fixed placement at bottom (full_screen=False)
    3. Concurrent input usage
    
    Features:
    - Bordered input frame with dynamic status bar title
    - Command history and auto-suggest
    - Async status bar refresh during processing
    """
    
    def __init__(self, history_file: str, completer: ReplCompleter, repl: "Repl", message_queue: asyncio.Queue):
        """Initialize prompt app.
        
        Args:
            history_file: Path to command history file.
            completer: ReplCompleter instance for tab completion.
            repl: Reference to parent Repl instance.
            message_queue: Queue to put submitted messages into.
        """
        self.repl = repl
        self.message_queue = message_queue
        
        # Unicode compatibility - detect and fallback to ASCII if needed
        self.SPINNER_FRAMES = get_animation_frames()
        self._separator = get_separator()
        
        # Status information
        self._model_name = ""
        self._current_agent = ""
        self._status_text = "Ready"
        self._is_processing = False
        self._start_time = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._refresh_task = None
        self._wave_offset = 0
        self._token_usage_pct = 0.0  # Token usage percentage for status bar
        self._total_cost = 0.0  # Total session cost
        
        # Task panel state
        self._task_panel_visible = False
        self._task_panel_content = ""  # Pre-rendered ANSI content

        # Style
        self.style = Style.from_dict({
            'frame.border': 'fg:ansiblue',
            'frame.label': 'fg:ansiwhite bold',
            'input-area': '', # Default text style
        })
        
        # Input Widget (TextArea)
        self.text_area = TextArea(
            multiline=True,
            completer=completer,
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            focusable=True,
            style="class:input-area",
            prompt='> ',
            height=Dimension(min=1, max=10),  # Dynamic height, 1-10 lines
        )

        # Track line count for dynamic height adjustment
        self._prev_line_count = 1

        def on_text_changed(buffer):
            """Handle text changes - force redraw when lines decrease."""
            if not hasattr(self, 'app'):
                return

            # Calculate current line count (capped at max height)
            current_lines = min(buffer.text.count('\n') + 1, 10)

            if current_lines < self._prev_line_count:
                # Lines decreased - erase and request full redraw
                try:
                    self.app.renderer.erase()
                except Exception:
                    pass

            self._prev_line_count = current_lines
            self.app.invalidate()

        self.text_area.buffer.on_text_changed += on_text_changed
        
        # Key bindings
        self.kb = create_key_bindings(self)
        # Apply key bindings to the text area's buffer
        # Note: TextArea creates its own Buffer/KeyBindings, we merge ours globally or attached to Window
        # The easiest way for TextArea is to handle accept_handler, but we want custom Enter logic
        # So we attach our KB to the Layout or Application
        
        # Main Layout Structure
        # 1. Task Panel (when active task)
        # 2. Processing status line (above input, only visible when processing)
        # 3. Input Area (TextArea) with top/bottom borders
        # 4. Status Bar (model/agent info) below Input
        # All wrapped in FloatContainer to support CompletionsMenu (dropdown)

        self.processing_control = FormattedTextControl(text=self.get_processing_formatted_text)
        self.status_control = FormattedTextControl(text=self.get_status_formatted_text)
        self.task_panel_control = FormattedTextControl(text=self._get_task_panel_text)

        self.root_container = HSplit([
            # Dynamic Task Panel (only visible when there's an active task)
            ConditionalContainer(
                Window(
                    content=self.task_panel_control,
                    height=Dimension(
                        min=6, 
                        max=20,  # Increased max height
                        preferred=20,  # Prefer max height if content available
                    ),
                    style="class:task-panel"
                ),
                filter=Condition(lambda: self._task_panel_visible)
            ),

            # Empty line for spacing (only when processing)
            ConditionalContainer(
                Window(height=1),
                filter=Condition(lambda: self._is_processing)
            ),

            # Processing status line (directly above input)
            ConditionalContainer(
                Window(
                    content=self.processing_control,
                    height=1,
                    style="class:processing-bar"
                ),
                filter=Condition(lambda: self._is_processing)
            ),

            # Dynamic Input Container
            DynamicContainer(self._get_input_container),

            # Status Bar below input (model/agent info)
            Window(
                content=self.status_control,
                height=1,
                style="class:status-bar"
            ),
        ])
        
        self.layout = Layout(
            FloatContainer(
                content=self.root_container,
                floats=[
                    Float(
                        xcursor=True,
                        ycursor=False,
                        bottom=2,  # Anchor above input line (Status=1 + BottomLine=1 => 2 from bottom)
                        content=CompletionsMenu(max_height=16, scroll_offset=1),
                    ),
                ],
            )
        )
        
        # Initialize Application
        self.app = Application(
            layout=self.layout,
            style=self.style,
            key_bindings=self.kb,
            mouse_support=False, # Disable mouse support to allow terminal scrolling
            full_screen=False, 
            refresh_interval=0.5,
        )
    
    def _create_horizontal_line(self, char: str = '─', style: str = 'fg:ansiblue'):
        """Create a horizontal line that spans terminal width."""
        def get_line():
            # Return a long line, prompt_toolkit will auto-truncate to terminal width
            return [(style, char * 500)]

        return Window(
            content=FormattedTextControl(get_line),
            height=1,
        )

    def _get_input_container(self):
        """Return input container with top/bottom horizontal lines (no side borders)."""
        return HSplit([
            self._create_horizontal_line('─', 'fg:ansiblue'),  # Top line
            self.text_area,
            self._create_horizontal_line('─', 'fg:ansiblue'),  # Bottom line
        ])

    def accept_input(self, buffer: Buffer):
        """Handle input submission from TextArea."""
        text = buffer.text

        # Add to history (TextArea handles file history automatically on submit usually,
        # but since we manual handle Enter, we should optimize)
        buffer.append_to_history()

        # Print to stdout so it appears in scrollback (show original text with @)
        # We must use print WITHOUT redirecting to the app buffer, so use proper stdout
        # But we are likely inside patch_stdout, so standard print works well to put text 'above'
        # Format: "> message" with light gray background (ANSI 256-color: 238)
        bg_color = "\033[48;5;238m"  # Light gray background
        reset = "\033[0m"
        # Handle multiline: prefix each line with "> "
        lines = text.split('\n')
        formatted_lines = [f"{bg_color}> {line}{reset}" for line in lines]
        print("\n" + "\n".join(formatted_lines))

        # Expand @path mentions to absolute paths before sending to agent
        text = self._expand_file_mentions(text)

        # Put into queue
        self.message_queue.put_nowait(text)

        # Reset buffer
        buffer.reset()

    def _expand_file_mentions(self, text: str) -> str:
        """Convert @relative/path to '/absolute/path' in text."""
        workspace = self._get_workspace()

        # Pattern: @ followed by a path (not starting with /)
        # Match @word or @path/to/file but not @/absolute/path
        pattern = r'@(?!/)([\w./\-]+)'

        def replace_path(match):
            rel_path = match.group(1)
            abs_path = (workspace / rel_path).resolve()
            if abs_path.exists():
                return f"'{abs_path}'"
            # Keep original if path doesn't exist
            return match.group(0)

        return re.sub(pattern, replace_path, text)

    def _get_workspace(self) -> Path:
        """Get workspace path from repl or fallback to PROJECT_ROOT."""
        if self.repl:
            try:
                endpoint = self.repl._chatroom._endpoint
                if endpoint and hasattr(endpoint, 'path'):
                    return endpoint.path
            except AttributeError:
                pass
        return PROJECT_ROOT
    
    async def run_async(self):
        """Run the application asynchronously."""
        # Ensure text area is focused
        self.app.layout.focus(self.text_area)
        await self.app.run_async()
    
    def get_processing_formatted_text(self) -> HTML:
        """Generate processing status line content (above input) with wave animation."""
        # Create wave text
        wave_text_parts = []
        clean_status = self._status_text

        # Apply wave color to each character
        for i, char in enumerate(clean_status):
            if char.isspace():
                wave_text_parts.append(char)
                continue
            color = get_wave_color(i, self._wave_offset)
            wave_text_parts.append(f'<style fg="{color}">{char}</style>')

        wave_html = "".join(wave_text_parts)

        # Only show token counts when we have output tokens
        if self._output_tokens > 0:
            token_info = f"{self._separator} {self._input_tokens} in, {self._output_tokens} out "
        else:
            token_info = ""

        return HTML(
            f"{self._current_spinner} {wave_html} {token_info}"
            f"{self._separator} {self._current_elapsed:.1f}s "
            f'{self._separator} <style fg="#888888">[Esc] cancel</style>'
        )

    def get_status_formatted_text(self) -> HTML:
        """Generate bottom status bar content (model/agent info) in muted gray."""
        usage_display = f"ctx: {self._token_usage_pct:.0f}%" if self._token_usage_pct > 0 else "ctx: 0%"
        if self._total_cost and self._total_cost > 0:
            usage_display += f" │ cost: ${self._total_cost:.4f}"
        status = "Processing..." if self._is_processing else "Ready"
        
        return HTML(
            f'<style fg="#666666">⏺ {self._model_name} │ agent: {self._current_agent} │ {usage_display} │ {status}</style>'
        )

    def start_processing(self, input_tokens: int = 0):
        """Mark processing start."""
        self._is_processing = True
        self._input_tokens = input_tokens
        self._output_tokens = 0
        self._current_elapsed = 0.0
        self._current_spinner = self.SPINNER_FRAMES[0]
        self._wave_offset = 0
        self._status_text = "Processing..."
        self.app.invalidate()
    
    def update_processing(
        self, 
        status: str = None, 
        output_tokens: int = None, 
        tool_name: str = None,
        spinner: str = None,
        elapsed: float = None,
        wave_offset: int = None
    ):
        """Update processing status from external driver."""
        if tool_name:
            self._status_text = f"Running {tool_name}..."
        elif status:
            self._status_text = status
            
        if output_tokens is not None:
            self._output_tokens = output_tokens
            
        if spinner:
            self._current_spinner = spinner
            
        if elapsed is not None:
            self._current_elapsed = elapsed

        if wave_offset is not None:
            self._wave_offset = wave_offset

        # Trigger redraw immediately
        self.app.invalidate()
    
    def stop_processing(self):
        """Mark processing complete."""
        self._is_processing = False
        self._status_text = "Ready"
        self.app.invalidate()
    
    def update_model(self, model_name: str):
        self._model_name = model_name
        self.app.invalidate()
    
    def update_agent(self, agent_name: str):
        self._current_agent = agent_name
        self.app.invalidate()
    
    def update_token_usage(self, usage_pct: float, total_cost: float = 0.0):
        """Update token usage percentage for status bar display."""
        self._token_usage_pct = usage_pct
        self._total_cost = total_cost
        self.app.invalidate()
    
    # === Task Panel Methods ===
    
    def _get_task_panel_text(self):
        """Generate task panel content for prompt_toolkit."""
        if not self._task_panel_content:
            return []
        # Return ANSI formatted text
        from prompt_toolkit.formatted_text import ANSI
        return ANSI(self._task_panel_content)
    

    def update_task_panel(self, ansi_content: str):
        """Update task panel content (called by TaskUIRenderer).
        
        Args:
            ansi_content: Pre-rendered ANSI string content
        """
        self._task_panel_content = ansi_content
        self.app.invalidate()
    
    def show_task_panel(self):
        """Show task panel (when there's an active task)."""
        self._task_panel_visible = True
        self.app.invalidate()
    
    def hide_task_panel(self):
        """Hide task panel (no active task)."""
        self._task_panel_visible = False
        self._task_panel_content = ""
        self.app.invalidate()
    
