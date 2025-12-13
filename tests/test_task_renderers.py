"""Tests for REPL Task UI Renderers."""
import pytest
from io import StringIO

from rich.console import Console

from pantheon.repl.task_renderers import (
    ToolCallInfo,
    AssistantStep,
    TaskUIState,
    TaskUIRenderer,
    NotifyUIRenderer,
)


class TestToolCallInfo:
    """Test ToolCallInfo dataclass."""
    
    def test_creation(self):
        info = ToolCallInfo(name="file_manager__read_file", key_param="/path/file.py", is_running=True)
        assert info.name == "file_manager__read_file"
        assert info.key_param == "/path/file.py"
        assert info.is_running is True


class TestAssistantStep:
    """Test AssistantStep dataclass."""
    
    def test_default(self):
        step = AssistantStep()
        assert step.content == ""
        assert step.tool_calls == []
    
    def test_with_content(self):
        step = AssistantStep(content="Hello")
        assert step.content == "Hello"
    
    def test_with_tools(self):
        tool = ToolCallInfo(name="test", key_param="", is_running=False)
        step = AssistantStep(tool_calls=[tool])
        assert len(step.tool_calls) == 1


class TestTaskUIState:
    """Test TaskUIState dataclass."""
    
    def test_initial_state(self):
        state = TaskUIState()
        assert state.task_name == ""
        assert state.mode == ""
        assert state.recent_steps == []
        assert state._current_step is None
    
    def test_reset(self):
        state = TaskUIState()
        state.task_name = "Test"
        state.mode = "PLANNING"
        state.recent_steps = [AssistantStep()]
        state._current_step = AssistantStep()
        
        state.reset()
        
        assert state.task_name == ""
        assert state.mode == ""
        assert state.recent_steps == []
        assert state._current_step is None


class TestTaskUIRenderer:
    """Test TaskUIRenderer class."""
    
    @pytest.fixture
    def console(self):
        return Console(file=StringIO(), force_terminal=True, width=80)
    
    @pytest.fixture
    def renderer(self, console):
        return TaskUIRenderer(console)
    
    def test_has_active_task_false(self, renderer):
        assert renderer.has_active_task() is False
    
    def test_has_active_task_true(self, renderer):
        renderer.update_task_boundary({
            "TaskName": "Test",
            "Mode": "PLANNING",
            "TaskStatus": "Status",
            "TaskSummary": "Summary"
        })
        assert renderer.has_active_task() is True
    
    def test_update_task_boundary_basic(self, renderer):
        renderer.update_task_boundary({
            "TaskName": "Test Task",
            "Mode": "EXECUTION",
            "TaskStatus": "Running test",
            "TaskSummary": "Testing"
        })
        
        assert renderer.state.task_name == "Test Task"
        assert renderer.state.mode == "EXECUTION"
        assert renderer.state.current_status == "Running test"
        assert renderer.state.summary == "Testing"
    
    def test_update_task_boundary_same_substitution(self, renderer):
        # First task
        renderer.update_task_boundary({
            "TaskName": "Initial Task",
            "Mode": "PLANNING",
            "TaskStatus": "Planning",
            "TaskSummary": "Starting"
        })
        
        # Update with %SAME%
        renderer.update_task_boundary({
            "TaskName": "%SAME%",
            "Mode": "%SAME%",
            "TaskStatus": "New status",
            "TaskSummary": "%SAME%"
        })
        
        assert renderer.state.task_name == "Initial Task"
        assert renderer.state.mode == "PLANNING"
        assert renderer.state.current_status == "New status"
        assert renderer.state.summary == "Starting"
    
    def test_status_history_accumulation(self, renderer):
        renderer.update_task_boundary({
            "TaskName": "Test",
            "Mode": "PLANNING",
            "TaskStatus": "Status 1",
            "TaskSummary": "Summary"
        })
        
        renderer.update_task_boundary({
            "TaskName": "%SAME%",
            "Mode": "%SAME%",
            "TaskStatus": "Status 2",
            "TaskSummary": "%SAME%"
        })
        
        assert len(renderer.state.status_history) == 1
        assert renderer.state.status_history[0] == "Status 1"
        assert renderer.state.current_status == "Status 2"
    
    def test_add_tool_call(self, renderer):
        renderer.add_tool_call("file_manager__read_file", args={"file_path": "/test.py"}, is_running=True)
        
        assert renderer.state._current_step is not None
        assert len(renderer.state._current_step.tool_calls) == 1
        tc = renderer.state._current_step.tool_calls[0]
        assert tc.name == "file_manager__read_file"
        assert tc.key_param == "/test.py"
        assert tc.is_running is True
    
    def test_update_tool_complete(self, renderer):
        renderer.add_tool_call("test_tool", is_running=True)
        renderer.update_tool_complete("test_tool")
        
        tc = renderer.state._current_step.tool_calls[0]
        assert tc.is_running is False
    
    def test_add_message(self, renderer):
        renderer.add_message("Short message")
        assert renderer.state._current_step is not None
        assert renderer.state._current_step.content == "Short message"
    
    def test_add_message_truncates_long(self, renderer):
        long_msg = "x" * 100
        renderer.add_message(long_msg)
        assert len(renderer.state._current_step.content) < 100
        assert renderer.state._current_step.content.endswith("...")
    
    def test_render_dynamic_panel_no_task(self, renderer):
        panel = renderer.render_dynamic_task_panel()
        assert panel is None
    
    def test_render_dynamic_panel_with_task(self, renderer):
        renderer.update_task_boundary({
            "TaskName": "Test Task",
            "Mode": "EXECUTION",
            "TaskStatus": "Running",
            "TaskSummary": "Summary"
        })
        
        panel = renderer.render_dynamic_task_panel()
        assert panel is not None
        assert "Test Task" in panel.title
    
    def test_on_notify_user_clears_state(self, renderer):
        renderer.update_task_boundary({
            "TaskName": "Test",
            "Mode": "PLANNING",
            "TaskStatus": "Status",
            "TaskSummary": "Summary"
        })
        
        renderer.on_notify_user()
        
        assert renderer.state.task_name == ""
        assert renderer.has_active_task() is False
    
    def test_finalize_step_on_status_change(self, renderer):
        # Add a tool (creates current step)
        renderer.add_tool_call("tool1", is_running=True)
        assert renderer.state._current_step is not None
        
        # Set initial task boundary
        renderer.update_task_boundary({
            "TaskName": "Test",
            "Mode": "PLANNING",
            "TaskStatus": "Status 1",
            "TaskSummary": "Summary"
        })
        
        # Add another tool
        renderer.add_tool_call("tool2", is_running=True)
        
        # Change status - should finalize current step
        renderer.update_task_boundary({
            "TaskName": "%SAME%",
            "Mode": "%SAME%",
            "TaskStatus": "Status 2",
            "TaskSummary": "%SAME%"
        })
        
        # Previous step should be finalized to recent_steps
        assert len(renderer.state.recent_steps) >= 1
    
    def test_flatten_step(self, renderer):
        step = AssistantStep(
            content="Message",
            tool_calls=[ToolCallInfo(name="test_tool", key_param="/path", is_running=False)]
        )
        items = renderer._flatten_step(step, is_current=False)
        assert len(items) == 2  # message + tool


class TestNotifyUIRenderer:
    """Test NotifyUIRenderer class."""
    
    @pytest.fixture
    def console(self):
        return Console(file=StringIO(), force_terminal=True, width=80)
    
    @pytest.fixture
    def renderer(self, console):
        return NotifyUIRenderer(console)
    
    def test_render_notification_not_blocked(self, renderer):
        result = {
            "success": True,
            "message": "Please review",
            "paths": ["/path/to/file.md"],
            "interrupt": False
        }
        
        blocked = renderer.render_notification(result)
        assert blocked is False
    
    def test_render_notification_blocked(self, renderer):
        result = {
            "success": True,
            "message": "Please approve",
            "paths": ["/path/to/plan.md"],
            "interrupt": True
        }
        
        blocked = renderer.render_notification(result)
        assert blocked is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
