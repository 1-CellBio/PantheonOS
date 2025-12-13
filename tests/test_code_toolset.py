"""Tests for CodeToolSet - code navigation tools using tree-sitter."""

import pytest
from tempfile import TemporaryDirectory
from pathlib import Path

from pantheon.toolsets.code import CodeToolSet


@pytest.fixture
def temp_toolset():
    """Create a CodeToolSet with a temporary directory."""
    with TemporaryDirectory() as temp_dir:
        yield CodeToolSet("code", temp_dir)


@pytest.fixture
def python_file(temp_toolset):
    """Create a sample Python file for testing."""
    content = '''"""Module docstring."""

class DataProcessor:
    """Process data efficiently."""
    
    def __init__(self, config):
        """Initialize processor."""
        self.config = config
    
    def validate(self, data):
        """Validate input data."""
        if not data:
            return False
        return True
    
    async def process(self, data):
        """Process data asynchronously."""
        return data


def helper_function(x, y):
    """A helper function."""
    return x + y


class AnotherClass:
    pass
'''
    file_path = temp_toolset.workspace_path / "sample.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def js_file(temp_toolset):
    """Create a sample JavaScript file for testing."""
    content = '''class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }
    
    async fetch(endpoint) {
        return await fetch(this.baseUrl + endpoint);
    }
}

function formatData(data) {
    return JSON.stringify(data);
}

const processItems = (items) => {
    return items.map(i => i.id);
};
'''
    file_path = temp_toolset.workspace_path / "api.js"
    file_path.write_text(content)
    return file_path


class TestViewFileOutline:
    """Tests for view_file_outline tool."""
    
    async def test_python_outline(self, temp_toolset, python_file):
        """Test outline extraction from Python file."""
        result = await temp_toolset.view_file_outline("sample.py")
        
        assert result["success"] is True
        assert result["language"] == "python"
        assert result["total_lines"] > 0
        assert len(result["symbols"]) >= 3  # 2 classes + 1 function
        
        # Check DataProcessor class
        dp = next(s for s in result["symbols"] if s["name"] == "DataProcessor")
        assert dp["kind"] == "class"
        assert "children" in dp
        assert len(dp["children"]) >= 3  # __init__, validate, process
        
        # Check method
        validate = next(c for c in dp["children"] if c["name"] == "validate")
        assert validate["kind"] == "method"
        assert validate["start_line"] < validate["end_line"]
    
    async def test_javascript_outline(self, temp_toolset, js_file):
        """Test outline extraction from JavaScript file."""
        result = await temp_toolset.view_file_outline("api.js")
        
        assert result["success"] is True
        assert result["language"] == "javascript"
        assert len(result["symbols"]) >= 2  # class + functions
        
        # Check ApiClient class
        api = next((s for s in result["symbols"] if s["name"] == "ApiClient"), None)
        assert api is not None
        assert api["kind"] == "class"
    
    async def test_unsupported_file(self, temp_toolset):
        """Test error handling for unsupported file types."""
        file_path = temp_toolset.workspace_path / "data.json"
        file_path.write_text('{"key": "value"}')
        
        result = await temp_toolset.view_file_outline("data.json")
        
        assert result["success"] is False
        assert "Unsupported file type" in result["error"]
    
    async def test_nonexistent_file(self, temp_toolset):
        """Test error handling for missing files."""
        result = await temp_toolset.view_file_outline("nonexistent.py")
        
        assert result["success"] is False
        assert "does not exist" in result["error"]


class TestViewCodeItem:
    """Tests for view_code_item tool."""
    
    async def test_get_class(self, temp_toolset, python_file):
        """Test extracting a class."""
        result = await temp_toolset.view_code_item("sample.py", "DataProcessor")
        
        assert result["success"] is True
        assert result["name"] == "DataProcessor"
        assert result["kind"] == "class"
        assert "class DataProcessor" in result["source"]
        assert "def validate" in result["source"]
    
    async def test_get_method(self, temp_toolset, python_file):
        """Test extracting a specific method."""
        result = await temp_toolset.view_code_item("sample.py", "DataProcessor.validate")
        
        assert result["success"] is True
        assert result["name"] == "validate"
        assert result["kind"] == "method"
        assert "def validate" in result["source"]
        assert "class DataProcessor" not in result["source"]
    
    async def test_get_function(self, temp_toolset, python_file):
        """Test extracting a top-level function."""
        result = await temp_toolset.view_code_item("sample.py", "helper_function")
        
        assert result["success"] is True
        assert result["name"] == "helper_function"
        assert result["kind"] == "function"
        assert "def helper_function" in result["source"]
    
    async def test_symbol_not_found(self, temp_toolset, python_file):
        """Test error handling for missing symbols."""
        result = await temp_toolset.view_code_item("sample.py", "NonExistent")
        
        assert result["success"] is False
        assert "not found" in result["error"]
    
    async def test_nested_symbol_not_found(self, temp_toolset, python_file):
        """Test error handling for missing nested symbols."""
        result = await temp_toolset.view_code_item("sample.py", "DataProcessor.nonexistent")
        
        assert result["success"] is False
        assert "not found" in result["error"]
