"""CodeToolSet - Code navigation and exploration tools.

Provides tools for viewing code structure and extracting specific code items
using tree-sitter for multi-language AST parsing.
"""

from pathlib import Path

from ...toolset import ToolSet, tool
from .tree_sitter_parser import get_file_outline, get_code_item, detect_language


class CodeToolSet(ToolSet):
    """ToolSet for code navigation and exploration.
    
    Provides:
    - view_file_outline: Get structured overview of classes/functions
    - view_code_item: Extract source code for specific symbols
    
    Requires: pip install 'pantheon-agents[toolsets]'
    
    Args:
        name: Name of this toolset instance.
        workspace_path: Root directory for file operations.
    """
    
    def __init__(self, name: str, workspace_path: str | Path | None = None, **kwargs):
        super().__init__(name, **kwargs)
        if workspace_path is None:
            workspace_path = Path.cwd()
        self.workspace_path = Path(workspace_path)
    
    @tool
    async def view_file_outline(self, file_path: str) -> dict:
        """Get a structured outline of classes and functions in a file.
        
        Returns the "skeleton" of a source file: all top-level classes and
        functions with their line ranges, signatures, and nested members.
        Useful for understanding large files without reading all the code.
        
        Args:
            file_path: Path to the source file (relative to workspace or absolute).
                      Supports: .py, .js, .ts, .jsx, .tsx
        
        Returns:
            dict: {
                "success": bool,
                "file": str,
                "language": str,        # "python", "javascript", etc.
                "total_lines": int,
                "symbols": [
                    {
                        "name": str,
                        "kind": str,    # "class", "function", "method"
                        "start_line": int,
                        "end_line": int,
                        "signature": str,
                        "docstring": str,  # First 100 chars
                        "children": [...]  # Nested symbols
                    }
                ]
            }
        
        Examples:
            # View outline of a Python file
            outline = await view_file_outline("src/utils.py")
            # outline["symbols"] → [{name: "MyClass", kind: "class", ...}]
            
            # View outline of a JavaScript file
            outline = await view_file_outline("lib/index.js")
        """
        # Resolve path
        if Path(file_path).is_absolute():
            target_path = Path(file_path)
        else:
            target_path = self.workspace_path / file_path
        
        return get_file_outline(target_path)
    
    @tool
    async def view_code_item(self, file_path: str, node_path: str) -> dict:
        """View the source code of a specific class, function, or method.
        
        Use this after view_file_outline to drill down into specific symbols.
        Useful for reading just the code you need without loading the entire file.
        
        Args:
            file_path: Path to the source file (relative to workspace or absolute).
            
            node_path: Qualified name of the symbol using dot notation.
                      Examples:
                      - "MyClass" → entire class
                      - "MyClass.my_method" → specific method
                      - "helper_function" → top-level function
        
        Returns:
            dict: {
                "success": bool,
                "name": str,
                "kind": str,        # "class", "function", "method"
                "start_line": int,
                "end_line": int,
                "source": str       # The actual source code
            }
        
        Examples:
            # Get a specific method
            code = await view_code_item("src/utils.py", "DataProcessor.validate")
            print(code["source"])
            
            # Get a class
            code = await view_code_item("lib/api.js", "ApiClient")
        """
        # Resolve path
        if Path(file_path).is_absolute():
            target_path = Path(file_path)
        else:
            target_path = self.workspace_path / file_path
        
        return get_code_item(target_path, node_path)
