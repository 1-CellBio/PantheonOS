"""
File search utilities: glob and grep implementations.

This module provides fast file and content search capabilities using
external tools (fd, ripgrep) with Python fallbacks.
"""

from pathlib import Path
import os
import subprocess
import re
import shutil
from datetime import datetime
from typing import Optional

from pantheon.utils.log import logger


# ============================================================================
# Constants
# ============================================================================

GITIGNORE_PATTERNS = {
    ".git",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".venv",
    ".pytest_cache",
}


# ============================================================================
# Helper Functions
# ============================================================================


def resolve_path(workspace_root: Path, path: Optional[str]) -> Path:
    """Resolve relative or absolute path.

    Args:
        workspace_root: The workspace root directory.
        path: Path to resolve (can be None, relative, or absolute).

    Returns:
        Resolved Path object.
    """
    if path is None:
        return workspace_root
    return Path(path) if os.path.isabs(path) else workspace_root / path


def build_file_info(file_path: Path, workspace_root: Path) -> dict:
    """Build file information dictionary.

    Args:
        file_path: Path to the file.
        workspace_root: Workspace root for calculating relative paths.

    Returns:
        Dictionary with file metadata.
    """
    stat = file_path.stat()

    # Calculate relative path
    try:
        rel_path = str(file_path.relative_to(workspace_root))
    except ValueError:
        rel_path = str(file_path)

    return {
        "path": rel_path,
        "name": file_path.name,
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "type": "directory" if file_path.is_dir() else "file",
    }


def should_ignore(file_path: Path, base_path: Path) -> bool:
    """Check if file should be ignored by gitignore patterns.

    Args:
        file_path: Path to check.
        base_path: Base directory for relative path calculation.

    Returns:
        True if file should be ignored, False otherwise.
    """
    try:
        parts = file_path.relative_to(base_path).parts
        return any(p.startswith(".") or p in GITIGNORE_PATTERNS for p in parts)
    except ValueError:
        return False


# ============================================================================
# Glob Implementation
# ============================================================================


def run_fd(
    pattern: str, search_dir: Path, workspace_root: Path, respect_git_ignore: bool
) -> list[dict]:
    """Run fd tool to find files.

    Args:
        pattern: Glob pattern to match.
        search_dir: Directory to search in.
        workspace_root: Workspace root for relative paths.
        respect_git_ignore: Whether to respect .gitignore.

    Returns:
        List of file information dictionaries.

    Raises:
        Exception: If fd command fails.
    """
    cmd = ["fd", "--type", "f", "--glob", pattern]

    if not respect_git_ignore:
        cmd.extend(["--no-ignore", "--hidden"])

    cmd.append(str(search_dir))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode not in [0, 1]:
        raise Exception(f"fd command failed: {result.stderr}")

    # Parse output and build file list
    files = []
    for line in result.stdout.strip().split("\n"):
        if line:
            file_path = Path(line)
            if file_path.exists():
                files.append(build_file_info(file_path, workspace_root))

    return files


def run_glob_fallback(
    pattern: str, search_dir: Path, workspace_root: Path, respect_git_ignore: bool
) -> list[dict]:
    """Fallback glob implementation using Python pathlib.

    Args:
        pattern: Glob pattern to match.
        search_dir: Directory to search in.
        workspace_root: Workspace root for relative paths.
        respect_git_ignore: Whether to respect .gitignore.

    Returns:
        List of file information dictionaries.
    """
    files = []

    for file_path in search_dir.glob(pattern):
        # Filter gitignored files
        if respect_git_ignore and should_ignore(file_path, search_dir):
            continue

        if file_path.is_file():
            files.append(build_file_info(file_path, workspace_root))

    return sorted(files, key=lambda x: x["path"])


def glob_search(
    pattern: str,
    workspace_root: Path,
    path: Optional[str] = None,
    respect_git_ignore: bool = True,
) -> dict:
    """Find files matching glob patterns.

    Args:
        pattern: Glob pattern to match files.
        workspace_root: Workspace root directory.
        path: Directory to search from (default: workspace root).
        respect_git_ignore: Whether to respect .gitignore patterns.

    Returns:
        Dictionary with search results or error.
    """
    try:
        search_dir = resolve_path(workspace_root, path)

        if not search_dir.exists():
            return {
                "success": False,
                "error": f"Directory does not exist: {path or 'workspace root'}",
            }

        if not search_dir.is_dir():
            return {"success": False, "error": f"Path is not a directory: {path}"}

        # Prevent glob at root directory
        resolved = search_dir.resolve()
        if str(resolved) == "/":
            return {
                "success": False,
                "error": "Cannot glob at root directory. Please specify a more specific path.",
            }

        # Try fd first, fallback to Python glob
        try:
            if shutil.which("fd"):
                files = run_fd(pattern, search_dir, workspace_root, respect_git_ignore)
            else:
                raise FileNotFoundError("fd not available")
        except Exception as e:
            logger.debug(f"fd failed ({e}), using Python fallback")
            files = run_glob_fallback(
                pattern, search_dir, workspace_root, respect_git_ignore
            )

        return {
            "success": True,
            "files": files,
            "total": len(files),
            "pattern": pattern,
            "message": f"Found {len(files)} file(s) matching '{pattern}'",
        }

    except Exception as e:
        logger.error(f"glob_search failed: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# Grep Implementation
# ============================================================================


def run_ripgrep(
    pattern: str,
    search_path: Path,
    workspace_root: Path,
    file_pattern: Optional[str],
    context_lines: int,
    case_sensitive: bool,
    respect_git_ignore: bool,
) -> dict:
    """Run ripgrep tool to search file contents.

    Args:
        pattern: Regex pattern to search for.
        search_path: Path to search in.
        workspace_root: Workspace root for relative paths.
        file_pattern: Glob pattern to filter files.
        context_lines: Number of context lines before/after match.
        case_sensitive: Whether search is case-sensitive.
        respect_git_ignore: Whether to respect .gitignore.

    Returns:
        Dictionary with matches and files_matched count.

    Raises:
        Exception: If ripgrep command fails.
    """
    import json as json_module

    cmd = ["rg", "--json", pattern]

    if not case_sensitive:
        cmd.append("--ignore-case")
    if context_lines > 0:
        cmd.extend(["-C", str(context_lines)])
    if not respect_git_ignore:
        cmd.extend(["--no-ignore", "--hidden"])
    if file_pattern:
        cmd.extend(["--glob", file_pattern])

    cmd.append(str(search_path))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode not in [0, 1]:
        raise Exception(f"ripgrep command failed: {result.stderr}")

    # Parse JSON output
    matches = []
    files_matched = set()

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue

        try:
            data = json_module.loads(line)
            if data.get("type") == "match":
                match_data = data["data"]
                file_path = Path(match_data["path"]["text"])

                # Calculate relative path
                try:
                    rel_path = str(file_path.relative_to(workspace_root))
                except ValueError:
                    rel_path = str(file_path)

                files_matched.add(rel_path)

                # Extract match information
                submatches = match_data.get("submatches", [])
                matches.append(
                    {
                        "file": rel_path,
                        "line_number": match_data["line_number"],
                        "line_content": match_data["lines"]["text"].rstrip("\n"),
                        "context_before": [],
                        "context_after": [],
                        "column": submatches[0]["start"] + 1 if submatches else 1,
                    }
                )

        except (json_module.JSONDecodeError, KeyError):
            continue

    return {"matches": matches, "files_matched": len(files_matched)}


def run_grep_fallback(
    pattern: str,
    search_path: Path,
    workspace_root: Path,
    file_pattern: Optional[str],
    context_lines: int,
    case_sensitive: bool,
    respect_git_ignore: bool,
) -> dict:
    """Fallback grep implementation using Python re module.

    Args:
        pattern: Regex pattern to search for.
        search_path: Path to search in.
        workspace_root: Workspace root for relative paths.
        file_pattern: Glob pattern to filter files.
        context_lines: Number of context lines before/after match.
        case_sensitive: Whether search is case-sensitive.
        respect_git_ignore: Whether to respect .gitignore.

    Returns:
        Dictionary with matches and files_matched count.

    Raises:
        ValueError: If regex pattern is invalid.
    """
    # Compile regex pattern
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}")

    # Get list of files to search
    if search_path.is_file():
        files_to_search = [search_path]
    elif file_pattern:
        files_to_search = list(search_path.glob(file_pattern))
    else:
        files_to_search = list(search_path.rglob("*"))

    matches = []
    files_matched = set()

    for file_path in files_to_search:
        if not file_path.is_file():
            continue

        # Filter gitignored files
        base = search_path if search_path.is_dir() else search_path.parent
        if respect_git_ignore and should_ignore(file_path, base):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                if match := regex.search(line):
                    # Calculate relative path
                    try:
                        rel_path = str(file_path.relative_to(workspace_root))
                    except ValueError:
                        rel_path = str(file_path)

                    files_matched.add(rel_path)

                    # Get context lines
                    context_before = []
                    context_after = []

                    if context_lines > 0:
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)

                        context_before = [
                            lines[j].rstrip("\n") for j in range(start, i)
                        ]
                        context_after = [
                            lines[j].rstrip("\n") for j in range(i + 1, end)
                        ]

                    matches.append(
                        {
                            "file": rel_path,
                            "line_number": i + 1,  # 1-indexed
                            "line_content": line.rstrip("\n"),
                            "context_before": context_before,
                            "context_after": context_after,
                            "column": match.start() + 1,  # 1-indexed
                        }
                    )

        except (UnicodeDecodeError, PermissionError):
            # Skip binary files and permission-denied files
            continue
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            continue

    return {"matches": matches, "files_matched": len(files_matched)}


def grep_search(
    pattern: str,
    workspace_root: Path,
    path: Optional[str] = None,
    file_pattern: Optional[str] = None,
    context_lines: int = 0,
    case_sensitive: bool = False,
    respect_git_ignore: bool = True,
) -> dict:
    """Search for text patterns within file contents.

    Args:
        pattern: Text or regex pattern to search for.
        workspace_root: Workspace root directory.
        path: Directory or file to search (default: workspace root).
        file_pattern: Glob pattern to filter files.
        context_lines: Number of context lines before/after each match.
        case_sensitive: Whether search is case-sensitive.
        respect_git_ignore: Whether to respect .gitignore patterns.

    Returns:
        Dictionary with search results or error.
    """
    try:
        search_path = resolve_path(workspace_root, path)

        if not search_path.exists():
            return {
                "success": False,
                "error": f"Path does not exist: {path or 'workspace root'}",
            }

        # Prevent grep at root directory
        resolved = search_path.resolve()
        if str(resolved) == "/":
            return {
                "success": False,
                "error": "Cannot grep at root directory. Please specify a more specific path.",
            }

        # Try ripgrep first, fallback to Python re
        try:
            if shutil.which("rg"):
                result = run_ripgrep(
                    pattern,
                    search_path,
                    workspace_root,
                    file_pattern,
                    context_lines,
                    case_sensitive,
                    respect_git_ignore,
                )
            else:
                raise FileNotFoundError("ripgrep not available")
        except Exception as e:
            logger.warning(f"ripgrep failed ({e}), using Python fallback")
            result = run_grep_fallback(
                pattern,
                search_path,
                workspace_root,
                file_pattern,
                context_lines,
                case_sensitive,
                respect_git_ignore,
            )

        return {
            "success": True,
            "matches": result["matches"],
            "total_matches": len(result["matches"]),
            "files_matched": result["files_matched"],
            "pattern": pattern,
            "message": f"Found {len(result['matches'])} match(es) in {result['files_matched']} file(s)",
        }

    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"grep_search failed: {e}")
        return {"success": False, "error": str(e)}
