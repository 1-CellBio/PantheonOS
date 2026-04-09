from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_VALID_WORKSPACE_MODES = {"isolated", "project"}
_TRUSTED_TEMP_UPLOAD_PREFIXES = ("pantheon",)


def slugify_project_name(name: str | None) -> str:
    """Convert a project name into a filesystem-safe slug."""
    if not name or not name.strip():
        return "default"
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower().strip())
    slug = slug.strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or "default"


def _normalize_optional_path(path_value: Any) -> Path | None:
    if not isinstance(path_value, str):
        return None
    value = path_value.strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _workspace_root(pantheon_dir: str | Path) -> Path:
    return Path(pantheon_dir).resolve() / "workspaces"


def is_trusted_temp_upload_path(path_value: str | Path) -> bool:
    """Return True for Pantheon-managed transient upload sources under tempdir."""
    try:
        candidate = Path(path_value).expanduser().resolve()
    except Exception:
        return False

    temp_roots: list[Path] = []
    for raw_root in (tempfile.gettempdir(), "/tmp"):
        try:
            root = Path(raw_root).resolve()
        except Exception:
            continue
        if root not in temp_roots:
            temp_roots.append(root)

    relative = None
    for root in temp_roots:
        try:
            relative = candidate.relative_to(root)
            break
        except ValueError:
            continue

    if relative is None:
        return False

    parts = tuple(part.lower() for part in relative.parts)
    if not parts:
        return False

    return any(
        any(part.startswith(prefix) for prefix in _TRUSTED_TEMP_UPLOAD_PREFIXES)
        for part in parts
    )


def find_chat_workspace(pantheon_dir: str | Path, chat_id: str) -> Path | None:
    """Locate an existing chat workspace in either nested or flat layouts."""
    ws_root = _workspace_root(pantheon_dir)

    flat = ws_root / chat_id
    if flat.is_dir():
        return flat.resolve()

    if ws_root.exists():
        for slug_dir in ws_root.iterdir():
            if not slug_dir.is_dir():
                continue
            nested = slug_dir / chat_id
            if nested.is_dir():
                return nested.resolve()
    return None


def load_project_metadata(pantheon_dir: str | Path, chat_id: str) -> dict[str, Any]:
    """Read project metadata for a chat directly from persisted memory if present.

    Supports both JSONL format (extra_data in .meta.json) and legacy JSON format.
    """
    pantheon_root = Path(pantheon_dir).resolve()

    # JSONL format: extra_data stored in <chat_id>.meta.json
    meta_file = pantheon_root / "memory" / f"{chat_id}.meta.json"
    if meta_file.exists():
        try:
            payload = json.loads(meta_file.read_text(encoding="utf-8"))
            project = (payload.get("extra_data") or {}).get("project")
            if isinstance(project, dict):
                return dict(project)
        except Exception:
            pass

    # Legacy JSON format: everything in <chat_id>.json
    memory_file = pantheon_root / "memory" / f"{chat_id}.json"
    if memory_file.exists():
        try:
            payload = json.loads(memory_file.read_text(encoding="utf-8"))
            extra_data = payload.get("extra_data")
            if isinstance(extra_data, dict):
                project = extra_data.get("project")
                if isinstance(project, dict):
                    return dict(project)
        except Exception:
            pass

    return {}


@dataclass(frozen=True)
class WorkspaceLayout:
    chat_id: str
    project_name: str | None
    project_slug: str
    workspace_mode: str
    chat_workspace_path: Path
    project_workspace_path: Path
    session_workspace_path: Path
    upload_workspace_path: Path
    session_upload_path: Path
    upload_files_path: Path
    workspace_override_path: Path | None
    effective_workspace_path: Path


def build_workspace_views(layout: WorkspaceLayout) -> dict[str, dict[str, str]]:
    """Return UI-facing roots for global/project and isolated/session views."""
    return {
        "global": {
            "root": str(layout.project_workspace_path),
            "workspace_scope": "project",
        },
        "isolated": {
            "root": str(layout.session_workspace_path),
            "workspace_scope": "session",
        },
    }


def build_upload_attachment_metadata(
    layout: WorkspaceLayout,
    path_value: str | Path,
) -> dict[str, str] | None:
    """Describe an uploaded file path for UI and LLM display."""
    try:
        candidate = Path(path_value).expanduser().resolve()
    except Exception:
        return None

    scope = None
    workspace_view = None
    upload_root = None

    project_upload_root = (layout.project_workspace_path / ".uploaded_files").resolve()
    session_upload_root = (layout.session_workspace_path / ".uploaded_files").resolve()

    if candidate == project_upload_root or candidate.is_relative_to(project_upload_root):
        scope = "project"
        workspace_view = "global"
        upload_root = project_upload_root
    elif candidate == session_upload_root or candidate.is_relative_to(session_upload_root):
        scope = "session"
        workspace_view = "isolated"
        upload_root = session_upload_root
    else:
        return None

    try:
        relative = candidate.relative_to(upload_root)
    except ValueError:
        relative = Path()

    if str(relative) == ".":
        virtual_path = ".uploaded_files"
    elif not relative.parts:
        virtual_path = ".uploaded_files"
    else:
        virtual_path = str(Path(".uploaded_files") / relative)

    scope_label = "项目上传区" if scope == "project" else "会话上传区"
    return {
        "name": candidate.name,
        "virtual_path": virtual_path,
        "workspace_scope": scope,
        "workspace_view": workspace_view,
        "absolute_path": str(candidate),
        "display_path": f"{scope_label} · {virtual_path}",
        "scope_label": scope_label,
    }


def resolve_upload_attachment_path(
    layout: WorkspaceLayout,
    virtual_path: str,
    *,
    workspace_scope: str | None = None,
    workspace_view: str | None = None,
) -> Path | None:
    """Resolve a virtual upload path into its concrete filesystem path."""
    if not isinstance(virtual_path, str) or not virtual_path.strip():
        return None

    candidate = Path(virtual_path.strip())
    if candidate.parts[:1] != (".uploaded_files",):
        return None

    scope = workspace_scope
    if scope not in {"project", "session"}:
        if workspace_view == "global":
            scope = "project"
        elif workspace_view == "isolated":
            scope = "session"
        else:
            return None

    base_root = (
        layout.project_workspace_path if scope == "project" else layout.session_workspace_path
    )
    return (base_root / candidate).resolve()


def resolve_workspace_layout(
    pantheon_dir: str | Path,
    chat_id: str,
    project: dict[str, Any] | None = None,
) -> WorkspaceLayout:
    """Resolve the authoritative workspace layout for a chat."""
    pantheon_root = Path(pantheon_dir).resolve()
    project_data = dict(project or load_project_metadata(pantheon_root, chat_id))
    existing_chat_workspace = find_chat_workspace(pantheon_root, chat_id)

    inferred_slug = None
    inferred_project_workspace = None
    if existing_chat_workspace is not None:
        ws_root = pantheon_root / "workspaces"
        try:
            relative = existing_chat_workspace.relative_to(ws_root)
            if len(relative.parts) >= 2:
                inferred_slug = relative.parts[0]
                inferred_project_workspace = ws_root / inferred_slug
        except ValueError:
            pass

    project_name = project_data.get("name")
    project_slug = (
        project_data.get("slug")
        or inferred_slug
        or slugify_project_name(project_name)
    )

    workspace_mode = project_data.get("workspace_mode")
    if workspace_mode not in _VALID_WORKSPACE_MODES:
        workspace_mode = "isolated"

    workspace_root = _workspace_root(pantheon_root)
    default_project_workspace = workspace_root / project_slug
    project_workspace_path = _normalize_optional_path(
        project_data.get("project_workspace_path")
    )
    if project_workspace_path is None:
        project_workspace_path = inferred_project_workspace or default_project_workspace
    default_chat_workspace = project_workspace_path / chat_id
    chat_workspace_path = default_chat_workspace
    if (
        existing_chat_workspace is not None
        and existing_chat_workspace.parent == workspace_root
        and not project_data
    ):
        chat_workspace_path = existing_chat_workspace

    workspace_override_path = _normalize_optional_path(
        project_data.get("workspace_override_path")
    )
    stored_workspace_path = _normalize_optional_path(project_data.get("workspace_path"))
    if (
        workspace_override_path is None
        and stored_workspace_path is not None
        and stored_workspace_path not in {chat_workspace_path, project_workspace_path}
    ):
        workspace_override_path = stored_workspace_path

    session_workspace_path = (
        workspace_override_path
        if workspace_override_path is not None
        else chat_workspace_path
    )
    upload_workspace_path = (
        project_workspace_path if workspace_mode == "project" else session_workspace_path
    )
    session_upload_path = session_workspace_path / ".uploaded_files"
    upload_files_path = upload_workspace_path / ".uploaded_files"

    return WorkspaceLayout(
        chat_id=chat_id,
        project_name=project_name if isinstance(project_name, str) else None,
        project_slug=project_slug,
        workspace_mode=workspace_mode,
        chat_workspace_path=chat_workspace_path.resolve(),
        project_workspace_path=project_workspace_path.resolve(),
        session_workspace_path=session_workspace_path.resolve(),
        upload_workspace_path=upload_workspace_path.resolve(),
        session_upload_path=session_upload_path.resolve(),
        upload_files_path=upload_files_path.resolve(),
        workspace_override_path=(
            workspace_override_path.resolve()
            if workspace_override_path is not None
            else None
        ),
        effective_workspace_path=session_workspace_path.resolve(),
    )


def normalize_project_metadata(
    pantheon_dir: str | Path,
    chat_id: str,
    project: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return project metadata normalized to the shared workspace rules."""
    project_data = dict(project or {})
    layout = resolve_workspace_layout(pantheon_dir, chat_id, project_data)

    project_data["slug"] = layout.project_slug
    project_data["workspace_mode"] = layout.workspace_mode
    project_data["chat_workspace_path"] = str(layout.chat_workspace_path)
    project_data["project_workspace_path"] = str(layout.project_workspace_path)
    project_data["workspace_path"] = str(layout.session_workspace_path)
    project_data["upload_workspace_path"] = str(layout.upload_workspace_path)
    if layout.workspace_override_path is not None:
        project_data["workspace_override_path"] = str(layout.workspace_override_path)
    else:
        project_data.pop("workspace_override_path", None)

    return project_data


def ensure_workspace_layout(
    layout: WorkspaceLayout,
    *,
    create_chat_workspace: bool = False,
    create_project_workspace: bool = False,
    create_upload_workspace: bool = False,
    create_effective_workspace: bool = False,
    create_attachment_bridge: bool = False,
) -> None:
    """Create the requested workspace directories for a resolved layout."""
    if create_project_workspace:
        layout.project_workspace_path.mkdir(parents=True, exist_ok=True)
    if create_chat_workspace:
        layout.chat_workspace_path.mkdir(parents=True, exist_ok=True)
    if create_upload_workspace:
        layout.upload_workspace_path.mkdir(parents=True, exist_ok=True)
        layout.upload_files_path.mkdir(parents=True, exist_ok=True)
    if create_effective_workspace:
        layout.effective_workspace_path.mkdir(parents=True, exist_ok=True)
    if create_attachment_bridge:
        layout.session_workspace_path.mkdir(parents=True, exist_ok=True)
        layout.upload_workspace_path.mkdir(parents=True, exist_ok=True)
        layout.upload_files_path.mkdir(parents=True, exist_ok=True)
