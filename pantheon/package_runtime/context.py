"""Helpers for exporting and loading Pantheon package context."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

CONTEXT_ENV = "PANTHEON_CONTEXT"
DEFAULT_PACKAGES_SUBDIR = ".pantheon/packages"


def _default_json(value: Any):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _normalize_path(path: str | Path | None) -> str | None:
    if not path:
        return None
    return str(Path(path).expanduser().resolve())


def derive_packages_path(workdir: str | Path | None = None) -> str:
    """Return the canonical packages directory for a workdir."""

    if workdir:
        root = Path(workdir).expanduser().resolve()
    else:
        root = Path.cwd()
    return str((root / DEFAULT_PACKAGES_SUBDIR).resolve())


def build_context_payload(
    *,
    workdir: str | Path | None = None,
    context_variables: Mapping[str, Any] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict:
    """Construct a normalized context payload ready for export."""

    normalized_workdir = _normalize_path(workdir)
    payload: dict[str, Any] = {
        "workdir": normalized_workdir,
        "context_variables": dict(context_variables or {}),
    }
    if extras:
        payload.update(extras)
    return payload


def build_context_env(
    *,
    workdir: str | Path | None = None,
    context_variables: Mapping[str, Any] | None = None,
    extras: Mapping[str, Any] | None = None,
    base_env: Mapping[str, Any] | None = None,
) -> dict:
    """Return a copy of base_env with PANTHEON_CONTEXT exported."""

    payload = build_context_payload(
        workdir=workdir,
        context_variables=context_variables,
        extras=extras,
    )
    env: dict[str, Any] = dict(base_env or {})
    export_context(payload, env=env)
    return env


def export_context(payload: Mapping[str, Any], env: dict | None = None) -> dict:
    """Serialize payload into env (defaults to os.environ) for downstream use."""

    target = env if env is not None else os.environ
    normalized = {
        "workdir": _normalize_path(payload.get("workdir")),
        "context_variables": _filter_serializable_context(
            payload.get("context_variables") or {}
        ),
    }
    extras = {
        key: value
        for key, value in payload.items()
        if key not in ("workdir", "context_variables")
    }
    if extras:
        normalized.update(extras)

    serialized = json.dumps(normalized, default=_default_json)
    target[CONTEXT_ENV] = serialized
    return target


def load_context(default: Mapping[str, Any] | None = None) -> dict:
    """Load the serialized payload from env, returning a normalized dict."""

    baseline = {"workdir": None, "context_variables": {}}
    if default:
        baseline.update(default)
        if "context_variables" in default:
            baseline["context_variables"] = dict(default["context_variables"])

    raw = os.environ.get(CONTEXT_ENV)
    if not raw:
        return baseline

    try:
        data = json.loads(raw)
    except Exception:
        return baseline

    if not isinstance(data, Mapping):
        return baseline

    payload = dict(baseline)
    payload.update(data)
    payload["workdir"] = _normalize_path(payload.get("workdir"))
    payload["context_variables"] = dict(payload.get("context_variables") or {})
    return payload
 

def _filter_serializable_context(values: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only entries whose values can be JSON-serialized."""

    clean: dict[str, Any] = {}
    for key, value in dict(values).items():
        normalized_key = key if isinstance(key, str) else str(key)
        if _is_json_serializable(value):
            clean[normalized_key] = value
    return clean


def _is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value, default=_default_json)
        return True
    except (TypeError, ValueError):
        return False


__all__ = [
    "CONTEXT_ENV",
    "build_context_payload",
    "derive_packages_path",
    "export_context",
    "load_context",
]
