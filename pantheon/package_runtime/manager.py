"""Runtime package discovery and invocation utilities."""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable

from pantheon.toolset import ToolSet
from pantheon.utils.log import logger


@dataclass
class PackageMethod:
    """Metadata for an exported package method."""

    name: str
    doc: str | None
    callable: Callable
    accepts_context: bool
    is_async: bool
    owner: str
    signature: str


@dataclass
class PackageRecord:
    """Tracked metadata for a package directory."""

    name: str
    path: Path | None
    mtime: float = 0.0
    module_name: str | None = None
    module: ModuleType | None = None
    classes: dict[str, Any] = field(default_factory=dict)
    class_defs: dict[str, type[Any]] = field(default_factory=dict)
    methods: dict[str, PackageMethod] = field(default_factory=dict)
    description: str | None = None
    status: str = "uninitialized"
    last_error: str | None = None
    last_loaded: float | None = None
    origin: str = "user"


class PackageManager:
    """Manages local Pantheon packages by scanning the workspace."""

    def __init__(self, packages_path: str | Path):
        self.packages_path = Path(packages_path).expanduser().resolve()
        self.packages_path.mkdir(parents=True, exist_ok=True)
        self._namespace = f"pantheon_packages_{hash(self.packages_path)}"
        self._packages: dict[str, PackageRecord] = {}
        self._lock = threading.Lock()
        self._system_names: list[str] = []
        self._system_sources: dict[str, type[ToolSet]] = {}
        self._load_system_toolsets()
        logger.debug(
            f"PackageManager initialized at {self.packages_path} "
            f"(system packages: {len(self._system_names)})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self) -> list[str]:
        """Return sorted list of package directory names."""

        names: list[str] = []
        if not self.packages_path.exists():
            return names
        for entry in sorted(self.packages_path.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                names.append(entry.name)
        return names

    def list_packages(self) -> list[dict]:
        """Return high-level metadata for all packages."""

        results = []
        all_names = sorted(set(self.discover() + self._system_names))
        for name in all_names:
            record = self._ensure_loaded(name)
            results.append(
                {
                    "name": name,
                    "status": record.status,
                    "description": record.description,
                    "methods": sorted(record.methods.keys()),
                    "path": str(record.path) if record.path else None,
                    "last_error": record.last_error,
                    "origin": record.origin,
                }
            )
        return results

    def describe_package(self, name: str) -> dict:
        """Return detailed metadata for a package."""

        record = self._ensure_loaded(name)
        methods = [
            {
                "name": method.name,
                "doc": method.doc,
                "owner": method.owner,
                "signature": method.signature,
                "async": method.is_async,
                "accepts_context": method.accepts_context,
            }
            for method in record.methods.values()
        ]
        class_names = set(record.classes.keys()) | set(record.class_defs.keys())
        return {
            "success": record.status == "ready",
            "package": {
                "name": record.name,
                "status": record.status,
                "description": record.description,
                "path": str(record.path) if record.path else None,
                "classes": sorted(class_names),
                "methods": methods,
                "last_loaded": record.last_loaded,
                "last_error": record.last_error,
                "origin": record.origin,
            },
        }

    def package_status(self, name: str) -> dict:
        record = self._ensure_loaded(name)
        return {
            "name": name,
            "status": record.status,
            "last_loaded": record.last_loaded,
            "last_error": record.last_error,
            "origin": record.origin,
        }

    def reload_package(self, name: str | None = None) -> dict:
        """Reload specific package or all packages."""

        if name is None:
            reloaded = {}
            for pkg_name in self.discover():
                reloaded[pkg_name] = self._load_package(pkg_name)
            return {"success": True, "reloaded": list(reloaded.keys())}
        record = self._packages.get(name)
        if record and record.origin == "system":
            return self._reload_system_package(name)
        self._load_package(name)
        return {"success": True, "reloaded": [name]}

    def call(
        self,
        package_name: str,
        method_name: str,
        *args,
        context_variables: dict | None = None,
        **kwargs,
    ) -> Any:
        """Invoke a package method and return the raw result."""

        record = self._ensure_loaded(package_name)
        if method_name not in record.methods:
            raise AttributeError(
                f"Package '{package_name}' has no exported method '{method_name}'"
            )
        method = record.methods[method_name]
        call_kwargs = dict(kwargs)
        if method.accepts_context and "context_variables" not in call_kwargs:
            call_kwargs["context_variables"] = context_variables or {}
        arg_preview = ", ".join(self._shorten_args(args, call_kwargs))
        logger.debug(
            f"Calling package '{package_name}.{method_name}' "
            f"with args: {arg_preview or 'none'}"
        )
        try:
            result = method.callable(*args, **call_kwargs)
        except Exception:
            logger.exception(
                f"Package '{package_name}.{method_name}' raised an exception"
            )
            raise
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self, name: str) -> PackageRecord:
        with self._lock:
            record = self._packages.get(name)
            if record and record.origin == "system":
                return record
            path = self.packages_path / name
            if not path.exists():
                raise FileNotFoundError(f"Package '{name}' not found at {path}")
            latest_mtime = self._calculate_latest_mtime(path)
            if record is None or latest_mtime > (record.mtime or 0):
                record = self._load_package(name)
            return record

    def _load_package(self, name: str) -> PackageRecord:
        path = self.packages_path / name
        entry_file = path / "__init__.py"
        module_suffix = "__init__"
        if not entry_file.exists():
            candidates = sorted(path.glob("*.py"))
            if candidates:
                entry_file = candidates[0]
                module_suffix = entry_file.stem
        logger.debug(f"Loading package '{name}' from {path}")
        record = PackageRecord(name=name, path=path)
        self._packages[name] = record

        if not entry_file.exists():
            record.status = "missing"
            record.last_error = "No Python entry file found"
            logger.warning(
                f"Package '{name}' is missing an entry file under {path}; skipping"
            )
            return record

        if module_suffix == "__init__":
            module_name = f"{self._namespace}.{name}"
        else:
            module_name = f"{self._namespace}.{name}.{module_suffix}"

        if module_name in sys.modules:
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(
            module_name,
            entry_file,
            submodule_search_locations=[str(path)],
        )
        if spec is None or spec.loader is None:
            record.status = "error"
            record.last_error = f"Unable to create spec for package '{name}'"
            logger.error(
                f"Failed to create import spec for package '{name}' "
                f"(entry={entry_file})"
            )
            return record

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - surfaced to users
            record.status = "error"
            record.last_error = str(exc)
            logger.exception(f"Error while importing package '{name}'")
            return record

        record.module = module
        record.module_name = module_name
        record.description = inspect.getdoc(module)

        class_defs = self._discover_class_defs(module)
        record.class_defs = class_defs
        record.classes = {}

        methods = self._collect_methods(name, class_defs)
        record.methods = methods
        record.mtime = self._calculate_latest_mtime(path)
        record.last_loaded = time.time()
        record.status = "ready" if methods else "empty"
        if not record.description:
            record.description = self._select_fallback_description(class_defs)
        record.last_error = None
        logger.info(
            f"Package '{name}' loaded with {len(class_defs)} class(es) "
            f"and {len(methods)} method(s); status={record.status}"
        )
        return record

    def _discover_class_defs(
        self, module: ModuleType
    ) -> dict[str, type[Any]]:
        class_defs: dict[str, type[Any]] = {}
        for cls_name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if cls_name.startswith("_"):
                continue
            class_defs[cls_name] = cls
        return class_defs

    def _collect_methods(
        self, package_name: str, class_defs: dict[str, type[Any]]
    ) -> dict[str, PackageMethod]:
        methods: dict[str, PackageMethod] = {}
        for cls_name, cls in class_defs.items():
            methods.update(
                self._collect_class_methods(package_name, cls_name, cls)
            )
        return methods

    def _collect_class_methods(
        self,
        package_name: str,
        class_name: str,
        cls: type[Any],
    ) -> dict[str, PackageMethod]:
        methods: dict[str, PackageMethod] = {}
        is_toolset = issubclass(cls, ToolSet)
        for attr_name, func in inspect.getmembers(cls, inspect.isfunction):
            if not self._should_export_function(attr_name, func, is_toolset):
                continue
            doc = inspect.getdoc(func)
            if not doc and not is_toolset:
                continue
            signature = self._safe_signature(func)
            accepts_context = (
                True
                if is_toolset
                else "context_variables" in signature.parameters
            )
            methods[attr_name] = PackageMethod(
                name=attr_name,
                doc=doc,
                callable=self._make_lazy_callable(
                    package_name, class_name, attr_name
                ),
                accepts_context=accepts_context,
                is_async=inspect.iscoroutinefunction(func),
                owner=class_name,
                signature=str(signature),
            )
        return methods

    def _should_export_function(
        self,
        name: str,
        func: Callable,
        is_toolset: bool,
    ) -> bool:
        if is_toolset:
            if name == "list_tools":
                return False
            if getattr(func, "_exclude", False):
                return False
            return getattr(func, "_is_tool", False)
        return not name.startswith("_")

    def _make_lazy_callable(
        self,
        package_name: str,
        class_name: str,
        method_name: str,
    ) -> Callable:
        def _call(*args, **kwargs):
            method = self._get_bound_method(package_name, class_name, method_name)
            return method(*args, **kwargs)

        return _call

    def _get_bound_method(
        self, package_name: str, class_name: str, method_name: str
    ) -> Callable:
        instance = self._ensure_instance(package_name, class_name)
        method = getattr(instance, method_name, None)
        if method is None:
            raise AttributeError(
                f"Package '{package_name}' has no method '{method_name}' in class '{class_name}'"
            )
        return method

    def _ensure_instance(self, package_name: str, class_name: str) -> Any:
        record = self._packages.get(package_name)
        if record is None:
            raise KeyError(f"Package '{package_name}' not loaded")

        instance = record.classes.get(class_name)
        if instance is not None:
            return instance

        with self._lock:
            record = self._packages.get(package_name)
            if record is None:
                raise KeyError(f"Package '{package_name}' not loaded")
            instance = record.classes.get(class_name)
            if instance is not None:
                return instance
            cls = record.class_defs.get(class_name)
            if cls is None:
                raise KeyError(
                    f"Class '{class_name}' not registered for package '{package_name}'"
                )
            try:
                if issubclass(cls, ToolSet):
                    identifier = (
                        package_name
                        if record.origin == "system"
                        else f"{package_name}.{class_name}"
                    )
                    instance = cls(identifier)
                else:
                    instance = cls()
            except Exception as exc:
                record.status = "error"
                record.last_error = str(exc)
                logger.exception(
                    f"Failed to instantiate class '{class_name}' for package '{package_name}'"
                )
                raise
            record.classes[class_name] = instance
        return instance

    def _select_fallback_description(self, class_defs: dict[str, type[Any]]) -> str | None:
        for cls in class_defs.values():
            doc = inspect.getdoc(cls)
            if doc:
                return doc
        return None

    def _calculate_latest_mtime(self, path: Path) -> float:
        if path is None or not path.exists():
            return 0.0
        latest = path.stat().st_mtime
        for file in path.rglob("*.py"):
            latest = max(latest, file.stat().st_mtime)
        return latest

    def _safe_signature(self, func: Callable) -> inspect.Signature:
        try:
            return inspect.signature(func)
        except (ValueError, TypeError):  # pragma: no cover - fallback
            return inspect.Signature()

    def _shorten_args(self, args: Iterable[Any], kwargs: Dict[str, Any]) -> list[str]:
        parts = [self._short_repr(arg) for arg in args]
        parts.extend(f"{k}={self._short_repr(v)}" for k, v in kwargs.items())
        return parts

    def _short_repr(self, value: Any) -> str:
        text = repr(value)
        return text if len(text) <= 32 else text[:29] + "..."

    # ------------------------------------------------------------------
    # System ToolSets
    # ------------------------------------------------------------------

    def _load_system_toolsets(self) -> None:
        try:
            import pantheon.toolsets as builtin_toolsets
        except Exception:  # pragma: no cover - import failure surfaced
            logger.exception("Unable to import built-in Pantheon toolsets")
            return

        exports = getattr(builtin_toolsets, "__all__", [])
        for export in exports:
            cls = getattr(builtin_toolsets, export, None)
            if not inspect.isclass(cls) or not issubclass(cls, ToolSet):
                continue
            self._register_system_toolset(export, cls)

    def _register_system_toolset(
        self,
        class_name: str,
        cls: type[ToolSet],
        target_name: str | None = None,
    ) -> str:
        base_name = target_name or self._normalize_system_name(class_name)
        final_name = base_name
        if target_name is None:
            suffix = 1
            while final_name in self._packages:
                suffix += 1
                final_name = f"{base_name}_{suffix}"
        else:
            self._packages.pop(final_name, None)

        record = PackageRecord(name=final_name, path=None, origin="system")
        self._packages[final_name] = record
        self._system_sources[final_name] = cls
        if final_name not in self._system_names:
            self._system_names.append(final_name)

        record.class_defs = {class_name: cls}
        record.classes = {}

        try:
            methods = self._collect_methods(final_name, record.class_defs)
            record.methods = methods
            record.description = inspect.getdoc(cls)
            record.mtime = time.time()
            record.last_loaded = record.mtime
            record.status = "ready" if record.methods else "empty"
        except Exception as exc:  # pragma: no cover - surfaced to logs
            record.status = "error"
            record.last_error = str(exc)
            logger.exception(
                f"Failed to register system ToolSet '{class_name}' "
                f"for package '{final_name}'"
            )
        else:
            logger.debug(
                f"Registered system package '{final_name}' "
                f"with {len(record.methods)} method(s)"
            )

        return final_name

    def _normalize_system_name(self, class_name: str) -> str:
        name = class_name
        if name.lower().endswith("toolset"):
            name = name[: -len("toolset")]
        snake = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snake)
        return snake.lower()

    def _reload_system_package(self, name: str) -> dict:
        cls = self._system_sources.get(name)
        if not cls:
            return {"success": False, "error": f"System package '{name}' not found"}
        self._register_system_toolset(class_name=cls.__name__, cls=cls, target_name=name)
        logger.info(f"Reloaded system package '{name}'")
        return {"success": True, "reloaded": [name]}


__all__ = ["PackageManager", "PackageMethod", "PackageRecord"]
