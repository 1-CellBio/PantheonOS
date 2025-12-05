"""Runtime helpers exposed inside python/shell interpreters."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from . import get_package_manager
from .context import load_context
from .manager import PackageManager

_RUNTIME: "PackageRuntime" | None = None


def get_runtime(packages_path: str | None = None) -> "PackageRuntime":
    global _RUNTIME
    manager = get_package_manager(packages_path)
    if _RUNTIME is None or _RUNTIME.manager is not manager:
        _RUNTIME = PackageRuntime(manager)
    return _RUNTIME


async def _ensure_awaitable(result: Any) -> Any:
    if asyncio.iscoroutine(result) or asyncio.isfuture(result):
        return await result
    return result


@dataclass
class PackageRuntimeMethod:
    manager: PackageManager
    package_name: str
    method_name: str

    def _invoke(self, *args, **kwargs):
        context = kwargs.pop("context_variables", None)
        if context is None:
            context_payload = load_context()
            context = dict(context_payload.get("context_variables") or {})
        return self.manager.call(
            self.package_name,
            self.method_name,
            *args,
            context_variables=context,
            **kwargs,
        )

    async def async_call(self, *args, **kwargs):
        result = self._invoke(*args, **kwargs)
        return await _ensure_awaitable(result)

    def __call__(self, *args, **kwargs):
        result = self._invoke(*args, **kwargs)
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            return asyncio.run(result)
        return result

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<PackageRuntimeMethod {self.package_name}.{self.method_name}>"


class PackageProxy:
    def __init__(self, name: str, manager: PackageManager):
        self._name = name
        self._manager = manager

    def __getattr__(self, item: str) -> PackageRuntimeMethod:
        return PackageRuntimeMethod(self._manager, self._name, item)

    def __dir__(self):  # pragma: no cover - UX helper
        record = None
        try:
            record = self._manager.describe_package(self._name)["package"]
        except Exception:
            return []
        return record.get("methods", [])


class PackageRuntime:
    def __init__(self, manager: PackageManager):
        self.manager = manager

    def list_packages(self):
        return self.manager.list_packages()

    def describe(self, name: str):
        return self.manager.describe_package(name)

    def reload(self, name: str | None = None):
        return self.manager.reload_package(name)

    def __getattr__(self, item: str) -> PackageProxy:
        return PackageProxy(item, self.manager)

    def __dir__(self):  # pragma: no cover
        return self.manager.discover()


__all__ = ["PackageRuntime", "get_runtime"]
