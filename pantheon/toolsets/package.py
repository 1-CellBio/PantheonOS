"""Unified package management toolset for Pantheon packages."""

from __future__ import annotations

from pathlib import Path

from ..package_runtime import get_package_manager
from ..toolset import ToolSet, tool


class PackageToolSet(ToolSet):
    """Expose read-only management functions for packages via tools."""

    def __init__(
        self,
        name: str,
        workdir: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        root = Path(workdir).expanduser().resolve() if workdir else Path.cwd()
        packages_path = root / ".pantheon" / "packages"
        packages_path.mkdir(parents=True, exist_ok=True)
        self.manager = get_package_manager(packages_path)

    @tool
    async def search_packages(self, query: str | None = None) -> dict:
        """Return packages whose name, description, or method docs match the query."""

        try:
            packages = self.manager.list_packages()
            if not query:
                return {"success": True, "packages": packages}
            q = query.lower()
            results = []
            for pkg in packages:
                haystacks = [
                    pkg.get("name") or "",
                    pkg.get("description") or "",
                    " ".join(pkg.get("methods") or []),
                ]
                if any(q in hay.lower() for hay in haystacks):
                    results.append(pkg)
            return {"success": True, "packages": results}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

__all__ = ["PackageToolSet"]
