"""Package installer for downloading and installing Store packages locally."""

from pathlib import Path
from typing import Dict, Optional

from loguru import logger


class PackageInstaller:
    """Install/uninstall agent, team, and skill packages from the Store.

    Packages are installed to the user's ~/.pantheon/ directory structure:
      - agents/{name}.md
      - teams/{name}.md  (+ bundled agents)
      - skills/{name}.md
    """

    def __init__(self, work_dir: Optional[Path] = None):
        from pantheon.settings import get_settings
        self.settings = get_settings(work_dir)

    def install(
        self,
        pkg_type: str,
        name: str,
        content: str,
        files: Optional[Dict[str, str]] = None,
    ) -> list[Path]:
        """Install a package locally.

        Args:
            pkg_type: One of "agent", "team", "skill".
            name: Package name (used as filename).
            content: Main .md file content.
            files: Optional dict of relative_path -> content for bundled files
                   (e.g., {"agents/researcher.md": "..."} for team packages).

        Returns:
            List of paths that were written.
        """
        written: list[Path] = []

        if pkg_type == "agent":
            target = self.settings.agents_dir / f"{name}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(target)

        elif pkg_type == "team":
            target = self.settings.teams_dir / f"{name}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(target)

            # Write bundled agent files
            if files:
                for rel_path, file_content in files.items():
                    # rel_path is like "agents/researcher.md"
                    file_target = self.settings.pantheon_dir / rel_path
                    file_target.parent.mkdir(parents=True, exist_ok=True)
                    file_target.write_text(file_content, encoding="utf-8")
                    written.append(file_target)

        elif pkg_type == "skill":
            target = self.settings.skills_dir / f"{name}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(target)

            # Write bundled skill files (for skill groups)
            if files:
                for rel_path, file_content in files.items():
                    # rel_path is like "skills/omics/quality_control.md"
                    file_target = self.settings.pantheon_dir / rel_path
                    file_target.parent.mkdir(parents=True, exist_ok=True)
                    file_target.write_text(file_content, encoding="utf-8")
                    written.append(file_target)

        else:
            raise ValueError(f"Unknown package type: {pkg_type}")

        for p in written:
            logger.info(f"Installed: {p}")

        return written

    def uninstall(self, pkg_type: str, name: str) -> list[Path]:
        """Uninstall a package by removing its files.

        Args:
            pkg_type: One of "agent", "team", "skill".
            name: Package name.

        Returns:
            List of paths that were removed.
        """
        removed: list[Path] = []

        if pkg_type == "agent":
            target = self.settings.agents_dir / f"{name}.md"
            if target.exists():
                target.unlink()
                removed.append(target)

        elif pkg_type == "team":
            target = self.settings.teams_dir / f"{name}.md"
            if target.exists():
                target.unlink()
                removed.append(target)
            # Note: bundled agents are NOT removed automatically
            # as they may be shared with other teams.

        elif pkg_type == "skill":
            target = self.settings.skills_dir / f"{name}.md"
            if target.exists():
                target.unlink()
                removed.append(target)

        else:
            raise ValueError(f"Unknown package type: {pkg_type}")

        for p in removed:
            logger.info(f"Removed: {p}")

        if not removed:
            logger.warning(f"No files found to remove for {pkg_type}/{name}")

        return removed
