"""Unified Pantheon CLI entry point.

Usage:
    pantheon cli [OPTIONS]       Start Pantheon CLI (REPL)
    pantheon ui [OPTIONS]        Start Pantheon UI (Chatroom)
"""

import warnings

# Suppress DeprecationWarnings before any third-party imports (fastapi, starlette, etc.)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import sys

import fire
from dotenv import load_dotenv

# Load .env files
load_dotenv()
load_dotenv(
    os.path.join(os.path.expanduser("~"), ".pantheon", ".env"), override=False
)

# Windows UTF-8 setup
if sys.platform == "win32":
    try:
        os.system("chcp 65001 > nul 2>&1")
        if sys.stdout:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def setup():
    """Launch the setup wizard to configure LLM provider API keys."""
    from pantheon.repl.setup_wizard import run_setup_wizard

    run_setup_wizard(standalone=True)


def main():
    # Skip auto-setup if user explicitly requested "pantheon setup"
    if len(sys.argv) < 2 or sys.argv[1] != "setup":
        from pantheon.repl.setup_wizard import check_and_run_setup

        check_and_run_setup()

    # Import REAL functions — Fire reads their signatures for --help
    from pantheon.repl.__main__ import start as cli
    from pantheon.chatroom.start import start_services as ui

    fire.Fire({"cli": cli, "ui": ui, "setup": setup}, name="pantheon")


if __name__ == "__main__":
    main()
