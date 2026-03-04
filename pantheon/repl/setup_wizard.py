"""
Setup Wizard - TUI wizard for configuring LLM provider API keys.

Launched automatically when no API keys are detected at REPL startup.
Guides the user through selecting providers and entering API keys,
then saves them to ~/.pantheon/.env for persistence across sessions.

Also provides PROVIDER_MENU and _save_key_to_env_file used by the
/keys REPL command.
"""

import os
from pathlib import Path

from pantheon.utils.model_selector import PROVIDER_API_KEYS

# Providers shown in the wizard/keys menu: (provider_key, display_name, env_var)
PROVIDER_MENU = [
    ("openai", "OpenAI", "OPENAI_API_KEY"),
    ("anthropic", "Anthropic", "ANTHROPIC_API_KEY"),
    ("gemini", "Google Gemini", "GEMINI_API_KEY"),
    ("google", "Google AI", "GOOGLE_API_KEY"),
    ("azure", "Azure OpenAI", "AZURE_API_KEY"),
    ("zai", "Z.ai (Zhipu)", "ZAI_API_KEY"),
    ("minimax", "MiniMax", "MINIMAX_API_KEY"),
    ("moonshot", "Moonshot", "MOONSHOT_API_KEY"),
    ("deepseek", "DeepSeek", "DEEPSEEK_API_KEY"),
    ("mistral", "Mistral", "MISTRAL_API_KEY"),
    ("groq", "Groq", "GROQ_API_KEY"),
    ("openrouter", "OpenRouter", "OPENROUTER_API_KEY"),
    ("together_ai", "Together AI", "TOGETHER_API_KEY"),
    ("cohere", "Cohere", "COHERE_API_KEY"),
    ("replicate", "Replicate", "REPLICATE_API_KEY"),
    ("huggingface", "Hugging Face", "HUGGINGFACE_API_KEY"),
]


def check_and_run_setup():
    """Check if any LLM provider API keys are set; launch wizard if none found.

    Called at startup before the event loop starts (sync context).
    Also checks for universal LLM_API_KEY (custom API endpoint).
    """
    if os.environ.get("LLM_API_KEY", ""):
        return
    for env_var in PROVIDER_API_KEYS.values():
        if os.environ.get(env_var, ""):
            return
    run_setup_wizard()


def run_setup_wizard(standalone: bool = False):
    """Interactive TUI wizard (sync, for use before event loop starts).

    Args:
        standalone: If True, called via ``pantheon setup`` (skip "Starting Pantheon" message).
    """
    from rich.console import Console
    from rich.panel import Panel
    from prompt_toolkit import prompt as pt_prompt

    console = Console()

    # Header
    console.print()
    console.print(
        Panel(
            "  No LLM provider API keys detected.\n"
            "  Let's set up at least one provider to get started.\n"
            "  You can also use [bold]/keys[/bold] command in the REPL to configure later.",
            title="Pantheon Setup",
            border_style="cyan",
        )
    )

    configured_any = False

    while True:
        # Show custom API endpoint option
        custom_set = " [green](configured)[/green]" if os.environ.get("LLM_API_KEY", "") else ""
        console.print(f"\n  [cyan][0][/cyan] Custom API Endpoint  (LLM_API_BASE + LLM_API_KEY){custom_set}")

        # Show provider menu
        console.print("\nAvailable providers:")
        for i, (_, display_name, env_var) in enumerate(PROVIDER_MENU, 1):
            already_set = " [green](configured)[/green]" if os.environ.get(env_var, "") else ""
            console.print(f"  [cyan][{i}][/cyan] {display_name:<16} ({env_var}){already_set}")
        console.print()

        try:
            selection = pt_prompt("Select providers to configure (comma-separated, e.g. 0,1,3): ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nSetup cancelled.")
            break

        # Parse selection
        indices = []
        has_custom = False
        for part in selection.split(","):
            part = part.strip()
            if part == "0":
                has_custom = True
            elif part.isdigit():
                idx = int(part)
                if 1 <= idx <= len(PROVIDER_MENU):
                    indices.append(idx - 1)

        if not indices and not has_custom:
            console.print("[yellow]No valid providers selected. Please try again.[/yellow]")
            continue

        # Handle custom API endpoint
        if has_custom:
            console.print("\n[bold]Configure Custom API Endpoint[/bold]")
            try:
                base_url = pt_prompt("LLM_API_BASE (e.g. https://your-proxy.com/v1): ")
            except (EOFError, KeyboardInterrupt):
                console.print("\nSkipped.")
                base_url = ""

            base_url = base_url.strip()
            if base_url:
                _save_key_to_env_file("LLM_API_BASE", base_url)
                os.environ["LLM_API_BASE"] = base_url
                console.print("[green]\u2713 LLM_API_BASE saved[/green]")

            try:
                api_key = pt_prompt("LLM_API_KEY: ", is_password=True)
            except (EOFError, KeyboardInterrupt):
                console.print("\nSkipped.")
                api_key = ""

            api_key = api_key.strip()
            if api_key:
                _save_key_to_env_file("LLM_API_KEY", api_key)
                os.environ["LLM_API_KEY"] = api_key
                console.print("[green]\u2713 LLM_API_KEY saved[/green]")
                configured_any = True
            elif not base_url:
                console.print("[yellow]Nothing entered, skipped.[/yellow]")

        # Collect API keys for selected providers
        for idx in indices:
            _, display_name, env_var = PROVIDER_MENU[idx]
            console.print(f"\n[bold]Enter API key for {display_name}[/bold]")
            try:
                api_key = pt_prompt(f"{env_var}: ", is_password=True)
            except (EOFError, KeyboardInterrupt):
                console.print("\nSkipped.")
                continue

            api_key = api_key.strip()
            if not api_key:
                console.print("[yellow]Empty key, skipped.[/yellow]")
                continue

            _save_key_to_env_file(env_var, api_key)
            os.environ[env_var] = api_key
            console.print("[green]\u2713 Saved[/green]")
            configured_any = True

        console.print()
        try:
            more = pt_prompt("Configure another provider? [y/N]: ")
        except (EOFError, KeyboardInterrupt):
            more = "n"
        if more.strip().lower() != "y":
            break

    if configured_any:
        env_path = Path.home() / ".pantheon" / ".env"
        console.print(f"\n[green]\u2713 API keys saved to {env_path}[/green]")
        if not standalone:
            console.print("  Starting Pantheon...\n")
    else:
        console.print(
            "\n[yellow]No API keys configured. "
            "Pantheon may not work correctly without provider keys.[/yellow]\n"
        )

    return configured_any


def _save_key_to_env_file(env_var: str, value: str):
    """Append or update a key in ~/.pantheon/.env."""
    env_dir = Path.home() / ".pantheon"
    env_dir.mkdir(parents=True, exist_ok=True)
    env_file = env_dir / ".env"

    # Read existing content to check for duplicates
    lines = []
    if env_file.exists():
        lines = env_file.read_text().splitlines()

    # Remove existing entry for this var if present
    lines = [line for line in lines if not line.startswith(f"{env_var}=")]

    # Append new entry
    lines.append(f"{env_var}={value}")

    env_file.write_text("\n".join(lines) + "\n")
