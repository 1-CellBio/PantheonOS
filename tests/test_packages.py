from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path

from pantheon.agent import Agent
from pantheon.package_runtime import (
    configure_package_manager,
    build_context_payload,
    export_context,
)
from pantheon.package_runtime.manager import PackageManager

SAMPLE_WORKSPACE = Path(__file__).parent / "sample_workspace"
SAMPLE_PACKAGES = {
    path.name
    for path in (SAMPLE_WORKSPACE / ".pantheon" / "packages").iterdir()
    if path.is_dir()
}


def _prepare_runtime(workspace: Path, packages_dir: Path, monkeypatch) -> object:
    payload = build_context_payload(workdir=str(workspace))
    env: dict[str, str] = {}
    export_context(payload, env=env)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    manager = configure_package_manager()
    assert manager.packages_path == packages_dir.resolve()
    sys.modules.pop("pantheon.packages", None)
    return importlib.import_module("pantheon.packages")


def test_export_context_filters_non_serializable_entries():
    payload = build_context_payload(
        workdir="/tmp/workspace",
        context_variables={
            "ok": "value",
            "number": 42,
            "callable": lambda x: x,  # type: ignore[arg-type]
            "nested": {"valid": 1, "bad": object()},
            "list": ["hello", object()],
        },
    )
    env: dict[str, str] = {}
    export_context(payload, env=env)

    serialized = env["PANTHEON_CONTEXT"]
    data = json.loads(serialized)
    ctx_vars = data["context_variables"]

    assert ctx_vars["ok"] == "value"
    assert ctx_vars["number"] == 42
    assert "callable" not in ctx_vars
    assert "nested" not in ctx_vars
    assert "list" not in ctx_vars


def test_packages_module_adds_packages_path(tmp_path, monkeypatch):
    workdir = tmp_path
    pkg_dir = workdir / ".pantheon" / "packages"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "demo.py").write_text("value = 42")

    payload = build_context_payload(workdir=str(workdir))
    env: dict[str, str] = {}
    export_context(payload, env=env)
    monkeypatch.setenv("PANTHEON_CONTEXT", env["PANTHEON_CONTEXT"])

    from pantheon import packages as pp  # noqa: WPS433

    importlib.reload(pp)
    assert str(pkg_dir.resolve()) in sys.path


def test_runtime_imports_sample_packages(monkeypatch):
    workspace = SAMPLE_WORKSPACE
    packages_dir = workspace / ".pantheon" / "packages"
    runtime = _prepare_runtime(workspace, packages_dir, monkeypatch)

    manager = PackageManager(packages_dir)
    packages = manager.list_packages()
    user_names = {pkg["name"] for pkg in packages if pkg.get("origin") == "user"}
    assert user_names == set(SAMPLE_PACKAGES)
    assert any(
        pkg.get("origin") == "system" and "python_interpreter" in pkg["name"]
        for pkg in packages
    )

    report = runtime.packages.sales_report.generate(
        date="2025-12-01", region="APAC"
    )
    assert report["region"] == "APAC"

    inventory = runtime.packages.inventory.restock(product="Widget", delta=3)
    assert inventory == {"product": "Widget", "delta": 3, "status": "restocked"}

    notification = asyncio.run(
        runtime.packages.ops_center.notify.async_call(
            payload={"event": "report-ready"},
            context_variables={
                "execution_context_id": "exec-1",
                "client_id": "cli-1",
            },
        )
    )
    assert notification["context_id"] == "exec-1"
    assert notification["payload"]["event"] == "report-ready"
    assert hasattr(runtime.packages, "python_interpreter")


def test_agent_end_to_end_package_pipeline(monkeypatch):
    workspace = SAMPLE_WORKSPACE
    packages_dir = workspace / ".pantheon" / "packages"
    _prepare_runtime(workspace, packages_dir, monkeypatch)

    async def _agent_flow():
        agent = Agent(
            name="package-orchestrator",
            instructions="Use local packages to build reports and notifications.",
        )

        @agent.tool
        async def orchestrate(
            date: str,
            region: str | None = None,
            context_variables: dict | None = None,
        ):
            from pantheon import packages as pp_local  # noqa: WPS433

            report = pp_local.packages.sales_report.generate(date=date, region=region)
            inventory = pp_local.packages.inventory.restock(
                product="Widget", delta=5
            )
            notification = await pp_local.packages.ops_center.notify.async_call(
                payload={"report_status": report["status"]},
                context_variables=context_variables,
            )
            return {
                "report": report,
                "inventory": inventory,
                "notification": notification,
            }

        context = {"client_id": "agent-test", "execution_context_id": "exec-42"}
        return await agent.call_tool(
            "orchestrate",
            {"date": "2025-12-02", "region": "APAC"},
            context_variables=context,
        )

    result = asyncio.run(_agent_flow())
    assert result["report"]["region"] == "APAC"
    assert result["inventory"]["delta"] == 5
    assert result["notification"]["context_id"] == "exec-42"
