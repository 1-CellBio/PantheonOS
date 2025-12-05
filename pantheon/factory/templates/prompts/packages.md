---
id: packages
name: Pantheon packages
description: Pantheon packages usage guide
---

# Pantheon Packages Usage Guide

Pantheon packages collect reusable packages under `.pantheon/packages/`. Use them whenever you need to run domain logic (analytics, workflows, notifications, etc.) inside Python code without requesting extra tools. Packages run the same way inside `shell`, `python_interpreter`, and `integrated_notebook` sessions.


## What Packages Are & When to Reach for Them
- Each folder inside `.pantheon/packages/` defines one package (e.g. `sales_report`).
- Use packages to group related functions or ToolSets so that one import unlocks an entire capability set.
- Prefer packages for multi-step logic: synthesize data, call APIs, write files, and coordinate results within one script.

## Discovering Capabilities
```python
from pantheon import packages as pp

catalog = pp.packages.list_packages()
sales = pp.packages.describe("sales_report")
matches = pp.packages.search("inventory")
```

**`list_packages()`** → returns a list of package summaries. Each entry contains:
- `name`: package directory name
- `origin`: `user` (workspace code) or `system` (built-in ToolSets)
- `status`: `ready`, `empty`, or `error`
- `methods`: exported method names
- `description`/`path` for quick context

**`describe(name)`** → returns one package with rich metadata:
- `classes`: class names found in the package
- `methods`: for each method, you get docstring, signature, async flag, and any
  parameters you should provide
- `last_loaded`, `last_error`, and filesystem path to help debug

**`search(query)`** → performs a fuzzy match across package names, method names,
docstrings, and descriptions. The result mirrors the list output but filtered to
relevant entries, so you can quickly jump to the right package or function.

## Calling Package Methods
```python
from pantheon import packages as pp

report = pp.packages.sales_report.generate(date="2025-12-01", region="APAC")

inventory = pp.packages.inventory.restock(product="Widget", delta=5)

notification = await pp.packages.ops_center.notify.async_call(
    payload={"event": "ready"},
)
```
- Treat regular methods like any other Python function.
- ToolSet methods remain async-aware: call them with `await ...` or use `.async_call(...)` explicitly.

## Authoring or Updating Packages
1. Create/modify files under `.pantheon/packages/<package_name>/` via `file_manager`, `shell`, or `python_interpreter`. No `__init__.py` is required.
2. Define a normal class whose public methods encapsulate your capability.
3. Any public method (name not starting with `_`) that has a docstring becomes callable through `pp.packages.<package>.<method>`.

Example:
```python
class SalesReportPackage:
    """Sales analytics helpers."""

    def generate(self, date: str, region: str | None = None):
        """Return a structured summary for the requested date."""
        ...

```
Save the file—future imports automatically pick up the latest code.

## Surfaces Where You Can Use Packages
- **python_interpreter**: write scripts, import `pantheon.packages`, call packages.
- **shell.execute_command**: run inline Python (e.g. `python - <<'PY' ... PY`) and use the same import.
- **integrated_notebook / jupyter_kernel**: start a cell with `from pantheon import packages as pp` and continue.

### Advanced / Direct Imports (optional)
- Always prefer `from pantheon import packages as pp`; it keeps discovery, context injection, and system packages consistent.
- If you *must* bypass the shim (for example, a class expects custom constructor args), first import `pantheon.packages` somewhere in the same process (that import ensures the package path is on `sys.path`). Afterwards you can import the module/class directly and instantiate it yourself:
  ```python
  import pantheon.packages  # sets up sys.path and package manager
  from sales_report.specialized import SpecializedPackage  # maps to .pantheon/packages/sales_report/specialized.py

  pkg = SpecializedPackage(config_path="/tmp/custom.yaml")
  pkg.generate(...)
  ```
- Treat this as a last-resort escape hatch; sticking with `pp.packages.*` keeps future runtime improvements transparent and consistent across all tools.

## Quick Checklist
- Need to understand what exists? `list_packages()` → `describe()` → `search()`.
- Need new behavior? Add files under `.pantheon/packages/<name>/`, ensure methods have docstrings, then import again.
- Need async work? Either `await` the method or call `.async_call(...)`.

Stick to this workflow and you can discover, modify, and invoke every package capability directly from your Python code without learning any extra tools or internal details.
