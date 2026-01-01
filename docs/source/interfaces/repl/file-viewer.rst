File Viewer
===========

The REPL includes a full-screen file viewer with syntax highlighting, accessible via the ``/view`` command.

Usage
-----

.. code-block:: text

   > /view <filepath>

Examples:

.. code-block:: text

   > /view src/main.py
   > /view README.md
   > /view config/settings.json

Features
--------

Syntax Highlighting
~~~~~~~~~~~~~~~~~~~

Files are highlighted based on their extension using Pygments. Supports 100+ programming languages including:

- Python, JavaScript, TypeScript
- Rust, Go, C, C++
- HTML, CSS, JSON, YAML
- Markdown, reStructuredText
- And many more

Line Numbers
~~~~~~~~~~~~

Lines are numbered for easy reference when discussing code with the agent.

Smooth Navigation
~~~~~~~~~~~~~~~~~

Navigate large files efficiently with vim-style keys.

Multiple Encodings
~~~~~~~~~~~~~~~~~~

Handles UTF-8, Latin-1, and other common encodings.

Navigation
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Action
   * - ``j`` or ``↓``
     - Scroll down one line
   * - ``k`` or ``↑``
     - Scroll up one line
   * - ``Space`` or ``Ctrl+F``
     - Page down
   * - ``Ctrl+B``
     - Page up
   * - ``g``
     - Jump to beginning of file
   * - ``G``
     - Jump to end of file
   * - ``q`` or ``Esc``
     - Exit viewer and return to REPL

Tips
----

**Quick Code Review**

Use ``/view`` to inspect files before asking the agent to modify them:

.. code-block:: text

   > /view src/api.py
   (review the code)
   > Please add error handling to the fetch_data function

**Comparing Changes**

After the agent makes changes, view the file to verify:

.. code-block:: text

   > Edit src/api.py to add logging
   (agent makes changes)
   > /view src/api.py
   (verify the changes)

**Large Files**

For very large files, use ``g`` and ``G`` to quickly jump between beginning and end.

Programmatic Access
-------------------

The file viewer can also be used programmatically:

.. code-block:: python

   from pantheon.repl.viewers import FileViewer

   async def view_file(path: str):
       viewer = FileViewer()
       await viewer.view(path)
