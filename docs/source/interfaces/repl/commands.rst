REPL Commands
=============

Complete reference for REPL slash commands.

Built-in Commands
-----------------

/help
~~~~~

Display available commands.

.. code-block:: text

   > /help

/view <filepath>
~~~~~~~~~~~~~~~~

Open full-screen file viewer with syntax highlighting.

.. code-block:: text

   > /view src/main.py
   > /view README.md

See :doc:`file-viewer` for navigation keys.

/clear
~~~~~~

Clear conversation context. Starts a fresh conversation while keeping the same session.

.. code-block:: text

   > /clear

/compress
~~~~~~~~~

Compress conversation history to reduce token usage. Useful for long conversations approaching context limits.

.. code-block:: text

   > /compress

The agent will summarize the conversation to preserve key information while reducing tokens.

/exit or /quit
~~~~~~~~~~~~~~

Exit the REPL.

.. code-block:: text

   > /exit
   > /quit

Shell Commands
--------------

Run shell commands directly by prefixing with ``!``:

.. code-block:: text

   > !ls -la
   > !git status
   > !python script.py

Output is displayed and can be referenced in the conversation.

Multi-line Input
----------------

For multi-line messages, wrap in triple backticks:

.. code-block:: text

   > ```
   Please review this code:

   def hello():
       print("Hello")
   ```

The entire block is sent as one message.

Keyboard Shortcuts
------------------

.. list-table::
   :header-rows: 1

   * - Key
     - Action
   * - ``↑`` / ``↓``
     - Navigate command history
   * - ``Tab``
     - Auto-complete
   * - ``Ctrl+C``
     - Cancel current operation
   * - ``Ctrl+D``
     - Exit REPL
   * - ``Ctrl+L``
     - Clear screen

File Viewer Keys
----------------

When in the file viewer (``/view``):

.. list-table::
   :header-rows: 1

   * - Key
     - Action
   * - ``j`` / ``↓``
     - Scroll down
   * - ``k`` / ``↑``
     - Scroll up
   * - ``Space`` / ``Ctrl+F``
     - Page down
   * - ``Ctrl+B``
     - Page up
   * - ``g``
     - Go to top
   * - ``G``
     - Go to bottom
   * - ``q`` / ``Esc``
     - Exit viewer

Interactive Dialogs
-------------------

When agents request approval (via ``notify_user``), an interactive dialog appears:

.. list-table::
   :header-rows: 1

   * - Key
     - Action
   * - ``a``
     - Approve
   * - ``c``
     - Continue planning
   * - ``1-9``
     - Switch between file previews
   * - ``Tab``
     - Next file
   * - ``Esc``
     - Cancel/Reject

Custom Commands
---------------

You can add custom commands by creating command handlers. See :doc:`advanced` for details.
