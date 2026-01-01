REPL Module
===========

.. module:: pantheon.repl

The REPL (Read-Eval-Print Loop) module provides a feature-rich interactive command-line interface for agents and teams.

Overview
--------

The REPL module provides:

- Interactive chat with agents and teams
- Syntax highlighting for code and markdown
- Full-screen file viewer with navigation
- Interactive approval workflows
- Command history and auto-completion
- Session management and persistence

Starting the REPL
-----------------

Command Line
~~~~~~~~~~~~

The easiest way to start the REPL:

.. code-block:: bash

   python -m pantheon.repl

With options:

.. code-block:: bash

   # Specify a team template
   python -m pantheon.repl --team myteam

   # Specify memory directory
   python -m pantheon.repl --memory-dir ./chats

   # Resume a specific chat
   python -m pantheon.repl --chat-id abc123

Programmatic Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from pantheon.repl import Repl
   from pantheon import Agent

   async def main():
       agent = Agent(
           name="assistant",
           instructions="You are helpful.",
           model="gpt-4o"
       )

       repl = Repl(agent=agent)
       await repl.run()

   asyncio.run(main())

With Teams
~~~~~~~~~~

.. code-block:: python

   from pantheon.repl import Repl
   from pantheon.team import PantheonTeam
   from pantheon import Agent

   async def main():
       agents = [
           Agent(name="researcher", instructions="..."),
           Agent(name="writer", instructions="...")
       ]

       team = PantheonTeam(agents)
       repl = Repl(agent=team)
       await repl.run()

With ChatRoom
~~~~~~~~~~~~~

.. code-block:: python

   from pantheon.repl import Repl
   from pantheon.chatroom import ChatRoom

   async def main():
       chatroom = ChatRoom()
       repl = Repl(chatroom=chatroom)
       await repl.run()

REPL Commands
-------------

Built-in Commands
~~~~~~~~~~~~~~~~~

The REPL provides several built-in slash commands:

- ``/help`` - Show available commands
- ``/view <filepath>`` - Open full-screen file viewer
- ``/clear`` - Clear conversation context
- ``/compress`` - Compress conversation history to save tokens
- ``/exit`` or ``/quit`` - Exit the REPL

File Viewer
~~~~~~~~~~~

The ``/view`` command opens a full-screen file viewer with:

- Syntax highlighting via Pygments
- Keyboard navigation:

  - ``j/k`` or arrow keys: Scroll up/down
  - ``Space`` or ``Ctrl-F``: Page down
  - ``Ctrl-B``: Page up
  - ``g``: Go to top
  - ``G``: Go to bottom
  - ``q`` or ``Esc``: Exit viewer

Example:

.. code-block:: text

   > /view src/main.py

Interactive Approval
~~~~~~~~~~~~~~~~~~~~

When agents request user approval (via ``notify_user`` with ``interrupt=True``), an interactive dialog appears with:

- Markdown-rendered notification message
- File preview with multi-file switching (keys 1-9, Tab)
- Action buttons:

  - ``a``: Approve
  - ``c``: Continue planning
  - ``Esc``: Reject/Cancel

Multi-line Input
~~~~~~~~~~~~~~~~

For multi-line messages, use triple backticks:

.. code-block:: text

   > ```
   This is a
   multi-line
   message
   ```

REPL Class
----------

.. autoclass:: pantheon.repl.core.Repl
   :members:
   :undoc-members:
   :show-inheritance:

Initialization Modes
~~~~~~~~~~~~~~~~~~~~

The Repl class supports multiple initialization modes:

1. **Agent Mode**: Pass an Agent or Team directly

   .. code-block:: python

      repl = Repl(agent=my_agent)

2. **ChatRoom Mode**: Pass an existing ChatRoom

   .. code-block:: python

      repl = Repl(chatroom=my_chatroom)

3. **Endpoint Mode**: Pass an Endpoint instance

   .. code-block:: python

      repl = Repl(endpoint=my_endpoint)

4. **Auto Mode**: Create everything automatically

   .. code-block:: python

      repl = Repl()  # Auto-creates ChatRoom and Endpoint

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   repl = Repl(
       agent=my_agent,
       memory_dir="./chat_history",  # Directory for persistence
       chat_id="session-123",        # Specific chat session ID
   )

UI Components
-------------

ReplUI
~~~~~~

Base class providing UI rendering capabilities:

- Console output with Rich
- Progress indicators
- Token statistics display
- Task rendering

TaskUIRenderer
~~~~~~~~~~~~~~

Renders task progress and status:

- Tool execution progress
- Agent thinking indicators
- Cost and token tracking

NotifyUIRenderer
~~~~~~~~~~~~~~~~

Renders notification panels and approval dialogs.

Viewers Module
--------------

FileViewer
~~~~~~~~~~

Full-screen file viewer using prompt_toolkit:

.. code-block:: python

   from pantheon.repl.viewers import FileViewer

   async def view_file():
       viewer = FileViewer()
       await viewer.view("path/to/file.py")

Features:

- Syntax highlighting for 100+ languages
- Line numbers
- Smooth scrolling
- Multiple encoding support

NotifyDialog
~~~~~~~~~~~~

Interactive dialog for agent approval workflows:

.. code-block:: python

   from pantheon.repl.viewers import NotifyDialog

   dialog = NotifyDialog(
       message="Agent wants to modify files",
       files=["file1.py", "file2.py"]
   )
   result = await dialog.show()

Command Handlers
----------------

Base Handler
~~~~~~~~~~~~

.. code-block:: python

   from pantheon.repl.handlers.base import CommandHandler

   class MyHandler(CommandHandler):
       def __init__(self, console, repl):
           super().__init__(console, repl)

       def can_handle(self, command: str) -> bool:
           return command.startswith("/mycommand")

       async def handle(self, command: str) -> bool:
           # Handle the command
           return True  # Consumed the command

Registering Custom Handlers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CustomRepl(Repl):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.handlers.append(MyCustomHandler(self.console, self))

Best Practices
--------------

1. **Use Templates**: Define agent/team configurations in ``.pantheon/`` for reusability
2. **Memory Management**: Use ``/compress`` periodically for long conversations
3. **File Viewer**: Use ``/view`` to inspect files before editing
4. **History**: Use arrow keys to navigate command history
5. **Interrupts**: Use Ctrl+C gracefully to interrupt long operations

Integration Examples
--------------------

Development Workflow
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pantheon.repl import Repl
   from pantheon import Agent
   from pantheon.toolsets import FileManagerToolSet, ShellToolSet

   async def dev_repl():
       agent = Agent(
           name="developer",
           instructions="You are a developer assistant.",
           model="gpt-4o",
           tools=[FileManagerToolSet(), ShellToolSet()]
       )

       repl = Repl(agent=agent)
       await repl.run()

Data Analysis
~~~~~~~~~~~~~

.. code-block:: python

   from pantheon.repl import Repl
   from pantheon import Agent
   from pantheon.toolsets import IntegratedNotebookToolSet

   async def analysis_repl():
       agent = Agent(
           name="analyst",
           instructions="You are a data analyst.",
           model="gpt-4o",
           tools=[IntegratedNotebookToolSet()]
       )

       repl = Repl(agent=agent)
       await repl.run()
