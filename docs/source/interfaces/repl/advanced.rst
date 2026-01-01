Advanced REPL Usage
===================

This section covers advanced REPL features and customization.

Programmatic Usage
------------------

You can create and run the REPL programmatically:

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

Initialization Modes
--------------------

The REPL supports several initialization modes:

**1. Agent Mode**

Pass an Agent or Team directly:

.. code-block:: python

   from pantheon import Agent
   from pantheon.repl import Repl

   agent = Agent(name="assistant", ...)
   repl = Repl(agent=agent)

**2. ChatRoom Mode**

Use an existing ChatRoom:

.. code-block:: python

   from pantheon.chatroom import ChatRoom
   from pantheon.repl import Repl

   chatroom = ChatRoom()
   repl = Repl(chatroom=chatroom)

**3. Endpoint Mode**

Pass an Endpoint instance:

.. code-block:: python

   from pantheon.endpoint import Endpoint
   from pantheon.repl import Repl

   endpoint = Endpoint(workspace_path="./workspace")
   repl = Repl(endpoint=endpoint)

**4. Auto Mode**

Let REPL create everything automatically:

.. code-block:: python

   repl = Repl()  # Auto-creates ChatRoom and Endpoint
   await repl.run()

Custom Command Handlers
-----------------------

Add custom commands by creating a handler:

.. code-block:: python

   from pantheon.repl.handlers.base import CommandHandler

   class MyHandler(CommandHandler):
       def __init__(self, console, repl):
           super().__init__(console, repl)

       def can_handle(self, command: str) -> bool:
           return command.startswith("/mycommand")

       async def handle(self, command: str) -> bool:
           # Parse and execute command
           self.console.print("My custom command executed!")
           return True  # Command was consumed

Register in a custom REPL:

.. code-block:: python

   class CustomRepl(Repl):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.handlers.append(MyHandler(self.console, self))

Interactive Approval Dialogs
----------------------------

The REPL supports interactive approval workflows. When an agent calls ``notify_user`` with ``interrupt=True``, a dialog appears:

.. code-block:: python

   # In agent tool
   context_variables.notify_user(
       message="I want to modify these files...",
       files=["file1.py", "file2.py"],
       interrupt=True  # Triggers interactive dialog
   )

Users see:

- Markdown-rendered message
- File previews (switch with 1-9 or Tab)
- Action keys: ``a`` (approve), ``c`` (continue), ``Esc`` (reject)

Session Management
------------------

**Resume Previous Chat**

.. code-block:: bash

   python -m pantheon.repl --chat-id <previous-id>

**Set Memory Directory**

.. code-block:: python

   repl = Repl(
       agent=agent,
       memory_dir="./my_chats"
   )

Environment Variables
---------------------

The REPL respects these environment variables:

.. list-table::
   :header-rows: 1

   * - Variable
     - Description
   * - ``OPENAI_API_KEY``
     - OpenAI API key
   * - ``ANTHROPIC_API_KEY``
     - Anthropic API key
   * - ``PANTHEON_LOG_LEVEL``
     - Log level (DEBUG, INFO, WARNING, ERROR)

Configuration File
------------------

REPL settings in ``.pantheon/settings.json``:

.. code-block:: json

   {
     "repl": {
       "quiet": false,
       "default_template": "default",
       "log_level": "ERROR"
     }
   }

**Options:**

- ``quiet``: Suppress startup messages
- ``default_template``: Team template to load by default
- ``log_level``: Logging verbosity

Debugging
---------

Enable debug logging:

.. code-block:: bash

   PANTHEON_LOG_LEVEL=DEBUG python -m pantheon.repl

View logs in ``~/.pantheon/logs/``.
