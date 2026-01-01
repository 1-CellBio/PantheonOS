ChatRoom Quick Start
====================

Get the web UI running in under 2 minutes.

Prerequisites
-------------

1. Pantheon installed: ``pip install pantheon-agents[toolsets]``
2. API key configured: ``export OPENAI_API_KEY="...``

Step 1: Start ChatRoom
----------------------

.. code-block:: bash

   python -m pantheon.chatroom

You'll see output like:

.. code-block:: text

   ╭─────────────────────────────────────╮
   │ ChatRoom started!                   │
   │                                     │
   │ Service ID: abc123-def456-...       │
   │                                     │
   │ Connect at:                         │
   │ https://pantheon-ui.vercel.app/     │
   ╰─────────────────────────────────────╯

Step 2: Connect the Web UI
--------------------------

1. Open https://pantheon-ui.vercel.app/ in your browser
2. Paste the service ID from the terminal
3. Click "Connect"

Step 3: Start Chatting
----------------------

Type your message in the chat input and press Enter. The agent will respond.

Using Team Templates
--------------------

ChatRoom uses team templates from ``.pantheon/teams/``. To use a specific template:

.. code-block:: bash

   python -m pantheon.chatroom --template data_research_team

Available options:

.. code-block:: bash

   # List available templates
   ls .pantheon/teams/

   # Use a template
   python -m pantheon.chatroom --template <template_name>

Creating Your First Template
----------------------------

Create a file ``.pantheon/teams/my_team.md``:

.. code-block:: markdown

   ---
   name: My Team
   icon: 🤖
   agents:
     - name: assistant
       model: openai/gpt-4o-mini
       instructions: You are a helpful assistant.
     - name: coder
       model: openai/gpt-4o
       instructions: You are a coding expert.
       toolsets:
         - python_interpreter
         - file_manager
   ---

   # My Custom Team

   This team helps with general tasks and coding.

Then start with your template:

.. code-block:: bash

   python -m pantheon.chatroom --template my_team

Common Options
--------------

.. code-block:: bash

   # Use specific memory directory
   python -m pantheon.chatroom --memory-dir ./my_chats

   # Use specific template
   python -m pantheon.chatroom --template developer_team

   # Quiet mode
   python -m pantheon.chatroom --quiet

Troubleshooting
---------------

**Connection Failed**

- Check that the ChatRoom is still running
- Verify the service ID was copied correctly
- Check for firewall issues

**Agent Not Responding**

- Verify your API key is set correctly
- Check the terminal for error messages

**Template Not Found**

- Ensure the template file exists in ``.pantheon/teams/``
- Check the file name matches (without ``.md`` extension)

Next Steps
----------

- :doc:`web-interface` - Explore UI features
- :doc:`/configuration/templates/teams` - Create custom team templates
