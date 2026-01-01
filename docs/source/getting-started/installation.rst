Installation
============

This guide will help you install Pantheon and set up your environment.

Basic Installation
------------------

Install from PyPI
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install pantheon-agents

With optional toolsets (Python execution, Jupyter notebooks, web browsing, etc.):

.. code-block:: bash

   pip install pantheon-agents[toolsets]

Install from Source
~~~~~~~~~~~~~~~~~~~

For the latest development version:

.. code-block:: bash

   git clone https://github.com/aristoteleo/pantheon-agents.git
   cd pantheon-agents
   pip install -e ".[toolsets]"

Installation Options
--------------------

Pantheon provides several optional dependency groups:

.. code-block:: bash

   # Core only (minimal installation)
   pip install pantheon-agents

   # With all toolsets (Python, Jupyter, Web, etc.)
   pip install pantheon-agents[toolsets]

   # With knowledge base support (LlamaIndex, Qdrant)
   pip install pantheon-agents[knowledge]

   # With Slack integration
   pip install pantheon-agents[slack]

   # Development dependencies
   pip install pantheon-agents[dev]

   # Multiple options
   pip install pantheon-agents[toolsets,knowledge]

Dependencies
------------

Core dependencies are automatically installed:

- Python 3.10+
- LiteLLM for unified LLM API access
- Rich for terminal UI
- prompt-toolkit for interactive REPL
- NATS for distributed messaging
- FastMCP for Model Context Protocol support

Optional dependencies for specific features:

- **Toolsets**: pandas, matplotlib, jupyter-client, tree-sitter
- **Knowledge**: llama-index, qdrant-client, fastembed
- **Slack**: slack-sdk, slack-bolt

Environment Setup
-----------------

API Keys
~~~~~~~~

Set up your LLM provider API keys:

.. code-block:: bash

   # OpenAI (default)
   export OPENAI_API_KEY="your-openai-key"

   # Anthropic Claude
   export ANTHROPIC_API_KEY="your-anthropic-key"

   # Or use other providers supported by LiteLLM
   export GEMINI_API_KEY="your-gemini-key"

Configuration Directory
~~~~~~~~~~~~~~~~~~~~~~~

Pantheon uses a layered configuration system:

1. **User global config**: ``~/.pantheon/``
2. **Project config**: ``./.pantheon/``

Create a project configuration:

.. code-block:: bash

   mkdir .pantheon
   # Configuration files will be auto-generated on first run

Starting the REPL
-----------------

The easiest way to start using Pantheon:

.. code-block:: bash

   python -m pantheon.repl

This launches the interactive REPL with default settings.

Verifying Installation
----------------------

.. code-block:: python

   import asyncio
   from pantheon import Agent

   async def main():
       agent = Agent(
           name="test",
           instructions="Say hello!",
           model="gpt-4o-mini"
       )
       response = await agent.run("Hello!")
       print(response.content)

   asyncio.run(main())

Next Steps
----------

- :doc:`quickstart` - Create your first agent
- :doc:`concepts` - Learn core concepts
- :doc:`examples/index` - Explore example implementations
