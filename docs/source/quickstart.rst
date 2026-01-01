Quick Start
===========

This guide will help you create your first Pantheon agents and teams.

Prerequisites
-------------

Before running the examples, make sure to set up your API key as an environment variable. For OpenAI models (default):

.. code-block:: bash

   export OPENAI_API_KEY="your-api-key-here"

For other LLM providers, see the model configuration details in the :ref:`Model Selection and Providers <model-selection-providers>` section.

Using the REPL (Recommended)
----------------------------

The easiest way to start using Pantheon is through the interactive REPL:

.. code-block:: bash

   python -m pantheon.repl

This launches a feature-rich command-line interface with:

- Syntax highlighting and auto-completion
- Full-screen file viewer (``/view <filepath>``)
- Interactive approval workflows
- Chat history and session management

REPL Commands
~~~~~~~~~~~~~

Common commands available in the REPL:

- ``/help`` - Show available commands
- ``/view <file>`` - View file with syntax highlighting
- ``/clear`` - Clear conversation context
- ``/compress`` - Compress conversation history
- ``/exit`` or ``/quit`` - Exit REPL

Your First Agent
----------------

The simplest way to create an agent programmatically:

.. code-block:: python

   import asyncio
   from pantheon import Agent

   async def main():
       # Create a basic agent
       agent = Agent(
           name="assistant",
           instructions="You are a helpful assistant.",
           model="gpt-4o-mini"
       )

       # Run a single query
       response = await agent.run("What is 2 + 2?")
       print(response.content)

       # Or chat interactively
       await agent.chat()

   if __name__ == "__main__":
       asyncio.run(main())

Adding Tools to Agents
----------------------

Agents become more powerful with tools. Here's a simple example:

.. code-block:: python

   import asyncio
   from pantheon import Agent
   import random

   async def main():
       # Create an agent
       agent = Agent(
           name="weather_assistant",
           instructions="You are a helpful weather assistant.",
           model="gpt-4o-mini"
       )

       # Define a custom tool using decorator
       @agent.tool
       def get_weather(city: str) -> str:
           """Get the current weather for a city."""
           temp = random.randint(60, 85)
           conditions = random.choice(["sunny", "cloudy", "rainy"])
           return f"The weather in {city} is {temp}F and {conditions}."

       # Define another tool
       @agent.tool
       def get_forecast(city: str, days: int = 3) -> str:
           """Get weather forecast for a city."""
           return f"The {days}-day forecast for {city} shows mixed conditions."

       # Chat with the agent - it can now use these tools
       await agent.chat()

   if __name__ == "__main__":
       asyncio.run(main())

Using Built-in Toolsets
-----------------------

Pantheon includes powerful built-in toolsets:

.. code-block:: python

   import asyncio
   from pantheon import Agent
   from pantheon.toolsets import FileManagerToolSet, ShellToolSet

   async def main():
       # Create toolsets
       file_tools = FileManagerToolSet()
       shell_tools = ShellToolSet()

       # Create an agent with toolsets
       agent = Agent(
           name="developer",
           instructions="You are a developer assistant who can read/write files and run commands.",
           model="gpt-4o",
           tools=[file_tools, shell_tools]
       )

       await agent.chat()

   if __name__ == "__main__":
       asyncio.run(main())

Available built-in toolsets:

- ``FileManagerToolSet`` - File operations (read, write, edit, search)
- ``ShellToolSet`` - Shell command execution
- ``PythonInterpreterToolSet`` - Python code execution
- ``IntegratedNotebookToolSet`` - Jupyter notebook integration
- ``WebToolSet`` - Web browsing and search
- ``KnowledgeToolSet`` - Knowledge base management
- ``TaskToolSet`` - Ephemeral task tracking

Creating Agent Teams
--------------------

Multiple agents can work together as a team.

PantheonTeam (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The default team type with intelligent agent delegation:

.. code-block:: python

   import asyncio
   from pantheon import Agent
   from pantheon.team import PantheonTeam

   async def main():
       # Create specialized agents
       researcher = Agent(
           name="researcher",
           instructions="You are an expert researcher. You gather and analyze information.",
           model="gpt-4o-mini"
       )

       writer = Agent(
           name="writer",
           instructions="You are a professional writer. You create clear, engaging content.",
           model="gpt-4o-mini"
       )

       # Create a team
       team = PantheonTeam([researcher, writer])

       # Start chatting - agents collaborate automatically
       await team.chat()

   if __name__ == "__main__":
       asyncio.run(main())

Swarm Team
~~~~~~~~~~

Agents dynamically transfer control to each other:

.. code-block:: python

   import asyncio
   from pantheon import Agent
   from pantheon.team import SwarmTeam

   async def main():
       researcher = Agent(
           name="Researcher",
           instructions="You are an expert researcher.",
           model="gpt-4o-mini"
       )

       writer = Agent(
           name="Writer",
           instructions="You are a professional writer.",
           model="gpt-4o-mini"
       )

       # Add transfer functions
       @researcher.tool
       def transfer_to_writer():
           """Transfer control to the writer when research is complete."""
           return writer

       @writer.tool
       def transfer_to_researcher():
           """Transfer control to the researcher when more information is needed."""
           return researcher

       team = SwarmTeam([researcher, writer])
       await team.chat()

   if __name__ == "__main__":
       asyncio.run(main())

For more team types (Sequential, SwarmCenter, MoA, AgentAsToolTeam), see the :doc:`team/index` documentation.

Using MCP Servers
-----------------

Pantheon supports Model Context Protocol (MCP) for external tool integration:

.. code-block:: python

   import asyncio
   from pantheon import Agent
   from pantheon.providers import MCPProvider

   async def main():
       # Connect to an MCP server
       mcp = MCPProvider("npx -y @anthropic/mcp-server-filesystem")

       agent = Agent(
           name="file_agent",
           instructions="You can access the filesystem via MCP tools.",
           model="gpt-4o",
           tools=[mcp]
       )

       await agent.chat()

   if __name__ == "__main__":
       asyncio.run(main())

Creating and Configuring ChatRoom
----------------------------------

The ChatRoom service provides a web interface for interacting with your agents:

Basic ChatRoom Setup
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Start the chatroom with default configuration
   export OPENAI_API_KEY=your_key
   python -m pantheon.chatroom

Then:

1. Copy the service ID from the output
2. Go to https://pantheon-ui.vercel.app/
3. Paste the service ID and click "Connect"
4. Start chatting with your agents!

ChatRoom with Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a YAML configuration file to define your ChatRoom:

.. code-block:: yaml

   # .pantheon/teams/default.md (frontmatter format)
   ---
   agents:
     - name: triage
       instructions: You are the triage agent...
       model: openai/gpt-4o
     - name: data_analyst
       instructions: You analyze data...
       model: openai/gpt-4o
       toolsets:
         - python_interpreter
         - file_manager
   ---

Then start the ChatRoom:

.. code-block:: bash

   python -m pantheon.chatroom

Smart Functions
---------------

Create LLM-powered functions with the ``@smart_func`` decorator:

.. code-block:: python

   import asyncio
   from pantheon import smart_func

   @smart_func(model="gpt-4o-mini")
   def analyze_sentiment(text: str) -> dict:
       """Analyze the sentiment of the given text.

       Returns a dict with 'sentiment' (positive/negative/neutral)
       and 'confidence' (0-1).
       """
       pass  # LLM handles the implementation!

   async def main():
       result = await analyze_sentiment("I love this product!")
       print(result)  # {'sentiment': 'positive', 'confidence': 0.95}

   asyncio.run(main())

Next Steps
----------

1. **Learn Pantheon's core concepts** - :doc:`concepts`
2. **Understand Pantheon's architecture** - :doc:`architecture`
3. **Deep dive into components** - :doc:`agent/index`, :doc:`team/index`, :doc:`toolsets/index`
4. **Explore examples** - :doc:`examples/index`
