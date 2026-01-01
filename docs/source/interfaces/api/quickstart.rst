API Quick Start
===============

Get started with the Pantheon Python API.

Installation
------------

.. code-block:: bash

   pip install pantheon-agents[toolsets]

Set your API key:

.. code-block:: bash

   export OPENAI_API_KEY="your-key"

Basic Agent
-----------

.. code-block:: python

   import asyncio
   from pantheon import Agent

   async def main():
       agent = Agent(
           name="assistant",
           instructions="You are a helpful assistant.",
           model="gpt-4o-mini"
       )

       response = await agent.run("Hello!")
       print(response.content)

   asyncio.run(main())

With Tools
----------

.. code-block:: python

   from pantheon import Agent
   from pantheon.toolsets import FileManagerToolSet, ShellToolSet

   async def main():
       agent = Agent(
           name="developer",
           instructions="You are a developer assistant.",
           model="gpt-4o",
           tools=[
               FileManagerToolSet(),
               ShellToolSet()
           ]
       )

       response = await agent.run("List files in the current directory")
       print(response.content)

Custom Tools
------------

.. code-block:: python

   from pantheon import Agent

   agent = Agent(name="calculator", instructions="Do math.")

   @agent.tool
   def calculate(expression: str) -> float:
       """Evaluate a math expression."""
       return eval(expression)

   response = await agent.run("What is 15 * 7?")

Interactive Chat
----------------

.. code-block:: python

   # Start interactive REPL
   await agent.chat()

Teams
-----

.. code-block:: python

   from pantheon import Agent
   from pantheon.team import PantheonTeam

   researcher = Agent(name="researcher", instructions="Research topics.")
   writer = Agent(name="writer", instructions="Write content.")

   team = PantheonTeam([researcher, writer])

   response = await team.run("Write about AI safety")
   print(response.content)

Streaming
---------

.. code-block:: python

   async for chunk in agent.stream("Tell me a story"):
       print(chunk.content, end="", flush=True)

Memory
------

.. code-block:: python

   # Enable memory persistence
   agent = Agent(
       name="assistant",
       instructions="...",
       use_memory=True
   )

   # Conversations are saved and can be resumed

Common Patterns
---------------

**Error Handling**

.. code-block:: python

   try:
       response = await agent.run("...")
   except Exception as e:
       print(f"Error: {e}")

**Timeout**

.. code-block:: python

   response = await asyncio.wait_for(
       agent.run("..."),
       timeout=60.0
   )

**Multiple Queries**

.. code-block:: python

   queries = ["Q1", "Q2", "Q3"]
   responses = await asyncio.gather(
       *[agent.run(q) for q in queries]
   )

Next Steps
----------

- :doc:`agent` - Full Agent API
- :doc:`team` - Team patterns
- :doc:`toolsets` - Available toolsets
