Agents
======

Deep dive into agent architecture and capabilities.

Agent Architecture
------------------

.. code-block:: text

   ┌───────────────────────────────────────┐
   │               Agent                   │
   │  ┌─────────────┐  ┌────────────────┐ │
   │  │ Instructions│  │     Model      │ │
   │  │  (System)   │  │  (LLM Engine)  │ │
   │  └─────────────┘  └────────────────┘ │
   │  ┌─────────────┐  ┌────────────────┐ │
   │  │    Tools    │  │    Memory      │ │
   │  │ (Capabilities│  │  (History)    │ │
   │  └─────────────┘  └────────────────┘ │
   └───────────────────────────────────────┘

Core Properties
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Property
     - Description
   * - ``name``
     - Unique identifier for the agent
   * - ``model``
     - LLM model to use (e.g., ``openai/gpt-4o``)
   * - ``instructions``
     - System prompt defining agent behavior
   * - ``tools``
     - List of tools/toolsets the agent can use
   * - ``memory``
     - Conversation history management

Creating Agents
---------------

**Basic Agent**

.. code-block:: python

   from pantheon import Agent

   agent = Agent(
       name="assistant",
       model="openai/gpt-4o",
       instructions="You are a helpful assistant."
   )

**Agent with Tools**

.. code-block:: python

   from pantheon import Agent
   from pantheon.toolsets import FileManagerToolSet, ShellToolSet

   agent = Agent(
       name="developer",
       model="openai/gpt-4o",
       instructions="You are a developer assistant.",
       tools=[
           FileManagerToolSet(),
           ShellToolSet()
       ]
   )

**Agent from Template**

.. code-block:: python

   from pantheon.factory import load_agent

   agent = load_agent("developer")  # Loads .pantheon/agents/developer.md

Agent Lifecycle
---------------

.. code-block:: text

   1. Creation
      Agent(name, model, instructions, tools)
           │
   2. Run/Chat
      agent.run("user message")
           │
   3. Model Call
      LLM processes message with tools
           │
   4. Tool Execution (if needed)
      Agent executes tool calls
           │
   5. Response
      Final response returned

Running Agents
--------------

**Single Response**

.. code-block:: python

   response = await agent.run("Help me write a function")
   print(response.content)

**Interactive Chat**

.. code-block:: python

   await agent.chat()  # Starts interactive loop

**Streaming**

.. code-block:: python

   async for chunk in agent.stream("Tell me a story"):
       print(chunk.content, end="", flush=True)

Tool Execution
--------------

When the model requests tool use:

1. Agent parses tool call from model response
2. Finds matching tool in registered tools
3. Executes tool with provided arguments
4. Sends result back to model
5. Model generates final response

.. code-block:: python

   @agent.tool
   def calculate(expression: str) -> float:
       """Calculate a math expression."""
       return eval(expression)

   response = await agent.run("What is 25 * 4?")
   # Agent calls calculate("25 * 4") -> 100
   # Response: "25 times 4 equals 100"

Memory Management
-----------------

Agents maintain conversation history:

.. code-block:: python

   from pantheon.memory import Memory

   memory = Memory()
   agent = Agent(
       name="assistant",
       memory=memory
   )

   # Conversation builds over multiple calls
   await agent.run("My name is Alice")
   await agent.run("What's my name?")  # Remembers "Alice"

**Clearing Memory**

.. code-block:: python

   agent.memory.clear()

**Saving/Loading**

.. code-block:: python

   agent.memory.save("conversation.json")
   agent.memory = Memory.load("conversation.json")

Agent Events
------------

Monitor agent activity:

.. code-block:: python

   from pantheon import Agent

   agent = Agent(...)

   @agent.on_tool_call
   def log_tool(name, args):
       print(f"Tool called: {name}({args})")

   @agent.on_response
   def log_response(response):
       print(f"Response: {response.content[:100]}...")

Advanced Patterns
-----------------

**Parallel Tool Execution**

.. code-block:: python

   agent = Agent(
       ...,
       parallel_tool_calls=True  # Execute multiple tools simultaneously
   )

**Custom Model Parameters**

.. code-block:: python

   agent = Agent(
       ...,
       temperature=0.7,
       max_tokens=2000,
       top_p=0.9
   )

**Agent Cloning**

.. code-block:: python

   # Create variation of existing agent
   creative_agent = agent.clone(
       name="creative_assistant",
       temperature=0.9
   )

API Reference
-------------

See :doc:`/interfaces/api/agent` for complete API documentation.
