Core Components
===============

Deep dive into Pantheon's core building blocks.

Overview
--------

Pantheon is built on these core components:

.. code-block:: text

   ┌─────────────────────────────────────────┐
   │              Your Application            │
   └─────────────────┬───────────────────────┘
                     │
   ┌─────────────────▼───────────────────────┐
   │     Agent / Team (Orchestration)         │
   └─────────────────┬───────────────────────┘
                     │
   ┌────────┬────────┼────────┬──────────────┐
   │        │        │        │              │
   ▼        ▼        ▼        ▼              ▼
 Tools   Memory  Providers  Models       Config
   │        │        │        │              │
   └────────┴────────┴────────┴──────────────┘

Component Summary
-----------------

**Agents**
   The fundamental unit of intelligence. An agent combines a model with instructions and tools.

**Teams**
   Multiple agents working together. Supports various collaboration patterns (orchestrated, sequential, swarm).

**Toolsets**
   Collections of tools that give agents capabilities like file access, code execution, and web search.

**Memory**
   Conversation history and context management. Supports persistence and compression.

**Providers**
   Abstractions for external tool sources (MCP servers, remote endpoints).

Relationships
-------------

.. code-block:: python

   from pantheon import Agent, Team
   from pantheon.toolsets import FileManagerToolSet
   from pantheon.memory import Memory
   from pantheon.providers import MCPProvider

   # An agent uses tools and memory
   agent = Agent(
       name="assistant",
       model="gpt-4o",
       tools=[FileManagerToolSet()],
       memory=Memory()
   )

   # A team coordinates multiple agents
   team = Team(
       name="dev_team",
       agents=[agent1, agent2, agent3]
   )

   # Providers connect to external tools
   mcp = MCPProvider("npx -y @anthropic/mcp-server-github")
   agent_with_mcp = Agent(
       name="github_agent",
       tools=[mcp]
   )

When to Use What
----------------

**Single Agent**
   Simple tasks, single domain expertise, straightforward workflows.

**Team**
   Complex tasks, multiple expertise areas, tasks requiring review/iteration.

**Custom Toolsets**
   When built-in tools don't cover your needs.

**MCP Providers**
   Integrating with external services, using community tools.

Next Sections
-------------

- :doc:`agents` - Agent architecture and API
- :doc:`teams` - Team patterns and orchestration
- :doc:`toolsets` - Tool system details
- :doc:`memory` - Memory and context management
- :doc:`providers` - Provider abstractions
