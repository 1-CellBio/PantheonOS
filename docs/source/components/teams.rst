Teams
=====

Multi-agent collaboration patterns and orchestration.

Team Architecture
-----------------

.. code-block:: text

   ┌─────────────────────────────────────────────┐
   │                   Team                       │
   │  ┌───────────────────────────────────────┐  │
   │  │           Orchestrator                │  │
   │  │   (Coordinates agent interactions)    │  │
   │  └─────────────────┬─────────────────────┘  │
   │          ┌─────────┼─────────┐              │
   │          ▼         ▼         ▼              │
   │     ┌────────┐ ┌────────┐ ┌────────┐       │
   │     │ Agent1 │ │ Agent2 │ │ Agent3 │       │
   │     └────────┘ └────────┘ └────────┘       │
   └─────────────────────────────────────────────┘

Team Types
----------

Pantheon supports multiple team patterns:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Description
   * - **PantheonTeam**
     - Orchestrator-managed with dynamic task delegation
   * - **SwarmTeam**
     - Agents hand off to each other dynamically
   * - **SequentialTeam**
     - Agents process in fixed order
   * - **MoATeam**
     - Mixture of Agents with aggregation

PantheonTeam
------------

Central orchestrator delegates tasks to specialized agents:

.. code-block:: python

   from pantheon.team import PantheonTeam
   from pantheon import Agent

   team = PantheonTeam(
       name="dev_team",
       agents=[
           Agent(name="planner", instructions="Plan tasks."),
           Agent(name="developer", instructions="Write code."),
           Agent(name="reviewer", instructions="Review code.")
       ]
   )

   response = await team.run("Build a REST API")

**When to use:**
- Complex tasks requiring coordination
- Tasks with unclear agent routing
- Need for dynamic task breakdown

SwarmTeam
---------

Agents can hand off to each other based on context:

.. code-block:: python

   from pantheon.team import SwarmTeam
   from pantheon import Agent

   triage = Agent(
       name="triage",
       instructions="Assess issues and route to specialists."
   )
   tech_support = Agent(
       name="tech_support",
       instructions="Handle technical issues."
   )
   billing = Agent(
       name="billing",
       instructions="Handle billing questions."
   )

   team = SwarmTeam(
       name="support_team",
       agents=[triage, tech_support, billing],
       handoffs={
           "triage": ["tech_support", "billing"],
           "tech_support": ["triage"],
           "billing": ["triage"]
       }
   )

**When to use:**
- Customer support flows
- Tasks with clear routing logic
- Conversational agents with specializations

SequentialTeam
--------------

Agents process in a fixed pipeline:

.. code-block:: python

   from pantheon.team import SequentialTeam
   from pantheon import Agent

   team = SequentialTeam(
       name="content_pipeline",
       agents=[
           Agent(name="researcher", instructions="Research the topic."),
           Agent(name="writer", instructions="Write the content."),
           Agent(name="editor", instructions="Edit and polish.")
       ]
   )

   response = await team.run("Write about AI trends")
   # researcher -> writer -> editor

**When to use:**
- Content creation pipelines
- Data processing workflows
- Tasks with clear sequential stages

MoATeam (Mixture of Agents)
---------------------------

Multiple agents contribute, then an aggregator synthesizes:

.. code-block:: python

   from pantheon.team import MoATeam
   from pantheon import Agent

   team = MoATeam(
       name="analysis_team",
       agents=[
           Agent(name="technical", instructions="Technical analysis."),
           Agent(name="business", instructions="Business analysis."),
           Agent(name="user", instructions="User perspective.")
       ],
       aggregator=Agent(
           name="synthesizer",
           instructions="Synthesize all perspectives."
       )
   )

   response = await team.run("Evaluate this proposal")

**When to use:**
- Multi-perspective analysis
- Decision making requiring diverse viewpoints
- Tasks benefiting from consensus

AgentAsToolTeam
---------------

Use agents as tools callable by other agents:

.. code-block:: python

   from pantheon.team import AgentAsToolTeam
   from pantheon import Agent

   expert_python = Agent(name="python_expert", ...)
   expert_sql = Agent(name="sql_expert", ...)

   main_agent = Agent(
       name="coordinator",
       instructions="Use experts as needed."
   )

   team = AgentAsToolTeam(
       main_agent=main_agent,
       tool_agents=[expert_python, expert_sql]
   )

   # main_agent can call python_expert() or sql_expert() as tools

Team Communication
------------------

**Shared Context**

Teams maintain shared context between agents:

.. code-block:: python

   team = PantheonTeam(
       ...,
       shared_context=True  # Agents see each other's messages
   )

**Context Variables**

Pass data between agents:

.. code-block:: python

   team = PantheonTeam(...)

   response = await team.run(
       "Analyze this data",
       context_variables={"dataset": "sales_2024.csv"}
   )

Team Templates
--------------

Define teams in ``.pantheon/teams/``:

.. code-block:: markdown

   ---
   name: Developer Team
   team_type: pantheon
   agents:
     - name: planner
       instructions: Plan the implementation.
     - name: developer
       toolsets: [file_manager, shell]
       instructions: Write the code.
     - name: reviewer
       instructions: Review for quality.
   ---

   # Developer Team

   A team for software development.

Load with:

.. code-block:: python

   from pantheon.factory import load_team

   team = load_team("developer_team")

Best Practices
--------------

1. **Clear Role Separation**: Each agent should have a distinct responsibility
2. **Minimal Overlap**: Avoid agents with overlapping capabilities
3. **Right Team Type**: Match the pattern to your workflow
4. **Appropriate Size**: Start small, add agents as needed

API Reference
-------------

See :doc:`/interfaces/api/team` for complete API documentation.

Detailed Team Documentation
---------------------------

For in-depth documentation of each team type:

.. toctree::
   :maxdepth: 1

   /team/pantheon_team
   /team/swarm_team
   /team/sequential_team
   /team/moa_team
   /team/swarm_center_team
