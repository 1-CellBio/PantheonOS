Learning System
===============

Agents that learn and improve from experience.

Overview
--------

The Learning System enables agents to:

- Record successful task completions as "skills"
- Retrieve relevant skills for new tasks
- Improve performance over time

.. code-block:: text

   ┌─────────────────────────────────────────┐
   │           Learning System               │
   │  ┌─────────────────────────────────┐   │
   │  │         Skillbook               │   │
   │  │  ┌─────┐ ┌─────┐ ┌─────┐       │   │
   │  │  │Skill│ │Skill│ │Skill│  ...  │   │
   │  │  └─────┘ └─────┘ └─────┘       │   │
   │  └─────────────────────────────────┘   │
   │  ┌─────────────────────────────────┐   │
   │  │     Trajectory Tracking         │   │
   │  │  (Records agent interactions)   │   │
   │  └─────────────────────────────────┘   │
   └─────────────────────────────────────────┘

Configuration
-------------

Enable learning in ``settings.json``:

.. code-block:: json

   {
     "learning": {
       "enabled": true,
       "skillbook_path": ".pantheon/skills",
       "trajectory_tracking": true
     }
   }

Skillbook
---------

The Skillbook stores learned patterns:

**Skill Structure**

.. code-block:: json

   {
     "id": "skill_123",
     "name": "Parse JSON API Response",
     "description": "Extract data from nested JSON structures",
     "pattern": "When parsing JSON with nested objects...",
     "example": {
       "input": "Parse this API response: {...}",
       "output": "Here's how to extract the data..."
     },
     "metadata": {
       "success_count": 15,
       "created_at": "2024-01-15T10:30:00Z",
       "tags": ["json", "api", "parsing"]
     }
   }

**Using Skills**

.. code-block:: python

   from pantheon import Agent
   from pantheon.learning import Skillbook

   skillbook = Skillbook(".pantheon/skills")

   agent = Agent(
       name="assistant",
       skillbook=skillbook
   )

   # Agent automatically retrieves relevant skills
   response = await agent.run("Parse this JSON response")

Recording Skills
----------------

**Automatic Recording**

With trajectory tracking enabled, successful interactions are recorded:

.. code-block:: python

   agent = Agent(
       name="learner",
       learning_enabled=True
   )

   # This interaction may be recorded as a skill
   await agent.run("Write a function to sort a list")

**Manual Recording**

.. code-block:: python

   from pantheon.learning import Skillbook, Skill

   skillbook = Skillbook()

   skill = Skill(
       name="API Error Handling",
       description="Handle common API errors gracefully",
       pattern="""
       When encountering API errors:
       1. Check status code
       2. Parse error message
       3. Implement retry logic for transient errors
       """,
       example={
           "input": "Handle this API error",
           "output": "Implemented retry with exponential backoff"
       }
   )

   skillbook.add(skill)

Trajectory Tracking
-------------------

Records agent interactions for later analysis:

.. code-block:: python

   from pantheon import Agent
   from pantheon.learning import TrajectoryTracker

   tracker = TrajectoryTracker(".pantheon/trajectories")

   agent = Agent(
       name="tracked_agent",
       trajectory_tracker=tracker
   )

   # All interactions are recorded
   await agent.run("Complex task")

   # Review trajectories
   trajectories = tracker.get_recent(n=10)
   for t in trajectories:
       print(f"Task: {t.task}")
       print(f"Steps: {len(t.steps)}")
       print(f"Success: {t.success}")

Skill Retrieval
---------------

When processing a new task, the agent retrieves relevant skills:

.. code-block:: python

   # Agent automatically searches skillbook
   response = await agent.run("Parse CSV data")

   # Behind the scenes:
   # 1. Agent embeds the task description
   # 2. Searches skillbook for similar patterns
   # 3. Includes relevant skills in context
   # 4. Generates response with skill guidance

**Manual Retrieval**

.. code-block:: python

   relevant_skills = skillbook.search(
       query="parsing structured data",
       top_k=5
   )

   for skill in relevant_skills:
       print(f"{skill.name}: {skill.description}")

Skill Refinement
----------------

Skills improve over time:

.. code-block:: python

   # Mark skill as successful
   skillbook.record_success(skill_id="skill_123")

   # Update skill with better example
   skillbook.update(
       skill_id="skill_123",
       example={
           "input": "Better example input",
           "output": "Improved solution"
       }
   )

   # Remove outdated skills
   skillbook.prune(min_success_rate=0.5)

Integration with Teams
----------------------

Skills can be shared across team members:

.. code-block:: python

   from pantheon.team import PantheonTeam
   from pantheon.learning import Skillbook

   shared_skillbook = Skillbook(".pantheon/team_skills")

   team = PantheonTeam(
       agents=[...],
       skillbook=shared_skillbook
   )

   # All agents benefit from shared learning

REPL Commands
-------------

In REPL mode:

.. code-block:: text

   > /skills                    # List learned skills
   > /skills search parsing     # Search skills
   > /learn                     # Start learning mode
   > /trajectory                # View recent trajectories

Best Practices
--------------

1. **Start Small**: Enable learning for specific task types first
2. **Review Skills**: Periodically review and prune the skillbook
3. **Tag Skills**: Use tags for better organization
4. **Share Skills**: Use shared skillbooks for teams
5. **Monitor Quality**: Track skill success rates
