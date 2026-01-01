API Reference
=============

Complete API documentation for Pantheon's classes and modules.

**Core Components:**

- :doc:`agent` - Agent API
- :doc:`team` - Team API
- :doc:`memory` - Memory API
- :doc:`chatroom` - ChatRoom API

**Utilities:**

- :doc:`reasoning` - Reasoning utilities
- :doc:`remote` - Remote execution
- :doc:`repl` - REPL API
- :doc:`utils` - Utility functions

Quick Reference
---------------

Core Classes
~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   pantheon.agent.Agent
   pantheon.team.Team
   pantheon.team.SequentialTeam
   pantheon.team.SwarmTeam
   pantheon.team.SwarmCenterTeam
   pantheon.team.MoATeam
   pantheon.memory.Memory

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from pantheon.agent import Agent
   from pantheon.team import SequentialTeam

   # Create an agent
   agent = Agent(
       name="my_agent",
       instructions="You are a helpful assistant.",
       model="gpt-4o-mini"
   )

   # Use in a team
   team = SequentialTeam([agent1, agent2])
   await team.chat()