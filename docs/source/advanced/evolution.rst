Evolution System
================

Automatic agent improvement through evaluation and iteration.

Overview
--------

The Evolution System automatically improves agents by:

- Evaluating agent performance on tasks
- Generating variations of agent configurations
- Selecting best-performing variations
- Iterating to refine agents over time

.. code-block:: text

   ┌─────────────────────────────────────────┐
   │          Evolution Pipeline             │
   │                                         │
   │  ┌─────────┐    ┌─────────┐            │
   │  │ Initial │───▶│ Evaluate │           │
   │  │ Agent   │    │ Performance│          │
   │  └─────────┘    └────┬──────┘          │
   │                      │                  │
   │  ┌─────────┐    ┌────▼──────┐          │
   │  │ Select  │◀───│ Generate  │          │
   │  │ Best    │    │ Variations│          │
   │  └────┬────┘    └───────────┘          │
   │       │                                 │
   │       ▼                                 │
   │  ┌─────────┐                           │
   │  │Improved │                           │
   │  │ Agent   │                           │
   │  └─────────┘                           │
   └─────────────────────────────────────────┘

Basic Usage
-----------

.. code-block:: python

   from pantheon import Agent
   from pantheon.evolution import evolve_agent

   # Starting agent
   agent = Agent(
       name="assistant",
       model="gpt-4o",
       instructions="You are helpful."
   )

   # Define evaluation tasks
   tasks = [
       {"input": "Summarize this text", "expected": "..."},
       {"input": "Answer this question", "expected": "..."},
   ]

   # Evolve the agent
   improved_agent = await evolve_agent(
       agent=agent,
       tasks=tasks,
       generations=5
   )

Evaluation Functions
--------------------

Define how to evaluate agent performance:

**Built-in Evaluators**

.. code-block:: python

   from pantheon.evolution import (
       exact_match_evaluator,
       semantic_similarity_evaluator,
       llm_judge_evaluator
   )

   # Exact match
   improved = await evolve_agent(
       agent=agent,
       tasks=tasks,
       evaluator=exact_match_evaluator
   )

   # Semantic similarity
   improved = await evolve_agent(
       agent=agent,
       tasks=tasks,
       evaluator=semantic_similarity_evaluator
   )

   # LLM as judge
   improved = await evolve_agent(
       agent=agent,
       tasks=tasks,
       evaluator=llm_judge_evaluator(judge_model="gpt-4o")
   )

**Custom Evaluator**

.. code-block:: python

   def custom_evaluator(response: str, expected: str) -> float:
       """Return score between 0 and 1."""
       # Your evaluation logic
       if "key_phrase" in response:
           return 1.0
       return 0.5

   improved = await evolve_agent(
       agent=agent,
       tasks=tasks,
       evaluator=custom_evaluator
   )

Mutation Strategies
-------------------

Control how variations are generated:

.. code-block:: python

   from pantheon.evolution import (
       instruction_mutation,
       temperature_mutation,
       tool_mutation
   )

   improved = await evolve_agent(
       agent=agent,
       tasks=tasks,
       mutations=[
           instruction_mutation(strength=0.3),
           temperature_mutation(range=(0.1, 0.9))
       ]
   )

**Available Mutations**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mutation
     - Description
   * - ``instruction_mutation``
     - Modify system instructions
   * - ``temperature_mutation``
     - Adjust model temperature
   * - ``tool_mutation``
     - Add/remove tools
   * - ``model_mutation``
     - Try different models

Evolution Parameters
--------------------

.. code-block:: python

   improved = await evolve_agent(
       agent=agent,
       tasks=tasks,
       generations=10,        # Number of evolution cycles
       population_size=5,     # Variants per generation
       selection_size=2,      # Top performers to keep
       mutation_rate=0.3,     # Probability of mutation
       crossover_rate=0.2     # Probability of combining agents
   )

Team Evolution
--------------

Evolve entire teams:

.. code-block:: python

   from pantheon.team import PantheonTeam
   from pantheon.evolution import evolve_team

   team = PantheonTeam(agents=[...])

   improved_team = await evolve_team(
       team=team,
       tasks=complex_tasks,
       generations=5
   )

Tracking Progress
-----------------

Monitor evolution progress:

.. code-block:: python

   from pantheon.evolution import evolve_agent, EvolutionCallback

   class ProgressCallback(EvolutionCallback):
       def on_generation(self, gen, population, scores):
           print(f"Gen {gen}: Best score = {max(scores):.3f}")

       def on_complete(self, best_agent, best_score):
           print(f"Evolution complete: {best_score:.3f}")

   improved = await evolve_agent(
       agent=agent,
       tasks=tasks,
       callback=ProgressCallback()
   )

Saving Results
--------------

Save evolved agents:

.. code-block:: python

   # Save as template
   improved.save_template(".pantheon/agents/evolved_assistant.md")

   # Save evolution history
   from pantheon.evolution import save_evolution_history

   save_evolution_history(
       history=evolution_history,
       path=".pantheon/evolution/run_001.json"
   )

Prompt Optimization
-------------------

Specific optimization for prompts:

.. code-block:: python

   from pantheon.evolution import optimize_prompt

   optimized_instructions = await optimize_prompt(
       initial_prompt="You are a helpful assistant.",
       tasks=tasks,
       model="gpt-4o",
       iterations=10
   )

   agent = Agent(
       name="optimized",
       instructions=optimized_instructions
   )

A/B Testing
-----------

Compare agent versions:

.. code-block:: python

   from pantheon.evolution import ab_test

   results = await ab_test(
       agent_a=original_agent,
       agent_b=evolved_agent,
       tasks=test_tasks,
       num_trials=100
   )

   print(f"Agent A: {results['a']['score']:.3f}")
   print(f"Agent B: {results['b']['score']:.3f}")
   print(f"Winner: {results['winner']}")

Best Practices
--------------

1. **Quality Tasks**: Use diverse, representative evaluation tasks
2. **Sufficient Data**: More tasks lead to better evolution
3. **Appropriate Evaluator**: Match evaluator to your success criteria
4. **Start Simple**: Begin with fewer generations, increase as needed
5. **Save Checkpoints**: Save intermediate results during long runs
6. **Validate Results**: Test evolved agents on held-out tasks
