Memory
======

Conversation history and context management.

Memory Architecture
-------------------

.. code-block:: text

   ┌─────────────────────────────────────┐
   │              Memory                  │
   │  ┌─────────────────────────────┐    │
   │  │     Message History         │    │
   │  │  [user, assistant, user...] │    │
   │  └─────────────────────────────┘    │
   │  ┌─────────────────────────────┐    │
   │  │    Context Variables        │    │
   │  │  {key: value, ...}          │    │
   │  └─────────────────────────────┘    │
   │  ┌─────────────────────────────┐    │
   │  │    Compression/Summary      │    │
   │  │  (for long conversations)   │    │
   │  └─────────────────────────────┘    │
   └─────────────────────────────────────┘

Basic Usage
-----------

**Default Memory**

Agents have memory by default:

.. code-block:: python

   from pantheon import Agent

   agent = Agent(name="assistant", ...)

   await agent.run("My name is Alice")
   await agent.run("What's my name?")  # Remembers "Alice"

**Custom Memory**

.. code-block:: python

   from pantheon.memory import Memory

   memory = Memory()
   agent = Agent(name="assistant", memory=memory)

Message Types
-------------

Memory stores different message types:

.. code-block:: python

   memory = Memory()

   # User message
   memory.append({"role": "user", "content": "Hello"})

   # Assistant message
   memory.append({"role": "assistant", "content": "Hi there!"})

   # System message
   memory.append({"role": "system", "content": "Be helpful"})

   # Tool results
   memory.append({
       "role": "tool",
       "tool_call_id": "call_123",
       "content": "Result of tool execution"
   })

Memory Operations
-----------------

**Access History**

.. code-block:: python

   # Get all messages
   messages = memory.messages

   # Get last N messages
   recent = memory.get_recent(n=10)

   # Get message count
   count = len(memory)

**Clear Memory**

.. code-block:: python

   memory.clear()

**Add Messages**

.. code-block:: python

   memory.append({"role": "user", "content": "Hello"})

   # Or multiple at once
   memory.extend([
       {"role": "user", "content": "Question"},
       {"role": "assistant", "content": "Answer"}
   ])

Persistence
-----------

**Save to File**

.. code-block:: python

   memory.save("conversation.json")

**Load from File**

.. code-block:: python

   memory = Memory.load("conversation.json")

   agent = Agent(
       name="assistant",
       memory=memory  # Resume conversation
   )

**Auto-save**

.. code-block:: python

   memory = Memory(auto_save="conversation.json")

   # Memory automatically saved after each interaction

Memory Compression
------------------

For long conversations, compress older messages:

.. code-block:: python

   from pantheon.internal.compression import compress_memory

   # Compress when memory gets large
   if len(memory) > 50:
       memory = await compress_memory(
           memory,
           model="gpt-4o-mini",
           keep_recent=10  # Keep last 10 messages unchanged
       )

**How Compression Works:**

1. Older messages are summarized by an LLM
2. Summary replaces the original messages
3. Recent messages are kept intact
4. Context is preserved while reducing token count

**Configuration**

In ``settings.json``:

.. code-block:: json

   {
     "context_compression": {
       "enabled": true,
       "threshold_tokens": 100000,
       "target_tokens": 50000
     }
   }

Context Variables
-----------------

Store structured data alongside conversation:

.. code-block:: python

   memory = Memory()

   # Set context
   memory.context["user_name"] = "Alice"
   memory.context["preferences"] = {"theme": "dark"}

   # Access in agent
   agent = Agent(
       name="assistant",
       memory=memory,
       instructions="""
       User name: {context[user_name]}
       Help them with their tasks.
       """
   )

Memory with Teams
-----------------

**Shared Memory**

Team agents can share memory:

.. code-block:: python

   from pantheon.team import PantheonTeam
   from pantheon.memory import Memory

   shared_memory = Memory()

   team = PantheonTeam(
       agents=[agent1, agent2],
       memory=shared_memory
   )

**Isolated Memory**

Each agent maintains separate memory:

.. code-block:: python

   team = PantheonTeam(
       agents=[
           Agent(name="a1", memory=Memory()),
           Agent(name="a2", memory=Memory())
       ],
       shared_memory=False
   )

Memory Limits
-------------

Prevent unbounded growth:

.. code-block:: python

   memory = Memory(max_messages=100)

   # Oldest messages are removed when limit reached

Token Counting
--------------

Monitor token usage:

.. code-block:: python

   # Estimate tokens in memory
   token_count = memory.estimate_tokens()

   if token_count > 50000:
       memory = await compress_memory(memory)

Session Management
------------------

For multi-session applications:

.. code-block:: python

   import os
   from pantheon.memory import Memory

   def get_memory(session_id: str) -> Memory:
       path = f"sessions/{session_id}.json"
       if os.path.exists(path):
           return Memory.load(path)
       return Memory(auto_save=path)

   # Each user gets their own memory
   memory = get_memory(user_session_id)
   agent = Agent(memory=memory)

Best Practices
--------------

1. **Use Compression** for long-running conversations
2. **Persist Important Conversations** with save/load
3. **Set Limits** to prevent memory from growing too large
4. **Clear When Done** to free resources
5. **Use Context Variables** for structured data, not conversation history
