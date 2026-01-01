Providers
=========

Providers abstract different tool sources and external connections.

Provider Types
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Provider
     - Description
   * - ``MCPProvider``
     - Connect to MCP (Model Context Protocol) servers
   * - ``LocalProvider``
     - Wrap local toolsets
   * - ``RemoteProvider``
     - Connect to remote Pantheon endpoints

MCP Provider
------------

Connect to MCP servers for external tools:

.. code-block:: python

   from pantheon.providers import MCPProvider

   # Filesystem server
   fs_mcp = MCPProvider("npx -y @anthropic/mcp-server-filesystem .")

   # GitHub server
   github_mcp = MCPProvider(
       "npx -y @anthropic/mcp-server-github",
       env={"GITHUB_TOKEN": os.environ["GITHUB_TOKEN"]}
   )

   agent = Agent(
       name="assistant",
       tools=[fs_mcp, github_mcp]
   )

**From Configuration**

Use servers defined in ``mcp.json``:

.. code-block:: python

   from pantheon.providers import MCPProvider

   # Load from .pantheon/mcp.json
   mcp = MCPProvider.from_config("github")

   agent = Agent(tools=[mcp])

**Tool Filtering**

Only expose specific tools:

.. code-block:: python

   mcp = MCPProvider(
       "npx -y @anthropic/mcp-server-filesystem",
       filter_tools=["read_file", "list_directory"]  # Only these tools
   )

   # Or by prefix
   mcp = MCPProvider(
       "...",
       filter_prefix="fs_"  # Only tools starting with "fs_"
   )

Local Provider
--------------

Wrap local toolsets with provider interface:

.. code-block:: python

   from pantheon.providers import LocalProvider
   from pantheon.toolsets import FileManagerToolSet

   provider = LocalProvider(FileManagerToolSet())

   # Useful for uniform handling of different tool sources
   agent = Agent(tools=[provider])

Remote Provider
---------------

Connect to remote Pantheon endpoints:

.. code-block:: python

   from pantheon.providers import RemoteProvider

   # Connect to remote endpoint
   remote = RemoteProvider(
       url="http://localhost:8080",
       api_key="..."
   )

   agent = Agent(tools=[remote])

**With NATS Backend**

.. code-block:: python

   from pantheon.providers import RemoteProvider

   remote = RemoteProvider(
       backend="nats",
       nats_url="nats://localhost:4222"
   )

Provider Lifecycle
------------------

.. code-block:: text

   1. Creation
      provider = MCPProvider(...)
           │
   2. Connection (on first use)
      provider connects to server/endpoint
           │
   3. Tool Discovery
      provider fetches available tools
           │
   4. Tool Execution
      agent calls tools through provider
           │
   5. Disconnection (on cleanup)
      provider disconnects

Managing Connections
--------------------

**Explicit Lifecycle**

.. code-block:: python

   mcp = MCPProvider("...")

   # Connect
   await mcp.connect()

   # Use
   agent = Agent(tools=[mcp])
   await agent.run("...")

   # Disconnect
   await mcp.disconnect()

**Context Manager**

.. code-block:: python

   async with MCPProvider("...") as mcp:
       agent = Agent(tools=[mcp])
       await agent.run("...")
   # Automatically disconnected

Multiple Providers
------------------

Combine multiple providers:

.. code-block:: python

   from pantheon.providers import MCPProvider, LocalProvider
   from pantheon.toolsets import PythonInterpreterToolSet

   agent = Agent(
       name="full_stack",
       tools=[
           MCPProvider("npx -y @anthropic/mcp-server-filesystem"),
           MCPProvider("npx -y @anthropic/mcp-server-github"),
           LocalProvider(PythonInterpreterToolSet())
       ]
   )

Provider Events
---------------

Monitor provider activity:

.. code-block:: python

   mcp = MCPProvider("...")

   @mcp.on_connect
   def on_connect():
       print("Connected to MCP server")

   @mcp.on_tool_call
   def on_tool(name, args, result):
       print(f"Tool {name} called: {result[:100]}")

   @mcp.on_disconnect
   def on_disconnect():
       print("Disconnected")

Error Handling
--------------

Handle provider errors:

.. code-block:: python

   from pantheon.exceptions import ProviderError

   try:
       mcp = MCPProvider("...")
       await mcp.connect()
   except ProviderError as e:
       print(f"Failed to connect: {e}")
       # Use fallback

**Reconnection**

.. code-block:: python

   mcp = MCPProvider(
       "...",
       auto_reconnect=True,  # Auto-reconnect on failure
       max_retries=3
   )

Custom Providers
----------------

Create custom providers:

.. code-block:: python

   from pantheon.providers import Provider

   class MyProvider(Provider):
       def __init__(self, config):
           self.config = config

       async def connect(self):
           # Connect to your service
           pass

       async def disconnect(self):
           # Clean up
           pass

       def get_tools(self):
           # Return list of tool definitions
           return [
               {
                   "name": "my_tool",
                   "description": "Does something",
                   "parameters": {...}
               }
           ]

       async def execute_tool(self, name, arguments):
           # Execute the tool
           if name == "my_tool":
               return await self._do_something(arguments)

Best Practices
--------------

1. **Reuse Providers**: Create once, use across multiple agents
2. **Handle Disconnections**: Use auto_reconnect or explicit error handling
3. **Filter Tools**: Only expose tools the agent needs
4. **Use Config Files**: Define MCP servers in ``mcp.json``
5. **Clean Up**: Disconnect providers when done
