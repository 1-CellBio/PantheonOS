MCP Configuration
=================

MCP (Model Context Protocol) enables agents to use external tool servers.

Configuration File
------------------

MCP servers are configured in ``.pantheon/mcp.json``:

.. code-block:: json

   {
     "mcpServers": {
       "filesystem": {
         "command": "npx",
         "args": ["-y", "@anthropic/mcp-server-filesystem", "/path/to/workspace"]
       },
       "github": {
         "command": "npx",
         "args": ["-y", "@anthropic/mcp-server-github"],
         "env": {
           "GITHUB_TOKEN": "${GITHUB_TOKEN}"
         }
       }
     }
   }

Server Configuration
--------------------

Each server entry has:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Description
   * - ``command``
     - Executable to run (e.g., ``npx``, ``python``, ``node``)
   * - ``args``
     - Command arguments as array
   * - ``env``
     - Environment variables (supports ``${VAR}`` substitution)
   * - ``cwd``
     - Working directory for the server

Common MCP Servers
------------------

Filesystem Server
~~~~~~~~~~~~~~~~~

Access local files:

.. code-block:: json

   {
     "mcpServers": {
       "filesystem": {
         "command": "npx",
         "args": ["-y", "@anthropic/mcp-server-filesystem", "."]
       }
     }
   }

GitHub Server
~~~~~~~~~~~~~

GitHub operations:

.. code-block:: json

   {
     "mcpServers": {
       "github": {
         "command": "npx",
         "args": ["-y", "@anthropic/mcp-server-github"],
         "env": {
           "GITHUB_TOKEN": "${GITHUB_TOKEN}"
         }
       }
     }
   }

PostgreSQL Server
~~~~~~~~~~~~~~~~~

Database operations:

.. code-block:: json

   {
     "mcpServers": {
       "postgres": {
         "command": "npx",
         "args": ["-y", "@anthropic/mcp-server-postgres"],
         "env": {
           "DATABASE_URL": "${DATABASE_URL}"
         }
       }
     }
   }

Puppeteer Server
~~~~~~~~~~~~~~~~

Browser automation:

.. code-block:: json

   {
     "mcpServers": {
       "puppeteer": {
         "command": "npx",
         "args": ["-y", "@anthropic/mcp-server-puppeteer"]
       }
     }
   }

Custom Python Server
~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "mcpServers": {
       "custom": {
         "command": "python",
         "args": ["-m", "my_mcp_server"],
         "cwd": "/path/to/server"
       }
     }
   }

Using MCP in Templates
----------------------

In Agent Templates
~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

   ---
   name: GitHub Assistant
   model: openai/gpt-4o
   mcp_servers:
     - github
     - filesystem
   ---

   You are a GitHub assistant.

In Team Templates
~~~~~~~~~~~~~~~~~

.. code-block:: markdown

   ---
   name: Dev Team
   agents:
     - name: developer
       mcp_servers:
         - github
         - filesystem
       instructions: Write and commit code.
   ---

Using MCP in Python API
-----------------------

.. code-block:: python

   from pantheon import Agent
   from pantheon.providers import MCPProvider

   # Create MCP provider
   mcp = MCPProvider("npx -y @anthropic/mcp-server-filesystem .")

   # Use with agent
   agent = Agent(
       name="assistant",
       model="gpt-4o",
       tools=[mcp]
   )

   response = await agent.run("List files in current directory")

Environment Variables
---------------------

Use ``${VAR}`` syntax for environment variable substitution:

.. code-block:: json

   {
     "mcpServers": {
       "database": {
         "command": "python",
         "args": ["-m", "db_server"],
         "env": {
           "DB_HOST": "${DB_HOST}",
           "DB_USER": "${DB_USER}",
           "DB_PASSWORD": "${DB_PASSWORD}"
         }
       }
     }
   }

Variables are resolved from:

1. Process environment
2. ``.env`` file in project root
3. Shell environment

Server Lifecycle
----------------

MCP servers are:

1. **Started on demand** when an agent with that server is created
2. **Kept running** for the session duration
3. **Stopped** when the session ends or explicitly disconnected

Multiple Usage
--------------

The same MCP server can be used by multiple agents:

.. code-block:: json

   {
     "mcpServers": {
       "shared-fs": {
         "command": "npx",
         "args": ["-y", "@anthropic/mcp-server-filesystem", "."]
       }
     }
   }

.. code-block:: markdown

   ---
   name: Team
   agents:
     - name: reader
       mcp_servers: [shared-fs]
       instructions: Read files.
     - name: writer
       mcp_servers: [shared-fs]
       instructions: Write files.
   ---

Troubleshooting
---------------

Server Not Starting
~~~~~~~~~~~~~~~~~~~

Check:

1. Command is installed (``npx``, ``python``, etc.)
2. Package exists (``npm install -g @anthropic/mcp-server-xxx``)
3. Environment variables are set

View Logs
~~~~~~~~~

Enable debug logging:

.. code-block:: bash

   export PANTHEON_LOG_LEVEL=DEBUG
   python -m pantheon.repl

Connection Issues
~~~~~~~~~~~~~~~~~

If a server fails to connect:

1. Test the command manually in terminal
2. Check for port conflicts
3. Verify environment variables
4. Check server-specific requirements

Finding MCP Servers
-------------------

- `Anthropic MCP Servers <https://github.com/anthropics/mcp-servers>`_
- `MCP Server Registry <https://github.com/anthropics/mcp>`_
- Community servers on GitHub

Creating Custom Servers
-----------------------

See the `MCP Specification <https://github.com/anthropics/mcp>`_ for creating your own servers.

Basic Python server:

.. code-block:: python

   from mcp import Server, Tool

   server = Server("my-server")

   @server.tool
   def my_tool(param: str) -> str:
       """Tool description."""
       return f"Result: {param}"

   if __name__ == "__main__":
       server.run()
