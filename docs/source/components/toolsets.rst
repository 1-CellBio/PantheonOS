Toolsets
========

Tools give agents capabilities to interact with the world.

Toolset Architecture
--------------------

.. code-block:: text

   ┌──────────────────────────────────────┐
   │              ToolSet                  │
   │  ┌──────┐ ┌──────┐ ┌──────┐         │
   │  │ Tool │ │ Tool │ │ Tool │  ...    │
   │  └──────┘ └──────┘ └──────┘         │
   │                                       │
   │  • Tool definitions (name, schema)   │
   │  • Execution handlers                │
   │  • Permission controls               │
   └──────────────────────────────────────┘

Built-in Toolsets
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Toolset
     - Description
   * - ``FileManagerToolSet``
     - Read, write, edit, search files
   * - ``ShellToolSet``
     - Execute shell commands
   * - ``PythonInterpreterToolSet``
     - Run Python code
   * - ``IntegratedNotebookToolSet``
     - Jupyter notebook operations
   * - ``WebToolSet``
     - Web search and fetch
   * - ``KnowledgeToolSet``
     - Vector store and RAG

Using Toolsets
--------------

.. code-block:: python

   from pantheon import Agent
   from pantheon.toolsets import (
       FileManagerToolSet,
       ShellToolSet,
       PythonInterpreterToolSet
   )

   agent = Agent(
       name="developer",
       model="gpt-4o",
       tools=[
           FileManagerToolSet(),
           ShellToolSet(),
           PythonInterpreterToolSet()
       ]
   )

FileManagerToolSet
------------------

File operations:

.. code-block:: python

   from pantheon.toolsets import FileManagerToolSet

   file_tools = FileManagerToolSet()

**Tools provided:**

- ``read_file(path)`` - Read file contents
- ``write_file(path, content)`` - Write to file
- ``edit_file(path, old, new)`` - Replace text in file
- ``search_files(pattern)`` - Search file contents
- ``list_directory(path)`` - List directory contents

ShellToolSet
------------

Shell command execution:

.. code-block:: python

   from pantheon.toolsets import ShellToolSet

   shell_tools = ShellToolSet(
       timeout=60,  # Command timeout in seconds
       allowed_commands=["git", "npm", "python"]  # Optional whitelist
   )

**Tools provided:**

- ``run_command(command)`` - Execute shell command

PythonInterpreterToolSet
------------------------

Python code execution:

.. code-block:: python

   from pantheon.toolsets import PythonInterpreterToolSet

   python_tools = PythonInterpreterToolSet()

**Tools provided:**

- ``execute_python(code)`` - Run Python code
- ``execute_python_file(path)`` - Run Python file

IntegratedNotebookToolSet
-------------------------

Jupyter notebook operations:

.. code-block:: python

   from pantheon.toolsets import IntegratedNotebookToolSet

   notebook_tools = IntegratedNotebookToolSet()

**Tools provided:**

- ``create_notebook(path)`` - Create new notebook
- ``add_cell(path, cell_type, content)`` - Add cell
- ``execute_cell(path, cell_index)`` - Execute cell
- ``read_notebook(path)`` - Read notebook contents

WebToolSet
----------

Web browsing capabilities:

.. code-block:: python

   from pantheon.toolsets import WebToolSet

   web_tools = WebToolSet()

**Tools provided:**

- ``web_search(query)`` - Search the web
- ``fetch_url(url)`` - Fetch page content

Creating Custom Toolsets
------------------------

**Simple Tool Function**

.. code-block:: python

   from pantheon import Agent

   agent = Agent(name="assistant", ...)

   @agent.tool
   def get_weather(city: str) -> str:
       """Get current weather for a city."""
       # Implementation
       return f"Weather in {city}: Sunny, 72°F"

**ToolSet Class**

.. code-block:: python

   from pantheon import ToolSet, tool

   class WeatherToolSet(ToolSet):
       def __init__(self, api_key: str):
           self.api_key = api_key

       @tool
       def get_weather(self, city: str) -> str:
           """Get current weather for a city.

           Args:
               city: Name of the city

           Returns:
               Weather description
           """
           # Use self.api_key for API call
           return f"Weather in {city}: ..."

       @tool
       async def get_forecast(self, city: str, days: int = 5) -> str:
           """Get weather forecast.

           Args:
               city: Name of the city
               days: Number of days (default 5)

           Returns:
               Forecast description
           """
           # Async implementation
           return f"{days}-day forecast for {city}: ..."

   # Use it
   agent = Agent(
       name="weather_assistant",
       tools=[WeatherToolSet(api_key="...")]
   )

Tool Docstrings
---------------

Docstrings become tool descriptions for the LLM:

.. code-block:: python

   @tool
   def analyze_data(
       self,
       file_path: str,
       method: str = "summary"
   ) -> dict:
       """Analyze data from a file.

       Use this tool when you need to analyze CSV or JSON data files.
       Supports various analysis methods.

       Args:
           file_path: Path to the data file (CSV or JSON)
           method: Analysis method - "summary", "correlation", or "distribution"

       Returns:
           Dictionary containing analysis results with keys:
           - statistics: Basic statistics
           - insights: Key findings
       """
       ...

Good docstrings help the model understand when and how to use tools.

Async Tools
-----------

For I/O-bound operations:

.. code-block:: python

   @tool
   async def fetch_data(self, url: str) -> str:
       """Fetch data from an API."""
       async with aiohttp.ClientSession() as session:
           async with session.get(url) as response:
               return await response.text()

Tool Permissions
----------------

Control what tools can do:

.. code-block:: python

   from pantheon.toolsets import FileManagerToolSet

   # Restrict to read-only
   file_tools = FileManagerToolSet(
       read_only=True
   )

   # Restrict to specific directory
   file_tools = FileManagerToolSet(
       workspace="/path/to/allowed/directory"
   )

Error Handling
--------------

Return informative errors:

.. code-block:: python

   @tool
   def read_config(self, path: str) -> str:
       """Read configuration file."""
       try:
           with open(path) as f:
               return f.read()
       except FileNotFoundError:
           return f"Error: Config file '{path}' not found"
       except PermissionError:
           return f"Error: No permission to read '{path}'"

Detailed Toolset Documentation
------------------------------

For in-depth documentation of each toolset:

.. toctree::
   :maxdepth: 1

   /toolsets/builtin_toolsets
   /toolsets/file_editor
   /toolsets/shell
   /toolsets/python_interpreter
   /toolsets/notebook
   /toolsets/web_browse
   /toolsets/custom_toolsets
