Toolsets API
============

Toolsets extend agent capabilities with tools for file operations, code execution, web search, and more.

Using Toolsets
--------------

.. code-block:: python

   from pantheon import Agent
   from pantheon.toolsets import FileManagerToolSet, ShellToolSet

   agent = Agent(
       name="developer",
       instructions="You are a developer assistant.",
       model="gpt-4o",
       tools=[
           FileManagerToolSet(),
           ShellToolSet()
       ]
   )

Available Toolsets
------------------

**File Operations**

.. code-block:: python

   from pantheon.toolsets import FileManagerToolSet

   # Read, write, edit, search files
   file_tools = FileManagerToolSet()

**Shell Commands**

.. code-block:: python

   from pantheon.toolsets import ShellToolSet

   # Run shell commands with timeout protection
   shell_tools = ShellToolSet()

**Python Execution**

.. code-block:: python

   from pantheon.toolsets import PythonInterpreterToolSet

   # Execute Python code
   python_tools = PythonInterpreterToolSet()

**Jupyter Notebooks**

.. code-block:: python

   from pantheon.toolsets import IntegratedNotebookToolSet

   # Create, edit, execute notebooks
   notebook_tools = IntegratedNotebookToolSet()

**Web Search**

.. code-block:: python

   from pantheon.toolsets import WebToolSet

   # Search and fetch web content
   web_tools = WebToolSet()

**Knowledge Base**

.. code-block:: python

   from pantheon.toolsets import KnowledgeToolSet

   # Vector store and RAG
   knowledge_tools = KnowledgeToolSet()

Creating Custom Tools
---------------------

**Simple Function**

.. code-block:: python

   @agent.tool
   def my_function(param: str) -> str:
       """Description shown to the LLM."""
       return f"Result: {param}"

**ToolSet Class**

.. code-block:: python

   from pantheon import ToolSet, tool

   class MyToolSet(ToolSet):
       @tool
       def calculate(self, expression: str) -> float:
           """Evaluate a math expression.

           Args:
               expression: Math expression to evaluate

           Returns:
               The result of the calculation
           """
           return eval(expression)

       @tool
       async def fetch_data(self, url: str) -> str:
           """Fetch data from a URL."""
           async with aiohttp.ClientSession() as session:
               async with session.get(url) as resp:
                   return await resp.text()

Tool Documentation
------------------

The docstring becomes the tool description:

.. code-block:: python

   @tool
   def search_files(self, pattern: str, directory: str = ".") -> list:
       """Search for files matching a pattern.

       Use this tool when you need to find files in the project.

       Args:
           pattern: Glob pattern (e.g., "*.py", "**/*.md")
           directory: Directory to search (default: current directory)

       Returns:
           List of matching file paths
       """
       return glob.glob(pattern, root_dir=directory)

Good docstrings help the LLM understand when and how to use the tool.

Async Tools
-----------

Use async for I/O operations:

.. code-block:: python

   @tool
   async def fetch_url(self, url: str) -> str:
       """Fetch content from a URL."""
       async with aiohttp.ClientSession() as session:
           async with session.get(url) as response:
               return await response.text()

Combining Toolsets
------------------

.. code-block:: python

   agent = Agent(
       name="full_stack",
       instructions="You are a full-stack developer.",
       tools=[
           FileManagerToolSet(),
           ShellToolSet(),
           PythonInterpreterToolSet(),
           WebToolSet()
       ]
   )

Best Practices
--------------

**Only Include Needed Tools**

Too many tools can confuse the model. Be selective.

**Clear Descriptions**

Write descriptive docstrings that explain when to use each tool.

**Error Handling**

Return informative error messages:

.. code-block:: python

   @tool
   def read_file(self, path: str) -> str:
       """Read a file."""
       try:
           with open(path) as f:
               return f.read()
       except FileNotFoundError:
           return f"Error: File '{path}' not found"
       except PermissionError:
           return f"Error: No permission to read '{path}'"
