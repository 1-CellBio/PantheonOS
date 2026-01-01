Notebook Toolset
================

The Notebook toolset provides integration with Jupyter notebooks, enabling agents to create, modify, and execute notebook cells programmatically.

Overview
--------

Pantheon provides multiple notebook-related toolsets:

- **IntegratedNotebookToolSet**: Full-featured notebook with kernel management (recommended)
- **JupyterKernelToolSet**: Direct Jupyter kernel access
- **NotebookToolSet**: Basic notebook file operations

Installation
------------

Notebook support requires the toolsets extra::

    pip install pantheon-agents[toolsets]

This includes:

- jupyter-client
- ipykernel
- nbformat
- jedi (for code completion)

IntegratedNotebookToolSet
-------------------------

The recommended toolset for notebook operations with integrated kernel management.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from pantheon import Agent
   from pantheon.toolsets import IntegratedNotebookToolSet

   async def main():
       notebook_tools = IntegratedNotebookToolSet()

       agent = Agent(
           name="analyst",
           instructions="You are a data analyst. Use Jupyter notebooks for analysis.",
           model="gpt-4o",
           tools=[notebook_tools]
       )

       await agent.chat()

Available Tools
~~~~~~~~~~~~~~~

The IntegratedNotebookToolSet provides:

**Notebook Management**

- ``create_notebook(path, kernel_name="python3")``: Create a new notebook
- ``open_notebook(path)``: Open and read an existing notebook
- ``read_notebook(path)``: Read notebook contents without opening
- ``close_notebook(path)``: Close a notebook and cleanup resources

**Cell Operations**

- ``add_cell(path, cell_type, source, execute=True)``: Add a cell

  - ``cell_type``: "code" or "markdown"
  - ``execute``: Auto-execute code cells (default: True for code, False for markdown)

- ``update_cell(path, cell_index, source, execute=True)``: Update existing cell
- ``delete_cell(path, cell_index)``: Remove a cell
- ``move_cell(path, from_index, to_index)``: Reorder cells

**Execution**

- ``execute_cell(path, cell_index)``: Execute a specific cell
- ``execute_all_cells(path)``: Execute all cells in order
- ``get_cell_output(path, cell_index)``: Get output of executed cell

**Kernel Management**

- ``restart_kernel(path)``: Restart the notebook kernel
- ``interrupt_kernel(path)``: Interrupt running execution

Example Workflow
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Agent can perform notebook operations like:

   # 1. Create a notebook
   await notebook_tools.create_notebook("analysis.ipynb")

   # 2. Add cells
   await notebook_tools.add_cell(
       "analysis.ipynb",
       "code",
       "import pandas as pd\nimport matplotlib.pyplot as plt",
       execute=True
   )

   await notebook_tools.add_cell(
       "analysis.ipynb",
       "markdown",
       "# Data Analysis\nLoading and exploring the dataset."
   )

   await notebook_tools.add_cell(
       "analysis.ipynb",
       "code",
       "df = pd.read_csv('data.csv')\ndf.head()"
   )

   # 3. Get output
   output = await notebook_tools.get_cell_output("analysis.ipynb", 2)

JupyterKernelToolSet
--------------------

Direct access to Jupyter kernels for code execution without notebook file management.

.. code-block:: python

   from pantheon.toolsets import JupyterKernelToolSet

   kernel_tools = JupyterKernelToolSet()

   agent = Agent(
       name="coder",
       instructions="Execute Python code interactively.",
       model="gpt-4o",
       tools=[kernel_tools]
   )

Available Tools
~~~~~~~~~~~~~~~

- ``execute_code(code)``: Execute Python code
- ``get_completions(code, cursor_pos)``: Get code completions
- ``restart_kernel()``: Restart the kernel
- ``interrupt_execution()``: Stop running code

Configuration
-------------

Kernel Settings
~~~~~~~~~~~~~~~

Configure kernel behavior:

.. code-block:: python

   notebook_tools = IntegratedNotebookToolSet(
       kernel_name="python3",      # Kernel to use
       timeout=60,                 # Execution timeout in seconds
       startup_timeout=30,         # Kernel startup timeout
   )

Working Directory
~~~~~~~~~~~~~~~~~

Set the working directory for notebook execution:

.. code-block:: python

   notebook_tools = IntegratedNotebookToolSet(
       working_dir="/path/to/notebooks"
   )

Best Practices
--------------

1. **Use IntegratedNotebookToolSet**: Provides the most complete functionality
2. **Execute on Add**: Use ``execute=True`` to immediately run code cells
3. **Handle Errors**: Check cell outputs for execution errors
4. **Cleanup**: Close notebooks when done to free kernel resources
5. **Markdown Cells**: Use markdown cells for documentation and structure

Integration with File Manager
-----------------------------

Combine with FileManagerToolSet for complete data analysis workflows:

.. code-block:: python

   from pantheon import Agent
   from pantheon.toolsets import IntegratedNotebookToolSet, FileManagerToolSet

   async def analysis_agent():
       agent = Agent(
           name="data_analyst",
           instructions="""You are a data analyst.
           Use notebooks for analysis and file operations for data management.
           Always document your analysis with markdown cells.""",
           model="gpt-4o",
           tools=[
               IntegratedNotebookToolSet(),
               FileManagerToolSet()
           ]
       )

       await agent.chat()

R Language Support
------------------

With R kernel support (requires rpy2)::

    pip install pantheon-agents[toolsets]

Then use an R kernel:

.. code-block:: python

   notebook_tools = IntegratedNotebookToolSet(kernel_name="ir")
