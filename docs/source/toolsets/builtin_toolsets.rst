Built-in Toolsets
=================

Pantheon provides production-ready toolsets that extend agent capabilities. Toolsets are included in the main package and can be used directly with agents.

Installation
------------

Install with the toolsets extra for full functionality::

    pip install pantheon-agents[toolsets]

Toolset Summary
---------------

File Operations
~~~~~~~~~~~~~~~

**FileManagerToolSet**

Comprehensive file operations with intelligent editing:

- ``read_file(path)``: Read file contents
- ``write_file(path, content)``: Write/create files
- ``edit_file(path, old_content, new_content)``: Diff-based editing
- ``glob(pattern)``: Pattern-based file search
- ``grep(pattern, path)``: Content search with regex
- ``patch_file(path, patch)``: Apply unified diff patches

.. code-block:: python

   from pantheon.toolsets import FileManagerToolSet

   agent = Agent(tools=[FileManagerToolSet()])

Code Execution
~~~~~~~~~~~~~~

**ShellToolSet**

Run shell commands with safety controls:

- ``run_command(command, timeout=60)``: Execute shell commands
- Configurable allowed/blocked commands
- Timeout protection

.. code-block:: python

   from pantheon.toolsets import ShellToolSet

   agent = Agent(tools=[ShellToolSet()])

**PythonInterpreterToolSet**

Execute Python code in isolated environment:

- ``execute(code)``: Run Python code
- Persistent state across executions
- Variable inspection

.. code-block:: python

   from pantheon.toolsets import PythonInterpreterToolSet

   agent = Agent(tools=[PythonInterpreterToolSet()])

**IntegratedNotebookToolSet**

Full Jupyter notebook integration:

- Create, read, update notebooks
- Execute cells with output capture
- Kernel lifecycle management

.. code-block:: python

   from pantheon.toolsets import IntegratedNotebookToolSet

   agent = Agent(tools=[IntegratedNotebookToolSet()])

Web & Search
~~~~~~~~~~~~

**WebToolSet**

Web browsing and search:

- ``search(query)``: Web search via DuckDuckGo
- ``fetch_url(url)``: Fetch and parse web pages

.. code-block:: python

   from pantheon.toolsets import WebToolSet

   agent = Agent(tools=[WebToolSet()])

**ScraperToolSet**

Advanced web scraping with crawl4ai:

- ``scrape(url)``: Scrape web content
- JavaScript rendering support
- Structured data extraction

.. code-block:: python

   from pantheon.toolsets import ScraperToolSet

   agent = Agent(tools=[ScraperToolSet()])

Knowledge & RAG
~~~~~~~~~~~~~~~

**KnowledgeToolSet**

Knowledge base management with LlamaIndex:

- ``add_documents(paths)``: Index documents
- ``query(question)``: Query knowledge base
- Vector store integration (Qdrant)

Requires knowledge extra::

    pip install pantheon-agents[knowledge]

.. code-block:: python

   from pantheon.toolsets import KnowledgeToolSet

   agent = Agent(tools=[KnowledgeToolSet()])

**VectorRAGToolSet**

Vector-based retrieval augmented generation:

- ``search(query, k=5)``: Semantic search
- ``add(text, metadata=None)``: Add to vector store

.. code-block:: python

   from pantheon.toolsets import VectorRAGToolSet

   agent = Agent(tools=[VectorRAGToolSet()])

Specialized Toolsets
~~~~~~~~~~~~~~~~~~~~

**TaskToolSet**

Ephemeral task tracking for workflows:

- ``create_task(description)``: Create a task
- ``update_task(id, status)``: Update task status
- ``list_tasks()``: List all tasks

.. code-block:: python

   from pantheon.toolsets import TaskToolSet

   agent = Agent(tools=[TaskToolSet()])

**PackageToolSet**

Python package discovery and analysis:

- ``discover_packages()``: Find available packages
- ``get_methods(package, class_name)``: Extract methods

.. code-block:: python

   from pantheon.toolsets import PackageToolSet

   agent = Agent(tools=[PackageToolSet()])

**DatabaseAPIQueryToolSet**

Database query capabilities:

- ``query(sql)``: Execute SQL queries
- Support for various databases

.. code-block:: python

   from pantheon.toolsets import DatabaseAPIQueryToolSet

   agent = Agent(tools=[DatabaseAPIQueryToolSet(connection_string="...")])

Language Interpreters
~~~~~~~~~~~~~~~~~~~~~

**RInterpreterToolSet**

R language code execution:

- ``execute_r(code)``: Run R code
- Requires R installation and rpy2

**JuliaInterpreterToolSet**

Julia language code execution:

- ``execute_julia(code)``: Run Julia code
- Requires Julia installation

Quick Start
-----------

Combining Multiple Toolsets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pantheon import Agent
   from pantheon.toolsets import (
       FileManagerToolSet,
       ShellToolSet,
       PythonInterpreterToolSet,
       WebToolSet
   )

   async def main():
       agent = Agent(
           name="developer",
           instructions="""You are a full-stack developer assistant.
           You can:
           - Read, write, and edit files
           - Run shell commands
           - Execute Python code
           - Search the web for information""",
           model="gpt-4o",
           tools=[
               FileManagerToolSet(),
               ShellToolSet(),
               PythonInterpreterToolSet(),
               WebToolSet()
           ]
       )

       await agent.chat()

Data Analysis Setup
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pantheon import Agent
   from pantheon.toolsets import (
       IntegratedNotebookToolSet,
       FileManagerToolSet
   )

   async def main():
       agent = Agent(
           name="analyst",
           instructions="""You are a data analyst.
           Use Jupyter notebooks for analysis.
           Document your work with markdown cells.""",
           model="gpt-4o",
           tools=[
               IntegratedNotebookToolSet(),
               FileManagerToolSet()
           ]
       )

       await agent.chat()

Best Practices
--------------

1. **Combine Related Toolsets**: Group toolsets that work together (e.g., File + Shell)
2. **Security**: Be cautious with ShellToolSet in production
3. **Resource Management**: Close resources when done (notebooks, connections)
4. **Error Handling**: Tools return error messages that agents can interpret
5. **Documentation**: Include toolset capabilities in agent instructions
