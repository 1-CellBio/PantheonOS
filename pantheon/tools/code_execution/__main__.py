import asyncio
from .python_interpreter import PythonInterpreterToolSet

toolset = PythonInterpreterToolSet("python_interpreter")
asyncio.run(toolset.run())
