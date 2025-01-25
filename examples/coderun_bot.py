import warnings
warnings.filterwarnings("ignore")

import asyncio
from pantheon.agent import Agent
from pantheon.tools.code_execution import PythonInterpreterToolSet
from pantheon.remote import run_toolsets


async def main():
    toolset = PythonInterpreterToolSet("python_interpreter")
    async with run_toolsets([toolset], log_level="INFO"):
        agent = Agent("coderun_bot", "You are an AI assistant that can run Python code.")
        await agent.remote_toolset(toolset.service_id)
        await agent.chat()


if __name__ == "__main__":
    asyncio.run(main())
