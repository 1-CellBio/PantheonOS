import asyncio
import os
from pathlib import Path

import fire
from magique.ai.tools.python import PythonInterpreterToolSet
from magique.ai.tools.file_manager import FileManagerToolSet
from magique.ai.toolset import run_toolsets

from . import ChatRoom
from ..remote.memory import MemoryManagerService, RemoteMemoryManager
from ..agent import Agent


async def main(
    memory_path: str = "./.pantheon-chatroom",
    workspace_path: str = "./.pantheon-chatroom-workspace",
    log_level: str = "INFO",
):
    w_path = Path(workspace_path)
    w_path.mkdir(parents=True, exist_ok=True)
    memory_service = MemoryManagerService(memory_path)
    asyncio.create_task(memory_service.run())
    await asyncio.sleep(0.5)
    remote_memory_manager = RemoteMemoryManager(memory_service.worker.service_id)
    await remote_memory_manager.connect()
    py_toolset = PythonInterpreterToolSet("python_interpreter", workdir=str(w_path))
    file_manager_toolset = FileManagerToolSet("file_manager", str(w_path))
    async with run_toolsets([py_toolset, file_manager_toolset], log_level=log_level):
        agent = Agent(
            name="Pantheon",
            instructions="You are a helpful assistant that can answer questions and help with tasks.",
            model="gpt-4o-mini",
        )
        await agent.remote_toolset(py_toolset.service_id)
        await agent.remote_toolset(file_manager_toolset.service_id)
        chat_room = ChatRoom(agent, remote_memory_manager)
        await chat_room.run()


if __name__ == "__main__":
    fire.Fire(main)
