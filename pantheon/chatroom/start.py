import asyncio
from pathlib import Path

from .room import ChatRoom

from magique.ai import connect_remote


async def start_services(
    service_name: str = "pantheon-chatroom",
    memory_dir: str = "./.pantheon-chatroom",
    id_hash: str | None = None,
    endpoint_service_id: str | None = None,
    workspace_path: str = "./.pantheon-chatroom-workspace",
    agents_template: dict | str | None = None,
    log_level: str = "INFO",
    endpoint_wait_time: int = 5,
    worker_params: dict | None = None,
    worker_params_endpoint: dict | None = None,
    endpoint_connect_params: dict | None = None,
    speech_to_text_model: str = "gpt-4o-mini-transcribe",
):
    if endpoint_service_id is None:
        from magique.ai.endpoint import Endpoint
        w_path = Path(workspace_path)
        w_path.mkdir(parents=True, exist_ok=True)
        endpoint = Endpoint(
            workspace_path=workspace_path,
            config={"log_level": log_level},
            worker_params=worker_params_endpoint,
        )
        asyncio.create_task(endpoint.run())
        endpoint_service_id = endpoint.worker.service_id
        await asyncio.sleep(endpoint_wait_time)

    endpoint_connect_params = endpoint_connect_params or {}
    endpoint = await connect_remote(endpoint_service_id, **endpoint_connect_params)

    if worker_params is None:
        worker_params = {}
    if "id_hash" not in worker_params:
        worker_params["id_hash"] = id_hash

    chat_room = ChatRoom(
        endpoint_service_id=endpoint_service_id,
        agents_template=agents_template,
        memory_dir=memory_dir,
        name=service_name,
        worker_params=worker_params,
        endpoint_connect_params=endpoint_connect_params,
        speech_to_text_model=speech_to_text_model,
    )
    await chat_room.run(log_level=log_level)
