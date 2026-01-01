Advanced ChatRoom Usage
=======================

Programmatic usage and advanced configuration for ChatRoom.

Programmatic Usage
------------------

**Basic Usage**

.. code-block:: python

   import asyncio
   from pantheon.chatroom import ChatRoom

   async def main():
       chatroom = ChatRoom()

       # Start and get service info
       info = await chatroom.start()
       print(f"Service ID: {info.service_id}")

       # Keep running
       await asyncio.Event().wait()

   asyncio.run(main())

**With Custom Endpoint**

.. code-block:: python

   from pantheon.chatroom import ChatRoom
   from pantheon.endpoint import Endpoint

   endpoint = Endpoint(workspace_path="./my_workspace")

   chatroom = ChatRoom(
       endpoint=endpoint,
       memory_dir="./chats"
   )

**From Team Instance**

.. code-block:: python

   from pantheon.chatroom import ChatRoom
   from pantheon.team import PantheonTeam
   from pantheon import Agent

   agents = [
       Agent(name="researcher", instructions="..."),
       Agent(name="writer", instructions="...")
   ]

   team = PantheonTeam(agents)
   chatroom = ChatRoom(team=team)

Configuration
-------------

ChatRoom settings in ``.pantheon/settings.json``:

.. code-block:: json

   {
     "chatroom": {
       "memory_dir": ".pantheon/memory",
       "enable_nats_streaming": false,
       "speech_to_text_model": null
     }
   }

**Options:**

- ``memory_dir``: Directory for conversation persistence
- ``enable_nats_streaming``: Enable NATS for distributed deployment
- ``speech_to_text_model``: Model for speech-to-text (if enabled)

Team Templates
--------------

ChatRoom uses team templates from ``.pantheon/teams/``:

.. code-block:: markdown

   ---
   name: Data Analysis Team
   icon: 📊
   agents:
     - name: analyst
       model: openai/gpt-4o
       instructions: You analyze data and find insights.
       toolsets:
         - python_interpreter
         - file_manager
     - name: visualizer
       model: openai/gpt-4o
       instructions: You create visualizations.
       toolsets:
         - python_interpreter
   ---

   # Data Analysis Team

   A team specialized in data analysis and visualization.

See :doc:`/configuration/templates/teams` for template format details.

NATS Streaming
--------------

For distributed deployments, enable NATS streaming:

.. code-block:: json

   {
     "chatroom": {
       "enable_nats_streaming": true
     },
     "remote": {
       "nats_url": "nats://localhost:4222"
     }
   }

This allows:

- Multiple ChatRoom instances sharing state
- Remote toolset connections
- Distributed agent execution

API Endpoints
-------------

ChatRoom exposes these endpoints:

.. list-table::
   :header-rows: 1

   * - Endpoint
     - Description
   * - ``/chat``
     - Main chat WebSocket
   * - ``/health``
     - Health check
   * - ``/info``
     - Service information

Custom Integration
------------------

**Embedding in Existing App**

.. code-block:: python

   from fastapi import FastAPI
   from pantheon.chatroom import ChatRoom

   app = FastAPI()
   chatroom = ChatRoom()

   @app.on_event("startup")
   async def startup():
       await chatroom.start()

   @app.websocket("/chat")
   async def websocket_endpoint(websocket):
       await chatroom.handle_websocket(websocket)

**Event Hooks**

.. code-block:: python

   class CustomChatRoom(ChatRoom):
       async def on_message(self, message):
           # Called when user sends a message
           print(f"User: {message}")
           return await super().on_message(message)

       async def on_response(self, response):
           # Called when agent responds
           print(f"Agent: {response}")
           return response

Monitoring
----------

**Logs**

ChatRoom logs are written to:

- Console (configurable level)
- ``~/.pantheon/logs/chatroom.log``

**Metrics**

Track token usage and costs via the returned response metadata.

Security Considerations
-----------------------

**API Keys**

Never expose API keys in the UI. They should be:

- Set as environment variables on the server
- Configured in ``settings.json`` on the server

**Workspace Isolation**

Each ChatRoom has its own workspace. Files are isolated per session.

**Network**

- The ChatRoom server should be behind a firewall in production
- Use HTTPS for the UI connection
- Consider authentication for multi-user deployments
