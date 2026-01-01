Settings Reference
==================

The ``settings.json`` file contains all Pantheon configuration.

Location
--------

- **Project**: ``./.pantheon/settings.json``
- **User global**: ``~/.pantheon/settings.json``

Project settings override user settings.

Format
------

JSONC format (JSON with comments):

.. code-block:: json

   {
     // This is a comment
     "repl": {
       "quiet": false
     }
   }

Full Reference
--------------

API Keys
~~~~~~~~

.. code-block:: json

   {
     "api_keys": {
       "openai": "sk-...",
       "anthropic": "sk-ant-...",
       "google": "...",
       "huggingface": "..."
     }
   }

**Recommended**: Use environment variables instead:

.. code-block:: bash

   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."

Models
~~~~~~

.. code-block:: json

   {
     "models": {
       "default": "gpt-4o",
       "fallback": ["gpt-4o-mini", "claude-3-sonnet"],
       "providers": {
         "openai": {
           "default": "gpt-4o",
           "mini": "gpt-4o-mini"
         },
         "anthropic": {
           "default": "claude-3-opus"
         }
       }
     }
   }

REPL Settings
~~~~~~~~~~~~~

.. code-block:: json

   {
     "repl": {
       "quiet": false,
       "default_template": "default",
       "log_level": "ERROR"
     }
   }

- ``quiet``: Suppress startup messages
- ``default_template``: Team template to load
- ``log_level``: DEBUG, INFO, WARNING, ERROR

ChatRoom Settings
~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "chatroom": {
       "memory_dir": ".pantheon/memory",
       "enable_nats_streaming": false,
       "speech_to_text_model": null
     }
   }

Endpoint Settings
~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "endpoint": {
       "workspace": ".pantheon/workspace",
       "timeout": 120,
       "execution_mode": "local"
     }
   }

- ``workspace``: Working directory for file operations
- ``timeout``: Tool execution timeout (seconds)
- ``execution_mode``: "local" or "remote"

Services
~~~~~~~~

Built-in toolset configuration:

.. code-block:: json

   {
     "services": {
       "tier1": ["file_manager"],
       "tier2": ["python_interpreter", "shell"],
       "tier3": ["web_browse"]
     }
   }

Learning Settings
~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "learning": {
       "enabled": true,
       "skillbook_path": ".pantheon/skills",
       "trajectory_tracking": true
     }
   }

Context Compression
~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "context_compression": {
       "enabled": true,
       "threshold_tokens": 100000,
       "target_tokens": 50000
     }
   }

Knowledge Base
~~~~~~~~~~~~~~

.. code-block:: json

   {
     "knowledge": {
       "vector_db": "qdrant",
       "qdrant_url": "http://localhost:6333",
       "embedding_model": "text-embedding-3-small"
     }
   }

Remote Backend
~~~~~~~~~~~~~~

For distributed deployments:

.. code-block:: json

   {
     "remote": {
       "nats_url": "nats://localhost:4222",
       "backend": "nats"
     }
   }

Environment Variables
---------------------

These environment variables are recognized:

.. list-table::
   :header-rows: 1

   * - Variable
     - Description
   * - ``OPENAI_API_KEY``
     - OpenAI API key
   * - ``ANTHROPIC_API_KEY``
     - Anthropic API key
   * - ``GOOGLE_API_KEY``
     - Google API key
   * - ``PANTHEON_LOG_LEVEL``
     - Log level
   * - ``PANTHEON_CONFIG_DIR``
     - Override config directory

Example Configuration
---------------------

Minimal:

.. code-block:: json

   {
     "models": {
       "default": "gpt-4o-mini"
     }
   }

Full:

.. code-block:: json

   {
     "api_keys": {
       "openai": "sk-..."
     },
     "models": {
       "default": "gpt-4o",
       "fallback": ["gpt-4o-mini"]
     },
     "repl": {
       "quiet": false,
       "default_template": "developer_team"
     },
     "chatroom": {
       "memory_dir": ".pantheon/memory"
     },
     "endpoint": {
       "workspace": ".pantheon/workspace",
       "timeout": 120
     },
     "learning": {
       "enabled": true
     }
   }
