Model Configuration
===================

Configure which LLM models your agents use.

Model Format
------------

Models are specified as ``provider/model-name``:

.. code-block:: text

   openai/gpt-4o
   openai/gpt-4o-mini
   anthropic/claude-3-opus
   anthropic/claude-3-sonnet
   google/gemini-pro

Supported Providers
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Provider
     - Prefix
     - Example Models
   * - OpenAI
     - ``openai/``
     - ``gpt-4o``, ``gpt-4o-mini``, ``gpt-4-turbo``
   * - Anthropic
     - ``anthropic/``
     - ``claude-3-opus``, ``claude-3-sonnet``, ``claude-3-haiku``
   * - Google
     - ``google/``
     - ``gemini-pro``, ``gemini-pro-vision``
   * - Azure OpenAI
     - ``azure/``
     - Your deployed model names
   * - Local (Ollama)
     - ``ollama/``
     - ``llama2``, ``mistral``, ``codellama``

Settings Configuration
----------------------

Configure defaults in ``settings.json``:

.. code-block:: json

   {
     "models": {
       "default": "openai/gpt-4o",
       "fallback": ["openai/gpt-4o-mini", "anthropic/claude-3-sonnet"]
     }
   }

Default Model
~~~~~~~~~~~~~

The ``default`` model is used when no model is specified:

.. code-block:: json

   {
     "models": {
       "default": "openai/gpt-4o"
     }
   }

Fallback Chain
~~~~~~~~~~~~~~

If the primary model fails, fallbacks are tried in order:

.. code-block:: json

   {
     "models": {
       "default": "openai/gpt-4o",
       "fallback": [
         "openai/gpt-4o-mini",
         "anthropic/claude-3-sonnet"
       ]
     }
   }

Provider-Specific Defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~

Set defaults per provider:

.. code-block:: json

   {
     "models": {
       "default": "openai/gpt-4o",
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

Setting Model in Templates
--------------------------

Agent Templates
~~~~~~~~~~~~~~~

.. code-block:: markdown

   ---
   name: Smart Agent
   model: openai/gpt-4o
   ---

   You are a helpful assistant.

Team Templates
~~~~~~~~~~~~~~

Default for all agents:

.. code-block:: markdown

   ---
   name: My Team
   model: openai/gpt-4o-mini
   agents:
     - name: agent1
       instructions: First agent.
     - name: agent2
       model: openai/gpt-4o  # Override default
       instructions: Second agent (uses better model).
   ---

Python API
~~~~~~~~~~

.. code-block:: python

   from pantheon import Agent

   agent = Agent(
       name="assistant",
       model="anthropic/claude-3-opus",
       instructions="You are helpful."
   )

API Keys
--------

Set API keys via environment variables (recommended):

.. code-block:: bash

   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   export GOOGLE_API_KEY="..."

Or in ``settings.json`` (not recommended for security):

.. code-block:: json

   {
     "api_keys": {
       "openai": "sk-...",
       "anthropic": "sk-ant-..."
     }
   }

Azure OpenAI
------------

Configure Azure endpoints:

.. code-block:: bash

   export AZURE_API_KEY="..."
   export AZURE_API_BASE="https://your-resource.openai.azure.com"
   export AZURE_API_VERSION="2024-02-01"

Use with:

.. code-block:: markdown

   ---
   name: Azure Agent
   model: azure/your-deployment-name
   ---

Local Models (Ollama)
---------------------

1. Install `Ollama <https://ollama.ai>`_
2. Pull a model: ``ollama pull llama2``
3. Use in template:

.. code-block:: markdown

   ---
   name: Local Agent
   model: ollama/llama2
   ---

Model Selection Tips
--------------------

**For complex reasoning:**

- ``openai/gpt-4o``
- ``anthropic/claude-3-opus``

**For fast, simple tasks:**

- ``openai/gpt-4o-mini``
- ``anthropic/claude-3-haiku``

**For code generation:**

- ``openai/gpt-4o``
- ``anthropic/claude-3-sonnet``

**For cost-sensitive applications:**

- ``openai/gpt-4o-mini``
- ``ollama/llama2`` (free, local)

**For vision tasks:**

- ``openai/gpt-4o`` (supports images)
- ``google/gemini-pro-vision``

Model Parameters
----------------

Some parameters can be set per-agent:

.. code-block:: markdown

   ---
   name: Creative Writer
   model: openai/gpt-4o
   temperature: 0.9
   max_tokens: 4000
   ---

   You are a creative writer.

.. code-block:: markdown

   ---
   name: Code Generator
   model: openai/gpt-4o
   temperature: 0.1
   ---

   You write precise, correct code.

Temperature Guidelines
~~~~~~~~~~~~~~~~~~~~~~

- **0.0-0.3**: Deterministic, factual (code, analysis)
- **0.4-0.7**: Balanced (general tasks)
- **0.8-1.0**: Creative (writing, brainstorming)

Troubleshooting
---------------

**"API key not found"**

.. code-block:: bash

   # Check if key is set
   echo $OPENAI_API_KEY

   # Set it
   export OPENAI_API_KEY="sk-..."

**"Model not found"**

- Check model name spelling
- Verify you have access to the model
- Check provider prefix is correct

**"Rate limit exceeded"**

- Configure fallback models
- Use a model with higher limits
- Add delays between requests
