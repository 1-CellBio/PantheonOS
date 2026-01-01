<div align="center">
  <h1> Pantheon </h1>

  <p> A framework for building distributed LLM-based multi-agent systems. </p>

  <p>
    <a href="https://github.com/aristoteleo/pantheon-agents/actions/workflows/test.yml">
        <img src="https://github.com/aristoteleo/pantheon-agents/actions/workflows/test.yml/badge.svg" alt="Build Status">
    </a>
    <a href="https://pypi.org/project/pantheon-agents/">
      <img src="https://img.shields.io/pypi/v/pantheon-agents.svg" alt="Install with PyPi" />
    </a>
    <a href="https://github.com/aristoteleo/pantheon-agents/blob/master/LICENSE">
      <img src="https://img.shields.io/github/license/aristoteleo/pantheon-agents" alt="MIT license" />
    </a>
  </p>
</div>

## Features

- **Multi-Agent Teams**: PantheonTeam, Sequential, Swarm, MoA, and AgentAsToolTeam patterns
- **Rich Toolsets**: File operations, Python/Shell execution, Jupyter notebooks, Web browsing, RAG
- **Interactive REPL**: Full-screen file viewer, syntax highlighting, approval workflows
- **MCP Integration**: Native Model Context Protocol support
- **Learning System**: Skillbook-based agent improvement
- **Distributed Architecture**: NATS-based communication for scalable deployments

## Installation

```bash
# Basic installation
pip install pantheon-agents

# With all toolsets
pip install pantheon-agents[toolsets]
```

## Quick Start

### Using the REPL

The easiest way to start:

```bash
python -m pantheon.repl
```

### Creating an Agent

```python
import asyncio
from pantheon import Agent

async def main():
    agent = Agent(
        name="assistant",
        instructions="You are a helpful assistant.",
        model="gpt-4o-mini"
    )
    await agent.chat()

asyncio.run(main())
```

### Using Toolsets

```python
from pantheon import Agent
from pantheon.toolsets import FileManagerToolSet, ShellToolSet

agent = Agent(
    name="developer",
    instructions="You are a developer assistant.",
    model="gpt-4o",
    tools=[FileManagerToolSet(), ShellToolSet()]
)
```

### Creating Teams

```python
from pantheon import Agent
from pantheon.team import PantheonTeam

researcher = Agent(name="researcher", instructions="Research topics.")
writer = Agent(name="writer", instructions="Write content.")

team = PantheonTeam([researcher, writer])
await team.chat()
```

## Documentation

See the [docs](docs/) folder for detailed documentation.

## Examples

See the [examples](examples/) folder for usage patterns and implementations.

## License

MIT License
