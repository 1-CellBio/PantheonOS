---
category: testing
description: 'Test team for agentic task modal workflow'
icon: 🧪
id: agentic_coding
name: Agentic Coding Team
type: team
version: 1.0.0
agents:
  - agentic_agent
agentic_agent:
  id: agentic_agent
  name: Agentic Agent
  model: openai/gpt-5
  icon: 🤖
  toolsets:
    - file_manager
    - python_interpreter
    - shell
    - integrated_notebook
    - task
    - code
---

{{agentic_coding}}
