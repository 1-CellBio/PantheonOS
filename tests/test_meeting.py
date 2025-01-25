from pantheon.agent import Agent
from pantheon.meeting import BrainStorm


async def test_brainstorm():
    agent1 = Agent(name="agent1", instructions="You are a biologist.")
    agent2 = Agent(name="agent2", instructions="You are a computer scientist.")
    agent3 = Agent(name="agent3", instructions="You are a doctor.")
    meeting = BrainStorm([agent1, agent2, agent3])
    report = await meeting.run("How to use AI in biology and medicine?", rounds=5)
    print(report)

