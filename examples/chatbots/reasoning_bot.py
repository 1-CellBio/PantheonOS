import asyncio
from pantheon.agent import Agent
from pantheon.reasoning import reasoning_deepseek_reasoner


reasoning_bot = Agent(
    name="reasoning_bot",
    instructions="You are an AI assistant with reasoning abilities. " \
        "You can use `reasoning` to solve complex problems.",
    model="deepseek/deepseek-chat",
    tools=[reasoning_deepseek_reasoner],
)


async def main():
    await reasoning_bot.chat()


if __name__ == "__main__":
    asyncio.run(main())
