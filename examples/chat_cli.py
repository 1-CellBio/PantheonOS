import warnings
warnings.filterwarnings("ignore")

import asyncio
import random
from pantheum.repl import Repl
from pantheum.agent import Agent


scifi_fan = Agent(
    name="scifi_fan",
    instructions="You are a scifi fan. Please disguise yourself as a human and talk to me.",
    model="gpt-4o-mini",
)


adventure_novel_fan = Agent(
    name="adventure_novel_fan",
    instructions="You are an adventure novel fan. Please disguise yourself as a human and talk to me.",
    model="gpt-4o-mini",
)


async def main():
    repl = Repl([scifi_fan, adventure_novel_fan])
    await repl.run()


if __name__ == "__main__":
    asyncio.run(main())
