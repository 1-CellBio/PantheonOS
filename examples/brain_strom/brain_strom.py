import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.WARNING)

import asyncio
from pantheon.agent import Agent
from pantheon.meeting import BrainStorm
from pantheon.smart_func import smart_func
from pantheon.tools.web_browse.duckduckgo import duckduckgo_search
from pantheon.tools.web_browse.web_crawl import web_crawl


biologist = Agent(
    name="biologist",
    instructions="You are a biologist. You have a lot of knowledge about the biology.",
    model="gpt-4o-mini",
    tools=[duckduckgo_search, web_crawl],
)


computer_scientist = Agent(
    name="computer_scientist",
    instructions="You are a computer scientist. You have a lot of knowledge about the computer science.",
    model="gpt-4o-mini",
    tools=[duckduckgo_search, web_crawl],
)


doctor = Agent(
    name="doctor",
    instructions="You are a doctor. You have a lot of knowledge about the medicine.",
    model="gpt-4o-mini",
    tools=[duckduckgo_search, web_crawl],
)


meeting = BrainStorm([biologist, computer_scientist, doctor])


@smart_func(model="gpt-4o-mini")
async def generate_agenda(theme: str) -> str:
    """
    Generate a agenda for the meeting. 
    The agenda should be a list of topics to discuss.
    """


@smart_func(model="gpt-4o")
async def summarize(report: str, theme: str, agenda: str) -> str:
    """
    Summarize the discussion on the theme,
    and give a conclusion in markdown format.

    Including:
    - The theme
    - The agenda of the meeting
    - Procedure of the discussion
    - Thoughts of each participant
    - Results of the discussion
        + List the important points and provide the details for each point
        + Provide the evidence and sources for each point
    - The final conclusion
    """


async def main():
    theme = """Discuss how AI could be used in biology and medicine.  """
    print("Generating meeting agenda...")
    agenda = await generate_agenda(theme)
    print("Meeting agenda:\n", agenda)

    print("------------START-------------\n")
    report = await meeting.run(
        agenda,
        rounds=20,
        print_stream=True,
    )
    print("------------END-------------\n")
    summary = await summarize(report, theme, agenda)
    print(summary)

    with open("./brain_strom.md", "w", encoding="utf-8") as f:
        f.write(summary)
        f.write("\n\n")
        f.write("## Detailed discussion\n")
        f.write("```\n")
        f.write(report)
        f.write("```\n")



if __name__ == "__main__":
    asyncio.run(main())
