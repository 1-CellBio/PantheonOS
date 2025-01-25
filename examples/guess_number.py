import random
import asyncio
from pantheon.agent import Agent


def validate_guess(guess: int) -> str:
    print(f"Guess: {guess}")
    if guess == number:
        msg = "You guessed the number correctly!"
    elif guess < number:
        msg = "You guessed too low!"
    else:
        msg = "You guessed too high!"
    print(msg)
    return msg


agent = Agent(
    name="guesser",
    instructions="You are a guesser, you need to guess the number.",
    tools=[validate_guess],
)


async def main():
    global number
    number = random.randint(1, 100)
    print(f"Start guessing, the truth is {number}")
    resp = await agent.run("Guess the number between 1 and 100.")
    print(resp.content)


if __name__ == "__main__":
    asyncio.run(main())
