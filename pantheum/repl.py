import asyncio

from pydantic import BaseModel, Field
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel

from .agent import Agent
from .types import AgentResponse


class ChatHistory:
    def __init__(self):
        self.messages = []

    def add_agent_response(self, response: AgentResponse):
        str_content: str
        if isinstance(response.content, BaseModel):
            str_content = response.content.model_dump_json()
        else:
            str_content = response.content
        self.messages.append({
            "role": "assistant",
            "content": str_content,
            "sender": "agent-" + response.agent_name,
        })

    def add_user_message(self, message: str):
        self.messages.append({
            "role": "user",
            "content": message,
            "sender": "user",
        })

    def to_conversation_prompt(self, agent_name: str):
        conversation = "This is a conversation between user and agents:\n"
        for message in self.messages:
            conversation += f"{message['sender']}: {message['content']}\n"
        conversation += f"Please continue the conversation as {agent_name}." + \
            "Just output your response, " + \
            "please do not include your name at the beginning." + \
            "And you should try to cooperate with other agents when necessary."
        return conversation


class WantAnswer(BaseModel):
    want_answer: bool = Field(description="Whether the agent wants to answer the question.")


class Repl:
    def __init__(self, agents: list[Agent] | Agent):
        if isinstance(agents, Agent):
            agents = [agents]
        self.agents = agents
        self.history = ChatHistory()
        self.console = Console()

    def print_greeting(self):
        self.console.print(
            "[bold]Welcome to the Pantheum REPL![/bold]\n" +
            "You can start by typing a message or type 'exit' to exit.\n"
        )
        # print current agents
        self.console.print("[bold]Current agents:[/bold]")
        for agent in self.agents:
            self.console.print(f"  - [blue]{agent.name}[/blue]")
            # print their instructions
            self.console.print(f"    - [green]Instructions:[/green] {agent.instructions}")
            # print their tools
            if agent.functions:
                self.console.print("    - [green]Tools:[/green]")
                for func in agent.functions.values():
                    self.console.print(f"      - {func.__name__}")

        self.console.print()

    async def run(self, message: str | dict | None = None):
        self.print_greeting()

        def ask_user():
            message = Prompt.ask("[red][bold]User[/bold][/red]")
            self.console.print()
            return message

        if message is None:
            message = ask_user()
            if message == "exit":
                return
        else:
            self.console.print(f"[red][bold]User[/bold][/red]: {message}\n")

        self.history.add_user_message(message)

        while True:
            for agent in self.agents:
                conversation_prompt = \
                    self.history.to_conversation_prompt(agent.name)
                resp = await agent.run(
                    [
                        conversation_prompt,
                        f"Should {agent.name} answer the last question?"
                    ],
                    response_format=WantAnswer,
                )
                if not resp.content.want_answer:
                    continue
                self.console.print(f"[blue][bold]{agent.name}[/bold][/blue]: ")
                content = ""
                markdown = Markdown(content)

                with Live(markdown, refresh_per_second=10) as live:
                    def process_chunk(chunk: dict):
                        nonlocal content
                        content += chunk.get("content", "") or ""
                        live.update(Markdown(content))

                    def process_step_message(message: dict):

                        def print_tool_message(message: str):
                            panel = Panel(message, title="Tool Message")
                            self.console.print(panel)

                        if tool_calls := message.get("tool_calls"):
                            for call in tool_calls:
                                print_tool_message(
                                    f"[bold]Agent [blue]{agent.name}[/blue] is using tool "
                                    f"[green]{call.get('function', {}).get('name')}[/green]"
                                    f" with arguments [yellow]{call.get('function', {}).get('arguments')}[/yellow]"
                                    "[/bold]"
                                )
                        if message.get("role") == "tool":
                            print_tool_message(
                                f"[bold]Agent [blue]{agent.name}[/blue] got result from tool "
                                f"[green]{message.get('tool_name')}[/green]:[/bold] "
                                f"[yellow]{message.get('content')}[/yellow]"
                            )

                    resp = await agent.run(
                        conversation_prompt,
                        process_chunk=process_chunk,
                        process_step_message=process_step_message,
                    )

                self.console.print()
                self.history.add_agent_response(resp)
            user_message = ask_user()
            if user_message == "exit":
                break
            self.history.add_user_message(user_message)


if __name__ == "__main__":
    repl = Repl([Agent("agent", "You are a helpful assistant.")])
    asyncio.run(repl.run())
