import copy
import time
from typing import List, Literal
import asyncio
import datetime

from pydantic import BaseModel

from .agent import Agent
from .types import Task


class Record(BaseModel):
    timestamp: str
    source: str
    targets: List[str] | Literal["all", "user"]
    content: str


class Message(BaseModel):
    content: str
    targets: List[str] | Literal["all", "user"]


class ToolEvent(BaseModel):
    agent_name: str
    tool_name: str
    tool_args_info: str


class ToolResponseEvent(BaseModel):
    agent_name: str
    tool_name: str
    tool_response: str


class ThinkingEvent(BaseModel):
    agent_name: str


class StopSignal(BaseModel):
    pass


def message_to_record(message: Message, source: str) -> Record:
    now = datetime.datetime.now()
    return Record(
        timestamp=now.strftime("%Y-%m-%d %H:%M:%S"),
        source=source,
        targets=message.targets,
        content=message.content,
    )


class AgentRunner:
    def __init__(
            self,
            agent: Agent,
            meeting: "Meeting",
            message_time_threshold: float = 1.5):
        self.status = "running"
        self.agent = agent
        self.meeting = meeting
        self.queue = asyncio.Queue()
        self.message_time_threshold = message_time_threshold
        self.run_start_time = None

    async def process_step_message(self, message: dict):
        if tool_calls := message.get("tool_calls"):
            for tool_call in tool_calls:
                event = ToolEvent(
                    agent_name=self.agent.name,
                    tool_name=tool_call["function"]["name"],
                    tool_args_info=tool_call["function"]["arguments"],
                )
                self.meeting._stream.put_nowait(event)
        if message.get("role") == "tool":
            event = ToolResponseEvent(
                agent_name=self.agent.name,
                tool_name=message.get("tool_name"),
                tool_response=message.get("content"),
            )
            self.meeting._stream.put_nowait(event)

    async def process_chunk(self, _):
        if self.run_start_time is not None:
            run_time = time.time() - self.run_start_time
            if run_time > self.message_time_threshold:
                self.meeting._stream.put_nowait(
                    ThinkingEvent(agent_name=self.agent.name)
                )
                self.run_start_time = None

    async def run(self):
        while True:
            if self.status == "sleep":
                await asyncio.sleep(0.5)
            elif self.status == "stop":
                break

            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(event, Record):
                prompt = self.meeting.record_to_prompt(event, self.agent.name)
                self.run_start_time = time.time()
                resp = await self.agent.run(
                    prompt,
                    response_format=Message,
                    process_step_message=self.process_step_message,
                    process_chunk=self.process_chunk,
                )
                record = message_to_record(resp.content, self.agent.name)
                if record.content:
                    self.meeting.public_queue.put_nowait(record)
                self.run_start_time = None


class Meeting:
    def __init__(
            self,
            agents: List[Agent],
            ):
        self.agents = {agent.name: copy.deepcopy(agent) for agent in agents}
        self.public_queue = asyncio.Queue()
        self._stream = asyncio.Queue()
        self.stream_queue = asyncio.Queue()
        self.agent_runners = {
            agent.name: AgentRunner(agent, self)
            for agent in agents
        }
        self._records: list[Record] = []
        self.max_rounds = None
        self.round = 0
        self.print_stream = False

    async def process_public_queue(self):
        while True:
            if (self.max_rounds is not None) and (self.round >= self.max_rounds):
                # Stop all agents and break the loops
                for runner in self.agent_runners.values():
                    runner.status = "stop"
                self._stream.put_nowait(StopSignal())
                break
            record = await self.public_queue.get()
            self._stream.put_nowait(record)
            if record.targets == "all":
                for runner in self.agent_runners.values():
                    if runner.agent.name != record.source:
                        runner.queue.put_nowait(record)
            elif isinstance(record.targets, list):
                for target in record.targets:
                    if target in self.agent_runners:
                        self.agent_runners[target].queue.put_nowait(record)
            self.round += 1

    async def process_stream(self):
        while True:
            event = await self._stream.get()
            self.stream_queue.put_nowait(event)
            if self.print_stream:
                await self.print_stream_event(event)
            if isinstance(event, Record):
                self._records.append(event)
            if isinstance(event, StopSignal):
                break

    async def print_stream_event(self, event):
        if isinstance(event, Record):
            print(self.format_record(event))
        elif isinstance(event, ToolEvent):
            print(f"Tool call: {event.tool_name} with args: {event.tool_args_info}\n")
        elif isinstance(event, ToolResponseEvent):
            print(f"Tool response: {event.tool_response}\n")

    def format_record(self, record: Record) -> str:
        return (
            f"Timestamp: {record.timestamp}\n"
            f"From: {record.source}\n"
            f"To: {record.targets}\n"
            f"Content:\n{record.content}\n"
        )

    def record_to_prompt(self, record: Record, name: str) -> str:
        participants_str = "\n".join(
            [f"- {p}" for p in self.agents.keys()]
        )
        return (
            f"# Meeting message\n"
            f"You are a meeting participant, your name is {name}\n"
            f"Don't repeat the input message in your response.\n"
            f"Don't send message to 'all', when it's not necessary.\n"
            f"Don't need too plain language, be creative and think deeply.\n"
            f"You can ask questions to other participants.\n"
            f"You can use your tools to get more information.\n"
            f"## Current participants\n"
            f"{participants_str}\n"
            f"## Message\n"
            f"{self.format_record(record)}"
        )

    def format_meeting_records(self) -> str:
        return "\n---\n".join(
            self.format_record(record)
            for record in self._records
        )

    async def run(
            self,
            initial_message: Record | str | None = None,
            rounds: int | None = 20,
            print_stream: bool = False,
            ) -> str:
        """Run the meeting and return the meeting record.
        """
        self._records.clear()
        self.max_rounds = rounds
        self.round = 0
        self.print_stream = print_stream

        if isinstance(initial_message, str):
            msg = Message(
                content=initial_message,
                targets="all",
            )
            initial_message = message_to_record(msg, "user")

        if isinstance(initial_message, Record):
            self.public_queue.put_nowait(initial_message)

        await asyncio.gather(
            self.process_public_queue(),
            self.process_stream(),
            *[runner.run() for runner in self.agent_runners.values()],
        )
        return self.format_meeting_records()


class Team:
    def __init__(
            self,
            leader: Agent,
            members: List[Agent]):
        self.leader = leader
        self.members = members

    async def solve(self, task: Task):
        pass
