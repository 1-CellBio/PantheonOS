import asyncio

from pydantic import BaseModel
from rich.console import Console

from .agent import Agent
from .utils.misc import print_agent_message


class Task(BaseModel):
    name: str
    goal: str


class SubTask(BaseModel):
    name: str
    goal: str
    inputs: list[str]
    outputs: list[str]


class TasksSolver:
    def __init__(
            self,
            tasks: list[Task] | Task,
            agent: Agent,
        ):
        if isinstance(tasks, Task):
            tasks = [tasks]
        self.tasks = tasks
        self.agent = agent
        self.console = Console()

    async def process_agent_messages(self):
        while True:
            message = await self.agent.events_queue.get()
            print_agent_message(self.agent.name, message, self.console)

    async def solve_sub_tasks(self, sub_tasks: list[SubTask]):
        for i, task in enumerate(sub_tasks):
            self.console.print(f"Solving sub-task [blue]{task.name}[/blue] ({i+1}/{len(sub_tasks)}): [yellow]{task.goal}[/yellow]")
            # Thinking
            prompt = f"Thinking about how to solve the sub-task: {task.name}\nGoal: {task.goal}\nInputs: {task.inputs}\nOutputs: {task.outputs}"
            await self.agent.run(prompt, tool_use=False)
            # Action
            prompt = f"Now, please using the tools to do the actions to solve the sub-task: {task.name}\nInputs: {task.inputs}\nOutputs: {task.outputs}"
            await self.agent.run(prompt, tool_use=True)
            # Reflexion
            await self.reflexion(task)

    async def judge_task_solved(self, task: Task | SubTask) -> bool:
        if isinstance(task, Task):
            prompt = f"The task '{task.name}': \n{task.goal} \n This task has been totally solved or not? You can use the tools to check the result."
        else:
            prompt = f"The sub-task '{task.name}': \nGoal: {task.goal} \nInputs: {task.inputs} \nOutputs: {task.outputs} \n This sub-task has been totally solved or not? You can use the tools to check the result."
        await self.agent.run(prompt, tool_use=True)
        resp = await self.agent.run(

            f"Now, Please tell me if the task '{task.name}' has been totally solved or not.",
            response_format=bool,
            tool_use=False,
        )
        return resp.content

    async def reflexion(self, task: Task | SubTask):
        while True:  # Reflexion loop
            if isinstance(task, Task):
                if await self.judge_task_solved(task):
                    self.console.print(f"[green]Task [blue]{task.name}[/blue] has been solved.[/green]")
                    break
                else:
                    self.console.print(f"[red]Task [blue]{task.name}[/blue] has not been solved.[/red]")
                    await self.agent.run("The task has not been solved, please analyze the reason", tool_use=False)
                    await self.agent.run("Please try again to solve the task.", tool_use=True)
            else:
                if await self.judge_task_solved(task):
                    self.console.print(f"[green]Sub-task [blue]{task.name}[/blue] has been solved.[/green]")
                    break
                else:
                    self.console.print(f"[red]Sub-task [blue]{task.name}[/blue] has not been solved.[/red]")
                    await self.agent.run("The sub-task has not been solved, please analyze the reason", tool_use=False)
                    await self.agent.run("Please try again to solve the sub-task.", tool_use=True)

    async def break_task(self, task: Task) -> list[SubTask]:
        # Breaking the task into smaller tasks
        prompt = (
            f"Please think about how to break the task: {task.name}: {task.goal}\ninto smaller sub-tasks. \n"
            "Please put necessary information(like the input file path, variables, etc.) into the sub-tasks's inputs."
            "Please put expected outputs into the sub-tasks's outputs."
        )
        await self.agent.run(prompt, tool_use=False)
        prompt = f"Output the sub-tasks"
        resp = await self.agent.run(prompt, response_format=list[SubTask], tool_use=False)
        sub_tasks = resp.content
        return sub_tasks

    async def solve_task_directly(self, task: Task):
        await self.agent.run(f"Thinking about how to solve the task: {task.name}\nGoal: {task.goal}", tool_use=False)
        await self.agent.run(f"Solve the task.", tool_use=True)

    async def solve_task(self, task: Task, split_task: bool = False):
        if split_task:
            sub_tasks = await self.break_task(task)
            self.console.print("Sub-tasks:")
            for i, sub_task in enumerate(sub_tasks):
                self.console.print(f"  {i+1}. {sub_task.name}:")
                self.console.print(f"    Goal: {sub_task.goal}")
                self.console.print(f"    Inputs: {sub_task.inputs}")
                self.console.print(f"    Outputs: {sub_task.outputs}")
            await self.solve_sub_tasks(sub_tasks)
            await self.judge_task_solved(task)
        else:
            await self.solve_task_directly(task)
            await self.reflexion(task)

    async def solve(self, split_task: bool = False):
        import logging
        logging.getLogger().setLevel(logging.WARNING)

        print_task = asyncio.create_task(self.process_agent_messages())

        for i, task in enumerate(self.tasks):  # Outer loop: Solve each task
            self.console.print(f"Solving task [blue]{task.name}[/blue] ({i+1}/{len(self.tasks)}): [yellow]{task.goal}[/yellow]")
            await self.solve_task(task, split_task)

        print_task.cancel()
