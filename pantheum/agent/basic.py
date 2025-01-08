import copy
import json
import inspect
from typing import Optional, List, Callable, Union
from uuid import uuid4

from funcdesc.parse import parse_func
from pydantic import BaseModel
from litellm import acompletion, stream_chunk_builder

from ..utils.misc import desc_to_openai_function
from ..types import AgentResponse, ResponseDetails, AgentInput


__CTX_VARS_NAME__ = "context_variables"


class Agent:
    def __init__(
        self,
        name: str,
        instructions: str,
        model: str = "gpt-4o-mini",
        tools: Optional[List[Callable]] = None,
        response_format: Optional[BaseModel] = None,
    ):
        self.id = uuid4()
        self.name = name
        self.instructions = instructions
        self.model = model
        self.functions = {}
        if tools:
            for func in tools:
                self.functions[func.__name__] = func
        self.response_format = response_format

    def tool(self, func: Callable):
        """
        Add a tool to the agent.
        """
        self.functions[func.__name__] = func
        return self

    def _convert_functions(self) -> List[dict]:
        """Convert function to the format that the model can understand."""
        functions = []

        for func in self.functions.values():
            func_dict = desc_to_openai_function(
                parse_func(func),
                skip_params=[__CTX_VARS_NAME__],
            )
            functions.append(func_dict)
        return functions

    async def handle_tool_calls(
            self, tool_calls: List, context_variables: dict):
        messages = []
        for call in tool_calls:
            func_name = call["function"]["name"]
            params = json.loads(call["function"]["arguments"])
            func = self.functions[func_name]
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                params[__CTX_VARS_NAME__] = context_variables
            if inspect.iscoroutinefunction(func):
                result = await func(**params)
            else:
                result = func(**params)
            context_variables[call["id"]] = result
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "tool_name": func_name,
                "content": repr(result),
            })
        return messages

    async def run_stream(
        self,
        messages: List[dict],
        process_chunk: Optional[Callable] = None,
        process_step_message: Optional[Callable] = None,
        max_turns: Union[int, float] = float("inf"),
        context_variables: Optional[dict] = None,
        response_format: Optional[BaseModel] = None,
    ):
        response_format = response_format or self.response_format
        history = copy.deepcopy(messages)
        history.insert(0, {"role": "system", "content": self.instructions})
        init_len = len(history)
        if context_variables is None:
            context_variables = {}

        while len(history) - init_len < max_turns:

            message = {}
            history_clear_parsed = copy.deepcopy(history)
            for msg in history_clear_parsed:
                if "parsed" in msg:
                    del msg["parsed"]
            response = await acompletion(
                model=self.model,
                messages=history_clear_parsed,
                tools=self._convert_functions() or None,
                response_format=response_format,
                stream=True,
            )
            async for chunk in response:
                if process_chunk:
                    choice = chunk.choices[0]
                    if choice.finish_reason == "stop":
                        break
                    process_chunk(choice.delta.model_dump())
            complete_resp = stream_chunk_builder(response.chunks)
            message = complete_resp.choices[0].message.model_dump()

            if response_format is not None:
                content = message.get("content")
                if content:
                    parsed = response_format.model_validate_json(content)
                    message["parsed"] = parsed

            history.append(message)
            if process_step_message:
                process_step_message(message)

            if not message["tool_calls"]:
                break

            tool_messages = await self.handle_tool_calls(
                message["tool_calls"],
                context_variables=context_variables,
            )
            history.extend(tool_messages)
            if process_step_message:
                for msg in tool_messages:
                    process_step_message(msg)

        return ResponseDetails(
            messages=history[init_len:],
            context_variables=context_variables,
        )

    async def run(
            self, msg: AgentInput,
            response_format: Optional[BaseModel] = None,
            context_variables: Optional[dict] = None,
            process_chunk: Optional[Callable] = None,
            process_step_message: Optional[Callable] = None,
            ) -> AgentResponse:
        assert isinstance(msg, (list, str, BaseModel, AgentResponse)), \
            "Message must be a list, string, BaseModel or AgentResponse"
        response_format = response_format or self.response_format
        if isinstance(msg, AgentResponse):
            # For acceping the result of previous run or other agent
            msg = msg.content

        # Convert message to the openai message format
        if isinstance(msg, BaseModel):
            messages = [{"role": "user", "content": msg.model_dump_json()}]
        elif isinstance(msg, str):
            messages = [{"role": "user", "content": msg}]
        elif isinstance(msg, list):
            new_messages = []
            for m in msg:
                if isinstance(m, BaseModel):
                    new_messages.append(
                        {"role": "user", "content": m.model_dump_json()})
                elif isinstance(m, str):
                    new_messages.append({"role": "user", "content": m})
                else:
                    assert isinstance(m, dict), \
                        "Message must be a string, BaseModel or dict"
                    new_messages.append(m)
            messages = new_messages
        details = await self.run_stream(
            messages=messages,
            response_format=response_format,
            context_variables=context_variables,
            process_chunk=process_chunk,
            process_step_message=process_step_message,
        )
        final_msg = details.messages[-1]
        if response_format:
            content = final_msg.get("parsed")
        else:
            content = final_msg.get("content")
        return AgentResponse(
            agent_name=self.name,
            content=content,
            details=details,
        )
