from typing import Callable, List

from funcdesc.desc import NotDef
from funcdesc.pydantic import desc_to_pydantic, Description
from openai import pydantic_function_tool


def desc_to_openai_dict(
        desc: Description,
        skip_params: List[str] = []) -> dict:
    pydantic_model = desc_to_pydantic(desc)['inputs']
    oai_func_dict = pydantic_function_tool(pydantic_model)
    oai_params = oai_func_dict["function"]["parameters"]["properties"]

    parameters = {}
    required = []

    for arg in desc.inputs:
        if arg.name in skip_params:
            continue

        pdict = {
            "description": arg.doc or "",
        }
        oai_pdict = oai_params[arg.name]
        if "type" in oai_pdict:
            pdict["type"] = oai_pdict["type"]
        if "items" in oai_pdict:
            pdict["items"] = oai_pdict["items"]
        if "anyOf" in oai_pdict:
            pdict["anyOf"] = oai_pdict["anyOf"]

        parameters[arg.name] = pdict

        if arg.default is NotDef:
            required.append(arg.name)

    func_dict = {
        "type": "function",
        "function": {
            "name": desc.name,
            "description": desc.doc or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
                "additionalProperties": False,
            },
            "strict": False,
        },
    }

    return func_dict
