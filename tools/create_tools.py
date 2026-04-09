import inspect
from typing import Any, get_type_hints

from pydantic import BaseModel, create_model
from pydantic.type_adapter import TypeAdapter


def make_tool_adapter(func):
    """
    Takes a python function.

    Returns tuple:
        (ArgsModel - all the arguments packages up into a pydantic struct
        tool_spec - The spec of the tool
        call_from_tool - a wrapper function that takes the dict from the LLM and unpacks the arguments and call your function
        )
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    fields: dict[str, tuple[type, Any]] = {}
    ordered_params: list[tuple[str, inspect.Parameter]] = []

    for pname, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ann = hints.get(pname, Any)
        default = ... if p.default is inspect._empty else p.default
        fields[pname] = (ann, default)
        ordered_params.append((pname, p))
    ArgsModel: type[BaseModel] = create_model(
        f"{func.__name__.capitalize()}Args", **fields
    )  # type ignore
    tool_spec = {
        # "type": "function",
        "name": func.__name__,
        "description": (func.__doc__ or f"Call {func.__name__}").strip(),
        "input_schema": {
            "type": "object",
            "properties": ArgsModel.model_json_schema()["properties"],
        },
    }

    ret_ann = hints.get("return", Any)
    ret_adapter = TypeAdapter(ret_ann)

    def call_from_tool(d: dict):
        m = ArgsModel.model_validate(d)

        posargs = []
        kwargs = {}
        for name, p in ordered_params:
            val = getattr(m, name)
            if p.kind == inspect.Parameter.POSITIONAL_ONLY:
                posargs.append(val)
            else:
                kwargs[name] = val
        result = func(*posargs, **kwargs)

        py_result = ret_adapter.dump_python(result)
        json_text = ret_adapter.dump_json(result).decode()
        return py_result, json_text

    return ArgsModel, tool_spec, call_from_tool


def create_tools(funcs):
    tools = {}

    for func in funcs:
        ArgsModel, tool_spec, call_from_array = make_tool_adapter(func)
        tools[func.__name__] = (ArgsModel, tool_spec, call_from_array)
    return tools
