from dataclasses import dataclass, field
from typing import Any, Optional, final

import jsonschema
from anthropic import BadRequestError

from core.tools.message_classes import ToolParam
from core.utils.dialog import DialogMessages
from core.types.agent_types import ToolCallParameters

ToolInputSchema = dict[str, Any]


class Tool:
    """
    A tool that can be called by an llm

    A general tool may require additional parameters the modle doe not provide
    It may also return arbitary structures output
    We need to define a general interface
    """

    name: str
    description: str
    input_schema: ToolInputSchema


class LLMTool:
    """
    A tool that fits into the standard LLM tool calling paradigm
    An llmTool can be called by supplying the parameters specied in its input schema
    and returns a string that can be shown to the LLM
    """

    name: str
    description: str
    input_schema: ToolInputSchema

    @property
    def should_stop(self) -> bool:
        """Whether the tool should stop the agent after being called"""
        return False

    @final
    def run(
        self,
        tool_input: dict[str, Any],
        dialog_messages: Optional[DialogMessages] = None,
    ) -> str:
        """Run the tool

        Args:
            tool_input: the input to the tool
            dialog_messages: The dialog messages so far, if available
            the tool is allowed to modify this object so the caller should mamake a copy
        """

        if dialog_messages:
            assert dialog_messages.is_user_turn()

        try:
            self._validate_tool_input(tool_input)
            result = self.run_impl(tool_input, dialog_messages)
            tool_output = result.tool_output
        except jsonschema.ValidationError as exc:
            tool_output = "Invalid Tool input" + exc.message
        except BadRequestError as exc:
            raise RuntimeError("Bad Request: ", exc.message)

        return tool_output

    def get_tool_start_message(self, tool_input: ToolInputSchema) -> str:
        """Return a user friendly message to be shown to the model when the tool is called"""
        return f"Calling tool '{self.name}'"

    def run_impl(
        self,
        tool_input: dict[str, Any],
        dialog_messages: Optional[DialogMessages] = None,
    ):
        """Sub calss should implement this

        Returns:
          A ToolImplOutput containing an output string, description and any auxzillary data
        """
        raise NotImplementedError()

    def get_tool_params(self) -> ToolParam:
        return ToolParam(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    def _validate_tool_input(self, tool_input: dict[str, Any]):
        """Validate the tool input against the tool's input schema"""
        jsonschema.validate(instance=tool_input, schema=self.input_schema)


def call_tools(
    tools: list[LLMTool],
    calls_to_make: list[ToolCallParameters],
    dialog_messages: Optional[DialogMessages] = None,
):
    """
    Call the requested tools and return their output

    Args:
    tools: tools to call
    calls_to_make: the calls to make
    dialog_messages: If supplied the call results will be recorded here"""
    tool_outputs = []

    for call in calls_to_make:
        tool = next(t for t in tools if t.name == call.tool_name)
        tool_outputs.append(tool.run(call.tool_input))
    if dialog_messages:
        dialog_messages.add_tool_call_results(calls_to_make, tool_outputs)

    return tool_outputs
