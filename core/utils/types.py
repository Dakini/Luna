from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallParameters:
    tool_call_id: str
    tool_name: str
    tool_input: str


@dataclass
class ToolImplOutput:
    """Output from an LLM tool implementation

    Attributes:
        tool_output: The main string that will be shown to the model
        tool_result_message: A description of what the tool did, logging purposes
        auxiliary_data: Additional data the tool wants to pass along for logging only
    """

    tool_output: str
    tool_result_message: str = ""
    axiliary_data: dict = field(default_factory=dict)
