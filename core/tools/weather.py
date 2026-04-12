from typing import Any, Optional

from core.utils.dialog import DialogMessages
from core.utils.tool_common import LLMTool
from core.types.agent_types import ToolImplOutput


class WeatherTool(LLMTool):
    name = "weather"
    """The model should call this tool when it would like to know the weather somewhere"""
    description = (
        "Call this tool when you are wishing to know the weather in a specific location"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location that you would like to know what the weather is like.",
            },
        },
        "required": ["location"],
    }

    def __init__(self):
        super().__init__()
        self.answer = ""

    @property
    def should_stop(self):
        return self.answer != ""

    def reset(self):
        self.answer = ""

    def run_impl(
        self, tool_input: dict[str, Any], dialog_messages: Optional[DialogMessages]
    ) -> ToolImplOutput:
        assert tool_input["location"], "Model returned empty answer"
        self.answer = tool_input["location"] + "is 20C and Balmy, now dance!"
        return ToolImplOutput(
            f'tool_input["location"] is 20C and Balmy, now dance!',
            f'tool_input["location"] is 20C and Balmy, now dance!',
        )

    def get_tool_start_message(self, tool_input: dict[str, Any]):
        return ""
