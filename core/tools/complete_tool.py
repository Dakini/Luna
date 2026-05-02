from typing import Any, Optional

from core.types.agent_types import ToolImplOutput
from core.utils.dialog import DialogMessages
from core.utils.tool_common import LLMTool
import mlflow
from mlflow.entities import SpanType


class CompleteTool(LLMTool):
    name = "complete"
    """The model should call this tool when it is done with its tas"""
    description = "Call this tool when you are done with the task, and supply your answer or summary"
    input_schema = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the question, or final summary of actions taken to accomplish the task",
            },
        },
        "required": ["answer"],
    }

    def __init__(self):
        super().__init__()
        self.answer = ""

    @property
    def should_stop(self):
        return self.answer != ""

    def reset(self):
        self.answer = ""

    @mlflow.trace(span_type=SpanType.TOOL)
    def run_impl(
        self, tool_input: dict[str, Any], dialog_messages: Optional[DialogMessages]
    ) -> ToolImplOutput:
        assert tool_input["answer"], "Model returned empty answer"
        self.answer = tool_input["answer"]
        return ToolImplOutput(
            tool_output="Task completed", tool_result_message="Task Completed"
        )

    def get_tool_start_message(self, tool_input: dict[str, Any]):
        return ""
