import json
from dataclasses import dataclass
from typing import Any

from anthropic.types import \
    RedactedThinkingBlock as AnthropicRedactedThinkingBlock
from anthropic.types import TextBlock as AnthropicTextBlock
from anthropic.types import ThinkingBlock as AnthropicThinkingBlock
from anthropic.types import ToolParam as AnthropicToolParam
from anthropic.types import \
    ToolResultBlockParam as AnthropicToolResultBlockParam
from anthropic.types import ToolUseBlock as AnthropicToolUseBlock
from dataclasses_json import DataClassJsonMixin


@dataclass
class ToolParam(DataClassJsonMixin):
    """Internal Representation of a LLM Tool"""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall(DataClassJsonMixin):
    """Internal Representation of a Tool Call"""

    tool_call_id: str
    tool_name: str
    tool_input: dict[str, Any]


@dataclass
class ToolResult(DataClassJsonMixin):
    """Internal Representation of result of a Tool Call"""

    tool_call_id: str
    tool_name: str
    tool_input: dict[str, Any]


@dataclass
class ToolFormattedResult(DataClassJsonMixin):
    """Internal Representation of a formatted result of a tool call"""

    tool_call_id: str
    tool_name: str
    tool_output: dict[str, Any]


@dataclass
class TextPrompt(DataClassJsonMixin):
    """An internal representation of a user text prompt"""

    text: str


@dataclass
class TextResult(DataClassJsonMixin):
    """An internal representation of LLM-generated text"""

    text: str


AssistantContentBlock = (
    TextResult | ToolCall | AnthropicRedactedThinkingBlock | AnthropicThinkingBlock
)
UserContentBlock = TextPrompt | ToolFormattedResult
GeneralContentBlock = UserContentBlock | AssistantContentBlock
LLMMessages = list[list[GeneralContentBlock]]
