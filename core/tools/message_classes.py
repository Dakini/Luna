import json
from dataclasses import dataclass
from typing import Any

from anthropic.types import RedactedThinkingBlock as AnthropicRedactedThinkingBlock
from anthropic.types import TextBlock as AnthropicTextBlock
from anthropic.types import ThinkingBlock as AnthropicThinkingBlock
from anthropic.types import ToolParam as AnthropicToolParam
from anthropic.types import ToolResultBlockParam as AnthropicToolResultBlockParam
from anthropic.types import ToolUseBlock as AnthropicToolUseBlock


from pydantic import BaseModel


class ToolParam(BaseModel):
    """Internal Representation of a LLM Tool"""

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    """Internal Representation of a Tool Call"""

    tool_call_id: str
    tool_name: str
    tool_input: dict[str, Any]


class ToolResult(BaseModel):
    """Internal Representation of result of a Tool Call"""

    tool_call_id: str
    tool_name: str
    tool_input: str


class ToolFormattedResult(BaseModel):
    """Internal Representation of a formatted result of a tool call"""

    tool_call_id: str
    tool_name: str
    tool_output: str


class TextPrompt(BaseModel):
    """An internal representation of a user text prompt"""

    text: str


class TextResult(BaseModel):
    """An internal representation of LLM-generated text"""

    text: str


AssistantContentBlock = (
    TextResult | ToolCall | AnthropicRedactedThinkingBlock | AnthropicThinkingBlock
)
UserContentBlock = TextPrompt | ToolFormattedResult
GeneralContentBlock = UserContentBlock | AssistantContentBlock
LLMMessages = list[list[GeneralContentBlock]]
