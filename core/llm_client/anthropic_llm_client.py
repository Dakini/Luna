import logging
import os
import random
import time
from typing import Any, Tuple, cast

import anthropic
from anthropic import NOT_GIVEN as anthropic_NOT_GIVEN
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import InternalServerError as AnthropicInternalServerError
from anthropic import RateLimitError as AnthropicRateLimitError
from anthropic._exceptions import OverloadedError as AnthropicOverloadedError
from anthropic.types.message_create_params import (
    ToolChoiceToolChoiceAny,
    ToolChoiceToolChoiceAuto,
    ToolChoiceToolChoiceTool,
)

from core.llm_client import LLMClient, recursively_remove_invoke_tags
from core.tools.message_classes import (
    AnthropicRedactedThinkingBlock,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicToolParam,
    AnthropicToolResultBlockParam,
    AnthropicToolUseBlock,
    AssistantContentBlock,
    LLMMessages,
    TextPrompt,
    TextResult,
    ToolCall,
    ToolFormattedResult,
    ToolParam,
)

# import mlflow

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("Luna-AI-Agent")

logging.getLogger("httpx").setLevel(logging.WARNING)
# mlflow.anthropic.autolog()


class AnthropicLLMClient(LLMClient):

    def __init__(
        self,
        model_name="claude-sonnet-4-6",
        max_retries: int = 2,
        use_caching: bool = True,
        thinking_tokens: int = 0,
    ):

        # disable retries as we handle it ourself
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"), max_retries=1, timeout=60 * 5
        )
        self.model_name = model_name
        self.max_retries = max_retries
        self.use_caching = use_caching
        self.prompt_caching_headers = {"antrhopic-beta": "prompt-caching-2024-07-31"}
        self.thinking_tokens = thinking_tokens

    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] = None,
        thinking_tokens: int | None = None,
        session_id: str | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """Generate a repsones from the LLM
        Args:
            messages: a list of messages
            max_tokens: the maximum number of tokens to generate
            system_prompt: the system prompt
            temperatrure: the temperture to generate a response with lower theless creative
            tools: The tools that can be called
            tool_choice a tool choice if used
            thinking tokens: Number of tokens if using a thinking model
        Returns:
            A generated response from the LLM

        """

        anthropic_messages = self.creating_message_list(messages)

        extra_headers, tool_choice_param, tool_params = self.extracting_params(
            tools, tool_choice
        )

        response = None
        if thinking_tokens is None:
            thinking_tokens = self.thinking_tokens

        extra_body = None
        if thinking_tokens and thinking_tokens > 0:
            extra_body = {
                "thinking": {"type": "enabled", "budget_tokens": thinking_tokens}
            }
            temperature = 1
            assert (
                max_tokens >= 32000 and thinking_tokens <= 8192
            ), f"Just for giggles best to have max_token generation as 32k tokens currently: {max_tokens} and thinking as 8k tokens currently : {thinking_tokens}"

        for retry in range(self.max_retries):
            try:

                response = self.client.messages.create(
                    max_tokens=max_tokens,
                    messages=anthropic_messages,
                    model=self.model_name,
                    temperature=temperature,
                    system=system_prompt or anthropic_NOT_GIVEN,
                    tool_choice=tool_choice_param,
                    tools=tool_params,
                    extra_body=extra_body,
                    extra_headers=extra_headers,
                )
                break
            except (
                AnthropicAPIConnectionError,
                AnthropicInternalServerError,
                AnthropicOverloadedError,
                AnthropicRateLimitError,
            ) as e:
                if retry == self.max_retries - 1:
                    print(f"Failed Antrhopic request after {self.max_retries} reties")
                    raise e
                else:
                    print("Retrying LLM request")
                    time.sleep(5 * random.uniform(0.8, 1.2))

        augment_messages = self.extract_augmented_messages(response)
        message_metadata = self.extract_metadata(response)

        return augment_messages, message_metadata

    def extract_augmented_messages(self, response):
        augment_messages = []

        assert response is not None
        for message in response.content:
            if "</invoke>" in str(message):
                warning_msg = "\n".join(
                    ["!" * 80, "WARNING: Unexpected 'invoke' in message", "!" * 80]
                )
                print(warning_msg)
            message_str_type = str(type(message))

            if message_str_type == str(AnthropicTextBlock):
                message = cast(AnthropicTextBlock, message)
                augment_messages.append(TextResult(message.text))

            elif message_str_type == str(AnthropicRedactedThinkingBlock):
                augment_messages.append(message)

            elif message_str_type == str(AnthropicThinkingBlock):
                message = cast(AnthropicThinkingBlock, message)
                augment_messages.append(message)
            elif message_str_type == str(AnthropicToolUseBlock):
                message = cast(AnthropicToolUseBlock, message)
                augment_messages.append(
                    ToolCall(
                        tool_call_id=message.id,
                        tool_name=message.name,
                        tool_input=recursively_remove_invoke_tags(message.input),
                    )
                )
            else:
                raise ValueError("Unknown Message Type")
        return augment_messages

    def extract_metadata(self, response):
        message_metadata = {
            "raw_response": response,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_input_tokens": getattr(
                response.usage, "cache_creation_input_tokens", -1
            ),
            "cache_read_input_tokens": getattr(
                response.usage, "cache_read_input_tokens", -1
            ),
        }

        return message_metadata

    def extracting_params(self, tools, tool_choice):
        extra_headers = None
        if self.use_caching:
            extra_headers = self.prompt_caching_headers
        if tool_choice is None:
            tool_choice_param = anthropic_NOT_GIVEN
        elif tool_choice["type"] == "any":
            tool_choice_param = ToolChoiceToolChoiceAny(type="any")
        elif tool_choice["type"] == "auto":
            tool_choice_param = ToolChoiceToolChoiceAuto(type="auto")
        elif tool_choice["type"] == "tool":
            tool_choice_param = ToolChoiceToolChoiceTool(
                type="tool", name=tool_choice["name"]
            )
        else:
            raise ValueError(f"Unknown tool_choice type: {tool_choice["type"]}")

        if len(tools) == 0:
            tool_params = anthropic_NOT_GIVEN
        else:
            tool_params = [
                AnthropicToolParam(
                    input_schema=tool.input_schema,
                    name=tool.name,
                    description=tool.description,
                )
                for tool in tools
            ]

        return extra_headers, tool_choice_param, tool_params

    def creating_message_list(self, messages):
        anthropic_messages = []
        for idx, message_list in enumerate(messages):
            role = "user" if idx % 2 == 0 else "assistant"
            message_content_list = []
            for message in message_list:
                # Check string type to avoid import issues
                message_content = self.check_message_type(message)
                message_content_list.append(message_content)
            # Anthropic supports up to 4. cache breakpoints so we append to the last 4 messages
            if self.use_caching and idx >= len(messages) - 4:
                if isinstance(message_content_list[-1], dict):
                    message_content_list[-1]["cache_control"] = {"type": "ephemeral"}
                else:
                    message_content_list[-1].cache_control = {"type": "ephemeral"}
            anthropic_messages.append({"role": role, "content": message_content_list})
        return anthropic_messages

    def check_message_type(self, message):
        if str(type(message)) == str(TextPrompt):
            message = cast(TextPrompt, message)
            message_content = AnthropicTextBlock(type="text", text=message.text)

        elif str(type(message)) == str(TextResult):
            message = cast(TextResult, message)
            message_content = AnthropicTextBlock(type="text", text=message.text)

        elif str(type(message)) == str(ToolCall):
            message = cast(ToolCall, message)
            message_content = AnthropicToolUseBlock(
                type="tool_use",
                id=message.tool_call_id,
                name=message.tool_name,
                input=message.tool_input,
            )

        elif str(type(message)) == str(ToolFormattedResult):
            message = cast(ToolFormattedResult, message)
            message_content = AnthropicToolResultBlockParam(
                type="tool_result",
                tool_use_id=message.tool_call_id,
                content=message.tool_output,
            )
        elif str(type(message)) == str(AnthropicRedactedThinkingBlock):
            message = cast(AnthropicRedactedThinkingBlock, message)
            message_content = message
        elif str(type(message)) == str(AnthropicThinkingBlock):
            message = cast(AnthropicThinkingBlock, message)
            message_content = message
        else:
            print(
                f"Unknown Message type: {type(message)}, expected one of {str(TextPrompt)}, {str(TextResult)}, {str(ToolCall)}, {str(ToolFormattedResult)}"
            )
            raise ValueError(
                f"Unknown message type: {type(message)}, expected one of {str(TextPrompt)}, {str(TextResult)}, {str(ToolCall)}, {str(ToolFormattedResult)}"
            )

        return message_content


def get_antropic_client(**kwargs):
    """Get a client for an agent"""

    return AnthropicLLMClient(**kwargs)
