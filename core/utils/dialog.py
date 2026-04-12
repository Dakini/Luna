import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, cast

import jsonschema

from core.tools.message_classes import (
    AnthropicThinkingBlock,
    AssistantContentBlock,
    GeneralContentBlock,
    LLMMessages,
    TextPrompt,
    TextResult,
    ToolCall,
    ToolFormattedResult,
)
from core.utils.token_counter import ClaudeTokenCounter
from core.types.agent_types import ToolCallParameters


class DialogMessages:
    """Keeps tracks of messages that compose a dialog

    A dialog alternates between the user and assistant turns. Each turn consists of one or more messages

    A user turn consists of one or more prompts and tool results
    An assistant turn consists of a model answer and tools call
    """

    def __init__(
        self, logger_for_agent_logs: logging.Logger, use_prompt_budgeting: bool = False
    ):
        self.logger_for_agent_logs = logger_for_agent_logs
        self._message_lists: List[List[GeneralContentBlock]] = []
        self.token_counter = ClaudeTokenCounter()
        self.use_prompt_budgeting = use_prompt_budgeting
        self.truncation_history_token_cts: list[int] = []
        self.token_budget_to_trigger_truncation = 120000
        self.truncate_all_but_N: int = 3

    def count_tokens(self) -> int:
        total_tokens = 0
        for i, message_list in enumerate(self._message_lists):
            is_last_message = i == len(self._message_lists) - 1

            for message in message_list:
                if isinstance(message, (TextPrompt, TextResult)):
                    total_tokens += self.token_counter.count_tokens(message.text)
                elif isinstance(message, ToolFormattedResult):
                    total_tokens += self.token_counter.count_tokens(message.tool_output)
                elif isinstance(message, ToolCall):
                    total_tokens += self.token_counter.count_tokens(
                        json.dumps(message.tool_input)
                    )
                elif isinstance(message, AnthropicThinkingBlock):

                    total_tokens += (
                        self.token_counter.count_tokens(message.thinking)
                        if is_last_message
                        else 0
                    )
                else:
                    raise ValueError(f"Unknown message type: {message}")
        return total_tokens

    def run_compaction_strategy(self):
        """Truncate all tool results apart from the last N turns"""

        print(
            f"Truncating all but the last {self.truncate_all_but_N} turns as we hit the token budget {self.token_budget_to_trigger_truncation}"
        )
        self.logger_for_agent_logs.info(
            f"Truncating all but the last {self.truncate_all_but_N} turns as we hit the token budget {self.token_budget_to_trigger_truncation}"
        )
        old_token_count = self.count_tokens()

        new_message_lists: list[list[GeneralContentBlock]] = copy.deepcopy(
            self._message_lists
        )
        for message_list in new_message_lists[: -self.truncate_all_but_N]:
            for message in message_list:
                if isinstance(message, ToolFormattedResult):
                    message.tool_output = "[Truncated...re-run tool if you need to see input/output again.]"
                elif isinstance(message, ToolCall):
                    if message.tool_name == "sequential_thinking":
                        message.tool_input["thought"] = (
                            "[Truncated...re-run tool if you need to see input/output again.]"
                        )
                    elif message.tool_name == "str_replace_editor":
                        if "file_text" in message.tool_input:
                            message.tool_input["file_text"] = (
                                "[Truncated...re-run tool if you need to see input/output again.]"
                            )
                        elif "old_str" in message.tool_input:
                            message.tool_input["old_str"] = (
                                "[Truncated...re-run tool if you need to see input/output again.]"
                            )
                        elif "new_str" in message.tool_input:
                            message.tool_input["new_str"] = (
                                "[Truncated...re-run tool if you need to see input/output again.]"
                            )

        self._message_lists = new_message_lists
        new_token_count = self.count_tokens()
        print(
            f"Token count before compaction: {old_token_count}, after compaction: {new_token_count}"
        )
        self.truncation_history_token_cts.append(old_token_count - new_token_count)

    def get_messages_for_llm_client(self) -> LLMMessages:
        """
        Get messages for the llm CLient Truncation and
        Compation strategy should be implemented here
        """
        if (
            self.use_prompt_budgeting
            and self.count_tokens() > self.token_budget_to_trigger_truncation
        ):
            self.run_compaction_strategy()
        return list(self._message_lists)

    def add_user_message(
        self, message: str, allow_append_to_tool_call_results: bool = False
    ):
        """
        Add a user message to the dialog

        Args:
            message: the message to add
            allow_append_to_tool_call_results: If true and the last message is a tool call result, then the message will be appended to that turn
        """
        if self.is_user_turn():
            self._message_lists.append([TextPrompt(message)])
        else:
            if allow_append_to_tool_call_results:
                user_messages = self._message_lists[-1]
                for user_message in user_messages:
                    if isinstance(user_message, TextPrompt):
                        raise ValueError(
                            f"Last user turn already contains a text prompt: {user_message}"
                        )

                user_messages.append(TextPrompt(message))
            else:
                self._assert_user_turn()

    def add_tool_call_result(self, parameters: ToolCallParameters, result: str):
        """Add the results of a tool call to the dialog"""
        self.add_tool_call_results([parameters], [result])

    def add_tool_call_results(
        self, parameters_list: list[ToolCallParameters], results: list[str]
    ):
        """Add the results of multiple tool calls to the dialog"""
        self._assert_user_turn()
        self._message_lists.append(
            [
                ToolFormattedResult(
                    tool_call_id=params.tool_call_id,
                    tool_name=params.tool_name,
                    tool_output=result,
                )
                for params, result in zip(parameters_list, results)
            ]
        )

    def add_model_response(self, response: list[AssistantContentBlock]):
        """Add the result of a model call to the dialog"""
        self._assert_assistant_turn()
        self._message_lists.append(cast(list[GeneralContentBlock], response))

    def clear(self):
        """Delete all messages"""
        self._message_lists = []

    def is_user_turn(self) -> bool:
        return len(self._message_lists) % 2 == 0

    def is_assistant_turn(self) -> bool:
        return len(self._message_lists) % 2 == 1

    def __str__(self) -> str:
        json_serialisable = [
            [message.to__dict() for message in message_list]
            for message_list in self._message_lists
        ]
        return json.dumps(json_serialisable, indent=4)

    def get_summary(self, max_str_len: int = 100):
        """Return a summary of the dialog"""

        def truncate_str(obj):
            if isinstance(obj, str):
                if len(obj) > max_str_len:
                    return obj[:max_str_len] + "..."
            elif isinstance(obj, dict):
                return {k: truncate_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [truncate_str(item) for item in obj]
            return obj

    def _assert_user_turn(self):
        assert self.is_user_turn(), "Can only add user prompts on users turn"

    def _assert_assistant_turn(self):
        assert (
            self.is_assistant_turn(),
            "Can only get/replace the last user prompt on assistants turn",
        )

    def drop_final_assistant_turn(self):
        """Remove the final assistant turn

        This allows dialog messages to be passed to tools as theya are called without containing the final tool call.
        """
        if self.is_user_turn():
            self._message_lists.pop()

    def drop_tool_calls_from_final_turn(self):
        """Remove tool calls from the final assistant turn
        This allows the dialog messages to be passed to the tools
        as they are called without containing the final tool call"
        """
        if self.is_user_turn():
            new_turn_messages = [
                message
                for message in self._message_lists[-1]
                if not isinstance(message, ToolCall)
            ]
            self._message_lists[-1] = new_turn_messages

    def get_pending_tool_calls(self) -> list[ToolCallParameters]:
        """Return tool calls from the last assistant turn

        returns an empty array list of no tool calls are pending
        """

        self._assert_user_turn()
        if len(self._message_lists) == 0:
            return []
        tool_calls = []
        for message in self._message_lists[-1]:
            if isinstance(message, ToolCall):
                tool_calls.append(
                    ToolCallParameters(
                        tool_call_id=message.tool_call_id,
                        tool_name=message.tool_name,
                        tool_input=message.tool_input,
                    )
                )
        return tool_calls

    def get_last_model_text_response(self):
        self._assert_user_turn()
        for message in self._message_lists[-1]:
            if isinstance(message, TextResult):
                return message.text
        raise ValueError("No text response found in last model response")

    def get_last_user_prompt(self):
        self._assert_assistant_turn()
        for message in self._message_lists[-1]:
            if isinstance(message, TextPrompt):
                return message.text
        raise ValueError("No text prompt found in last user turn")

    def replace_last_user_prompt(self, new_prompt: str):
        """Replace the last prompt with a new one."""
        self._assert_assistant_turn()
        for i, message in enumerate(self._message_lists[-1]):
            if isinstance(message, TextPrompt):
                self._message_lists[-1][i] = TextPrompt(new_prompt)
                return
        raise ValueError("No text prompt found in last user turn")
