import logging

import pytest

from core.tools.message_classes import (
    TextPrompt,
    TextResult,
    ToolCall,
    ToolFormattedResult,
)
from core.utils.dialog import DialogMessages
from core.utils.types import ToolCallParameters


@pytest.fixture
def logger():
    """Create a logger for testing"""
    return logging.getLogger("test")


@pytest.fixture
def dialog(logger):
    """Create a DialogMessages instance for testing"""
    return DialogMessages(logger_for_agent_logs=logger)


class TestDialogMessages:
    """Test suite for DialogMessages class"""

    def test_initialization(self, dialog):
        """Test that DialogMessages initializes correctly"""
        assert dialog._message_lists == []
        assert dialog.use_prompt_budgeting == False
        assert dialog.token_budget_to_trigger_truncation == 120000

    def test_is_user_turn_initially_true(self, dialog):
        """Test that the first turn is a user turn"""
        assert dialog.is_user_turn() == True
        assert dialog.is_assistant_turn() == False

    def test_add_user_prompt(self, dialog):
        """Test adding a user prompt"""
        message = "Hello, world!"
        dialog.add_user_prompt(message)

        assert len(dialog._message_lists) == 1
        assert isinstance(dialog._message_lists[0][0], TextPrompt)
        assert dialog._message_lists[0][0].text == message

    def test_add_user_prompt_on_user_turn_fails(self, dialog):
        """Test that adding a user prompt on user turn without allow_append_to_tool_call_results raises error"""
        dialog.add_user_prompt("First message")
        dialog._message_lists.append([TextResult("Assistant response")])

        with pytest.raises(AssertionError):
            dialog.add_user_prompt("Second message")

    def test_is_user_turn_alternates(self, dialog):
        """Test that user and assistant turns alternate"""
        dialog.add_user_prompt("User message")
        assert dialog.is_assistant_turn() == True
        assert dialog.is_user_turn() == False

        dialog._message_lists.append([TextResult("Assistant response")])
        assert dialog.is_user_turn() == True
        assert dialog.is_assistant_turn() == False

    def test_add_tool_call_result(self, dialog):
        """Test adding a tool call result"""
        dialog.add_user_prompt("Run tool")
        dialog._message_lists.append([TextResult("Response")])

        params = ToolCallParameters(
            tool_call_id="123", tool_name="test_tool", tool_input='{"test": "input"}'
        )
        dialog.add_tool_call_result(params, "Tool output")

        assert len(dialog._message_lists) == 3
        tool_result = dialog._message_lists[2][0]
        assert isinstance(tool_result, ToolFormattedResult)
        assert tool_result.tool_call_id == "123"
        assert tool_result.tool_name == "test_tool"
        assert tool_result.tool_output == "Tool output"

    def test_add_multiple_tool_call_results(self, dialog):
        """Test adding multiple tool call results"""
        dialog.add_user_prompt("Run tools")
        dialog._message_lists.append([TextResult("Response")])

        params_list = [
            ToolCallParameters("id1", "tool1", '{"a": 1}'),
            ToolCallParameters("id2", "tool2", '{"b": 2}'),
        ]
        results = ["output1", "output2"]

        dialog.add_tool_call_results(params_list, results)

        assert len(dialog._message_lists[2]) == 2
        assert dialog._message_lists[2][0].tool_name == "tool1"
        assert dialog._message_lists[2][1].tool_name == "tool2"

    def test_get_pending_tool_calls(self, dialog):
        """Test retrieving pending tool calls"""
        dialog.add_user_prompt("Message")

        tool_calls = [
            ToolCall(tool_call_id="123", tool_name="test", tool_input={"key": "value"})
        ]
        dialog._message_lists.append(tool_calls)

        pending = dialog.get_pending_tool_calls()
        assert len(pending) == 1
        assert pending[0].tool_call_id == "123"
        assert pending[0].tool_name == "test"

    def test_clear(self, dialog):
        """Test clearing all messages"""
        dialog.add_user_prompt("Message 1")
        dialog.add_user_prompt("Message 2", allow_append_to_tool_call_results=True)

        assert len(dialog._message_lists) > 0
        dialog.clear()
        assert len(dialog._message_lists) == 0

    def test_get_messages_for_llm_client(self, dialog):
        """Test getting formatted messages for LLM client"""
        dialog.add_user_prompt("User message")
        dialog._message_lists.append([TextResult("Assistant response")])

        messages = dialog.get_messages_for_llm_client()
        assert isinstance(messages, list)
        assert len(messages) == 2

    def test_get_last_user_prompt(self, dialog):
        """Test retrieving the last user prompt"""
        text = "What is the answer?"
        dialog.add_user_prompt(text)
        dialog._message_lists.append([TextResult("42")])

        last_prompt = dialog.get_last_user_prompt()
        assert last_prompt == text

    def test_get_last_model_text_response(self, dialog):
        """Test retrieving the last model text response"""
        dialog.add_user_prompt("Question")
        response_text = "Answer"
        dialog._message_lists.append([TextResult(response_text)])

        last_response = dialog.get_last_model_text_response()
        assert last_response == response_text

    def test_replace_last_user_prompt(self, dialog):
        """Test replacing the last user prompt"""
        dialog.add_user_prompt("Original")
        dialog._message_lists.append([TextResult("Response")])

        dialog.replace_last_user_prompt("Replacement")
        assert dialog.get_last_user_prompt() == "Replacement"

    def test_drop_final_assistant_turn(self, dialog):
        """Test dropping the final assistant turn"""
        dialog.add_user_prompt("User message")
        dialog._message_lists.append([TextResult("Assistant response")])

        assert dialog.is_user_turn() == True
        dialog.drop_final_assistant_turn()
        assert dialog.is_user_turn() == False  # Now it's assistant turn again

    def test_drop_tool_calls_from_final_turn(self, dialog):
        """Test removing tool calls from the final assistant turn"""
        dialog.add_user_prompt("Message")

        tool_call = ToolCall(tool_call_id="123", tool_name="test", tool_input={})
        text_result = TextResult("Some text")
        dialog._message_lists.append([tool_call, text_result])

        dialog.drop_tool_calls_from_final_turn()

        final_turn = dialog._message_lists[-1]
        assert len(final_turn) == 1
        assert isinstance(final_turn[0], TextResult)

    def test_count_tokens(self, dialog):
        """Test token counting"""
        dialog.add_user_prompt("This is a test message with multiple words")
        dialog._message_lists.append([TextResult("Response text")])

        token_count = dialog.count_tokens()
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_initialization_with_prompt_budgeting(self, logger):
        """Test initialization with prompt budgeting enabled"""
        dialog = DialogMessages(logger_for_agent_logs=logger, use_prompt_budgeting=True)
        assert dialog.use_prompt_budgeting == True

    def test_str_representation(self, dialog):
        """Test string representation"""
        dialog.add_user_prompt("Test message")

        str_repr = str(dialog)
        assert isinstance(str_repr, str)
        assert "Test message" in str_repr
