from typing import Any, Optional

import pytest

from core.tools.message_classes import ToolParam
from core.utils.tool_common import LLMTool, Tool, ToolImplOutput, call_tools
from core.utils.types import ToolCallParameters


class SimpleTool(LLMTool):
    """A simple tool for testing"""

    def __init__(self):
        self.name = "simple_tool"
        self.description = "A simple test tool"
        self.input_schema = {
            "type": "object",
            "properties": {
                "input_text": {"type": "string", "description": "Input text"}
            },
            "required": ["input_text"],
        }

    def run_impl(
        self, tool_input: dict[str, Any], dialog_messages: Optional[Any] = None
    ) -> ToolImplOutput:
        """Simple implementation that echoes input"""
        return ToolImplOutput(
            tool_output=f"Processed: {tool_input['input_text']}",
            tool_result_message="Successfully processed input",
        )


class StoppingTool(LLMTool):
    """A tool that stops execution"""

    def __init__(self):
        self.name = "stopping_tool"
        self.description = "A tool that stops execution"
        self.input_schema = {"type": "object", "properties": {}, "required": []}

    @property
    def should_stop(self) -> bool:
        return True

    def run_impl(
        self, tool_input: dict[str, Any], dialog_messages: Optional[Any] = None
    ) -> ToolImplOutput:
        return ToolImplOutput(tool_output="Stopping now")


class TestToolCallParameters:
    """Test suite for ToolCallParameters dataclass"""

    def test_initialization(self):
        """Test ToolCallParameters initialization"""
        params = ToolCallParameters(
            tool_call_id="123", tool_name="test_tool", tool_input='{"key": "value"}'
        )

        assert params.tool_call_id == "123"
        assert params.tool_name == "test_tool"
        assert params.tool_input == '{"key": "value"}'

    def test_equality(self):
        """Test ToolCallParameters equality"""
        params1 = ToolCallParameters("id", "tool", '{"a": 1}')
        params2 = ToolCallParameters("id", "tool", '{"a": 1}')

        assert params1 == params2

    def test_inequality(self):
        """Test ToolCallParameters inequality"""
        params1 = ToolCallParameters("id1", "tool", "{}")
        params2 = ToolCallParameters("id2", "tool", "{}")

        assert params1 != params2


class TestToolImplOutput:
    """Test suite for ToolImplOutput dataclass"""

    def test_initialization_with_defaults(self):
        """Test ToolImplOutput initialization with defaults"""
        output = ToolImplOutput(tool_output="Test output")

        assert output.tool_output == "Test output"
        assert output.tool_result_message == ""
        assert output.axiliary_data == {}

    def test_initialization_with_all_fields(self):
        """Test ToolImplOutput initialization with all fields"""
        aux_data = {"key": "value"}
        output = ToolImplOutput(
            tool_output="Output", tool_result_message="Message", axiliary_data=aux_data
        )

        assert output.tool_output == "Output"
        assert output.tool_result_message == "Message"
        assert output.axiliary_data == aux_data


class TestLLMTool:
    """Test suite for LLMTool base class"""

    def test_simple_tool_initialization(self):
        """Test SimpleTool initialization"""
        tool = SimpleTool()

        assert tool.name == "simple_tool"
        assert tool.description == "A simple test tool"
        assert "input_text" in tool.input_schema["properties"]

    def test_tool_run_success(self):
        """Test successful tool execution"""
        tool = SimpleTool()
        result = tool.run({"input_text": "hello"})

        assert "Processed: hello" in result

    def test_tool_should_stop_default(self):
        """Test that should_stop defaults to False"""
        tool = SimpleTool()
        assert tool.should_stop == False

    def test_tool_should_stop_override(self):
        """Test tool that overrides should_stop"""
        tool = StoppingTool()
        assert tool.should_stop == True

    def test_tool_validation_error(self):
        """Test tool execution with invalid input"""
        tool = SimpleTool()
        result = tool.run({"wrong_field": "value"})

        assert "Invalid Tool input" in result

    def test_tool_get_params(self):
        """Test getting tool parameters"""
        tool = SimpleTool()
        params = tool.get_tool_params()

        assert isinstance(params, ToolParam)
        assert params.name == "simple_tool"
        assert params.description == "A simple test tool"
        assert params.input_schema == tool.input_schema

    def test_tool_get_start_message(self):
        """Test getting tool start message"""
        tool = SimpleTool()
        message = tool.get_tool_start_message({})

        assert "simple_tool" in message
        assert "Calling tool" in message


class TestCallTools:
    """Test suite for call_tools function"""

    def test_call_single_tool(self):
        """Test calling a single tool"""
        tool = SimpleTool()
        calls = [ToolCallParameters("123", "simple_tool", '{"input_text": "test"}')]

        outputs = call_tools([tool], calls)

        assert len(outputs) == 1
        assert "Processed: test" in outputs[0]

    def test_call_multiple_tools(self):
        """Test calling multiple tools"""
        tool1 = SimpleTool()
        tool2 = SimpleTool()

        calls = [
            ToolCallParameters("id1", "simple_tool", '{"input_text": "first"}'),
            ToolCallParameters("id2", "simple_tool", '{"input_text": "second"}'),
        ]

        outputs = call_tools([tool1, tool2], calls)

        assert len(outputs) == 2
        assert "first" in outputs[0]
        assert "second" in outputs[1]

    def test_call_tools_with_invalid_tool_name(self):
        """Test calling with invalid tool name raises error"""
        tool = SimpleTool()
        calls = [ToolCallParameters("id", "nonexistent_tool", "{}")]

        with pytest.raises(StopIteration):
            call_tools([tool], calls)


class TestToolValidation:
    """Test suite for tool input validation"""

    def test_valid_input_passes_validation(self):
        """Test that valid input passes validation"""
        tool = SimpleTool()
        tool_input = {"input_text": "valid input"}

        # Should not raise
        tool._validate_tool_input(tool_input)

    def test_missing_required_field_fails_validation(self):
        """Test that missing required field fails validation"""
        tool = SimpleTool()
        tool_input = {}

        with pytest.raises(Exception):
            tool._validate_tool_input(tool_input)

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed"""
        tool = SimpleTool()
        tool_input = {"input_text": "test", "extra_field": "extra"}

        # Should not raise since extra fields are allowed
        tool._validate_tool_input(tool_input)
