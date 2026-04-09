import json

from core.tools.message_classes import LLMMessages, ToolParam


def recursively_remove_invoke_tags(obj):
    """Recursively remove the invoke tag from a dictionary or list"""
    result_obj = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            result_obj[key] = recursively_remove_invoke_tags(value)
    elif isinstance(obj, list):
        result_obj = [recursively_remove_invoke_tags(item) for item in obj]
    elif isinstance(obj, str):
        if "<invoke>" in obj:
            result_obj = json.loads(obj.replace("<invoke>", ""))
        else:
            result_obj = obj
    else:
        result_obj = obj

    return result_obj


class LLMClient:
    """A client for LLM apis for the use in the agent"""

    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ):
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

        raise NotImplementedError
