from typing import Any, Optional

from core.utils.dialog import DialogMessages
from core.utils.tool_common import LLMTool
from core.types.agent_types import ToolImplOutput
from core.commands.loader import CommandLoader


def process_template(template: str, arguments: str) -> str:
    """Process $! $1 $3 Arguemtns"""
    # for simplicity cba at the moment

    return template


class CommandTool(LLMTool):
    """Wrapper for Command loader that exposes command() as a Tool"""

    name = "commands"
    description = "Use this tool if you see input beginning with a /command, use this tool to load and execute it. If the command doesn't exist, let the user know."
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The command name (without the / prefix).",
            },
            "arguments": {
                "type": "string",
                "description": "Optional arguments to substitute into the template.",
            },
        },
        "required": ["name"],
    }

    def __init__(self, command_loader: CommandLoader):
        super().__init__()
        self.command_loader = command_loader

    @property
    def should_stop(self):
        return self.command is not None

    def reset(self):
        self.command = None

    def get_tool_start_message(self, tool_input: dict[str, Any]):
        return ""

    def run_impl(
        self, tool_input: dict[str, Any], dialog_messages: Optional[DialogMessages]
    ) -> ToolImplOutput:
        assert tool_input["name"], "Model returned empty command"

        name = tool_input["name"]
        if name.startswith("/"):
            name = name.lstrip("/")

        self.command = self.command_loader.load_command(name)

        if self.command is None:
            return ToolImplOutput(
                f"Command not found: /{name}",
                f"Command not found: /{name}",
            )
        content = process_template(
            self.command.template, tool_input.get("arguments", "")
        )
        return ToolImplOutput(content, content)
