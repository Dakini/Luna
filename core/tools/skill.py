from typing import Any, Optional

from core.skills.loader import SkillLoader
from core.types.agent_types import ToolImplOutput
from core.utils.dialog import DialogMessages
from core.utils.tool_common import LLMTool


class SkillsTool(LLMTool):
    """Wrapper for skill loader that exposes skill() as a Tool"""

    name = "skills"
    description = "Call this tool load a skill to get specialised instructions"
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "the exact name of the skill to load.",
            },
        },
        "required": ["name"],
    }

    def __init__(self, skill_loader: SkillLoader):
        super().__init__()
        self.skill_loader = skill_loader

    @property
    def should_stop(self):
        return self.answer is not None

    def reset(self):
        self.answer = None

    def get_tool_start_message(self, tool_input: dict[str, Any]):
        return ""

    def run_impl(
        self, tool_input: dict[str, Any], dialog_messages: Optional[DialogMessages]
    ) -> ToolImplOutput:
        assert tool_input["name"], "Model returned empty answer"
        self.answer = self.skill_loader.load_skill(tool_input["name"])

        return ToolImplOutput(
            self.answer.model_dump_json(),
            self.answer.model_dump_json(),
        )
