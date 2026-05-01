import logging
from pathlib import Path

import frontmatter
from pydantic import BaseModel

from core.utils.common import _find_git_root

logger = logging.getLogger(__name__)


class Command(BaseModel):
    name: str
    description: str
    template: str


class CommandLoader:
    def __init__(self, command_dir: Path | str = None):
        self.command_dir = _find_git_root(Path(".")) / "core/commands"

    def load_command(self, name) -> Command:
        command_file = self.command_dir / name / f"{name}.md"
        if not command_file.exists():
            return None
        parsed = frontmatter.load(command_file, encoding="utf-8")
        metadata = dict(parsed.metadata)

        return Command(
            name=name,
            description=metadata.get("description", ""),
            template=parsed.content,
        )

    def list_commands(self) -> list[Command]:
        """List all avaiable commands"""
        commands = []

        if not self.command_dir.exists():
            return commands

        for md_file in sorted(self.command_dir.glob("*/*.md")):
            name = md_file.stem
            command = self.load_command(name)
            commands.append(command)
        return commands


if __name__ == "__main__":

    c = CommandLoader(".")
    print(c.list_commands())
    print(c)
