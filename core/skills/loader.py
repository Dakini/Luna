import logging
from pathlib import Path
from typing import Any

import frontmatter
from pydantic import BaseModel

from core.utils.common import _find_git_root

logger = logging.getLogger(__name__)


class Skill(BaseModel):
    name: str
    description: str
    metadata: dict[str, Any]
    content: str


class SkillLoader:

    def __init__(self, path: str | None = None):
        self.skills_dir = _find_git_root(Path(path)) / "core/skills"

    def load_skill(self, name):

        skill_file = self.skills_dir / name / "SKILL.md"
        if not skill_file.exists():
            logger.warning(f"The skill %s was not found", name)
            return None

        parsed = frontmatter.load(skill_file)

        return Skill(
            name=parsed.metadata.get("name", "name"),
            description=parsed.metadata.get("description", ""),
            metadata=parsed.metadata,
            content=parsed.content,
        )

    def list_skills(
        self,
    ) -> list[Skill]:
        skills = []

        for skill_dir in sorted(self.skills_dir.iterdir()):
            if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
                continue

            skill = self.load_skill(skill_dir.name)
            skills.append(skill)
        return skills

    def get_descriptions(self) -> str:
        skills = self.list_skills()
        skills_listing = "\n".join(f"- {s.name}: {s.description}" for s in skills)
        return skills_listing


if __name__ == "__main__":
    s = SkillLoader(".")
    print(s.load_skill("hello"))
    print(s.get_descriptions())
    print(s)
