from pathlib import Path
from typing import Optional


def _find_git_root(start: Path) -> Optional[Path]:
    """walk from start and its parents in search of a .git folder
    Returns the directory containing .git
    """

    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None
