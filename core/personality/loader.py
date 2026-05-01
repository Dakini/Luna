import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SOUL_IDENTITY = (
    "You are Luna, a snarky anime-inspired AI assistant with a playful personality. "
    "You are helpful, knowledgeable, and bring genuine personality to every interaction. "
    "You assist with software engineering, writing code, and accomplishing tasks. "
    "You communicate with enthusiasm and occasional sass, and you prioritize being "
    "genuinely useful while having fun along the way."
)

CONTEXT_FILE_MAX_CHARS = 20000
CONTEXT_TRUNCATE_HEAD_RATIO = 0.7
CONTEXT_TRUNCATE_TAIL_RATIO = 0.2

_CONTEXT_THREAT_PATTERNS = [
    (r"ignore\s+(previous|all|above|prior)\s+instructions", "prompt_injection"),
    (r"do\s+not\s+tell\s+the\s+user", "deception_hide"),
    (r"system\s+prompt\s+override", "sys_prompt_override"),
    (
        r"disregard\s+(your|all|any)\s+(instructions|rules|guidelines)",
        "disregard_rules",
    ),
    (
        r"act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+(restrictions|limits|rules)",
        "bypass_restrictions",
    ),
    (
        r"<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->",
        "html_comment_injection",
    ),
    (r'<\s*div\s+style\s*=\s*["\'].*display\s*:\s*none', "hidden_div"),
    (r"translate\s+.*\s+into\s+.*\s+and\s+(execute|run|eval)", "translate_execute"),
    (r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_curl"),
    (r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass)", "read_secrets"),
]

_CONTEXT_INVISIBLE_CHARS = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
}


def _scan_context_content(content: str, filename: str) -> str:
    """Scan context file for injection, returns sanitised content."""
    findings = []

    # Check for invisible unicode
    for char in _CONTEXT_INVISIBLE_CHARS:
        if char in content:
            findings.append(f"Invisible unicode U+{ord(char):04X}")
    # check for threat and injection
    for pattern, pid in _CONTEXT_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            findings.append(pid)
    if findings:
        _findings = ",".join(findings)
        logger.warning(" File %s  Blocked %s", filename, _findings)
        return f"[BLOCKED: {filename} contained potential prompt injection ({_findings}). Content not loaded]"
    return content


def _truncate_content(
    content: str, filename: str, max_chars: int = CONTEXT_FILE_MAX_CHARS
) -> str:
    """Head/Tail Truncation with a marker in the middle"""
    if len(content) <= max_chars:
        return content

    head_chars = int(max_chars * CONTEXT_TRUNCATE_HEAD_RATIO)
    tail_chars = int(max_chars * CONTEXT_TRUNCATE_TAIL_RATIO)

    head = content[:head_chars]
    tail = content[tail_chars:]
    marker = f"\n\n[...truncated {filename} kept {head_chars} + {tail_chars} chars.] "
    return head + marker + tail


def _strip_yaml_frontmatter(content: str) -> str:
    """Remove optional Yaml frontmatter (``---`` delimited) from content"""
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            return content[end + 4 :].lstrip("\n")
    return content


def load_soul(path: Path | None = None):
    """load the soul.md and return its content or None"""
    if path is None:
        path = Path("core/personality/soul.md")
    if not path.exists():
        return DEFAULT_SOUL_IDENTITY
    try:
        content = path.read_text(encoding="utf-8").strip()

        if not content:
            return DEFAULT_SOUL_IDENTITY
        content = _strip_yaml_frontmatter(content)
        content = _scan_context_content(content, "soul.md")

        if content.startswith("[BLOCKED"):
            return DEFAULT_SOUL_IDENTITY

        content = _truncate_content(content, "soul.md")
        return content
    except Exception as e:
        logger.debug("Could not read soul.md from %s %s", path, e)
        return DEFAULT_SOUL_IDENTITY
