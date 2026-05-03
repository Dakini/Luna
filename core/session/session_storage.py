import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.types.agent_types import (
    AgentPermissions,
    AgentRuntimeConfig,
    BudgetConfig,
    ModelConfig,
    ModelPricing,
    OutputSchemaConfig,
    UsageStats,
)


class StoredSession(BaseModel):
    model_config = ConfigDict(frozen=True)
    session_id: str
    messages: tuple[str, ...]
    input_tokens: int
    output_tokens: int


DEFAULT_SESSION_DIR = Path(".port_sessions")
DEFAULT_AGENT_SESSION_DIR = DEFAULT_SESSION_DIR / "agent"


def save_session(session: StoredSession, directory: Path | None = None) -> Path:
    target_dir = directory or DEFAULT_SESSION_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{session.session_id}.json"
    path.write_text(session.model_dump_json(indent=2))
    return path


def load_session(session_id: str, directory: Path | None = None) -> StoredSession:
    target_dir = directory or DEFAULT_SESSION_DIR
    data = json.loads((target_dir / f"{session_id}.json").read_text())
    return StoredSession(
        session_id=data["session_id"],
        messages=tuple(data["messages"]),
        input_tokens=data["input_tokens"],
        output_tokens=data["output_tokens"],
    )


class StoredAgentSession(BaseModel):
    model_config = ConfigDict(frozen=True)
    session_id: str
    agent_model_config: dict[str, Any]
    runtime_config: dict[str, Any]
    system_prompt_parts: tuple[str]
    user_context: dict[str, Any]
    system_context: dict[str, Any]
    messages: tuple[dict[str, Any], ...]
    turns: int
    tool_calls: int
    usages: dict[str, Any]
    total_cost_usd: float
    file_history: tuple[dict[str, Any], ...]
    budget_state: dict[str, Any]
    scratch_pad_dir: str | None = None


def save_agent_session(session: StoredSession, directory: Path | None = None) -> Path:
    target_dir = directory or DEFAULT_AGENT_SESSION_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{session.session_id}.json"
    path.write_text(session.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_agent_session(
    session_id: str, directory: Path | None = None
) -> StoredAgentSession:
    target_dir = directory or DEFAULT_AGENT_SESSION_DIR
    data = json.loads((target_dir / f"{session_id}.json").read_text(encoding="utf-8"))
    return StoredAgentSession(
        session_id=data["session_id"],
        agent_model_config=dict(data["agent_model_config"]),
        runtime_config=dict(data["runtime_config"]),
        system_prompt_parts=tuple(data["system_prompt_parts"]),
        user_context=dict(data["user_context"]),
        system_context=dict(data["system_context"]),
        messages=tuple(
            message for message in data["messages"] if isinstance(message, dict)
        ),
        turns=int(data["turns"]),
        tool_calls=int(data["tool_calls"]),
        usages=dict(data.get("usages", {})),
        total_cost_usd=float(data.get("total_cost_usd", 0.0)),
        file_history=tuple(
            entry for entry in data.get("file_history", []) if isinstance(entry, dict)
        ),
        budget_state=dict(
            data.get("budget_state", {})
            if isinstance(data.get("budget_start"), dict)
            else {}
        ),
        scratch_pad_dir=(
            str(data["scratch_pad_dir"])
            if isinstance(data["scratch_pad_dir"], str)
            else None
        ),
    )
