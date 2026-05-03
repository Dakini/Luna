from pathlib import Path
from typing import Any, Tuple

from pydantic import BaseModel, ConfigDict, Field


class UsageStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def __add__(self, other: "UsageStats") -> "UsageStats":
        return UsageStats(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens
            + other.cache_read_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class OutputSchemaConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    output_schema: dict[str, Any]
    strict: bool = False


class ModelPricing(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_cost_per_mill_tokens: float = 0.0
    output_cost_per_mill_tokens: float = 0.0
    cache_creation_input_cost_per_mill_tokens: float = 0.0
    cache_read_input_cost_per_mill_tokens: float = 0.0

    def estimate_cost_usd(self, usage: UsageStats) -> float:
        return (
            (usage.input_tokens / 1e6) * self.input_cost_per_mill_tokens
            + (usage.output_tokens / 1e6) * self.output_cost_per_mill_tokens
            + (usage.cache_creation_input_tokens / 1e6)
            * self.cache_creation_input_cost_per_mill_tokens
            + (usage.cache_read_input_tokens / 1e6)
            * self.cache_read_input_cost_per_mill_tokens
        )


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    api_key: str = "local-token"
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    pricing: ModelPricing = Field(default_factory=ModelPricing)


class BudgetConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_total_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_reasoning_tokens: int | None = None
    max_total_cost_usd: float | None = None
    max_tool_calls: int | None = None
    max_model_calls: int | None = None
    max_session_turns: int | None = None


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any]


class AssistantTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    tool_calls: Tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    raw_message: dict[str, Any] = Field(default_factory=dict)
    usage: UsageStats = Field(default_factory=UsageStats)


class StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    delta: str = ""
    tool_call_index: int | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_delta: str = ""
    finish_reason: str | None = None
    usage: UsageStats = Field(default_factory=UsageStats)
    raw_event: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class AgentPermissions(BaseModel):
    model_config = ConfigDict(frozen=True)

    allow_file_write: bool = False
    allow_shell_commands: bool = False
    allow_destructive_shell_commands: bool = False


class AgentRuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    cwd: Path
    max_turns: int = 200
    command_timeout_seconds: float = 30.0
    max_output_chars: int = 12000
    stream_model_responses: bool = False
    auto_snip_threshold_tokens: int | None = None
    auto_compact_threshold_tokens: int | None = None
    compact_preserve_message: int = 4
    permissions: AgentPermissions = Field(default_factory=AgentPermissions)
    additiona_working_directories: Tuple[Path, ...] = ()
    disable_md_discovery: bool = False
    budget_config: BudgetConfig = Field(default_factory=BudgetConfig)
    output_schema: OutputSchemaConfig | None = None
    session_dir: Path = Field(
        default_factory=lambda: (Path(".port_sessions") / "agent").resolve()
    )
    scratch_pad_dir: Path = Field(
        default_factory=lambda: (Path(".port_sessions") / "scratch_pad").resolve()
    )


class ToolCallParameters(BaseModel):
    tool_call_id: str
    tool_name: str
    tool_input: Any


class ToolImplOutput(BaseModel):
    tool_output: str
    tool_result_message: str = ""
    auxiliary_data: dict[str, Any] = Field(default_factory=dict)


if __name__ == "__main__":
    usage = UsageStats(
        input_tokens=100,
        output_tokens=5,
        cache_creation_input_tokens=1e6,
        cache_read_input_tokens=1e9,
    )

    s = ModelPricing(2, 1, 2, 3)
    print(s.estimate_cost_usd(usage))
    assert usage.total_tokens == 105 + 1e6 + 1e9
