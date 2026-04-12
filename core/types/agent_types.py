from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UsageStats:
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
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass(frozen=True)
class OutputSchemaConfig:
    name: str
    schema: dict[str, Any]
    strict: bool = False


@dataclass(frozen=True)
class ModelPricing:
    input_cost_per_mill_tokens: float = 0.0
    output_cost_per_mill_tokens: float = 0.0
    cache_creation_input_cost_per_mill_tokens: float = 0.0
    cache_read_input_cost_per_mill_tokens: float = 0.0

    def estimate_cost_usd(self, usage: UsageStats) -> float:
        return (
            (usage.input_tokens / 1e6) * self.input_cost_per_mill_tokens
            + ((usage.output_tokens / 1e6) * self.output_cost_per_mill_tokens)
            + (
                (usage.cache_creation_input_tokens / 1e6)
                * self.cache_creation_input_cost_per_mill_tokens
            )
            + (
                (usage.cache_read_input_tokens / 1e6)
                * self.cache_read_input_cost_per_mill_tokens
            )
        )


@dataclass(frozen=True)
class ModelConfig:
    model: str
    # base_url: str = "None"
    api_key: str = "local-token"
    temperature: int = 0.0
    timeout_seconds: float = 120.0
    pricing: ModelPricing = field(default_factory=ModelPricing)


@dataclass(frozen=True)
class BudgetConfig:
    max_total_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_reasoning_tokens: int | None = None
    max_total_cost_usd: int | None = None
    max_tool_calls: int | None = None
    max_model_calls: int | None = None
    max_session_turns: int | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AssistantTurn:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    raw_message: dict[str, Any] = field(default_factory=dict)
    usage: UsageStats = field(default_factory=UsageStats)


@dataclass(frozen=True)
class StreamEvent:
    type: str
    delta: str = ""
    tool_call_index: int | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_delta: str = ""
    finish_reason: str = None
    usage: UsageStats = field(default_factory=UsageStats)
    raw_event: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "delta": self.delta,
            "tool_call_index": self.tool_call_index,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments_delta": self.arguments_delta,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict(),
            "raw_event": dict(self.raw_event),
        }


@dataclass(frozen=True)
class AgentPermissions:
    allow_file_write: bool = False
    allow_shell_commands: bool = False
    allow_destructive_shell_commands: bool = False


@dataclass
class ToolCallParameters:
    tool_call_id: str
    tool_name: str
    tool_input: str


@dataclass
class ToolImplOutput:
    """Output from an LLM tool implementation

    Attributes:
        tool_output: The main string that will be shown to the model
        tool_result_message: A description of what the tool did, logging purposes
        auxiliary_data: Additional data the tool wants to pass along for logging only
    """

    tool_output: str
    tool_result_message: str = ""
    axiliary_data: dict = field(default_factory=dict)


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
