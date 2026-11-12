"""Configuration model for the Agent Orchestrator."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OrchestratorConfig(BaseModel):
    """Validated execution settings for the orchestrator.

    All numeric limits are strictly positive. Invalid values raise a clear
    Pydantic validation error.
    """

    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=10, gt=0, description="Maximum agent loop iterations.")
    tool_timeout: float = Field(
        default=30.0, gt=0, description="Timeout in seconds for a single tool execution."
    )
    max_tool_calls: int = Field(default=20, gt=0, description="Maximum number of tool calls per run.")
    enable_tracing: bool = Field(
        default=True, description="Whether tracing/observability hooks are enabled."
    )

    @field_validator("max_iterations", "max_tool_calls")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be a positive integer")
        return value

    @field_validator("tool_timeout")
    @classmethod
    def _validate_positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be a positive number")
        return value