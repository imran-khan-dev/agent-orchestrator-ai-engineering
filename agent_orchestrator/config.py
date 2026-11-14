"""Configuration model for the Agent Orchestrator."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_orchestrator.retry import RetryPolicy


class OrchestratorConfig(BaseModel):
    """Validated execution settings for the orchestrator.

    All numeric limits are strictly positive. Invalid values raise a clear
    Pydantic validation error.
    """

    model_config = ConfigDict(extra="forbid")  # what is this

    max_iterations: int = Field(default=10, gt=0, description="Maximum agent loop iterations.")
    
    tool_timeout: float = Field(
        default=30.0, gt=0, description="Timeout in seconds for a single tool execution."
    )

    max_tool_calls: int = Field(default=20, gt=0, description="Maximum number of tool calls per run.")
    
    enable_tracing: bool = Field(
        default=True, description="Whether tracing/observability hooks are enabled."
    )
   
    retry_policy: RetryPolicy = Field(
        default_factory=RetryPolicy,
        description="Default retry policy (one attempt, no retries unless configured).",
    )
    provider_retry_policy: RetryPolicy | None = Field(
        default=None,
        description="Optional provider-specific retry policy override; defaults to retry_policy.",
    )
    tool_retry_policy: RetryPolicy | None = Field(
        default=None,
        description="Optional tool-specific retry policy override; defaults to retry_policy.",
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