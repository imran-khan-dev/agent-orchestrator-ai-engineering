"""Provider request and response models.

These models normalize provider interactions without depending on any specific
LLM provider implementation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_orchestrator.tools.models import ToolCall, ToolDefinition


class ProviderRequest(BaseModel):
    """Normalized request sent to an LLM provider."""

    model_config = ConfigDict(extra="forbid")

    messages: list[dict[str, Any]] = Field(
        ..., description="Conversation messages in provider-agnostic dictionary form."
    )
    tools: list[ToolDefinition] | None = Field(
        default=None, description="Optional tool definitions available to the provider."
    )
    tool_choice: Any | None = Field(
        default=None, description="Optional provider-agnostic tool choice hint."
    )


class ProviderResponse(BaseModel):
    """Normalized response returned by an LLM provider."""

    model_config = ConfigDict(extra="forbid")

    content: str | None = Field(default=None, description="Text content from the provider.")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Tool calls requested by the provider."
    )
    finish_reason: str | None = Field(
        default=None, description="Reason the provider finished generating."
    )