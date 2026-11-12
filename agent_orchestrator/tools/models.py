"""Tool contracts for the Agent Orchestrator.

These models are intentionally independent of any LLM provider. They describe
tools, calls made against them, and the results returned — all using
JSON-compatible data types.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolDefinition(BaseModel):
    """Normalized description of a callable tool."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Unique tool name.")
    description: str = Field(..., min_length=1, description="Human-readable tool description.")
    parameters_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing the tool's parameters.",
    )


class ToolCall(BaseModel):
    """A single invocation request for a tool."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Unique call identifier.")
    name: str = Field(..., min_length=1, description="Tool name to invoke.")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="JSON-compatible arguments for the tool."
    )


class ToolResult(BaseModel):
    """The outcome of executing a single tool call."""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str = Field(..., min_length=1, description="Identifier of the originating ToolCall.")
    tool_name: str = Field(..., min_length=1, description="Name of the tool that was executed.")
    output: Any = Field(default=None, description="JSON-compatible output produced by the tool.")
    success: bool = Field(..., description="Whether the tool call succeeded.")
    error: str | None = Field(default=None, description="Error message when success is False.")