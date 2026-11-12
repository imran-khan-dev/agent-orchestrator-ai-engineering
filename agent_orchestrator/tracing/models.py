"""Tracing contracts for the Agent Orchestrator.

Provider-independent, framework-independent models and interfaces for
execution tracing. Instrumentation and collectors are added in later tasks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class SpanStatus(str, Enum):
    """Status of a trace span."""

    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


class ExecutionEvent(BaseModel):
    """A single tracing event emitted during execution.

    Metadata is flexible but validated; callers should avoid storing raw
    prompts, tool arguments, tool outputs, or secrets by default.
    """

    model_config = ConfigDict(extra="forbid")

    event_type: str = Field(..., min_length=1, description="Event type or name.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp (UTC recommended).",
    )
    trace_id: str = Field(..., min_length=1, description="Identifier of the trace.")
    span_id: str = Field(..., min_length=1, description="Identifier of the span.")
    parent_span_id: str | None = Field(
        default=None, description="Identifier of the parent span, if any."
    )
    component: str = Field(
        ..., min_length=1, description="Component or source emitting the event."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional execution metadata (e.g. execution_id, iteration, "
            "tool_call_id, tool_name)."
        ),
    )


class Span(BaseModel):
    """A unit of work within a trace, optionally nested under a parent span."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(..., min_length=1, description="Identifier of the trace.")
    span_id: str = Field(..., min_length=1, description="Identifier of the span.")
    parent_span_id: str | None = Field(
        default=None, description="Identifier of the parent span, if any."
    )
    name: str = Field(..., min_length=1, description="Span name.")
    component: str = Field(
        ..., min_length=1, description="Component or source owning the span."
    )
    start_time: datetime = Field(..., description="Span start timestamp.")
    end_time: datetime | None = Field(default=None, description="Span end timestamp.")
    status: SpanStatus = Field(default=SpanStatus.UNSET, description="Span status.")
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Optional span attributes/metadata."
    )


class Trace(BaseModel):
    """Collection of spans representing an execution trace."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(..., min_length=1, description="Identifier of the trace.")
    spans: list[Span] = Field(default_factory=list, description="Spans in the trace.")


@runtime_checkable
class Tracer(Protocol):
    """Provider-independent tracing interface.

    Implementations record execution events for in-memory or external tracing.
    """

    def record(self, event: ExecutionEvent) -> None:
        """Record a single execution event."""
        ...


class NoOpTracer:
    """Tracer implementation that discards all events."""

    def record(self, event: ExecutionEvent) -> None:
        """Discard the event."""
        return None