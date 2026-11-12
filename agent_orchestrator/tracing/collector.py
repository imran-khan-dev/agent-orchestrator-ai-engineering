"""In-memory tracing implementation.

Provides a thread-safe :class:`InMemoryTracer` that records execution events
and manages trace/span lifecycles using the tracing contracts. No coupling to
AgentLoop or ToolExecutor.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from agent_orchestrator.tracing.models import ExecutionEvent, Span, SpanStatus, Trace


class InMemoryTracer:
    """Thread-safe in-memory tracer satisfying the :class:`Tracer` protocol.

    Collects execution events and spans by trace ID. Internal collections are
    never exposed directly; retrieval methods return copies.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: dict[str, list[ExecutionEvent]] = {}
        self._spans: dict[str, dict[str, Span]] = {}

    def record(self, event: ExecutionEvent) -> None:
        """Record a single execution event for its trace."""
        with self._lock:
            self._events.setdefault(event.trace_id, []).append(event)

    def get_events(self, trace_id: str) -> list[ExecutionEvent]:
        """Return events for a trace in recorded order.

        Returns an empty list for unknown trace IDs.
        """
        with self._lock:
            return list(self._events.get(trace_id, []))

    def list_traces(self) -> list[str]:
        """Return trace IDs known to this tracer."""
        with self._lock:
            ids = set(self._events) | set(self._spans)
            return sorted(ids)

    def start_span(
        self,
        trace_id: str,
        span_id: str,
        name: str,
        component: str,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start and register a new span.

        Raises:
            ValueError: If a span with the same ID already exists in the trace.
        """
        start_time = datetime.now(timezone.utc)
        with self._lock:
            trace_spans = self._spans.setdefault(trace_id, {})
            if span_id in trace_spans:
                raise ValueError(
                    f"Span '{span_id}' already exists in trace '{trace_id}'."
                )
            span = Span(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                name=name,
                component=component,
                start_time=start_time,
                end_time=None,
                status=SpanStatus.UNSET,
                attributes=dict(attributes or {}),
            )
            trace_spans[span_id] = span
            return span.model_copy(deep=True)

    def end_span(
        self, span_id: str, trace_id: str, status: SpanStatus = SpanStatus.OK
    ) -> Span:
        """End an open span and record its status.

        Raises:
            ValueError: If the trace or span is unknown, or the span already ended.
        """
        end_time = datetime.now(timezone.utc)
        with self._lock:
            trace_spans = self._spans.get(trace_id)
            if trace_spans is None or span_id not in trace_spans:
                raise ValueError(
                    f"Unknown span '{span_id}' in trace '{trace_id}'."
                )
            span = trace_spans[span_id]
            if span.end_time is not None:
                raise ValueError(
                    f"Span '{span_id}' in trace '{trace_id}' has already ended."
                )
            ended = span.model_copy(
                update={"end_time": end_time, "status": status}, deep=True
            )
            trace_spans[span_id] = ended
            return ended.model_copy(deep=True)

    def get_trace(self, trace_id: str) -> Trace | None:
        """Return the complete trace for an ID, or ``None`` if unknown."""
        with self._lock:
            if trace_id not in self._events and trace_id not in self._spans:
                return None
            spans = [
                s.model_copy(deep=True)
                for s in self._spans.get(trace_id, {}).values()
            ]
            return Trace(trace_id=trace_id, spans=spans)

    @staticmethod
    def span_duration(span: Span) -> timedelta | None:
        """Calculate span duration from start/end timestamps.

        Returns ``None`` if the span has not ended.
        """
        if span.end_time is None:
            return None
        return span.end_time - span.start_time