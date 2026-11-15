
"""Tracing helpers for AgentLoop."""

from __future__ import annotations

import uuid

from agent_orchestrator.core.models import ExecutionResult, ExecutionStatus
from agent_orchestrator.tracing import ExecutionEvent, SpanStatus


class TracingHelpersMixin:
    """Provides safe, best-effort tracing helpers."""

    def _start_span(
        self,
        trace_id,
        name,
        component,
        parent_span_id,
        attributes=None,
    ):
        span_id = str(uuid.uuid4())

        try:
            start = getattr(self._tracer, "start_span", None)

            if callable(start):
                start(
                    trace_id,
                    span_id,
                    name,
                    component,
                    parent_span_id,
                    attributes or {},
                )

            return span_id

        except Exception:
            return span_id

    def _end_span(self, span_id, trace_id, status):
        if not span_id or not trace_id:
            return

        try:
            end = getattr(self._tracer, "end_span", None)

            if callable(end):
                end(span_id, trace_id, status)

        except Exception:
            return

    def _record_event(
        self,
        event_type,
        trace_id,
        span_id,
        component,
        metadata=None,
    ):
        if not span_id or not trace_id:
            return

        try:
            self._tracer.record(
                ExecutionEvent(
                    event_type=event_type,
                    trace_id=trace_id,
                    span_id=span_id,
                    component=component,
                    metadata=metadata or {},
                )
            )

        except Exception:
            return

    def _record_precomputed_tool_event(
        self,
        trace_id,
        parent_span_id,
        state,
        iteration,
        tool_call,
        result,
    ):
        """Record events for malformed or duplicate tool calls."""

        error = result.error or ""

        if "Duplicate tool_call id" in error:
            event_type = "tool.call.duplicate"
        else:
            event_type = "tool.call.invalid"

        meta = {
            "execution_id": state.execution_id,
            "iteration": iteration,
            "tool_call_id": result.tool_call_id,
            "tool_name": result.tool_name,
            "success": False,
        }

        self._record_event(
            event_type,
            trace_id,
            parent_span_id,
            "agent_loop",
            meta,
        )

    def _emit_completion_events(self, trace_id, root_span, result):
        meta = {
            "execution_id": trace_id,
            "iterations": result.iterations,
            "tool_calls_count": result.tool_calls_count,
        }

        if result.status == ExecutionStatus.COMPLETED:
            self._record_event(
                "agent.execution.final_answer",
                trace_id,
                root_span,
                "agent_loop",
                meta,
            )
            self._record_event(
                "agent.execution.completed",
                trace_id,
                root_span,
                "agent_loop",
                meta,
            )

        elif result.status == ExecutionStatus.MAX_ITERATIONS_REACHED:
            self._record_event(
                "agent.execution.max_iterations",
                trace_id,
                root_span,
                "agent_loop",
                meta,
            )
            self._record_event(
                "agent.execution.completed",
                trace_id,
                root_span,
                "agent_loop",
                meta,
            )

        else:
            self._record_event(
                "agent.execution.failed",
                trace_id,
                root_span,
                "agent_loop",
                meta,
            )

    @staticmethod
    def _status_for_result(result: ExecutionResult) -> SpanStatus:
        if result.status in (
            ExecutionStatus.COMPLETED,
            ExecutionStatus.MAX_ITERATIONS_REACHED,
        ):
            # Defined terminal outcomes are treated as successfully closed spans.
            return SpanStatus.OK

        return SpanStatus.ERROR
