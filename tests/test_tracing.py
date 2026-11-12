"""Unit tests for tracing contracts and no-op tracer."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agent_orchestrator.tracing import (
    ExecutionEvent,
    NoOpTracer,
    Span,
    SpanStatus,
    Trace,
    Tracer,
)


class TestExecutionEvent:
    def test_valid_event_fields(self):
        ts = datetime(2026, 11, 12, tzinfo=timezone.utc)
        ev = ExecutionEvent(
            event_type="tool.started",
            timestamp=ts,
            trace_id="trace-1",
            span_id="span-1",
            parent_span_id="span-0",
            component="tool_executor",
            metadata={
                "execution_id": "exec-1",
                "iteration": 2,
                "tool_call_id": "call-1",
                "tool_name": "add",
            },
        )
        assert ev.event_type == "tool.started"
        assert ev.timestamp == ts
        assert ev.trace_id == "trace-1"
        assert ev.span_id == "span-1"
        assert ev.parent_span_id == "span-0"
        assert ev.component == "tool_executor"
        assert ev.metadata["tool_name"] == "add"

    def test_defaults(self):
        ev = ExecutionEvent(
            event_type="run.started",
            trace_id="t",
            span_id="s",
            component="agent_loop",
        )
        assert ev.parent_span_id is None
        assert isinstance(ev.timestamp, datetime)
        assert ev.metadata == {}

    def test_metadata_is_flexible_but_isolated(self):
        a = ExecutionEvent(event_type="e", trace_id="t", span_id="s1", component="c")
        b = ExecutionEvent(event_type="e", trace_id="t", span_id="s2", component="c")
        a.metadata["x"] = 1
        assert b.metadata == {}

    def test_requires_non_empty_identifiers(self):
        with pytest.raises(ValidationError):
            ExecutionEvent(event_type="", trace_id="t", span_id="s", component="c")
        with pytest.raises(ValidationError):
            ExecutionEvent(event_type="e", trace_id="", span_id="s", component="c")
        with pytest.raises(ValidationError):
            ExecutionEvent(event_type="e", trace_id="t", span_id="", component="c")
        with pytest.raises(ValidationError):
            ExecutionEvent(event_type="e", trace_id="t", span_id="s", component="")


class TestSpanAndTrace:
    def test_span_parent_child_relationship(self):
        start = datetime(2026, 11, 12, tzinfo=timezone.utc)
        end = datetime(2026, 11, 12, 0, 0, 1, tzinfo=timezone.utc)
        parent = Span(
            trace_id="t1",
            span_id="root",
            name="execution",
            component="agent_loop",
            start_time=start,
            end_time=end,
            status=SpanStatus.OK,
        )
        child = Span(
            trace_id="t1",
            span_id="child",
            parent_span_id=parent.span_id,
            name="tool_call",
            component="tool_executor",
            start_time=start,
            status=SpanStatus.UNSET,
            attributes={"tool_name": "add", "tool_call_id": "c1"},
        )
        assert child.parent_span_id == "root"
        assert parent.parent_span_id is None
        assert parent.status == SpanStatus.OK
        assert child.attributes["tool_name"] == "add"

    def test_span_defaults(self):
        span = Span(
            trace_id="t",
            span_id="s",
            name="op",
            component="core",
            start_time=datetime.now(timezone.utc),
        )
        assert span.parent_span_id is None
        assert span.end_time is None
        assert span.status == SpanStatus.UNSET
        assert span.attributes == {}

    def test_trace_holds_spans(self):
        start = datetime.now(timezone.utc)
        spans = [
            Span(trace_id="t1", span_id="s1", name="a", component="c", start_time=start),
            Span(
                trace_id="t1",
                span_id="s2",
                parent_span_id="s1",
                name="b",
                component="c",
                start_time=start,
            ),
        ]
        trace = Trace(trace_id="t1", spans=spans)
        assert trace.trace_id == "t1"
        assert len(trace.spans) == 2
        assert trace.spans[1].parent_span_id == "s1"

    def test_trace_defaults_empty(self):
        trace = Trace(trace_id="t")
        assert trace.spans == []


class TestNoOpTracer:
    def test_satisfies_protocol(self):
        tracer = NoOpTracer()
        assert isinstance(tracer, Tracer)

    def test_record_discards_without_error(self):
        tracer = NoOpTracer()
        ev = ExecutionEvent(
            event_type="x", trace_id="t", span_id="s", component="c", metadata={"k": "v"}
        )
        assert tracer.record(ev) is None
        assert tracer.record(ev) is None