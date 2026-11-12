"""Tests for InMemoryTracer collector, spans, and timing."""

import threading
from datetime import datetime, timedelta, timezone

import pytest

from agent_orchestrator.tracing import (
    ExecutionEvent,
    InMemoryTracer,
    SpanStatus,
    Tracer,
)


def _event(trace_id: str, span_id: str = "s", n: int = 0) -> ExecutionEvent:
    return ExecutionEvent(
        event_type=f"evt.{n}",
        trace_id=trace_id,
        span_id=span_id,
        component="test",
        metadata={"n": n},
    )


class TestInMemoryTracerEvents:
    def test_satisfies_protocol(self):
        assert isinstance(InMemoryTracer(), Tracer)

    def test_collects_events_by_trace_and_preserves_order(self):
        tracer = InMemoryTracer()
        for i in range(5):
            tracer.record(_event("t1", n=i))
        events = tracer.get_events("t1")
        assert [e.metadata["n"] for e in events] == [0, 1, 2, 3, 4]

    def test_trace_isolation(self):
        tracer = InMemoryTracer()
        tracer.record(_event("t1", n=1))
        tracer.record(_event("t2", n=2))
        assert [e.metadata["n"] for e in tracer.get_events("t1")] == [1]
        assert [e.metadata["n"] for e in tracer.get_events("t2")] == [2]

    def test_get_events_unknown_trace_returns_empty(self):
        assert InMemoryTracer().get_events("missing") == []

    def test_list_traces(self):
        tracer = InMemoryTracer()
        tracer.record(_event("b"))
        tracer.record(_event("a"))
        tracer.start_span("c", "s1", "op", "comp")
        assert tracer.list_traces() == ["a", "b", "c"]

    def test_events_not_mutable_via_returned_list(self):
        tracer = InMemoryTracer()
        tracer.record(_event("t1", n=1))
        events = tracer.get_events("t1")
        events.clear()
        assert len(tracer.get_events("t1")) == 1


class TestInMemoryTracerSpans:
    def test_span_lifecycle_and_parent_child(self):
        tracer = InMemoryTracer()
        parent = tracer.start_span(
            "t1", "root", "execution", "agent_loop", attributes={"execution_id": "e1"}
        )
        child = tracer.start_span(
            "t1", "child", "tool", "tool_executor", parent_span_id="root"
        )
        assert parent.parent_span_id is None
        assert child.parent_span_id == "root"
        assert parent.status == SpanStatus.UNSET and parent.end_time is None

        ended = tracer.end_span("child", "t1", status=SpanStatus.OK)
        assert ended.end_time is not None
        assert ended.status == SpanStatus.OK

        trace = tracer.get_trace("t1")
        assert trace is not None and len(trace.spans) == 2
        by_id = {s.span_id: s for s in trace.spans}
        assert by_id["child"].parent_span_id == "root"
        assert by_id["child"].status == SpanStatus.OK

    def test_multiple_spans_one_trace(self):
        tracer = InMemoryTracer()
        for i in range(3):
            tracer.start_span("t1", f"s{i}", f"op{i}", "c")
            tracer.end_span(f"s{i}", "t1")
        trace = tracer.get_trace("t1")
        assert trace is not None
        assert {s.span_id for s in trace.spans} == {"s0", "s1", "s2"}

    def test_trace_retrieval_unknown_returns_none(self):
        assert InMemoryTracer().get_trace("nope") is None

    def test_trace_data_isolated(self):
        tracer = InMemoryTracer()
        tracer.start_span("t1", "s1", "a", "c")
        tracer.start_span("t2", "s2", "b", "c")
        t1 = tracer.get_trace("t1")
        t2 = tracer.get_trace("t2")
        assert t1 and [s.span_id for s in t1.spans] == ["s1"]
        assert t2 and [s.span_id for s in t2.spans] == ["s2"]

    def test_get_trace_returns_copies(self):
        tracer = InMemoryTracer()
        tracer.start_span("t1", "s1", "a", "c")
        trace = tracer.get_trace("t1")
        assert trace is not None
        trace.spans.clear()
        again = tracer.get_trace("t1")
        assert again is not None and len(again.spans) == 1

    def test_end_unknown_span_raises(self):
        tracer = InMemoryTracer()
        with pytest.raises(ValueError):
            tracer.end_span("missing", "t1")
        tracer.start_span("t1", "s1", "a", "c")
        with pytest.raises(ValueError):
            tracer.end_span("s1", "other-trace")

    def test_duplicate_end_raises(self):
        tracer = InMemoryTracer()
        tracer.start_span("t1", "s1", "a", "c")
        tracer.end_span("s1", "t1")
        with pytest.raises(ValueError):
            tracer.end_span("s1", "t1")

    def test_duplicate_start_raises(self):
        tracer = InMemoryTracer()
        tracer.start_span("t1", "s1", "a", "c")
        with pytest.raises(ValueError):
            tracer.start_span("t1", "s1", "a", "c")


class TestInMemoryTracerTiming:
    def test_timestamps_are_utc_and_duration_accurate(self):
        tracer = InMemoryTracer()
        span = tracer.start_span("t1", "s1", "op", "c")
        ended = tracer.end_span("s1", "t1")
        assert span.start_time.tzinfo is not None
        assert span.start_time.utcoffset() == timedelta(0)
        assert ended.end_time is not None
        assert ended.end_time.tzinfo is not None
        duration = InMemoryTracer.span_duration(ended)
        assert duration is not None
        assert duration == ended.end_time - ended.start_time
        assert duration >= timedelta(0)

    def test_duration_none_when_not_ended(self):
        tracer = InMemoryTracer()
        span = tracer.start_span("t1", "s1", "op", "c")
        assert InMemoryTracer.span_duration(span) is None


class TestInMemoryTracerConcurrency:
    def test_concurrent_record_and_span_ops(self):
        tracer = InMemoryTracer()
        errors: list[Exception] = []

        def record_worker(trace_id: str):
            try:
                for i in range(50):
                    tracer.record(_event(trace_id, n=i))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        def span_worker(trace_id: str):
            try:
                for i in range(20):
                    sid = f"{trace_id}-{i}"
                    tracer.start_span(trace_id, sid, "op", "c")
                    tracer.end_span(sid, trace_id)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = []
        for t in ["t1", "t2", "t3", "t4"]:
            threads.append(threading.Thread(target=record_worker, args=(t,)))
            threads.append(threading.Thread(target=span_worker, args=(t,)))
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == []
        for t in ["t1", "t2", "t3", "t4"]:
            assert len(tracer.get_events(t)) == 50
            trace = tracer.get_trace(t)
            assert trace is not None and len(trace.spans) == 20
            assert all(s.end_time is not None for s in trace.spans)