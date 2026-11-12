"""Tests for ToolExecutor retry behavior (M6.3)."""

import asyncio
import time
import uuid

import pytest

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.retry import RetryPolicy
from agent_orchestrator.tools.executor import ToolExecutor
from agent_orchestrator.tools.models import ToolCall, ToolDefinition
from agent_orchestrator.tools.registry import ToolRegistry
from agent_orchestrator.tracing import ExecutionEvent, SpanStatus


def _def(name: str = "add") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} tool",
        parameters_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    )


def make_executor(
    registry: ToolRegistry | None = None,
    config: OrchestratorConfig | None = None,
) -> tuple[ToolExecutor, ToolRegistry]:
    reg = registry or ToolRegistry()
    cfg = config or OrchestratorConfig()
    return ToolExecutor(reg, config=cfg), reg


class RecordingTracer:
    def __init__(self):
        self.events = []
        self.spans_started = []
        self.spans_ended = []
        self._span_ids = {}

    def record(self, event):
        self.events.append(event)

    def start_span(self, trace_id, span_id, name, component, parent_span_id=None, attributes=None):
        self.spans_started.append((trace_id, span_id, name, component, parent_span_id, attributes))
        class S:
            pass
        s = S()
        s.span_id = span_id
        self._span_ids[span_id] = s
        return s

    def end_span(self, span_id, trace_id, status=SpanStatus.OK):
        self.spans_ended.append((span_id, trace_id, status))
        class S:
            pass
        s = S()
        s.span_id = span_id
        s.status = status
        return s


class TestToolRetrySync:
    def test_sync_handler_succeeds_after_retry(self, monkeypatch):
        calls = {"n": 0}
        sleeps = []

        def boom_then_ok(a: int, b: int) -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return a + b

        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        executor, reg = make_executor()
        reg.register(_def("add"), boom_then_ok)
        policy = RetryPolicy(max_attempts=2, initial_delay=0.01, max_delay=0.01)
        result = executor.execute(
            ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3}),
            tool_retry_policy=policy,
        )
        assert result.success is True
        assert result.output == 5
        assert calls["n"] == 2
        assert len(sleeps) == 1
        assert sleeps[0] == 0.01

    def test_retry_exhaustion_sync(self, monkeypatch):
        calls = {"n": 0}
        sleeps = []

        def always_boom(a: int, b: int) -> int:
            calls["n"] += 1
            raise ValueError("boom")

        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        executor, reg = make_executor()
        reg.register(_def("add"), always_boom)
        policy = RetryPolicy(max_attempts=3, initial_delay=0.1, max_delay=0.5)
        result = executor.execute(
            ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}),
            tool_retry_policy=policy,
        )
        assert result.success is False
        assert "ValueError" in result.error
        assert "boom" in result.error
        assert calls["n"] == 3
        assert len(sleeps) == 2
        assert sleeps == [0.1, 0.1]

    def test_non_retryable_exceptions_sync(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        executor, reg = make_executor()
        reg.register(_def("add"), lambda a, b: a + b)
        policy = RetryPolicy(max_attempts=5, initial_delay=1.0, max_delay=1.0)
        result = executor.execute(
            ToolCall(id="c1", name="add", arguments={"a": "no", "b": 2}),
            tool_retry_policy=policy,
        )
        assert result.success is False
        assert "ToolValidationError" in result.error
        assert sleeps == []

    def test_unknown_tool_not_retried(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
        executor, _ = make_executor()
        policy = RetryPolicy(max_attempts=4, initial_delay=0.5, max_delay=0.5)
        result = executor.execute(
            ToolCall(id="c1", name="missing", arguments={}),
            tool_retry_policy=policy,
        )
        assert result.success is False
        assert "ToolNotFoundError" in result.error
        assert sleeps == []

    def test_custom_retry_on_configuration(self, monkeypatch):
        calls = {"n": 0}
        sleeps = []

        class RetryableErr(Exception):
            pass

        class NonRetryableErr(Exception):
            pass

        def handler(a: int, b: int) -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RetryableErr("r")
            raise NonRetryableErr("nr")

        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        executor, reg = make_executor()
        reg.register(_def("add"), handler)
        policy = RetryPolicy(
            max_attempts=10,
            initial_delay=0.01,
            max_delay=0.01,
            retry_on=(RetryableErr,),
        )
        result = executor.execute(
            ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}),
            tool_retry_policy=policy,
        )
        assert result.success is False
        assert "NonRetryableErr" in result.error
        assert calls["n"] == 2
        assert len(sleeps) == 1


class TestToolRetryAsync:
    def test_async_handler_succeeds_after_retry(self, monkeypatch):
        calls = {"n": 0}
        sleeps = []

        async def boom_then_ok(a: int, b: int) -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return a + b

        async def fake_sleep(s):
            sleeps.append(s)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        executor, reg = make_executor()
        reg.register(_def("add"), boom_then_ok)
        policy = RetryPolicy(max_attempts=2, initial_delay=0.01, max_delay=0.01)
        result = asyncio.run(
            executor.execute_async(
                ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3}),
                tool_retry_policy=policy,
            )
        )
        assert result.success is True
        assert result.output == 5
        assert calls["n"] == 2
        assert len(sleeps) == 1

    def test_retry_exhaustion_async(self, monkeypatch):
        calls = {"n": 0}
        sleeps = []

        async def always_boom(a: int, b: int) -> int:
            calls["n"] += 1
            raise ValueError("boom")

        async def fake_sleep(s):
            sleeps.append(s)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        executor, reg = make_executor()
        reg.register(_def("add"), always_boom)
        policy = RetryPolicy(max_attempts=3, initial_delay=0.2, max_delay=0.2)
        result = asyncio.run(
            executor.execute_async(
                ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}),
                tool_retry_policy=policy,
            )
        )
        assert result.success is False
        assert calls["n"] == 3
        assert len(sleeps) == 2

    def test_cancellation_not_retried(self, monkeypatch):
        calls = {"n": 0}
        sleeps = []

        async def cancelled(a: int, b: int) -> int:
            calls["n"] += 1
            raise asyncio.CancelledError()

        async def fake_sleep(s):
            sleeps.append(s)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        executor, reg = make_executor()
        reg.register(_def("add"), cancelled)
        policy = RetryPolicy(max_attempts=5, initial_delay=1.0, max_delay=1.0)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                executor.execute_async(
                    ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}),
                    tool_retry_policy=policy,
                )
            )
        assert calls["n"] == 1
        assert sleeps == []


class TestToolRetryBehavior:
    def test_tool_call_id_and_args_preserved(self, monkeypatch):
        calls = {"n": 0}
        received = []

        def boom_then_ok(a: int, b: int) -> int:
            calls["n"] += 1
            received.append((a, b))
            if calls["n"] == 1:
                raise RuntimeError("x")
            return a + b

        monkeypatch.setattr(time, "sleep", lambda s: None)

        executor, reg = make_executor()
        reg.register(_def("add"), boom_then_ok)
        policy = RetryPolicy(max_attempts=2, initial_delay=0.0, max_delay=0.0)
        result = executor.execute(
            ToolCall(id="req-123", name="add", arguments={"a": 7, "b": 9}),
            tool_retry_policy=policy,
        )
        assert result.success is True
        assert result.tool_call_id == "req-123"
        assert result.tool_name == "add"
        assert received == [(7, 9), (7, 9)]

    def test_max_attempts_and_backoff_behavior(self, monkeypatch):
        calls = {"n": 0}
        sleeps = []

        def always_boom(a: int, b: int) -> int:
            calls["n"] += 1
            raise RuntimeError("x")

        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        executor, reg = make_executor()
        reg.register(_def("add"), always_boom)
        policy = RetryPolicy(
            max_attempts=4,
            initial_delay=1.0,
            max_delay=10.0,
            backoff_multiplier=2.0,
        )
        result = executor.execute(
            ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1}),
            tool_retry_policy=policy,
        )
        assert result.success is False
        assert calls["n"] == 4
        assert sleeps == [1.0, 2.0, 4.0]

    def test_no_sleep_after_final_attempt(self, monkeypatch):
        calls = {"n": 0}
        sleeps = []

        def always_boom(a: int, b: int) -> int:
            calls["n"] += 1
            raise RuntimeError("x")

        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        executor, reg = make_executor()
        reg.register(_def("add"), always_boom)
        policy = RetryPolicy(max_attempts=1, initial_delay=5.0, max_delay=5.0)
        result = executor.execute(
            ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1}),
            tool_retry_policy=policy,
        )
        assert result.success is False
        assert calls["n"] == 1
        assert sleeps == []

    def test_retry_tracing_events_and_span_lifecycle(self, monkeypatch):
        calls = {"n": 0}

        def boom_then_ok(a: int, b: int) -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("t")
            return a + b

        monkeypatch.setattr(time, "sleep", lambda s: None)

        tracer = RecordingTracer()
        executor, reg = make_executor()
        reg.register(_def("add"), boom_then_ok)
        executor.tracer = tracer
        policy = RetryPolicy(max_attempts=2, initial_delay=0.01, max_delay=0.01)
        result = executor.execute(
            ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}),
            tool_retry_policy=policy,
            trace_id="trace-1",
            parent_span_id="iter-1",
            execution_id="exec-1",
            iteration=1,
        )
        assert result.success is True
        types = [e.event_type for e in tracer.events]
        assert "tool.execution.started" in types
        assert "tool.attempt.started" in types
        assert "tool.attempt.failed" in types
        assert "tool.retry.scheduled" in types
        assert "tool.attempt.succeeded" in types
        assert "tool.execution.completed" in types
        assert len(tracer.spans_started) == 1
        assert len(tracer.spans_ended) == 1
        # retry metadata sanity
        retry_evt = next(e for e in tracer.events if e.event_type == "tool.retry.scheduled")
        assert retry_evt.metadata.get("retry_number") == 1
        assert retry_evt.metadata.get("attempt") == 1
        assert retry_evt.metadata.get("max_attempts") == 2
        assert "delay" in retry_evt.metadata
        assert "tool_call_id" in retry_evt.metadata and retry_evt.metadata["tool_call_id"] == "c1"
        assert retry_evt.metadata.get("execution_id") == "exec-1"
        assert retry_evt.metadata.get("iteration") == 1
        # span lifecycle status ok
        ended = tracer.spans_ended[0]
        assert ended[2] == SpanStatus.OK

    def test_policy_precedence_config_tool_retry(self, monkeypatch):
        calls = {"n": 0}
        sleeps = []

        def boom_then_ok(a: int, b: int) -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("t")
            return a + b

        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        cfg = OrchestratorConfig(tool_retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.01, max_delay=0.01))
        executor, reg = make_executor(config=cfg)
        reg.register(_def("add"), boom_then_ok)
        result = executor.execute(
            ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1}),
        )
        assert result.success is True
        assert calls["n"] == 2
        assert len(sleeps) == 1

    def test_policy_precedence_default_retry_policy(self, monkeypatch):
        calls = {"n": 0}
        sleeps = []

        def always_boom(a: int, b: int) -> int:
            calls["n"] += 1
            raise RuntimeError("t")

        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        cfg = OrchestratorConfig(retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.05, max_delay=0.05))
        executor, reg = make_executor(config=cfg)
        reg.register(_def("add"), always_boom)
        result = executor.execute(
            ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1}),
        )
        assert result.success is False
        assert calls["n"] == 3
        assert len(sleeps) == 2
