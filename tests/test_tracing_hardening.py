"""Regression tests for tracing hardening (M5.4)."""

import asyncio
import concurrent.futures

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.core.loop import AgentLoop
from agent_orchestrator.core.models import ExecutionStatus
from agent_orchestrator.providers.models import ProviderResponse
from agent_orchestrator.tools.executor import ToolExecutor
from agent_orchestrator.tools.models import ToolCall, ToolDefinition
from agent_orchestrator.tools.registry import ToolRegistry
from agent_orchestrator.tracing import InMemoryTracer, NoOpTracer, SpanStatus


def _tool_def(name="add"):
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


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, request):
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class ResponseLike:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class TestMalformedDuplicateToolCallTracing:
    def test_malformed_and_duplicate_emit_events_without_fake_tool_spans(self):
        reg = ToolRegistry()
        reg.register(_tool_def("add"), lambda a, b: a + b)
        tracer = InMemoryTracer()
        loop = AgentLoop(tool_executor=ToolExecutor(reg), tracer=tracer)
        provider = ScriptedProvider([
            ResponseLike(tool_calls=[
                {"name": "add", "arguments": {"a": 1, "b": 2}},  # missing id -> invalid
                ToolCall(id="dup", name="add", arguments={"a": 2, "b": 2}),
                ToolCall(id="dup", name="add", arguments={"a": 3, "b": 3}),
            ]),
            ProviderResponse(content="recovered"),
        ])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        events = tracer.get_events(result.execution_id)
        types = [e.event_type for e in events]
        assert "tool.call.invalid" in types
        assert "tool.call.duplicate" in types
        spans = tracer.get_trace(result.execution_id).spans
        # executed tool call gets a tool.execute span; malformed/duplicate do not
        tool_spans = [s for s in spans if s.name == "tool.execute"]
        assert len(tool_spans) == 1
        assert tool_spans[0].attributes["tool_call_id"] == "dup"
        # lifecycle events attached to iteration span
        iter_span = next(s for s in spans if s.name == "iteration")
        special = [e for e in events if e.event_type in ("tool.call.invalid", "tool.call.duplicate")]
        assert special and all(e.span_id == iter_span.span_id for e in special)
        assert all(e.metadata["success"] is False for e in special)


class TestTraceIsolationAndReuse:
    def test_multiple_executions_same_loop_have_isolated_traces(self):
        tracer = InMemoryTracer()
        loop = AgentLoop(tracer=tracer)
        _, r1 = loop.run(messages=[], provider=ScriptedProvider([ProviderResponse(content="a")]))
        _, r2 = loop.run(messages=[], provider=ScriptedProvider([ProviderResponse(content="b")]))
        assert r1.execution_id != r2.execution_id
        assert set(tracer.list_traces()) == {r1.execution_id, r2.execution_id}
        t1 = tracer.get_trace(r1.execution_id)
        t2 = tracer.get_trace(r2.execution_id)
        assert t1 and t2
        assert all(s.trace_id == r1.execution_id for s in t1.spans)
        assert all(s.trace_id == r2.execution_id for s in t2.spans)
        assert all(e.trace_id == r1.execution_id for e in tracer.get_events(r1.execution_id))
        assert all(e.trace_id == r2.execution_id for e in tracer.get_events(r2.execution_id))

    def test_concurrent_executions_isolated(self):
        tracer = InMemoryTracer()

        def run_one(i):
            loop = AgentLoop(tracer=tracer)
            _, res = loop.run(messages=[], provider=ScriptedProvider([ProviderResponse(content=str(i))]))
            return res

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(run_one, range(6)))
        ids = [r.execution_id for r in results]
        assert len(set(ids)) == 6
        for eid in ids:
            trace = tracer.get_trace(eid)
            assert trace is not None
            assert all(s.trace_id == eid for s in trace.spans)
            assert all(s.end_time is not None for s in trace.spans)
            assert all(e.trace_id == eid for e in tracer.get_events(eid))


class TestTracingDisabledAndResilience:
    def test_tracing_disabled_records_nothing(self):
        cfg = OrchestratorConfig(enable_tracing=False)
        loop = AgentLoop(config=cfg)
        provider = ScriptedProvider([ProviderResponse(content="ok")])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        assert isinstance(loop.tracer, NoOpTracer)
        # NoOpTracer has no storage; protocol behavior is no-op record
        assert hasattr(loop.tracer, "record")

    def test_tracer_failures_during_lifecycle_do_not_break(self):
        class FlakyTracer:
            def __init__(self):
                self.phase = 0
            def record(self, event):
                self.phase += 1
                if self.phase in (1, 3):
                    raise RuntimeError("record boom")
            def start_span(self, *a, **k):
                self.phase += 1
                if self.phase == 2:
                    raise RuntimeError("start boom")
            def end_span(self, *a, **k):
                self.phase += 1
                if self.phase % 5 == 0:
                    raise RuntimeError("end boom")

        reg = ToolRegistry(); reg.register(_tool_def("add"), lambda a,b:a+b)
        tracer = FlakyTracer()
        executor = ToolExecutor(reg, tracer=tracer)
        loop = AgentLoop(tool_executor=executor, tracer=tracer)
        provider = ScriptedProvider([
            ProviderResponse(tool_calls=[ToolCall(id="c1", name="add", arguments={"a":2,"b":2})]),
            ProviderResponse(content="still ok"),
        ])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "still ok"
        assert result.tool_calls_count == 1

    def test_provider_exception_closes_all_spans(self):
        tracer = InMemoryTracer()
        loop = AgentLoop(tracer=tracer)
        provider = ScriptedProvider([RuntimeError("boom")])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.FAILED
        trace = tracer.get_trace(result.execution_id)
        assert trace is not None
        assert trace.spans  # has spans
        assert all(s.end_time is not None for s in trace.spans)
        statuses = {s.name: s.status for s in trace.spans}
        assert statuses["agent.execution"] == SpanStatus.ERROR
        assert statuses["provider.request"] == SpanStatus.ERROR
        assert statuses["iteration"] == SpanStatus.ERROR
