"""Tests for AgentLoop and ToolExecutor tracing instrumentation (M5.3)."""

import asyncio

import pytest

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.core.loop import AgentLoop, IterationDecision, IterationOutcome
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


class AsyncScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)

    async def complete_async(self, request):
        await asyncio.sleep(0)
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def _setup():
    reg = ToolRegistry()
    reg.register(_tool_def("add"), lambda a, b: a + b)
    cfg = OrchestratorConfig(max_iterations=10, max_tool_calls=20)
    tracer = InMemoryTracer()
    executor = ToolExecutor(reg, config=cfg)
    loop = AgentLoop(config=cfg, tool_executor=executor, tracer=tracer)
    return loop, tracer, reg


def _span_map(tracer, trace_id):
    trace = tracer.get_trace(trace_id)
    assert trace is not None
    return {s.span_id: s for s in trace.spans}, trace.spans


class TestAgentLoopTracingSuccess:
    def test_sync_execution_trace_final_answer(self):
        loop, tracer, _ = _setup()
        provider = ScriptedProvider([ProviderResponse(content="hello")])
        state, result = loop.run(messages=[{"role": "user", "content": "hi"}], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED and result.final_response == "hello"
        trace_id = result.execution_id
        spans, _ = _span_map(tracer, trace_id)
        events = tracer.get_events(trace_id)
        assert all(e.trace_id == trace_id for e in events)
        assert all(s.trace_id == trace_id for s in spans.values())
        root = next(s for s in spans.values() if s.name == "agent.execution")
        assert root.status == SpanStatus.OK and root.end_time is not None
        iter_span = next(s for s in spans.values() if s.name == "iteration")
        assert iter_span.parent_span_id == root.span_id
        prov = next(s for s in spans.values() if s.name == "provider.request")
        assert prov.parent_span_id == iter_span.span_id and prov.status == SpanStatus.OK
        types = [e.event_type for e in events]
        assert "agent.execution.started" in types
        assert "agent.iteration.started" in types
        assert "provider.request.started" in types
        assert "provider.response.received" in types
        assert "agent.execution.final_answer" in types
        assert "agent.execution.completed" in types

    def test_sync_tool_execution_spans_and_events(self):
        loop, tracer, _ = _setup()
        provider = ScriptedProvider([
            ProviderResponse(tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]),
            ProviderResponse(content="sum is 5"),
        ])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED and result.tool_calls_count == 1
        spans, _ = _span_map(tracer, result.execution_id)
        root = next(s for s in spans.values() if s.name == "agent.execution")
        iter1 = [s for s in spans.values() if s.name == "iteration"][0]
        tool_span = next(s for s in spans.values() if s.name == "tool.execute")
        assert tool_span.parent_span_id == iter1.span_id
        assert tool_span.status == SpanStatus.OK
        assert tool_span.attributes["tool_name"] == "add"
        assert tool_span.attributes["tool_call_id"] == "c1"
        assert tool_span.attributes["execution_id"] == result.execution_id
        assert tool_span.attributes["iteration"] == 1
        assert InMemoryTracer.span_duration(tool_span) is not None
        events = tracer.get_events(result.execution_id)
        types = [e.event_type for e in events]
        assert "tool.execution.started" in types
        assert "tool.execution.completed" in types
        tool_ev = next(e for e in events if e.event_type == "tool.execution.completed")
        assert tool_ev.metadata["tool_name"] == "add"
        assert tool_ev.metadata["success"] is True
        assert tool_ev.metadata["iteration"] == 1

    def test_async_execution_trace(self):
        async def run():
            loop, tracer, _ = _setup()
            provider = AsyncScriptedProvider([
                ProviderResponse(tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})]),
                ProviderResponse(content="ok"),
            ])
            state, result = await loop.run_async(messages=[], provider=provider)
            return tracer, result

        tracer, result = asyncio.run(run())
        assert result.status == ExecutionStatus.COMPLETED
        spans, _ = _span_map(tracer, result.execution_id)
        assert any(s.name == "tool.execute" and s.status == SpanStatus.OK for s in spans.values())
        types = [e.event_type for e in tracer.get_events(result.execution_id)]
        assert "provider.response.received" in types and "agent.execution.completed" in types


class TestAgentLoopTracingFailures:
    def test_provider_exception_tracing(self):
        loop, tracer, _ = _setup()
        provider = ScriptedProvider([RuntimeError("provider down")])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.FAILED
        spans, _ = _span_map(tracer, result.execution_id)
        root = next(s for s in spans.values() if s.name == "agent.execution")
        assert root.status == SpanStatus.ERROR
        prov = next(s for s in spans.values() if s.name == "provider.request")
        assert prov.status == SpanStatus.ERROR
        types = [e.event_type for e in tracer.get_events(result.execution_id)]
        assert "provider.request.failed" in types
        assert "agent.execution.failed" in types

    def test_tool_validation_timeout_and_failure_paths(self):
        async def run_timeout():
            reg = ToolRegistry()
            async def slow(a, b): await asyncio.sleep(1); return a + b
            reg.register(_tool_def("add"), slow)
            cfg = OrchestratorConfig(max_iterations=5, max_tool_calls=5, tool_timeout=0.05)
            tracer = InMemoryTracer()
            loop = AgentLoop(config=cfg, tool_executor=ToolExecutor(reg, config=cfg), tracer=tracer)
            provider = AsyncScriptedProvider([
                ProviderResponse(tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1})]),
                ProviderResponse(content="handled"),
            ])
            return await loop.run_async(messages=[], provider=provider), tracer

        (state, result), tracer = asyncio.run(run_timeout())
        assert result.status == ExecutionStatus.COMPLETED
        spans, _ = _span_map(tracer, result.execution_id)
        tool_span = next(s for s in spans.values() if s.name == "tool.execute")
        assert tool_span.status == SpanStatus.ERROR
        types = [e.event_type for e in tracer.get_events(result.execution_id)]
        assert "tool.execution.timeout" in types

        # validation failure
        loop, tracer, _ = _setup()
        provider = ScriptedProvider([
            ProviderResponse(tool_calls=[ToolCall(id="c2", name="add", arguments={"a": "bad", "b": 1})]),
            ProviderResponse(content="recovered"),
        ])
        state, result = loop.run(messages=[], provider=provider)
        spans, _ = _span_map(tracer, result.execution_id)
        tool_span = next(s for s in spans.values() if s.name == "tool.execute")
        assert tool_span.status == SpanStatus.ERROR
        types = [e.event_type for e in tracer.get_events(result.execution_id)]
        assert "tool.validation.failed" in types

    def test_max_iterations_tracing(self):
        cfg = OrchestratorConfig(max_iterations=2, max_tool_calls=20)
        reg = ToolRegistry(); reg.register(_tool_def("add"), lambda a,b:a+b)
        tracer = InMemoryTracer()
        loop = AgentLoop(config=cfg, tool_executor=ToolExecutor(reg, config=cfg), tracer=tracer)
        provider = ScriptedProvider([
            ProviderResponse(tool_calls=[ToolCall(id="c1", name="add", arguments={"a":1,"b":1})]),
            ProviderResponse(tool_calls=[ToolCall(id="c2", name="add", arguments={"a":1,"b":1})]),
            ProviderResponse(content="never"),
        ])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.MAX_ITERATIONS_REACHED
        spans, _ = _span_map(tracer, result.execution_id)
        root = next(s for s in spans.values() if s.name == "agent.execution")
        assert root.status == SpanStatus.OK  # defined terminal limit
        types = [e.event_type for e in tracer.get_events(result.execution_id)]
        assert "agent.execution.max_iterations" in types

    def test_step_exception_closes_spans(self):
        tracer = InMemoryTracer()
        loop = AgentLoop(OrchestratorConfig(enable_tracing=True), tracer=tracer)

        def step(state):
            raise RuntimeError("boom")

        state, result = loop.run(step=step)
        assert result.status == ExecutionStatus.FAILED
        spans, _ = _span_map(tracer, result.execution_id)
        root = next(s for s in spans.values() if s.name == "agent.execution")
        iter_span = next(s for s in spans.values() if s.name == "iteration")
        assert root.status == SpanStatus.ERROR and root.end_time is not None
        assert iter_span.status == SpanStatus.ERROR and iter_span.end_time is not None


class TestTracingConfigAndResilience:
    def test_tracing_disabled_uses_noop(self):
        cfg = OrchestratorConfig(enable_tracing=False)
        loop = AgentLoop(config=cfg)
        provider = ScriptedProvider([ProviderResponse(content="ok")])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        assert isinstance(loop.tracer, NoOpTracer)

    def test_custom_tracer_injection_and_shared_with_executor(self):
        tracer = InMemoryTracer()
        reg = ToolRegistry(); reg.register(_tool_def("add"), lambda a,b:a+b)
        executor = ToolExecutor(reg)
        loop = AgentLoop(tool_executor=executor, tracer=tracer)
        assert loop.tracer is tracer
        assert executor.tracer is tracer  # shared, no conflicting instance
        provider = ScriptedProvider([
            ProviderResponse(tool_calls=[ToolCall(id="c1", name="add", arguments={"a":1,"b":1})]),
            ProviderResponse(content="ok"),
        ])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        assert tracer.get_trace(result.execution_id) is not None

    def test_tracer_exceptions_do_not_break_execution(self):
        class BoomTracer:
            def record(self, event): raise RuntimeError("nope")
            def start_span(self, *a, **k): raise RuntimeError("nope")
            def end_span(self, *a, **k): raise RuntimeError("nope")

        reg = ToolRegistry(); reg.register(_tool_def("add"), lambda a,b:a+b)
        executor = ToolExecutor(reg, tracer=BoomTracer())
        loop = AgentLoop(tool_executor=executor, tracer=BoomTracer())
        provider = ScriptedProvider([
            ProviderResponse(tool_calls=[ToolCall(id="c1", name="add", arguments={"a":2,"b":2})]),
            ProviderResponse(content="still ok"),
        ])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "still ok"
        assert result.tool_calls_count == 1
