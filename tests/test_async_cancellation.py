"""Async cancellation and cleanup tests (M6.4)."""

import asyncio

import pytest

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.core.loop import AgentLoop
from agent_orchestrator.core.models import ExecutionStatus
from agent_orchestrator.providers.models import ProviderResponse
from agent_orchestrator.retry import RetryPolicy
from agent_orchestrator.tools.executor import ToolExecutor
from agent_orchestrator.tools.models import ToolCall, ToolDefinition
from agent_orchestrator.tools.registry import ToolRegistry
from agent_orchestrator.tracing import InMemoryTracer, SpanStatus


class CountingTracer(InMemoryTracer):
    def __init__(self):
        super().__init__()
        self.end_calls = 0
        self.end_call_ids = []

    def end_span(self, span_id, trace_id, status=SpanStatus.OK):
        self.end_calls += 1
        self.end_call_ids.append((span_id, status))
        return super().end_span(span_id, trace_id, status)


def _tool_def(name="slow"):
    return ToolDefinition(
        name=name,
        description="slow tool",
        parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )


class TestCancellationProvider:
    def test_cancel_during_provider_async_propagates_no_retry(self):
        calls = {"n": 0}

        class SlowProvider:
            async def complete_async(self, request):
                calls["n"] += 1
                await asyncio.sleep(10)

        async def run():
            tracer = CountingTracer()
            loop = AgentLoop(
                config=OrchestratorConfig(provider_retry_policy=RetryPolicy(max_attempts=5, initial_delay=0.1, max_delay=0.1)),
                tracer=tracer,
            )
            task = asyncio.create_task(loop.run_async(messages=[{"role": "user", "content": "hi"}], provider=SlowProvider()))
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return tracer

        tracer = asyncio.run(run())
        assert calls["n"] == 1
        traces = tracer.list_traces()
        assert len(traces) == 1
        trace = tracer.get_trace(traces[0])
        assert trace is not None
        root = next(s for s in trace.spans if s.name == "agent.execution")
        assert root.status == SpanStatus.ERROR
        assert root.end_time is not None
        iter_spans = [s for s in trace.spans if s.name == "iteration"]
        assert len(iter_spans) == 1 and iter_spans[0].status == SpanStatus.ERROR
        prov_spans = [s for s in trace.spans if s.name == "provider.request"]
        assert len(prov_spans) == 1 and prov_spans[0].status == SpanStatus.ERROR
        # provider span ended exactly once
        prov_id = prov_spans[0].span_id
        assert sum(1 for sid, _ in tracer.end_call_ids if sid == prov_id) == 1
        # no success/completion events
        events = tracer.get_events(traces[0])
        types = [e.event_type for e in events]
        assert "agent.execution.completed" not in types
        assert "agent.execution.final_answer" not in types
        assert "provider.response.received" not in types

    def test_cancel_during_retry_backoff_provider(self):
        calls = {"n": 0}

        class FlakyProvider:
            async def complete_async(self, request):
                calls["n"] += 1
                raise RuntimeError("boom")

        async def run():
            tracer = CountingTracer()
            loop = AgentLoop(
                config=OrchestratorConfig(provider_retry_policy=RetryPolicy(max_attempts=5, initial_delay=10.0, max_delay=10.0)),
                tracer=tracer,
            )
            task = asyncio.create_task(loop.run_async(messages=[{"role": "user", "content": "hi"}], provider=FlakyProvider()))
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return tracer

        tracer = asyncio.run(run())
        assert calls["n"] == 1
        trace = tracer.get_trace(tracer.list_traces()[0])
        root = next(s for s in trace.spans if s.name == "agent.execution")
        assert root.status == SpanStatus.ERROR
        assert root.end_time is not None
        prov_spans = [s for s in trace.spans if s.name == "provider.request"]
        assert len(prov_spans) == 1 and prov_spans[0].status == SpanStatus.ERROR
        prov_id = prov_spans[0].span_id
        assert sum(1 for sid, _ in tracer.end_call_ids if sid == prov_id) == 1
        iter_spans = [s for s in trace.spans if s.name == "iteration"]
        assert len(iter_spans) == 1 and iter_spans[0].status == SpanStatus.ERROR
        events = tracer.get_events(tracer.list_traces()[0])
        types = [e.event_type for e in events]
        assert "provider.response.received" not in types
        assert "agent.execution.completed" not in types

    def test_provider_span_ended_once_on_retry_exhaustion_failure(self):
        class AlwaysFailProvider:
            async def complete_async(self, request):
                raise RuntimeError("boom")

        async def run():
            tracer = CountingTracer()
            loop = AgentLoop(
                config=OrchestratorConfig(provider_retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.0, max_delay=0.0)),
                tracer=tracer,
            )
            state, result = await loop.run_async(messages=[{"role": "user", "content": "hi"}], provider=AlwaysFailProvider())
            return tracer, result

        tracer, result = asyncio.run(run())
        assert result.status == ExecutionStatus.FAILED
        trace = tracer.get_trace(tracer.list_traces()[0])
        prov_spans = [s for s in trace.spans if s.name == "provider.request"]
        assert len(prov_spans) == 1 and prov_spans[0].status == SpanStatus.ERROR
        prov_id = prov_spans[0].span_id
        assert sum(1 for sid, _ in tracer.end_call_ids if sid == prov_id) == 1


class TestCancellationTool:
    def _setup(self):
        reg = ToolRegistry()
        tracer = CountingTracer()
        cfg = OrchestratorConfig()
        executor = ToolExecutor(reg, config=cfg, tracer=tracer)
        loop = AgentLoop(config=cfg, tool_executor=executor, tracer=tracer)
        return loop, reg, tracer

    def test_cancel_during_async_tool_execution(self):
        loop, reg, tracer = self._setup()
        started = asyncio.Event()

        async def slow():
            started.set()
            await asyncio.sleep(10)
            return "x"

        reg.register(_tool_def("slow"), slow)

        class Provider:
            async def complete_async(self, request):
                return ProviderResponse(tool_calls=[ToolCall(id="c1", name="slow", arguments={})])

        async def run():
            task = asyncio.create_task(loop.run_async(messages=[{"role": "u", "content": "x"}], provider=Provider()))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        trace = tracer.get_trace(tracer.list_traces()[0])
        assert trace is not None
        for s in trace.spans:
            assert s.end_time is not None
        root = next(s for s in trace.spans if s.name == "agent.execution")
        assert root.status == SpanStatus.ERROR
        tool_spans = [s for s in trace.spans if s.name == "tool.execute"]
        assert len(tool_spans) == 1
        assert tool_spans[0].status == SpanStatus.ERROR
        tool_id = tool_spans[0].span_id
        assert sum(1 for sid, _ in tracer.end_call_ids if sid == tool_id) == 1
        events = tracer.get_events(tracer.list_traces()[0])
        types = [e.event_type for e in events]
        assert "tool.execution.completed" not in types
        assert "agent.execution.completed" not in types

    def test_cancel_during_tool_retry_backoff(self):
        loop, reg, tracer = self._setup()
        calls = {"n": 0}

        async def flaky():
            calls["n"] += 1
            raise RuntimeError("boom")

        reg.register(_tool_def("slow"), flaky)
        tool_policy = RetryPolicy(max_attempts=5, initial_delay=10.0, max_delay=10.0)

        class Provider:
            async def complete_async(self, request):
                return ProviderResponse(tool_calls=[ToolCall(id="c1", name="slow", arguments={})])

        async def run():
            task = asyncio.create_task(
                loop.run_async(
                    messages=[{"role": "u", "content": "x"}],
                    provider=Provider(),
                    tool_retry_policy=tool_policy,
                )
            )
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        assert calls["n"] == 1
        trace = tracer.get_trace(tracer.list_traces()[0])
        for s in trace.spans:
            assert s.end_time is not None
        root = next(s for s in trace.spans if s.name == "agent.execution")
        assert root.status == SpanStatus.ERROR
        tool_spans = [s for s in trace.spans if s.name == "tool.execute"]
        assert len(tool_spans) == 1 and tool_spans[0].status == SpanStatus.ERROR
        tool_id = tool_spans[0].span_id
        assert sum(1 for sid, _ in tracer.end_call_ids if sid == tool_id) == 1
        events = tracer.get_events(tracer.list_traces()[0])
        types = [e.event_type for e in events]
        assert "tool.execution.completed" not in types
        assert "agent.execution.completed" not in types


class TestCancellationStep:
    def test_cancel_during_async_step(self):
        started = asyncio.Event()

        async def step(state):
            started.set()
            await asyncio.sleep(10)
            return None

        async def run():
            tracer = CountingTracer()
            loop = AgentLoop(tracer=tracer)
            task = asyncio.create_task(loop.run_async(messages=[], step=step))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return tracer

        tracer = asyncio.run(run())
        trace = tracer.get_trace(tracer.list_traces()[0])
        root = next(s for s in trace.spans if s.name == "agent.execution")
        assert root.status == SpanStatus.ERROR
        assert root.end_time is not None
        iter_spans = [s for s in trace.spans if s.name == "iteration"]
        assert len(iter_spans) == 1 and iter_spans[0].status == SpanStatus.ERROR
        root_id = root.span_id
        assert sum(1 for sid, _ in tracer.end_call_ids if sid == root_id) == 1
        for s in trace.spans:
            assert s.end_time is not None
        events = tracer.get_events(tracer.list_traces()[0])
        types = [e.event_type for e in events]
        assert "agent.execution.completed" not in types


class TestSyncUnaffected:
    def test_sync_run_unchanged(self):
        reg = ToolRegistry()
        reg.register(_tool_def("ok"), lambda: "ok")
        loop = AgentLoop(tool_executor=ToolExecutor(reg))

        class Provider:
            def complete(self, request):
                return ProviderResponse(content="done")

        state, result = loop.run(messages=[{"role": "user", "content": "hi"}], provider=Provider())
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "done"
