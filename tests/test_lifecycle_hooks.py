"""Lifecycle hook tests (M6.5)."""

from __future__ import annotations

import asyncio

import pytest

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.core.loop import AgentLoop, IterationDecision, IterationOutcome
from agent_orchestrator.core.models import ExecutionResult, ExecutionStatus
from agent_orchestrator.hooks import HookEventType, HookManager, LifecycleEvent, LifecycleHook
from agent_orchestrator.providers.models import ProviderResponse
from agent_orchestrator.retry import RetryPolicy
from agent_orchestrator.state.models import AgentState
from agent_orchestrator.tools.executor import ToolExecutor
from agent_orchestrator.tools.models import ToolCall, ToolDefinition
from agent_orchestrator.tools.registry import ToolRegistry
from agent_orchestrator.tracing import InMemoryTracer


class RecordingHook:
    def __init__(self, name: str = "h", fail_on: set[HookEventType] | None = None):
        self.name = name
        self.events: list[LifecycleEvent] = []
        self.fail_on = fail_on or set()

    def on_event(self, event: LifecycleEvent):
        self.events.append(event)
        if event.event_type in self.fail_on:
            raise RuntimeError(f"{self.name}-boom")


class AsyncRecordingHook:
    def __init__(self, name: str = "ah", fail_on: set[HookEventType] | None = None):
        self.name = name
        self.events: list[LifecycleEvent] = []
        self.fail_on = fail_on or set()

    async def on_event(self, event: LifecycleEvent):
        self.events.append(event)
        if event.event_type in self.fail_on:
            raise RuntimeError(f"{self.name}-boom")


class AsyncCoroutineReturningHook:
    """Sync-looking hook that returns coroutine in sync paths (explicit handling)."""

    def __init__(self):
        self.called = 0

    def on_event(self, event: LifecycleEvent):
        self.called += 1

        async def _coro():
            return None

        return _coro()


def _tool_def(name: str = "add") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="add",
        parameters_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    )


def _setup_tool_loop(hooks=None, config=None):
    reg = ToolRegistry()
    reg.register(_tool_def("add"), lambda a, b: a + b)
    cfg = config or OrchestratorConfig()
    executor = ToolExecutor(reg, config=cfg, hooks=hooks)
    loop = AgentLoop(config=cfg, tool_executor=executor, hooks=hooks)
    return loop, reg, executor


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


class AsyncScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def complete_async(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


class TestHookBasics:
    def test_registration_order_and_multiple_hooks(self):
        h1 = RecordingHook("h1")
        h2 = RecordingHook("h2")
        loop = AgentLoop(OrchestratorConfig(max_iterations=1), hooks=[h1, h2])
        state, result = loop.run(messages=[{"role": "user", "content": "hi"}])
        assert result.status == ExecutionStatus.COMPLETED
        types1 = [e.event_type for e in h1.events]
        types2 = [e.event_type for e in h2.events]
        assert types1 == types2
        assert types1[0] == HookEventType.AGENT_STARTED
        assert types1[-1] == HookEventType.AGENT_COMPLETED
        # order: h1 recorded before h2? we only know both got same sequence; dispatch is ordered.
        # verify dispatch order by shared recorder
        shared = []

        class O:
            def __init__(self, n):
                self.n = n

            def on_event(self, e):
                shared.append((self.n, e.event_type))

        loop2 = AgentLoop(OrchestratorConfig(max_iterations=1), hooks=[O(1), O(2)])
        loop2.run(messages=[])
        assert shared[0][0] == 1 and shared[1][0] == 2
        assert shared[0][1] == shared[1][1]

    def test_payloads_and_execution_ids(self):
        h = RecordingHook()
        loop = AgentLoop(OrchestratorConfig(max_iterations=2), hooks=[h])
        state, result = loop.run(messages=[{"role": "u", "content": "x"}], execution_id="exec-1")
        assert result.execution_id == "exec-1"
        assert all(e.execution_id == "exec-1" for e in h.events)
        assert all(e.component == "agent_loop" for e in h.events if e.event_type.value.startswith("AGENT") or e.event_type.value.startswith("ITERATION"))
        iters = [e for e in h.events if e.event_type == HookEventType.ITERATION_STARTED]
        assert len(iters) == 1 and iters[0].iteration == 1
        assert all(e.timestamp is not None for e in h.events)

    def test_sync_hooks_in_async_execution(self):
        h = RecordingHook()
        loop = AgentLoop(OrchestratorConfig(max_iterations=1), hooks=[h])

        async def run():
            return await loop.run_async(messages=[{"role": "u", "content": "x"}])

        state, result = asyncio.run(run())
        assert result.status == ExecutionStatus.COMPLETED
        types = [e.event_type for e in h.events]
        assert HookEventType.AGENT_STARTED in types
        assert HookEventType.AGENT_COMPLETED in types

    def test_async_hooks_in_async_execution(self):
        h = AsyncRecordingHook()

        async def step(state):
            return IterationOutcome(decision=IterationDecision.COMPLETE, final_response="ok")

        loop = AgentLoop(OrchestratorConfig(max_iterations=1), hooks=[h])

        async def run():
            return await loop.run_async(messages=[], step=step)

        state, result = asyncio.run(run())
        assert result.status == ExecutionStatus.COMPLETED
        types = [e.event_type for e in h.events]
        assert HookEventType.AGENT_STARTED in types
        assert HookEventType.AGENT_COMPLETED in types

    def test_async_hook_in_sync_execution_explicit_behavior(self):
        # coroutine-returning hook must not leave unawaited coroutine / crash sync run
        h = AsyncCoroutineReturningHook()
        loop = AgentLoop(OrchestratorConfig(max_iterations=1), hooks=[h])
        state, result = loop.run(messages=[])
        assert result.status == ExecutionStatus.COMPLETED
        assert h.called >= 1  # hook invoked; coroutine closed best-effort


class TestHookErrors:
    def test_best_effort_exceptions_do_not_interrupt(self):
        h1 = RecordingHook("h1", fail_on={HookEventType.AGENT_STARTED, HookEventType.ITERATION_STARTED})
        h2 = RecordingHook("h2")
        loop = AgentLoop(OrchestratorConfig(max_iterations=1), hooks=[h1, h2])
        state, result = loop.run(messages=[{"role": "u", "content": "x"}])
        assert result.status == ExecutionStatus.COMPLETED
        assert state.status == ExecutionStatus.COMPLETED
        # h2 still receives events despite h1 failures
        assert any(e.event_type == HookEventType.AGENT_COMPLETED for e in h2.events)

    def test_hook_exception_does_not_change_retry_behavior(self):
        calls = {"n": 0}

        def flaky(a: int, b: int) -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return a + b

        h = RecordingHook(fail_on={HookEventType.TOOL_RETRY_SCHEDULED, HookEventType.TOOL_EXECUTION_STARTED})
        reg = ToolRegistry()
        reg.register(_tool_def(), flaky)
        cfg = OrchestratorConfig(tool_retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.0, max_delay=0.0))
        executor = ToolExecutor(reg, config=cfg, hooks=[h])
        result = executor.execute(ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}))
        assert result.success is True
        assert result.output == 3
        assert calls["n"] == 2

    def test_hook_exception_does_not_suppress_original_cancellation(self):
        h = AsyncRecordingHook(fail_on={HookEventType.AGENT_CANCELLED, HookEventType.ITERATION_FAILED})

        async def step(state):
            await asyncio.sleep(10)
            return None

        async def run():
            loop = AgentLoop(OrchestratorConfig(max_iterations=2), hooks=[h])
            task = asyncio.create_task(loop.run_async(messages=[], step=step))
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())


class TestAgentAndIterationEvents:
    def test_agent_success_failure_events(self):
        h = RecordingHook()
        loop = AgentLoop(OrchestratorConfig(max_iterations=1), hooks=[h])
        state, result = loop.run(messages=[])
        types = [e.event_type for e in h.events]
        assert types.count(HookEventType.AGENT_STARTED) == 1
        assert types.count(HookEventType.AGENT_COMPLETED) == 1
        assert HookEventType.AGENT_FAILED not in types

        h2 = RecordingHook()

        def fail_step(state):
            return IterationOutcome(decision=IterationDecision.FAIL, error="nope")

        loop2 = AgentLoop(OrchestratorConfig(max_iterations=2), hooks=[h2])
        state2, result2 = loop2.run(messages=[], step=fail_step)
        assert result2.status == ExecutionStatus.FAILED
        types2 = [e.event_type for e in h2.events]
        assert HookEventType.AGENT_FAILED in types2
        assert HookEventType.AGENT_COMPLETED not in types2

    def test_iteration_lifecycle_events(self):
        h = RecordingHook()
        steps = iter([
            IterationOutcome(decision=IterationDecision.CONTINUE),
            IterationOutcome(decision=IterationDecision.COMPLETE, final_response="ok"),
        ])

        def step(state):
            return next(steps)

        loop = AgentLoop(OrchestratorConfig(max_iterations=5), hooks=[h])
        state, result = loop.run(messages=[], step=step)
        assert result.status == ExecutionStatus.COMPLETED
        started = [e for e in h.events if e.event_type == HookEventType.ITERATION_STARTED]
        completed = [e for e in h.events if e.event_type == HookEventType.ITERATION_COMPLETED]
        assert [e.iteration for e in started] == [1, 2]
        assert [e.iteration for e in completed] == [1, 2]
        assert not [e for e in h.events if e.event_type == HookEventType.ITERATION_FAILED]

    def test_agent_cancelled_event_async(self):
        h = AsyncRecordingHook()
        started = asyncio.Event()

        async def slow_step(state):
            started.set()
            await asyncio.sleep(10)
            return None

        async def run():
            loop = AgentLoop(OrchestratorConfig(max_iterations=2), hooks=[h])
            task = asyncio.create_task(loop.run_async(messages=[], step=slow_step))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return h

        h = asyncio.run(run())
        types = [e.event_type for e in h.events]
        assert HookEventType.AGENT_STARTED in types
        assert HookEventType.AGENT_CANCELLED in types
        assert HookEventType.AGENT_COMPLETED not in types


class TestCancellationAndDispatchAudit:
    def test_genuine_task_cancellation_while_awaiting_agent_started_hook(self):
        """Genuine task cancellation during hook await must propagate (not swallowed)."""

        class BlockHook:
            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def on_event(self, event: LifecycleEvent):
                if event.event_type == HookEventType.AGENT_STARTED:
                    self.started.set()
                    await self.release.wait()

        hook = BlockHook()
        executed = {"step": 0}

        async def step(state):
            executed["step"] += 1
            return IterationOutcome(decision=IterationDecision.COMPLETE, final_response="ok")

        async def run():
            loop = AgentLoop(OrchestratorConfig(max_iterations=2), hooks=[hook])
            task = asyncio.create_task(loop.run_async(messages=[], step=step))
            await hook.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        assert executed["step"] == 0

    def test_genuine_task_cancellation_while_awaiting_tool_hook_prevents_continuation(self):
        """Cancellation during tool hook dispatch must not allow further execution."""

        class BlockToolHook:
            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def on_event(self, event: LifecycleEvent):
                if event.event_type == HookEventType.TOOL_EXECUTION_STARTED:
                    self.started.set()
                    await self.release.wait()

        hook = BlockToolHook()
        reg = ToolRegistry()
        tool_calls = {"n": 0}
        provider_calls = {"n": 0}

        def tool(a: int, b: int) -> int:
            tool_calls["n"] += 1
            return a + b

        reg.register(_tool_def("add"), tool)
        executor = ToolExecutor(reg, hooks=[hook])
        loop = AgentLoop(tool_executor=executor, hooks=[hook])

        class Provider:
            async def complete_async(self, request):
                provider_calls["n"] += 1
                if provider_calls["n"] == 1:
                    return ProviderResponse(
                        tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})]
                    )
                return ProviderResponse(content="done")

        async def run():
            task = asyncio.create_task(
                loop.run_async(messages=[{"role": "u", "content": "x"}], provider=Provider())
            )
            await hook.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        assert tool_calls["n"] == 0
        assert provider_calls["n"] == 1

    def test_hook_independently_raising_cancelled_error_is_best_effort(self):
        """Hook-raised CancelledError without task cancellation must not break execution."""

        class CancelRaisingHook:
            def __init__(self):
                self.events = []

            async def on_event(self, event: LifecycleEvent):
                self.events.append(event.event_type)
                if event.event_type == HookEventType.ITERATION_STARTED:
                    raise asyncio.CancelledError()

        hook = CancelRaisingHook()
        loop = AgentLoop(OrchestratorConfig(max_iterations=1), hooks=[hook])
        state, result = asyncio.run(loop.run_async(messages=[{"role": "u", "content": "x"}]))
        assert result.status == ExecutionStatus.COMPLETED
        assert HookEventType.AGENT_COMPLETED in hook.events
        assert HookEventType.ITERATION_STARTED in hook.events

    def test_original_execution_exception_preserved_when_cleanup_hooks_fail(self):
        boom = ValueError("original-step-failure")
        hook = AsyncRecordingHook(fail_on={HookEventType.ITERATION_FAILED, HookEventType.AGENT_FAILED})

        async def step(state):
            raise boom

        async def run():
            loop = AgentLoop(OrchestratorConfig(max_iterations=2), hooks=[hook])
            state, result = await loop.run_async(messages=[], step=step)
            return result

        result = asyncio.run(run())
        assert result.status == ExecutionStatus.FAILED
        assert "original-step-failure" in (result.error or "")
        types = [e.event_type for e in hook.events]
        assert types.count(HookEventType.ITERATION_FAILED) == 1
        assert types.count(HookEventType.AGENT_FAILED) == 1

    def test_cancellation_during_cleanup_preserves_original_and_no_duplicate_terminals(self):
        """Cancellation while emitting cleanup hooks must preserve original cancellation."""

        class CancelOnCleanupHook:
            def __init__(self):
                self.events = []
                self.step_started = asyncio.Event()
                self.release_step = asyncio.Event()

            async def on_event(self, event: LifecycleEvent):
                self.events.append(event.event_type)
                if event.event_type == HookEventType.ITERATION_STARTED:
                    self.step_started.set()
                    await self.release_step.wait()
                elif event.event_type == HookEventType.AGENT_CANCELLED:
                    raise asyncio.CancelledError()

        hook = CancelOnCleanupHook()

        async def run():
            loop = AgentLoop(OrchestratorConfig(max_iterations=2), hooks=[hook])
            task = asyncio.create_task(loop.run_async(messages=[]))
            await hook.step_started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        types = list(hook.events)
        assert types.count(HookEventType.AGENT_CANCELLED) == 1
        assert HookEventType.AGENT_COMPLETED not in types
        assert types.count(HookEventType.ITERATION_STARTED) == 1
        # iteration failed may be emitted once in cleanup path
        assert types.count(HookEventType.ITERATION_FAILED) <= 1

    def test_genuine_task_cancellation_while_awaiting_provider_request_started_hook(self):
        """Cancellation during PROVIDER_REQUEST_STARTED hook must stop before provider call."""

        class BlockProviderRequestHook:
            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()
                self.events = []

            async def on_event(self, event: LifecycleEvent):
                self.events.append(event.event_type)
                if event.event_type == HookEventType.PROVIDER_REQUEST_STARTED:
                    self.started.set()
                    await self.release.wait()

        hook = BlockProviderRequestHook()
        provider_calls = {"n": 0}

        class Provider:
            async def complete_async(self, request):
                provider_calls["n"] += 1
                return ProviderResponse(content="ok")

        async def run():
            tracer = InMemoryTracer()
            loop = AgentLoop(tracer=tracer, hooks=[hook])
            task = asyncio.create_task(
                loop.run_async(messages=[{"role": "u", "content": "x"}], provider=Provider())
            )
            await hook.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return tracer, hook

        tracer, hook = asyncio.run(run())
        assert provider_calls["n"] == 0
        types = list(hook.events)
        assert types.count(HookEventType.PROVIDER_REQUEST_STARTED) == 1
        assert HookEventType.PROVIDER_RESPONSE_RECEIVED not in types
        assert HookEventType.PROVIDER_REQUEST_FAILED not in types
        assert HookEventType.PROVIDER_RETRY_SCHEDULED not in types
        assert HookEventType.AGENT_COMPLETED not in types
        assert types.count(HookEventType.AGENT_CANCELLED) == 1
        trace = tracer.get_trace(tracer.list_traces()[0])
        assert trace is not None
        for s in trace.spans:
            assert s.end_time is not None
        root = next(s for s in trace.spans if s.name == "agent.execution")
        assert root.status.name == "ERROR"

    def test_genuine_task_cancellation_while_awaiting_provider_retry_scheduled_hook(self):
        """Cancellation during PROVIDER_RETRY_SCHEDULED hook must prevent subsequent retry."""

        class BlockProviderRetryHook:
            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()
                self.events = []

            async def on_event(self, event: LifecycleEvent):
                self.events.append(event.event_type)
                if event.event_type == HookEventType.PROVIDER_RETRY_SCHEDULED:
                    self.started.set()
                    await self.release.wait()

        hook = BlockProviderRetryHook()
        provider_calls = {"n": 0}

        class FlakyProvider:
            async def complete_async(self, request):
                provider_calls["n"] += 1
                raise RuntimeError("boom")

        async def run():
            tracer = InMemoryTracer()
            loop = AgentLoop(
                config=OrchestratorConfig(
                    provider_retry_policy=RetryPolicy(max_attempts=3, initial_delay=10.0, max_delay=10.0)
                ),
                tracer=tracer,
                hooks=[hook],
            )
            task = asyncio.create_task(
                loop.run_async(messages=[{"role": "u", "content": "x"}], provider=FlakyProvider())
            )
            await hook.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return tracer, hook

        tracer, hook = asyncio.run(run())
        assert provider_calls["n"] == 1
        types = list(hook.events)
        assert types.count(HookEventType.PROVIDER_REQUEST_STARTED) == 1
        assert types.count(HookEventType.PROVIDER_RETRY_SCHEDULED) == 1
        assert HookEventType.PROVIDER_RESPONSE_RECEIVED not in types
        # provider request failed/cancelled terminal may be emitted once in cleanup
        assert types.count(HookEventType.PROVIDER_REQUEST_FAILED) <= 1
        assert HookEventType.AGENT_COMPLETED not in types
        assert types.count(HookEventType.AGENT_CANCELLED) == 1
        trace = tracer.get_trace(tracer.list_traces()[0])
        assert trace is not None
        for s in trace.spans:
            assert s.end_time is not None
        prov_spans = [s for s in trace.spans if s.name == "provider.request"]
        assert len(prov_spans) == 1 and prov_spans[0].status.name == "ERROR"
        root = next(s for s in trace.spans if s.name == "agent.execution")
        assert root.status.name == "ERROR"

    def test_genuine_task_cancellation_while_awaiting_tool_retry_scheduled_hook(self):
        """Cancellation during TOOL_RETRY_SCHEDULED hook must prevent subsequent tool attempt."""

        class BlockToolRetryHook:
            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()
                self.events = []

            async def on_event(self, event: LifecycleEvent):
                self.events.append(event.event_type)
                if event.event_type == HookEventType.TOOL_RETRY_SCHEDULED:
                    self.started.set()
                    await self.release.wait()

        hook = BlockToolRetryHook()
        tool_calls = {"n": 0}
        reg = ToolRegistry()

        async def flaky(a: int, b: int) -> int:
            tool_calls["n"] += 1
            raise RuntimeError("boom")

        reg.register(_tool_def("add"), flaky)
        tracer = InMemoryTracer()
        cfg = OrchestratorConfig(tool_retry_policy=RetryPolicy(max_attempts=3, initial_delay=10.0, max_delay=10.0))
        executor = ToolExecutor(reg, config=cfg, tracer=tracer, hooks=[hook])
        loop = AgentLoop(config=cfg, tool_executor=executor, tracer=tracer, hooks=[hook])

        class Provider:
            async def complete_async(self, request):
                return ProviderResponse(tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})])

        async def run():
            task = asyncio.create_task(
                loop.run_async(messages=[{"role": "u", "content": "x"}], provider=Provider())
            )
            await hook.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        assert tool_calls["n"] == 1
        types = list(hook.events)
        assert types.count(HookEventType.TOOL_EXECUTION_STARTED) == 1
        assert types.count(HookEventType.TOOL_RETRY_SCHEDULED) == 1
        # no successful tool completion; terminal failed event may or may not be emitted depending on cancel boundary
        assert HookEventType.TOOL_EXECUTION_COMPLETED not in types
        assert HookEventType.AGENT_COMPLETED not in types
        assert types.count(HookEventType.AGENT_CANCELLED) == 1
        trace = tracer.get_trace(tracer.list_traces()[0])
        assert trace is not None
        for s in trace.spans:
            assert s.end_time is not None
        root = next(s for s in trace.spans if s.name == "agent.execution")
        assert root.status.name == "ERROR"

    def test_sync_coroutine_returning_hook_warning_free(self):
        h = AsyncCoroutineReturningHook()
        loop = AgentLoop(OrchestratorConfig(max_iterations=1), hooks=[h])
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            state, result = loop.run(messages=[])
        assert result.status == ExecutionStatus.COMPLETED
        assert h.called >= 1

    def test_provider_and_tool_terminal_events_exactly_once_across_paths(self):
        # provider retry exhaustion failure
        h = RecordingHook()

        class AlwaysFailProvider:
            def complete(self, request):
                raise RuntimeError("boom")

        loop = AgentLoop(
            config=OrchestratorConfig(provider_retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.0, max_delay=0.0)),
            hooks=[h],
        )
        state, result = loop.run(messages=[{"role": "u", "content": "x"}], provider=AlwaysFailProvider())
        assert result.status == ExecutionStatus.FAILED
        types = [e.event_type for e in h.events]
        assert types.count(HookEventType.PROVIDER_REQUEST_STARTED) == 1
        assert types.count(HookEventType.PROVIDER_REQUEST_FAILED) == 1
        assert types.count(HookEventType.PROVIDER_RESPONSE_RECEIVED) == 0
        assert types.count(HookEventType.AGENT_FAILED) == 1
        assert types.count(HookEventType.AGENT_COMPLETED) == 0

        # tool retry exhaustion failure (single executor execution => single terminal)
        h2 = RecordingHook()
        calls = {"n": 0}

        def flaky(a: int, b: int) -> int:
            calls["n"] += 1
            raise RuntimeError("boom")

        reg = ToolRegistry()
        reg.register(_tool_def(), flaky)
        cfg = OrchestratorConfig(tool_retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.0, max_delay=0.0))
        executor = ToolExecutor(reg, config=cfg, hooks=[h2])
        res = executor.execute(ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}))
        assert res.success is False
        t2 = [e.event_type for e in h2.events]
        assert t2.count(HookEventType.TOOL_EXECUTION_STARTED) == 1
        assert t2.count(HookEventType.TOOL_EXECUTION_FAILED) == 1
        assert t2.count(HookEventType.TOOL_EXECUTION_COMPLETED) == 0
        assert t2.count(HookEventType.TOOL_RETRY_SCHEDULED) == 1
        assert calls["n"] == 2

        h = RecordingHook()
        calls = {"n": 0}

        class FlakyProvider:
            def complete(self, request):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("boom")
                return ProviderResponse(content="ok")

        loop = AgentLoop(
            config=OrchestratorConfig(provider_retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.0, max_delay=0.0)),
            hooks=[h],
        )
        state, result = loop.run(messages=[{"role": "u", "content": "x"}], provider=FlakyProvider())
        assert result.status == ExecutionStatus.COMPLETED
        types = [e.event_type for e in h.events]
        assert types.count(HookEventType.PROVIDER_REQUEST_STARTED) == 1
        assert types.count(HookEventType.PROVIDER_RETRY_SCHEDULED) == 1
        assert types.count(HookEventType.PROVIDER_RESPONSE_RECEIVED) == 1
        assert HookEventType.PROVIDER_REQUEST_FAILED not in types
        retry = next(e for e in h.events if e.event_type == HookEventType.PROVIDER_RETRY_SCHEDULED)
        assert retry.metadata.get("retry_number") == 1
        assert retry.iteration == 1

    def test_tool_lifecycle_and_retry_scheduling(self):
        h = RecordingHook()
        calls = {"n": 0}

        def flaky(a: int, b: int) -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return a + b

        reg = ToolRegistry()
        reg.register(_tool_def(), flaky)
        cfg = OrchestratorConfig(tool_retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.0, max_delay=0.0))
        executor = ToolExecutor(reg, config=cfg, hooks=[h])
        result = executor.execute(ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}))
        assert result.success is True
        types = [e.event_type for e in h.events]
        assert types.count(HookEventType.TOOL_EXECUTION_STARTED) == 1
        assert types.count(HookEventType.TOOL_RETRY_SCHEDULED) == 1
        assert types.count(HookEventType.TOOL_EXECUTION_COMPLETED) == 1
        assert HookEventType.TOOL_EXECUTION_FAILED not in types

    def test_no_duplicate_events(self):
        h = RecordingHook()
        loop = AgentLoop(OrchestratorConfig(max_iterations=1), hooks=[h])
        loop.run(messages=[])
        types = [e.event_type for e in h.events]
        assert types.count(HookEventType.AGENT_STARTED) == 1
        assert types.count(HookEventType.AGENT_COMPLETED) == 1
        assert types.count(HookEventType.ITERATION_STARTED) == 1
        assert types.count(HookEventType.ITERATION_COMPLETED) == 1

        # provider single request
        h2 = RecordingHook()
        loop2 = AgentLoop(hooks=[h2])
        loop2.run(messages=[], provider=ScriptedProvider([ProviderResponse(content="ok")]))
        types2 = [e.event_type for e in h2.events]
        assert types2.count(HookEventType.PROVIDER_REQUEST_STARTED) == 1
        assert types2.count(HookEventType.PROVIDER_RESPONSE_RECEIVED) == 1
        assert types2.count(HookEventType.AGENT_COMPLETED) == 1

    def test_existing_tracing_behavior_unchanged(self):
        # tracer should still receive expected events alongside hooks
        tracer = InMemoryTracer()
        h = RecordingHook()
        loop = AgentLoop(OrchestratorConfig(max_iterations=1), tracer=tracer, hooks=[h])
        state, result = loop.run(messages=[])
        assert result.status == ExecutionStatus.COMPLETED
        events = tracer.get_events(result.execution_id)
        types = [e.event_type for e in events]
        assert "agent.execution.started" in types
        assert "agent.execution.completed" in types
        # hooks also fired
        assert any(e.event_type == HookEventType.AGENT_COMPLETED for e in h.events)
