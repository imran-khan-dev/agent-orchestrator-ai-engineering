"""Tests for provider retry integration (M6.2)."""

import asyncio
import time

import pytest

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.core.loop import AgentLoop
from agent_orchestrator.core.models import ExecutionStatus
from agent_orchestrator.providers.models import ProviderRequest, ProviderResponse
from agent_orchestrator.retry import RetryPolicy
from agent_orchestrator.tools.executor import ToolExecutor
from agent_orchestrator.tools.models import ToolCall, ToolDefinition
from agent_orchestrator.tools.registry import ToolRegistry


def _tool_def(name: str = "add") -> ToolDefinition:
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


def _setup(cfg: OrchestratorConfig | None = None):
    reg = ToolRegistry()
    reg.register(_tool_def("add"), lambda a, b: a + b)
    cfg = cfg or OrchestratorConfig()
    executor = ToolExecutor(reg, config=cfg)
    loop = AgentLoop(config=cfg, tool_executor=executor)
    return loop, reg


class RecordingTracer:
    def __init__(self):
        self.events = []
        self.spans_started = []
        self.spans_ended = []

    def record(self, event):
        self.events.append(event)

    def start_span(self, *args, **kwargs):
        self.spans_started.append((args, kwargs))

    def end_span(self, *args, **kwargs):
        self.spans_ended.append((args, kwargs))


class FlakySyncProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        self.requests.append(request)
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class FlakyAsyncProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.calls = 0

    async def complete_async(self, request):
        await asyncio.sleep(0)
        self.calls += 1
        self.requests.append(request)
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class SyncOnlyProviderForAsync:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        self.requests.append(request)
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class TestProviderRetrySync:
    def test_sync_fails_once_then_succeeds(self):
        policy = RetryPolicy(max_attempts=2, initial_delay=0.0, max_delay=0.0)
        cfg = OrchestratorConfig(provider_retry_policy=policy)
        loop, _ = _setup(cfg)
        provider = FlakySyncProvider([RuntimeError("boom"), ProviderResponse(content="ok")])
        state, result = loop.run(messages=[{"role": "user", "content": "hi"}], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "ok"
        assert provider.calls == 2
        assert len(provider.requests) == 2
        assert provider.requests[0].messages == provider.requests[1].messages

    def test_retries_exhausted_preserves_error(self):
        policy = RetryPolicy(max_attempts=3, initial_delay=0.0, max_delay=0.0)
        cfg = OrchestratorConfig(provider_retry_policy=policy)
        loop, _ = _setup(cfg)
        provider = FlakySyncProvider([
            RuntimeError("fail-1"), RuntimeError("fail-2"), RuntimeError("fail-3")
        ])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.FAILED
        assert "RuntimeError" in (result.error or "")
        assert "fail-3" in (result.error or "")
        assert provider.calls == 3
        assert result.iterations == 1

    def test_non_retryable_exception_not_retried(self):
        policy = RetryPolicy(max_attempts=5, initial_delay=0.0, max_delay=0.0, retry_on=(ValueError,))
        cfg = OrchestratorConfig(provider_retry_policy=policy)
        loop, _ = _setup(cfg)
        provider = FlakySyncProvider([RuntimeError("no-retry")])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.FAILED
        assert "RuntimeError" in (result.error or "")
        assert provider.calls == 1

    def test_default_one_attempt_no_retry(self):
        loop, _ = _setup(OrchestratorConfig())
        provider = FlakySyncProvider([RuntimeError("boom")])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.FAILED
        assert provider.calls == 1

    def test_backoff_uses_sleep_and_max_delay_no_sleep_after_final(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda d: sleeps.append(d))
        policy = RetryPolicy(
            max_attempts=4, initial_delay=1.0, max_delay=3.0, backoff_multiplier=2.0, jitter=False
        )
        cfg = OrchestratorConfig(provider_retry_policy=policy)
        loop, _ = _setup(cfg)
        provider = FlakySyncProvider([
            RuntimeError("1"), RuntimeError("2"), RuntimeError("3"), RuntimeError("4")
        ])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.FAILED
        assert provider.calls == 4
        assert sleeps == [1.0, 2.0, 3.0]  # capped, no sleep after final

    def test_provider_response_and_history_correct_across_retry(self):
        policy = RetryPolicy(max_attempts=2, initial_delay=0.0, max_delay=0.0)
        cfg = OrchestratorConfig(provider_retry_policy=policy)
        loop, _ = _setup(cfg)
        provider = FlakySyncProvider([
            RuntimeError("temp"),
            ProviderResponse(tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]),
            ProviderResponse(content="5"),
        ])
        state, result = loop.run(messages=[{"role": "user", "content": "add"}], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "5"
        assert result.iterations == 2
        assert provider.calls == 3
        # first logical request retried with same history
        assert provider.requests[0].messages == [{"role": "user", "content": "add"}]
        assert provider.requests[1].messages == [{"role": "user", "content": "add"}]

    def test_does_not_retry_malformed_response_or_tool_failure(self):
        # malformed response -> loop turns into failed result (no exception), so no provider retry
        policy = RetryPolicy(max_attempts=3, initial_delay=0.0, max_delay=0.0)
        cfg = OrchestratorConfig(provider_retry_policy=policy)
        loop, _ = _setup(cfg)
        provider = FlakySyncProvider(["not-a-response"])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.FAILED
        assert provider.calls == 1

        # tool execution failure is handled as ToolResult fed back (no provider exception)
        loop, _ = _setup(cfg)
        provider = FlakySyncProvider([
            ProviderResponse(tool_calls=[ToolCall(id="c1", name="missing", arguments={})]),
            ProviderResponse(content="handled"),
        ])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        assert provider.calls == 2

    def test_run_level_provider_retry_policy_override(self):
        # config says no retries, call overrides to allow
        cfg = OrchestratorConfig(provider_retry_policy=RetryPolicy(max_attempts=1))
        loop, _ = _setup(cfg)
        provider = FlakySyncProvider([RuntimeError("boom"), ProviderResponse(content="ok")])
        state, result = loop.run(
            messages=[], provider=provider,
            provider_retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.0, max_delay=0.0),
        )
        assert result.status == ExecutionStatus.COMPLETED
        assert provider.calls == 2


class TestProviderRetryAsync:
    def test_async_fails_once_then_succeeds(self):
        async def run():
            policy = RetryPolicy(max_attempts=2, initial_delay=0.0, max_delay=0.0)
            cfg = OrchestratorConfig(provider_retry_policy=policy)
            loop, _ = _setup(cfg)
            provider = FlakyAsyncProvider([RuntimeError("boom"), ProviderResponse(content="ok")])
            state, result = await loop.run_async(messages=[{"role": "user", "content": "hi"}], provider=provider)
            return state, result, provider

        state, result, provider = asyncio.run(run())
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "ok"
        assert provider.calls == 2
        assert provider.requests[0].messages == provider.requests[1].messages

    def test_sync_provider_in_async_path_retries(self):
        async def run():
            policy = RetryPolicy(max_attempts=2, initial_delay=0.0, max_delay=0.0)
            cfg = OrchestratorConfig(provider_retry_policy=policy)
            loop, _ = _setup(cfg)
            provider = SyncOnlyProviderForAsync([RuntimeError("boom"), ProviderResponse(content="ok")])
            return await loop.run_async(messages=[], provider=provider), provider

        (state, result), provider = asyncio.run(run())
        assert result.status == ExecutionStatus.COMPLETED
        assert provider.calls == 2

    def test_async_backoff_uses_asyncio_sleep(self, monkeypatch):
        async def run():
            sleeps = []
            orig_sleep = asyncio.sleep

            async def fake_sleep(d):
                sleeps.append(d)

            monkeypatch.setattr(asyncio, "sleep", fake_sleep)
            # but FlakyAsyncProvider uses asyncio.sleep(0); patch after construction not needed except our calls
            # easier: track only nonzero or all; provider does sleep(0) in complete_async in our helper
            policy = RetryPolicy(max_attempts=3, initial_delay=1.0, max_delay=5.0, backoff_multiplier=2.0)
            cfg = OrchestratorConfig(provider_retry_policy=policy)
            loop, _ = _setup(cfg)

            class P:
                def __init__(self):
                    self.calls = 0
                async def complete_async(self, request):
                    self.calls += 1
                    if self.calls < 3:
                        raise RuntimeError("x")
                    return ProviderResponse(content="ok")

            provider = P()
            state, result = await loop.run_async(messages=[], provider=provider)
            return result, provider, sleeps

        result, provider, sleeps = asyncio.run(run())
        assert result.status == ExecutionStatus.COMPLETED
        assert provider.calls == 3
        assert sleeps == [1.0, 2.0]

    def test_async_cancellation_not_retried(self):
        async def run():
            policy = RetryPolicy(max_attempts=5, initial_delay=0.0, max_delay=0.0)
            cfg = OrchestratorConfig(provider_retry_policy=policy)
            loop, _ = _setup(cfg)

            class P:
                def __init__(self):
                    self.calls = 0
                async def complete_async(self, request):
                    self.calls += 1
                    raise asyncio.CancelledError()

            provider = P()
            try:
                await loop.run_async(messages=[], provider=provider)
                return None, provider
            except BaseException as e:
                return e, provider

        err, provider = asyncio.run(run())
        # cancellation is process-control; must not be retried
        assert provider.calls == 1
        assert err is not None

    def test_async_retries_exhausted(self):
        async def run():
            policy = RetryPolicy(max_attempts=2, initial_delay=0.0, max_delay=0.0)
            cfg = OrchestratorConfig(provider_retry_policy=policy)
            loop, _ = _setup(cfg)
            provider = FlakyAsyncProvider([RuntimeError("a"), RuntimeError("b")])
            return await loop.run_async(messages=[], provider=provider), provider

        (state, result), provider = asyncio.run(run())
        assert result.status == ExecutionStatus.FAILED
        assert provider.calls == 2
        assert "RuntimeError" in (result.error or "")


class TestProviderRetryTracing:
    def test_retry_events_recorded_on_tracer_when_available(self):
        policy = RetryPolicy(max_attempts=3, initial_delay=0.0, max_delay=0.0)
        cfg = OrchestratorConfig(provider_retry_policy=policy)
        loop, _ = _setup(cfg)
        tracer = RecordingTracer()
        # attach for tracers supported by loop implementation
        setattr(loop, "_tracer", tracer)
        setattr(loop, "tracer", tracer)
        provider = FlakySyncProvider([RuntimeError("x"), RuntimeError("y"), ProviderResponse(content="ok")])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        # extract event types and metadata from either ExecutionEvent-like or dict
        def etype(e):
            return getattr(e, "event_type", None) or (e.get("event_type") if isinstance(e, dict) else None)
        def meta(e):
            return getattr(e, "metadata", None) or (e.get("metadata") if isinstance(e, dict) else {}) or {}
        types = [etype(e) for e in tracer.events]
        assert "provider.attempt.started" in types
        assert "provider.attempt.failed" in types
        assert "provider.retry.scheduled" in types
        retry_events = [e for e in tracer.events if etype(e) == "provider.retry.scheduled"]
        assert len(retry_events) == 2
        m0 = meta(retry_events[0])
        assert m0.get("attempt") == 1
        assert m0.get("max_attempts") == 3
        assert "delay" in m0
        assert m0.get("error_type") == "RuntimeError"

    def test_tracer_failure_does_not_affect_retries(self):
        class BoomTracer:
            def record(self, event):
                raise RuntimeError("tracer boom")
            def start_span(self, *a, **k):
                raise RuntimeError("tracer boom")
            def end_span(self, *a, **k):
                raise RuntimeError("tracer boom")

        policy = RetryPolicy(max_attempts=2, initial_delay=0.0, max_delay=0.0)
        cfg = OrchestratorConfig(provider_retry_policy=policy)
        loop, _ = _setup(cfg)
        setattr(loop, "_tracer", BoomTracer())
        setattr(loop, "tracer", BoomTracer())
        provider = FlakySyncProvider([RuntimeError("x"), ProviderResponse(content="ok")])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "ok"
        assert provider.calls == 2
