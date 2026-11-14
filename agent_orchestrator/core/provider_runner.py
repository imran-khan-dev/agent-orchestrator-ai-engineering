"""Provider execution, retry, and iteration logic for AgentLoop."""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any

from agent_orchestrator.core.models import ExecutionResult, ExecutionStatus
from agent_orchestrator.hooks import HookEventType
from agent_orchestrator.providers.base import LLMProvider
from agent_orchestrator.providers.models import ProviderRequest, ProviderResponse
from agent_orchestrator.retry import RetryPolicy
from agent_orchestrator.state.models import AgentState
from agent_orchestrator.tools.executor import ExecutionContext
from agent_orchestrator.tools.models import ToolDefinition
from agent_orchestrator.tracing import ExecutionEvent, SpanStatus


class ProviderRunnerMixin:
    """Runs provider iterations and applies provider retry policies."""

    def _run_with_provider_sync(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        provider: LLMProvider,
        tools: list[ToolDefinition] | None,
        provider_retry_policy: RetryPolicy | None,
        tool_retry_policy: RetryPolicy | None,
        trace_id: str,
        parent_span_id: str | None,
    ) -> tuple[AgentState, ExecutionResult]:
        available_tools = self._resolve_tools(tools)
        policy = self._resolve_provider_retry_policy(provider_retry_policy)
        while True:
            if state.iteration_count >= self._config.max_iterations:
                return self._max_iterations(state, tool_context)

            state.iteration_count += 1
            iter_span = self._start_span(
                trace_id,
                "iteration",
                "agent_loop",
                parent_span_id,
                {"execution_id": trace_id, "iteration": state.iteration_count},
            )
            self._record_event(
                "agent.iteration.started",
                trace_id,
                iter_span,
                "agent_loop",
                {"execution_id": trace_id, "iteration": state.iteration_count},
            )
            self._emit_hook(
                HookEventType.ITERATION_STARTED,
                execution_id=trace_id,
                iteration=state.iteration_count,
                metadata={"iteration": state.iteration_count},
            )
            try:
                request = ProviderRequest(messages=list(state.messages), tools=available_tools)
                response = self._provider_call_with_retry_sync(
                    provider, request, policy, trace_id, iter_span, state.iteration_count
                )
                terminal = self._handle_provider_response_sync(
                    state, tool_context, response, trace_id, iter_span, state.iteration_count,
                    tool_retry_policy=tool_retry_policy,
                )
            except Exception as exc:
                self._end_span(iter_span, trace_id, SpanStatus.ERROR)
                self._emit_hook(
                    HookEventType.ITERATION_FAILED,
                    execution_id=trace_id,
                    iteration=state.iteration_count,
                    metadata={"iteration": state.iteration_count},
                    error=exc,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
                raise
            if terminal is not None:
                if terminal[1].status == ExecutionStatus.FAILED:
                    self._end_span(iter_span, trace_id, SpanStatus.ERROR)
                    self._emit_hook(
                        HookEventType.ITERATION_FAILED,
                        execution_id=trace_id,
                        iteration=state.iteration_count,
                        metadata={"iteration": state.iteration_count, "status": terminal[1].status.value},
                        error_message=terminal[1].error,
                    )
                else:
                    self._end_span(iter_span, trace_id, self._status_for_result(terminal[1]))
                    self._emit_hook(
                        HookEventType.ITERATION_COMPLETED,
                        execution_id=trace_id,
                        iteration=state.iteration_count,
                        metadata={"iteration": state.iteration_count, "status": terminal[1].status.value},
                    )
                return terminal
            self._end_span(iter_span, trace_id, SpanStatus.OK)
            self._emit_hook(
                HookEventType.ITERATION_COMPLETED,
                execution_id=trace_id,
                iteration=state.iteration_count,
                metadata={"iteration": state.iteration_count},
            )

    async def _run_with_provider_async(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        provider: LLMProvider,
        tools: list[ToolDefinition] | None,
        provider_retry_policy: RetryPolicy | None,
        tool_retry_policy: RetryPolicy | None,
        trace_id: str,
        parent_span_id: str | None,
    ) -> tuple[AgentState, ExecutionResult]:
        available_tools = self._resolve_tools(tools)
        policy = self._resolve_provider_retry_policy(provider_retry_policy)
        while True:
            if state.iteration_count >= self._config.max_iterations:
                return self._max_iterations(state, tool_context)

            state.iteration_count += 1
            iter_span = self._start_span(
                trace_id,
                "iteration",
                "agent_loop",
                parent_span_id,
                {"execution_id": trace_id, "iteration": state.iteration_count},
            )
            self._record_event(
                "agent.iteration.started",
                trace_id,
                iter_span,
                "agent_loop",
                {"execution_id": trace_id, "iteration": state.iteration_count},
            )
            await self._aemit_hook(
                HookEventType.ITERATION_STARTED,
                execution_id=trace_id,
                iteration=state.iteration_count,
                metadata={"iteration": state.iteration_count},
            )
            try:
                request = ProviderRequest(messages=list(state.messages), tools=available_tools)
                response = await self._provider_call_with_retry_async(
                    provider, request, policy, trace_id, iter_span, state.iteration_count
                )
                terminal = await self._handle_provider_response_async(
                    state, tool_context, response, trace_id, iter_span, state.iteration_count,
                    tool_retry_policy=tool_retry_policy,
                )
            except asyncio.CancelledError as exc:
                self._end_span(iter_span, trace_id, SpanStatus.ERROR)
                try:
                    await self._aemit_hook(
                        HookEventType.ITERATION_FAILED,
                        execution_id=trace_id,
                        iteration=state.iteration_count,
                        metadata={"iteration": state.iteration_count, "cancelled": True},
                        error=exc,
                        error_message=f"{type(exc).__name__}",
                    )
                except asyncio.CancelledError:
                    # Preserve original execution cancellation during cleanup.
                    pass
                except Exception:
                    pass
                raise
            except Exception as exc:
                self._end_span(iter_span, trace_id, SpanStatus.ERROR)
                await self._aemit_hook(
                    HookEventType.ITERATION_FAILED,
                    execution_id=trace_id,
                    iteration=state.iteration_count,
                    metadata={"iteration": state.iteration_count},
                    error=exc,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
                raise
            if terminal is not None:
                if terminal[1].status == ExecutionStatus.FAILED:
                    self._end_span(iter_span, trace_id, SpanStatus.ERROR)
                    await self._aemit_hook(
                        HookEventType.ITERATION_FAILED,
                        execution_id=trace_id,
                        iteration=state.iteration_count,
                        metadata={"iteration": state.iteration_count, "status": terminal[1].status.value},
                        error_message=terminal[1].error,
                    )
                else:
                    self._end_span(iter_span, trace_id, self._status_for_result(terminal[1]))
                    await self._aemit_hook(
                        HookEventType.ITERATION_COMPLETED,
                        execution_id=trace_id,
                        iteration=state.iteration_count,
                        metadata={"iteration": state.iteration_count, "status": terminal[1].status.value},
                    )
                return terminal
            self._end_span(iter_span, trace_id, SpanStatus.OK)
            await self._aemit_hook(
                HookEventType.ITERATION_COMPLETED,
                execution_id=trace_id,
                iteration=state.iteration_count,
                metadata={"iteration": state.iteration_count},
            )

    def _provider_call_with_retry_sync(
        self,
        provider: LLMProvider,
        request: ProviderRequest,
        policy: RetryPolicy,
        trace_id: str,
        parent_span_id: str | None,
        iteration: int,
    ) -> ProviderResponse:
        span_id = self._start_span(
            trace_id,
            "provider.request",
            "provider",
            parent_span_id,
            {"execution_id": trace_id, "iteration": iteration},
        )
        self._record_event(
            "provider.request.started",
            trace_id,
            span_id,
            "provider",
            {"execution_id": trace_id, "iteration": iteration},
        )
        self._emit_hook(
            HookEventType.PROVIDER_REQUEST_STARTED,
            execution_id=trace_id,
            iteration=iteration,
            metadata={"iteration": iteration},
        )
        attempt = 1
        span_ended = False
        try:
            while True:
                try:
                    self._record_retry_event(
                        "provider.attempt.started",
                        attempt,
                        policy.max_attempts,
                        trace_id=trace_id,
                        span_id=span_id,
                    )
                    response = provider.complete(request)
                    self._record_retry_event(
                        "provider.attempt.succeeded",
                        attempt,
                        policy.max_attempts,
                        trace_id=trace_id,
                        span_id=span_id,
                    )
                    self._record_event(
                        "provider.response.received",
                        trace_id,
                        span_id,
                        "provider",
                        {
                            "execution_id": trace_id,
                            "iteration": iteration,
                            "has_tool_calls": bool(getattr(response, "tool_calls", None)),
                            "has_content": getattr(response, "content", None) is not None,
                        },
                    )
                    self._end_span(span_id, trace_id, SpanStatus.OK)
                    span_ended = True
                    self._emit_hook(
                        HookEventType.PROVIDER_RESPONSE_RECEIVED,
                        execution_id=trace_id,
                        iteration=iteration,
                        metadata={
                            "iteration": iteration,
                            "has_tool_calls": bool(getattr(response, "tool_calls", None)),
                            "has_content": getattr(response, "content", None) is not None,
                        },
                    )
                    return response
                except Exception as exc:
                    self._record_retry_event(
                        "provider.attempt.failed",
                        attempt,
                        policy.max_attempts,
                        error=exc,
                        trace_id=trace_id,
                        span_id=span_id,
                    )
                    if self._is_process_control_exception(exc):
                        self._record_event(
                            "provider.request.failed",
                            trace_id,
                            span_id,
                            "provider",
                            {"execution_id": trace_id, "iteration": iteration},
                        )
                        self._emit_hook(
                            HookEventType.PROVIDER_REQUEST_FAILED,
                            execution_id=trace_id,
                            iteration=iteration,
                            metadata={"iteration": iteration},
                            error=exc,
                            error_message=f"{type(exc).__name__}: {exc}",
                        )
                        raise
                    if policy.should_retry(attempt, exc):
                        delay = float(policy.next_delay(attempt))
                        self._record_retry_event(
                            "provider.retry.scheduled",
                            attempt,
                            policy.max_attempts,
                            error=exc,
                            delay=delay,
                            retry_number=attempt,
                            trace_id=trace_id,
                            span_id=span_id,
                        )
                        self._emit_hook(
                            HookEventType.PROVIDER_RETRY_SCHEDULED,
                            execution_id=trace_id,
                            iteration=iteration,
                            metadata={
                                "iteration": iteration,
                                "attempt": attempt,
                                "max_attempts": policy.max_attempts,
                                "retry_number": attempt,
                                "delay": delay,
                                "error_type": type(exc).__name__,
                            },
                            error=exc,
                        )
                        if delay > 0:
                            time.sleep(delay)
                        attempt += 1
                        continue
                    self._record_event(
                        "provider.request.failed",
                        trace_id,
                        span_id,
                        "provider",
                        {"execution_id": trace_id, "iteration": iteration},
                    )
                    self._emit_hook(
                        HookEventType.PROVIDER_REQUEST_FAILED,
                        execution_id=trace_id,
                        iteration=iteration,
                        metadata={"iteration": iteration},
                        error=exc,
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                    raise
        except BaseException:
            if not span_ended:
                try:
                    self._end_span(span_id, trace_id, SpanStatus.ERROR)
                except Exception:
                    pass
            raise

    async def _provider_call_with_retry_async(
        self,
        provider: LLMProvider,
        request: ProviderRequest,
        policy: RetryPolicy,
        trace_id: str,
        parent_span_id: str | None,
        iteration: int,
    ) -> ProviderResponse:
        span_id = self._start_span(
            trace_id,
            "provider.request",
            "provider",
            parent_span_id,
            {"execution_id": trace_id, "iteration": iteration},
        )
        self._record_event(
            "provider.request.started",
            trace_id,
            span_id,
            "provider",
            {"execution_id": trace_id, "iteration": iteration},
        )
        attempt = 1
        span_ended = False
        try:
            await self._aemit_hook(
                HookEventType.PROVIDER_REQUEST_STARTED,
                execution_id=trace_id,
                iteration=iteration,
                metadata={"iteration": iteration},
            )
            while True:
                try:
                    self._record_retry_event(
                        "provider.attempt.started",
                        attempt,
                        policy.max_attempts,
                        trace_id=trace_id,
                        span_id=span_id,
                    )
                    response = await self._complete_async(provider, request)
                    self._record_retry_event(
                        "provider.attempt.succeeded",
                        attempt,
                        policy.max_attempts,
                        trace_id=trace_id,
                        span_id=span_id,
                    )
                    self._record_event(
                        "provider.response.received",
                        trace_id,
                        span_id,
                        "provider",
                        {
                            "execution_id": trace_id,
                            "iteration": iteration,
                            "has_tool_calls": bool(getattr(response, "tool_calls", None)),
                            "has_content": getattr(response, "content", None) is not None,
                        },
                    )
                    self._end_span(span_id, trace_id, SpanStatus.OK)
                    span_ended = True
                    await self._aemit_hook(
                        HookEventType.PROVIDER_RESPONSE_RECEIVED,
                        execution_id=trace_id,
                        iteration=iteration,
                        metadata={
                            "iteration": iteration,
                            "has_tool_calls": bool(getattr(response, "tool_calls", None)),
                            "has_content": getattr(response, "content", None) is not None,
                        },
                    )
                    return response
                except asyncio.CancelledError as exc:
                    self._record_retry_event(
                        "provider.attempt.failed",
                        attempt,
                        policy.max_attempts,
                        error=exc,
                        trace_id=trace_id,
                        span_id=span_id,
                    )
                    self._record_event(
                        "provider.request.failed",
                        trace_id,
                        span_id,
                        "provider",
                        {"execution_id": trace_id, "iteration": iteration},
                    )
                    try:
                        await self._aemit_hook(
                            HookEventType.PROVIDER_REQUEST_FAILED,
                            execution_id=trace_id,
                            iteration=iteration,
                            metadata={"iteration": iteration, "cancelled": True},
                            error=exc,
                            error_message=f"{type(exc).__name__}",
                        )
                    except asyncio.CancelledError:
                        # Preserve original cancellation raised from provider flow.
                        pass
                    except Exception:
                        pass
                    raise
                except Exception as exc:
                    self._record_retry_event(
                        "provider.attempt.failed",
                        attempt,
                        policy.max_attempts,
                        error=exc,
                        trace_id=trace_id,
                        span_id=span_id,
                    )
                    if self._is_process_control_exception(exc):
                        self._record_event(
                            "provider.request.failed",
                            trace_id,
                            span_id,
                            "provider",
                            {"execution_id": trace_id, "iteration": iteration},
                        )
                        try:
                            await self._aemit_hook(
                                HookEventType.PROVIDER_REQUEST_FAILED,
                                execution_id=trace_id,
                                iteration=iteration,
                                metadata={"iteration": iteration},
                                error=exc,
                                error_message=f"{type(exc).__name__}: {exc}",
                            )
                        except asyncio.CancelledError:
                            # Preserve original process-control exception from provider flow.
                            pass
                        except Exception:
                            pass
                        raise
                    if policy.should_retry(attempt, exc):
                        delay = float(policy.next_delay(attempt))
                        self._record_retry_event(
                            "provider.retry.scheduled",
                            attempt,
                            policy.max_attempts,
                            error=exc,
                            delay=delay,
                            retry_number=attempt,
                            trace_id=trace_id,
                            span_id=span_id,
                        )
                        await self._aemit_hook(
                            HookEventType.PROVIDER_RETRY_SCHEDULED,
                            execution_id=trace_id,
                            iteration=iteration,
                            metadata={
                                "iteration": iteration,
                                "attempt": attempt,
                                "max_attempts": policy.max_attempts,
                                "retry_number": attempt,
                                "delay": delay,
                                "error_type": type(exc).__name__,
                            },
                            error=exc,
                        )
                        if delay > 0:
                            await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    self._record_event(
                        "provider.request.failed",
                        trace_id,
                        span_id,
                        "provider",
                        {"execution_id": trace_id, "iteration": iteration},
                    )
                    await self._aemit_hook(
                        HookEventType.PROVIDER_REQUEST_FAILED,
                        execution_id=trace_id,
                        iteration=iteration,
                        metadata={"iteration": iteration},
                        error=exc,
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                    raise
        except BaseException:
            if not span_ended:
                try:
                    self._end_span(span_id, trace_id, SpanStatus.ERROR)
                except Exception:
                    pass
            raise

    async def _complete_async(
        self, provider: LLMProvider, request: ProviderRequest
    ) -> ProviderResponse:
        complete_async = getattr(provider, "complete_async", None)
        if callable(complete_async):
            result = complete_async(request)
            if inspect.isawaitable(result):
                return await result
            return result
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, provider.complete, request)

    def _resolve_provider_retry_policy(self, override: RetryPolicy | None) -> RetryPolicy:
        if override is not None:
            return override
        provider_policy = getattr(self._config, "provider_retry_policy", None)
        if provider_policy is not None:
            return provider_policy
        return getattr(self._config, "retry_policy", RetryPolicy())

    @staticmethod
    def _is_process_control_exception(exc: BaseException) -> bool:
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            return True
        name = type(exc).__name__
        return name in {"CancelledError", "TaskCancelledException"}

    def _record_retry_event(
        self,
        event_type: str,
        attempt: int,
        max_attempts: int,
        *,
        error: BaseException | None = None,
        delay: float | None = None,
        retry_number: int | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        # Tracing is best-effort and optional; must not affect retries/results.
        try:
            tracer = getattr(self, "_tracer", None) or getattr(self, "tracer", None)
            if tracer is None:
                return
            record = getattr(tracer, "record", None)
            if not callable(record):
                return
            metadata: dict[str, Any] = {
                "attempt": attempt,
                "max_attempts": max_attempts,
            }
            if retry_number is not None:
                metadata["retry_number"] = retry_number
            if delay is not None:
                metadata["delay"] = delay
            if error is not None:
                metadata["error_type"] = type(error).__name__
            ev_trace = trace_id or "provider-retry"
            ev_span = span_id or "provider-retry"
            try:
                event = ExecutionEvent(
                    event_type=event_type,
                    trace_id=ev_trace,
                    span_id=ev_span,
                    component="provider",
                    metadata=metadata,
                )
                record(event)
            except Exception:
                record(
                    {
                        "event_type": event_type,
                        "trace_id": ev_trace,
                        "span_id": ev_span,
                        "component": "provider",
                        "metadata": metadata,
                    }
                )
        except Exception:
            return
