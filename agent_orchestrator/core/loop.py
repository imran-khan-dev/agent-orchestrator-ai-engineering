"""Agent execution loop foundation.

Provides lifecycle management, iteration tracking, and termination handling for
agent runs. Supports provider-independent provider/tool round trips when an
LLM provider is supplied. No concrete provider SDKs are used.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable

from pydantic import ValidationError

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.core.models import ExecutionResult, ExecutionStatus
from agent_orchestrator.providers.base import LLMProvider
from agent_orchestrator.providers.models import ProviderRequest, ProviderResponse
from agent_orchestrator.retry import RetryPolicy
from agent_orchestrator.state.models import AgentState
from agent_orchestrator.tools.executor import ExecutionContext, ToolExecutor
from agent_orchestrator.tools.models import ToolCall, ToolDefinition, ToolResult
from agent_orchestrator.tracing import ExecutionEvent, InMemoryTracer, NoOpTracer, SpanStatus, Tracer


class IterationDecision(str, Enum):
    """Outcome of a single loop iteration."""

    CONTINUE = "CONTINUE"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"


@dataclass(frozen=True)
class IterationOutcome:
    """Result returned by an iteration step."""

    decision: IterationDecision
    final_response: str | None = None
    error: str | None = None


IterationStep = Callable[[AgentState], IterationOutcome]
AsyncIterationStep = Callable[[AgentState], Awaitable[IterationOutcome]]


@dataclass(frozen=True)
class _NormalizedResponse:
    content: str | None = None
    tool_calls: list[tuple[ToolCall, ToolResult | None]] | None = None
    error: str | None = None


class AgentLoop:
    """Reusable agent execution loop.

    Manages execution state transitions, iteration counting, maximum-iteration
    enforcement, and structured results. When a provider is supplied, performs
    provider-independent tool-calling round trips using :class:`ToolExecutor`.
    A custom iteration step may be supplied instead for lifecycle testing.
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        tool_executor: ToolExecutor | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._config = config or OrchestratorConfig()
        self._tool_executor = tool_executor
        if tracer is not None:
            self._tracer: Tracer = tracer
        elif self._config.enable_tracing:
            self._tracer = InMemoryTracer()
        else:
            self._tracer = NoOpTracer()
        # Share tracer with executor to avoid conflicting instances.
        if self._tool_executor is not None:
            self._tool_executor.tracer = self._tracer

    @property
    def config(self) -> OrchestratorConfig:
        return self._config

    @property
    def tracer(self) -> Tracer:
        return self._tracer

    @tracer.setter
    def tracer(self, tracer: Tracer) -> None:
        self._tracer = tracer
        if self._tool_executor is not None:
            self._tool_executor.tracer = tracer

    def run(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        execution_id: str | None = None,
        step: IterationStep | None = None,
        provider: LLMProvider | None = None,
        tools: list[ToolDefinition] | None = None,
        provider_retry_policy: RetryPolicy | None = None,
    ) -> tuple[AgentState, ExecutionResult]:
        """Run an agent execution synchronously.

        Args:
            messages: Initial message history.
            execution_id: Optional execution identifier; generated if omitted.
            step: Optional iteration step (takes precedence when provided).
            provider: Optional LLM provider for provider/tool round trips.
            tools: Optional tool definitions for provider requests; defaults to
                definitions from the configured tool executor's registry.
            provider_retry_policy: Optional provider retry policy override.

        Returns:
            The final agent state and execution result.
        """
        state = self._initial_state(execution_id, messages)
        tool_context = self._new_tool_context()
        state.status = ExecutionStatus.RUNNING
        trace_id = state.execution_id
        root_span = self._start_span(
            trace_id, "agent.execution", "agent_loop", None, {"execution_id": trace_id}
        )
        self._record_event(
            "agent.execution.started", trace_id, root_span, "agent_loop", {"execution_id": trace_id}
        )

        try:
            if step is not None:
                result = self._run_with_step(state, tool_context, step, trace_id, root_span)
            elif provider is not None:
                result = self._run_with_provider_sync(
                    state, tool_context, provider, tools, provider_retry_policy, trace_id, root_span
                )
            else:
                result = self._run_with_step(
                    state, tool_context, self._default_step, trace_id, root_span
                )
            self._emit_completion_events(trace_id, root_span, result[1])
            self._end_span(root_span, trace_id, self._status_for_result(result[1]))
            return result
        except Exception as exc:
            res = self._finish(
                state,
                ExecutionStatus.FAILED,
                tool_context,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._record_event(
                "agent.execution.failed", trace_id, root_span, "agent_loop", {"execution_id": trace_id}
            )
            self._end_span(root_span, trace_id, SpanStatus.ERROR)
            return res

    async def run_async(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        execution_id: str | None = None,
        step: AsyncIterationStep | IterationStep | None = None,
        provider: LLMProvider | None = None,
        tools: list[ToolDefinition] | None = None,
        provider_retry_policy: RetryPolicy | None = None,
    ) -> tuple[AgentState, ExecutionResult]:
        """Run an agent execution asynchronously.

        The iteration step may be synchronous or asynchronous. When a provider is
        supplied, ``complete_async`` is used if available; otherwise
        ``complete`` runs in a worker thread to avoid blocking the event loop.
        """
        state = self._initial_state(execution_id, messages)
        tool_context = self._new_tool_context()
        state.status = ExecutionStatus.RUNNING
        trace_id = state.execution_id
        root_span = self._start_span(
            trace_id, "agent.execution", "agent_loop", None, {"execution_id": trace_id}
        )
        self._record_event(
            "agent.execution.started", trace_id, root_span, "agent_loop", {"execution_id": trace_id}
        )

        try:
            if step is not None:
                result = await self._run_with_step_async(
                    state, tool_context, step, trace_id, root_span
                )
            elif provider is not None:
                result = await self._run_with_provider_async(
                    state, tool_context, provider, tools, provider_retry_policy, trace_id, root_span
                )
            else:
                result = await self._run_with_step_async(
                    state, tool_context, self._default_step, trace_id, root_span
                )
            self._emit_completion_events(trace_id, root_span, result[1])
            self._end_span(root_span, trace_id, self._status_for_result(result[1]))
            return result
        except Exception as exc:
            res = self._finish(
                state,
                ExecutionStatus.FAILED,
                tool_context,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._record_event(
                "agent.execution.failed", trace_id, root_span, "agent_loop", {"execution_id": trace_id}
            )
            self._end_span(root_span, trace_id, SpanStatus.ERROR)
            return res

    def _run_with_step(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        step_fn: IterationStep,
        trace_id: str,
        parent_span_id: str | None,
    ) -> tuple[AgentState, ExecutionResult]:
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
            try:
                outcome = step_fn(state)
                terminal = self._apply_outcome(state, tool_context, outcome)
            except Exception:
                self._end_span(iter_span, trace_id, SpanStatus.ERROR)
                raise
            if terminal is not None:
                self._end_span(iter_span, trace_id, self._status_for_result(terminal[1]))
                return terminal
            self._end_span(iter_span, trace_id, SpanStatus.OK)

    async def _run_with_step_async(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        step_fn: AsyncIterationStep | IterationStep,
        trace_id: str,
        parent_span_id: str | None,
    ) -> tuple[AgentState, ExecutionResult]:
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
            try:
                outcome = step_fn(state)
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                terminal = self._apply_outcome(state, tool_context, outcome)
            except Exception:
                self._end_span(iter_span, trace_id, SpanStatus.ERROR)
                raise
            if terminal is not None:
                self._end_span(iter_span, trace_id, self._status_for_result(terminal[1]))
                return terminal
            self._end_span(iter_span, trace_id, SpanStatus.OK)

    def _run_with_provider_sync(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        provider: LLMProvider,
        tools: list[ToolDefinition] | None,
        provider_retry_policy: RetryPolicy | None,
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
            try:
                request = ProviderRequest(messages=list(state.messages), tools=available_tools)
                response = self._provider_call_with_retry_sync(
                    provider, request, policy, trace_id, iter_span, state.iteration_count
                )
                terminal = self._handle_provider_response_sync(
                    state, tool_context, response, trace_id, iter_span, state.iteration_count
                )
            except Exception:
                self._end_span(iter_span, trace_id, SpanStatus.ERROR)
                raise
            if terminal is not None:
                self._end_span(iter_span, trace_id, self._status_for_result(terminal[1]))
                return terminal
            self._end_span(iter_span, trace_id, SpanStatus.OK)

    async def _run_with_provider_async(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        provider: LLMProvider,
        tools: list[ToolDefinition] | None,
        provider_retry_policy: RetryPolicy | None,
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
            try:
                request = ProviderRequest(messages=list(state.messages), tools=available_tools)
                response = await self._provider_call_with_retry_async(
                    provider, request, policy, trace_id, iter_span, state.iteration_count
                )
                terminal = await self._handle_provider_response_async(
                    state, tool_context, response, trace_id, iter_span, state.iteration_count
                )
            except Exception:
                self._end_span(iter_span, trace_id, SpanStatus.ERROR)
                raise
            if terminal is not None:
                self._end_span(iter_span, trace_id, self._status_for_result(terminal[1]))
                return terminal
            self._end_span(iter_span, trace_id, SpanStatus.OK)

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
        attempt = 1
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
                        self._end_span(span_id, trace_id, SpanStatus.ERROR)
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
                    self._end_span(span_id, trace_id, SpanStatus.ERROR)
                    raise
        except Exception:
            # Ensure span closure on any unexpected path.
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
                        self._end_span(span_id, trace_id, SpanStatus.ERROR)
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
                    self._end_span(span_id, trace_id, SpanStatus.ERROR)
                    raise
        except Exception:
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
            # Prefer execution trace/span context when available.
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

    def _handle_provider_response_sync(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        response: ProviderResponse,
        trace_id: str,
        parent_span_id: str | None,
        iteration: int,
    ) -> tuple[AgentState, ExecutionResult] | None:
        normalized = self._normalize_provider_response(response)
        if normalized.error is not None:
            return self._finish(
                state, ExecutionStatus.FAILED, tool_context, error=normalized.error
            )
        if normalized.tool_calls is not None:
            self._append_assistant_tool_call_message(
                state, normalized.tool_calls, normalized.content
            )
            for tool_call, precomputed in normalized.tool_calls:
                if precomputed is not None:
                    self._consume_tool_call_slot(tool_context, precomputed)
                    self._record_precomputed_tool_event(
                        trace_id, parent_span_id, state, iteration, tool_call, precomputed
                    )
                    tool_result = precomputed
                else:
                    tool_result = self._execute_tool_sync(
                        tool_call, tool_context, trace_id, parent_span_id, state.execution_id, iteration
                    )
                self._append_tool_result_message(state, tool_result)
            return None
        return self._finish(
            state, ExecutionStatus.COMPLETED, tool_context, final_response=normalized.content
        )

    async def _handle_provider_response_async(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        response: ProviderResponse,
        trace_id: str,
        parent_span_id: str | None,
        iteration: int,
    ) -> tuple[AgentState, ExecutionResult] | None:
        normalized = self._normalize_provider_response(response)
        if normalized.error is not None:
            return self._finish(
                state, ExecutionStatus.FAILED, tool_context, error=normalized.error
            )
        if normalized.tool_calls is not None:
            self._append_assistant_tool_call_message(
                state, normalized.tool_calls, normalized.content
            )
            for tool_call, precomputed in normalized.tool_calls:
                if precomputed is not None:
                    self._consume_tool_call_slot(tool_context, precomputed)
                    self._record_precomputed_tool_event(
                        trace_id, parent_span_id, state, iteration, tool_call, precomputed
                    )
                    tool_result = precomputed
                else:
                    tool_result = await self._execute_tool_async(
                        tool_call, tool_context, trace_id, parent_span_id, state.execution_id, iteration
                    )
                self._append_tool_result_message(state, tool_result)
            return None
        return self._finish(
            state, ExecutionStatus.COMPLETED, tool_context, final_response=normalized.content
        )

    def _execute_tool_sync(self, tool_call, tool_context, trace_id, parent_span_id, execution_id, iteration):
        if self._tool_executor is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                output=None,
                success=False,
                error="ToolExecutionError: No tool executor configured.",
            )
        return self._tool_executor.execute(
            tool_call,
            context=tool_context,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            execution_id=execution_id,
            iteration=iteration,
        )

    async def _execute_tool_async(self, tool_call, tool_context, trace_id, parent_span_id, execution_id, iteration):
        if self._tool_executor is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                output=None,
                success=False,
                error="ToolExecutionError: No tool executor configured.",
            )
        return await self._tool_executor.execute_async(
            tool_call,
            context=tool_context,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            execution_id=execution_id,
            iteration=iteration,
        )

    @staticmethod
    def _consume_tool_call_slot(tool_context: ExecutionContext | None, result: ToolResult) -> None:
        if tool_context is None:
            return
        try:
            tool_context.consume()
        except Exception:
            pass

    @staticmethod
    def _append_assistant_tool_call_message(state: AgentState, tool_calls, content):
        state.messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [tc.model_dump() for tc, _ in tool_calls],
            }
        )

    @staticmethod
    def _append_tool_result_message(state: AgentState, result: ToolResult) -> None:
        state.messages.append(
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "name": result.tool_name,
                "content": AgentLoop._serialize_tool_result(result),
            }
        )

    @staticmethod
    def _serialize_tool_result(result: ToolResult) -> str:
        payload = {"success": result.success, "output": result.output, "error": result.error}
        return json.dumps(payload, ensure_ascii=False)

    def _apply_outcome(self, state, tool_context, outcome):
        if outcome.decision is IterationDecision.CONTINUE:
            return None
        if outcome.decision is IterationDecision.COMPLETE:
            return self._finish(
                state, ExecutionStatus.COMPLETED, tool_context, final_response=outcome.final_response
            )
        if outcome.decision is IterationDecision.FAIL:
            return self._finish(
                state, ExecutionStatus.FAILED, tool_context, error=outcome.error or "Execution failed."
            )
        return self._finish(
            state,
            ExecutionStatus.FAILED,
            tool_context,
            error=f"Unknown iteration decision: {outcome.decision!r}",
        )

    def _max_iterations(self, state, tool_context):
        return self._finish(
            state,
            ExecutionStatus.MAX_ITERATIONS_REACHED,
            tool_context,
            error=f"Maximum iterations reached ({self._config.max_iterations}).",
        )

    def _resolve_tools(self, tools):
        if tools is not None:
            return tools
        if self._tool_executor is not None:
            return self._tool_executor._registry.get_all_definitions() or None
        return None

    def _normalize_provider_response(self, response: Any) -> _NormalizedResponse:
        if isinstance(response, ProviderResponse):
            raw_calls = list(response.tool_calls or [])
            content = response.content
        elif hasattr(response, "tool_calls") or hasattr(response, "content"):
            raw_calls = list(getattr(response, "tool_calls", None) or [])
            content = getattr(response, "content", None)
        else:
            return _NormalizedResponse(
                error=(
                    "ProviderResponseError: Provider must return a ProviderResponse "
                    f"instance, got {type(response).__name__}."
                )
            )
        if raw_calls:
            normalized: list[tuple[ToolCall, ToolResult | None]] = []
            seen_ids: set[str] = set()
            for item in raw_calls:
                tool_call, precomputed = self._coerce_tool_call(item)
                if precomputed is not None:
                    normalized.append((tool_call, precomputed))
                    continue
                if tool_call.id in seen_ids:
                    precomputed = ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        output=None,
                        success=False,
                        error=f"ToolValidationError: Duplicate tool_call id '{tool_call.id}'.",
                    )
                    normalized.append((tool_call, precomputed))
                    continue
                seen_ids.add(tool_call.id)
                normalized.append((tool_call, None))
            return _NormalizedResponse(content=content, tool_calls=normalized)
        return _NormalizedResponse(content=content)

    @staticmethod
    def _coerce_tool_call(item):
        if isinstance(item, ToolCall):
            return item, None
        try:
            return ToolCall.model_validate(item), None
        except ValidationError as exc:
            fallback_id = ""
            fallback_name = ""
            if isinstance(item, dict):
                fallback_id = str(item.get("id", "") or "")
                fallback_name = str(item.get("name", "") or "")
            err_msg = f"ToolValidationError: Malformed tool call: {exc.errors()[0].get('msg', str(exc))}"
            try:
                placeholder = ToolCall(id=fallback_id or "invalid", name=fallback_name or "unknown")
            except ValidationError:
                placeholder = ToolCall(id="invalid", name="unknown")
                fallback_id, fallback_name = "invalid", "unknown"
            return placeholder, ToolResult(
                tool_call_id=fallback_id or placeholder.id,
                tool_name=fallback_name or placeholder.name,
                output=None,
                success=False,
                error=err_msg,
            )
        except Exception as exc:
            return ToolCall(id="invalid", name="unknown"), ToolResult(
                tool_call_id="invalid",
                tool_name="unknown",
                output=None,
                success=False,
                error=f"ToolValidationError: Malformed tool call: {type(exc).__name__}: {exc}",
            )

    def _initial_state(self, execution_id, messages):
        return AgentState(
            execution_id=execution_id or str(uuid.uuid4()),
            messages=list(messages or []),
            iteration_count=0,
            status=ExecutionStatus.PENDING,
            metadata={},
        )

    def _new_tool_context(self):
        if self._tool_executor is None:
            return None
        return self._tool_executor.new_context()

    @staticmethod
    def _default_step(state):
        return IterationOutcome(decision=IterationDecision.COMPLETE)

    @staticmethod
    def _finish(state, status, tool_context, *, final_response=None, error=None):
        state.status = status
        result = ExecutionResult(
            execution_id=state.execution_id,
            status=status,
            final_response=final_response,
            iterations=state.iteration_count,
            tool_calls_count=tool_context.tool_calls_count if tool_context else 0,
            error=error,
        )
        return state, result

    # --- tracing helpers (all safe/no-throw) ---

    def _start_span(self, trace_id, name, component, parent_span_id, attributes=None):
        span_id = str(uuid.uuid4())
        try:
            start = getattr(self._tracer, "start_span", None)
            if callable(start):
                start(trace_id, span_id, name, component, parent_span_id, attributes or {})
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

    def _record_event(self, event_type, trace_id, span_id, component, metadata=None):
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

    def _record_precomputed_tool_event(self, trace_id, parent_span_id, state, iteration, tool_call, result):
        """Record lifecycle event for malformed/duplicate tool calls handled without ToolExecutor."""
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
        self._record_event(event_type, trace_id, parent_span_id, "agent_loop", meta)

    def _emit_completion_events(self, trace_id, root_span, result):
        meta = {
            "execution_id": trace_id,
            "iterations": result.iterations,
            "tool_calls_count": result.tool_calls_count,
        }
        if result.status == ExecutionStatus.COMPLETED:
            self._record_event("agent.execution.final_answer", trace_id, root_span, "agent_loop", meta)
            self._record_event("agent.execution.completed", trace_id, root_span, "agent_loop", meta)
        elif result.status == ExecutionStatus.MAX_ITERATIONS_REACHED:
            self._record_event("agent.execution.max_iterations", trace_id, root_span, "agent_loop", meta)
            self._record_event("agent.execution.completed", trace_id, root_span, "agent_loop", meta)
        else:
            self._record_event("agent.execution.failed", trace_id, root_span, "agent_loop", meta)

    @staticmethod
    def _status_for_result(result):
        if result.status in (ExecutionStatus.COMPLETED, ExecutionStatus.MAX_ITERATIONS_REACHED):
            # Defined terminal outcomes (including limit) treated as closed spans.
            return SpanStatus.OK
        return SpanStatus.ERROR
