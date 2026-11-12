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
    ) -> None:
        self._config = config or OrchestratorConfig()
        self._tool_executor = tool_executor

    @property
    def config(self) -> OrchestratorConfig:
        return self._config

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

        try:
            if step is not None:
                return self._run_with_step(state, tool_context, step)
            if provider is not None:
                return self._run_with_provider_sync(
                    state, tool_context, provider, tools, provider_retry_policy
                )
            return self._run_with_step(state, tool_context, self._default_step)
        except Exception as exc:
            return self._finish(
                state,
                ExecutionStatus.FAILED,
                tool_context,
                error=f"{type(exc).__name__}: {exc}",
            )

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

        try:
            if step is not None:
                return await self._run_with_step_async(state, tool_context, step)
            if provider is not None:
                return await self._run_with_provider_async(
                    state, tool_context, provider, tools, provider_retry_policy
                )
            return await self._run_with_step_async(state, tool_context, self._default_step)
        except Exception as exc:
            return self._finish(
                state,
                ExecutionStatus.FAILED,
                tool_context,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _run_with_step(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        step_fn: IterationStep,
    ) -> tuple[AgentState, ExecutionResult]:
        while True:
            if state.iteration_count >= self._config.max_iterations:
                return self._max_iterations(state, tool_context)

            state.iteration_count += 1
            outcome = step_fn(state)
            terminal = self._apply_outcome(state, tool_context, outcome)
            if terminal is not None:
                return terminal

    async def _run_with_step_async(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        step_fn: AsyncIterationStep | IterationStep,
    ) -> tuple[AgentState, ExecutionResult]:
        while True:
            if state.iteration_count >= self._config.max_iterations:
                return self._max_iterations(state, tool_context)

            state.iteration_count += 1
            outcome = step_fn(state)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            terminal = self._apply_outcome(state, tool_context, outcome)
            if terminal is not None:
                return terminal

    def _run_with_provider_sync(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        provider: LLMProvider,
        tools: list[ToolDefinition] | None,
        provider_retry_policy: RetryPolicy | None = None,
    ) -> tuple[AgentState, ExecutionResult]:
        available_tools = self._resolve_tools(tools)
        policy = self._resolve_provider_retry_policy(provider_retry_policy)
        while True:
            if state.iteration_count >= self._config.max_iterations:
                return self._max_iterations(state, tool_context)

            state.iteration_count += 1
            request = ProviderRequest(messages=list(state.messages), tools=available_tools)
            response = self._complete_with_retry_sync(provider, request, policy)
            terminal = self._handle_provider_response_sync(state, tool_context, response)
            if terminal is not None:
                return terminal

    async def _run_with_provider_async(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        provider: LLMProvider,
        tools: list[ToolDefinition] | None,
        provider_retry_policy: RetryPolicy | None = None,
    ) -> tuple[AgentState, ExecutionResult]:
        available_tools = self._resolve_tools(tools)
        policy = self._resolve_provider_retry_policy(provider_retry_policy)
        while True:
            if state.iteration_count >= self._config.max_iterations:
                return self._max_iterations(state, tool_context)

            state.iteration_count += 1
            request = ProviderRequest(messages=list(state.messages), tools=available_tools)
            response = await self._complete_with_retry_async(provider, request, policy)
            terminal = await self._handle_provider_response_async(
                state, tool_context, response
            )
            if terminal is not None:
                return terminal

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

    def _resolve_provider_retry_policy(
        self, override: RetryPolicy | None
    ) -> RetryPolicy:
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

    def _complete_with_retry_sync(
        self, provider: LLMProvider, request: ProviderRequest, policy: RetryPolicy
    ) -> ProviderResponse:
        attempt = 1
        while True:
            try:
                self._record_retry_event("provider.attempt.started", attempt, policy.max_attempts)
                response = provider.complete(request)
                self._record_retry_event(
                    "provider.attempt.succeeded", attempt, policy.max_attempts
                )
                return response
            except Exception as exc:
                if self._is_process_control_exception(exc):
                    self._record_retry_event(
                        "provider.attempt.failed", attempt, policy.max_attempts, error=exc
                    )
                    raise
                if policy.should_retry(attempt, exc):
                    retry_number = attempt
                    delay = float(policy.next_delay(retry_number))
                    self._record_retry_event(
                        "provider.attempt.failed", attempt, policy.max_attempts, error=exc
                    )
                    self._record_retry_event(
                        "provider.retry.scheduled",
                        attempt,
                        policy.max_attempts,
                        error=exc,
                        delay=delay,
                        retry_number=retry_number,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    attempt += 1
                    continue
                self._record_retry_event(
                    "provider.attempt.failed", attempt, policy.max_attempts, error=exc
                )
                raise

    async def _complete_with_retry_async(
        self, provider: LLMProvider, request: ProviderRequest, policy: RetryPolicy
    ) -> ProviderResponse:
        attempt = 1
        while True:
            try:
                self._record_retry_event("provider.attempt.started", attempt, policy.max_attempts)
                response = await self._complete_async(provider, request)
                self._record_retry_event(
                    "provider.attempt.succeeded", attempt, policy.max_attempts
                )
                return response
            except Exception as exc:
                if self._is_process_control_exception(exc):
                    self._record_retry_event(
                        "provider.attempt.failed", attempt, policy.max_attempts, error=exc
                    )
                    raise
                if policy.should_retry(attempt, exc):
                    retry_number = attempt
                    delay = float(policy.next_delay(retry_number))
                    self._record_retry_event(
                        "provider.attempt.failed", attempt, policy.max_attempts, error=exc
                    )
                    self._record_retry_event(
                        "provider.retry.scheduled",
                        attempt,
                        policy.max_attempts,
                        error=exc,
                        delay=delay,
                        retry_number=retry_number,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    attempt += 1
                    continue
                self._record_retry_event(
                    "provider.attempt.failed", attempt, policy.max_attempts, error=exc
                )
                raise

    def _record_retry_event(
        self,
        event_type: str,
        attempt: int,
        max_attempts: int,
        error: BaseException | None = None,
        delay: float | None = None,
        retry_number: int | None = None,
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
            # Use minimal ExecutionEvent-like object if tracing models are present;
            # otherwise pass a plain dict to tolerant tracers.
            try:
                from agent_orchestrator.tracing.models import ExecutionEvent  # type: ignore

                event = ExecutionEvent(
                    event_type=event_type,
                    trace_id="provider-retry",
                    span_id="provider-retry",
                    component="provider",
                    metadata=metadata,
                )
                record(event)
            except Exception:
                record(
                    {
                        "event_type": event_type,
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
    ) -> tuple[AgentState, ExecutionResult] | None:
        normalized = self._normalize_provider_response(response)
        if normalized.error is not None:
            return self._finish(
                state,
                ExecutionStatus.FAILED,
                tool_context,
                error=normalized.error,
            )
        if normalized.tool_calls is not None:
            self._append_assistant_tool_call_message(state, normalized.tool_calls, normalized.content)
            for tool_call, precomputed in normalized.tool_calls:
                if precomputed is not None:
                    self._consume_tool_call_slot(tool_context, precomputed)
                    tool_result = precomputed
                else:
                    tool_result = self._execute_tool_sync(tool_call, tool_context)
                self._append_tool_result_message(state, tool_result)
            return None
        return self._finish(
            state,
            ExecutionStatus.COMPLETED,
            tool_context,
            final_response=normalized.content,
        )

    async def _handle_provider_response_async(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        response: ProviderResponse,
    ) -> tuple[AgentState, ExecutionResult] | None:
        normalized = self._normalize_provider_response(response)
        if normalized.error is not None:
            return self._finish(
                state,
                ExecutionStatus.FAILED,
                tool_context,
                error=normalized.error,
            )
        if normalized.tool_calls is not None:
            self._append_assistant_tool_call_message(state, normalized.tool_calls, normalized.content)
            for tool_call, precomputed in normalized.tool_calls:
                if precomputed is not None:
                    self._consume_tool_call_slot(tool_context, precomputed)
                    tool_result = precomputed
                else:
                    tool_result = await self._execute_tool_async(tool_call, tool_context)
                self._append_tool_result_message(state, tool_result)
            return None
        return self._finish(
            state,
            ExecutionStatus.COMPLETED,
            tool_context,
            final_response=normalized.content,
        )

    def _execute_tool_sync(
        self, tool_call: ToolCall, tool_context: ExecutionContext | None
    ) -> ToolResult:
        if self._tool_executor is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                output=None,
                success=False,
                error="ToolExecutionError: No tool executor configured.",
            )
        return self._tool_executor.execute(tool_call, context=tool_context)

    async def _execute_tool_async(
        self, tool_call: ToolCall, tool_context: ExecutionContext | None
    ) -> ToolResult:
        if self._tool_executor is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                output=None,
                success=False,
                error="ToolExecutionError: No tool executor configured.",
            )
        return await self._tool_executor.execute_async(tool_call, context=tool_context)

    @staticmethod
    def _consume_tool_call_slot(
        tool_context: ExecutionContext | None, result: ToolResult
    ) -> None:
        if tool_context is None:
            return
        try:
            tool_context.consume()
        except Exception:
            # If the slot cannot be consumed, keep the precomputed result but do
            # not overwrite its specific error; count remains at the limit.
            pass

    @staticmethod
    def _append_assistant_tool_call_message(
        state: AgentState,
        tool_calls: list[tuple[ToolCall, ToolResult | None]],
        content: str | None,
    ) -> None:
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
        payload = {
            "success": result.success,
            "output": result.output,
            "error": result.error,
        }
        return json.dumps(payload, ensure_ascii=False)

    def _apply_outcome(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        outcome: IterationOutcome,
    ) -> tuple[AgentState, ExecutionResult] | None:
        if outcome.decision is IterationDecision.CONTINUE:
            return None
        if outcome.decision is IterationDecision.COMPLETE:
            return self._finish(
                state,
                ExecutionStatus.COMPLETED,
                tool_context,
                final_response=outcome.final_response,
            )
        if outcome.decision is IterationDecision.FAIL:
            return self._finish(
                state,
                ExecutionStatus.FAILED,
                tool_context,
                error=outcome.error or "Execution failed.",
            )
        return self._finish(
            state,
            ExecutionStatus.FAILED,
            tool_context,
            error=f"Unknown iteration decision: {outcome.decision!r}",
        )

    def _max_iterations(
        self, state: AgentState, tool_context: ExecutionContext | None
    ) -> tuple[AgentState, ExecutionResult]:
        return self._finish(
            state,
            ExecutionStatus.MAX_ITERATIONS_REACHED,
            tool_context,
            error=f"Maximum iterations reached ({self._config.max_iterations}).",
        )

    def _resolve_tools(
        self, tools: list[ToolDefinition] | None
    ) -> list[ToolDefinition] | None:
        if tools is not None:
            return tools
        if self._tool_executor is not None:
            # ToolExecutor stores the registry; use it to expose tool definitions.
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
    def _coerce_tool_call(item: Any) -> tuple[ToolCall, ToolResult | None]:
        if isinstance(item, ToolCall):
            return item, None
        try:
            if isinstance(item, dict):
                tool_call = ToolCall.model_validate(item)
            else:
                tool_call = ToolCall.model_validate(item)
            return tool_call, None
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
                fallback_id = "invalid"
                fallback_name = "unknown"
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

    def _initial_state(
        self, execution_id: str | None, messages: list[dict[str, Any]] | None
    ) -> AgentState:
        return AgentState(
            execution_id=execution_id or str(uuid.uuid4()),
            messages=list(messages or []),
            iteration_count=0,
            status=ExecutionStatus.PENDING,
            metadata={},
        )

    def _new_tool_context(self) -> ExecutionContext | None:
        if self._tool_executor is None:
            return None
        return self._tool_executor.new_context()

    @staticmethod
    def _default_step(state: AgentState) -> IterationOutcome:
        return IterationOutcome(decision=IterationDecision.COMPLETE)

    @staticmethod
    def _finish(
        state: AgentState,
        status: ExecutionStatus,
        tool_context: ExecutionContext | None,
        *,
        final_response: str | None = None,
        error: str | None = None,
    ) -> tuple[AgentState, ExecutionResult]:
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