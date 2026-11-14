"""Agent execution loop foundation.

Provides lifecycle management, iteration tracking, and termination handling for
agent runs. Supports provider-independent provider/tool round trips when an
LLM provider is supplied. No concrete provider SDKs are used.
"""
# what does lifecycle management

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Sequence

from pydantic import ValidationError

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.core.models import ExecutionResult, ExecutionStatus
from agent_orchestrator.hooks import HookEventType, HookManager, LifecycleEvent, LifecycleHook
from agent_orchestrator.providers.base import LLMProvider
from agent_orchestrator.retry import RetryPolicy
from agent_orchestrator.state.models import AgentState
from agent_orchestrator.tools.executor import ToolExecutor
from agent_orchestrator.tools.models import ToolDefinition
from agent_orchestrator.tracing import ExecutionEvent, InMemoryTracer, NoOpTracer, SpanStatus, Tracer

from agent_orchestrator.core.step_runner import StepRunnerMixin
from agent_orchestrator.core.provider_runner import ProviderRunnerMixin
from agent_orchestrator.core.response_handler import ResponseHandlerMixin
from agent_orchestrator.core.response_normalizer import ResponseNormalizerMixin


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


class AgentLoop(
    StepRunnerMixin,
    ProviderRunnerMixin,
    ResponseHandlerMixin,
    ResponseNormalizerMixin,
):
    """Reusable agent execution loop.

    Manages execution state transitions, iteration counting, maximum-iteration
    enforcement, and structured results. 
    
    When a provider is supplied, performs
    provider-independent tool-calling round trips using :class:`ToolExecutor`.
    A custom iteration step may be supplied instead for lifecycle testing.
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        tool_executor: ToolExecutor | None = None,
        tracer: Tracer | None = None,
        hooks: Sequence[LifecycleHook] | None = None,
    ) -> None:
        
        self._config = config or OrchestratorConfig()
        self._tool_executor = tool_executor

        if tracer is not None:
            self._tracer: Tracer = tracer
        elif self._config.enable_tracing:
            self._tracer = InMemoryTracer()
        else:
            self._tracer = NoOpTracer()
        
        self._hook_manager = HookManager(hooks)
        # Share tracer and hooks with executor to avoid conflicting instances.
        if self._tool_executor is not None:
            self._tool_executor.tracer = self._tracer
            self._tool_executor._hook_manager = self._hook_manager

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

    @property
    def hooks(self) -> tuple[LifecycleHook, ...]:
        return self._hook_manager.hooks

# helper
    def _emit_hook(
        self,
        event_type: HookEventType,
        *,
        execution_id: str | None = None,
        iteration: int | None = None,
        metadata: dict[str, Any] | None = None,
        error: BaseException | None = None,
        error_message: str | None = None,
    ) -> None:
        try:
            self._hook_manager.emit(
                LifecycleEvent(
                    event_type=event_type,
                    execution_id=execution_id,
                    iteration=iteration,
                    component="agent_loop",
                    metadata=dict(metadata or {}),
                    error=error,
                    error_message=error_message,
                )
            )
        except BaseException:
            # Best-effort for sync path; callers preserve original execution exceptions.
            return

# helper
    async def _aemit_hook(
        self,
        event_type: HookEventType,
        *,
        execution_id: str | None = None,
        iteration: int | None = None,
        metadata: dict[str, Any] | None = None,
        error: BaseException | None = None,
        error_message: str | None = None,
    ) -> None:
        await self._hook_manager.aemit(
            LifecycleEvent(
                event_type=event_type,
                execution_id=execution_id,
                iteration=iteration,
                component="agent_loop",
                metadata=dict(metadata or {}),
                error=error,
                error_message=error_message,
            )
        )

# helper
    def _emit_terminal_agent_hook(self, execution_id: str, result: ExecutionResult) -> None:
        meta = {
            "status": result.status.value,
            "iterations": result.iterations,
            "tool_calls_count": result.tool_calls_count,
        }
        if result.status == ExecutionStatus.COMPLETED:
            self._emit_hook(
                HookEventType.AGENT_COMPLETED, execution_id=execution_id, metadata=meta
            )
        else:
            self._emit_hook(
                HookEventType.AGENT_FAILED,
                execution_id=execution_id,
                metadata=meta,
                error_message=result.error,
            )

# helper
    async def _aemit_terminal_agent_hook(self, execution_id: str, result: ExecutionResult) -> None:
        meta = {
            "status": result.status.value,
            "iterations": result.iterations,
            "tool_calls_count": result.tool_calls_count,
        }
        if result.status == ExecutionStatus.COMPLETED:
            await self._aemit_hook(
                HookEventType.AGENT_COMPLETED, execution_id=execution_id, metadata=meta
            )
        else:
            await self._aemit_hook(
                HookEventType.AGENT_FAILED,
                execution_id=execution_id,
                metadata=meta,
                error_message=result.error,
            )

    def run(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        execution_id: str | None = None,
        step: IterationStep | None = None,
        provider: LLMProvider | None = None,
        tools: list[ToolDefinition] | None = None,
        provider_retry_policy: RetryPolicy | None = None,
        tool_retry_policy: RetryPolicy | None = None,
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
            tool_retry_policy: Optional tool retry policy override.

        Returns:
            The final agent state and execution result.
        """
        state = self._initial_state(execution_id, messages) # why and what is?
        tool_context = self._new_tool_context() # what is this?
        state.status = ExecutionStatus.RUNNING
        trace_id = state.execution_id # why and what is?
        root_span = self._start_span(
            trace_id, "agent.execution", "agent_loop", None, {"execution_id": trace_id}
        ) # why and what is?
        self._record_event(
            "agent.execution.started", trace_id, root_span, "agent_loop", {"execution_id": trace_id}
        ) # why and what is?
        self._emit_hook(
            HookEventType.AGENT_STARTED,
            execution_id=trace_id,
            metadata={"execution_id": trace_id},
        ) # why and what is?

        try:
            if step is not None:
                result = self._run_with_step(state, tool_context, step, trace_id, root_span)
            elif provider is not None:
                result = self._run_with_provider_sync(
                    state, tool_context, provider, tools, provider_retry_policy, tool_retry_policy, trace_id, root_span
                )
            else:
                result = self._run_with_step(
                    state, tool_context, self._default_step, trace_id, root_span
                )
            self._emit_completion_events(trace_id, root_span, result[1])
            self._end_span(root_span, trace_id, self._status_for_result(result[1]))
            self._emit_terminal_agent_hook(trace_id, result[1])
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
            self._emit_hook(
                HookEventType.AGENT_FAILED,
                execution_id=trace_id,
                metadata={"status": res[1].status.value, "iterations": res[1].iterations},
                error=exc,
                error_message=res[1].error,
            )
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
        tool_retry_policy: RetryPolicy | None = None,
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
        await self._aemit_hook(
            HookEventType.AGENT_STARTED,
            execution_id=trace_id,
            metadata={"execution_id": trace_id},
        )

        try:
            if step is not None:
                result = await self._run_with_step_async(
                    state, tool_context, step, trace_id, root_span
                )
            elif provider is not None:
                result = await self._run_with_provider_async(
                    state, tool_context, provider, tools, provider_retry_policy, tool_retry_policy, trace_id, root_span
                )
            else:
                result = await self._run_with_step_async(
                    state, tool_context, self._default_step, trace_id, root_span
                )
            self._emit_completion_events(trace_id, root_span, result[1])
            self._end_span(root_span, trace_id, self._status_for_result(result[1]))
            await self._aemit_terminal_agent_hook(trace_id, result[1])
            return result
        except asyncio.CancelledError as exc:
            try:
                self._record_event(
                    "agent.execution.failed", trace_id, root_span, "agent_loop", {"execution_id": trace_id}
                )
            except Exception:
                pass
            try:
                self._end_span(root_span, trace_id, SpanStatus.ERROR)
            except Exception:
                pass
            try:
                await self._aemit_hook(
                    HookEventType.AGENT_CANCELLED,
                    execution_id=trace_id,
                    metadata={"execution_id": trace_id},
                    error=exc,
                    error_message=f"{type(exc).__name__}",
                )
            except asyncio.CancelledError:
                # A new genuine task cancellation during cleanup hook dispatch
                # must not replace the original execution cancellation.
                pass
            except Exception:
                pass
            raise
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
            await self._aemit_hook(
                HookEventType.AGENT_FAILED,
                execution_id=trace_id,
                metadata={"status": res[1].status.value, "iterations": res[1].iterations},
                error=exc,
                error_message=res[1].error,
            )
            return res

# helper
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

# helper
    def _max_iterations(self, state, tool_context):
        return self._finish(
            state,
            ExecutionStatus.MAX_ITERATIONS_REACHED,
            tool_context,
            error=f"Maximum iterations reached ({self._config.max_iterations}).",
        )

# helper
    def _resolve_tools(self, tools):
        if tools is not None:
            return tools
        if self._tool_executor is not None:
            return self._tool_executor._registry.get_all_definitions() or None
        return None


# helper
    @staticmethod

    def _initial_state(self, execution_id, messages):
        return AgentState(
            execution_id=execution_id or str(uuid.uuid4()),
            messages=list(messages or []),
            iteration_count=0,
            status=ExecutionStatus.PENDING,
            metadata={},
        )

# helper
    def _new_tool_context(self):
        if self._tool_executor is None:
            return None
        return self._tool_executor.new_context()

# helper
    @staticmethod
    def _default_step(state):
        return IterationOutcome(decision=IterationDecision.COMPLETE)

# helper
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
  
    # helper
    def _start_span(self, trace_id, name, component, parent_span_id, attributes=None):
        span_id = str(uuid.uuid4())
        try:
            start = getattr(self._tracer, "start_span", None)
            if callable(start):
                start(trace_id, span_id, name, component, parent_span_id, attributes or {})
            return span_id
        except Exception:
            return span_id

    # helper
    def _end_span(self, span_id, trace_id, status):
        if not span_id or not trace_id:
            return
        try:
            end = getattr(self._tracer, "end_span", None)
            if callable(end):
                end(span_id, trace_id, status)
        except Exception:
            return

    # helper
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

    # helper
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

    # helper
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

    # helper
    @staticmethod
    def _status_for_result(result):
        if result.status in (ExecutionStatus.COMPLETED, ExecutionStatus.MAX_ITERATIONS_REACHED):
            # Defined terminal outcomes (including limit) treated as closed spans.
            return SpanStatus.OK
        return SpanStatus.ERROR
