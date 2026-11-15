"""Agent execution loop foundation.

Provides lifecycle management, iteration tracking, and termination handling for
agent runs. Supports provider-independent provider/tool round trips when an
LLM provider is supplied. No concrete provider SDKs are used.
"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.core.lifecycle_helpers import (
    AsyncIterationStep,
    IterationStep,
    LifecycleHelpersMixin,
)
from agent_orchestrator.core.models import ExecutionResult, ExecutionStatus
from agent_orchestrator.core.provider_runner import ProviderRunnerMixin
from agent_orchestrator.core.response_handler import ResponseHandlerMixin
from agent_orchestrator.core.response_normalizer import ResponseNormalizerMixin
from agent_orchestrator.core.step_runner import StepRunnerMixin
from agent_orchestrator.core.tracing_helpers import TracingHelpersMixin
from agent_orchestrator.hooks import HookEventType, HookManager, LifecycleHook
from agent_orchestrator.providers.base import LLMProvider
from agent_orchestrator.retry import RetryPolicy
from agent_orchestrator.state.models import AgentState
from agent_orchestrator.tools.executor import ToolExecutor
from agent_orchestrator.tools.models import ToolDefinition
from agent_orchestrator.tracing import InMemoryTracer, NoOpTracer, SpanStatus, Tracer


class AgentLoop(
    StepRunnerMixin,
    ProviderRunnerMixin,
    ResponseHandlerMixin,
    ResponseNormalizerMixin,
    LifecycleHelpersMixin,
    TracingHelpersMixin,
):
    """Reusable agent execution loop.

    Manages execution state transitions, iteration counting, maximum-iteration
    enforcement, and structured results.

    When a provider is supplied, performs provider-independent tool-calling
    round trips using :class:`ToolExecutor`. A custom iteration step may be
    supplied instead for lifecycle testing.
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
        state = self._initial_state(execution_id, messages)
        tool_context = self._new_tool_context()
        state.status = ExecutionStatus.RUNNING
        trace_id = state.execution_id
        root_span = self._start_span(
            trace_id, "agent.execution", "agent_loop", None, {"execution_id": trace_id}
        )
        self._record_event(
            "agent.execution.started",
            trace_id,
            root_span,
            "agent_loop",
            {"execution_id": trace_id},
        )
        self._emit_hook(
            HookEventType.AGENT_STARTED,
            execution_id=trace_id,
            metadata={"execution_id": trace_id},
        )

        try:
            if step is not None:
                # why run with stp?
                result = self._run_with_step(
                    state, tool_context, step, trace_id, root_span
                )
            elif provider is not None:
                result = self._run_with_provider_sync(
                    state,
                    tool_context,
                    provider,
                    tools,
                    provider_retry_policy,
                    tool_retry_policy,
                    trace_id,
                    root_span,
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
                "agent.execution.failed",
                trace_id,
                root_span,
                "agent_loop",
                {"execution_id": trace_id},
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
        supplied, ``complete_async`` is used if available; otherwise ``complete``
        runs in a worker thread to avoid blocking the event loop.
        """
        state = self._initial_state(execution_id, messages)
        tool_context = self._new_tool_context()
        state.status = ExecutionStatus.RUNNING
        trace_id = state.execution_id
        root_span = self._start_span(
            trace_id, "agent.execution", "agent_loop", None, {"execution_id": trace_id}
        )
        self._record_event(
            "agent.execution.started",
            trace_id,
            root_span,
            "agent_loop",
            {"execution_id": trace_id},
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
                    state,
                    tool_context,
                    provider,
                    tools,
                    provider_retry_policy,
                    tool_retry_policy,
                    trace_id,
                    root_span,
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
                    "agent.execution.failed",
                    trace_id,
                    root_span,
                    "agent_loop",
                    {"execution_id": trace_id},
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
                "agent.execution.failed",
                trace_id,
                root_span,
                "agent_loop",
                {"execution_id": trace_id},
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
