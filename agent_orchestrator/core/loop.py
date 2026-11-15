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
    IterationDecision,
    IterationOutcome,
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
        # Create a new execution state using the supplied messages and execution ID.
        state = self._initial_state(execution_id, messages)

        # Create a separate tool execution context for this agent execution.
        tool_context = self._new_tool_context()

        # Mark the execution as RUNNING because the agent is about to start working.
        state.status = ExecutionStatus.RUNNING

        # Use the execution ID as the trace ID to connect all tracing events and spans.
        trace_id = state.execution_id

        # Start the root tracing span representing the entire agent execution.
        root_span = self._start_span(
            trace_id, "agent.execution", "agent_loop", None, {"execution_id": trace_id}
        )

        # Record an event indicating that the agent execution has started.
        self._record_event(
            "agent.execution.started",
            trace_id,
            root_span,
            "agent_loop",
            {"execution_id": trace_id},
        )

        # Notify registered lifecycle hooks that a new agent execution has started.
        self._emit_hook(
            HookEventType.AGENT_STARTED,
            execution_id=trace_id,
            metadata={"execution_id": trace_id},
        )

        try:
            # If the caller provides a custom step, use it as the execution strategy.
            # This takes priority even if a provider is also supplied.
            if step is not None:
                result = self._run_with_step(
                    state, tool_context, step, trace_id, root_span
                )

            # Otherwise, if a provider is supplied, use the LLM/tool execution flow.
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

            # If neither a custom step nor a provider is supplied,
            # use the default step, which immediately completes the execution.
            else:
                result = self._run_with_step(
                    state, tool_context, self._default_step, trace_id, root_span
                )

            # Emit tracing events describing the final execution result.
            self._emit_completion_events(trace_id, root_span, result[1])

            # Close the root span using the appropriate status (OK or ERROR).
            self._end_span(root_span, trace_id, self._status_for_result(result[1]))

            # Notify lifecycle hooks that the entire agent execution has finished.
            self._emit_terminal_agent_hook(trace_id, result[1])

            # Return the final agent state and execution result to the caller.
            return result

        except Exception as exc:
            # An unexpected exception occurred.
            # Convert it into a FAILED execution result.
            res = self._finish(
                state,
                ExecutionStatus.FAILED,
                tool_context,
                error=f"{type(exc).__name__}: {exc}",
            )

            # Record that the overall agent execution failed.
            self._record_event(
                "agent.execution.failed",
                trace_id,
                root_span,
                "agent_loop",
                {"execution_id": trace_id},
            )

            # Mark the root tracing span as failed.
            self._end_span(root_span, trace_id, SpanStatus.ERROR)

            # Notify registered hooks about the failed execution.
            self._emit_hook(
                HookEventType.AGENT_FAILED,
                execution_id=trace_id,
                metadata={"status": res[1].status.value, "iterations": res[1].iterations},
                error=exc,
                error_message=res[1].error,
            )

            # Return the failed execution result instead of propagating the exception.
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
        # Initialize a new execution state, just like the synchronous run().
        state = self._initial_state(execution_id, messages)

        # Create an independent tool context for this execution.
        tool_context = self._new_tool_context()

        # Mark this execution as RUNNING.
        state.status = ExecutionStatus.RUNNING

        # Use the execution ID as the trace ID.
        trace_id = state.execution_id

        # Start the root span for the entire asynchronous agent execution.
        root_span = self._start_span(
            trace_id, "agent.execution", "agent_loop", None, {"execution_id": trace_id}
        )

        # Record that the agent execution has started.
        self._record_event(
            "agent.execution.started",
            trace_id,
            root_span,
            "agent_loop",
            {"execution_id": trace_id},
        )

        # Dispatch the start hook asynchronously.
        # Await it because hook execution may itself involve asynchronous work.
        await self._aemit_hook(
            HookEventType.AGENT_STARTED,
            execution_id=trace_id,
            metadata={"execution_id": trace_id},
        )

        try:
            # A caller-provided step takes priority over the provider.
            # The async step runner supports synchronous or asynchronous step functions.
            if step is not None:
                result = await self._run_with_step_async(
                    state, tool_context, step, trace_id, root_span
                )

            # If a provider is supplied, use the asynchronous provider execution flow.
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

            # If no step or provider is supplied, execute the default completion step.
            else:
                result = await self._run_with_step_async(
                    state, tool_context, self._default_step, trace_id, root_span
                )

            # Record tracing events for the final execution result.
            self._emit_completion_events(trace_id, root_span, result[1])

            # Close the root span with the final execution status.
            self._end_span(root_span, trace_id, self._status_for_result(result[1]))

            # Dispatch the terminal lifecycle hook asynchronously.
            await self._aemit_terminal_agent_hook(trace_id, result[1])

            # Return the final state and result.
            return result

        except asyncio.CancelledError as exc:
            # The caller cancelled this async execution.
            # Cancellation is different from an ordinary failure:
            # do not convert it into a FAILED result.

            try:
                # Best-effort recording of the interrupted execution.
                self._record_event(
                    "agent.execution.failed",
                    trace_id,
                    root_span,
                    "agent_loop",
                    {"execution_id": trace_id},
                )
            except Exception:
                # Tracing cleanup must not prevent cancellation from propagating.
                pass

            try:
                # Mark the root span as failed because execution did not finish.
                self._end_span(root_span, trace_id, SpanStatus.ERROR)
            except Exception:
                # Even if tracing cleanup fails, continue cancellation handling.
                pass

            try:
                # Notify lifecycle hooks that this execution was cancelled.
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
                # Hook cleanup errors must not hide the original cancellation.
                pass

            # Re-raise cancellation so the caller knows the task was cancelled.
            raise

        except Exception as exc:
            # An ordinary execution error occurred.
            # Convert it into a FAILED execution result.
            res = self._finish(
                state,
                ExecutionStatus.FAILED,
                tool_context,
                error=f"{type(exc).__name__}: {exc}",
            )

            # Record the overall execution failure.
            self._record_event(
                "agent.execution.failed",
                trace_id,
                root_span,
                "agent_loop",
                {"execution_id": trace_id},
            )

            # Close the root span with ERROR status.
            self._end_span(root_span, trace_id, SpanStatus.ERROR)

            # Notify lifecycle hooks asynchronously about the failure.
            await self._aemit_hook(
                HookEventType.AGENT_FAILED,
                execution_id=trace_id,
                metadata={"status": res[1].status.value, "iterations": res[1].iterations},
                error=exc,
                error_message=res[1].error,
            )

            # Return the failed execution result.
            return res
