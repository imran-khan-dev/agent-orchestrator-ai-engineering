
"""Custom-step iteration runners for AgentLoop."""

import asyncio
import inspect

from agent_orchestrator.core.models import ExecutionResult, ExecutionStatus
from agent_orchestrator.hooks import HookEventType
from agent_orchestrator.state.models import AgentState
from agent_orchestrator.tools.executor import ExecutionContext
from agent_orchestrator.tracing import SpanStatus


class StepRunnerMixin:
    """Provides synchronous and asynchronous custom-step execution."""

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
            self._emit_hook(
                HookEventType.ITERATION_STARTED,
                execution_id=trace_id,
                iteration=state.iteration_count,
                metadata={"iteration": state.iteration_count},
            )
            try:
                outcome = step_fn(state)
                terminal = self._apply_outcome(state, tool_context, outcome)
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
                        metadata={
                            "iteration": state.iteration_count,
                            "status": terminal[1].status.value,
                        },
                        error_message=terminal[1].error,
                    )
                else:
                    self._end_span(
                        iter_span, trace_id, self._status_for_result(terminal[1])
                    )
                    self._emit_hook(
                        HookEventType.ITERATION_COMPLETED,
                        execution_id=trace_id,
                        iteration=state.iteration_count,
                        metadata={
                            "iteration": state.iteration_count,
                            "status": terminal[1].status.value,
                        },
                    )
                return terminal

            self._end_span(iter_span, trace_id, SpanStatus.OK)
            self._emit_hook(
                HookEventType.ITERATION_COMPLETED,
                execution_id=trace_id,
                iteration=state.iteration_count,
                metadata={"iteration": state.iteration_count},
            )

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
            await self._aemit_hook(
                HookEventType.ITERATION_STARTED,
                execution_id=trace_id,
                iteration=state.iteration_count,
                metadata={"iteration": state.iteration_count},
            )
            try:
                outcome = step_fn(state)
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                terminal = self._apply_outcome(state, tool_context, outcome)
            except asyncio.CancelledError as exc:
                self._end_span(iter_span, trace_id, SpanStatus.ERROR)
                try:
                    await self._aemit_hook(
                        HookEventType.ITERATION_FAILED,
                        execution_id=trace_id,
                        iteration=state.iteration_count,
                        metadata={
                            "iteration": state.iteration_count,
                            "cancelled": True,
                        },
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
                        metadata={
                            "iteration": state.iteration_count,
                            "status": terminal[1].status.value,
                        },
                        error_message=terminal[1].error,
                    )
                else:
                    self._end_span(
                        iter_span, trace_id, self._status_for_result(terminal[1])
                    )
                    await self._aemit_hook(
                        HookEventType.ITERATION_COMPLETED,
                        execution_id=trace_id,
                        iteration=state.iteration_count,
                        metadata={
                            "iteration": state.iteration_count,
                            "status": terminal[1].status.value,
                        },
                    )
                return terminal

            self._end_span(iter_span, trace_id, SpanStatus.OK)
            await self._aemit_hook(
                HookEventType.ITERATION_COMPLETED,
                execution_id=trace_id,
                iteration=state.iteration_count,
                metadata={"iteration": state.iteration_count},
            )
