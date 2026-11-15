
"""Custom-step iteration runners for AgentLoop."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from agent_orchestrator.core.models import ExecutionResult, ExecutionStatus
from agent_orchestrator.hooks import HookEventType
from agent_orchestrator.state.models import AgentState
from agent_orchestrator.tools.executor import ExecutionContext
from agent_orchestrator.tracing import SpanStatus

IterationStep = Callable[[AgentState], Any]
AsyncIterationStep = Callable[[AgentState], Awaitable[Any]]


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
        # Stop execution if the maximum allowed iterations have been reached.
        # This prevents the agent from running forever.
        if state.iteration_count >= self._config.max_iterations:
            return self._max_iterations(state, tool_context)

        # Increment the counter because a new iteration is starting.
        state.iteration_count += 1

        # Start a tracing span for this iteration.
        # This lets us monitor the duration and status of each iteration.
        iter_span = self._start_span(
            trace_id,
            "iteration",
            "agent_loop",
            parent_span_id,
            {"execution_id": trace_id, "iteration": state.iteration_count},
        )

        # Record an event showing that this iteration has started.
        # Spans track the duration of work; events record what happened.
        self._record_event(
            "agent.iteration.started",
            trace_id,
            iter_span,
            "agent_loop",
            {"execution_id": trace_id, "iteration": state.iteration_count},
        )

        # Notify registered lifecycle hooks that a new iteration has started.
        # Other components can use this event for logging or monitoring.
        self._emit_hook(
            HookEventType.ITERATION_STARTED,
            execution_id=trace_id,
            iteration=state.iteration_count,
            metadata={"iteration": state.iteration_count},
        )

        try:
            # Execute the actual work for this iteration.
            # step_fn receives the current agent state and returns an outcome.
            outcome = step_fn(state)

            # Interpret the outcome and update the agent state.
            # Returns None if execution should continue, or a result tuple if finished.
            terminal = self._apply_outcome(state, tool_context, outcome)

        except Exception as exc:
            # Mark the current iteration's tracing span as failed.
            self._end_span(iter_span, trace_id, SpanStatus.ERROR)

            # Notify registered hooks that the iteration failed.
            # Include the exception so listeners can inspect or log the error.
            self._emit_hook(
                HookEventType.ITERATION_FAILED,
                execution_id=trace_id,
                iteration=state.iteration_count,
                metadata={"iteration": state.iteration_count},
                error=exc,
                error_message=f"{type(exc).__name__}: {exc}",
            )

            # Pass the exception to the outer AgentLoop.run() method.
            # The outer method decides how the overall execution should handle it.
            raise

        # Check whether the outcome indicates that the entire execution has finished.
        if terminal is not None:

            # Handle a terminal result that represents failure.
            if terminal[1].status == ExecutionStatus.FAILED:

                # Mark the iteration's tracing span as failed.
                self._end_span(iter_span, trace_id, SpanStatus.ERROR)

                # Notify listeners that the iteration ended in failure.
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
                # Mark the iteration span with the appropriate final status.
                self._end_span(
                    iter_span, trace_id, self._status_for_result(terminal[1])
                )

                # Notify listeners that the iteration completed successfully.
                self._emit_hook(
                    HookEventType.ITERATION_COMPLETED,
                    execution_id=trace_id,
                    iteration=state.iteration_count,
                    metadata={
                        "iteration": state.iteration_count,
                        "status": terminal[1].status.value,
                    },
                )

            # Return the final agent state and execution result.
            # Returning stops the while loop and hands control back to the caller.
            return terminal

        # No terminal result means the agent should continue to another iteration.
        # Close the current iteration's tracing span successfully.
        self._end_span(iter_span, trace_id, SpanStatus.OK)

        # Notify listeners that this iteration finished.
        # This does not mean the entire agent execution has finished.
        self._emit_hook(
            HookEventType.ITERATION_COMPLETED,
            execution_id=trace_id,
            iteration=state.iteration_count,
            metadata={"iteration": state.iteration_count},
        )

        # Reaching the end of the while loop automatically starts the next iteration.
    
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
