
"""Lifecycle and execution-state helpers for AgentLoop."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable

from agent_orchestrator.core.models import ExecutionResult, ExecutionStatus
from agent_orchestrator.hooks import (
    HookEventType,
    LifecycleEvent,
    LifecycleHook,
)
from agent_orchestrator.state.models import AgentState
from agent_orchestrator.tools.executor import ExecutionContext


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


class LifecycleHelpersMixin:
    """Provides lifecycle, state, and execution-result helpers."""

    def _emit_hook(
        self,
        event_type: HookEventType,
        *,
        execution_id: str | None = None,
        iteration: int | None = None,
        metadata: dict | None = None,
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
            # Best-effort for sync path; preserve execution exceptions.
            return

    async def _aemit_hook(
        self,
        event_type: HookEventType,
        *,
        execution_id: str | None = None,
        iteration: int | None = None,
        metadata: dict | None = None,
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

    def _emit_terminal_agent_hook(
        self,
        execution_id: str,
        result: ExecutionResult,
    ) -> None:
        meta = {
            "status": result.status.value,
            "iterations": result.iterations,
            "tool_calls_count": result.tool_calls_count,
        }

        if result.status == ExecutionStatus.COMPLETED:
            self._emit_hook(
                HookEventType.AGENT_COMPLETED,
                execution_id=execution_id,
                metadata=meta,
            )
        else:
            self._emit_hook(
                HookEventType.AGENT_FAILED,
                execution_id=execution_id,
                metadata=meta,
                error_message=result.error,
            )

    async def _aemit_terminal_agent_hook(
        self,
        execution_id: str,
        result: ExecutionResult,
    ) -> None:
        meta = {
            "status": result.status.value,
            "iterations": result.iterations,
            "tool_calls_count": result.tool_calls_count,
        }

        if result.status == ExecutionStatus.COMPLETED:
            await self._aemit_hook(
                HookEventType.AGENT_COMPLETED,
                execution_id=execution_id,
                metadata=meta,
            )
        else:
            await self._aemit_hook(
                HookEventType.AGENT_FAILED,
                execution_id=execution_id,
                metadata=meta,
                error_message=result.error,
            )

    def _apply_outcome(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        outcome: IterationOutcome,
    ):
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
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
    ):
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

    def _initial_state(
        self,
        execution_id: str | None,
        messages: list[dict] | None,
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
    ):
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
