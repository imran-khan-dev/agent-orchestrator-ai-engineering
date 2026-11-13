"""Lifecycle hook contracts for the Agent Orchestrator.

Hooks provide user-defined reactions to execution lifecycle events. They are
distinct from tracing: tracing is for observability data collection, while hooks
are for lightweight callbacks/side effects at defined execution boundaries.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Protocol, Sequence, runtime_checkable


class HookEventType(str, Enum):
    """Lifecycle event types emitted by the orchestrator."""

    AGENT_STARTED = "AGENT_STARTED"
    AGENT_COMPLETED = "AGENT_COMPLETED"
    AGENT_FAILED = "AGENT_FAILED"
    AGENT_CANCELLED = "AGENT_CANCELLED"

    ITERATION_STARTED = "ITERATION_STARTED"
    ITERATION_COMPLETED = "ITERATION_COMPLETED"
    ITERATION_FAILED = "ITERATION_FAILED"

    PROVIDER_REQUEST_STARTED = "PROVIDER_REQUEST_STARTED"
    PROVIDER_RESPONSE_RECEIVED = "PROVIDER_RESPONSE_RECEIVED"
    PROVIDER_REQUEST_FAILED = "PROVIDER_REQUEST_FAILED"
    PROVIDER_RETRY_SCHEDULED = "PROVIDER_RETRY_SCHEDULED"

    TOOL_EXECUTION_STARTED = "TOOL_EXECUTION_STARTED"
    TOOL_EXECUTION_COMPLETED = "TOOL_EXECUTION_COMPLETED"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    TOOL_RETRY_SCHEDULED = "TOOL_RETRY_SCHEDULED"


@dataclass(frozen=True)
class LifecycleEvent:
    """Immutable payload for a lifecycle hook event."""

    event_type: HookEventType
    execution_id: str | None = None
    iteration: int | None = None
    component: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    error: BaseException | None = None
    error_message: str | None = None


@runtime_checkable
class LifecycleHook(Protocol):
    """Protocol for lifecycle hooks."""

    def on_event(self, event: LifecycleEvent) -> Any | Awaitable[Any]:
        """Handle a lifecycle event."""
        ...


def _is_process_control(exc: BaseException) -> bool:
    return isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit))


class HookManager:
    """Ordered lifecycle hook dispatcher.

    Default policy is best-effort: ordinary hook exceptions do not interrupt
    execution. Process-control exceptions from hook execution are not treated as
    ordinary hook failures and may propagate unless callers preserve an original
    execution exception/cancellation.
    """

    def __init__(self, hooks: Sequence[LifecycleHook] | None = None) -> None:
        self._hooks: list[LifecycleHook] = list(hooks or [])

    @property
    def hooks(self) -> tuple[LifecycleHook, ...]:
        return tuple(self._hooks)

    def emit(self, event: LifecycleEvent) -> None:
        """Emit event to synchronous-capable hooks (best-effort).

        Synchronous execution does not execute awaitable hook results. Coroutine
        objects returned by hooks are explicitly closed (best-effort) to avoid
        unawaited coroutine warnings/leaks. Custom non-coroutine awaitables
        cannot be generically closed/disposed and are not executed.
        """
        for hook in self._hooks:
            try:
                result = hook.on_event(event)
                if inspect.isawaitable(result):
                    # Explicit and predictable: do not leave unawaited coroutine.
                    if inspect.iscoroutine(result):
                        result.close()
                    # Not executed in sync path (custom awaitables cannot be
                    # generically closed/disposed).
                    continue
            except Exception:
                # Ordinary hook exceptions must not interrupt execution.
                continue

    async def aemit(self, event: LifecycleEvent) -> None:
        """Emit event supporting sync and async hooks (best-effort).

        Genuine cancellation of the currently executing task while awaiting a
        hook propagates immediately. A hook that independently raises
        ``asyncio.CancelledError`` when the current task is not cancelling is
        treated as a best-effort hook failure.
        """
        task = asyncio.current_task()
        for hook in self._hooks:
            try:
                result = hook.on_event(event)
                if inspect.isawaitable(result):
                    cancelling_before = task.cancelling() if task is not None and hasattr(task, "cancelling") else 0
                    try:
                        await result
                    except asyncio.CancelledError:
                        cancelling_after = task.cancelling() if task is not None and hasattr(task, "cancelling") else 0
                        # Genuine task cancellation: propagate immediately.
                        if task is not None and cancelling_after > cancelling_before:
                            raise
                        # Hook independently raised CancelledError without a
                        # new task cancellation request: best-effort hook failure.
                        continue
            except asyncio.CancelledError:
                raise
            except Exception:
                # Ordinary hook exceptions must not interrupt execution.
                continue
