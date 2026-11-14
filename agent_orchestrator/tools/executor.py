"""Tool execution engine for the Agent Orchestrator.

Executes registered tool handlers after argument validation. Supports
synchronous and asynchronous execution (with timeouts) and per-session
execution limits while remaining independent of LLM providers. No
orchestration, retries, or parallel batch execution are implemented.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid 
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.exceptions import ExecutionLimitError, ToolNotFoundError, ToolValidationError
from agent_orchestrator.hooks import HookEventType, HookManager, LifecycleEvent, LifecycleHook
from agent_orchestrator.retry import RetryPolicy
from agent_orchestrator.tools.models import ToolCall, ToolResult
from agent_orchestrator.tools.registry import RegisteredTool, ToolRegistry
from agent_orchestrator.tools.validator import ToolValidator
from agent_orchestrator.tracing import ExecutionEvent, InMemoryTracer, NoOpTracer, SpanStatus, Tracer


@dataclass
class ExecutionContext:
    """Lightweight per-session execution state.

    Tracks the number of tool executions and enforces the configured maximum.
    Instances are independent and hold no global state.
    """

    max_tool_calls: int
    tool_calls_count: int = 0

    def consume(self) -> None:
        """Reserve one execution slot or raise if the limit is reached."""
        if self.tool_calls_count >= self.max_tool_calls:
            raise ExecutionLimitError(
                f"Maximum tool calls reached ({self.max_tool_calls})."
            )
        self.tool_calls_count += 1


class ToolExecutor:
    """Execute tool calls using a registry and optional validator.

    Arguments are validated before handler invocation. Failures (lookup,
    validation, handler errors, timeouts, execution limits) are captured as
    failed :class:`ToolResult` instances with preserved error information.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        validator: ToolValidator | None = None,
        config: OrchestratorConfig | None = None,
        tracer: Tracer | None = None,
        hooks: Sequence[LifecycleHook] | None = None,
    ) -> None:
        self._registry = registry
        self._validator = validator or ToolValidator()
        self._config = config or OrchestratorConfig()
        self.tracer: Tracer = tracer or (
            InMemoryTracer() if self._config.enable_tracing else NoOpTracer()
        )
        self._hook_manager = HookManager(hooks)

    @property
    def hooks(self) -> tuple[LifecycleHook, ...]:
        return self._hook_manager.hooks

    def new_context(self) -> ExecutionContext:
        """Start a fresh execution session with an independent counter."""
        return ExecutionContext(max_tool_calls=self._config.max_tool_calls)

    def execute(
        self,
        tool_call: ToolCall,
        context: ExecutionContext | None = None,
        *,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        execution_id: str | None = None,
        iteration: int | None = None,
        tool_retry_policy: RetryPolicy | None = None,
    ) -> ToolResult:
        """Execute a single tool call synchronously and return its result."""
        policy = self._resolve_tool_retry_policy(tool_retry_policy)
        return self._traced_execute(
            tool_call,
            context,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            execution_id=execution_id,
            iteration=iteration,
            invoke=lambda h, a: self._invoke_sync(h, a),
            policy=policy,
        )

    async def execute_async(
        self,
        tool_call: ToolCall,
        timeout: float | None = None,
        context: ExecutionContext | None = None,
        *,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        execution_id: str | None = None,
        iteration: int | None = None,
        tool_retry_policy: RetryPolicy | None = None,
    ) -> ToolResult:
        """Execute a single tool call asynchronously and return its result.

        Supports both synchronous and asynchronous handlers. Synchronous handlers
        run in a worker thread to avoid blocking the event loop. Async execution
        is bounded by ``timeout`` (seconds), defaulting to
        ``OrchestratorConfig.tool_timeout``.
        """
        effective_timeout = timeout if timeout is not None else self._config.tool_timeout
        policy = self._resolve_tool_retry_policy(tool_retry_policy)
        return await self._traced_execute_async(
            tool_call,
            context,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            execution_id=execution_id,
            iteration=iteration,
            timeout=effective_timeout,
            policy=policy,
        )

    def _resolve_tool_retry_policy(self, override: RetryPolicy | None) -> RetryPolicy:
        if override is not None:
            return override
        tool_policy = getattr(self._config, "tool_retry_policy", None)
        if tool_policy is not None:
            return tool_policy
        return getattr(self._config, "retry_policy", RetryPolicy())

    @staticmethod
    def _is_process_control_exception(exc: BaseException) -> bool:
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            return True
        name = type(exc).__name__
        return name in {"CancelledError", "TaskCancelledException"}

    @staticmethod
    def _is_non_retryable_tool_exception(exc: BaseException) -> bool:
        # Validation/schema/unknown-tool and other pre-execution or structural failures
        # must not be retried.
        return isinstance(exc, (ToolValidationError, ToolNotFoundError, ExecutionLimitError))

    def _should_retry_tool_exception(self, policy: RetryPolicy, attempt: int, exc: Exception) -> bool:
        if self._is_process_control_exception(exc):
            return False
        if self._is_non_retryable_tool_exception(exc):
            return False
        return policy.should_retry(attempt, exc)

    def _record_retry_event(
        self,
        event_type: str,
        trace_id: str | None,
        span_id: str | None,
        tool_call: ToolCall,
        execution_id: str | None,
        iteration: int | None,
        attempt: int,
        max_attempts: int,
        *,
        error: BaseException | None = None,
        delay: float | None = None,
        retry_number: int | None = None,
    ) -> None:
        extra: dict[str, Any] = {
            "attempt": attempt,
            "max_attempts": max_attempts,
        }
        if retry_number is not None:
            extra["retry_number"] = retry_number
        if delay is not None:
            extra["delay"] = delay
        if error is not None:
            extra["error_type"] = type(error).__name__
        self._record_tool_event(
            event_type,
            trace_id,
            span_id,
            tool_call,
            execution_id,
            iteration,
            extra,
        )

    def _emit_hook(
        self,
        event_type: HookEventType,
        *,
        execution_id: str | None,
        iteration: int | None = None,
        metadata: dict[str, Any] | None = None,
        error: BaseException | None = None,
        error_message: str | None = None,
    ) -> None:
        # Best-effort dispatch; hook manager swallows ordinary exceptions.
        # Execution process-control exceptions must not be suppressed by callers.
        try:
            self._hook_manager.emit(
                LifecycleEvent(
                    event_type=event_type,
                    execution_id=execution_id,
                    iteration=iteration,
                    component="tool_executor",
                    metadata=dict(metadata or {}),
                    error=error,
                    error_message=error_message,
                )
            )
        except BaseException:
            # Never swallow process-control/cancellation from hook dispatch in a
            # way that would mask execution flow is handled by call sites; keep
            # emission best-effort for unexpected BaseExceptions from hooks too.
            return

    async def _aemit_hook(
        self,
        event_type: HookEventType,
        *,
        execution_id: str | None,
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
                component="tool_executor",
                metadata=dict(metadata or {}),
                error=error,
                error_message=error_message,
            )
        )

    def _traced_execute(
        self,
        tool_call: ToolCall,
        context: ExecutionContext | None,
        *,
        trace_id: str | None,
        parent_span_id: str | None,
        execution_id: str | None,
        iteration: int | None,
        invoke: Callable[[Callable[..., Any], dict[str, Any]], Any],
        policy: RetryPolicy,
    ) -> ToolResult:
        span_id = self._start_tool_span(
            trace_id, parent_span_id, tool_call, execution_id, iteration
        )
        self._record_tool_event(
            "tool.execution.started",
            trace_id,
            span_id,
            tool_call,
            execution_id,
            iteration,
            {"success": None},
        )
        self._emit_hook(
            HookEventType.TOOL_EXECUTION_STARTED,
            execution_id=execution_id,
            iteration=iteration,
            metadata={"tool_name": tool_call.name, "tool_call_id": tool_call.id},
        )
        status = SpanStatus.OK
        result: ToolResult | None = None
        try:
            try:
                self._consume(context)
                registered = self._prepare(tool_call)
            except Exception as exc:
                status = self._status_for_exception(exc)
                self._record_tool_event(
                    self._failure_event_type(exc),
                    trace_id,
                    span_id,
                    tool_call,
                    execution_id,
                    iteration,
                    {"success": False, "error_type": type(exc).__name__},
                )
                result = self._failure(tool_call, exc)
                self._emit_hook(
                    HookEventType.TOOL_EXECUTION_FAILED,
                    execution_id=execution_id,
                    iteration=iteration,
                    metadata={
                        "tool_name": tool_call.name,
                        "tool_call_id": tool_call.id,
                        "success": False,
                        "error_type": type(exc).__name__,
                    },
                    error=exc,
                    error_message=result.error,
                )
            else:
                attempt = 1
                while True:
                    self._record_retry_event(
                        "tool.attempt.started",
                        trace_id,
                        span_id,
                        tool_call,
                        execution_id,
                        iteration,
                        attempt,
                        policy.max_attempts,
                    )
                    try:
                        output = invoke(registered.handler, tool_call.arguments)
                    except BaseException as exc:
                        self._record_retry_event(
                            "tool.attempt.failed",
                            trace_id,
                            span_id,
                            tool_call,
                            execution_id,
                            iteration,
                            attempt,
                            policy.max_attempts,
                            error=exc,
                        )
                        if self._is_process_control_exception(exc):
                            status = SpanStatus.ERROR
                            raise
                        if isinstance(exc, Exception) and self._should_retry_tool_exception(policy, attempt, exc):
                            delay = float(policy.next_delay(attempt))
                            self._record_retry_event(
                                "tool.retry.scheduled",
                                trace_id,
                                span_id,
                                tool_call,
                                execution_id,
                                iteration,
                                attempt,
                                policy.max_attempts,
                                error=exc,
                                delay=delay,
                                retry_number=attempt,
                            )
                            self._emit_hook(
                                HookEventType.TOOL_RETRY_SCHEDULED,
                                execution_id=execution_id,
                                iteration=iteration,
                                metadata={
                                    "tool_name": tool_call.name,
                                    "tool_call_id": tool_call.id,
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
                        if isinstance(exc, Exception):
                            status = self._status_for_exception(exc)
                            self._record_tool_event(
                                self._failure_event_type(exc),
                                trace_id,
                                span_id,
                                tool_call,
                                execution_id,
                                iteration,
                                {"success": False, "error_type": type(exc).__name__},
                            )
                            result = self._failure(tool_call, exc)
                            self._emit_hook(
                                HookEventType.TOOL_EXECUTION_FAILED,
                                execution_id=execution_id,
                                iteration=iteration,
                                metadata={
                                    "tool_name": tool_call.name,
                                    "tool_call_id": tool_call.id,
                                    "success": False,
                                    "error_type": type(exc).__name__,
                                },
                                error=exc,
                                error_message=result.error,
                            )
                            break
                        status = SpanStatus.ERROR
                        raise
                    else:
                        self._record_retry_event(
                            "tool.attempt.succeeded",
                            trace_id,
                            span_id,
                            tool_call,
                            execution_id,
                            iteration,
                            attempt,
                            policy.max_attempts,
                        )
                        self._record_tool_event(
                            "tool.execution.completed",
                            trace_id,
                            span_id,
                            tool_call,
                            execution_id,
                            iteration,
                            {"success": True},
                        )
                        result = self._success(tool_call, output)
                        self._emit_hook(
                            HookEventType.TOOL_EXECUTION_COMPLETED,
                            execution_id=execution_id,
                            iteration=iteration,
                            metadata={
                                "tool_name": tool_call.name,
                                "tool_call_id": tool_call.id,
                                "success": True,
                            },
                        )
                        break
        except BaseException:
            status = SpanStatus.ERROR
            raise
        finally:
            self._end_span(span_id, trace_id, status)
        assert result is not None
        return result

    async def _traced_execute_async(
        self,
        tool_call: ToolCall,
        context: ExecutionContext | None,
        *,
        trace_id: str | None,
        parent_span_id: str | None,
        execution_id: str | None,
        iteration: int | None,
        timeout: float,
        policy: RetryPolicy,
    ) -> ToolResult:
        span_id = self._start_tool_span(
            trace_id, parent_span_id, tool_call, execution_id, iteration
        )
        self._record_tool_event(
            "tool.execution.started",
            trace_id,
            span_id,
            tool_call,
            execution_id,
            iteration,
            {"success": None},
        )
        await self._aemit_hook(
            HookEventType.TOOL_EXECUTION_STARTED,
            execution_id=execution_id,
            iteration=iteration,
            metadata={"tool_name": tool_call.name, "tool_call_id": tool_call.id},
        )
        status = SpanStatus.OK
        result: ToolResult | None = None
        try:
            try:
                self._consume(context)
                registered = self._prepare(tool_call)
            except Exception as exc:
                status = self._status_for_exception(exc)
                self._record_tool_event(
                    self._failure_event_type(exc),
                    trace_id,
                    span_id,
                    tool_call,
                    execution_id,
                    iteration,
                    {"success": False, "error_type": type(exc).__name__},
                )
                result = self._failure(tool_call, exc)
                try:
                    await self._aemit_hook(
                        HookEventType.TOOL_EXECUTION_FAILED,
                        execution_id=execution_id,
                        iteration=iteration,
                        metadata={
                            "tool_name": tool_call.name,
                            "tool_call_id": tool_call.id,
                            "success": False,
                            "error_type": type(exc).__name__,
                        },
                        error=exc,
                        error_message=result.error,
                    )
                except asyncio.CancelledError:
                    status = SpanStatus.ERROR
                    raise
            else:
                attempt = 1
                while True:
                    self._record_retry_event(
                        "tool.attempt.started",
                        trace_id,
                        span_id,
                        tool_call,
                        execution_id,
                        iteration,
                        attempt,
                        policy.max_attempts,
                    )
                    try:
                        output = await self._invoke_async(
                            registered.handler, tool_call.arguments, timeout
                        )
                    except BaseException as exc:
                        self._record_retry_event(
                            "tool.attempt.failed",
                            trace_id,
                            span_id,
                            tool_call,
                            execution_id,
                            iteration,
                            attempt,
                            policy.max_attempts,
                            error=exc,
                        )
                        if self._is_process_control_exception(exc):
                            status = SpanStatus.ERROR
                            raise
                        if isinstance(exc, Exception) and self._should_retry_tool_exception(policy, attempt, exc):
                            delay = float(policy.next_delay(attempt))
                            self._record_retry_event(
                                "tool.retry.scheduled",
                                trace_id,
                                span_id,
                                tool_call,
                                execution_id,
                                iteration,
                                attempt,
                                policy.max_attempts,
                                error=exc,
                                delay=delay,
                                retry_number=attempt,
                            )
                            await self._aemit_hook(
                                HookEventType.TOOL_RETRY_SCHEDULED,
                                execution_id=execution_id,
                                iteration=iteration,
                                metadata={
                                    "tool_name": tool_call.name,
                                    "tool_call_id": tool_call.id,
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
                        if isinstance(exc, Exception):
                            status = self._status_for_exception(exc)
                            self._record_tool_event(
                                self._failure_event_type(exc),
                                trace_id,
                                span_id,
                                tool_call,
                                execution_id,
                                iteration,
                                {"success": False, "error_type": type(exc).__name__},
                            )
                            result = self._failure(tool_call, exc)
                            try:
                                await self._aemit_hook(
                                    HookEventType.TOOL_EXECUTION_FAILED,
                                    execution_id=execution_id,
                                    iteration=iteration,
                                    metadata={
                                        "tool_name": tool_call.name,
                                        "tool_call_id": tool_call.id,
                                        "success": False,
                                        "error_type": type(exc).__name__,
                                    },
                                    error=exc,
                                    error_message=result.error,
                                )
                            except asyncio.CancelledError:
                                status = SpanStatus.ERROR
                                raise
                            break
                        status = SpanStatus.ERROR
                        raise
                    else:
                        self._record_retry_event(
                            "tool.attempt.succeeded",
                            trace_id,
                            span_id,
                            tool_call,
                            execution_id,
                            iteration,
                            attempt,
                            policy.max_attempts,
                        )
                        self._record_tool_event(
                            "tool.execution.completed",
                            trace_id,
                            span_id,
                            tool_call,
                            execution_id,
                            iteration,
                            {"success": True},
                        )
                        result = self._success(tool_call, output)
                        await self._aemit_hook(
                            HookEventType.TOOL_EXECUTION_COMPLETED,
                            execution_id=execution_id,
                            iteration=iteration,
                            metadata={
                                "tool_name": tool_call.name,
                                "tool_call_id": tool_call.id,
                                "success": True,
                            },
                        )
                        break
        except BaseException:
            status = SpanStatus.ERROR
            raise
        finally:
            self._end_span(span_id, trace_id, status)
        assert result is not None
        return result

    @staticmethod
    def _invoke_sync(handler: Callable[..., Any], arguments: dict[str, Any]) -> Any:
        output = handler(**arguments)
        if inspect.isawaitable(output):
            if inspect.iscoroutine(output):
                output.close()
            raise TypeError("Async tool handlers require execute_async().")
        return output

    async def _invoke_async(
        self, handler: Callable[..., Any], arguments: dict[str, Any], timeout: float
    ) -> Any:
        if inspect.iscoroutinefunction(handler):
            return await asyncio.wait_for(handler(**arguments), timeout=timeout)
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, lambda: handler(**arguments))
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            future.cancel()
            raise

    @staticmethod
    def _consume(context: ExecutionContext | None) -> None:
        if context is not None:
            context.consume()

    def _prepare(self, tool_call: ToolCall) -> RegisteredTool:
        registered = self._registry.get(tool_call.name)
        self._validator.validate(registered.definition, tool_call.arguments)
        return registered

    @staticmethod
    def _success(tool_call: ToolCall, output: Any) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            output=output,
            success=True,
            error=None,
        )

    @staticmethod
    def _failure(tool_call: ToolCall, exc: Exception) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            output=None,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    @staticmethod
    def _failure_event_type(exc: Exception) -> str:
        if isinstance(exc, ToolValidationError):
            return "tool.validation.failed"
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return "tool.execution.timeout"
        return "tool.execution.failed"

    @staticmethod
    def _status_for_exception(exc: Exception) -> SpanStatus:
        return SpanStatus.ERROR

    def _start_tool_span(
        self,
        trace_id: str | None,
        parent_span_id: str | None,
        tool_call: ToolCall,
        execution_id: str | None,
        iteration: int | None,
    ) -> str | None:
        if trace_id is None:
            return None
        span_id = str(uuid.uuid4())
        attrs = {
            "tool_name": tool_call.name,
            "tool_call_id": tool_call.id,
        }
        if execution_id is not None:
            attrs["execution_id"] = execution_id
        if iteration is not None:
            attrs["iteration"] = iteration
        self._safe_call(
            lambda: getattr(self.tracer, "start_span", lambda *a, **k: None)(
                trace_id, span_id, "tool.execute", "tool_executor", parent_span_id, attrs
            )
        )
        return span_id

    def _end_span(self, span_id: str | None, trace_id: str | None, status: SpanStatus) -> None:
        if trace_id is None or span_id is None:
            return
        self._safe_call(
            lambda: getattr(self.tracer, "end_span", lambda *a, **k: None)(
                span_id, trace_id, status
            )
        )

    def _record_tool_event(
        self,
        event_type: str,
        trace_id: str | None,
        span_id: str | None,
        tool_call: ToolCall,
        execution_id: str | None,
        iteration: int | None,
        extra: dict[str, Any],
    ) -> None:
        if trace_id is None or span_id is None:
            return
        metadata: dict[str, Any] = {
            "tool_name": tool_call.name,
            "tool_call_id": tool_call.id,
        }
        if execution_id is not None:
            metadata["execution_id"] = execution_id
        if iteration is not None:
            metadata["iteration"] = iteration
        metadata.update(extra)
        event = ExecutionEvent(
            event_type=event_type,
            trace_id=trace_id,
            span_id=span_id,
            component="tool_executor",
            metadata=metadata,
        )
        self._safe_call(lambda: self.tracer.record(event))

    @staticmethod
    def _safe_call(fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception:
            return None