"""Tool execution engine for the Agent Orchestrator.

Executes registered tool handlers after argument validation. Supports
synchronous and asynchronous execution (with timeouts) and per-session
execution limits while remaining independent of LLM providers. No
orchestration, retries, or parallel batch execution are implemented.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.exceptions import ExecutionLimitError
from agent_orchestrator.tools.models import ToolCall, ToolResult
from agent_orchestrator.tools.registry import RegisteredTool, ToolRegistry
from agent_orchestrator.tools.validator import ToolValidator


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
    ) -> None:
        self._registry = registry
        self._validator = validator or ToolValidator()
        self._config = config or OrchestratorConfig()

    def new_context(self) -> ExecutionContext:
        """Start a fresh execution session with an independent counter."""
        return ExecutionContext(max_tool_calls=self._config.max_tool_calls)

    def execute(
        self, tool_call: ToolCall, context: ExecutionContext | None = None
    ) -> ToolResult:
        """Execute a single tool call synchronously and return its result."""
        try:
            self._consume(context)
            registered = self._prepare(tool_call)
            output = registered.handler(**tool_call.arguments)
            if inspect.isawaitable(output):
                if inspect.iscoroutine(output):
                    output.close()
                raise TypeError(
                    "Async tool handlers require execute_async()."
                )
            return self._success(tool_call, output)
        except Exception as exc:
            return self._failure(tool_call, exc)

    async def execute_async(
        self,
        tool_call: ToolCall,
        timeout: float | None = None,
        context: ExecutionContext | None = None,
    ) -> ToolResult:
        """Execute a single tool call asynchronously and return its result.

        Supports both synchronous and asynchronous handlers. Synchronous handlers
        run in a worker thread to avoid blocking the event loop. Async execution
        is bounded by ``timeout`` (seconds), defaulting to
        ``OrchestratorConfig.tool_timeout``.
        """
        try:
            self._consume(context)
            registered = self._prepare(tool_call)
            effective_timeout = (
                timeout if timeout is not None else self._config.tool_timeout
            )
            output = await self._invoke_async(
                registered.handler, tool_call.arguments, effective_timeout
            )
            return self._success(tool_call, output)
        except asyncio.TimeoutError as exc:
            return self._failure(tool_call, exc)
        except Exception as exc:
            return self._failure(tool_call, exc)

    @staticmethod
    def _consume(context: ExecutionContext | None) -> None:
        if context is not None:
            context.consume()

    def _prepare(self, tool_call: ToolCall) -> RegisteredTool:
        registered = self._registry.get(tool_call.name)
        self._validator.validate(registered.definition, tool_call.arguments)
        return registered

    async def _invoke_async(
        self, handler: Callable[..., Any], arguments: dict[str, Any], timeout: float
    ) -> Any:
        if inspect.iscoroutinefunction(handler):
            # asyncio.wait_for cancels the awaited coroutine/future on timeout.
            return await asyncio.wait_for(handler(**arguments), timeout=timeout)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, lambda: handler(**arguments))
        return await asyncio.wait_for(future, timeout=timeout)

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