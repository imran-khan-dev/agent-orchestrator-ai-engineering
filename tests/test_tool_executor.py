"""Unit tests for ToolExecutor."""

import asyncio
import time

import pytest

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.exceptions import ToolNotFoundError, ToolValidationError
from agent_orchestrator.tools.executor import ToolExecutor
from agent_orchestrator.tools.models import ToolCall, ToolDefinition
from agent_orchestrator.tools.registry import ToolRegistry
from agent_orchestrator.tools.validator import ToolValidator


def _def(name: str = "add") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} tool",
        parameters_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    )


def add(a: int, b: int) -> int:
    return a + b


def make_executor(registry: ToolRegistry | None = None) -> tuple[ToolExecutor, ToolRegistry]:
    reg = registry or ToolRegistry()
    return ToolExecutor(reg), reg


class TestToolExecutorSuccess:
    def test_successful_handler_execution(self):
        executor, reg = make_executor()
        reg.register(_def("add"), add)
        result = executor.execute(ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3}))
        assert result.success is True
        assert result.output == 5
        assert result.error is None
        assert result.tool_call_id == "c1"
        assert result.tool_name == "add"

    def test_arguments_passed_correctly_to_handlers(self):
        received = {}

        def capture(**kwargs):
            received.update(kwargs)
            return "ok"

        executor, reg = make_executor()
        reg.register(
            ToolDefinition(
                name="cap",
                description="cap",
                parameters_schema={
                    "type": "object",
                    "properties": {"x": {"type": "string"}, "y": {"type": "integer"}},
                    "required": ["x", "y"],
                },
            ),
            capture,
        )
        result = executor.execute(ToolCall(id="c2", name="cap", arguments={"x": "hi", "y": 7}))
        assert result.success is True
        assert received == {"x": "hi", "y": 7}

    def test_multiple_independent_tool_executions(self):
        executor, reg = make_executor()
        reg.register(_def("add"), add)

        def mul(a: int, b: int) -> int:
            return a * b

        reg.register(
            ToolDefinition(
                name="mul",
                description="mul",
                parameters_schema={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            ),
            mul,
        )
        r1 = executor.execute(ToolCall(id="1", name="add", arguments={"a": 1, "b": 2}))
        r2 = executor.execute(ToolCall(id="2", name="mul", arguments={"a": 3, "b": 4}))
        assert r1.success is True and r1.output == 3
        assert r2.success is True and r2.output == 12
        assert r1.tool_call_id == "1" and r2.tool_call_id == "2"


class TestToolExecutorFailures:
    def test_invalid_arguments_prevent_handler_execution(self):
        called = {"v": False}

        def handler(a: int, b: int) -> int:
            called["v"] = True
            return a + b

        executor, reg = make_executor()
        reg.register(_def("add"), handler)
        result = executor.execute(ToolCall(id="c1", name="add", arguments={"a": "no", "b": 2}))
        assert result.success is False
        assert called["v"] is False
        assert result.tool_call_id == "c1"
        assert result.tool_name == "add"
        assert result.output is None
        assert result.error is not None
        assert "ToolValidationError" in result.error

    def test_unknown_tool_name(self):
        executor, _ = make_executor()
        result = executor.execute(ToolCall(id="c1", name="missing", arguments={}))
        assert result.success is False
        assert result.tool_call_id == "c1"
        assert result.tool_name == "missing"
        assert result.output is None
        assert "ToolNotFoundError" in result.error
        assert "missing" in result.error

    def test_handler_exception_returns_failed_result(self):
        def boom(a: int, b: int) -> int:
            raise ValueError("boom")

        executor, reg = make_executor()
        reg.register(_def("add"), boom)
        result = executor.execute(ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}))
        assert result.success is False
        assert result.output is None
        assert result.tool_call_id == "c1"
        assert result.tool_name == "add"
        assert "ValueError" in result.error
        assert "boom" in result.error

    def test_accepts_explicit_validator(self):
        executor, reg = make_executor()
        reg.register(_def("add"), add)
        exec2 = ToolExecutor(reg, validator=ToolValidator())
        result = exec2.execute(ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}))
        assert result.success is True and result.output == 3
class TestToolExecutorAsync:
    def test_async_handler_execution(self):
        async def add_async(a: int, b: int) -> int:
            await asyncio.sleep(0)
            return a + b

        executor, reg = make_executor()
        reg.register(_def("add"), add_async)
        result = asyncio.run(
            executor.execute_async(ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3}))
        )
        assert result.success is True
        assert result.output == 5
        assert result.error is None
        assert result.tool_call_id == "c1"
        assert result.tool_name == "add"

    def test_sync_handler_through_execute_async(self):
        executor, reg = make_executor()
        reg.register(_def("add"), add)
        result = asyncio.run(
            executor.execute_async(ToolCall(id="c1", name="add", arguments={"a": 1, "b": 4}))
        )
        assert result.success is True
        assert result.output == 5

    def test_async_handler_exception(self):
        async def boom(a: int, b: int) -> int:
            await asyncio.sleep(0)
            raise ValueError("async-boom")

        executor, reg = make_executor()
        reg.register(_def("add"), boom)
        result = asyncio.run(
            executor.execute_async(ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}))
        )
        assert result.success is False
        assert result.output is None
        assert result.tool_call_id == "c1"
        assert result.tool_name == "add"
        assert "ValueError" in result.error
        assert "async-boom" in result.error

    def test_sync_handler_exception_through_execute_async(self):
        def boom(a: int, b: int) -> int:
            raise RuntimeError("sync-boom")

        executor, reg = make_executor()
        reg.register(_def("add"), boom)
        result = asyncio.run(
            executor.execute_async(ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}))
        )
        assert result.success is False
        assert "RuntimeError" in result.error
        assert "sync-boom" in result.error

    def test_async_handler_timeout(self):
        async def slow(a: int, b: int) -> int:
            await asyncio.sleep(1.0)
            return a + b

        executor, reg = make_executor()
        reg.register(_def("add"), slow)
        result = asyncio.run(
            executor.execute_async(
                ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}), timeout=0.05
            )
        )
        assert result.success is False
        assert result.output is None
        assert result.tool_call_id == "c1"
        assert result.tool_name == "add"
        assert "TimeoutError" in result.error

    def test_async_cancellation_on_timeout(self):
        cancelled = asyncio.Event()

        async def slow(a: int, b: int) -> int:
            try:
                await asyncio.sleep(1.0)
                return a + b
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def run():
            executor, reg = make_executor()
            reg.register(_def("add"), slow)
            result = await executor.execute_async(
                ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}), timeout=0.05
            )
            await asyncio.sleep(0)  # let cancellation propagate
            return result

        result = asyncio.run(run())
        assert result.success is False
        assert "TimeoutError" in result.error
        assert cancelled.is_set()

    def test_event_loop_responsive_with_sync_handler(self):
        def blocking(a: int, b: int) -> int:
            time.sleep(0.2)
            return a + b

        async def run():
            executor, reg = make_executor()
            reg.register(_def("add"), blocking)
            ticks = 0

            async def ticker():
                nonlocal ticks
                for _ in range(5):
                    await asyncio.sleep(0.02)
                    ticks += 1

            exec_task = asyncio.create_task(
                executor.execute_async(ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}))
            )
            tick_task = asyncio.create_task(ticker())
            result = await exec_task
            await tick_task
            return result, ticks

        result, ticks = asyncio.run(run())
        assert result.success is True
        assert result.output == 3
        assert ticks >= 3  # loop made progress while sync handler ran in thread

    def test_uses_config_default_timeout(self):
        async def slow(a: int, b: int) -> int:
            await asyncio.sleep(1.0)
            return a + b

        cfg = OrchestratorConfig(tool_timeout=0.05)
        reg = ToolRegistry()
        reg.register(_def("add"), slow)
        executor = ToolExecutor(reg, config=cfg)
        result = asyncio.run(
            executor.execute_async(ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}))
        )
        assert result.success is False
        assert "TimeoutError" in result.error

class TestToolExecutorLimits:
    def _setup(self, max_tool_calls=2):
        cfg = OrchestratorConfig(max_tool_calls=max_tool_calls)
        reg = ToolRegistry()
        reg.register(_def("add"), add)
        executor = ToolExecutor(reg, config=cfg)
        return executor, reg

    def test_execution_within_configured_limit(self):
        executor, _ = self._setup(max_tool_calls=2)
        ctx = executor.new_context()
        r1 = executor.execute(ToolCall(id="1", name="add", arguments={"a": 1, "b": 1}), context=ctx)
        r2 = executor.execute(ToolCall(id="2", name="add", arguments={"a": 2, "b": 2}), context=ctx)
        assert r1.success is True and r1.output == 2
        assert r2.success is True and r2.output == 4
        assert ctx.tool_calls_count == 2

    def test_execution_rejected_after_reaching_limit(self):
        executor, _ = self._setup(max_tool_calls=1)
        ctx = executor.new_context()
        r1 = executor.execute(ToolCall(id="1", name="add", arguments={"a": 1, "b": 1}), context=ctx)
        r2 = executor.execute(ToolCall(id="2", name="add", arguments={"a": 2, "b": 2}), context=ctx)
        assert r1.success is True
        assert r2.success is False
        assert r2.tool_call_id == "2"
        assert r2.tool_name == "add"
        assert r2.output is None
        assert "ExecutionLimitError" in r2.error
        assert ctx.tool_calls_count == 1

    def test_sync_and_async_share_same_session_limit(self):
        async def run():
            cfg = OrchestratorConfig(max_tool_calls=2)
            reg = ToolRegistry()
            reg.register(_def("add"), add)
            executor = ToolExecutor(reg, config=cfg)
            ctx = executor.new_context()
            r1 = executor.execute(ToolCall(id="1", name="add", arguments={"a": 1, "b": 1}), context=ctx)
            r2 = await executor.execute_async(ToolCall(id="2", name="add", arguments={"a": 2, "b": 2}), context=ctx)
            r3 = executor.execute(ToolCall(id="3", name="add", arguments={"a": 3, "b": 3}), context=ctx)
            return r1, r2, r3, ctx

        r1, r2, r3, ctx = asyncio.run(run())
        assert r1.success is True and r2.success is True
        assert r3.success is False
        assert "ExecutionLimitError" in r3.error
        assert ctx.tool_calls_count == 2

    def test_independent_sessions_maintain_independent_counters(self):
        executor, _ = self._setup(max_tool_calls=1)
        ctx_a = executor.new_context()
        ctx_b = executor.new_context()
        ra = executor.execute(ToolCall(id="a1", name="add", arguments={"a": 1, "b": 1}), context=ctx_a)
        rb = executor.execute(ToolCall(id="b1", name="add", arguments={"a": 2, "b": 2}), context=ctx_b)
        ra2 = executor.execute(ToolCall(id="a2", name="add", arguments={"a": 3, "b": 3}), context=ctx_a)
        assert ra.success is True and rb.success is True
        assert ra2.success is False and "ExecutionLimitError" in ra2.error
        assert ctx_a.tool_calls_count == 1 and ctx_b.tool_calls_count == 1

    def test_failed_tool_executions_count_toward_limit(self):
        def boom(a: int, b: int) -> int:
            raise ValueError("boom")

        cfg = OrchestratorConfig(max_tool_calls=1)
        reg = ToolRegistry()
        reg.register(_def("add"), boom)
        executor = ToolExecutor(reg, config=cfg)
        ctx = executor.new_context()
        r1 = executor.execute(ToolCall(id="1", name="add", arguments={"a": 1, "b": 2}), context=ctx)
        r2 = executor.execute(ToolCall(id="2", name="add", arguments={"a": 1, "b": 2}), context=ctx)
        assert r1.success is False and "ValueError" in r1.error
        assert r2.success is False and "ExecutionLimitError" in r2.error
        assert ctx.tool_calls_count == 1

    def test_fresh_session_resets_counter(self):
        executor, _ = self._setup(max_tool_calls=1)
        ctx1 = executor.new_context()
        executor.execute(ToolCall(id="1", name="add", arguments={"a": 1, "b": 1}), context=ctx1)
        r_blocked = executor.execute(ToolCall(id="2", name="add", arguments={"a": 1, "b": 1}), context=ctx1)
        assert "ExecutionLimitError" in r_blocked.error
        ctx2 = executor.new_context()
        r3 = executor.execute(ToolCall(id="3", name="add", arguments={"a": 1, "b": 1}), context=ctx2)
        assert r3.success is True
        assert ctx2.tool_calls_count == 1

    def test_existing_behavior_unchanged_without_context(self):
        executor, _ = self._setup(max_tool_calls=1)
        r1 = executor.execute(ToolCall(id="1", name="add", arguments={"a": 1, "b": 1}))
        r2 = executor.execute(ToolCall(id="2", name="add", arguments={"a": 2, "b": 2}))
        assert r1.success is True and r2.success is True
