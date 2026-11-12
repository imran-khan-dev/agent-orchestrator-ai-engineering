"""Integration tests for AgentLoop provider/tool-call round trips."""

import asyncio
import json

import pytest

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.core.loop import AgentLoop
from agent_orchestrator.core.models import ExecutionStatus
from agent_orchestrator.providers.models import ProviderRequest, ProviderResponse
from agent_orchestrator.tools.executor import ToolExecutor
from agent_orchestrator.tools.models import ToolCall, ToolDefinition
from agent_orchestrator.tools.registry import ToolRegistry


def _tool_def(name: str = "add") -> ToolDefinition:
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


def _setup_executor(*, fail: bool = False, max_tool_calls: int = 20):
    reg = ToolRegistry()
    if fail:
        def boom(a: int, b: int) -> int:
            raise ValueError("boom")
        reg.register(_tool_def("add"), boom)
    else:
        reg.register(_tool_def("add"), lambda a, b: a + b)
    reg.register(_tool_def("mul"), lambda a, b: a * b)
    cfg = OrchestratorConfig(max_tool_calls=max_tool_calls, max_iterations=10)
    executor = ToolExecutor(reg, config=cfg)
    loop = AgentLoop(config=cfg, tool_executor=executor)
    return loop, reg, executor, cfg


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[ProviderRequest] = []

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class AsyncScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[ProviderRequest] = []

    async def complete_async(self, request: ProviderRequest) -> ProviderResponse:
        await asyncio.sleep(0)
        self.requests.append(request)
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class TestProviderRoundTripsSync:
    def test_final_answer_without_tool_calls(self):
        loop, *_ = _setup_executor()
        provider = ScriptedProvider([ProviderResponse(content="hello", finish_reason="stop")])
        state, result = loop.run(
            messages=[{"role": "user", "content": "hi"}], provider=provider
        )
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "hello"
        assert result.iterations == 1
        assert result.tool_calls_count == 0
        assert len(provider.requests) == 1
        assert provider.requests[0].messages == [{"role": "user", "content": "hi"}]

    def test_one_tool_call_round_trip(self):
        loop, *_ = _setup_executor()
        provider = ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})],
                    finish_reason="tool_calls",
                ),
                ProviderResponse(content="sum is 5", finish_reason="stop"),
            ]
        )
        state, result = loop.run(
            messages=[{"role": "user", "content": "add 2+3"}], provider=provider
        )
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "sum is 5"
        assert result.iterations == 2
        assert result.tool_calls_count == 1
        # history: user, assistant(tool_calls), tool, (implicit in final request)
        assert state.messages[0] == {"role": "user", "content": "add 2+3"}
        assert state.messages[1]["role"] == "assistant"
        assert state.messages[1]["tool_calls"][0]["id"] == "c1"
        assert state.messages[1]["tool_calls"][0]["name"] == "add"
        assert state.messages[2]["role"] == "tool"
        assert state.messages[2]["tool_call_id"] == "c1"
        assert state.messages[2]["name"] == "add"
        payload = json.loads(state.messages[2]["content"])
        assert payload == {"success": True, "output": 5, "error": None}
        # second request sees tool result
        assert provider.requests[1].messages[-1]["role"] == "tool"

    def test_multiple_tool_calls_in_one_response(self):
        loop, *_ = _setup_executor()
        provider = ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[
                        ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}),
                        ToolCall(id="c2", name="mul", arguments={"a": 3, "b": 4}),
                    ],
                ),
                ProviderResponse(content="done", finish_reason="stop"),
            ]
        )
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.iterations == 2
        assert result.tool_calls_count == 2
        tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
        assert [m["name"] for m in tool_msgs] == ["add", "mul"]
        assert json.loads(tool_msgs[0]["content"])["output"] == 3
        assert json.loads(tool_msgs[1]["content"])["output"] == 12

    def test_failed_tool_result_returned_to_provider(self):
        loop, *_ = _setup_executor(fail=True)
        provider = ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})],
                ),
                ProviderResponse(content="handled failure", finish_reason="stop"),
            ]
        )
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "handled failure"
        tool_msg = [m for m in state.messages if m.get("role") == "tool"][0]
        payload = json.loads(tool_msg["content"])
        assert payload["success"] is False
        assert payload["output"] is None
        assert "ValueError" in payload["error"]
        assert "boom" in payload["error"]

    def test_multiple_iterations_before_completion(self):
        loop, *_ = _setup_executor()
        provider = ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1})],
                ),
                ProviderResponse(
                    tool_calls=[ToolCall(id="c2", name="mul", arguments={"a": 2, "b": 3})],
                ),
                ProviderResponse(content="final", finish_reason="stop"),
            ]
        )
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.iterations == 3
        assert result.tool_calls_count == 2
        assert len(provider.requests) == 3

    def test_maximum_iteration_termination(self):
        cfg = OrchestratorConfig(max_iterations=2, max_tool_calls=20)
        reg = ToolRegistry()
        reg.register(_tool_def("add"), lambda a, b: a + b)
        executor = ToolExecutor(reg, config=cfg)
        loop = AgentLoop(config=cfg, tool_executor=executor)
        provider = ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1})],
                ),
                ProviderResponse(
                    tool_calls=[ToolCall(id="c2", name="add", arguments={"a": 1, "b": 1})],
                ),
                ProviderResponse(content="never", finish_reason="stop"),
            ]
        )
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.MAX_ITERATIONS_REACHED
        assert result.iterations == 2
        assert "Maximum iterations reached" in (result.error or "")
        assert len(provider.requests) == 2  # stopped before 3rd provider call

    def test_provider_exception_handling(self):
        loop, *_ = _setup_executor()
        provider = ScriptedProvider([RuntimeError("provider down")])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.FAILED
        assert "RuntimeError" in (result.error or "")
        assert "provider down" in (result.error or "")

    def test_tool_definitions_passed_to_provider(self):
        loop, reg, *_ = _setup_executor()
        provider = ScriptedProvider([ProviderResponse(content="ok")])
        loop.run(messages=[], provider=provider)
        tool_names = {t.name for t in (provider.requests[0].tools or [])}
        assert "add" in tool_names and "mul" in tool_names

    def test_tool_call_ids_and_result_association(self):
        loop, *_ = _setup_executor()
        provider = ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[
                        ToolCall(id="id-A", name="add", arguments={"a": 1, "b": 1}),
                        ToolCall(id="id-B", name="add", arguments={"a": 2, "b": 2}),
                    ],
                ),
                ProviderResponse(content="ok"),
            ]
        )
        state, _ = loop.run(messages=[], provider=provider)
        assistant = state.messages[0]
        assert [tc["id"] for tc in assistant["tool_calls"]] == ["id-A", "id-B"]
        tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["id-A", "id-B"]

    def test_correct_conversation_history_across_iterations(self):
        loop, *_ = _setup_executor()
        provider = ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})],
                ),
                ProviderResponse(content="final"),
            ]
        )
        state, _ = loop.run(messages=[{"role": "user", "content": "q"}], provider=provider)
        assert [m.get("role") for m in provider.requests[0].messages] == ["user"]
        assert [m.get("role") for m in provider.requests[1].messages] == [
            "user",
            "assistant",
            "tool",
        ]
        assert state.messages[0]["content"] == "q"


class TestProviderRoundTripsAsync:
    def test_async_provider_interaction(self):
        async def run():
            loop, *_ = _setup_executor()
            provider = AsyncScriptedProvider(
                [
                    ProviderResponse(
                        tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})],
                    ),
                    ProviderResponse(content="async done"),
                ]
            )
            return await loop.run_async(messages=[{"role": "user", "content": "hi"}], provider=provider)

        state, result = asyncio.run(run())
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "async done"
        assert result.iterations == 2
        assert result.tool_calls_count == 1

    def test_sync_provider_in_async_path_runs_without_blocking(self):
        # sync-only provider (no complete_async) should work in run_async
        async def run():
            loop, *_ = _setup_executor()
            provider = ScriptedProvider([ProviderResponse(content="ok")])
            return await loop.run_async(messages=[], provider=provider)

        state, result = asyncio.run(run())
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "ok"