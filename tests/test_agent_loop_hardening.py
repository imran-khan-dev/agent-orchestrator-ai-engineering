"""Edge-case tests for hardened AgentLoop execution flow."""

import asyncio
import json

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


def _setup(max_iterations=10, max_tool_calls=20):
    reg = ToolRegistry()
    reg.register(_tool_def("add"), lambda a, b: a + b)
    cfg = OrchestratorConfig(max_iterations=max_iterations, max_tool_calls=max_tool_calls)
    executor = ToolExecutor(reg, config=cfg)
    loop = AgentLoop(config=cfg, tool_executor=executor)
    return loop, reg, executor


class TestAgentLoopHardening:
    def test_invalid_provider_response_type_fails(self):
        loop, *_ = _setup()

        class BadProvider:
            def complete(self, request):
                return {"not": "a provider response"}

        state, result = loop.run(messages=[], provider=BadProvider())
        assert result.status == ExecutionStatus.FAILED
        assert "ProviderResponseError" in (result.error or "")
        assert result.iterations == 1

    def test_empty_provider_response_completes_with_none(self):
        loop, *_ = _setup()
        provider = ScriptedProvider([ProviderResponse()])
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response is None
        assert result.iterations == 1

    def test_missing_or_duplicate_tool_call_ids_handled(self):
        loop, *_ = _setup()

        class ResponseLike:
            def __init__(self, content=None, tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls or []

        provider = ScriptedProvider(
            [
                ResponseLike(
                    tool_calls=[
                        {"name": "add", "arguments": {"a": 1, "b": 2}},  # missing id
                        ToolCall(id="dup", name="add", arguments={"a": 2, "b": 2}),
                        ToolCall(id="dup", name="add", arguments={"a": 3, "b": 3}),  # duplicate
                    ]
                ),
                ProviderResponse(content="recovered"),
            ]
        )
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "recovered"
        tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 3
        payloads = [json.loads(m["content"]) for m in tool_msgs]
        assert payloads[0]["success"] is False
        assert "ToolValidationError" in payloads[0]["error"]
        assert payloads[1]["success"] is True and payloads[1]["output"] == 4
        assert payloads[2]["success"] is False
        assert "Duplicate" in (payloads[2]["error"] or "")
        # malformed/duplicate attempts still consume execution slots
        assert result.tool_calls_count == 3

    def test_failed_tool_does_not_corrupt_history(self):
        loop, *_ = _setup()
        provider = ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[ToolCall(id="c1", name="missing_tool", arguments={})],
                ),
                ProviderResponse(content="continued"),
            ]
        )
        state, result = loop.run(messages=[{"role": "user", "content": "q"}], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        assert state.messages[0] == {"role": "user", "content": "q"}
        assert state.messages[1]["role"] == "assistant"
        assert state.messages[2]["role"] == "tool"
        payload = json.loads(state.messages[2]["content"])
        assert payload["success"] is False
        assert "ToolNotFoundError" in (payload["error"] or "")
        assert [m.get("role") for m in provider.requests[1].messages] == [
            "user",
            "assistant",
            "tool",
        ]

    def test_max_tool_calls_enforced_across_round_trips(self):
        loop, *_ = _setup(max_iterations=10, max_tool_calls=2)
        provider = ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[
                        ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1}),
                        ToolCall(id="c2", name="add", arguments={"a": 2, "b": 2}),
                        ToolCall(id="c3", name="add", arguments={"a": 3, "b": 3}),
                    ],
                ),
                ProviderResponse(content="after limit"),
            ]
        )
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
        payloads = [json.loads(m["content"]) for m in tool_msgs]
        assert payloads[0]["success"] is True and payloads[1]["success"] is True
        assert payloads[2]["success"] is False
        assert "ExecutionLimitError" in (payloads[2]["error"] or "")
        assert result.tool_calls_count == 2  # context consumed slots

    def test_tool_timeout_effective_in_round_trip(self):
        async def run():
            reg = ToolRegistry()

            async def slow(a: int, b: int) -> int:
                await asyncio.sleep(1.0)
                return a + b

            reg.register(_tool_def("add"), slow)
            cfg = OrchestratorConfig(max_iterations=5, max_tool_calls=5, tool_timeout=0.05)
            executor = ToolExecutor(reg, config=cfg)
            loop = AgentLoop(config=cfg, tool_executor=executor)
            provider = ScriptedProvider(
                [
                    ProviderResponse(
                        tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1})],
                    ),
                    ProviderResponse(content="timed out handled"),
                ]
            )
            return await loop.run_async(messages=[], provider=provider)

        state, result = asyncio.run(run())
        tool_msg = [m for m in state.messages if m.get("role") == "tool"][0]
        payload = json.loads(tool_msg["content"])
        assert payload["success"] is False
        assert "TimeoutError" in (payload["error"] or "")
        assert result.status == ExecutionStatus.COMPLETED

    def test_each_run_has_independent_state_and_context(self):
        loop, *_ = _setup(max_tool_calls=1)
        p1 = ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1})],
                ),
                ProviderResponse(content="r1"),
            ]
        )
        p2 = ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[ToolCall(id="c2", name="add", arguments={"a": 2, "b": 2})],
                ),
                ProviderResponse(content="r2"),
            ]
        )
        s1, r1 = loop.run(messages=[{"role": "user", "content": "a"}], provider=p1)
        s2, r2 = loop.run(messages=[{"role": "user", "content": "b"}], provider=p2)
        assert r1.status == ExecutionStatus.COMPLETED and r2.status == ExecutionStatus.COMPLETED
        assert r1.tool_calls_count == 1 and r2.tool_calls_count == 1
        assert s1.execution_id != s2.execution_id
        assert s1.messages[0]["content"] == "a" and s2.messages[0]["content"] == "b"

    def test_async_path_handles_malformed_and_invalid_response(self):
        async def run_invalid():
            loop, *_ = _setup()

            class BadAsyncProvider:
                async def complete_async(self, request):
                    return "bad"

            return await loop.run_async(messages=[], provider=BadAsyncProvider())

        _, result = asyncio.run(run_invalid())
        assert result.status == ExecutionStatus.FAILED
        assert "ProviderResponseError" in (result.error or "")

        async def run_empty():
            loop, *_ = _setup()
            provider = AsyncScriptedProvider([ProviderResponse()])
            return await loop.run_async(messages=[], provider=provider)

        _, result = asyncio.run(run_empty())
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response is None

    def test_no_tool_executor_tool_calls_return_failed_results(self):
        loop = AgentLoop(OrchestratorConfig(max_iterations=5))
        provider = ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})],
                ),
                ProviderResponse(content="handled"),
            ]
        )
        state, result = loop.run(messages=[], provider=provider)
        assert result.status == ExecutionStatus.COMPLETED
        payload = json.loads([m for m in state.messages if m["role"] == "tool"][0]["content"])
        assert payload["success"] is False
        assert "No tool executor configured" in (payload["error"] or "")