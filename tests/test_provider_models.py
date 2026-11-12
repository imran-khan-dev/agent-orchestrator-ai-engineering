"""Unit tests for provider request/response models and interface."""

import pytest
from pydantic import ValidationError

from agent_orchestrator.providers.base import LLMProvider
from agent_orchestrator.providers.models import ProviderRequest, ProviderResponse
from agent_orchestrator.tools.models import ToolCall, ToolDefinition


class TestProviderRequest:
    def test_minimal_request(self):
        req = ProviderRequest(messages=[{"role": "user", "content": "hi"}])
        assert req.messages[0]["role"] == "user"
        assert req.tools is None
        assert req.tool_choice is None

    def test_with_tools_and_choice(self):
        td = ToolDefinition(name="search", description="Search")
        req = ProviderRequest(
            messages=[{"role": "user", "content": "hi"}],
            tools=[td],
            tool_choice="auto",
        )
        assert len(req.tools) == 1
        assert req.tools[0].name == "search"
        assert req.tool_choice == "auto"

    def test_messages_required(self):
        with pytest.raises(ValidationError):
            ProviderRequest()

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ProviderRequest(messages=[], stream=True)


class TestProviderResponse:
    def test_defaults(self):
        resp = ProviderResponse()
        assert resp.content is None
        assert resp.tool_calls == []
        assert resp.finish_reason is None

    def test_with_content(self):
        resp = ProviderResponse(content="hello", finish_reason="stop")
        assert resp.content == "hello"
        assert resp.finish_reason == "stop"

    def test_with_tool_calls(self):
        tc = ToolCall(id="c1", name="search", arguments={"q": "x"})
        resp = ProviderResponse(tool_calls=[tc], finish_reason="tool_calls")
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ProviderResponse(content="x", usage={"tokens": 1})


class TestLLMProviderProtocol:
    def test_mock_implementation_satisfies_protocol(self):
        class MockProvider:
            def complete(self, request: ProviderRequest) -> ProviderResponse:
                return ProviderResponse(content="ok", finish_reason="stop")

        provider = MockProvider()
        assert isinstance(provider, LLMProvider)
        resp = provider.complete(ProviderRequest(messages=[]))
        assert resp.content == "ok"

    def test_runtime_checkable_rejects_incomplete(self):
        class NotAProvider:
            pass

        assert not isinstance(NotAProvider(), LLMProvider)