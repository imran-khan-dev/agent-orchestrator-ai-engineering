"""Unit tests for tool contract models."""

import pytest
from pydantic import ValidationError

from agent_orchestrator.tools.models import ToolCall, ToolDefinition, ToolResult


class TestToolDefinition:
    def test_required_fields(self):
        td = ToolDefinition(
            name="search",
            description="Search the web",
            parameters_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        assert td.name == "search"
        assert td.parameters_schema["properties"]["query"]["type"] == "string"

    def test_schema_defaults_to_empty_dict(self):
        td = ToolDefinition(name="ping", description="Ping service")
        assert td.parameters_schema == {}

    def test_name_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            ToolDefinition(name="", description="x")

    def test_description_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            ToolDefinition(name="x", description="")

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ToolDefinition(name="x", description="y", runtime="z")


class TestToolCall:
    def test_required_fields(self):
        tc = ToolCall(id="c1", name="search", arguments={"query": "hi"})
        assert tc.id == "c1"
        assert tc.arguments == {"query": "hi"}

    def test_arguments_default_empty(self):
        tc = ToolCall(id="c1", name="ping")
        assert tc.arguments == {}

    def test_id_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            ToolCall(id="", name="x")

    def test_name_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            ToolCall(id="c1", name="")


class TestToolResult:
    def test_success_result(self):
        tr = ToolResult(
            tool_call_id="c1",
            tool_name="search",
            output={"items": []},
            success=True,
        )
        assert tr.success is True
        assert tr.error is None

    def test_failure_result_with_error(self):
        tr = ToolResult(
            tool_call_id="c1",
            tool_name="search",
            output=None,
            success=False,
            error="timeout",
        )
        assert tr.success is False
        assert tr.error == "timeout"

    def test_success_must_be_provided(self):
        with pytest.raises(ValidationError):
            ToolResult(tool_call_id="c1", tool_name="search", output=None)

    def test_tool_call_id_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            ToolResult(tool_call_id="", tool_name="search", success=True)

    def test_tool_name_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            ToolResult(tool_call_id="c1", tool_name="", success=True)

    def test_json_compatible_output(self):
        tr = ToolResult(
            tool_call_id="c1",
            tool_name="calc",
            output={"value": 42, "list": [1, 2, 3], "nested": {"a": True}},
            success=True,
        )
        assert tr.output["nested"]["a"] is True