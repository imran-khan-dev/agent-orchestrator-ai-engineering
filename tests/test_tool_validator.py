"""Unit tests for ToolValidator."""

import pytest

from agent_orchestrator.exceptions import ToolValidationError
from agent_orchestrator.tools.models import ToolDefinition
from agent_orchestrator.tools.validator import ToolValidator


def _def(schema: dict) -> ToolDefinition:
    return ToolDefinition(name="demo", description="Demo tool", parameters_schema=schema)


class TestToolValidatorValid:
    def test_valid_arguments(self):
        td = _def(
            {
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["query"],
            }
        )
        args = {"query": "hello", "limit": 5}
        ToolValidator().validate(td, args)
        assert args == {"query": "hello", "limit": 5}  # unmodified

    def test_empty_argument_object_with_empty_schema(self):
        td = _def({})
        ToolValidator().validate(td, {})

    def test_empty_argument_object_when_no_required(self):
        td = _def({"type": "object", "properties": {"q": {"type": "string"}}})
        ToolValidator().validate(td, {})


class TestToolValidatorInvalid:
    def test_missing_required_property(self):
        td = _def(
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        )
        with pytest.raises(ToolValidationError) as exc:
            ToolValidator().validate(td, {})
        assert "query" in str(exc.value)

    def test_incorrect_property_type(self):
        td = _def({"type": "object", "properties": {"limit": {"type": "integer"}}})
        with pytest.raises(ToolValidationError) as exc:
            ToolValidator().validate(td, {"limit": "ten"})
        assert "limit" in str(exc.value) or "integer" in str(exc.value)

    def test_additional_properties_disallowed(self):
        td = _def(
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "additionalProperties": False,
            }
        )
        with pytest.raises(ToolValidationError) as exc:
            ToolValidator().validate(td, {"query": "hi", "extra": 1})
        msg = str(exc.value)
        assert "extra" in msg or "Additional" in msg or "additional" in msg

    def test_invalid_schema_definition(self):
        td = _def({"type": "not-a-real-type"})
        with pytest.raises(ToolValidationError) as exc:
            ToolValidator().validate(td, {})
        assert "schema" in str(exc.value).lower()

    def test_error_message_includes_tool_name(self):
        td = _def({"type": "object", "required": ["x"], "properties": {"x": {"type": "string"}}})
        with pytest.raises(ToolValidationError) as exc:
            ToolValidator().validate(td, {})
        assert "demo" in str(exc.value)

    def test_nested_path_in_error_message(self):
        td = _def(
            {
                "type": "object",
                "properties": {
                    "config": {
                        "type": "object",
                        "properties": {"retries": {"type": "integer"}},
                        "required": ["retries"],
                    }
                },
            }
        )
        with pytest.raises(ToolValidationError) as exc:
            ToolValidator().validate(td, {"config": {"retries": "no"}})
        assert "config" in str(exc.value) or "retries" in str(exc.value)