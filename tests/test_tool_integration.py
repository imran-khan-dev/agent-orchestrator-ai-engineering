"""Integration tests for ToolRegistry and ToolValidator."""

import pytest

from agent_orchestrator.exceptions import ToolNotFoundError, ToolValidationError
from agent_orchestrator.tools.models import ToolDefinition
from agent_orchestrator.tools.registry import ToolRegistry
from agent_orchestrator.tools.validator import ToolValidator


def _def(name: str = "search") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} tool",
        parameters_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def _handler(**kwargs):
    return kwargs


class TestRegistryValidatorIntegration:
    def test_register_and_retrieve_tool(self):
        reg = ToolRegistry()
        td = _def("search")
        reg.register(td, _handler)
        tool = reg.get("search")
        assert tool.definition == td
        assert tool.handler is _handler

    def test_validate_arguments_for_registered_tool(self):
        reg = ToolRegistry()
        reg.register(_def("search"), _handler)
        reg.validate_arguments("search", {"query": "hello"})

    def test_validate_with_explicit_validator(self):
        reg = ToolRegistry()
        reg.register(_def("search"), _handler)
        reg.validate_arguments("search", {"query": "hello"}, validator=ToolValidator())

    def test_reject_invalid_arguments_for_registered_tool(self):
        reg = ToolRegistry()
        reg.register(_def("search"), _handler)
        with pytest.raises(ToolValidationError):
            reg.validate_arguments("search", {"query": 123})
        with pytest.raises(ToolValidationError):
            reg.validate_arguments("search", {})
        with pytest.raises(ToolValidationError):
            reg.validate_arguments("search", {"query": "hi", "extra": 1})

    def test_unknown_tool_name_on_validate(self):
        reg = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            reg.validate_arguments("missing", {"query": "x"})

    def test_retrieve_all_definitions_for_provider_requests(self):
        reg = ToolRegistry()
        a, b = _def("a"), _def("b")
        reg.register(a, _handler)
        reg.register(b, _handler)
        defs = reg.get_all_definitions()
        assert defs == [a, b]
        # definitions are suitable ToolDefinition instances
        assert all(isinstance(d, ToolDefinition) for d in defs)