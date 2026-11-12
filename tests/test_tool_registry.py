"""Unit tests for ToolRegistry."""

import pytest

from agent_orchestrator.exceptions import ToolError, ToolNotFoundError
from agent_orchestrator.tools.models import ToolDefinition
from agent_orchestrator.tools.registry import RegisteredTool, ToolRegistry


def _def(name: str = "search") -> ToolDefinition:
    return ToolDefinition(name=name, description=f"{name} tool")


def _handler(*args, **kwargs):
    return "ok"


class TestToolRegistryRegistration:
    def test_successful_registration(self):
        reg = ToolRegistry()
        td = _def("search")
        reg.register(td, _handler)
        assert reg.get_definition("search") == td

    def test_duplicate_registration_raises_tool_error(self):
        reg = ToolRegistry()
        reg.register(_def("search"), _handler)
        with pytest.raises(ToolError) as exc:
            reg.register(_def("search"), _handler)
        assert "search" in str(exc.value)

    def test_invalid_handler_not_callable(self):
        reg = ToolRegistry()
        with pytest.raises(ToolError) as exc:
            reg.register(_def("bad"), "not-a-callable")  # type: ignore[arg-type]
        assert "callable" in str(exc.value).lower()

    def test_callable_object_is_accepted(self):
        class Handler:
            def __call__(self):
                return 1

        reg = ToolRegistry()
        reg.register(_def("obj"), Handler())
        assert reg.get("obj").handler() == 1


class TestToolRegistryRetrieval:
    def test_get_returns_registered_tool(self):
        reg = ToolRegistry()
        td = _def("ping")
        reg.register(td, _handler)
        tool = reg.get("ping")
        assert isinstance(tool, RegisteredTool)
        assert tool.definition == td
        assert tool.handler is _handler

    def test_get_definition(self):
        reg = ToolRegistry()
        td = _def("ping")
        reg.register(td, _handler)
        assert reg.get_definition("ping") is td

    def test_unknown_tool_lookup_get(self):
        reg = ToolRegistry()
        with pytest.raises(ToolNotFoundError) as exc:
            reg.get("missing")
        assert "missing" in str(exc.value)

    def test_unknown_tool_lookup_get_definition(self):
        reg = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            reg.get_definition("missing")


class TestToolRegistryListing:
    def test_list_tools(self):
        reg = ToolRegistry()
        a, b = _def("a"), _def("b")
        reg.register(a, _handler)
        reg.register(b, _handler)
        tools = reg.list_tools()
        assert tools == [a, b]

    def test_get_all_definitions(self):
        reg = ToolRegistry()
        a, b = _def("a"), _def("b")
        reg.register(a, _handler)
        reg.register(b, _handler)
        assert reg.get_all_definitions() == [a, b]

    def test_empty_registry_behavior(self):
        reg = ToolRegistry()
        assert reg.list_tools() == []
        assert reg.get_all_definitions() == []


class TestToolRegistryUnregister:
    def test_unregister_removes_tool(self):
        reg = ToolRegistry()
        reg.register(_def("x"), _handler)
        reg.unregister("x")
        assert reg.list_tools() == []
        with pytest.raises(ToolNotFoundError):
            reg.get("x")

    def test_unregister_unknown_raises(self):
        reg = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            reg.unregister("missing")