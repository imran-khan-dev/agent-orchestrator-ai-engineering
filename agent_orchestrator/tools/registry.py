"""Tool registry for the Agent Orchestrator.

Stores tool definitions alongside their callable handlers. This module is
independent of LLM providers and does not execute tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent_orchestrator.exceptions import ToolError, ToolNotFoundError
from agent_orchestrator.tools.models import ToolDefinition
from agent_orchestrator.tools.validator import ToolValidator


@dataclass(frozen=True)
class RegisteredTool:
    """A tool definition paired with its handler."""

    definition: ToolDefinition
    handler: Callable[..., object]


class ToolRegistry:
    """Registry for managing callable tools.

    Tools are identified by the name on their :class:`ToolDefinition`. Duplicate
    names are rejected and handlers must be callable.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, definition: ToolDefinition, handler: Callable[..., object]) -> None:
        """Register a tool definition with its handler.

        Args:
            definition: The tool's definition.
            handler: A callable that implements the tool.

        Raises:
            ToolError: If the handler is not callable or the tool name is already
                registered.
        """
        if not callable(handler):
            raise ToolError(f"Handler for tool '{definition.name}' must be callable.")
        if definition.name in self._tools:
            raise ToolError(f"Tool '{definition.name}' is already registered.")
        self._tools[definition.name] = RegisteredTool(definition=definition, handler=handler)

    def get(self, name: str) -> RegisteredTool:
        """Retrieve a registered tool by name.

        Args:
            name: The tool name.

        Returns:
            The registered tool (definition and handler).

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool '{name}' is not registered.") from exc

    def get_definition(self, name: str) -> ToolDefinition:
        """Retrieve a tool's definition by name.

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        return self.get(name).definition

    def list_tools(self) -> list[ToolDefinition]:
        """Return all registered tool definitions."""
        return [entry.definition for entry in self._tools.values()]

    def unregister(self, name: str) -> None:
        """Remove a registered tool.

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' is not registered.")
        del self._tools[name]

    def get_all_definitions(self) -> list[ToolDefinition]:
        """Return all registered definitions suitable for providing to an LLM."""
        return self.list_tools()

    def validate_arguments(
        self,
        name: str,
        arguments: dict[str, Any],
        validator: ToolValidator | None = None,
    ) -> None:
        """Validate arguments for a registered tool using its definition.

        Args:
            name: The registered tool name.
            arguments: The argument dictionary to validate.
            validator: Optional validator instance; a default is used when omitted.

        Raises:
            ToolNotFoundError: If the tool is not registered.
            ToolValidationError: If the arguments do not conform to the schema.
        """
        definition = self.get_definition(name)
        (validator or ToolValidator()).validate(definition, arguments)