"""JSON Schema validator for tool arguments.

Validates argument dictionaries against a :class:`ToolDefinition`'s
``parameters_schema`` using the :mod:`jsonschema` package. This module is
independent of the registry and execution engine and does not execute tools.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator, SchemaError, ValidationError

from agent_orchestrator.exceptions import ToolValidationError
from agent_orchestrator.tools.models import ToolDefinition


class ToolValidator:
    """Validate tool arguments against a tool's JSON Schema.

    The validator checks the schema itself (where appropriate) and then
    validates the provided arguments. Valid arguments are accepted without
    modification. Invalid arguments or schemas raise
    :class:`ToolValidationError` with a useful message.
    """

    def validate(self, definition: ToolDefinition, arguments: dict[str, Any]) -> None:
        """Validate ``arguments`` against ``definition.parameters_schema``.

        Args:
            definition: The tool definition containing the parameters schema.
            arguments: The argument dictionary to validate.

        Raises:
            ToolValidationError: If the schema is invalid or the arguments do
                not conform to the schema.
        """
        schema = definition.parameters_schema or {}

        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ToolValidationError(
                f"Invalid parameters schema for tool '{definition.name}': {exc.message}"
            ) from exc

        validator = Draft202012Validator(schema)
        try:
            validator.validate(arguments)
        except ValidationError as exc:
            location = ".".join(str(part) for part in exc.absolute_path)
            if location:
                message = f"Invalid arguments for tool '{definition.name}' at '{location}': {exc.message}"
            else:
                message = f"Invalid arguments for tool '{definition.name}': {exc.message}"
            raise ToolValidationError(message) from exc