"""Unit tests for the exception hierarchy."""

import pytest

from agent_orchestrator.exceptions import (
    ConfigurationError,
    ExecutionError,
    ExecutionLimitError,
    OrchestratorError,
    ProviderError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
)


class TestExceptionHierarchy:
    def test_orchestrator_error_is_exception(self):
        assert issubclass(OrchestratorError, Exception)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigurationError,
            ToolError,
            ProviderError,
            ExecutionError,
        ],
    )
    def test_direct_children_of_orchestrator_error(self, exc_cls):
        assert issubclass(exc_cls, OrchestratorError)

    @pytest.mark.parametrize(
        "exc_cls",
        [ToolNotFoundError, ToolValidationError, ToolExecutionError],
    )
    def test_tool_error_children(self, exc_cls):
        assert issubclass(exc_cls, ToolError)
        assert issubclass(exc_cls, OrchestratorError)

    def test_execution_limit_error_is_execution_error(self):
        assert issubclass(ExecutionLimitError, ExecutionError)
        assert issubclass(ExecutionLimitError, OrchestratorError)


class TestExceptionMessages:
    @pytest.mark.parametrize(
        "exc_cls",
        [
            OrchestratorError,
            ConfigurationError,
            ToolError,
            ToolNotFoundError,
            ToolValidationError,
            ToolExecutionError,
            ProviderError,
            ExecutionError,
            ExecutionLimitError,
        ],
    )
    def test_message_preserved(self, exc_cls):
        err = exc_cls("something went wrong")
        assert str(err) == "something went wrong"

    def test_can_be_raised_and_caught_as_base(self):
        with pytest.raises(OrchestratorError):
            raise ToolNotFoundError("missing")

        with pytest.raises(ToolError):
            raise ToolValidationError("bad args")

        with pytest.raises(ExecutionError):
            raise ExecutionLimitError("too many")