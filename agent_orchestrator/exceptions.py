"""Exception hierarchy for the Agent Orchestrator."""

from __future__ import annotations


class OrchestratorError(Exception):
    """Base exception for all Agent Orchestrator errors."""


class ConfigurationError(OrchestratorError):
    """Raised when orchestrator configuration is invalid."""


class ToolError(OrchestratorError):
    """Base exception for tool-related errors."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool cannot be found."""


class ToolValidationError(ToolError):
    """Raised when tool definition or call data is invalid."""


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""


class ProviderError(OrchestratorError):
    """Raised when an LLM provider operation fails."""


class ExecutionError(OrchestratorError):
    """Raised when an orchestration execution fails."""


class ExecutionLimitError(ExecutionError):
    """Raised when an execution limit is reached or exceeded."""