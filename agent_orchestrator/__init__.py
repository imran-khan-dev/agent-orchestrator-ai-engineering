"""Agent Orchestrator package.

A provider-independent tool-calling and execution engine for AI applications.
"""

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.core.models import ExecutionResult, ExecutionStatus
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
from agent_orchestrator.providers.base import LLMProvider
from agent_orchestrator.providers.models import ProviderRequest, ProviderResponse
from agent_orchestrator.state.models import AgentState
from agent_orchestrator.tools.models import ToolCall, ToolDefinition, ToolResult

__all__ = [
    "OrchestratorConfig",
    "ExecutionResult",
    "ExecutionStatus",
    "AgentState",
    "LLMProvider",
    "ProviderRequest",
    "ProviderResponse",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "OrchestratorError",
    "ConfigurationError",
    "ToolError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolExecutionError",
    "ProviderError",
    "ExecutionError",
    "ExecutionLimitError",
]

__version__ = "0.1.0"