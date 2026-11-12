"""Core execution models for the Agent Orchestrator."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ExecutionStatus(str, Enum):
    """Lifecycle status of an orchestration run."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"


class ExecutionResult(BaseModel):
    """Outcome of an orchestration run."""

    model_config = ConfigDict(extra="forbid")

    execution_id: str = Field(..., min_length=1, description="Unique execution identifier.")
    status: ExecutionStatus = Field(..., description="Final status of the execution.")
    final_response: str | None = Field(
        default=None, description="Final response content, if produced."
    )
    iterations: int = Field(..., ge=0, description="Number of iterations performed.")
    tool_calls_count: int = Field(..., ge=0, description="Total number of tool calls made.")
    error: str | None = Field(default=None, description="Error message when execution failed.")