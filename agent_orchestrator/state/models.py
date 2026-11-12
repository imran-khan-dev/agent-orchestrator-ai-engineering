"""Agent state model for the Agent Orchestrator."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_orchestrator.core.models import ExecutionStatus


class AgentState(BaseModel):
    """Mutable state tracked during an orchestration run."""

    model_config = ConfigDict(extra="forbid")

    execution_id: str = Field(..., min_length=1, description="Unique execution identifier.")
    messages: list[dict[str, Any]] = Field(
        default_factory=list, description="Conversation messages accumulated so far."
    )
    iteration_count: int = Field(
        default=0, ge=0, description="Number of iterations completed."
    )
    status: ExecutionStatus = Field(
        default=ExecutionStatus.PENDING, description="Current execution status."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata for the run."
    )