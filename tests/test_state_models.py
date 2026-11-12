"""Unit tests for AgentState model."""

import pytest
from pydantic import ValidationError

from agent_orchestrator.core.models import ExecutionStatus
from agent_orchestrator.state.models import AgentState


class TestAgentState:
    def test_defaults(self):
        state = AgentState(execution_id="run-1")
        assert state.execution_id == "run-1"
        assert state.messages == []
        assert state.iteration_count == 0
        assert state.status == ExecutionStatus.PENDING
        assert state.metadata == {}

    def test_explicit_values(self):
        state = AgentState(
            execution_id="run-2",
            messages=[{"role": "user", "content": "hi"}],
            iteration_count=2,
            status=ExecutionStatus.RUNNING,
            metadata={"k": "v"},
        )
        assert state.messages == [{"role": "user", "content": "hi"}]
        assert state.iteration_count == 2
        assert state.status == ExecutionStatus.RUNNING
        assert state.metadata["k"] == "v"

    def test_messages_use_independent_defaults(self):
        a = AgentState(execution_id="a")
        b = AgentState(execution_id="b")
        a.messages.append({"role": "user", "content": "x"})
        assert b.messages == []

    def test_metadata_uses_independent_defaults(self):
        a = AgentState(execution_id="a")
        b = AgentState(execution_id="b")
        a.metadata["x"] = 1
        assert b.metadata == {}

    def test_iteration_count_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            AgentState(execution_id="run-1", iteration_count=-1)

    def test_execution_id_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            AgentState(execution_id="")

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            AgentState(execution_id="run-1", unknown=1)