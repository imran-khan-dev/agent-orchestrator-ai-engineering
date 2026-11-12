"""Unit tests for core execution models."""

import pytest
from pydantic import ValidationError

from agent_orchestrator.core.models import ExecutionResult, ExecutionStatus


class TestExecutionStatus:
    def test_all_statuses_exist(self):
        assert ExecutionStatus.PENDING.value == "PENDING"
        assert ExecutionStatus.RUNNING.value == "RUNNING"
        assert ExecutionStatus.COMPLETED.value == "COMPLETED"
        assert ExecutionStatus.FAILED.value == "FAILED"
        assert ExecutionStatus.MAX_ITERATIONS_REACHED.value == "MAX_ITERATIONS_REACHED"

    def test_member_count(self):
        assert len(ExecutionStatus) == 5


class TestExecutionResult:
    def test_minimal_valid_result(self):
        res = ExecutionResult(
            execution_id="run-1",
            status=ExecutionStatus.COMPLETED,
            iterations=1,
            tool_calls_count=2,
        )
        assert res.execution_id == "run-1"
        assert res.final_response is None
        assert res.error is None

    def test_full_result(self):
        res = ExecutionResult(
            execution_id="run-2",
            status=ExecutionStatus.FAILED,
            final_response=None,
            iterations=3,
            tool_calls_count=5,
            error="boom",
        )
        assert res.status == ExecutionStatus.FAILED
        assert res.error == "boom"

    @pytest.mark.parametrize("status", list(ExecutionStatus))
    def test_accepts_every_status(self, status):
        res = ExecutionResult(
            execution_id="run-x",
            status=status,
            iterations=0,
            tool_calls_count=0,
        )
        assert res.status == status

    def test_execution_id_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            ExecutionResult(
                execution_id="",
                status=ExecutionStatus.COMPLETED,
                iterations=0,
                tool_calls_count=0,
            )

    def test_iterations_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            ExecutionResult(
                execution_id="run-1",
                status=ExecutionStatus.COMPLETED,
                iterations=-1,
                tool_calls_count=0,
            )

    def test_tool_calls_count_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            ExecutionResult(
                execution_id="run-1",
                status=ExecutionStatus.COMPLETED,
                iterations=0,
                tool_calls_count=-1,
            )

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ExecutionResult(
                execution_id="run-1",
                status=ExecutionStatus.COMPLETED,
                iterations=0,
                tool_calls_count=0,
                extra="no",
            )