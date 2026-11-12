"""Unit tests for the agent execution loop foundation."""

import asyncio

import pytest

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.core.loop import AgentLoop, IterationDecision, IterationOutcome
from agent_orchestrator.core.models import ExecutionResult, ExecutionStatus
from agent_orchestrator.state.models import AgentState
from agent_orchestrator.tools.executor import ToolExecutor
from agent_orchestrator.tools.registry import ToolRegistry


class TestAgentLoopInitialState:
    def test_initial_and_final_state_on_default_completion(self):
        loop = AgentLoop(OrchestratorConfig(max_iterations=5))
        observed = {}

        def step(state: AgentState) -> IterationOutcome:
            observed["status"] = state.status
            observed["iteration"] = state.iteration_count
            observed["id"] = state.execution_id
            return IterationOutcome(decision=IterationDecision.COMPLETE, final_response="done")

        state, result = loop.run(messages=[{"role": "user", "content": "hi"}], step=step)
        assert observed["status"] == ExecutionStatus.RUNNING
        assert observed["iteration"] == 1
        assert observed["id"] == state.execution_id
        assert state.messages == [{"role": "user", "content": "hi"}]
        assert state.status == ExecutionStatus.COMPLETED
        assert state.iteration_count == 1

    def test_explicit_execution_id_preserved(self):
        loop = AgentLoop()
        state, result = loop.run(execution_id="run-123")
        assert state.execution_id == "run-123"
        assert result.execution_id == "run-123"


class TestAgentLoopTransitionsAndTermination:
    def test_successful_termination(self):
        loop = AgentLoop()

        def step(state: AgentState) -> IterationOutcome:
            return IterationOutcome(decision=IterationDecision.COMPLETE, final_response="ok")

        state, result = loop.run(step=step)
        assert state.status == ExecutionStatus.COMPLETED
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "ok"
        assert result.iterations == 1
        assert result.tool_calls_count == 0
        assert result.error is None

    def test_failed_execution_state(self):
        loop = AgentLoop()

        def step(state: AgentState) -> IterationOutcome:
            return IterationOutcome(decision=IterationDecision.FAIL, error="boom")

        state, result = loop.run(step=step)
        assert state.status == ExecutionStatus.FAILED
        assert result.status == ExecutionStatus.FAILED
        assert result.error == "boom"
        assert result.final_response is None

    def test_iteration_tracking_and_continue(self):
        loop = AgentLoop(OrchestratorConfig(max_iterations=10))

        def step(state: AgentState) -> IterationOutcome:
            if state.iteration_count < 3:
                return IterationOutcome(decision=IterationDecision.CONTINUE)
            return IterationOutcome(decision=IterationDecision.COMPLETE, final_response="done")

        state, result = loop.run(step=step)
        assert state.iteration_count == 3
        assert result.iterations == 3
        assert result.status == ExecutionStatus.COMPLETED

    def test_maximum_iteration_enforcement(self):
        loop = AgentLoop(OrchestratorConfig(max_iterations=2))

        def step(state: AgentState) -> IterationOutcome:
            return IterationOutcome(decision=IterationDecision.CONTINUE)

        state, result = loop.run(step=step)
        assert state.status == ExecutionStatus.MAX_ITERATIONS_REACHED
        assert result.status == ExecutionStatus.MAX_ITERATIONS_REACHED
        assert state.iteration_count == 2
        assert result.iterations == 2
        assert "Maximum iterations reached" in (result.error or "")

    def test_step_exception_results_in_failed_state(self):
        loop = AgentLoop()

        def step(state: AgentState) -> IterationOutcome:
            raise RuntimeError("step exploded")

        state, result = loop.run(step=step)
        assert state.status == ExecutionStatus.FAILED
        assert result.status == ExecutionStatus.FAILED
        assert "RuntimeError" in (result.error or "")
        assert "step exploded" in (result.error or "")


class TestAgentLoopStateIndependenceAndAsync:
    def test_independent_execution_states(self):
        loop = AgentLoop()

        def step(state: AgentState) -> IterationOutcome:
            state.messages.append({"role": "assistant", "content": state.execution_id})
            return IterationOutcome(decision=IterationDecision.COMPLETE)

        s1, r1 = loop.run(execution_id="a", messages=[{"role": "user", "content": "x"}])
        s2, r2 = loop.run(execution_id="b", messages=[{"role": "user", "content": "y"}])
        assert s1.execution_id == "a" and s2.execution_id == "b"
        assert s1.messages != s2.messages
        assert r1.execution_id == "a" and r2.execution_id == "b"

    def test_async_run_with_async_step(self):
        async def run():
            loop = AgentLoop(OrchestratorConfig(max_iterations=5))

            async def step(state: AgentState) -> IterationOutcome:
                await asyncio.sleep(0)
                if state.iteration_count < 2:
                    return IterationOutcome(decision=IterationDecision.CONTINUE)
                return IterationOutcome(decision=IterationDecision.COMPLETE, final_response="async-ok")

            return await loop.run_async(step=step)

        state, result = asyncio.run(run())
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "async-ok"
        assert result.iterations == 2

    def test_async_run_with_sync_step(self):
        async def run():
            loop = AgentLoop()

            def step(state: AgentState) -> IterationOutcome:
                return IterationOutcome(decision=IterationDecision.COMPLETE, final_response="sync-ok")

            return await loop.run_async(step=step)

        state, result = asyncio.run(run())
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "sync-ok"

    def test_structured_execution_result_fields(self):
        loop = AgentLoop(OrchestratorConfig(max_iterations=3))

        def step(state: AgentState) -> IterationOutcome:
            return IterationOutcome(decision=IterationDecision.COMPLETE, final_response="r")

        state, result = loop.run(execution_id="rid", step=step)
        assert isinstance(result, ExecutionResult)
        assert result.execution_id == "rid"
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_response == "r"
        assert result.iterations == 1
        assert result.tool_calls_count == 0
        assert result.error is None

    def test_tool_executor_context_initialized_when_provided(self):
        # Architecture integration: executor/context is wired (no round trips yet).
        reg = ToolRegistry()
        executor = ToolExecutor(reg)
        loop = AgentLoop(tool_executor=executor)
        state, result = loop.run()
        assert result.tool_calls_count == 0
        assert result.status == ExecutionStatus.COMPLETED