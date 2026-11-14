"""Provider-response handling and tool execution for AgentLoop."""

from __future__ import annotations

import json

from agent_orchestrator.core.models import ExecutionResult, ExecutionStatus
from agent_orchestrator.providers.models import ProviderResponse
from agent_orchestrator.retry import RetryPolicy
from agent_orchestrator.state.models import AgentState
from agent_orchestrator.tools.executor import ExecutionContext
from agent_orchestrator.tools.models import ToolCall, ToolResult


class ResponseHandlerMixin:
    """Normalizes provider output and coordinates tool-call execution."""

    def _handle_provider_response_sync(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        response: ProviderResponse,
        trace_id: str,
        parent_span_id: str | None,
        iteration: int,
        tool_retry_policy: RetryPolicy | None = None,
    ) -> tuple[AgentState, ExecutionResult] | None:
        # Converts the provider's response into a standard format that your Agent Orchestrator understands
        normalized = self._normalize_provider_response(response)
        if normalized.error is not None:
            return self._finish(
                state, ExecutionStatus.FAILED, tool_context, error=normalized.error
            )
        if normalized.tool_calls is not None:
            self._append_assistant_tool_call_message(
                state, normalized.tool_calls, normalized.content
            )
            for tool_call, precomputed in normalized.tool_calls:
                if precomputed is not None:
                    self._consume_tool_call_slot(tool_context, precomputed)
                    self._record_precomputed_tool_event(
                        trace_id, parent_span_id, state, iteration, tool_call, precomputed
                    )
                    tool_result = precomputed
                else:
                    tool_result = self._execute_tool_sync(
                        tool_call, tool_context, trace_id, parent_span_id, state.execution_id, iteration,
                        tool_retry_policy=tool_retry_policy,
                    )
                self._append_tool_result_message(state, tool_result)
            return None
        
        return self._finish(
            state, ExecutionStatus.COMPLETED, tool_context, final_response=normalized.content
        )

    async def _handle_provider_response_async(
        self,
        state: AgentState,
        tool_context: ExecutionContext | None,
        response: ProviderResponse,
        trace_id: str,
        parent_span_id: str | None,
        iteration: int,
        tool_retry_policy: RetryPolicy | None = None,
    ) -> tuple[AgentState, ExecutionResult] | None:
        normalized = self._normalize_provider_response(response)
        if normalized.error is not None:
            return self._finish(
                state, ExecutionStatus.FAILED, tool_context, error=normalized.error
            )
        if normalized.tool_calls is not None:
            self._append_assistant_tool_call_message(
                state, normalized.tool_calls, normalized.content
            )
            for tool_call, precomputed in normalized.tool_calls:
                if precomputed is not None:
                    self._consume_tool_call_slot(tool_context, precomputed)
                    self._record_precomputed_tool_event(
                        trace_id, parent_span_id, state, iteration, tool_call, precomputed
                    )
                    tool_result = precomputed
                else:
                    tool_result = await self._execute_tool_async(
                        tool_call, tool_context, trace_id, parent_span_id, state.execution_id, iteration,
                        tool_retry_policy=tool_retry_policy,
                    )
                self._append_tool_result_message(state, tool_result)
            return None
        return self._finish(
            state, ExecutionStatus.COMPLETED, tool_context, final_response=normalized.content
        )

    def _execute_tool_sync(self, tool_call, tool_context, trace_id, parent_span_id, execution_id, iteration, *, tool_retry_policy: RetryPolicy | None = None):
        if self._tool_executor is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                output=None,
                success=False,
                error="ToolExecutionError: No tool executor configured.",
            )
        return self._tool_executor.execute(
            tool_call,
            context=tool_context,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            execution_id=execution_id,
            iteration=iteration,
            tool_retry_policy=tool_retry_policy,
        )

    async def _execute_tool_async(self, tool_call, tool_context, trace_id, parent_span_id, execution_id, iteration, *, tool_retry_policy: RetryPolicy | None = None):
        if self._tool_executor is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                output=None,
                success=False,
                error="ToolExecutionError: No tool executor configured.",
            )
        return await self._tool_executor.execute_async(
            tool_call,
            context=tool_context,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            execution_id=execution_id,
            iteration=iteration,
            tool_retry_policy=tool_retry_policy,
        )

    @staticmethod
    def _consume_tool_call_slot(tool_context: ExecutionContext | None, result: ToolResult) -> None:
        if tool_context is None:
            return
        try:
            tool_context.consume()
        except Exception:
            pass

    @staticmethod
    def _append_assistant_tool_call_message(state: AgentState, tool_calls, content):
        state.messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [tc.model_dump() for tc, _ in tool_calls],
            }
        )

    @staticmethod
    def _append_tool_result_message(state: AgentState, result: ToolResult) -> None:
        state.messages.append(
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "name": result.tool_name,
                "content": self._serialize_tool_result(result),
            }
        )

    @staticmethod
    def _serialize_tool_result(result: ToolResult) -> str:
        payload = {"success": result.success, "output": result.output, "error": result.error}
        return json.dumps(payload, ensure_ascii=False)
