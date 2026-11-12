# Agent Orchestrator

A provider-independent tool-calling and execution engine for AI applications.

## What It Is

**Agent Orchestrator** is a standalone, reusable Python package that provides a
provider-independent tool-calling and execution engine. It gives developers a
clean foundation for building AI agents that can reason, call tools, and
maintain state — without being locked into any specific LLM provider or tool
framework.

## Purpose

Modern AI applications need to orchestrate LLMs with tool use, but most
solutions are tightly coupled to a single provider (OpenAI, Anthropic, Groq,
Ollama, etc.). This makes it difficult to:

- Swap providers without rewriting tool-calling logic.
- Reuse tool definitions across different applications.
- Test orchestration logic in isolation.
- Maintain a consistent execution model across projects.

Agent Orchestrator solves this by providing normalized, provider-agnostic
contracts for tools, provider requests/responses, execution results, and
agent state.

## Architecture

- **Provider-independent**: The core package does not depend on any specific
  LLM provider. Concrete providers implement the `LLMProvider` protocol
  (`complete(request) -> ProviderResponse`) and may optionally provide
  `complete_async` for asynchronous execution.
- **Tool-independent**: Tool definitions, calls, and results are normalized
  data models; registration, schema validation, and execution are handled by
  reusable components without bundling provider-specific logic.
- **Minimal and focused**: Each component has a single responsibility with
  explicit, per-run state (no hidden global mutable execution state).
- **Testable**: Contracts are pure data models/protocols and can be unit-tested
  in isolation without network access or external services.

## Project Structure

```text
agent-orchestrator/
├── agent_orchestrator/
│   ├── __init__.py
│   ├── config.py
│   ├── exceptions.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── loop.py
│   │   └── models.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── executor.py
│   │   ├── models.py
│   │   ├── registry.py
│   │   └── validator.py
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── models.py
│   └── state/
│       ├── __init__.py
│       └── models.py
├── tests/
│   ├── __init__.py
│   ├── test_agent_loop.py
│   ├── test_agent_loop_hardening.py
│   ├── test_agent_loop_provider.py
│   ├── test_config.py
│   ├── test_core_models.py
│   ├── test_exceptions.py
│   ├── test_provider_models.py
│   ├── test_state_models.py
│   ├── test_tool_executor.py
│   ├── test_tool_integration.py
│   ├── test_tool_models.py
│   ├── test_tool_registry.py
│   └── test_tool_validator.py
├── examples/
│   └── .gitkeep
├── README.md
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Agent Execution Lifecycle (M4 Complete)

`AgentLoop` provides a reusable, provider-independent execution loop with sync
(`run`) and async (`run_async`) entry points.

1. **Initialize**: create an independent `AgentState` (execution id, message
   history, iteration/tool counters, status `PENDING`) and a fresh per-run
   `ExecutionContext` for tool-call limits.
2. **Run**: transition to `RUNNING`, build a `ProviderRequest` from current
   messages and available tool definitions, and call the provider.
3. **Provider response**:
   - **Final answer** (no tool calls): append/complete and return
     `ExecutionStatus.COMPLETED` with the final response.
   - **Tool calls**: append an assistant tool-call message, execute each call
     sequentially via `ToolExecutor`, append provider-independent tool-result
     messages (including failed results so the provider can recover), and iterate.
4. **Terminate**: stop on completion, `max_iterations` exhaustion
   (`MAX_ITERATIONS_REACHED`), or failure (provider/response/execution errors)
   with a structured `ExecutionResult` containing status, final response/error,
   iterations, and tool-call count.

Malformed/invalid provider responses or tool calls are handled predictably
(failed results or failed execution) without silently discarding tool failures
or corrupting history.

## Provider Abstraction

- `LLMProvider` is the integration boundary (`providers/base.py`) and requires
  `complete(request: ProviderRequest) -> ProviderResponse`.
- Optional async support: if a provider implements `complete_async`, the async
  loop uses it; otherwise `complete` runs in a worker thread so the event loop
  is not blocked.
- `ProviderRequest`/`ProviderResponse` and tool/result messages use
  provider-agnostic structures. Conversion to/from provider-specific formats
  belongs in concrete adapters outside the core orchestrator (no provider SDKs
  included).

## Tool-Call Round Trips

- Tool definitions come from explicit input or the executor’s `ToolRegistry`.
- Each requested tool call is validated and executed through `ToolExecutor`
  (sync/async handlers, timeouts), preserving `tool_call_id` and tool name.
- Tool results (success or failure, with error info) are serialized into tool
  messages and fed back to the provider on the next iteration. A shared
  `ExecutionContext` enforces `max_tool_calls` for the entire run.

## Configuration and Execution Limits

`OrchestratorConfig` (validated, positive limits):

- `max_iterations` (default 10): maximum agent-loop iterations.
- `tool_timeout` (default 30.0s): per-tool async execution timeout.
- `max_tool_calls` (default 20): per-session tool execution limit enforced via
  `ExecutionContext` (failed/attempted executions count; independent per run).
- `enable_tracing` (default True).

## Basic Usage Example

```python
from agent_orchestrator import (
    AgentLoop,
    OrchestratorConfig,
    ProviderRequest,
    ProviderResponse,
    ToolDefinition,
    ToolExecutor,
    ToolRegistry,
)

# Register a tool
registry = ToolRegistry()
registry.register(
    ToolDefinition(
        name="add",
        description="Add two integers",
        parameters_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    ),
    lambda a, b: a + b,
)

executor = ToolExecutor(registry)
loop = AgentLoop(config=OrchestratorConfig(), tool_executor=executor)

# Implement LLMProvider in your application (example fake provider)
class MyProvider:
    def complete(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(content="done")

state, result = loop.run(
    messages=[{"role": "user", "content": "What is 2+3?"}],
    provider=MyProvider(),
)
print(result.status, result.final_response)
```

## Current Scope (Milestone 1 — Complete)

- Foundation: package layout, `pyproject.toml`, `requirements.txt`, editable
  install, pytest configuration.
- `OrchestratorConfig`, core tool/provider/execution/state contracts, and the
  `OrchestratorError` exception hierarchy with public exports.

## Current Scope (Milestone 2 — Complete)

- `ToolRegistry` for registering/retrieving/listing/unregistering tools
  (definition + handler) without execution.
- `ToolValidator` for JSON Schema argument validation (`jsonschema`) with
  `ToolValidationError` reporting.
- Integrated tool management (`validate_arguments`, `get_all_definitions`).

## Current Scope (Milestone 3 — Complete)

- `ToolExecutor` for sync/async execution with validation, timeouts (async,
  `tool_timeout`), structured `ToolResult` success/failure (including preserved
  errors), and per-session `ExecutionContext` limits (`max_tool_calls`).

## Current Scope (Milestone 4 — Complete)

- `AgentLoop` lifecycle and provider-independent tool-call round trips
  (sync/async, optional provider `complete_async`), shared per-run execution
  context, correct history/association, iteration and tool-call limits, and
  hardened handling of provider/response/tool failures and edge cases.
- No concrete provider SDK integrations.

## Planned but NOT implemented

- Concrete provider implementations (OpenAI, Anthropic, etc.).
- Automatic retries and parallel tool/batch execution.
- Tool discovery.
- State persistence, database integration, conversation, or agent memory.
- Streaming support.
- Tracing and observability integrations.

## Installation

```bash
pip install -e .
```

## Running Tests

```bash
pytest
```

## License

MIT
