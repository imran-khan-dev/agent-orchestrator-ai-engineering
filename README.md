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
  LLM provider. Concrete providers will implement the `LLMProvider` protocol
  in a later milestone.
- **Tool-independent**: Tool definitions, calls, and results are normalized
  data models — no registration or execution logic is bundled.
- **Minimal and focused**: Each model has a single responsibility. No
  unnecessary abstractions or hidden coupling.
- **Testable**: Contracts are pure data models and protocols that can be
  unit-tested in isolation without network access or external services.

## Project Structure

```text
agent-orchestrator/
├── agent_orchestrator/
│   ├── __init__.py
│   ├── config.py
│   ├── exceptions.py
│   ├── core/
│   │   ├── __init__.py
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

## Current Scope (Milestone 1 — Complete)

This milestone establishes the **project foundation and core contracts**.
Implemented in M1:

- Package structure and configuration (`pyproject.toml`, `requirements.txt`).
- Editable-installable package layout.
- Pydantic and pytest declared as dependencies (plus `jsonschema` for tool argument validation).
- Configured pytest test paths.
- `OrchestratorConfig` with validated execution settings.
- Tool contracts: `ToolDefinition`, `ToolCall`, `ToolResult`.
- Provider contracts: `ProviderRequest`, `ProviderResponse`, `LLMProvider`
  protocol.
- Execution models: `ExecutionStatus`, `ExecutionResult`.
- Agent state: `AgentState`.
- Exception hierarchy with `OrchestratorError` as the common base.
- Public package exports from `agent_orchestrator` and subpackages.
- Unit tests covering configuration, tools, providers, execution/state models,
  and exceptions.

## Current Scope (Milestone 2 — Complete)

Implemented in M2:

- `ToolRegistry` for registering, retrieving, listing, and unregistering tools
  (definition + handler pairs) **without executing them**.
- Registry validation: rejects duplicate tool names, unknown lookups, and
  non-callable handlers.
- `ToolValidator` for validating tool argument dictionaries against
  `ToolDefinition.parameters_schema` using JSON Schema (`jsonschema`), with
  useful `ToolValidationError` messages and schema validation.
- Integrated tool management: `ToolRegistry.validate_arguments(name, arguments)`
  validates arguments for a registered tool using its definition and
  `ToolValidator`, and `get_all_definitions()` returns definitions suitable for
  LLM provider requests. Tool execution remains outside this layer.
- Unit and integration tests covering registry, validator, and their combined
  behavior.

## Current Scope (Milestone 3 — Complete)

Implemented in M3:

- `ToolExecutor` for tool execution: retrieves a registered tool, validates
  arguments before execution, invokes the handler, and returns a `ToolResult`
  (success with output, or failure with preserved error information on
  validation/lookup, handler, timeout, or execution-limit failures).
  - Synchronous `execute()` and asynchronous `execute_async()` (sync and async
    handlers; sync handlers run in a worker thread to keep the event loop
    responsive).
  - Configurable per-call timeout for async execution, defaulting to
    `OrchestratorConfig.tool_timeout`; timeouts produce failed results and
    cancel async tasks appropriately.
  - Per-session execution safeguards via `ExecutionContext` (explicit, no
    global state): tracks tool call count and enforces
    `OrchestratorConfig.max_tool_calls` consistently for sync/async; fresh
    sessions via `new_context()`. Failed executions count toward the limit.
- Provider-independent; no retries, parallel batch execution, or orchestration.

### Planned but NOT implemented

- Actual orchestration / agent loop execution engine.
- Concrete provider implementations (OpenAI, Anthropic, etc.).
- Automatic retries and parallel batch execution.
- Tool discovery.
- State persistence, database integration, conversation, or agent memory.
- Streaming support.
- Tracing and observability integrations.

## Installation

```bash
pip install -e .
```

This installs the package in editable mode so you can develop and test
locally without rebuilding.

## Running Tests

```bash
pytest
```

## License

MIT