# Agent Orchestrator

A reusable, provider-independent agent execution engine for Python.

## What It Is

**Agent Orchestrator** is a standalone Python package that provides the execution
core for building tool-calling AI agents. It gives you the loop, the tool
machinery, the retry/reliability layer, and the observability hooks — while
leaving the LLM itself, the tools themselves, and the application integration
entirely up to you.

It is deliberately **not** a complete AI agent application. It ships no
provider adapters, no built-in tools, no persistence layer, and no memory or
retrieval. It is the plumbing you would otherwise rewrite in every project that
calls an LLM and lets that LLM invoke functions.

## What It Handles / What You Handle

| Agent Orchestrator handles | Your application handles |
| --- | --- |
| The agent execution loop and iteration limits | Supplying an LLM provider (any API, any SDK) |
| Provider-independent request/response contracts | Converting provider payloads to/from the contracts |
| Registering tool definitions + Python handlers | Writing the actual tool logic (DB, HTTP, files, etc.) |
| Validating tool arguments against JSON Schema | Supplying the JSON Schema for each tool |
| Executing tools (sync/async) with timeouts and limits | Deciding which tools to register |
| Provider retries and tool retries (backoff, jitter, caps) | Choosing retry policies and exception eligibility |
| Structured `ExecutionResult` (status, iterations, error) | Persisting conversation state if you need it |
| Tracing spans/events and lifecycle hooks | Supplying system prompts and message history |
| Sync and async execution paths | Long-term memory / RAG (separate integration) |

## Features

- **Agent execution loop** — `AgentLoop` drives a provider/tool round-trip loop,
  appending assistant tool-call messages and tool-result messages until the
  provider produces a final answer, a limit is hit, or execution fails.
- **Synchronous and asynchronous execution** — `run()` and `run_async()` are
  first-class entry points. Async providers use `complete_async` when available;
  otherwise `complete` runs in a worker thread so the event loop is never blocked.
- **Custom LLM provider integration** — implement the `LLMProvider` protocol
  (`complete(request) -> ProviderResponse`); optionally add `complete_async`.
  No provider SDK is bundled.
- **Tool registration and execution** — `ToolRegistry` pairs each
  `ToolDefinition` with a Python handler; `ToolExecutor` looks up, validates, and
  invokes handlers.
- **Tool argument validation** — `ToolValidator` checks arguments against each
  tool's `parameters_schema` (JSON Schema Draft 2020-12) via `jsonschema`.
- **Execution context and tool-call limits** — a per-run `ExecutionContext`
  enforces `max_tool_calls` for the entire run; independent per execution.
- **Provider retry and tool retry policies** — independently configurable
  `RetryPolicy` objects for provider calls and tool execution, with per-call
  overrides.
- **Exponential backoff, delay limits, and jitter** — `RetryPolicy.next_delay()`
  computes capped exponential backoff with optional deterministic jitter
  (injectable random source). No sleeping happens in the policy itself.
- **Execution tracing, events, spans, and in-memory tracer** — span/event
  contracts (`Tracer`, `ExecutionEvent`, `Span`, `Trace`) with a thread-safe
  `InMemoryTracer` and a `NoOpTracer`. Tracing is best-effort and never affects
  execution.
- **Lifecycle hooks** — ordered, best-effort callbacks (`HookManager`) for
  logging/metrics/side effects at agent, iteration, provider, and tool
  boundaries, distinct from tracing.
- **Async cancellation handling** — `run_async()` propagates
  `asyncio.CancelledError` unchanged and never retries it; spans are closed with
  `ERROR` and failure hooks are emitted best-effort during cleanup.
- **Error handling and execution results** — provider/response/tool failures are
  captured into a structured `ExecutionResult`; failed tool results are fed back
  to the provider so it can recover. Process-control exceptions
  (`CancelledError`, `KeyboardInterrupt`, `SystemExit`) are never retried.

## Architecture

- **Provider-independent** — the core depends only on the `LLMProvider` protocol
  and normalized `ProviderRequest`/`ProviderResponse` models. Concrete adapters
  live in your application, not here.
- **Tool-independent** — `ToolDefinition`, `ToolCall`, and `ToolResult` are
  provider-agnostic data models. Definitions are separated from the Python
  handlers that actually run.
- **Minimal and focused** — each component has a single responsibility and
  per-run state; there is no hidden global mutable execution state.
- **Testable** — the contracts are pure data models/protocols, unit-testable
  without network access or external services.

## Architectural Decisions

These are the deliberate choices made during development, and why.

- **Provider independence via a protocol.** Concrete providers implement
  `LLMProvider.complete()`; the loop never imports an SDK. This keeps tool-calling
  logic reusable across OpenAI, Anthropic, local models, or a test double, and
  makes the orchestration logic testable in isolation.
- **Separating tool definitions from handlers.** `ToolDefinition` is the
  provider-facing description (name, description, JSON Schema); the handler is a
  plain Python callable. This lets the same tool be advertised to any provider
  while keeping its implementation provider-agnostic.
- **A `ToolRegistry` to manage available tools.** Centralizing registration
  avoids scattering tool wiring through the loop, enforces unique names, and
  gives the loop a single place to source `get_all_definitions()` for requests.
- **Separating `AgentLoop` from `ToolExecutor`.** The loop owns state, iteration,
  and termination; the executor owns validation, invocation, timeouts, and tool
  limits. Each can be tested and reasoned about independently, and the executor
  can be reused outside the loop.
- **Both sync and async execution paths.** Sync-only providers run in a worker
  thread on the async path, and async providers use `complete_async`. This avoids
  blocking the event loop without forcing every provider to be async.
- **Independent provider vs. tool retry policies.** Provider calls and tool
  handlers fail for different reasons and at different rates, so each has its own
  `RetryPolicy` (config-level and per-call). Coupling them would force one
  policy to serve two unrelated concerns.
- **Retry behavior is configurable but backward compatible.** The default
  `RetryPolicy` is a single attempt with no retries, so enabling retries is an
  explicit, opt-in decision and existing behavior is preserved by default.
- **Exponential backoff with optional jitter.** Backoff avoids tight retry loops
  against flaky services; optional deterministic jitter spreads retries so
  clients do not retry in lockstep. Capping via `max_delay` bounds worst-case
  wait time, and no sleep happens after the final failed attempt.
- **Tracing kept separate from lifecycle hooks.** Tracing (`Tracer`,
  `ExecutionEvent`) is observability data for spans/events; hooks
  (`LifecycleHook`, `LifecycleEvent`) are user callbacks for side effects.
  Mixing them would couple data collection to application reactions and let one
  affect the other's failure semantics.
- **Lifecycle hooks are best-effort by design.** Ordinary hook exceptions never
  interrupt execution, change retry decisions, or alter results. This keeps
  observability from becoming a new failure mode; there is no RAISE policy.
- **Cancellation handled separately from ordinary errors and retries.**
  `asyncio.CancelledError`, `KeyboardInterrupt`, and `SystemExit` are
  process-control signals: they are never retried, are propagated (not converted
  to a `FAILED` result), and receive best-effort cleanup. Treating them as
  ordinary failures would corrupt control flow.
- **Execution state kept separate from tool execution context.** `AgentState`
  (messages, iteration, status) and `ExecutionContext` (tool-call budget) have
  different lifetimes and responsibilities; merging them would couple
  conversation state to per-run tool accounting.
- **Mixins split the large `AgentLoop`.** The original monolithic loop was
  reorganized into focused mixins (`StepRunnerMixin`, `ProviderRunnerMixin`,
  `ResponseHandlerMixin`, `ResponseNormalizerMixin`, `LifecycleHelpersMixin`,
  `TracingHelpersMixin`) composed by `AgentLoop`. This makes each responsibility
  independently readable, testable, and maintainable. Explanatory comments were
  added around the important execution flows in the loop and step runner to aid
  readability (comments are not a runtime feature).

## Project Structure

```text
Agent Orchestrator/
├── agent_orchestrator/
│   ├── __init__.py              # Public package exports
│   ├── config.py                # OrchestratorConfig
│   ├── exceptions.py            # OrchestratorError hierarchy
│   ├── hooks.py                 # HookEventType, LifecycleEvent, LifecycleHook, HookManager
│   ├── retry.py                 # RetryPolicy (config + decision/delay helpers)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── loop.py              # AgentLoop (composes the mixins below)
│   │   ├── models.py            # ExecutionResult, ExecutionStatus
│   │   ├── step_runner.py       # Custom-step iteration (sync/async)
│   │   ├── provider_runner.py   # Provider iteration + provider retry
│   │   ├── response_handler.py  # Provider response handling + tool execution
│   │   ├── response_normalizer.py # Provider/tool-call normalization
│   │   ├── lifecycle_helpers.py # State/outcome/finish helpers
│   │   └── tracing_helpers.py   # Best-effort span/event helpers
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── executor.py          # ToolExecutor, ExecutionContext
│   │   ├── models.py            # ToolDefinition, ToolCall, ToolResult
│   │   ├── registry.py          # ToolRegistry, RegisteredTool
│   │   └── validator.py         # ToolValidator (JSON Schema)
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py              # LLMProvider protocol
│   │   └── models.py            # ProviderRequest, ProviderResponse
│   ├── state/
│   │   ├── __init__.py
│   │   └── models.py            # AgentState
│   └── tracing/
│       ├── __init__.py
│       ├── collector.py         # InMemoryTracer
│       └── models.py            # ExecutionEvent, Span, Trace, Tracer, NoOpTracer
├── tests/                       # pytest suite (see Running Tests)
├── README.md
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

### How `AgentLoop` is composed

`AgentLoop` inherits from six focused mixins, each owning one responsibility:

| Mixin | Responsibility |
| --- | --- |
| `StepRunnerMixin` (`step_runner.py`) | Runs a caller-supplied iteration step (sync/async) with iteration spans, events, and hooks. |
| `ProviderRunnerMixin` (`provider_runner.py`) | Runs provider/tool round trips and applies provider retry policy. |
| `ResponseHandlerMixin` (`response_handler.py`) | Normalizes a provider response, executes tool calls, and appends messages. |
| `ResponseNormalizerMixin` (`response_normalizer.py`) | Coerces provider output into `ToolCall`s; flags malformed/duplicate calls. |
| `LifecycleHelpersMixin` (`lifecycle_helpers.py`) | State init, outcome application, finishing, and terminal-hook emission. |
| `TracingHelpersMixin` (`tracing_helpers.py`) | Best-effort span/event helpers that never affect execution. |

This keeps each concern in one readable place and lets the loop body itself stay
small.

## Installation

Requires Python 3.11+. Install in editable mode from the project root:

```bash
pip install -e .
```

Runtime dependencies (declared in `pyproject.toml`):

- `pydantic>=2.0`
- `jsonschema>=4.0`

Development dependencies (optional):

```bash
pip install -e ".[dev]"
```

which adds `pytest>=7.0`.

## Running Tests

```bash
python -m pytest -q
```

## Usage

### Create an `AgentLoop` and register a tool

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

registry = ToolRegistry()
registry.register(
    ToolDefinition(
        name="add",
        description="Add two integers.",
        parameters_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    ),
    lambda a, b: a + b,
)

executor = ToolExecutor(registry)
loop = AgentLoop(config=OrchestratorConfig(), tool_executor=executor)
```

### Implement a custom LLM provider

```python
class MyProvider:
    def complete(self, request: ProviderRequest) -> ProviderResponse:
        # Call your model here; return a normalized response.
        return ProviderResponse(content="done")

    # Optional: enable true async execution
    async def complete_async(self, request: ProviderRequest) -> ProviderResponse:
        return self.complete(request)
```

### Run synchronously

```python
state, result = loop.run(
    messages=[{"role": "user", "content": "What is 2 + 3?"}],
    provider=MyProvider(),
)
print(result.status, result.final_response)
```

### Run asynchronously

```python
import asyncio

state, result = asyncio.run(
    loop.run_async(
        messages=[{"role": "user", "content": "What is 2 + 3?"}],
        provider=MyProvider(),
    )
)
print(result.status, result.final_response)
```

### Configure retry policies

```python
from agent_orchestrator import OrchestratorConfig, RetryPolicy

config = OrchestratorConfig(
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_delay=0.5,
        max_delay=2.0,
        backoff_multiplier=2.0,
        jitter=True,
        # retry_on=None retries any exception; set to a tuple to narrow it.
    ),
    provider_retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.1, max_delay=0.1),
    tool_retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.1, max_delay=0.1),
)
loop = AgentLoop(config=config, tool_executor=executor)

# Per-call overrides take precedence over config-level policies.
state, result = loop.run(
    messages=[{"role": "user", "content": "hi"}],
    provider=MyProvider(),
    provider_retry_policy=RetryPolicy(max_attempts=5, initial_delay=0.0, max_delay=0.0),
    tool_retry_policy=RetryPolicy(max_attempts=1),
)
```

Precedence for retries: per-call override → config-level
`provider_retry_policy` / `tool_retry_policy` → `config.retry_policy`. The
default `RetryPolicy()` is a single attempt with no retries.

### Use tracing

```python
from agent_orchestrator import InMemoryTracer

tracer = InMemoryTracer()
loop = AgentLoop(config=OrchestratorConfig(enable_tracing=True),
                 tool_executor=executor, tracer=tracer)

state, result = loop.run(messages=[{"role": "user", "content": "hi"}],
                         provider=MyProvider())

print(tracer.list_traces())          # trace ids seen
print(tracer.get_events(state.execution_id))  # events for the run
print(tracer.get_trace(state.execution_id))   # spans for the run
```

`enable_tracing=False` installs a `NoOpTracer` when no tracer is supplied.
Tracing is best-effort: a misbehaving tracer never changes execution or results.

### Register lifecycle hooks

```python
from agent_orchestrator import (
    AgentLoop,
    HookEventType,
    LifecycleEvent,
    OrchestratorConfig,
)

class MyHook:
    def on_event(self, event: LifecycleEvent) -> None:
        print(event.event_type, event.execution_id, event.iteration)

loop = AgentLoop(config=OrchestratorConfig(max_iterations=3), hooks=[MyHook()])
state, result = loop.run(messages=[{"role": "user", "content": "hi"}])
```

Hooks may be sync or async. Async execution (`run_async`) awaits both sync and
`async def on_event` hooks; sync execution (`run`) runs sync hooks only and does
not execute awaitable results (coroutines are closed best-effort). Hooks run in
registration order and are best-effort: ordinary hook exceptions never interrupt
execution or change results.

### Custom iteration step (optional)

For tests or custom strategies, supply a `step` instead of a `provider`:

```python
from agent_orchestrator import IterationDecision, IterationOutcome

def step(state):
    return IterationOutcome(decision=IterationDecision.COMPLETE,
                            final_response="done")

state, result = loop.run(messages=[], step=step)
```

## Context and Application Responsibilities

- **Messages flow through `run()` / `run_async()`.** You pass an initial
  `messages` list; the loop accumulates assistant tool-call messages and tool
  result messages and returns the final `AgentState.messages`.
- **Tool results are added to conversation history.** After each tool executes,
  the loop appends a `{"role": "tool", ...}` message so the provider sees the
  outcome on the next iteration. Failed tool results are also appended, letting
  the provider recover.
- **Persistent conversation storage is the application's responsibility.** The
  loop keeps state only in memory for the duration of a run. Persisting messages
  to a database or file is up to you.
- **System prompts are supplied through `messages`.** Put your system prompt in
  the initial `messages` list; the orchestrator does not inject one.
- **Database / external access goes through registered tools.** Wrap DB, HTTP, or
  filesystem access in a tool handler and register it; the loop has no built-in
  external I/O.
- **Long-term memory and RAG are separate integrations.** They are not built-in
  features; implement them as tools or application-level orchestration around the
  loop.

## Execution Lifecycle

1. **Initialize** — create an `AgentState` (execution id, messages, iteration
   counter, status `PENDING`) and a fresh `ExecutionContext` for tool-call limits.
2. **Run** — transition to `RUNNING`, build a `ProviderRequest` from the current
   messages and available tool definitions, and call the provider.
3. **Provider response**:
   - **Final answer** (no tool calls): complete with
     `ExecutionStatus.COMPLETED` and the final response.
   - **Tool calls**: append an assistant tool-call message, execute each call via
     `ToolExecutor`, append tool-result messages (including failures so the
     provider can recover), and iterate.
4. **Terminate** — stop on completion, `max_iterations` exhaustion
   (`MAX_ITERATIONS_REACHED`), or failure (provider/response/execution errors),
   returning a structured `ExecutionResult`.

## Configuration and Execution Limits

`OrchestratorConfig` (validated, positive limits):

- `max_iterations` (default 10) — maximum agent-loop iterations.
- `tool_timeout` (default 30.0s) — per-tool async execution timeout.
- `max_tool_calls` (default 20) — per-run tool execution limit enforced via
  `ExecutionContext` (attempted executions count; independent per run).
- `enable_tracing` (default `True`).
- `retry_policy` (default `RetryPolicy()`) — shared/default retry contract.
- `provider_retry_policy` (default `None`) — optional provider override.
- `tool_retry_policy` (default `None`) — optional tool override.

`RetryPolicy` fields: `max_attempts` (total, default 1), `initial_delay`,
`max_delay`, `backoff_multiplier`, optional `jitter`, and optional `retry_on`
(exception types). Defaults preserve single-attempt behavior.

## Cancellation and Error Handling

- `run_async()` propagates `asyncio.CancelledError` unchanged; it is never
  converted to a `FAILED` result and never retried. Open spans are closed with
  `ERROR` and cancellation hooks are emitted best-effort during cleanup.
- `run()` (sync) converts ordinary exceptions into a `FAILED` `ExecutionResult`
  rather than raising.
- Process-control exceptions (`asyncio.CancelledError`, `KeyboardInterrupt`,
  `SystemExit`) are never retried in either provider or tool paths.

## Limitations and Future Extensions

Genuine current limitations:

- No concrete provider adapters (OpenAI, Anthropic, local models, etc.).
- No built-in tools.
- No parallel / batch tool execution.
- No tool discovery.
- No persistence, database integration, conversation storage, or agent memory.
- No streaming support.
- No RAISE error policy for lifecycle hooks (best-effort only); no per-run hook
  overrides.
- Synchronous execution does not await async hooks (use async hooks with async
  execution).

Possible future extensions (not implemented):

- Concrete provider implementations.
- Streaming responses.
- Parallel tool execution and tool discovery.
- Persistence/memory integrations and RAISE hook policy.

## License

MIT