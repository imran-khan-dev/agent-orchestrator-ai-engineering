
"""Normalize provider responses into the orchestrator's internal format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from agent_orchestrator.providers.models import ProviderResponse
from agent_orchestrator.tools.models import ToolCall, ToolResult


@dataclass(frozen=True)
class _NormalizedResponse:
    content: str | None = None
    tool_calls: list[tuple[ToolCall, ToolResult | None]] | None = None
    error: str | None = None


class ResponseNormalizerMixin:
    """Provides provider-response and tool-call normalization."""

    def _normalize_provider_response(
        self, response: Any
    ) -> _NormalizedResponse:
        if isinstance(response, ProviderResponse):
            raw_calls = list(response.tool_calls or [])
            content = response.content
        elif hasattr(response, "tool_calls") or hasattr(response, "content"):
            raw_calls = list(getattr(response, "tool_calls", None) or [])
            content = getattr(response, "content", None)
        else:
            return _NormalizedResponse(
                error=(
                    "ProviderResponseError: Provider must return a ProviderResponse "
                    f"instance, got {type(response).__name__}."
                )
            )

        if raw_calls:
            normalized: list[tuple[ToolCall, ToolResult | None]] = []
            seen_ids: set[str] = set()

            for item in raw_calls:
                tool_call, precomputed = self._coerce_tool_call(item)

                if precomputed is not None:
                    normalized.append((tool_call, precomputed))
                    continue

                if tool_call.id in seen_ids:
                    precomputed = ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        output=None,
                        success=False,
                        error=(
                            "ToolValidationError: Duplicate tool_call "
                            f"id '{tool_call.id}'."
                        ),
                    )
                    normalized.append((tool_call, precomputed))
                    continue

                seen_ids.add(tool_call.id)
                normalized.append((tool_call, None))

            return _NormalizedResponse(
                content=content,
                tool_calls=normalized,
            )

        return _NormalizedResponse(content=content)

    @staticmethod
    def _coerce_tool_call(item):
        if isinstance(item, ToolCall):
            return item, None

        try:
            return ToolCall.model_validate(item), None

        except ValidationError as exc:
            fallback_id = ""
            fallback_name = ""

            if isinstance(item, dict):
                fallback_id = str(item.get("id", "") or "")
                fallback_name = str(item.get("name", "") or "")

            err_msg = (
                "ToolValidationError: Malformed tool call: "
                f"{exc.errors()[0].get('msg', str(exc))}"
            )

            try:
                placeholder = ToolCall(
                    id=fallback_id or "invalid",
                    name=fallback_name or "unknown",
                )
            except ValidationError:
                placeholder = ToolCall(id="invalid", name="unknown")
                fallback_id, fallback_name = "invalid", "unknown"

            return placeholder, ToolResult(
                tool_call_id=fallback_id or placeholder.id,
                tool_name=fallback_name or placeholder.name,
                output=None,
                success=False,
                error=err_msg,
            )

        except Exception as exc:
            return ToolCall(id="invalid", name="unknown"), ToolResult(
                tool_call_id="invalid",
                tool_name="unknown",
                output=None,
                success=False,
                error=(
                    "ToolValidationError: Malformed tool call: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
