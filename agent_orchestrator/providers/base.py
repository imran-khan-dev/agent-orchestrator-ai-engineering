"""Abstract provider interface for the Agent Orchestrator.

Defines the provider-independent :class:`LLMProvider` protocol. No concrete
provider adapters or network calls are implemented here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_orchestrator.providers.models import ProviderRequest, ProviderResponse


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol describing the provider contract.

    Any object that implements :meth:`complete` with the correct signature is
    considered a valid provider.
    """

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Send a request to the provider and return a normalized response."""
        ...