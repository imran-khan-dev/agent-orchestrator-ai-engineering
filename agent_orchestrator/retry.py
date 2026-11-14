"""Retry policy contracts and decision logic.

Provides provider-independent, reusable retry configuration and pure decision/
delay helpers. No sleeping or execution retries are performed here.
"""

from __future__ import annotations

import random
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, model_validator


RandomFunc = Callable[[], float]


class RetryPolicy(BaseModel):
    """Configuration and decision logic for retry behavior.

    Defaults preserve existing behavior: one attempt total, no retries.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    max_attempts: int = Field(
        default=1, ge=1, description="Maximum total attempts, including the initial attempt."
    )
    initial_delay: float = Field(
        default=0.0, ge=0.0, description="Delay in seconds before the first retry."
    )  # why delay?

    max_delay: float = Field(
        default=0.0, ge=0.0, description="Maximum backoff delay in seconds."
    )
    backoff_multiplier: float = Field(
        default=1.0, gt=0.0, description="Exponential backoff multiplier."
    )  # why exponential backoff what it is?

    jitter: bool = Field(
        default=False, description="Apply randomization to retry delays when enabled."
    )  # why jitter what it is?
    
    retry_on: tuple[type[BaseException], ...] | None = Field(
        default=None,
        description=(
            "Optional exception types eligible for retry. When None, any exception "
            "is considered retryable."
        ),
    )

    @model_validator(mode="after")
    def _validate_delay_bounds(self) -> "RetryPolicy":
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must not be less than initial_delay")
        return self

    def allows_retry(self, attempt: int) -> bool:
        """Return True if another attempt is allowed after ``attempt`` failed.

        ``attempt`` is 1-based (the attempt that just completed/failed).
        """
        return attempt < self.max_attempts

    def is_retryable(self, exc: BaseException) -> bool:
        """Return True if ``exc`` is eligible for retry under this policy."""
        if self.retry_on is None:
            return True
        return isinstance(exc, self.retry_on)

    def should_retry(self, attempt: int, exc: BaseException) -> bool:
        """Return True when ``exc`` on ``attempt`` should trigger a retry."""
        return self.allows_retry(attempt) and self.is_retryable(exc)

    def next_delay(
        self, retry_number: int, *, rand: RandomFunc | None = None
    ) -> float:
        """Calculate delay before the given retry.

        Args:
            retry_number: 1-based retry index (1 = delay before second attempt).
            rand: Optional callable returning a float in [0.0, 1.0) for jitter.
                Defaults to ``random.random`` when jitter is enabled.

        Returns:
            Delay in seconds, capped by ``max_delay`` and never negative.
        """
        if retry_number <= 0:
            return 0.0
        delay = self.initial_delay * (self.backoff_multiplier ** (retry_number - 1))
        delay = min(delay, self.max_delay)
        if self.jitter and delay > 0:
            source = rand if rand is not None else random.random
            factor = float(source())
            if factor < 0.0:
                factor = 0.0
            elif factor >= 1.0:
                factor = 0.9999999999999999
            delay = delay * factor
        return max(delay, 0.0)
