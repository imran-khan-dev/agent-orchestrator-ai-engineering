"""Unit tests for RetryPolicy and retry configuration."""

import pytest
from pydantic import ValidationError

from agent_orchestrator.config import OrchestratorConfig
from agent_orchestrator.retry import RetryPolicy


class TestRetryPolicyDefaults:
    def test_default_one_attempt_no_retries(self):
        policy = RetryPolicy()
        assert policy.max_attempts == 1
        assert policy.initial_delay == 0.0
        assert policy.max_delay == 0.0
        assert policy.backoff_multiplier == 1.0
        assert policy.jitter is False
        assert policy.retry_on is None
        assert policy.allows_retry(1) is False
        assert policy.should_retry(1, RuntimeError("x")) is False

    def test_default_next_delay(self):
        policy = RetryPolicy()
        assert policy.next_delay(1) == 0.0


class TestRetryPolicyAttemptsAndExceptions:
    def test_max_attempt_boundaries(self):
        policy = RetryPolicy(max_attempts=3)
        assert policy.allows_retry(1) is True
        assert policy.allows_retry(2) is True
        assert policy.allows_retry(3) is False
        assert policy.allows_retry(4) is False

    def test_retryable_and_non_retryable_exceptions(self):
        policy = RetryPolicy(max_attempts=3, retry_on=(ValueError, TimeoutError))
        assert policy.is_retryable(ValueError("a")) is True
        assert policy.is_retryable(TimeoutError("a")) is True
        assert policy.is_retryable(RuntimeError("a")) is False
        assert policy.should_retry(1, ValueError("a")) is True
        assert policy.should_retry(1, RuntimeError("a")) is False
        assert policy.should_retry(3, ValueError("a")) is False  # no attempts left

    def test_retry_on_none_means_all_retryable(self):
        policy = RetryPolicy(max_attempts=2, retry_on=None)
        assert policy.is_retryable(RuntimeError("x")) is True
        assert policy.is_retryable(Exception("x")) is True


class TestRetryPolicyDelayCalculation:
    def test_exponential_backoff(self):
        policy = RetryPolicy(
            max_attempts=5, initial_delay=1.0, max_delay=100.0, backoff_multiplier=2.0
        )
        assert policy.next_delay(1) == 1.0
        assert policy.next_delay(2) == 2.0
        assert policy.next_delay(3) == 4.0
        assert policy.next_delay(4) == 8.0

    def test_max_delay_enforcement(self):
        policy = RetryPolicy(
            max_attempts=10, initial_delay=1.0, max_delay=5.0, backoff_multiplier=2.0
        )
        assert policy.next_delay(1) == 1.0
        assert policy.next_delay(2) == 2.0
        assert policy.next_delay(3) == 4.0
        assert policy.next_delay(4) == 5.0
        assert policy.next_delay(10) == 5.0

    def test_jitter_uses_injected_random_deterministically(self):
        policy = RetryPolicy(
            max_attempts=5,
            initial_delay=10.0,
            max_delay=100.0,
            backoff_multiplier=1.0,
            jitter=True,
        )
        assert policy.next_delay(1, rand=lambda: 0.5) == 5.0
        assert policy.next_delay(1, rand=lambda: 0.0) == 0.0
        assert policy.next_delay(1, rand=lambda: 0.999) == 9.99
        assert policy.next_delay(2, rand=lambda: 0.25) == 2.5

    def test_jitter_disabled_ignores_random(self):
        policy = RetryPolicy(
            max_attempts=3, initial_delay=2.0, max_delay=10.0, jitter=False
        )
        assert policy.next_delay(1, rand=lambda: 0.0) == 2.0

    def test_non_positive_retry_number_returns_zero(self):
        policy = RetryPolicy(max_attempts=3, initial_delay=1.0, max_delay=5.0)
        assert policy.next_delay(0) == 0.0
        assert policy.next_delay(-1) == 0.0


class TestRetryPolicyValidation:
    def test_invalid_max_attempts(self):
        with pytest.raises(ValidationError):
            RetryPolicy(max_attempts=0)
        with pytest.raises(ValidationError):
            RetryPolicy(max_attempts=-1)

    def test_invalid_delays(self):
        with pytest.raises(ValidationError):
            RetryPolicy(initial_delay=-0.1)
        with pytest.raises(ValidationError):
            RetryPolicy(max_delay=-1.0)

    def test_invalid_backoff_multiplier(self):
        with pytest.raises(ValidationError):
            RetryPolicy(backoff_multiplier=0.0)
        with pytest.raises(ValidationError):
            RetryPolicy(backoff_multiplier=-2.0)

    def test_max_delay_less_than_initial_delay(self):
        with pytest.raises(ValidationError) as exc:
            RetryPolicy(initial_delay=5.0, max_delay=1.0)
        assert "max_delay" in str(exc.value)

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            RetryPolicy(max_attempts=2, unknown=True)


class TestOrchestratorConfigRetryIntegration:
    def test_default_retry_policy_preserves_behavior(self):
        cfg = OrchestratorConfig()
        assert isinstance(cfg.retry_policy, RetryPolicy)
        assert cfg.retry_policy.max_attempts == 1
        assert cfg.retry_policy.allows_retry(1) is False
        # existing defaults unchanged
        assert cfg.max_iterations == 10
        assert cfg.tool_timeout == 30.0
        assert cfg.max_tool_calls == 20
        assert cfg.enable_tracing is True

    def test_custom_retry_policy(self):
        policy = RetryPolicy(max_attempts=3, initial_delay=0.5, max_delay=2.0)
        cfg = OrchestratorConfig(retry_policy=policy)
        assert cfg.retry_policy.max_attempts == 3
        assert cfg.retry_policy.initial_delay == 0.5

    def test_constructor_compatibility(self):
        cfg = OrchestratorConfig(max_iterations=5, tool_timeout=10.0, max_tool_calls=7, enable_tracing=False)
        assert cfg.max_iterations == 5
        assert cfg.retry_policy.max_attempts == 1

    def test_invalid_retry_policy_rejected(self):
        with pytest.raises(ValidationError):
            OrchestratorConfig(retry_policy=RetryPolicy(max_attempts=0))
