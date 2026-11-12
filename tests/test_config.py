"""Unit tests for OrchestratorConfig."""

import pytest
from pydantic import ValidationError

from agent_orchestrator.config import OrchestratorConfig


class TestOrchestratorConfigDefaults:
    def test_defaults(self):
        cfg = OrchestratorConfig()
        assert cfg.max_iterations == 10
        assert cfg.tool_timeout == 30.0
        assert cfg.max_tool_calls == 20
        assert cfg.enable_tracing is True

    def test_explicit_values(self):
        cfg = OrchestratorConfig(
            max_iterations=5,
            tool_timeout=15.5,
            max_tool_calls=40,
            enable_tracing=False,
        )
        assert cfg.max_iterations == 5
        assert cfg.tool_timeout == 15.5
        assert cfg.max_tool_calls == 40
        assert cfg.enable_tracing is False


class TestOrchestratorConfigValidation:
    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_max_iterations_must_be_positive(self, bad):
        with pytest.raises(ValidationError):
            OrchestratorConfig(max_iterations=bad)

    @pytest.mark.parametrize("bad", [0, -0.5, -10])
    def test_tool_timeout_must_be_positive(self, bad):
        with pytest.raises(ValidationError):
            OrchestratorConfig(tool_timeout=bad)

    @pytest.mark.parametrize("bad", [0, -1, -50])
    def test_max_tool_calls_must_be_positive(self, bad):
        with pytest.raises(ValidationError):
            OrchestratorConfig(max_tool_calls=bad)

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            OrchestratorConfig(unknown_field=1)

    def test_error_message_mentions_limit(self):
        with pytest.raises(ValidationError) as exc:
            OrchestratorConfig(max_iterations=0)
        assert "max_iterations" in str(exc.value)