"""Tests for OpenAI provider temperature fallback logic.

The ``openai`` SDK may not be installed in the test environment, so all
OpenAI-specific classes are mocked via ``sys.modules``.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Create a fake openai module so imports resolve without the real SDK
# ---------------------------------------------------------------------------


class _FakeBadRequestError(Exception):
    """Stand-in for ``openai.BadRequestError``."""


_fake_openai = types.ModuleType("openai")
_fake_openai.BadRequestError = _FakeBadRequestError  # type: ignore[attr-defined]
_fake_openai.OpenAI = MagicMock  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _patch_openai_module(monkeypatch):
    """Ensure the fake openai module is used for every test."""
    monkeypatch.setitem(sys.modules, "openai", _fake_openai)
    yield


# Import AFTER the fake module is in place (module-level won't work
# because autouse fixtures run per-test).  We re-import inside each
# test class / helper to be safe.


def _import_provider():
    """(Re-)import the provider module with the fake openai in place."""
    import importlib
    import talkingtomachines.gateway.openai_provider as mod

    importlib.reload(mod)
    return mod


# ---------------------------------------------------------------------------
# _is_temperature_error
# ---------------------------------------------------------------------------


class TestIsTemperatureError:
    """Tests for the ``_is_temperature_error`` helper."""

    def test_true_for_temperature_bad_request(self):
        mod = _import_provider()
        exc = _FakeBadRequestError(
            "Unsupported parameter: 'temperature' is not supported with this model."
        )
        assert mod._is_temperature_error(exc) is True

    def test_false_for_non_temperature_bad_request(self):
        mod = _import_provider()
        exc = _FakeBadRequestError("Invalid model name.")
        assert mod._is_temperature_error(exc) is False

    def test_false_for_generic_exception(self):
        mod = _import_provider()
        assert mod._is_temperature_error(ValueError("temperature")) is False

    def test_false_for_runtime_error(self):
        mod = _import_provider()
        assert mod._is_temperature_error(RuntimeError("something")) is False


# ---------------------------------------------------------------------------
# _call_api / _call_api_chat temperature retry
# ---------------------------------------------------------------------------


class TestTemperatureRetry:
    """Verify that ``_call_api`` and ``_call_api_chat`` retry without temperature."""

    @pytest.fixture(autouse=True)
    def _clean_no_temp_models(self):
        """Remove test models from the global set after each test."""
        mod = _import_provider()
        yield
        mod.NO_TEMPERATURE_MODELS.discard("test-model")

    @staticmethod
    def _make_provider():
        mod = _import_provider()
        provider = mod.OpenAIProvider.__new__(mod.OpenAIProvider)
        provider._client = MagicMock()
        return provider, mod

    def test_call_api_retries_without_temperature(self):
        provider, mod = self._make_provider()
        mock_response = MagicMock()

        provider._client.responses.create = MagicMock(
            side_effect=[
                _FakeBadRequestError(
                    "Unsupported parameter: 'temperature' is not supported."
                ),
                mock_response,
            ]
        )

        result = provider._call_api(
            [{"role": "user", "content": "hi"}],
            "test-model",
            temperature=0.7,
        )

        assert result is mock_response
        assert "test-model" in mod.NO_TEMPERATURE_MODELS
        # Second call should NOT have temperature
        _, second_kwargs = provider._client.responses.create.call_args_list[1]
        assert "temperature" not in second_kwargs

    def test_call_api_chat_retries_without_temperature(self):
        provider, mod = self._make_provider()
        mock_response = MagicMock()

        provider._client.chat.completions.create = MagicMock(
            side_effect=[
                _FakeBadRequestError(
                    "Unsupported parameter: 'temperature' is not supported."
                ),
                mock_response,
            ]
        )

        result = provider._call_api_chat(
            [{"role": "user", "content": "hi"}],
            "test-model",
            temperature=0.7,
        )

        assert result is mock_response
        assert "test-model" in mod.NO_TEMPERATURE_MODELS
        _, second_kwargs = provider._client.chat.completions.create.call_args_list[1]
        assert "temperature" not in second_kwargs

    def test_call_api_reraises_non_temperature_error(self):
        provider, mod = self._make_provider()
        provider._client.responses.create = MagicMock(
            side_effect=_FakeBadRequestError("Invalid model name.")
        )

        with pytest.raises(_FakeBadRequestError, match="Invalid model"):
            provider._call_api(
                [{"role": "user", "content": "hi"}],
                "test-model",
                temperature=0.7,
            )
        assert "test-model" not in mod.NO_TEMPERATURE_MODELS

    def test_known_model_skips_temperature_upfront(self):
        """Models already in NO_TEMPERATURE_MODELS never send temperature."""
        provider, mod = self._make_provider()
        mock_response = MagicMock()
        provider._client.responses.create = MagicMock(return_value=mock_response)

        provider._call_api(
            [{"role": "user", "content": "hi"}],
            "o3",  # already in NO_TEMPERATURE_MODELS
            temperature=0.7,
        )

        call_kwargs = provider._client.responses.create.call_args.kwargs
        assert "temperature" not in call_kwargs
