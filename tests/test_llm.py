"""
Tests for the LLM gateway router (new API).

Replaces old tests of ``talkingtomachines.generative.llm.query_llm``
with tests of ``talkingtomachines.gateway.router.LLMRouter``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from talkingtomachines.gateway.base import LLMProvider, LLMResponse
from talkingtomachines.gateway.router import LLMRouter, _detect_provider
from talkingtomachines.gateway.cost_tracker import CostTracker


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected_provider",
    [
        ("gpt-4o", "openai"),
        ("gpt-4o-mini", "openai"),
        ("gpt-5", "openai"),
        ("o1", "openai"),
        ("o3", "openai"),
        ("o4-mini", "openai"),
        ("claude-3-5-sonnet-20241022", "anthropic"),
        ("claude-sonnet-4-6", "anthropic"),
        ("gemini-1.5-pro", "google"),
        ("mistral-large", "mistral"),
        ("grok-2", "grok"),
        ("deepseek-chat", "deepseek"),
        ("openrouter/anthropic/claude-3", "openrouter"),
        ("hf-inference", "huggingface"),
    ],
)
def test_detect_provider(model, expected_provider):
    assert _detect_provider(model) == expected_provider


def test_detect_provider_unknown_defaults_to_openrouter():
    """Unknown models default to openrouter (router passes them through)."""
    assert _detect_provider("some-truly-unknown-model-xyz-zzzzzz") == "openrouter"


# ---------------------------------------------------------------------------
# MockProvider for router tests
# ---------------------------------------------------------------------------


class _MockProvider(LLMProvider):
    provider_name = "mock"

    def __init__(self, content="response", cost=0.001):
        self._content = content
        self._cost = cost
        self.calls = []

    def generate(
        self, message_history, model, temperature=0.0, **kwargs
    ) -> LLMResponse:
        self.calls.append({"message_history": message_history, "model": model})
        return LLMResponse(
            content=self._content,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model=model,
            provider=self.provider_name,
            latency_ms=200.0,
            cost_usd=self._cost,
        )


# ---------------------------------------------------------------------------
# LLMRouter.generate
# ---------------------------------------------------------------------------


def test_router_generate_delegates_to_provider():
    cost_tracker = CostTracker(budget_cap_usd=0.0)
    router = LLMRouter(cost_tracker=cost_tracker)
    mock_provider = _MockProvider(content="Hello!")

    with patch.object(router, "_get_provider", return_value=mock_provider):
        response = router.generate(
            message_history=[{"role": "user", "content": "Hi"}],
            model="gpt-4o",
            temperature=0.5,
        )

    assert response.content == "Hello!"
    assert len(mock_provider.calls) == 1


def test_router_generate_tracks_cost():
    cost_tracker = CostTracker(budget_cap_usd=0.0)
    router = LLMRouter(cost_tracker=cost_tracker)
    mock_provider = _MockProvider(content="OK", cost=0.01)

    with patch.object(router, "_get_provider", return_value=mock_provider):
        router.generate(
            message_history=[{"role": "user", "content": "Hi"}],
            model="gpt-4o",
        )

    assert cost_tracker.total_usd == pytest.approx(0.01)


def test_router_generate_empty_message_history():
    cost_tracker = CostTracker(budget_cap_usd=0.0)
    router = LLMRouter(cost_tracker=cost_tracker)
    mock_provider = _MockProvider(content="empty")

    with patch.object(router, "_get_provider", return_value=mock_provider):
        response = router.generate(
            message_history=[],
            model="gpt-4o",
        )

    assert response.content == "empty"


def test_router_budget_exhausted_raises():
    """BudgetExhaustedError is raised when a response would exceed the budget cap."""
    from talkingtomachines.gateway.cost_tracker import BudgetExhaustedError

    # Cap is 0.001 USD; provider response costs 0.005 → tracker raises after add()
    cost_tracker = CostTracker(budget_cap_usd=0.001)
    router = LLMRouter(cost_tracker=cost_tracker)
    mock_provider = _MockProvider(content="answer", cost=0.005)

    with pytest.raises(BudgetExhaustedError):
        with patch.object(router, "_get_provider", return_value=mock_provider):
            router.generate(
                message_history=[{"role": "user", "content": "Hi"}],
                model="gpt-4o",
            )


# ---------------------------------------------------------------------------
# LLMResponse structure
# ---------------------------------------------------------------------------


def test_llm_response_has_required_fields():
    resp = LLMResponse(
        content="test",
        prompt_tokens=5,
        completion_tokens=3,
        total_tokens=8,
        model="gpt-4o",
        provider="openai",
        latency_ms=100.0,
    )
    assert resp.content == "test"
    assert resp.total_tokens == 8
    assert resp.cost_usd == 0.0  # default


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


def test_cost_tracker_accumulates():
    ct = CostTracker(budget_cap_usd=0.0)
    ct.add(0.005)
    ct.add(0.003)
    assert ct.total_usd == pytest.approx(0.008)


def test_cost_tracker_budget_cap():
    from talkingtomachines.gateway.cost_tracker import BudgetExhaustedError

    ct = CostTracker(budget_cap_usd=0.01)
    ct.add(0.005)
    with pytest.raises(BudgetExhaustedError):
        ct.add(0.006)  # total would be 0.011 > 0.01
