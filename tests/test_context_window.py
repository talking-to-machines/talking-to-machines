"""
Tests for context window estimation and overflow handling.

Covers:
  - Model registry: exact match, prefix match, date-suffix stripping, default
  - Token estimator: heuristic token counting, message list estimation
  - Context guard: terminate, truncate, summarize policies
  - Context window validator: compile-time estimation warnings/errors
  - Settings parsing: CONTEXT_OVERFLOW_POLICY validation
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from talkingtomachines.gateway.model_registry import (
    ModelSpec,
    get_model_spec,
    _DEFAULT_SPEC,
)
from talkingtomachines.gateway.token_estimator import (
    estimate_tokens,
    estimate_messages_tokens,
)
from talkingtomachines.gateway.context_guard import (
    ContextGuard,
    ContextWindowExceededError,
)
from talkingtomachines.authoring.validators.context_window_validator import (
    ContextWindowValidator,
)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def test_model_registry_exact_match():
    spec = get_model_spec("gpt-4o")
    assert spec.max_context_tokens == 128_000


def test_model_registry_date_suffix_stripped():
    spec = get_model_spec("gpt-4o-2024-05-13")
    assert spec.max_context_tokens == 128_000


def test_model_registry_latest_suffix_stripped():
    spec = get_model_spec("mistral-large-latest")
    assert spec.max_context_tokens == 128_000


def test_model_registry_prefix_match():
    spec = get_model_spec("claude-opus-4-6")
    assert spec.max_context_tokens == 200_000


def test_model_registry_unknown_returns_default():
    spec = get_model_spec("some-unknown-model")
    assert spec == _DEFAULT_SPEC
    assert spec.max_context_tokens == 128_000


def test_model_registry_case_insensitive():
    spec = get_model_spec("GPT-4O")
    assert spec.max_context_tokens == 128_000


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------


def test_estimate_tokens_basic():
    """Heuristic: ~4 chars per token."""
    text = "Hello, world!"  # 13 chars → ~3 tokens
    tokens = estimate_tokens(text)
    assert tokens >= 1
    assert tokens < 20  # Sanity bound


def test_estimate_tokens_empty():
    assert estimate_tokens("") >= 1  # min 1


def test_estimate_messages_tokens():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
    total = estimate_messages_tokens(messages)
    assert total > 0
    # Should include per-message overhead (4 tokens each)
    assert total >= 8  # At least 4 overhead per message


def test_estimate_messages_tokens_multimodal():
    """Text parts in multimodal content blocks are counted."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.png"},
                },
            ],
        },
    ]
    total = estimate_messages_tokens(messages)
    assert total > 4  # More than just overhead


# ---------------------------------------------------------------------------
# Context guard — terminate
# ---------------------------------------------------------------------------


def test_context_guard_terminate_raises():
    """Terminate policy raises ContextWindowExceededError."""
    guard = ContextGuard(model_name="gpt-4o", policy="terminate")
    # Override effective limit to something tiny
    guard._effective_limit = 10

    messages = [
        {"role": "system", "content": "System prompt " * 50},
        {"role": "user", "content": "Hello " * 50},
    ]
    with pytest.raises(ContextWindowExceededError, match="terminate"):
        guard.apply(messages)


def test_context_guard_within_limit_returns_unchanged():
    """Messages within limit are returned unchanged."""
    guard = ContextGuard(model_name="gpt-4o", policy="terminate")
    messages = [
        {"role": "system", "content": "Short."},
        {"role": "user", "content": "Hi."},
    ]
    result = guard.apply(messages)
    assert result == messages


# ---------------------------------------------------------------------------
# Context guard — truncate
# ---------------------------------------------------------------------------


def test_context_guard_truncate_preserves_structure():
    """Truncate keeps system, first exchange, and current prompt."""
    guard = ContextGuard(model_name="gpt-4o", policy="truncate")
    guard._effective_limit = 50  # Very small

    # Use longer content to ensure we exceed the limit
    messages = [
        {"role": "system", "content": "System prompt " * 5},
        {"role": "user", "content": "First exchange " * 5},
        {"role": "assistant", "content": "Old reply one " * 20},
        {"role": "user", "content": "Old message two " * 20},
        {"role": "assistant", "content": "Old reply two " * 20},
        {"role": "user", "content": "Current prompt"},
    ]
    result = guard.apply(messages)

    # System prompt preserved
    assert result[0]["content"] == "System prompt " * 5
    # First exchange preserved
    assert result[1]["content"] == "First exchange " * 5
    # Current prompt is last
    assert result[-1]["content"] == "Current prompt"
    # A truncation marker should exist
    assert any("[Note:" in msg.get("content", "") for msg in result)


def test_context_guard_truncate_short_messages_unchanged():
    """3 or fewer messages should not be truncated."""
    guard = ContextGuard(model_name="gpt-4o", policy="truncate")
    guard._effective_limit = 10

    messages = [
        {"role": "system", "content": "A" * 100},
        {"role": "user", "content": "B" * 100},
        {"role": "user", "content": "C" * 100},
    ]
    result = guard.apply(messages)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# Context guard — summarize
# ---------------------------------------------------------------------------


def test_context_guard_summarize_calls_router():
    """Summarize policy calls router.generate() to summarize old messages."""
    guard = ContextGuard(model_name="gpt-4o", policy="summarize")
    # Set limit so that raw messages exceed it, but after summarizing the
    # older half, the result fits (system + first + summary + kept + current).
    guard._effective_limit = 400

    mock_router = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Brief summary."
    mock_router.generate.return_value = mock_response

    messages = [
        {"role": "system", "content": "System prompt " * 5},
        {"role": "user", "content": "First exchange " * 5},
        {"role": "assistant", "content": "Reply one " * 30},
        {"role": "user", "content": "Message two " * 30},
        {"role": "assistant", "content": "Reply two " * 30},
        {"role": "user", "content": "Message three " * 30},
        {"role": "assistant", "content": "Reply three " * 30},
        {"role": "user", "content": "Current prompt"},
    ]
    result = guard.apply(messages, router=mock_router, temperature=0.0)

    # Router should have been called for summarization
    mock_router.generate.assert_called_once()
    # Result should contain a summary message
    assert any("[Summary" in msg.get("content", "") for msg in result)


def test_context_guard_summarize_falls_back_to_truncate_on_failure():
    """If summarization LLM call fails, fall back to truncation."""
    guard = ContextGuard(model_name="gpt-4o", policy="summarize")
    guard._effective_limit = 50

    mock_router = MagicMock()
    mock_router.generate.side_effect = Exception("API error")

    messages = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Msg 2"},
        {"role": "user", "content": "Current"},
    ]
    result = guard.apply(messages, router=mock_router, temperature=0.0)

    # Should not raise; should fall back to truncation
    assert result[0]["content"] == "Sys"
    assert result[-1]["content"] == "Current"


def test_context_guard_summarize_no_router_falls_back():
    """If no router provided, summarize falls back to truncation."""
    guard = ContextGuard(model_name="gpt-4o", policy="summarize")
    guard._effective_limit = 50

    messages = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Reply"},
        {"role": "user", "content": "Current"},
    ]
    result = guard.apply(messages, router=None)
    assert result[0]["content"] == "Sys"
    assert result[-1]["content"] == "Current"


# ---------------------------------------------------------------------------
# Context guard — concurrent truncation
# ---------------------------------------------------------------------------


def test_context_guard_truncate_concurrent():
    """Multiple threads using truncate simultaneously must not corrupt results."""
    guard = ContextGuard(model_name="gpt-4o", policy="truncate")
    guard._effective_limit = 80
    errors = []

    def apply_guard(i):
        try:
            messages = [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": f"Reply {i} " * 20},
                {"role": "user", "content": f"Msg {i} " * 20},
                {"role": "user", "content": f"Current {i}"},
            ]
            result = guard.apply(messages)
            assert result[0]["content"] == "System"
            assert result[-1]["content"] == f"Current {i}"
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=apply_guard, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors


# ---------------------------------------------------------------------------
# Context window validator
# ---------------------------------------------------------------------------


def test_context_window_validator_no_error_small_experiment():
    """Small experiment should not trigger warnings or errors."""
    prompts = {
        "task1": [SimpleNamespace(llm_text="What is 2+2?", is_displayed=None)],
    }
    validator = ContextWindowValidator(
        model_name="gpt-4o",
        prompts=prompts,
        constants={"task1": {"MAX_NUM_ROUNDS": 2, "PLAYERS_PER_GROUP": 2}},
        task_sequence=["task1"],
        num_agents_per_session=2,
    )
    errors = validator.validate()
    assert errors == []


def test_context_window_validator_error_exceeds_limit():
    """Large experiment should produce an error."""
    # Use a model with very small context window via direct spec override
    long_prompt = "x " * 2000  # ~1000 tokens
    prompts = {
        "task1": [SimpleNamespace(llm_text=long_prompt, is_displayed=None)],
    }
    validator = ContextWindowValidator(
        model_name="open-mistral-7b",  # 32,000 context window
        prompts=prompts,
        constants={"task1": {"MAX_NUM_ROUNDS": 100, "PLAYERS_PER_GROUP": 10}},
        task_sequence=["task1"],
        num_agents_per_session=10,
    )
    errors = validator.validate()
    assert len(errors) >= 1
    assert "EXCEEDS" in str(errors[0])


def test_context_window_validator_warning_near_limit(caplog):
    """Experiment near limit should produce a warning."""
    prompts = {
        "task1": [SimpleNamespace(llm_text="Short prompt.", is_displayed=None)],
    }
    # Use open-mistral-7b (32K) with moderate settings to trigger >80% warning
    validator = ContextWindowValidator(
        model_name="open-mistral-7b",
        prompts=prompts,
        constants={"task1": {"MAX_NUM_ROUNDS": 50, "PLAYERS_PER_GROUP": 3}},
        task_sequence=["task1"],
        num_agents_per_session=3,
    )
    with caplog.at_level(logging.WARNING):
        errors = validator.validate()
    # May or may not produce a blocking error depending on exact estimate,
    # but should produce some output (error or warning)
    has_warning = any("context" in msg.lower() for msg in caplog.messages)
    has_error = len(errors) > 0
    assert has_warning or has_error


def test_context_window_validator_summary():
    """get_summary() returns model info."""
    validator = ContextWindowValidator(
        model_name="gpt-4o",
        prompts={},
        constants={},
        task_sequence=[],
        num_agents_per_session=1,
    )
    summary = validator.get_summary()
    assert summary["model_name"] == "gpt-4o"
    assert summary["max_context_tokens"] == 128_000


# ---------------------------------------------------------------------------
# Settings parsing — CONTEXT_OVERFLOW_POLICY
# ---------------------------------------------------------------------------


def test_settings_parser_valid_overflow_policy():
    """Valid overflow policies are accepted."""
    from talkingtomachines.authoring.parsers.settings_parser import parse_settings

    for policy in ("terminate", "summarize", "truncate"):
        df = pd.DataFrame(
            [
                {"name": "EXPERIMENT_ID", "value": "exp1"},
                {"name": "MODEL_NAME", "value": "gpt-4o"},
                {"name": "RANDOM_SEED", "value": 42},
                {"name": "NUM_AGENTS_PER_SESSION", "value": 2},
                {"name": "TASK_SEQUENCE", "value": "task1"},
                {"name": "CONTEXT_OVERFLOW_POLICY", "value": policy},
            ]
        )
        config = parse_settings(df)
        assert config.context_overflow_policy == policy


def test_settings_parser_invalid_overflow_policy():
    """Invalid overflow policy raises ValueError."""
    from talkingtomachines.authoring.parsers.settings_parser import parse_settings

    df = pd.DataFrame(
        [
            {"name": "EXPERIMENT_ID", "value": "exp1"},
            {"name": "MODEL_NAME", "value": "gpt-4o"},
            {"name": "RANDOM_SEED", "value": 42},
            {"name": "NUM_AGENTS_PER_SESSION", "value": 2},
            {"name": "TASK_SEQUENCE", "value": "task1"},
            {"name": "CONTEXT_OVERFLOW_POLICY", "value": "invalid_policy"},
        ]
    )
    with pytest.raises(ValueError, match="CONTEXT_OVERFLOW_POLICY"):
        parse_settings(df)


def test_settings_parser_default_overflow_policy():
    """Missing CONTEXT_OVERFLOW_POLICY defaults to 'terminate'."""
    from talkingtomachines.authoring.parsers.settings_parser import parse_settings

    df = pd.DataFrame(
        [
            {"name": "EXPERIMENT_ID", "value": "exp1"},
            {"name": "MODEL_NAME", "value": "gpt-4o"},
            {"name": "RANDOM_SEED", "value": 42},
            {"name": "NUM_AGENTS_PER_SESSION", "value": 2},
            {"name": "TASK_SEQUENCE", "value": "task1"},
        ]
    )
    config = parse_settings(df)
    assert config.context_overflow_policy == "terminate"
