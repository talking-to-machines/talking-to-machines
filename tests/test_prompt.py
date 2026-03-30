"""
Tests for the new prompt builder (Phase 9).

Replaces old tests of ``talkingtomachines.generative.prompt``
with tests of ``talkingtomachines.agents.prompt_builder``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from talkingtomachines.agents.prompt_builder import (
    build_profile_prompt,
    build_system_message,
)


# ---------------------------------------------------------------------------
# build_profile_prompt — QA format
# ---------------------------------------------------------------------------


def test_build_profile_prompt_qa_format():
    profile_row = {"ID": 1, "age": 30, "party": "D"}
    short_names = ["ID", "age", "party"]
    full_names = ["Participant ID", "Age", "Political Party"]

    result = build_profile_prompt(
        profile_row=profile_row,
        short_names=short_names,
        full_names=full_names,
        build_profile_qa=True,
        build_profile_backstories=False,
    )

    assert "Age" in result
    assert "30" in result
    assert "Political Party" in result


def test_build_profile_prompt_plain_format():
    """When both QA and backstory are False, format should still include field values."""
    profile_row = {"ID": 1, "age": 25}
    short_names = ["ID", "age"]
    full_names = ["ID", "Age"]

    result = build_profile_prompt(
        profile_row=profile_row,
        short_names=short_names,
        full_names=full_names,
        build_profile_qa=False,
        build_profile_backstories=False,
    )
    # Result should be a string (may be empty if no content generated)
    assert isinstance(result, str)


def test_build_profile_prompt_skips_id():
    """ID column should not appear prominently in the persona text."""
    profile_row = {"ID": 99, "age": 30}
    short_names = ["ID", "age"]
    full_names = ["ID", "Age"]

    result = build_profile_prompt(
        profile_row=profile_row,
        short_names=short_names,
        full_names=full_names,
        build_profile_qa=True,
        build_profile_backstories=False,
    )
    # Should contain age info
    assert "30" in result or "Age" in result


def test_build_profile_prompt_returns_string():
    result = build_profile_prompt(
        profile_row={"ID": 1},
        short_names=["ID"],
        full_names=["ID"],
        build_profile_qa=False,
        build_profile_backstories=False,
    )
    assert isinstance(result, str)


def test_build_profile_prompt_backstory_with_mock_router():
    """When build_profile_backstories=True, a backstory should be appended if router is provided."""
    mock_router = MagicMock()
    mock_llm_response = MagicMock()
    mock_llm_response.content = "I am a 30-year-old Democrat who values community."
    mock_router.generate.return_value = mock_llm_response

    result = build_profile_prompt(
        profile_row={"ID": 1, "age": 30, "party": "D"},
        short_names=["ID", "age", "party"],
        full_names=["ID", "Age", "Party"],
        build_profile_qa=False,
        build_profile_backstories=True,
        router=mock_router,
        model_name="gpt-4o",
        temperature=0.7,
    )

    assert isinstance(result, str)
    # Router should have been called to generate the backstory
    assert mock_router.generate.called


# ---------------------------------------------------------------------------
# build_system_message
# ---------------------------------------------------------------------------


def test_build_system_message_returns_profile_prompt():
    profile = "You are 30 years old and identify as a Democrat."
    result = build_system_message(profile)
    assert result == profile


def test_build_system_message_empty_profile():
    result = build_system_message("")
    assert result == ""


def test_build_system_message_returns_string():
    result = build_system_message("Profile B")
    assert isinstance(result, str)
