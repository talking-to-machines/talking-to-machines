"""
Tests for the visibility policy filter (Phase 7).

Covers:
  - filter_message_history(): all four visibility scopes
  - make_message(): message construction helper
  - Unknown visibility scope fallback (defaults to group_only)
  - Mixed-visibility history filtering
"""

from __future__ import annotations

import pytest

from talkingtomachines.core.visibility import (
    filter_message_history,
    make_message,
    VISIBILITY_ALL_PLAYERS,
    VISIBILITY_GROUP_ONLY,
    VISIBILITY_PRIVATE,
    VISIBILITY_TREATMENT_GROUP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(
    content: str,
    visibility: str = VISIBILITY_GROUP_ONLY,
    sender: str = "agent_a",
    group_id: str = "G1",
    treatment: str = "T1",
) -> dict:
    return make_message(
        role="assistant",
        content=content,
        sender_agent_id=sender,
        group_id=group_id,
        treatment_label=treatment,
        visibility=visibility,
    )


def _filter(
    history: list,
    requesting_agent: str = "agent_a",
    group_id: str = "G1",
    treatment: str = "T1",
) -> list:
    return filter_message_history(
        shared_history=history,
        requesting_agent_id=requesting_agent,
        requesting_group_id=group_id,
        requesting_treatment_label=treatment,
    )


# ---------------------------------------------------------------------------
# make_message
# ---------------------------------------------------------------------------


def test_make_message_has_required_fields():
    msg = make_message(
        role="user",
        content="Hello",
        sender_agent_id="a1",
        group_id="G1",
        treatment_label="T1",
        visibility=VISIBILITY_GROUP_ONLY,
    )
    assert msg["role"] == "user"
    assert msg["content"] == "Hello"
    assert msg["sender_agent_id"] == "a1"
    assert msg["group_id"] == "G1"
    assert msg["treatment_label"] == "T1"
    assert msg["visibility"] == VISIBILITY_GROUP_ONLY


def test_make_message_extra_fields_merged():
    msg = make_message(
        role="assistant",
        content="Hi",
        extra={"prompt_id": "p1", "round": 2},
    )
    assert msg["prompt_id"] == "p1"
    assert msg["round"] == 2


def test_make_message_defaults():
    msg = make_message(role="user", content="text")
    assert msg["visibility"] == VISIBILITY_GROUP_ONLY
    assert msg["sender_agent_id"] == ""
    assert msg["group_id"] == ""
    assert msg["treatment_label"] == ""


# ---------------------------------------------------------------------------
# VISIBILITY_ALL_PLAYERS
# ---------------------------------------------------------------------------


def test_all_players_visible_to_any_agent():
    history = [_msg("broadcast", visibility=VISIBILITY_ALL_PLAYERS, group_id="G1")]
    # Agent in same group sees it
    assert len(_filter(history, requesting_agent="agent_a", group_id="G1")) == 1
    # Agent in a different group also sees it
    assert len(_filter(history, requesting_agent="agent_b", group_id="G2")) == 1


def test_all_players_visible_regardless_of_treatment():
    history = [_msg("global", visibility=VISIBILITY_ALL_PLAYERS, treatment="T1")]
    assert len(_filter(history, requesting_agent="agent_x", treatment="T2")) == 1


# ---------------------------------------------------------------------------
# VISIBILITY_GROUP_ONLY
# ---------------------------------------------------------------------------


def test_group_only_visible_within_same_group():
    history = [_msg("group msg", visibility=VISIBILITY_GROUP_ONLY, group_id="G1")]
    assert len(_filter(history, group_id="G1")) == 1


def test_group_only_hidden_from_other_group():
    history = [_msg("group msg", visibility=VISIBILITY_GROUP_ONLY, group_id="G1")]
    assert len(_filter(history, group_id="G2")) == 0


def test_group_only_is_default_scope():
    """Messages without an explicit visibility field default to group_only."""
    msg = {"role": "assistant", "content": "implicit", "group_id": "G1"}
    history = [msg]
    assert len(_filter(history, group_id="G1")) == 1
    assert len(_filter(history, group_id="G2")) == 0


# ---------------------------------------------------------------------------
# VISIBILITY_PRIVATE
# ---------------------------------------------------------------------------


def test_private_visible_only_to_sender():
    history = [_msg("private msg", visibility=VISIBILITY_PRIVATE, sender="agent_a")]
    # Requesting agent is the sender
    assert len(_filter(history, requesting_agent="agent_a")) == 1
    # Different agent cannot see it
    assert len(_filter(history, requesting_agent="agent_b")) == 0


def test_private_same_group_but_different_agent_cannot_see():
    history = [
        _msg("my note", visibility=VISIBILITY_PRIVATE, sender="agent_a", group_id="G1")
    ]
    assert len(_filter(history, requesting_agent="agent_b", group_id="G1")) == 0


# ---------------------------------------------------------------------------
# VISIBILITY_TREATMENT_GROUP
# ---------------------------------------------------------------------------


def test_treatment_group_visible_to_same_treatment():
    history = [_msg("T1 only", visibility=VISIBILITY_TREATMENT_GROUP, treatment="T1")]
    assert len(_filter(history, treatment="T1")) == 1


def test_treatment_group_hidden_from_other_treatment():
    history = [_msg("T1 only", visibility=VISIBILITY_TREATMENT_GROUP, treatment="T1")]
    assert len(_filter(history, treatment="T2")) == 0


def test_treatment_group_independent_of_group_id():
    """Same treatment in different groups can still see each other's treatment-scoped messages."""
    history = [
        _msg(
            "T1 msg",
            visibility=VISIBILITY_TREATMENT_GROUP,
            treatment="T1",
            group_id="G1",
        )
    ]
    # Agent in G2 but same treatment T1
    assert len(_filter(history, group_id="G2", treatment="T1")) == 1


# ---------------------------------------------------------------------------
# Unknown visibility fallback
# ---------------------------------------------------------------------------


def test_unknown_visibility_falls_back_to_group_only():
    msg = _msg("unknown scope", visibility="some_future_scope", group_id="G1")
    history = [msg]
    assert len(_filter(history, group_id="G1")) == 1
    assert len(_filter(history, group_id="G2")) == 0


# ---------------------------------------------------------------------------
# Mixed-visibility history
# ---------------------------------------------------------------------------


def test_mixed_history_returns_correct_subset():
    """A history with all four visibility types filtered correctly."""
    history = [
        _msg(
            "global", visibility=VISIBILITY_ALL_PLAYERS, sender="agent_b", group_id="G2"
        ),
        _msg(
            "group G1",
            visibility=VISIBILITY_GROUP_ONLY,
            sender="agent_b",
            group_id="G1",
        ),
        _msg(
            "group G2",
            visibility=VISIBILITY_GROUP_ONLY,
            sender="agent_b",
            group_id="G2",
        ),
        _msg(
            "private a", visibility=VISIBILITY_PRIVATE, sender="agent_a", group_id="G1"
        ),
        _msg(
            "private b", visibility=VISIBILITY_PRIVATE, sender="agent_b", group_id="G1"
        ),
        _msg("T1 msg", visibility=VISIBILITY_TREATMENT_GROUP, treatment="T1"),
        _msg("T2 msg", visibility=VISIBILITY_TREATMENT_GROUP, treatment="T2"),
    ]
    # agent_a in G1, treatment T1
    visible = _filter(
        history, requesting_agent="agent_a", group_id="G1", treatment="T1"
    )
    contents = [m["content"] for m in visible]

    assert "global" in contents  # all_players
    assert "group G1" in contents  # same group
    assert "group G2" not in contents  # different group
    assert "private a" in contents  # own private message
    assert "private b" not in contents  # other agent's private
    assert "T1 msg" in contents  # same treatment
    assert "T2 msg" not in contents  # different treatment


def test_empty_history_returns_empty():
    assert _filter([]) == []


def test_all_private_none_visible_to_other_agent():
    history = [
        _msg("note 1", visibility=VISIBILITY_PRIVATE, sender="agent_a"),
        _msg("note 2", visibility=VISIBILITY_PRIVATE, sender="agent_a"),
    ]
    assert _filter(history, requesting_agent="agent_b") == []


def test_ordering_preserved():
    """Visible messages must retain their original order."""
    history = [
        _msg("first", visibility=VISIBILITY_ALL_PLAYERS),
        _msg("second", visibility=VISIBILITY_ALL_PLAYERS),
        _msg("third", visibility=VISIBILITY_ALL_PLAYERS),
    ]
    visible = _filter(history)
    assert [m["content"] for m in visible] == ["first", "second", "third"]
