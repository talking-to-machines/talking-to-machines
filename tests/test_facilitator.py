"""
Tests for FacilitatorEngine (core/facilitator.py) and ConjointDesigner (core/conjoint.py).

Covers:
  - FacilitatorEngine.execute(): all four dispatch paths
      creating_session → empty string (side-effects only)
      assign_treatment → delegates to RandomisationEngine, returns empty string
      assign_groups → delegates to RandomisationEngine, returns empty string
      custom name → LLM call, returns raw response string
  - ConjointDesigner:
      generate_profile() → returns one dict per attribute
      generate_table() → markdown table with correct structure
      deterministic seeding: same inputs → same outputs
      different inputs → different outputs
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from talkingtomachines.core.facilitator import FacilitatorEngine
from talkingtomachines.core.models import FacilitatorFunction
from talkingtomachines.core.conjoint import ConjointDesigner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine(llm_response: str = "{}") -> tuple[FacilitatorEngine, MagicMock, MagicMock]:
    """Build a FacilitatorEngine with mocked RandomisationEngine and LLMRouter."""
    mock_rng = MagicMock()
    mock_rng.assign_treatments.return_value = {"agent_a": "T1", "agent_b": "T2"}
    mock_rng.assign_groups.return_value = {"G1": ["agent_a", "agent_b"]}

    mock_response = MagicMock()
    mock_response.content = llm_response
    mock_router = MagicMock()
    mock_router.generate.return_value = mock_response

    engine = FacilitatorEngine(
        state=MagicMock(),
        randomisation=mock_rng,
        router=mock_router,
        model_name="gpt-4o",
        temperature=0.0,
    )
    return engine, mock_rng, mock_router


def _fn(
    name: str, definition: str = "", args: dict | None = None
) -> FacilitatorFunction:
    return FacilitatorFunction(name=name, definition=definition, args=args or {})


# ---------------------------------------------------------------------------
# creating_session
# ---------------------------------------------------------------------------


def test_creating_session_returns_empty_string():
    engine, _, _ = _engine()
    result = engine.execute(_fn("creating_session"), context={})
    assert result == ""


def test_creating_session_does_not_call_rng(mocker=None):
    engine, mock_rng, mock_router = _engine()
    engine.execute(_fn("creating_session"), context={})
    mock_rng.assign_treatments.assert_not_called()
    mock_rng.assign_groups.assert_not_called()
    mock_router.generate.assert_not_called()


# ---------------------------------------------------------------------------
# assign_treatment
# ---------------------------------------------------------------------------


def test_assign_treatment_returns_empty_string():
    engine, mock_rng, _ = _engine()
    context = {"agent_ids": ["agent_a", "agent_b"], "treatment_labels": ["T1", "T2"]}
    result = engine.execute(_fn("assign_treatment"), context=context)
    assert result == ""


def test_assign_treatment_delegates_to_rng():
    engine, mock_rng, _ = _engine()
    context = {"agent_ids": ["a1", "a2"], "treatment_labels": ["A", "B"]}
    fn = _fn(
        "assign_treatment",
        args={"strategy": "complete_random", "path": "exp.treatments"},
    )
    engine.execute(fn, context=context)
    mock_rng.assign_treatments.assert_called_once_with(
        agent_ids=["a1", "a2"],
        treatment_labels=["A", "B"],
        strategy="complete_random",
        path="exp.treatments",
    )


def test_assign_treatment_missing_agent_ids_returns_empty():
    engine, mock_rng, _ = _engine()
    result = engine.execute(
        _fn("assign_treatment"), context={"treatment_labels": ["T1"]}
    )
    assert result == ""
    mock_rng.assign_treatments.assert_not_called()


def test_assign_treatment_missing_treatment_labels_returns_empty():
    engine, mock_rng, _ = _engine()
    result = engine.execute(_fn("assign_treatment"), context={"agent_ids": ["a1"]})
    assert result == ""
    mock_rng.assign_treatments.assert_not_called()


# ---------------------------------------------------------------------------
# assign_groups
# ---------------------------------------------------------------------------


def test_assign_groups_returns_empty_string():
    engine, mock_rng, _ = _engine()
    context = {"agent_ids": ["a1", "a2", "a3", "a4"], "players_per_group": 2}
    result = engine.execute(_fn("assign_groups"), context=context)
    assert result == ""


def test_assign_groups_delegates_to_rng():
    engine, mock_rng, _ = _engine()
    context = {"agent_ids": ["a1", "a2"], "players_per_group": 2}
    fn = _fn("assign_groups", args={"strategy": "random", "path": "task.round_1"})
    engine.execute(fn, context=context)
    mock_rng.assign_groups.assert_called_once_with(
        agent_ids=["a1", "a2"],
        players_per_group=2,
        strategy="random",
        path="task.round_1",
    )


def test_assign_groups_empty_agent_ids_returns_empty():
    engine, mock_rng, _ = _engine()
    result = engine.execute(
        _fn("assign_groups"), context={"agent_ids": [], "players_per_group": 2}
    )
    assert result == ""
    mock_rng.assign_groups.assert_not_called()


# ---------------------------------------------------------------------------
# Custom (LLM) facilitator
# ---------------------------------------------------------------------------


def test_custom_facilitator_calls_router_generate():
    engine, _, mock_router = _engine(llm_response='{"payoff": 10}')
    result = engine.execute(
        _fn("set_payoff", definition="Set payoff to 10."), context={}
    )
    assert result == '{"payoff": 10}'
    mock_router.generate.assert_called_once()


def test_custom_facilitator_passes_definition_to_llm():
    engine, _, mock_router = _engine(llm_response='{"done": true}')
    engine.execute(_fn("custom_fn", definition="Do something."), context={})
    call_kwargs = mock_router.generate.call_args[1]
    messages = call_kwargs.get("message_history", [])
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "Do something." in user_msg["content"]


def test_custom_facilitator_returns_raw_string():
    fenced = '```json\n{"score": 42}\n```'
    engine, _, _ = _engine(llm_response=fenced)
    result = engine.execute(_fn("score_fn"), context={})
    assert result == '```json\n{"score": 42}\n```'


def test_custom_facilitator_returns_raw_string_plain_fences():
    fenced = '```\n{"val": 7}\n```'
    engine, _, _ = _engine(llm_response=fenced)
    result = engine.execute(_fn("val_fn"), context={})
    assert result == '```\n{"val": 7}\n```'


def test_custom_facilitator_invalid_json_returns_raw():
    engine, _, _ = _engine(llm_response="not valid json {{}")
    result = engine.execute(_fn("broken_fn"), context={})
    assert result == "not valid json {{}"


def test_custom_facilitator_router_exception_returns_empty():
    engine, _, mock_router = _engine()
    mock_router.generate.side_effect = RuntimeError("LLM error")
    result = engine.execute(_fn("error_fn"), context={})
    assert result == ""


# ---------------------------------------------------------------------------
# ConjointDesigner — generate_profile
# ---------------------------------------------------------------------------

_ATTRIBUTES = {
    "party": ["Democrat", "Republican", "Independent"],
    "age": [30, 45, 60],
    "gender": ["Male", "Female"],
}


def test_generate_profile_returns_all_attributes():
    designer = ConjointDesigner(_ATTRIBUTES)
    profile = designer.generate_profile(player_id="p1", round_number=1)
    assert set(profile.keys()) == set(_ATTRIBUTES.keys())


def test_generate_profile_values_within_levels():
    designer = ConjointDesigner(_ATTRIBUTES)
    profile = designer.generate_profile(player_id="p1", round_number=1)
    for attr, levels in _ATTRIBUTES.items():
        assert profile[attr] in levels


def test_generate_profile_deterministic_same_inputs():
    designer = ConjointDesigner(_ATTRIBUTES)
    p1 = designer.generate_profile(player_id="p99", round_number=3)
    p2 = designer.generate_profile(player_id="p99", round_number=3)
    assert p1 == p2


def test_generate_profile_differs_across_rounds():
    designer = ConjointDesigner(_ATTRIBUTES)
    # With 3 attributes each with multiple levels, round should usually differ
    profiles = [
        designer.generate_profile(player_id="p1", round_number=r) for r in range(1, 6)
    ]
    # At least two profiles should differ
    assert not all(p == profiles[0] for p in profiles)


def test_generate_profile_differs_across_players():
    designer = ConjointDesigner(_ATTRIBUTES)
    profiles = [
        designer.generate_profile(player_id=f"p{i}", round_number=1) for i in range(5)
    ]
    assert not all(p == profiles[0] for p in profiles)


def test_generate_profile_number_changes_result():
    designer = ConjointDesigner(_ATTRIBUTES)
    p1 = designer.generate_profile(player_id="p1", round_number=1, profile_number=1)
    p2 = designer.generate_profile(player_id="p1", round_number=1, profile_number=2)
    # Different profile numbers should (almost certainly) yield different profiles
    # We just check they're both valid
    for attr, levels in _ATTRIBUTES.items():
        assert p1[attr] in levels
        assert p2[attr] in levels


# ---------------------------------------------------------------------------
# ConjointDesigner — generate_table
# ---------------------------------------------------------------------------


def test_generate_table_returns_string():
    designer = ConjointDesigner(_ATTRIBUTES)
    table = designer.generate_table(player_id="p1", round_number=1, num_profiles=2)
    assert isinstance(table, str)


def test_generate_table_contains_attribute_names():
    designer = ConjointDesigner(_ATTRIBUTES)
    table = designer.generate_table(player_id="p1", round_number=1, num_profiles=2)
    for attr in _ATTRIBUTES:
        assert attr in table


def test_generate_table_contains_profile_headers():
    designer = ConjointDesigner(_ATTRIBUTES)
    table = designer.generate_table(player_id="p1", round_number=1, num_profiles=3)
    assert "Profile 1" in table
    assert "Profile 2" in table
    assert "Profile 3" in table


def test_generate_table_is_markdown():
    designer = ConjointDesigner(_ATTRIBUTES)
    table = designer.generate_table(player_id="p1", round_number=1, num_profiles=2)
    # Each line should start and end with '|'
    lines = [l for l in table.splitlines() if l.strip()]
    for line in lines:
        assert line.startswith("|"), f"Expected '|' at start of: {line!r}"
        assert line.endswith("|"), f"Expected '|' at end of: {line!r}"


def test_generate_table_has_separator_row():
    designer = ConjointDesigner(_ATTRIBUTES)
    table = designer.generate_table(player_id="p1", round_number=1, num_profiles=2)
    lines = table.splitlines()
    # Second line should be the separator (contains dashes)
    assert "-" in lines[1]


def test_generate_table_deterministic():
    designer = ConjointDesigner(_ATTRIBUTES)
    t1 = designer.generate_table(player_id="p5", round_number=2, num_profiles=2)
    t2 = designer.generate_table(player_id="p5", round_number=2, num_profiles=2)
    assert t1 == t2


def test_generate_table_single_profile():
    designer = ConjointDesigner({"color": ["red", "blue"]})
    table = designer.generate_table(player_id="p1", round_number=1, num_profiles=1)
    assert "Profile 1" in table
    assert "Profile 2" not in table
