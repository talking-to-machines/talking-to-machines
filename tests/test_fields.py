"""
Tests for ExperimentState field management (Phase 6).

Covers:
  - All four scopes (session, agent, group, player): read/write/default
  - Thread-safety (concurrent writes don't corrupt state)
  - build_jinja_context(): namespaced access, treatment shortcut, constants
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from talkingtomachines.core.fields import ExperimentState
from talkingtomachines.core.models import Agent, Player, Group, Session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(agent_id="a1", treatment="T1") -> Agent:
    return Agent(
        agent_id=agent_id,
        agent_instance_id=f"inst_{agent_id}",
        profile_info={"age": 30, "party": "D"},
        treatment_label=treatment,
    )


def _make_player(player_id="p1", agent_instance_id="inst_a1", group_id="g1") -> Player:
    return Player(
        player_id=player_id,
        agent_instance_id=agent_instance_id,
        agent_id="a1",
        group_id=group_id,
    )


def _make_group(group_id="g1") -> Group:
    return Group(
        group_id=group_id,
        subsession_id="sub1",
        players=[],
        turn_order=[],
    )


def _make_session(session_id="s1", run_id="run1") -> Session:
    return Session(
        session_id=session_id,
        run_id=run_id,
        experiment_id="exp1",
        cep_hash="abc",
    )


# ---------------------------------------------------------------------------
# Session scope
# ---------------------------------------------------------------------------


def test_session_set_and_get():
    state = ExperimentState()
    state.set_session("pgg", "total_contributions", 42)
    assert state.get_session("pgg", "total_contributions") == 42


def test_session_get_default():
    state = ExperimentState()
    assert state.get_session("pgg", "missing", default=-1) == -1


def test_session_get_module_returns_dict_copy():
    state = ExperimentState()
    state.set_session("pgg", "x", 1)
    state.set_session("pgg", "y", 2)
    module_data = state.get_session_module("pgg")
    assert module_data == {"x": 1, "y": 2}
    # Mutation of copy should not affect stored state
    module_data["x"] = 999
    assert state.get_session("pgg", "x") == 1


# ---------------------------------------------------------------------------
# Agent scope
# ---------------------------------------------------------------------------


def test_agent_set_and_get():
    state = ExperimentState()
    state.set_agent("a1", "pgg", "contribution", 5)
    assert state.get_agent("a1", "pgg", "contribution") == 5


def test_agent_isolation():
    state = ExperimentState()
    state.set_agent("a1", "pgg", "decision", 10)
    state.set_agent("a2", "pgg", "decision", 20)
    assert state.get_agent("a1", "pgg", "decision") == 10
    assert state.get_agent("a2", "pgg", "decision") == 20


def test_agent_default():
    state = ExperimentState()
    assert state.get_agent("missing", "pgg", "decision", default=0) == 0


# ---------------------------------------------------------------------------
# Group scope
# ---------------------------------------------------------------------------


def test_group_set_and_get():
    state = ExperimentState()
    state.set_group("g1", "survey", "avg_age", 35.5)
    assert state.get_group("g1", "survey", "avg_age") == 35.5


def test_group_default():
    state = ExperimentState()
    assert state.get_group("g1", "pgg", "total", default=0) == 0


# ---------------------------------------------------------------------------
# Player scope
# ---------------------------------------------------------------------------


def test_player_set_and_get():
    state = ExperimentState()
    state.set_player("p1", "pgg", "payoff", 100)
    assert state.get_player("p1", "pgg", "payoff") == 100


def test_player_overwrite():
    state = ExperimentState()
    state.set_player("p1", "pgg", "decision", 5)
    state.set_player("p1", "pgg", "decision", 7)
    assert state.get_player("p1", "pgg", "decision") == 7


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


def test_concurrent_writes_are_safe():
    """Concurrent set_player calls from many threads must not corrupt data."""
    state = ExperimentState()
    errors = []

    def writer(i):
        try:
            state.set_player(f"p{i}", "pgg", "decision", i)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    for i in range(50):
        assert state.get_player(f"p{i}", "pgg", "decision") == i


# ---------------------------------------------------------------------------
# Jinja context construction
# ---------------------------------------------------------------------------


def test_build_jinja_context_basic():
    state = ExperimentState()
    agent = _make_agent("a1", treatment="T1")
    player = _make_player("p1", "inst_a1", "g1")
    group = _make_group("g1")
    session = _make_session("s1", "run1")

    ctx = state.build_jinja_context(
        agent=agent,
        player=player,
        group=group,
        session=session,
        module="pgg",
        round_number=2,
    )

    assert ctx["round_number"] == 2
    assert ctx["group_id"] == "g1"
    assert ctx["session_id"] == "s1"
    assert ctx["run_id"] == "run1"
    assert ctx["treatment"] == "T1"


def test_build_jinja_context_player_profile_fields():
    state = ExperimentState()
    agent = _make_agent("a1")
    player = _make_player("p1")
    group = _make_group("g1")
    session = _make_session("s1", "run1")

    ctx = state.build_jinja_context(
        agent=agent,
        player=player,
        group=group,
        session=session,
        module="pgg",
        round_number=1,
    )
    # profile fields accessible via player namespace
    assert ctx["player"].age == 30
    assert ctx["player"].party == "D"


def test_build_jinja_context_includes_player_state():
    state = ExperimentState()
    state.set_player("p1", "pgg", "payoff", 99)
    agent = _make_agent("a1")
    player = _make_player("p1")
    group = _make_group("g1")
    session = _make_session("s1", "run1")

    ctx = state.build_jinja_context(
        agent=agent,
        player=player,
        group=group,
        session=session,
        module="pgg",
        round_number=1,
    )
    assert ctx["player"].payoff == 99


def test_build_jinja_context_constants_namespace():
    state = ExperimentState()
    agent = _make_agent("a1")
    player = _make_player("p1")
    group = _make_group("g1")
    session = _make_session("s1", "run1")

    constants = {"pgg": {"ENDOWMENT": 20, "MULTIPLIER": 2.0}}
    ctx = state.build_jinja_context(
        agent=agent,
        player=player,
        group=group,
        session=session,
        module="pgg",
        round_number=1,
        constants=constants,
    )
    assert ctx["C"].pgg.ENDOWMENT == 20
    assert ctx["C"].pgg.MULTIPLIER == 2.0


def test_build_jinja_context_no_constants_key():
    state = ExperimentState()
    agent = _make_agent("a1")
    player = _make_player("p1")
    group = _make_group("g1")
    session = _make_session("s1", "run1")

    ctx = state.build_jinja_context(
        agent=agent,
        player=player,
        group=group,
        session=session,
        module="pgg",
        round_number=1,
    )
    assert "C" not in ctx


# ---------------------------------------------------------------------------
# Group/Session scope — accumulation
# ---------------------------------------------------------------------------


def test_accumulate_group_basic():
    """Multiple players writing to the same group field produce a dict keyed by player_id."""
    state = ExperimentState()
    state.accumulate_group("g1", "pgg", "vote", "p1", "yes")
    state.accumulate_group("g1", "pgg", "vote", "p2", "no")
    state.accumulate_group("g1", "pgg", "vote", "p3", "yes")
    result = state.get_group("g1", "pgg", "vote")
    assert result == {"p1": "yes", "p2": "no", "p3": "yes"}


def test_accumulate_session_basic():
    """Multiple players writing to the same session field produce a dict keyed by player_id."""
    state = ExperimentState()
    state.accumulate_session("survey", "satisfaction", "p1", 8)
    state.accumulate_session("survey", "satisfaction", "p2", 6)
    result = state.get_session("survey", "satisfaction")
    assert result == {"p1": 8, "p2": 6}


def test_accumulate_group_overwrites_per_player():
    """If the same player accumulates again, their value is updated."""
    state = ExperimentState()
    state.accumulate_group("g1", "pgg", "vote", "p1", "yes")
    state.accumulate_group("g1", "pgg", "vote", "p1", "no")
    result = state.get_group("g1", "pgg", "vote")
    assert result == {"p1": "no"}


def test_accumulate_group_concurrent():
    """Concurrent accumulate_group calls from many threads must not corrupt data."""
    state = ExperimentState()
    errors = []

    def writer(i):
        try:
            state.accumulate_group("g1", "pgg", "vote", f"p{i}", i)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    result = state.get_group("g1", "pgg", "vote")
    assert len(result) == 50
    for i in range(50):
        assert result[f"p{i}"] == i
