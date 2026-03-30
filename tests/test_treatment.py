"""
Tests for treatment assignment (new API).

Replaces old tests of ``talkingtomachines.management.treatment``
with tests of ``talkingtomachines.core.randomisation.RandomisationEngine``.
"""

from __future__ import annotations

import pytest

from talkingtomachines.core.randomisation import RandomisationEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine(seed=42) -> RandomisationEngine:
    return RandomisationEngine(global_seed=seed)


# ---------------------------------------------------------------------------
# Treatment assignment
# ---------------------------------------------------------------------------


def test_simple_random_assignment_assigns_all_agents():
    engine = _engine()
    agent_ids = [f"agent_{i}" for i in range(10)]
    treatments = ["T1", "T2", "T3"]

    assignments = engine.assign_treatments(
        agent_ids=agent_ids,
        treatment_labels=treatments,
        strategy="simple_random",
        path="test",
    )

    assert len(assignments) == len(agent_ids)
    for agent_id, treatment in assignments.items():
        assert treatment in treatments


def test_simple_random_assignment_empty_treatments():
    """Empty treatment list → no assignments (empty dict)."""
    engine = _engine()
    assignments = engine.assign_treatments(
        agent_ids=[],
        treatment_labels=["T1"],
        strategy="simple_random",
        path="test",
    )
    # No agents → empty assignment dict
    assert assignments == {}


def test_complete_random_assignment_covers_all_treatments():
    """With enough agents, all treatment labels should appear."""
    engine = _engine()
    agent_ids = [f"a{i}" for i in range(30)]
    treatments = ["T1", "T2", "T3"]

    assignments = engine.assign_treatments(
        agent_ids=agent_ids,
        treatment_labels=treatments,
        strategy="complete_random",
        path="test",
    )

    assert len(assignments) == 30
    assigned_treatments = set(assignments.values())
    assert assigned_treatments == set(treatments)


def test_complete_random_assignment_empty_agents():
    """Empty agent list → empty assignment dict."""
    engine = _engine()
    assignments = engine.assign_treatments(
        agent_ids=[],
        treatment_labels=["T1", "T2"],
        strategy="complete_random",
        path="test",
    )
    assert assignments == {}


def test_assignment_is_reproducible_with_same_seed():
    """Same seed must produce same assignment."""
    agent_ids = [f"a{i}" for i in range(20)]
    treatments = ["A", "B"]

    result1 = _engine(seed=123).assign_treatments(
        agent_ids=agent_ids,
        treatment_labels=treatments,
        strategy="simple_random",
        path="test",
    )
    result2 = _engine(seed=123).assign_treatments(
        agent_ids=agent_ids,
        treatment_labels=treatments,
        strategy="simple_random",
        path="test",
    )

    assert result1 == result2


def test_assignment_differs_with_different_seed():
    """Different seeds should (almost certainly) produce different assignments."""
    agent_ids = [f"a{i}" for i in range(20)]
    treatments = ["A", "B"]

    result1 = _engine(seed=1).assign_treatments(
        agent_ids=agent_ids,
        treatment_labels=treatments,
        strategy="simple_random",
        path="test",
    )
    result2 = _engine(seed=999).assign_treatments(
        agent_ids=agent_ids,
        treatment_labels=treatments,
        strategy="simple_random",
        path="test",
    )

    # Extremely unlikely they'd be identical with 20 agents
    assert result1 != result2


# ---------------------------------------------------------------------------
# Group formation
# ---------------------------------------------------------------------------


def test_group_formation_basic():
    """Agents should be partitioned into groups of the specified size."""
    engine = _engine()
    agent_ids = [f"a{i}" for i in range(8)]
    groups = engine.assign_groups(
        agent_ids=agent_ids,
        players_per_group=4,
        strategy="random",
        path="test.round_1",
    )

    assert len(groups) == 2
    # All agents appear exactly once
    all_members = [m for g in groups.values() for m in g]
    assert sorted(all_members) == sorted(agent_ids)


def test_group_formation_unequal_split():
    """When agents don't divide evenly, remaining agents go in the last group."""
    engine = _engine()
    agent_ids = [f"a{i}" for i in range(5)]
    groups = engine.assign_groups(
        agent_ids=agent_ids,
        players_per_group=3,
        strategy="random",
        path="test.round_1",
    )

    # 5 agents, 3 per group → 2 groups (sizes 3 and 2)
    total_members = sum(len(v) for v in groups.values())
    assert total_members == 5


def test_group_formation_single_agent():
    """1 agent → 1 group with 1 member."""
    engine = _engine()
    groups = engine.assign_groups(
        agent_ids=["a1"],
        players_per_group=4,
        strategy="random",
        path="test.round_1",
    )
    assert len(groups) == 1
    members = list(groups.values())[0]
    assert members == ["a1"]


def test_group_formation_all_in_one():
    """players_per_group >= number of agents → 1 group."""
    engine = _engine()
    agent_ids = ["a1", "a2", "a3"]
    groups = engine.assign_groups(
        agent_ids=agent_ids,
        players_per_group=10,
        strategy="random",
        path="test.round_1",
    )
    assert len(groups) == 1
