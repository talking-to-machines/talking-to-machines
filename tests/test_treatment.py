"""
Tests for randomisation utilities (group formation, shuffling, seeding).

Tests of ``talkingtomachines.core.randomisation.RandomisationEngine``.
Treatment assignment strategies have been removed — treatment is now
a regular field set via the Manual_ sheet.
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


def test_group_formation_reproducible_with_same_seed():
    """Same seed must produce same groups."""
    agent_ids = [f"a{i}" for i in range(8)]
    result1 = _engine(seed=123).assign_groups(
        agent_ids=agent_ids,
        players_per_group=4,
        strategy="random",
        path="test.round_1",
    )
    result2 = _engine(seed=123).assign_groups(
        agent_ids=agent_ids,
        players_per_group=4,
        strategy="random",
        path="test.round_1",
    )
    assert result1 == result2


def test_group_formation_differs_with_different_seed():
    """Different seeds should (almost certainly) produce different groups."""
    agent_ids = [f"a{i}" for i in range(20)]
    result1 = _engine(seed=1).assign_groups(
        agent_ids=agent_ids,
        players_per_group=4,
        strategy="random",
        path="test.round_1",
    )
    result2 = _engine(seed=999).assign_groups(
        agent_ids=agent_ids,
        players_per_group=4,
        strategy="random",
        path="test.round_1",
    )
    assert result1 != result2
