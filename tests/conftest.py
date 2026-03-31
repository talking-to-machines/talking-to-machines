"""
Shared pytest fixtures for the T2M test suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from talkingtomachines.gateway.base import LLMProvider, LLMResponse
from talkingtomachines.core.models import Agent, Player, Group, Session
from talkingtomachines.core.fields import ExperimentState


# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------


class MockProvider(LLMProvider):
    """Deterministic mock provider — returns a fixed string for every call."""

    provider_name = "mock"

    def __init__(self, response: str = "mock response"):
        self._response = response
        self.call_count = 0

    def generate(
        self, message_history, model, temperature=0.0, **kwargs
    ) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            content=self._response,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model=model,
            provider=self.provider_name,
            latency_ms=100.0,
            cost_usd=0.0,
        )


# ---------------------------------------------------------------------------
# Shared model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def experiment_state():
    return ExperimentState()


@pytest.fixture
def sample_agent():
    return Agent(
        agent_id="agent_001",
        agent_instance_id="inst_agent_001",
        profile_info={"ID": 1, "age": 30, "party": "D"},
        treatment_label="T1",
    )


@pytest.fixture
def sample_player(sample_agent):
    return Player(
        player_id="player_001",
        agent_instance_id=sample_agent.agent_instance_id,
        agent_id=sample_agent.agent_id,
        group_id="group_001",
    )


@pytest.fixture
def sample_group(sample_player):
    return Group(
        group_id="group_001",
        subsession_id="sub_001",
        players=[sample_player],
        turn_order=[sample_player.agent_instance_id],
    )


@pytest.fixture
def sample_session():
    return Session(
        session_id="session_001",
        run_id="run_001",
        experiment_id="test_exp",
        cep_hash="abc123",
    )


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def golden_cep(fixtures_dir):
    from talkingtomachines.compiler.cep_schema import CompiledExperiment
    import json

    data = json.loads((fixtures_dir / "golden_cep.json").read_text())
    return CompiledExperiment(**data)
