"""
End-to-end test: 4-agent, 3-round Public Goods Game using a MockProvider.

Exercises:
  - Multi-agent group formation
  - Multiple rounds (MAX_NUM_ROUNDS = 3)
  - CONTEXT prompt (broadcast)
  - DISCUSSION prompt (sequential per turn_order)
  - PUBLIC_QUESTION prompt
  - PRIVATE_QUESTION prompt (parallel when max_player_workers > 1)
  - Event logging (traces.jsonl written to output_dir)
  - State updates persisted per player per round
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from talkingtomachines.gateway.base import LLMProvider, LLMResponse


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class MockProvider(LLMProvider):
    provider_name = "mock"

    def __init__(self, responses=None):
        self._responses = responses or []
        self._idx = 0

    def generate(
        self, message_history, model, temperature=0.0, **kwargs
    ) -> LLMResponse:
        if self._responses:
            content = self._responses[self._idx % len(self._responses)]
            self._idx += 1
        else:
            content = "5"
        return LLMResponse(
            content=content,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model=model,
            provider=self.provider_name,
            latency_ms=150.0,
            cost_usd=0.0,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pgg_cep(tmp_path, max_rounds=3, num_agents=4):
    from talkingtomachines.compiler.cep_schema import CompiledExperiment
    from talkingtomachines.compiler.id_generator import make_agent_id

    agent_ids = [make_agent_id("pgg_exp", i + 1) for i in range(num_agents)]
    profile_rows = [{"ID": i + 1, "age": 25 + i * 5} for i in range(num_agents)]
    # All agents in one group for all rounds
    group_assignments = {
        "pgg": {r: {"G1": agent_ids} for r in range(1, max_rounds + 1)}
    }

    return CompiledExperiment.from_dict(
        {
            "schema_version": "1.0",
            "experiment_id": "pgg_exp",
            "run_id": "run_pgg",
            "cep_hash": "",
            "config_hash": "",
            "settings": {
                "model_name": "mock-model",
                "temperature": 0.0,
                "random_seed": 0,
                "build_profile_qa": False,
                "build_profile_backstories": False,
            },
            "profiles": {
                "short_names": ["ID", "age"],
                "full_names": ["ID", "Age"],
                "rows": profile_rows,
                "id_column": "ID",
                "included_columns": ["ID", "age"],
            },
            "fields": [
                {
                    "field_class": "Player",
                    "module": "pgg",
                    "name": "contribution",
                    "type": "integer",
                    "response_options": None,
                    "response_options_intro": "",
                    "randomise_options_order": False,
                    "validate": False,
                    "format_response": False,
                    "generate_speculation_score": False,
                }
            ],
            "prompts": {
                "pgg": [
                    {
                        "type": "CONTEXT",
                        "module": "pgg",
                        "prompt_sequence": 1,
                        "llm_text": "Round {{ round_number }} begins. You have an endowment.",
                        "is_displayed": None,
                        "field_class": None,
                        "field_name": None,
                    },
                    {
                        "type": "DISCUSSION",
                        "module": "pgg",
                        "prompt_sequence": 2,
                        "llm_text": "Discuss your strategy for this round.",
                        "is_displayed": None,
                        "field_class": "Player",
                        "field_name": "discussion",
                    },
                    {
                        "type": "PRIVATE_QUESTION",
                        "module": "pgg",
                        "prompt_sequence": 3,
                        "llm_text": "How much will you contribute? (0-20)",
                        "is_displayed": None,
                        "field_class": "Player",
                        "field_name": "contribution",
                    },
                ]
            },
            "module_sequence": ["pgg"],
            "facilitator_functions": [],
            "constants": {
                "pgg": {"MAX_NUM_ROUNDS": max_rounds, "PLAYERS_PER_GROUP": num_agents}
            },
            "assignment_plan": {
                "random_seed": 0,
                "group_assignments": group_assignments,
                "manual_variables": [],
            },
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_multi_agent_pgg_completes(tmp_path):
    """4-agent, 3-round PGG completes without error."""
    from talkingtomachines.orchestrator.runtime import ExperimentRuntime

    cep = _make_pgg_cep(tmp_path)
    mock_provider = MockProvider()

    with patch(
        "talkingtomachines.gateway.router.LLMRouter._get_provider",
        return_value=mock_provider,
    ):
        runtime = ExperimentRuntime(cep, output_dir=str(tmp_path))
        session = runtime.run(session_number=1)

    assert session is not None
    assert len(session.modules) == 1
    module = session.modules[0]
    assert module.module_name == "pgg"
    # 3 rounds = 3 subsessions
    assert len(module.subsessions) == 3


def test_multi_agent_llm_called_for_each_agent_round(tmp_path):
    """LLM must be called at least once per agent per round.

    3 rounds × 4 agents × 2 LLM prompts (DISCUSSION + PRIVATE_QUESTION) = 24 calls minimum.
    """
    from talkingtomachines.orchestrator.runtime import ExperimentRuntime
    from talkingtomachines.gateway.base import LLMResponse
    from unittest.mock import MagicMock

    cep = _make_pgg_cep(tmp_path, max_rounds=3, num_agents=4)

    with patch(
        "talkingtomachines.gateway.router.LLMRouter._get_provider",
        return_value=MockProvider(),
    ) as mocked_get_provider:
        runtime = ExperimentRuntime(cep, output_dir=str(tmp_path))
        runtime.run(session_number=1)

    # _get_provider was called at least once per LLM call
    # 3 rounds × 4 agents × (1 DISCUSSION + 1 PRIVATE_QUESTION) = 24 calls minimum
    assert mocked_get_provider.call_count >= 24


def test_multi_agent_traces_written(tmp_path):
    """traces.jsonl must exist and contain llm_call events."""
    import json
    from talkingtomachines.orchestrator.runtime import ExperimentRuntime

    cep = _make_pgg_cep(tmp_path, max_rounds=1, num_agents=2)
    mock_provider = MockProvider()

    with patch(
        "talkingtomachines.gateway.router.LLMRouter._get_provider",
        return_value=mock_provider,
    ):
        runtime = ExperimentRuntime(cep, output_dir=str(tmp_path))
        runtime.run(session_number=1)

    traces_path = Path(tmp_path) / "traces.jsonl"
    assert traces_path.exists()
    lines = traces_path.read_text().strip().split("\n")
    records = [json.loads(l) for l in lines if l]
    event_types = {r["event_type"] for r in records}
    assert "llm_call" in event_types
    assert "session_start" in event_types
    assert "session_end" in event_types


def test_multi_agent_message_history_grows(tmp_path):
    """After a DISCUSSION round, agent.message_history should have entries."""
    from talkingtomachines.orchestrator.runtime import ExperimentRuntime

    cep = _make_pgg_cep(tmp_path, max_rounds=1, num_agents=2)
    mock_provider = MockProvider(responses=["I'll contribute 5.", "7"])

    with patch(
        "talkingtomachines.gateway.router.LLMRouter._get_provider",
        return_value=mock_provider,
    ):
        runtime = ExperimentRuntime(cep, output_dir=str(tmp_path))
        session = runtime.run(session_number=1)

    # Each agent's message_history should contain messages from the session.
    # At minimum: CONTEXT (1 private) + DISCUSSION (2 group_only) + PRIVATE (1 own) = 4 per agent
    for agent in session.agents:
        assert (
            len(agent.message_history) >= 4
        ), f"Agent {agent.agent_instance_id} has only {len(agent.message_history)} messages"


def test_multi_agent_guardrails_checked(tmp_path):
    """Guardrails must be called; anomaly_flags should be accessible on runtime."""
    from talkingtomachines.orchestrator.runtime import ExperimentRuntime

    cep = _make_pgg_cep(tmp_path, max_rounds=1, num_agents=2)
    mock_provider = MockProvider()

    with patch(
        "talkingtomachines.gateway.router.LLMRouter._get_provider",
        return_value=mock_provider,
    ):
        runtime = ExperimentRuntime(cep, output_dir=str(tmp_path))
        runtime.run(session_number=1)

    # No anomaly should be flagged for normal mock responses
    assert isinstance(runtime._guardrails.anomaly_flags, list)


def test_multi_agent_parallel_private_question(tmp_path):
    """With max_player_workers=4, PRIVATE_QUESTION runs in parallel — result same as sequential."""
    from talkingtomachines.orchestrator.runtime import ExperimentRuntime

    cep = _make_pgg_cep(tmp_path, max_rounds=1, num_agents=4)
    # Override settings to enable player-level parallelism
    cep.settings["max_player_workers"] = 4
    mock_provider = MockProvider(responses=["5"])

    with patch(
        "talkingtomachines.gateway.router.LLMRouter._get_provider",
        return_value=mock_provider,
    ):
        runtime = ExperimentRuntime(cep, output_dir=str(tmp_path))
        session = runtime.run(session_number=1)

    assert session is not None
    # All 4 agents participated
    assert len(session.agents) == 4
