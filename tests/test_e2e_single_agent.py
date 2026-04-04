"""
End-to-end test: single-agent questionnaire using a MockProvider.

Exercises the full runtime loop with:
  - 1 agent
  - 1 module ("survey")
  - 1 round
  - 2 prompts: PRIVATE_QUESTION + PUBLIC_QUESTION
  - No actual LLM calls (MockProvider always returns a fixed string)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from talkingtomachines.core.models import (
    Agent,
    Player,
    Group,
    Session,
    Subsession,
    Module,
    PromptDefinition,
    FieldDefinition,
)
from talkingtomachines.core.fields import ExperimentState
from talkingtomachines.gateway.base import LLMProvider, LLMResponse
from talkingtomachines.gateway.router import LLMRouter
from talkingtomachines.gateway.cost_tracker import CostTracker


# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------


class MockProvider(LLMProvider):
    provider_name = "mock"

    def __init__(self, response: str = "Mock response"):
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
# Helpers
# ---------------------------------------------------------------------------


def _make_cep(tmp_path):
    """Return a minimal CompiledExperiment for a 1-agent survey."""
    from talkingtomachines.compiler.cep_schema import CompiledExperiment
    from talkingtomachines.compiler.id_generator import make_agent_id

    agent_id = make_agent_id("test_exp", 1)

    return CompiledExperiment.from_dict(
        {
            "schema_version": "1.0",
            "experiment_id": "test_exp",
            "run_id": "run_e2e_single",
            "cep_hash": "",
            "config_hash": "",
            "settings": {
                "model_name": "mock-model",
                "temperature": 0.0,
                "random_seed": 42,
                "build_profile_qa": False,
                "build_profile_backstories": False,
            },
            "profiles": {
                "short_names": ["ID", "age"],
                "full_names": ["ID", "Age"],
                "rows": [{"ID": 1, "age": 30}],
                "id_column": "ID",
                "included_columns": ["ID", "age"],
            },
            "fields": [
                {
                    "field_class": "Player",
                    "module": "survey",
                    "name": "satisfaction",
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
                "survey": [
                    {
                        "type": "PRIVATE_QUESTION",
                        "module": "survey",
                        "prompt_sequence": 1,
                        "llm_text": "Rate your satisfaction from 1-10.",
                        "is_displayed": None,
                        "field_class": "Player",
                        "field_name": "satisfaction",
                    },
                ]
            },
            "module_sequence": ["survey"],
            "facilitator_functions": [],
            "constants": {"survey": {"MAX_NUM_ROUNDS": 1, "PLAYERS_PER_GROUP": 1}},
            "assignment_plan": {
                "random_seed": 42,
                "group_assignments": {"survey": {1: {"G1": [agent_id]}}},
                "manual_variables": [],
            },
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_agent_e2e(tmp_path):
    """Full runtime loop completes without error for a 1-agent survey."""
    from talkingtomachines.orchestrator.runtime import ExperimentRuntime

    cep = _make_cep(tmp_path)
    mock_provider = MockProvider("7")

    with patch(
        "talkingtomachines.gateway.router.LLMRouter._get_provider",
        return_value=mock_provider,
    ):
        runtime = ExperimentRuntime(cep, output_dir=str(tmp_path))
        session = runtime.run(session_number=1)

    assert session is not None
    assert session.session_id is not None
    # Provider was called at least once (one PRIVATE_QUESTION prompt)
    assert mock_provider.call_count >= 1


def test_single_agent_e2e_cost_is_zero(tmp_path):
    """MockProvider returns 0 cost — total cost should be 0.0."""
    from talkingtomachines.orchestrator.runtime import ExperimentRuntime

    cep = _make_cep(tmp_path)
    mock_provider = MockProvider("7")

    with patch(
        "talkingtomachines.gateway.router.LLMRouter._get_provider",
        return_value=mock_provider,
    ):
        runtime = ExperimentRuntime(cep, output_dir=str(tmp_path))
        runtime.run(session_number=1)

    assert runtime.total_cost_usd == 0.0


def test_single_agent_e2e_state_updated(tmp_path):
    """Player state is written for the PRIVATE_QUESTION field."""
    from talkingtomachines.orchestrator.runtime import ExperimentRuntime
    from talkingtomachines.compiler.id_generator import (
        make_agent_id,
        make_agent_instance_id,
        make_session_id,
        make_module_id,
        make_subsession_id,
        make_group_id,
        make_player_id,
    )

    cep = _make_cep(tmp_path)
    mock_provider = MockProvider("7")

    with patch(
        "talkingtomachines.gateway.router.LLMRouter._get_provider",
        return_value=mock_provider,
    ):
        runtime = ExperimentRuntime(cep, output_dir=str(tmp_path))
        runtime.run(session_number=1)

    # The player state should contain the mock response for "satisfaction"
    agent_id = make_agent_id("test_exp", 1)
    agent_instance_id = make_agent_instance_id("run_e2e_single", agent_id)
    session_id = make_session_id("run_e2e_single", 1)
    module_id = make_module_id(session_id, "survey")
    subsession_id = make_subsession_id(module_id, 1)
    group_id = make_group_id(subsession_id, 1)
    player_id = make_player_id(agent_instance_id, subsession_id)

    satisfaction = runtime.state.get_player(player_id, "survey", "satisfaction")
    # The mock returns "7" — stored as raw string (format_response=False)
    assert satisfaction is not None


def test_single_agent_test_mode(tmp_path):
    """test_mode=True should still complete a single group run."""
    from talkingtomachines.orchestrator.runtime import ExperimentRuntime

    cep = _make_cep(tmp_path)
    mock_provider = MockProvider("5")

    with patch(
        "talkingtomachines.gateway.router.LLMRouter._get_provider",
        return_value=mock_provider,
    ):
        runtime = ExperimentRuntime(cep, output_dir=str(tmp_path))
        session = runtime.run(session_number=1, test_mode=True)

    assert session is not None
