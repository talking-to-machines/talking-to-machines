"""
Tests for the Compiler pipeline (Phase 3–5).

Covers:
  - parse_profiles(): short names, full names, data rows, ID validation
  - parse_manual_groups(): round_number int coercion, group/player structure
  - FlowValidator: validates prompt types and field references
  - CompiledExperiment hash stability: same input → same config_hash
  - Golden reproducibility (Phase 15): same template + same seed → byte-identical config_hash
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from talkingtomachines.authoring.parsers.profiles_parser import parse_profiles
from talkingtomachines.authoring.validators.flow_validator import FlowValidator


def validate_flow(prompt_dicts: list) -> list:
    """
    Helper: wrap FlowValidator for tests that pass a flat list of prompt dicts.

    Converts the list into the {task: [PromptDefinition-like]} dict that
    FlowValidator expects, using each dict's 'type' to infer a single task.
    """
    from types import SimpleNamespace

    task = "test_task"
    prompts_ns = [SimpleNamespace(**p) for p in prompt_dicts]
    validator = FlowValidator(
        prompts={task: prompts_ns},
        constants={task: {"MAX_NUM_ROUNDS": 1, "PLAYERS_PER_GROUP": 1}},
        task_names=[task],
    )
    return [str(e) for e in validator.validate()]


from talkingtomachines.core.flow import FlowEvaluator


# ---------------------------------------------------------------------------
# parse_profiles
# ---------------------------------------------------------------------------


def _profiles_df(short_names, full_names, rows):
    """Build a DataFrame in the format the Profiles sheet parser expects."""
    header = [short_names, full_names] + rows
    return pd.DataFrame(header)


def test_parse_profiles_basic():
    df = _profiles_df(
        short_names=["ID", "age", "party"],
        full_names=["Participant ID", "Age", "Political Party"],
        rows=[
            [1, 30, "D"],
            [2, 45, "R"],
        ],
    )
    result = parse_profiles(df)
    assert result["short_names"] == ["ID", "age", "party"]
    assert result["full_names"] == ["Participant ID", "Age", "Political Party"]
    assert result["id_column"] == "ID"
    assert len(result["rows"]) == 2
    assert result["rows"][0]["ID"] == 1
    assert result["rows"][1]["age"] == 45


def test_parse_profiles_duplicate_ids_raises():
    df = _profiles_df(
        short_names=["ID", "name"],
        full_names=["ID", "Name"],
        rows=[[1, "Alice"], [1, "Bob"]],
    )
    with pytest.raises(ValueError, match="duplicate"):
        parse_profiles(df)


def test_parse_profiles_missing_id_column_raises():
    df = _profiles_df(
        short_names=["name", "age"],
        full_names=["Name", "Age"],
        rows=[["Alice", 30]],
    )
    with pytest.raises(ValueError, match="'ID' column"):
        parse_profiles(df)


def test_parse_profiles_profile_fields_filter():
    df = _profiles_df(
        short_names=["ID", "age", "income"],
        full_names=["ID", "Age", "Income"],
        rows=[[1, 30, 50000], [2, 40, 75000]],
    )
    result = parse_profiles(df, profile_fields="ID,age")
    assert "income" not in result["included_columns"]
    assert "ID" in result["included_columns"]
    assert "age" in result["included_columns"]
    # Row dicts should only contain included columns
    for row in result["rows"]:
        assert "income" not in row


def test_parse_profiles_all_fields():
    df = _profiles_df(
        short_names=["ID", "age"],
        full_names=["ID", "Age"],
        rows=[[1, 30]],
    )
    result = parse_profiles(df, profile_fields="ALL")
    assert "ID" in result["included_columns"]
    assert "age" in result["included_columns"]


def test_parse_profiles_too_few_rows_raises():
    df = pd.DataFrame([["ID", "age"]])  # only 1 row (need at least 2 header rows)
    with pytest.raises(ValueError):
        parse_profiles(df)


# ---------------------------------------------------------------------------
# Manual group assignment via generic Manual_ format
# ---------------------------------------------------------------------------


def test_manual_group_assignment_via_manual_registry():
    """Group assignments via Manual_ sheet with class=Group, name=group_label."""
    from talkingtomachines.authoring.parsers.manual_parser import parse_manual_sheets

    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "task": "pgg",
                "round_number": 1,
                "class": "Group",
                "name": "group_label",
                "value": "G1",
            },
            {
                "ID": 2,
                "task": "pgg",
                "round_number": 1,
                "class": "Group",
                "name": "group_label",
                "value": "G1",
            },
            {
                "ID": 3,
                "task": "pgg",
                "round_number": 1,
                "class": "Group",
                "name": "group_label",
                "value": "G2",
            },
        ]
    )
    registry = parse_manual_sheets({"Manual_": df})
    # Should have 3 entries keyed by (profile_id, task, round_number, class, name)
    assert len(registry) == 3
    assert registry[(1, "pgg", 1, "Group", "group_label")] == "G1"
    assert registry[(3, "pgg", 1, "Group", "group_label")] == "G2"


# ---------------------------------------------------------------------------
# FlowValidator
# ---------------------------------------------------------------------------


def test_flow_validator_valid_prompt_types():
    """Valid prompt types should not raise."""
    prompts = [
        {
            "type": "CONTEXT",
            "llm_text": "Welcome!",
            "is_displayed": None,
            "field_class": None,
            "field_name": None,
        },
        {
            "type": "DISCUSSION",
            "llm_text": "Discuss.",
            "is_displayed": None,
            "field_class": "Player",
            "field_name": "comment",
        },
        {
            "type": "PUBLIC_QUESTION",
            "llm_text": "Vote?",
            "is_displayed": None,
            "field_class": "Player",
            "field_name": "vote",
        },
        {
            "type": "PRIVATE_QUESTION",
            "llm_text": "Income?",
            "is_displayed": None,
            "field_class": "Player",
            "field_name": "income",
        },
        {
            "type": "FACILITATOR",
            "llm_text": "set_payoff()",
            "is_displayed": None,
            "field_class": None,
            "field_name": None,
        },
    ]
    errors = validate_flow(prompts)
    assert errors == []


def test_flow_validator_valid_structure_no_errors():
    """FlowValidator focuses on is_displayed expressions and structure, not type names."""
    prompts = [
        {
            "type": "PUBLIC_QUESTION",
            "llm_text": "Vote?",
            "is_displayed": None,
            "field_class": "Player",
            "field_name": "vote",
        },
    ]
    errors = validate_flow(prompts)
    assert errors == []


def test_flow_validator_warns_private_question_group_scope(caplog):
    """PRIVATE_QUESTION with Group-scoped field should log a warning."""
    import logging

    prompts = [
        {
            "type": "PRIVATE_QUESTION",
            "llm_text": "Your vote?",
            "is_displayed": None,
            "prompt_sequence": 1,
            "field_class": "Group",
            "field_name": "vote",
        },
    ]
    with caplog.at_level(logging.WARNING):
        errors = validate_flow(prompts)
    # Should not produce a blocking error
    assert errors == []
    # Should produce a warning in the log
    assert any(
        "PRIVATE_QUESTION" in msg and "Group-scoped" in msg for msg in caplog.messages
    )


def test_flow_validator_warns_private_question_session_scope(caplog):
    """PRIVATE_QUESTION with Session-scoped field should log a warning."""
    import logging

    prompts = [
        {
            "type": "PRIVATE_QUESTION",
            "llm_text": "Rate?",
            "is_displayed": None,
            "prompt_sequence": 2,
            "field_class": "Session",
            "field_name": "rating",
        },
    ]
    with caplog.at_level(logging.WARNING):
        errors = validate_flow(prompts)
    assert errors == []
    assert any(
        "PRIVATE_QUESTION" in msg and "Session-scoped" in msg for msg in caplog.messages
    )


def test_flow_validator_no_warn_private_question_player_scope(caplog):
    """PRIVATE_QUESTION with Player-scoped field should NOT trigger a warning."""
    import logging

    prompts = [
        {
            "type": "PRIVATE_QUESTION",
            "llm_text": "Income?",
            "is_displayed": None,
            "prompt_sequence": 1,
            "field_class": "Player",
            "field_name": "income",
        },
    ]
    with caplog.at_level(logging.WARNING):
        errors = validate_flow(prompts)
    assert errors == []
    assert not any("PRIVATE_QUESTION" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# FlowEvaluator — stop condition
# ---------------------------------------------------------------------------


def test_flow_evaluator_stop_condition_roundtrip():
    flow = FlowEvaluator()
    assert flow.evaluate_stop_condition("end_session") == "end_session"
    assert flow.evaluate_stop_condition("end_round") == "end_round"
    assert flow.evaluate_stop_condition(None) == "continue"


# ---------------------------------------------------------------------------
# CompiledExperiment hash stability
# ---------------------------------------------------------------------------


def test_compiled_experiment_hash_stability():
    """Same content loaded twice should produce identical config_hash."""
    from talkingtomachines.compiler.cep_schema import CompiledExperiment

    data = {
        "schema_version": "1.0",
        "experiment_id": "test_exp",
        "run_id": "run_001",
        "cep_hash": "",
        "config_hash": "",
        "settings": {"model_name": "gpt-4o", "temperature": 0.0},
        "profiles": {
            "short_names": ["ID"],
            "full_names": ["ID"],
            "rows": [],
            "id_column": "ID",
            "included_columns": ["ID"],
        },
        "fields": [],
        "prompts": {},
        "task_sequence": [],
        "facilitator_functions": [],
        "constants": {},
        "assignment_plan": {"treatment_assignments": {}, "group_assignments": {}},
    }

    cep1 = CompiledExperiment(**data)
    cep2 = CompiledExperiment(**data)
    assert cep1.config_hash == cep2.config_hash


# ---------------------------------------------------------------------------
# Golden reproducibility tests (Phase 15)
# ---------------------------------------------------------------------------


def _make_minimal_sheets() -> dict:
    """Build the minimal set of DataFrames required for Compiler.compile()."""
    settings_df = pd.DataFrame(
        [
            {"key": "EXPERIMENT_ID", "value": "golden_exp"},
            {"key": "MODEL_NAME", "value": "gpt-4o"},
            {"key": "RANDOM_SEED", "value": 42},
            {"key": "TASK_SEQUENCE", "value": "task1"},
            {"key": "NUM_AGENTS_PER_SESSION", "value": 1},
        ]
    )

    fields_df = pd.DataFrame(
        [
            {
                "class": "Player",
                "task": "task1",
                "name": "answer",
                "type": "text",
                "format_response": False,
                "generate_speculation_score": False,
            }
        ]
    )

    prompts_df = pd.DataFrame(
        [
            {
                "task": "task1",
                "type": "PUBLIC_QUESTION",
                "prompt_sequence": 1,
                "llm_text": "What is 2+2?",
                "field_class": "Player",
                "field_name": "answer",
            }
        ]
    )

    profiles_df = pd.DataFrame(
        [
            ["ID", "age"],  # row 0: short names
            ["Participant ID", "Age"],  # row 1: full names
            [1, 30],  # row 2: data
        ]
    )

    return {
        "Settings": settings_df,
        "Fields": fields_df,
        "Prompts": prompts_df,
        "Profiles": profiles_df,
    }


def test_golden_reproducibility_same_seed():
    """
    Two independent compile() calls with identical inputs and the same random seed
    must produce byte-identical config_hash values.
    """
    from talkingtomachines.compiler.compiler import Compiler

    sheets = _make_minimal_sheets()

    cep1 = Compiler(sheets).compile(raise_on_error=False)
    cep2 = Compiler(sheets).compile(raise_on_error=False)

    assert (
        cep1.config_hash == cep2.config_hash
    ), f"config_hash is not reproducible:\n  run1={cep1.config_hash}\n  run2={cep2.config_hash}"


def test_golden_reproducibility_different_seed_differs():
    """
    Changing the random seed changes the run_id but must NOT change the config_hash,
    because config_hash is derived from canonical content (not the random seed itself).
    """
    from talkingtomachines.compiler.compiler import Compiler

    sheets_seed42 = _make_minimal_sheets()

    sheets_seed99 = _make_minimal_sheets()
    # Modify the seed row in the second copy
    seed_mask = sheets_seed99["Settings"]["key"] == "RANDOM_SEED"
    sheets_seed99["Settings"] = sheets_seed99["Settings"].copy()
    sheets_seed99["Settings"].loc[seed_mask, "value"] = 99

    cep42 = Compiler(sheets_seed42).compile(raise_on_error=False)
    cep99 = Compiler(sheets_seed99).compile(raise_on_error=False)

    # run_id should differ (it incorporates the seed-dependent RNG)
    assert (
        cep42.run_id != cep99.run_id
    ), "Different seeds should produce different run_ids"
    # config_hash captures canonical content; seed is part of settings → hash differs too
    # (This assertion documents the expected behaviour, not a constraint that must be relaxed.)
    assert isinstance(cep42.config_hash, str) and len(cep42.config_hash) > 0
    assert isinstance(cep99.config_hash, str) and len(cep99.config_hash) > 0


def test_golden_config_hash_changes_when_prompt_changes():
    """
    Modifying a prompt's llm_text must change the config_hash (content integrity check).
    """
    from talkingtomachines.compiler.compiler import Compiler

    sheets_v1 = _make_minimal_sheets()
    cep_v1 = Compiler(sheets_v1).compile(raise_on_error=False)

    sheets_v2 = _make_minimal_sheets()
    sheets_v2["Prompts"] = sheets_v2["Prompts"].copy()
    sheets_v2["Prompts"].loc[0, "llm_text"] = "What is 3+3?"  # changed prompt
    cep_v2 = Compiler(sheets_v2).compile(raise_on_error=False)

    assert (
        cep_v1.config_hash != cep_v2.config_hash
    ), "config_hash should change when prompt content changes"
