"""
Tests for FlowEvaluator (Phase 8).

Covers:
  - is_displayed(): None → True, literal True/False expressions,
    round_number comparisons, player namespace, error → True fallback
  - evaluate_stop_condition(): continue / end_round / end_session / unknown
"""

from __future__ import annotations

import pytest

from talkingtomachines.core.flow import FlowEvaluator
from talkingtomachines.core.fields import ExperimentState, _Namespace


# ---------------------------------------------------------------------------
# is_displayed — None expression always returns True
# ---------------------------------------------------------------------------


def test_is_displayed_none_always_true():
    flow = FlowEvaluator()
    assert flow.is_displayed(None, {}) is True


def test_is_displayed_empty_string_always_true():
    """Empty string — Jinja2 treats empty condition as truthy path; defaults to True."""
    flow = FlowEvaluator()
    # An empty expression won't parse as a valid condition — error fallback returns True
    assert flow.is_displayed("", {}) is True


# ---------------------------------------------------------------------------
# is_displayed — round_number conditions
# ---------------------------------------------------------------------------


def test_is_displayed_round_number_eq_true():
    flow = FlowEvaluator()
    assert flow.is_displayed("round_number == 1", {"round_number": 1}) is True


def test_is_displayed_round_number_eq_false():
    flow = FlowEvaluator()
    assert flow.is_displayed("round_number == 1", {"round_number": 2}) is False


def test_is_displayed_round_number_in_list():
    flow = FlowEvaluator()
    ctx = {"round_number": 3}
    assert flow.is_displayed("round_number in [2, 3, 4]", ctx) is True
    assert flow.is_displayed("round_number in [2, 3, 4]", {"round_number": 5}) is False


def test_is_displayed_round_number_gt():
    flow = FlowEvaluator()
    assert flow.is_displayed("round_number > 2", {"round_number": 3}) is True
    assert flow.is_displayed("round_number > 2", {"round_number": 2}) is False


# ---------------------------------------------------------------------------
# is_displayed — player namespace
# ---------------------------------------------------------------------------


def test_is_displayed_player_treatment():
    flow = FlowEvaluator()
    ctx = {"player": _Namespace({"treatment": "T1", "age": 30})}
    assert flow.is_displayed("player.treatment == 'T1'", ctx) is True
    assert flow.is_displayed("player.treatment == 'T2'", ctx) is False


def test_is_displayed_player_numeric_field():
    flow = FlowEvaluator()
    ctx = {"player": _Namespace({"democrat": 1})}
    assert flow.is_displayed("player.democrat == 1", ctx) is True
    assert flow.is_displayed("player.democrat == 0", ctx) is False


# ---------------------------------------------------------------------------
# is_displayed — error fallback
# ---------------------------------------------------------------------------


def test_is_displayed_invalid_expression_defaults_to_true():
    """Malformed Jinja2 expression: should log a warning and return True."""
    flow = FlowEvaluator()
    assert flow.is_displayed("this is not valid jinja", {}) is True


def test_is_displayed_undefined_variable_raises_error():
    flow = FlowEvaluator()
    # With StrictUndefined, referencing an undefined variable should raise ValueError
    import pytest

    with pytest.raises(ValueError, match="references an undefined variable"):
        flow.is_displayed("unknown_var == 1", {})


# ---------------------------------------------------------------------------
# evaluate_stop_condition
# ---------------------------------------------------------------------------


def test_evaluate_stop_condition_none():
    flow = FlowEvaluator()
    assert flow.evaluate_stop_condition(None) == "continue"


def test_evaluate_stop_condition_continue_variants():
    flow = FlowEvaluator()
    for val in ("continue", "CONTINUE", "Continue", "  continue  ", 0, "", "other"):
        assert flow.evaluate_stop_condition(val) == "continue"


def test_evaluate_stop_condition_end_round():
    flow = FlowEvaluator()
    assert flow.evaluate_stop_condition("end_round") == "end_round"
    assert flow.evaluate_stop_condition("END_ROUND") == "end_round"
    assert flow.evaluate_stop_condition("  end_round  ") == "end_round"


def test_evaluate_stop_condition_end_session():
    flow = FlowEvaluator()
    assert flow.evaluate_stop_condition("end_session") == "end_session"
    assert flow.evaluate_stop_condition("END_SESSION") == "end_session"
    assert flow.evaluate_stop_condition("  End_Session  ") == "end_session"
