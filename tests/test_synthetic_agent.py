"""
Tests for the new ConversationalSyntheticSubject (Phase 9).

Replaces old tests of ``talkingtomachines.generative.synthetic_subject``
with tests of ``talkingtomachines.agents.synthetic_subject``.
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

import pytest

from talkingtomachines.agents.synthetic_subject import ConversationalSyntheticSubject
from talkingtomachines.core.models import (
    Agent,
    Player,
    Group,
    Session,
    PromptDefinition,
    FieldDefinition,
)
from talkingtomachines.core.fields import ExperimentState
from talkingtomachines.gateway.base import LLMProvider, LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockRouter:
    """Minimal stand-in for LLMRouter."""

    def __init__(self, content="mock answer"):
        self._content = content
        self.call_count = 0

    def generate(
        self, message_history, model, temperature=0.0, **kwargs
    ) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            content=self._content,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model=model,
            provider="mock",
            latency_ms=100.0,
            cost_usd=0.0,
        )


def _make_subject(router=None, content="mock answer") -> ConversationalSyntheticSubject:
    agent = Agent(
        agent_id="a1",
        agent_instance_id="inst_a1",
        profile_info={"ID": 1, "age": 30},
    )
    player = Player(
        player_id="p1",
        agent_instance_id="inst_a1",
        agent_id="a1",
        group_id="g1",
    )
    group = Group(
        group_id="g1",
        subsession_id="sub1",
        players=[player],
        turn_order=["inst_a1"],
    )
    session = Session(
        session_id="s1",
        run_id="r1",
        experiment_id="exp1",
        cep_hash="abc",
    )
    state = ExperimentState()
    r = router or MockRouter(content=content)

    return ConversationalSyntheticSubject(
        agent=agent,
        player=player,
        group=group,
        session=session,
        state=state,
        router=r,
        model_name="gpt-4o",
        temperature=0.0,
    )


def _make_prompt(
    prompt_type="DISCUSSION", text="What do you think?"
) -> PromptDefinition:
    return PromptDefinition(
        module="pgg",
        prompt_sequence=1,
        type=prompt_type,
        llm_text=text,
        is_displayed=None,
        field_class=None,
        field_name=None,
    )


def _make_field_def(name="decision", field_type="text") -> FieldDefinition:
    return FieldDefinition(
        field_class="Player",
        module="pgg",
        name=name,
        type=field_type,
        format_response=False,
        generate_speculation_score=False,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_subject_construction():
    subject = _make_subject()
    assert subject.agent.agent_id == "a1"
    assert subject.player.player_id == "p1"


def test_subject_system_message_is_string():
    subject = _make_subject()
    assert isinstance(subject.get_system_message(), str)


# ---------------------------------------------------------------------------
# respond()
# ---------------------------------------------------------------------------


def test_respond_returns_tuple():
    subject = _make_subject(content="5")
    prompt = _make_prompt()
    content, rendered_prompt = subject.respond("pgg", 1, prompt)
    assert isinstance(content, str)
    assert isinstance(rendered_prompt, str)


def test_respond_calls_router():
    router = MockRouter(content="I contribute 10.")
    subject = _make_subject(router=router)
    prompt = _make_prompt(prompt_type="DISCUSSION", text="Discuss.")
    subject.respond("pgg", 1, prompt)
    assert router.call_count == 1


def test_respond_stores_player_state_when_field_def():
    router = MockRouter(content="happy")
    subject = _make_subject(router=router)
    prompt = _make_prompt(prompt_type="PRIVATE_QUESTION", text="How are you?")
    field_def = _make_field_def(name="mood")

    subject.respond("survey", 1, prompt, field_def=field_def)

    stored = subject._state.get_player("p1", "survey", "mood")
    assert stored == "happy"


def test_respond_jinja_rendering_in_prompt():
    router = MockRouter(content="ok")
    subject = _make_subject(router=router)
    # Template refers to player.age — should render without error
    prompt = _make_prompt(
        text="You are {{ player.age }} years old. What is your decision?"
    )
    content, rendered_prompt = subject.respond("pgg", 1, prompt)
    assert isinstance(content, str)
    assert isinstance(rendered_prompt, str)


def test_respond_with_response_options():
    """format_response=True causes JSON parsing attempt — raw fallback if not valid JSON."""
    router = MockRouter(content='{"vote": "yes"}')
    subject = _make_subject(router=router)
    prompt = _make_prompt(prompt_type="PUBLIC_QUESTION")
    field_def = FieldDefinition(
        field_class="Player",
        module="pgg",
        name="vote",
        type="text",
        response_options=["yes", "no"],
        response_options_intro="Choose:",
        randomise_options_order=False,
        validate=False,
        format_response=True,
        generate_speculation_score=False,
    )
    content, rendered_prompt = subject.respond("pgg", 1, prompt, field_def=field_def)
    assert isinstance(content, str)


# ---------------------------------------------------------------------------
# validate_response
# ---------------------------------------------------------------------------


def test_validate_response_valid_option():
    subject = _make_subject()
    field_def = FieldDefinition(
        field_class="Player",
        module="pgg",
        name="decision",
        type="text",
        response_options=["A", "B", "C"],
        format_response=False,
        generate_speculation_score=False,
    )
    assert subject._validate_response("A", ["A", "B", "C"], field_def) is True


def test_validate_response_invalid_option():
    subject = _make_subject()
    field_def = FieldDefinition(
        field_class="Player",
        module="pgg",
        name="decision",
        type="text",
        response_options=["A", "B"],
        format_response=False,
        generate_speculation_score=False,
    )
    assert subject._validate_response("X", ["A", "B"], field_def) is False


# ---------------------------------------------------------------------------
