"""Visibility policies for per-message access control.

This module provides per-message visibility control that determines which
messages each agent can see during an experiment session. Visibility is
enforced by tagging every message with a scope and then filtering the
shared history before it is fed into an agent's context window.

Visibility scopes:
    all_players: Every player in the session can see the message.
    group_only: Only players in the same group this round can see it.
    private: Only the sending player can see it.
    treatment_group: Only players sharing the same treatment condition can see it.
    facilitator: Only visible to the facilitator, invisible to all players.
"""

from __future__ import annotations

from typing import Any


VISIBILITY_ALL_PLAYERS = "all_players"
VISIBILITY_GROUP_ONLY = "group_only"
VISIBILITY_PRIVATE = "private"
VISIBILITY_TREATMENT_GROUP = "treatment_group"
VISIBILITY_FACILITATOR = "facilitator"

_VALID_VISIBILITIES = {
    VISIBILITY_ALL_PLAYERS,
    VISIBILITY_GROUP_ONLY,
    VISIBILITY_PRIVATE,
    VISIBILITY_TREATMENT_GROUP,
    VISIBILITY_FACILITATOR,
}


def filter_message_history(
    shared_history: list[dict],
    requesting_agent_id: str,
    requesting_group_id: str,
    requesting_treatment_label: str,
) -> list[dict]:
    """Filter shared message history based on visibility rules.

    Applies visibility rules to ``shared_history`` and returns the subset
    of messages that the requesting agent is entitled to see. Messages
    with an unrecognized visibility scope default to ``group_only``
    behaviour.

    Args:
        shared_history: Full list of message dicts from the session.
            Each dict may contain keys ``visibility``,
            ``sender_agent_id``, ``group_id``, and
            ``treatment_label``.
        requesting_agent_id: Agent ID of the agent whose visible
            history is being assembled.
        requesting_group_id: Group ID the requesting agent belongs to
            in the current round.
        requesting_treatment_label: Treatment condition label assigned
            to the requesting agent.

    Returns:
        A new list containing only the messages visible to the
        requesting agent.
    """
    visible: list[dict] = []
    for msg in shared_history:
        vis = msg.get("visibility", VISIBILITY_GROUP_ONLY)
        sender = msg.get("sender_agent_id", "")
        msg_group = msg.get("group_id", "")
        msg_treatment = msg.get("treatment_label", "")

        if vis == VISIBILITY_FACILITATOR:
            # Facilitator-only messages are never visible to players
            continue
        elif vis == VISIBILITY_ALL_PLAYERS:
            visible.append(msg)
        elif vis == VISIBILITY_GROUP_ONLY:
            if msg_group == requesting_group_id:
                visible.append(msg)
        elif vis == VISIBILITY_PRIVATE:
            if sender == requesting_agent_id:
                visible.append(msg)
        elif vis == VISIBILITY_TREATMENT_GROUP:
            if msg_treatment == requesting_treatment_label:
                visible.append(msg)
        else:
            # Unknown visibility defaults to group_only
            if msg_group == requesting_group_id:
                visible.append(msg)

    return visible


def make_message(
    role: str,
    content: str,
    sender_agent_id: str = "",
    group_id: str = "",
    treatment_label: str = "",
    visibility: str = VISIBILITY_GROUP_ONLY,
    module: str = "",
    round_number: int = 0,
    extra: dict | None = None,
) -> dict[str, Any]:
    """Build a standardised message dict with visibility metadata.

    Args:
        role: Message role, typically ``"user"`` or ``"assistant"``.
        content: The textual content of the message.
        sender_agent_id: Agent ID of the agent that produced this
            message. Defaults to ``""``.
        group_id: Group the sender belongs to in the current round.
            Defaults to ``""``.
        treatment_label: Treatment condition of the sender. Defaults
            to ``""``.
        visibility: One of the four visibility scope constants.
            Defaults to ``VISIBILITY_GROUP_ONLY``.
        module: Module name that produced this message.
            Defaults to ``""``.
        round_number: Round number within the module. Defaults to ``0``.
        extra: Optional additional key-value pairs to merge into the
            message dict.

    Returns:
        A dict containing all standard message fields plus any extras.
    """
    msg: dict[str, Any] = {
        "role": role,
        "content": content,
        "visibility": visibility,
        "sender_agent_id": sender_agent_id,
        "group_id": group_id,
        "treatment_label": treatment_label,
        "module": module,
        "round_number": round_number,
    }
    if extra:
        msg.update(extra)
    return msg
