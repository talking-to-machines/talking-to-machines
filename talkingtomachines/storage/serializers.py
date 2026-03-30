"""
Storage serializers (Phase 13 supplement).

Centralised serialization helpers used across the storage layer.

Provides:
  - ``to_json_safe``    — recursively convert arbitrary objects to JSON-safe types
  - ``serialize_session`` — convert a Session object tree to a plain dict
  - ``deserialize_session`` — restore a Session from a plain dict
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# JSON-safe conversion
# ---------------------------------------------------------------------------


def to_json_safe(obj: Any) -> Any:
    """Recursively convert an arbitrary object to a JSON-serializable form.

    Handles primitives, ``datetime``, ``Path``, sets, lists, tuples, dicts,
    objects with a ``to_dict()`` method, and dataclasses. Falls back to
    ``str()`` for unsupported types.

    Args:
        obj: The object to convert.

    Returns:
        A JSON-safe equivalent (composed of ``None``, ``bool``, ``int``,
        ``float``, ``str``, ``list``, and ``dict`` values).
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, (datetime,)):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (set, frozenset)):
        return [to_json_safe(v) for v in sorted(str(v) for v in obj)]

    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]

    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}

    if hasattr(obj, "to_dict"):
        return to_json_safe(obj.to_dict())

    # Dataclass fallback
    try:
        import dataclasses

        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return to_json_safe(dataclasses.asdict(obj))
    except Exception:
        pass

    # Last resort: str
    return str(obj)


# ---------------------------------------------------------------------------
# Serialization helpers for experiment data structures
# ---------------------------------------------------------------------------


def serialize_message(msg: dict) -> dict:
    """Ensure a message dictionary is JSON-safe.

    Args:
        msg: A message dictionary (typically from a conversation history).

    Returns:
        A recursively sanitized copy of the message dictionary.
    """
    return to_json_safe(msg)


def serialize_player(player: Any) -> dict:
    """Serialize a Player object to a plain dictionary.

    Args:
        player: A ``Player`` instance with ``player_id``,
            ``agent_instance_id``, ``agent_id``, and ``group_id`` attributes.

    Returns:
        A dictionary containing the player's identifying fields.
    """
    return {
        "player_id": player.player_id,
        "agent_instance_id": player.agent_instance_id,
        "agent_id": player.agent_id,
        "group_id": player.group_id,
    }


def serialize_group(group: Any) -> dict:
    """Serialize a Group object to a plain dictionary.

    Args:
        group: A ``Group`` instance with ``group_id``, ``subsession_id``,
            ``players``, and ``turn_order`` attributes.

    Returns:
        A dictionary containing group metadata, serialized players, and
        turn order.
    """
    return {
        "group_id": group.group_id,
        "subsession_id": group.subsession_id,
        "players": [serialize_player(p) for p in group.players],
        "turn_order": list(group.turn_order),
    }


def serialize_subsession(subsession: Any) -> dict:
    """Serialize a Subsession object to a plain dictionary.

    Args:
        subsession: A ``Subsession`` instance with ``subsession_id``,
            ``module_id``, ``round_number``, and ``groups`` attributes.

    Returns:
        A dictionary containing subsession metadata and serialized groups.
    """
    return {
        "subsession_id": subsession.subsession_id,
        "module_id": subsession.module_id,
        "round_number": subsession.round_number,
        "groups": [serialize_group(g) for g in subsession.groups],
    }


def serialize_module(module: Any) -> dict:
    """Serialize a Module object to a plain dictionary.

    Args:
        module: A ``Module`` instance with ``module_id``, ``session_id``,
            ``task_name``, and ``subsessions`` attributes.

    Returns:
        A dictionary containing module metadata and serialized subsessions.
    """
    return {
        "module_id": module.module_id,
        "session_id": module.session_id,
        "task_name": module.task_name,
        "subsessions": [serialize_subsession(s) for s in module.subsessions],
    }


def serialize_agent(agent: Any) -> dict:
    """Serialize an Agent object to a plain dictionary.

    Args:
        agent: An ``Agent`` instance with ``agent_id``,
            ``agent_instance_id``, ``profile_info``, ``treatment_label``,
            and optional ``is_human`` and ``state`` attributes.

    Returns:
        A dictionary containing agent metadata, profile information,
        treatment label, human flag, and agent state.
    """
    return {
        "agent_id": agent.agent_id,
        "agent_instance_id": agent.agent_instance_id,
        "profile_info": to_json_safe(agent.profile_info),
        "treatment_label": agent.treatment_label,
        "is_human": getattr(agent, "is_human", False),
        "state": to_json_safe(getattr(agent, "state", {})),
    }


def serialize_session(session: Any) -> dict:
    """Serialize a full Session object tree to a plain dictionary.

    This produces the canonical checkpoint format used for saving and
    restoring experiment state.

    Args:
        session: A ``Session`` instance with ``session_id``, ``run_id``,
            ``experiment_id``, ``cep_hash``, ``agents``, and ``modules``
            attributes.

    Returns:
        A nested dictionary representing the entire session hierarchy
        (session -> modules -> subsessions -> groups -> players).
    """
    return {
        "session_id": session.session_id,
        "run_id": session.run_id,
        "experiment_id": session.experiment_id,
        "cep_hash": session.cep_hash,
        "agents": [serialize_agent(a) for a in session.agents],
        "modules": [serialize_module(m) for m in session.modules],
    }


def deserialize_session(data: dict) -> Any:
    """Restore a Session from a serialized dictionary.

    Reconstructs the full object hierarchy: ``Session`` -> ``Module`` ->
    ``Subsession`` -> ``Group`` -> ``Player``, plus the list of ``Agent``
    objects.

    Args:
        data: A dictionary previously produced by :func:`serialize_session`.

    Returns:
        A ``Session`` object with fully populated ``agents`` and ``modules``.
    """
    from talkingtomachines.core.models import (
        Session,
        Agent,
        Player,
        Group,
        Subsession,
        Module,
    )

    session = Session(
        session_id=data["session_id"],
        run_id=data["run_id"],
        experiment_id=data["experiment_id"],
        cep_hash=data.get("cep_hash", ""),
    )

    session.agents = [
        Agent(
            agent_id=a["agent_id"],
            agent_instance_id=a["agent_instance_id"],
            profile_info=a.get("profile_info", {}),
            treatment_label=a.get("treatment_label", ""),
            is_human=a.get("is_human", False),
            state=a.get("state", {}),
        )
        for a in data.get("agents", [])
    ]

    for m_data in data.get("modules", []):
        module = Module(
            module_id=m_data["module_id"],
            session_id=m_data["session_id"],
            task_name=m_data["task_name"],
        )
        for s_data in m_data.get("subsessions", []):
            subsession = Subsession(
                subsession_id=s_data["subsession_id"],
                module_id=s_data["module_id"],
                round_number=s_data["round_number"],
            )
            for g_data in s_data.get("groups", []):
                players = [
                    Player(
                        player_id=p["player_id"],
                        agent_instance_id=p["agent_instance_id"],
                        agent_id=p["agent_id"],
                        group_id=p["group_id"],
                    )
                    for p in g_data.get("players", [])
                ]
                group = Group(
                    group_id=g_data["group_id"],
                    subsession_id=g_data["subsession_id"],
                    players=players,
                    turn_order=g_data.get("turn_order", []),
                )
                subsession.groups.append(group)
            module.subsessions.append(subsession)
        session.modules.append(module)

    return session


# ---------------------------------------------------------------------------
# JSON file helpers
# ---------------------------------------------------------------------------


def load_json(path: str | Path) -> Any:
    """Load a JSON file and return its parsed contents.

    Args:
        path: Filesystem path to the JSON file.

    Returns:
        The deserialized Python object (typically a ``dict`` or ``list``).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data as a JSON file, creating parent directories as needed.

    The data is first converted to a JSON-safe form via :func:`to_json_safe`
    before serialization.

    Args:
        data: The object to serialize and write.
        path: Destination file path.
        indent: Number of spaces for JSON indentation.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(to_json_safe(data), indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
