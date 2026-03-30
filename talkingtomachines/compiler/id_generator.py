"""
ID generator (Phase 3).

Derives all stable, deterministic IDs per the ID derivation scheme:

    experiment_id   → from Settings sheet
    config_hash     → SHA256 of canonical JSON of all parsed sheets (16 chars)
    cep_hash        → SHA256 of serialized CEP
    run_id          → {experiment_id}_{datetime_utc}_{random_suffix}
    session_id      → {run_id}_s{n}
    module_id       → {session_id}_{task_name}
    subsession_id   → {module_id}_r{round_number}
    group_id        → {subsession_id}_g{group_number}
    agent_id        → {experiment_id}_a{profile_ID}
    agent_instance_id → {run_id}_{agent_id}
    player_id       → {agent_instance_id}_{subsession_id}
    turn_id         → {group_id}_t{turn_number}_{agent_instance_id}
"""

from __future__ import annotations

import hashlib
import json
import random
import string
from datetime import datetime, timezone
from typing import Any


def _sha256_truncated(data: str | bytes, length: int = 16) -> str:
    """Compute a truncated SHA-256 hex digest.

    Args:
        data: Input data as a string or bytes. Strings are encoded
            to UTF-8 before hashing.
        length: Number of hex characters to return from the digest.

    Returns:
        The first *length* characters of the SHA-256 hex digest.
    """
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()[:length]


def compute_config_hash(parsed_sheets: dict[str, Any]) -> str:
    """Compute a truncated SHA-256 hash of all parsed worksheet data.

    Serialises the parsed sheets to canonical JSON (sorted keys,
    non-ASCII preserved) and returns the first 16 hex characters of
    the SHA-256 digest.

    Args:
        parsed_sheets: Dictionary of all parsed worksheet data, as
            returned by the template parser pipeline.

    Returns:
        A 16-character hexadecimal config hash string.
    """
    canonical = json.dumps(
        parsed_sheets, sort_keys=True, default=str, ensure_ascii=False
    )
    return _sha256_truncated(canonical, 16)


def compute_cep_hash(cep_dict: dict) -> str:
    """Compute the full SHA-256 hash of a serialised CEP dictionary.

    Serialises the compiled experiment package (CEP) to canonical JSON
    and returns the complete SHA-256 hex digest.

    Args:
        cep_dict: The CEP dictionary to hash, typically produced by
            ``ExperimentConfig.to_dict()``.

    Returns:
        A 64-character hexadecimal SHA-256 digest string.
    """
    canonical = json.dumps(cep_dict, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def make_run_id(experiment_id: str, rng: random.Random | None = None) -> str:
    """Generate a unique run ID with a UTC timestamp and random suffix.

    Format: ``{experiment_id}_{YYYYMMDDTHHMMSSZ}_{6-char suffix}``.

    Args:
        experiment_id: The experiment identifier from the Settings
            worksheet.
        rng: Optional ``random.Random`` instance for deterministic
            suffix generation. A new instance is created if ``None``.

    Returns:
        A unique run ID string.
    """
    rng = rng or random.Random()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "".join(rng.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{experiment_id}_{ts}_{suffix}"


def make_session_id(run_id: str, session_number: int) -> str:
    """Derive a session ID from a run ID and session number.

    Format: ``{run_id}_s{session_number}``.

    Args:
        run_id: The parent run identifier.
        session_number: Zero-based or one-based session index.

    Returns:
        A deterministic session ID string.
    """
    return f"{run_id}_s{session_number}"


def make_module_id(session_id: str, task_name: str) -> str:
    """Derive a module ID from a session ID and task name.

    Format: ``{session_id}_{task_name}``.

    Args:
        session_id: The parent session identifier.
        task_name: Name of the task (from ``TASK_SEQUENCE``).

    Returns:
        A deterministic module ID string.
    """
    return f"{session_id}_{task_name}"


def make_subsession_id(module_id: str, round_number: int) -> str:
    """Derive a subsession ID from a module ID and round number.

    Format: ``{module_id}_r{round_number}``.

    Args:
        module_id: The parent module identifier.
        round_number: The round number within the task.

    Returns:
        A deterministic subsession ID string.
    """
    return f"{module_id}_r{round_number}"


def make_group_id(subsession_id: str, group_number: int) -> str:
    """Derive a group ID from a subsession ID and group number.

    Format: ``{subsession_id}_g{group_number}``.

    Args:
        subsession_id: The parent subsession identifier.
        group_number: The group number within the subsession.

    Returns:
        A deterministic group ID string.
    """
    return f"{subsession_id}_g{group_number}"


def make_agent_id(experiment_id: str, profile_id: Any) -> str:
    """Derive an agent ID from an experiment ID and profile ID.

    Format: ``{experiment_id}_a{profile_id}``.

    Args:
        experiment_id: The experiment identifier from the Settings
            worksheet.
        profile_id: The agent's profile identifier (from the Profiles
            worksheet).

    Returns:
        A deterministic agent ID string.
    """
    return f"{experiment_id}_a{profile_id}"


def make_agent_instance_id(run_id: str, agent_id: str) -> str:
    """Derive an agent instance ID from a run ID and agent ID.

    An agent instance represents a specific agent participating in a
    specific run. Format: ``{run_id}_{agent_id}``.

    Args:
        run_id: The parent run identifier.
        agent_id: The agent identifier (from ``make_agent_id``).

    Returns:
        A deterministic agent instance ID string.
    """
    return f"{run_id}_{agent_id}"


def make_player_id(agent_instance_id: str, subsession_id: str) -> str:
    """Derive a player ID from an agent instance ID and subsession ID.

    A player represents an agent instance's participation in a
    specific subsession (round). Format:
    ``{agent_instance_id}_{subsession_id}``.

    Args:
        agent_instance_id: The agent instance identifier.
        subsession_id: The subsession identifier.

    Returns:
        A deterministic player ID string.
    """
    return f"{agent_instance_id}_{subsession_id}"


def make_turn_id(group_id: str, turn_number: int, agent_instance_id: str) -> str:
    """Derive a turn ID from a group ID, turn number, and agent instance.

    Format: ``{group_id}_t{turn_number}_{agent_instance_id}``.

    Args:
        group_id: The parent group identifier.
        turn_number: The sequential turn number within the group.
        agent_instance_id: The agent instance taking this turn.

    Returns:
        A deterministic turn ID string.
    """
    return f"{group_id}_t{turn_number}_{agent_instance_id}"
