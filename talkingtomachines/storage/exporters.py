"""
Multi-table CSV exporter

Produces research-ready export tables from a completed Session object.

Tables:
  session_table.csv   — one row per session
  agent_table.csv     — one row per agent per session
  group_table.csv     — one row per group per subsession
  player.csv          — one row per player per subsession
  assignments.csv     — one row per agent per module per round
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from talkingtomachines.core.models import Session
from talkingtomachines.compiler.cep_schema import CompiledExperiment
from talkingtomachines.core.fields import ExperimentState

logger = logging.getLogger(__name__)


def _compute_run_time_sec(
    start_time: Optional[str], end_time: Optional[str]
) -> Optional[float]:
    """Parse ISO timestamps and return elapsed seconds, or ``None`` on failure.

    Args:
        start_time: ISO-format start timestamp string.
        end_time: ISO-format end timestamp string.

    Returns:
        Elapsed seconds rounded to two decimals, or ``None`` if either
        timestamp is missing or unparseable.
    """
    if not start_time or not end_time:
        return None
    try:
        t0 = datetime.fromisoformat(start_time)
        t1 = datetime.fromisoformat(end_time)
        return round((t1 - t0).total_seconds(), 2)
    except (ValueError, TypeError):
        return None


def export_all(
    session: Session,
    cep: CompiledExperiment,
    state: ExperimentState,
    output_dir: str | Path,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    total_cost_usd: float = 0.0,
    stop_reason: str = "completed",
) -> dict[str, str]:
    """Export all research-ready CSV tables to the specified directory.

    Produces ``session_table.csv``, ``agent_table.csv``, ``group_table.csv``,
    ``responses.csv``, ``assignments.csv``, ``metrics.csv``, and optionally
    ``events.csv`` (if ``traces.jsonl`` exists).

    Args:
        session: The completed session object containing experiment data.
        cep: The compiled experiment package with field and module definitions.
        state: The experiment state holding all recorded field values.
        output_dir: Directory where CSV files will be written.
        start_time: ISO-format timestamp for when the run started.
        end_time: ISO-format timestamp for when the run ended.
        total_cost_usd: Cumulative LLM API cost in USD for the run.
        stop_reason: Reason the run terminated (e.g. ``"completed"``).

    Returns:
        A dictionary mapping table name (e.g. ``"session_table"``) to its
        file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    paths["session_table"] = _export_session_table(
        session, cep, out, start_time, end_time, total_cost_usd, stop_reason
    )
    paths["agent_table"] = _export_agent_table(session, cep, state, out)
    paths["group_table"] = _export_group_table(session, cep, state, out)
    paths["player"] = _export_player_table(session, cep, state, out)
    paths["assignments"] = _export_assignments(session, cep, out)
    paths["metrics"] = _export_metrics(
        session, cep, out, total_cost_usd, start_time, end_time
    )

    # traces.jsonl is written live by EventLogger during the run;
    # include its path in the returned map if it exists.
    traces_path = out / "traces.jsonl"
    if traces_path.exists():
        paths["traces"] = str(traces_path)
        # events.csv — filtered view of traces covering retries, errors, tool calls,
        # stop reasons (all non llm_call event types per design doc)
        paths["events"] = _export_events_csv(traces_path, out)

    return paths


# ---------------------------------------------------------------------------
# Table exporters
# ---------------------------------------------------------------------------


def _export_session_table(
    session: Session,
    cep: CompiledExperiment,
    out: Path,
    start_time: Optional[str],
    end_time: Optional[str],
    total_cost_usd: float,
    stop_reason: str,
) -> str:
    """Export session-level metadata to ``session_table.csv``.

    Produces one row per session with run identifiers, timing, cost, and
    stop reason.

    Args:
        session: The session object.
        cep: The compiled experiment package.
        out: Output directory path.
        start_time: ISO-format run start timestamp.
        end_time: ISO-format run end timestamp.
        total_cost_usd: Total API cost in USD.
        stop_reason: Reason the run terminated.

    Returns:
        The file path of the exported CSV.
    """
    path = str(out / "session_table.csv")
    rows = [
        {
            "run_id": session.run_id,
            "session_id": session.session_id,
            "experiment_id": session.experiment_id,
            "config_hash": cep.config_hash,
            "cep_hash": cep.cep_hash,
            "start_time": start_time or "",
            "end_time": end_time or "",
            "total_cost_usd": total_cost_usd,
            "run_time_sec": _compute_run_time_sec(start_time, end_time) or "",
            "stop_reason": stop_reason,
        }
    ]
    _write_csv(path, rows)
    return path


def _export_agent_table(
    session: Session,
    cep: CompiledExperiment,
    state: ExperimentState,
    out: Path,
) -> str:
    """Export agent-level data to ``agent_table.csv``.

    Produces one row per agent per session, including profile fields and
    agent-scoped field values for each module.

    Args:
        session: The session object.
        cep: The compiled experiment package.
        state: The experiment state with recorded field values.
        out: Output directory path.

    Returns:
        The file path of the exported CSV.
    """
    path = str(out / "agent_table.csv")
    rows: list[dict] = []
    for agent in session.agents:
        row: dict[str, Any] = {
            "session_id": session.session_id,
            "agent_id": agent.agent_id,
            "agent_instance_id": agent.agent_instance_id,
            "treatment_label": agent.treatment_label,
        }
        # Profile fields
        row.update(agent.profile_info)

        # Agent-scoped field values for each module
        for module in cep.module_sequence:
            agent_fields = state.get_agent_module(agent.agent_id, module)
            for k, v in agent_fields.items():
                row[f"{module}.{k}"] = v

        # System message and message history for debugging
        row["system_message"] = agent.state.get("system_message", "")
        row["message_history"] = json.dumps(
            agent.message_history, ensure_ascii=False, default=str
        )

        rows.append(row)
    _write_csv(path, rows)
    return path


def _export_group_table(
    session: Session,
    cep: CompiledExperiment,
    state: ExperimentState,
    out: Path,
) -> str:
    """Export group-level data to ``group_table.csv``.

    Produces one row per group per subsession, including group-scoped
    field values.

    Args:
        session: The session object.
        cep: The compiled experiment package.
        state: The experiment state with recorded field values.
        out: Output directory path.

    Returns:
        The file path of the exported CSV.
    """
    path = str(out / "group_table.csv")
    rows: list[dict] = []
    for module in session.modules:
        for subsession in module.subsessions:
            for group in subsession.groups:
                row: dict[str, Any] = {
                    "session_id": session.session_id,
                    "module_id": module.module_id,
                    "subsession_id": subsession.subsession_id,
                    "group_id": group.group_id,
                    "round_number": subsession.round_number,
                    "module": module.module_name,
                    "num_players": len(group.players),
                }
                # Group-scoped field values
                group_fields = state.get_group_module(
                    group.group_id, module.module_name
                )
                row.update(group_fields)
                rows.append(row)
    _write_csv(path, rows)
    return path


def _export_player_table(
    session: Session,
    cep: CompiledExperiment,
    state: ExperimentState,
    out: Path,
) -> str:
    """Export player-level responses to ``player.csv``.

    Produces one row per player per subsession, including player-scoped
    field values.

    Args:
        session: The session object.
        cep: The compiled experiment package.
        state: The experiment state with recorded field values.
        out: Output directory path.

    Returns:
        The file path of the exported CSV.
    """
    path = str(out / "player.csv")
    rows: list[dict] = []
    for module in session.modules:
        for subsession in module.subsessions:
            for group in subsession.groups:
                for player in group.players:
                    row: dict[str, Any] = {
                        "session_id": session.session_id,
                        "player_id": player.player_id,
                        "agent_instance_id": player.agent_instance_id,
                        "agent_id": player.agent_id,
                        "group_id": group.group_id,
                        "subsession_id": subsession.subsession_id,
                        "round_number": subsession.round_number,
                        "module": module.module_name,
                    }
                    # Player-scoped field values
                    player_fields = state.get_player_module(
                        player.player_id, module.module_name
                    )
                    row.update(player_fields)
                    rows.append(row)
    _write_csv(path, rows)
    return path


def _export_assignments(
    session: Session,
    cep: CompiledExperiment,
    out: Path,
) -> str:
    """Export treatment and group assignments to ``assignments.csv``.

    Produces one row per agent per module per round, documenting group
    membership and treatment labels.

    Args:
        session: The session object.
        cep: The compiled experiment package with assignment plan.
        out: Output directory path.

    Returns:
        The file path of the exported CSV.
    """
    path = str(out / "assignments.csv")
    plan = cep.assignment_plan
    rows: list[dict] = []
    for agent in session.agents:
        for module in cep.module_sequence:
            module_groups = plan.group_assignments.get(module, {})
            for round_num, groups in module_groups.items():
                group_id = ""
                for gid, member_ids in groups.items():
                    if agent.agent_id in member_ids:
                        group_id = gid
                        break
                treatment_label = plan.get_treatment(
                    agent.agent_id, module, int(round_num)
                )
                if not treatment_label:
                    treatment_label = agent.treatment_label
                rows.append(
                    {
                        "session_id": session.session_id,
                        "agent_id": agent.agent_id,
                        "agent_instance_id": agent.agent_instance_id,
                        "module": module,
                        "round_number": round_num,
                        "treatment_label": treatment_label,
                        "group_id": group_id,
                    }
                )
    _write_csv(path, rows)
    return path


def _export_metrics(
    session: Session,
    cep: CompiledExperiment,
    out: Path,
    total_cost_usd: float,
    start_time: Optional[str],
    end_time: Optional[str],
) -> str:
    """Export aggregate metrics to ``metrics.csv``.

    Produces one row per session with experiment-level statistics
    including agent counts, module counts, total rounds, groups, players,
    messages, cumulative API cost, and timing information.

    Args:
        session: The session object containing experiment hierarchy data.
        cep: The compiled experiment package with module definitions.
        out: Output directory path.
        total_cost_usd: Cumulative LLM API cost in USD for the run.
        start_time: ISO-format timestamp for when the run started.
        end_time: ISO-format timestamp for when the run ended.

    Returns:
        The file path of the exported CSV.
    """
    path = str(out / "metrics.csv")

    # Count totals
    total_groups = sum(
        len(sub.groups) for module in session.modules for sub in module.subsessions
    )
    total_players = sum(
        len(group.players)
        for module in session.modules
        for sub in module.subsessions
        for group in sub.groups
    )
    total_messages = sum(len(agent.message_history) for agent in session.agents)
    total_rounds = sum(len(module.subsessions) for module in session.modules)

    rows = [
        {
            "session_id": session.session_id,
            "run_id": session.run_id,
            "experiment_id": session.experiment_id,
            "num_agents": len(session.agents),
            "num_modules": len(cep.module_sequence),
            "num_rounds_total": total_rounds,
            "num_groups_total": total_groups,
            "num_players_total": total_players,
            "num_messages_total": total_messages,
            "total_cost_usd": total_cost_usd,
            "run_time_sec": _compute_run_time_sec(start_time, end_time) or "",
            "start_time": start_time or "",
            "end_time": end_time or "",
        }
    ]
    _write_csv(path, rows)
    return path


def _export_events_csv(traces_path: Path, out: Path) -> str:
    """Extract non-LLM-call events from ``traces.jsonl`` into ``events.csv``.

    Includes retries, errors, tool calls, stop-rule evaluations, facilitator
    calls, validation outcomes, checkpoint saves/loads, and budget checks
    (all event types except ``llm_call``, ``session_start``, and
    ``session_end``).

    Args:
        traces_path: Path to the ``traces.jsonl`` file written by
            :class:`EventLogger` during the run.
        out: Output directory path.

    Returns:
        The file path of the exported ``events.csv``.
    """
    path = str(out / "events.csv")

    # Event types to include in events.csv (non-routine LLM call events)
    _INCLUDE_TYPES = {
        "llm_retry",
        "llm_fallback",
        "facilitator_call",
        "facilitator_llm_call",
        "validation_pass",
        "validation_fail",
        "stop_rule_evaluated",
        "budget_check",
        "tool_call",
        "rag_call",
        "checkpoint_saved",
        "checkpoint_loaded",
        "randomisation",
        "group_assignment",
        "treatment_assignment",
    }

    # Structural keys written by EventLogger for every record; everything else
    # is event-specific payload merged in via record.update(data).
    _STRUCT_KEYS = {
        "timestamp",
        "event_type",
        "run_id",
        "session_id",
        "group_id",
        "agent_instance_id",
    }

    rows: list[dict] = []
    try:
        with traces_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_type = record.get("event_type", "")
                if event_type in _INCLUDE_TYPES:
                    # EventLogger.log() merges data keys into the top-level record
                    # via record.update(data), so reconstruct the payload by
                    # excluding the known structural keys.
                    data_payload = {
                        k: v for k, v in record.items() if k not in _STRUCT_KEYS
                    }
                    rows.append(
                        {
                            "timestamp": record.get("timestamp", ""),
                            "event_type": event_type,
                            "run_id": record.get("run_id", ""),
                            "session_id": record.get("session_id", ""),
                            "group_id": record.get("group_id", ""),
                            "agent_instance_id": record.get("agent_instance_id", ""),
                            "data": json.dumps(data_payload, ensure_ascii=False),
                        }
                    )
    except OSError:
        pass

    _write_csv(path, rows)
    return path


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _write_csv(path: str, rows: list[dict]) -> None:
    """Write a list of row dictionaries to a CSV file.

    Field names are inferred from the keys of the first row. Additional
    keys found in subsequent rows are appended to the column list. If
    ``rows`` is empty, an empty file is created.

    Args:
        path: Destination file path.
        rows: List of dictionaries where each dictionary represents one
            CSV row.
    """
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    # Collect any extra keys from later rows
    for row in rows[1:]:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
