"""
Event logger (Phase 13).

Appends structured event records to a JSONL file.  Thread-safe via file lock.

Event types logged:
  llm_call, llm_retry, llm_fallback, facilitator_call, facilitator_llm_call,
  validation_pass, validation_fail, randomisation, group_assignment,
  treatment_assignment, stop_rule_evaluated, budget_check, tool_call,
  rag_call, checkpoint_saved, checkpoint_loaded
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EventLogger:
    """Thread-safe append-mode JSONL event logger.

    Each event is written as a single JSON line to the log file. File-level
    thread safety is ensured via a ``threading.Lock``.

    Attributes:
        _path: Path to the JSONL log file.
        _lock: Threading lock used to synchronize concurrent writes.
    """

    def __init__(self, log_path: str | Path):
        """Initialize the event logger.

        Creates parent directories for the log file if they do not already
        exist.

        Args:
            log_path: Filesystem path for the JSONL log file.
        """
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        event_type: str,
        run_id: str = "",
        session_id: str = "",
        group_id: str = "",
        agent_instance_id: str = "",
        data: Optional[dict[str, Any]] = None,
        **extra,
    ) -> None:
        """Append a single event record to the JSONL log.

        Args:
            event_type: Category of the event (e.g. ``"llm_call"``,
                ``"validation_fail"``).
            run_id: Identifier for the current experiment run.
            session_id: Identifier for the current session.
            group_id: Identifier for the group associated with this event.
            agent_instance_id: Identifier for the agent instance that
                triggered the event.
            data: Optional dictionary of additional event payload fields
                merged into the record.
            **extra: Arbitrary keyword arguments merged into the record.
        """
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "run_id": run_id,
            "session_id": session_id,
            "group_id": group_id,
            "agent_instance_id": agent_instance_id,
        }
        if data:
            record.update(data)
        if extra:
            record.update(extra)

        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
