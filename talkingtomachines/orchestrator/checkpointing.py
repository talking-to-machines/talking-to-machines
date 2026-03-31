"""Session checkpointing for fault-tolerant experiment execution.

Saves and loads session state as JSON files at subsession boundaries,
enabling experiment resumption after interruptions or failures.

Checkpoint path convention::

    <run_folder>/checkpoints/<session_id>.json
    <run_folder>/checkpoints/<session_id>_state.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from talkingtomachines.core.fields import ExperimentState
from talkingtomachines.core.models import Session

logger = logging.getLogger(__name__)


class Checkpointer:
    """Manages checkpoint save/load for experiment sessions.

    Persists ``Session`` objects as JSON files in a ``checkpoints/``
    subdirectory within the run folder. Supports saving, loading, and
    existence checks by session ID.

    Attributes:
        _checkpoint_dir: Path to the ``checkpoints/`` directory.
    """

    def __init__(self, run_folder: str | Path) -> None:
        """Initialize the checkpointer.

        Args:
            run_folder: Root directory for the experiment run. A
                ``checkpoints/`` subdirectory is created automatically.
        """
        self._checkpoint_dir = Path(run_folder) / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, session: Session, state: Optional[ExperimentState] = None) -> str:
        """Save session and experiment state to JSON checkpoint files.

        Serializes the session via ``session.to_dict()`` and writes it
        as pretty-printed JSON. If an ``ExperimentState`` is provided,
        it is saved alongside as ``<session_id>_state.json``.

        Args:
            session: The session object to persist.
            state: Optional experiment state to persist alongside the
                session checkpoint.

        Returns:
            Absolute filesystem path to the saved session checkpoint file.
        """
        path = self._checkpoint_dir / f"{session.session_id}.json"
        data = session.to_dict()
        path.write_text(
            json.dumps(data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

        if state is not None:
            state_path = self._checkpoint_dir / f"{session.session_id}_state.json"
            state_path.write_text(
                json.dumps(state.to_dict(), indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )

        logger.info("Checkpoint saved: %s", path)
        return str(path)

    def load(
        self, session_id: str
    ) -> tuple[Optional[Session], Optional[ExperimentState]]:
        """Load a session checkpoint and experiment state from disk.

        Args:
            session_id: The unique session identifier to look up.

        Returns:
            A tuple of ``(Session, ExperimentState)``. Either element
            may be ``None`` — the session is ``None`` when no checkpoint
            file exists, and the state is ``None`` when no state file
            exists (backward compatibility with older checkpoints).
        """
        path = self._checkpoint_dir / f"{session_id}.json"
        if not path.exists():
            return None, None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            session = Session.from_dict(data)
            logger.info("Checkpoint loaded: %s", path)
        except Exception as exc:
            logger.error("Failed to load checkpoint %s: %s", path, exc)
            return None, None

        # Load experiment state if available
        state: Optional[ExperimentState] = None
        state_path = self._checkpoint_dir / f"{session_id}_state.json"
        if state_path.exists():
            try:
                state_data = json.loads(state_path.read_text(encoding="utf-8"))
                state = ExperimentState.from_dict(state_data)
                logger.info("Experiment state loaded: %s", state_path)
            except Exception as exc:
                logger.error("Failed to load experiment state %s: %s", state_path, exc)

        return session, state

    def exists(self, session_id: str) -> bool:
        """Check whether a checkpoint file exists for the given session.

        Args:
            session_id: The unique session identifier to look up.

        Returns:
            True if a checkpoint file exists on disk, False otherwise.
        """
        return (self._checkpoint_dir / f"{session_id}.json").exists()
