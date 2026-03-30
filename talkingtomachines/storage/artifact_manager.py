"""
Artifact manager.

Manages the run folder structure and orchestrates artifact saving.

Run folder structure::

    experiment_results/
      <experiment_id>/
        <run_id>/
          config.json
          compiled_experiment.json
          assignments.csv
          traces.jsonl
          events.csv
          metrics.csv
          responses.csv
          agent_table.csv
          group_table.csv
          session_table.csv
          codebook.json
          checkpoints/
            <session_id>.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from talkingtomachines.compiler.cep_schema import CompiledExperiment
from talkingtomachines.core.models import Session
from talkingtomachines.core.fields import ExperimentState
from talkingtomachines.storage.codebook import save_codebook
from talkingtomachines.storage.exporters import export_all

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Creates and manages the run folder for a single experiment run.

    The run folder follows a structured layout containing configuration files,
    compiled experiment data, CSV export tables, JSONL traces, and checkpoint
    files.

    Attributes:
        _run_dir: Path to the run directory
            (``<base_dir>/<experiment_id>/<run_id>``).
    """

    def __init__(
        self,
        base_dir: str | Path,
        experiment_id: str,
        run_id: str,
    ):
        """Initialize the artifact manager and create the run directory.

        Args:
            base_dir: Root directory for all experiment results.
            experiment_id: Unique identifier for the experiment.
            run_id: Unique identifier for this particular run.
        """
        self._run_dir = Path(base_dir) / experiment_id / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        """Return the path to the run directory."""
        return self._run_dir

    def save_config(self, cep: CompiledExperiment) -> str:
        """Save the compiled experiment package and a lightweight config summary.

        Writes two files to the run directory:
        - ``compiled_experiment.json``: the full CEP.
        - ``config.json``: a summary of experiment settings.

        Args:
            cep: The compiled experiment package to persist.

        Returns:
            The file path of the saved ``compiled_experiment.json``.
        """
        # Full CEP
        cep_path = self._run_dir / "compiled_experiment.json"
        cep.save(cep_path)

        # Lightweight config summary
        config_path = self._run_dir / "config.json"
        config_path.write_text(
            json.dumps(cep.settings, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Config saved to %s", self._run_dir)
        return str(cep_path)

    def save_all_artifacts(
        self,
        session: Session,
        cep: CompiledExperiment,
        state: ExperimentState,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        total_cost_usd: float = 0.0,
        stop_reason: str = "completed",
    ) -> dict[str, str]:
        """Save all research-ready export artifacts to the run directory.

        Delegates to :func:`exporters.export_all` for CSV tables and
        :func:`codebook.save_codebook` for the codebook JSON.

        Args:
            session: The completed session object containing all experiment
                data.
            cep: The compiled experiment package with field and prompt
                definitions.
            state: The experiment state holding all recorded field values.
            start_time: ISO-format timestamp for when the run started. If
                ``None``, left empty in exports.
            end_time: ISO-format timestamp for when the run ended. Defaults
                to the current UTC time if ``None``.
            total_cost_usd: Cumulative LLM API cost in USD for the run.
            stop_reason: Reason the run terminated (e.g. ``"completed"``,
                ``"budget_exceeded"``).

        Returns:
            A dictionary mapping artifact name to its file path.
        """
        end_time = end_time or datetime.now(timezone.utc).isoformat()
        paths = export_all(
            session=session,
            cep=cep,
            state=state,
            output_dir=self._run_dir,
            start_time=start_time,
            end_time=end_time,
            total_cost_usd=total_cost_usd,
            stop_reason=stop_reason,
        )
        # Codebook
        paths["codebook"] = save_codebook(cep, self._run_dir)
        logger.info("All artifacts saved to %s", self._run_dir)
        return paths
