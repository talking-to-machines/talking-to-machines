"""Three-tier parallelism model for experiment execution.

Provides concurrent execution at three levels of the experiment hierarchy:

- **Session-level**: ``ProcessPoolExecutor`` runs multiple sessions as
  separate processes so that global state cannot interfere across sessions.
- **Group-level**: ``ThreadPoolExecutor`` runs groups within a subsession
  concurrently, suitable for I/O-bound LLM calls.
- **Player-level**: ``ThreadPoolExecutor`` runs PRIVATE_QUESTION prompts
  concurrently within a single group.

Session-level usage::

    from talkingtomachines.orchestrator.parallel import run_sessions_parallel
    results = run_sessions_parallel(
        cep_path="run_dir/compiled_experiment.json",
        output_base="experiment_results",
        session_numbers=[1, 2, 3],
        budget_cap_usd=5.0,
        test_mode=False,
        max_workers=3,
    )
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Module-level worker — must be a top-level function to be picklable
# ---------------------------------------------------------------------------


@dataclass
class SessionResult:
    """Outcome of a single session execution in a parallel run.

    Attributes:
        session_number: Numeric index of the session.
        session_id: Unique identifier assigned to the session.
        total_cost_usd: Cumulative LLM API cost for the session in USD.
        output_dir: Filesystem path where session artifacts were written.
        success: Whether the session completed without errors.
        error: Error message if the session failed; empty string on success.
    """

    session_number: int
    session_id: str
    total_cost_usd: float
    output_dir: str
    success: bool
    error: str = ""


def _session_worker(
    cep_path: str,
    output_base: str,
    session_number: int,
    budget_cap_usd: float,
    test_mode: bool,
) -> SessionResult:
    """Execute a single experiment session inside a subprocess.

    This is a module-level function (required for pickling by
    ``ProcessPoolExecutor``). It loads the compiled experiment from disk,
    constructs a fresh ``ExperimentRuntime``, and runs the session. All
    objects are created within the subprocess to avoid shared-state issues.

    Args:
        cep_path: Absolute path to the ``compiled_experiment.json`` file.
        output_base: Root output directory; session artifacts are written
            to a subdirectory derived from experiment/run/session identifiers.
        session_number: Numeric index of the session to execute.
        budget_cap_usd: Maximum spend in USD for this session (0 = unlimited).
        test_mode: If True, only one group per module is executed.

    Returns:
        A ``SessionResult`` with success/failure status and metadata.
    """
    from talkingtomachines.compiler.cep_schema import CompiledExperiment
    from talkingtomachines.orchestrator.runtime import ExperimentRuntime
    from talkingtomachines.storage.artifact_manager import ArtifactManager

    cep = CompiledExperiment.load(cep_path)
    run_dir = str(
        Path(output_base)
        / cep.experiment_id
        / cep.run_id
        / f"session_{session_number:03d}"
    )

    try:
        runtime = ExperimentRuntime(
            cep=cep,
            output_dir=run_dir,
            budget_cap_usd=budget_cap_usd,
        )
        session = runtime.run(session_number=session_number, test_mode=test_mode)
        return SessionResult(
            session_number=session_number,
            session_id=session.session_id,
            total_cost_usd=runtime.total_cost_usd,
            output_dir=run_dir,
            success=True,
        )
    except Exception as exc:
        return SessionResult(
            session_number=session_number,
            session_id="",
            total_cost_usd=0.0,
            output_dir=run_dir,
            success=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Session-level parallelism (ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def run_sessions_parallel(
    cep_path: str | Path,
    output_base: str | Path,
    session_numbers: list[int],
    budget_cap_usd: float = 0.0,
    test_mode: bool = False,
    max_workers: int = 4,
) -> list[SessionResult]:
    """
    Run multiple experiment sessions in parallel using ``ProcessPoolExecutor``.

    Each session runs in an isolated subprocess so that concurrent LLM calls
    and shared global state cannot interfere with each other.

    Args:
        cep_path:         Path to the ``compiled_experiment.json`` file on disk.
        output_base:      Root output directory — each session writes to a sub-folder.
        session_numbers:  Ordered list of session indices to execute (e.g. [1, 2, 3]).
        budget_cap_usd:   Per-session budget cap (0 = unlimited).
        test_mode:        Pass ``test_mode=True`` to each session.
        max_workers:      Maximum number of concurrent processes.

    Returns:
        List of ``SessionResult`` objects in the same order as *session_numbers*.
    """
    cep_path = str(Path(cep_path).resolve())
    output_base = str(Path(output_base).resolve())

    results: dict[int, SessionResult] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _session_worker,
                cep_path,
                output_base,
                sn,
                budget_cap_usd,
                test_mode,
            ): i
            for i, sn in enumerate(session_numbers)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            sn = session_numbers[idx]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error("Session %d raised an unexpected error: %s", sn, exc)
                results[idx] = SessionResult(
                    session_number=sn,
                    session_id="",
                    total_cost_usd=0.0,
                    output_dir="",
                    success=False,
                    error=str(exc),
                )

    ordered = [results[i] for i in range(len(session_numbers))]
    failures = [r for r in ordered if not r.success]
    if failures:
        msgs = "; ".join(f"session {r.session_number}: {r.error}" for r in failures)
        logger.error("Session execution failures: %s", msgs)

    return ordered


# ---------------------------------------------------------------------------
# Group-level parallelism (ThreadPoolExecutor)
# ---------------------------------------------------------------------------


def run_groups_parallel(
    group_fn: Callable[..., T],
    group_args_list: list[tuple],
    max_workers: int = 4,
) -> list[T]:
    """Run a callable concurrently for each group using a thread pool.

    Submits ``group_fn(*args)`` for every ``args`` tuple in
    *group_args_list* and collects results preserving input order.

    Args:
        group_fn: Callable to invoke for each group. Receives unpacked
            positional arguments from each tuple in *group_args_list*.
        group_args_list: List of argument tuples, one per group.
        max_workers: Maximum number of concurrent threads.

    Returns:
        List of results in the same order as *group_args_list*.

    Raises:
        RuntimeError: If one or more group tasks fail. The error message
            aggregates all failure descriptions.
    """
    results: dict[int, T] = {}
    errors: list[Exception] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(group_fn, *args): i
            for i, args in enumerate(group_args_list)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error("Group task %d failed: %s", idx, exc)
                errors.append(exc)

    if errors:
        # Python 3.10-compatible error aggregation (ExceptionGroup requires 3.11+)
        error_messages = "; ".join(str(e) for e in errors)
        raise RuntimeError(
            f"Group execution errors ({len(errors)} failure(s)): {error_messages}"
        ) from errors[0]

    return [results[i] for i in range(len(group_args_list))]


# ---------------------------------------------------------------------------
# Player-level parallelism (ThreadPoolExecutor)
# ---------------------------------------------------------------------------


def run_private_questions_parallel(
    fn: Callable[..., T],
    player_args_list: list[tuple],
    max_workers: int = 8,
) -> list[T]:
    """Run PRIVATE_QUESTION responses concurrently for players within a group.

    Delegates to ``run_groups_parallel`` with a higher default worker count
    since private questions are independent and benefit from more parallelism.

    Args:
        fn: Callable that produces a private response for a single player.
        player_args_list: List of argument tuples, one per player.
        max_workers: Maximum number of concurrent threads.

    Returns:
        List of results in the same order as *player_args_list*.

    Raises:
        RuntimeError: If one or more player tasks fail.
    """
    return run_groups_parallel(fn, player_args_list, max_workers=max_workers)
