"""Runtime guardrails for safety checks and anomaly detection.

Provides monitoring for suspicious LLM response patterns such as
repeated identical outputs and suspiciously fast completions. Anomalies
are recorded as ``AnomalyFlag`` instances for post-hoc analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class SchemaFailureLimitError(Exception):
    """Raised when consecutive schema validation failures exceed the retry limit.

    This exception signals that the LLM has failed to produce a valid
    response conforming to the expected schema after the maximum number
    of retry attempts.
    """


@dataclass
class AnomalyFlag:
    """Record of a single detected anomaly during experiment execution.

    Attributes:
        turn_id: Identifier of the turn where the anomaly was detected.
        flag_type: Category of anomaly (e.g., ``"suspiciously_fast"``,
            ``"repeated_response"``).
        details: Human-readable description of the anomaly.
    """

    turn_id: str
    flag_type: str
    details: str


class Guardrails:
    """Monitors LLM responses for anomalies and enforces safety limits.

    Performs two runtime checks on every response:

    1. **Fast completion**: Flags responses with latency below
       ``MIN_LATENCY_MS`` as suspicious (may indicate caching or errors).
    2. **Repeated response**: Flags when the last ``REPEAT_WINDOW``
       consecutive responses from the same agent are identical.

    Attributes:
        MIN_LATENCY_MS: Minimum expected latency in milliseconds. Responses
            faster than this threshold are flagged.
        REPEAT_WINDOW: Number of consecutive identical responses that
            trigger a repeated-response anomaly flag.
        anomaly_flags: Accumulated list of all detected anomaly records.
    """

    MIN_LATENCY_MS: float = 50.0  # Sub-50ms responses are suspicious
    REPEAT_WINDOW: int = 3  # Flag if last N responses are identical

    def __init__(self) -> None:
        """Initialize the guardrails with empty response history and flags."""
        self._agent_response_history: dict[str, list[str]] = {}
        self.anomaly_flags: list[AnomalyFlag] = []

    def check_response(
        self,
        turn_id: str,
        agent_instance_id: str,
        content: str,
        latency_ms: float,
    ) -> list[str]:
        """Check a single LLM response for anomalies.

        Runs all configured anomaly detectors against the response and
        records any flags to ``self.anomaly_flags``.

        Args:
            turn_id: Unique identifier for this turn, used in log messages
                and anomaly records.
            agent_instance_id: Identifier of the agent that produced the
                response.
            content: The text content of the LLM response.
            latency_ms: Wall-clock time in milliseconds that the LLM call
                took to complete.

        Returns:
            List of flag type strings detected (e.g.,
            ``["suspiciously_fast"]``). Empty if no anomalies found.
        """
        flags: list[str] = []

        # Fast completion check
        if latency_ms < self.MIN_LATENCY_MS and latency_ms > 0:
            flag = AnomalyFlag(
                turn_id=turn_id,
                flag_type="suspiciously_fast",
                details=f"Latency {latency_ms:.1f}ms < {self.MIN_LATENCY_MS}ms threshold.",
            )
            self.anomaly_flags.append(flag)
            flags.append("suspiciously_fast")
            logger.warning(
                "Anomaly [%s]: suspiciously fast response (%.1f ms)",
                turn_id,
                latency_ms,
            )

        # Repeated identical response check
        history = self._agent_response_history.setdefault(agent_instance_id, [])
        history.append(content)
        if len(history) >= self.REPEAT_WINDOW:
            window = history[-self.REPEAT_WINDOW :]
            if len(set(window)) == 1:
                flag = AnomalyFlag(
                    turn_id=turn_id,
                    flag_type="repeated_response",
                    details=f"Agent gave identical response {self.REPEAT_WINDOW} times in a row.",
                )
                self.anomaly_flags.append(flag)
                flags.append("repeated_response")
                logger.warning(
                    "Anomaly [%s]: agent %s gave identical response %d times in a row.",
                    turn_id,
                    agent_instance_id,
                    self.REPEAT_WINDOW,
                )

        return flags
