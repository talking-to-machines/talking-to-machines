"""Conditional control flow and stop-condition evaluator.

Provides ``FlowEvaluator``, which uses ``jinja2.sandbox.SandboxedEnvironment``
exclusively -- ``eval()`` is never used directly -- to evaluate ``is_displayed``
expressions and stop conditions at runtime.

Supported ``is_displayed`` expressions::

    (none/null)                      -- always display
    "round_number == 1"              -- only first round
    "round_number in [2, 4]"         -- rounds 2 and 4
    "player.treatment == 'T1'"       -- Treatment 1 agents only
    "agent.democrat == 1"            -- profile field condition
    "pgg.Player.decision > 10"       -- field-value condition

Stop condition signals returned by ``evaluate_stop_condition``::

    "end_round"   -- break the current group loop
    "end_session" -- break the entire session
    "continue"    -- no stop signal detected
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FlowEvaluator:
    """
    Evaluates ``is_displayed`` expressions and stop conditions
    using a Jinja2 sandboxed environment.
    """

    def __init__(self) -> None:
        """Initialise the evaluator with a Jinja2 sandboxed environment.

        Uses ``StrictUndefined`` so that references to missing variables
        raise immediately rather than silently rendering as empty strings.
        """
        from jinja2.sandbox import SandboxedEnvironment
        from jinja2 import StrictUndefined

        self._env = SandboxedEnvironment(undefined=StrictUndefined)

    def is_displayed(self, expression: Optional[str], context: dict[str, Any]) -> bool:
        """Evaluate whether a prompt should be shown.

        Args:
            expression: The ``is_displayed`` string. Pass ``None`` to
                unconditionally display the prompt.
            context: The current Jinja2 rendering context dict.

        Returns:
            ``True`` if the prompt should be shown, ``False`` otherwise.
            Defaults to ``True`` on non-undefined evaluation errors.

        Raises:
            ValueError: If the expression references an undefined
                variable (wraps the Jinja2 ``UndefinedError``).
        """
        if expression is None:
            return True
        try:
            template = self._env.from_string(
                f"{{% if {expression} %}}true{{% else %}}false{{% endif %}}"
            )
            result = template.render(**context).strip()
            return result == "true"
        except Exception as exc:
            from jinja2 import UndefinedError

            if isinstance(exc, UndefinedError):
                raise ValueError(
                    f"is_displayed expression '{expression}' references an undefined variable: {exc}"
                ) from exc
            logger.warning(
                "is_displayed evaluation error '%s': %s — defaulting to True.",
                expression,
                exc,
            )
            return True

    def evaluate_stop_condition(self, value: Any) -> str:
        """Determine the stop signal from a facilitator return value or response.

        Args:
            value: Raw value to interpret. Typically a string returned
                by a facilitator function or LLM response. ``None`` is
                treated as no stop signal.

        Returns:
            One of ``"end_round"``, ``"end_session"``, or ``"continue"``.
        """
        if value is None:
            return "continue"
        s = str(value).strip().lower()
        if "end_session" in s:
            return "end_session"
        if "end_round" in s:
            return "end_round"
        return "continue"
