"""Thread-safe cost tracker with budget cap enforcement.

Provides :class:`CostTracker`, a thread-safe accumulator that records
the cost of each LLM call and raises :class:`BudgetExhaustedError` when
a configurable budget cap is exceeded.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


class BudgetExhaustedError(Exception):
    """Raised when accumulated cost exceeds the configured budget cap.

    The exception message includes both the budget cap and the current
    total cost in USD.
    """


@dataclass
class CostTracker:
    """Accumulates cost across all LLM calls for a single experiment run.

    All mutations are protected by a threading lock so the tracker is
    safe to use from concurrent threads.

    Attributes:
        budget_cap_usd: Maximum allowed spend in US dollars. A value of
            ``0.0`` (the default) disables budget enforcement.
    """

    budget_cap_usd: float = 0.0
    _total: float = field(default=0.0, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def add(self, cost_usd: float) -> None:
        """Add *cost_usd* to the running total.

        Args:
            cost_usd: The cost of the most recent LLM call in US dollars.

        Raises:
            BudgetExhaustedError: If the new total exceeds
                :attr:`budget_cap_usd` (when the cap is greater than zero).
        """
        with self._lock:
            self._total += cost_usd
            if self.budget_cap_usd > 0 and self._total > self.budget_cap_usd:
                raise BudgetExhaustedError(
                    f"Budget cap of ${self.budget_cap_usd:.4f} USD exceeded "
                    f"(current total: ${self._total:.4f} USD)."
                )

    @property
    def total_usd(self) -> float:
        """The accumulated cost in US dollars so far.

        Returns:
            The current running total.
        """
        with self._lock:
            return self._total

    def reset(self) -> None:
        """Reset the accumulated cost to zero."""
        with self._lock:
            self._total = 0.0
