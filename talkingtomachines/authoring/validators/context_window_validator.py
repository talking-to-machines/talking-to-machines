"""Context window validator (compile-time estimation).

Estimates worst-case context window usage for each task in an
experiment and warns or errors when the estimate approaches or
exceeds the model's context window limit.
"""

from __future__ import annotations

import logging
from typing import Any

from .schema_validator import ValidationError
from talkingtomachines.gateway.model_registry import (
    get_model_spec,
    WARNING_THRESHOLD,
    SYSTEM_PROMPT_ESTIMATE,
    AVG_RESPONSE_TOKENS,
)
from talkingtomachines.gateway.token_estimator import estimate_tokens

logger = logging.getLogger(__name__)


class ContextWindowValidator:
    """Estimates worst-case context window usage at compile time.

    For each task, computes a conservative upper-bound token estimate
    based on prompt lengths, number of rounds, and players per group.
    Emits a ``ValidationError`` when the estimate exceeds the model's
    context window, or logs a warning when utilisation passes
    ``WARNING_THRESHOLD``.
    """

    def __init__(
        self,
        model_name: str,
        prompts: dict[str, list],
        constants: dict[str, dict[str, Any]],
        task_sequence: list[str],
        num_agents_per_session: int,
    ) -> None:
        """Initialise the context window validator.

        Args:
            model_name: Name of the LLM model (used to look up the
                context window size via the model registry).
            prompts: Mapping ``{task: [PromptDefinition, ...]}`` as
                returned by the prompts parser.
            constants: Nested dictionary ``{task: {name: value}}``
                as returned by the constants parser.
            task_sequence: Ordered list of task names from the
                ``TASK_SEQUENCE`` setting.
            num_agents_per_session: Number of agents per session,
                used as the default ``PLAYERS_PER_GROUP`` fallback.
        """
        self._model_name = model_name
        self._prompts = prompts
        self._constants = constants
        self._task_sequence = task_sequence
        self._num_agents = num_agents_per_session
        self._errors: list[ValidationError] = []

    def validate(self) -> list[ValidationError]:
        """Run context window estimation for each task and return errors.

        Computes worst-case token usage assuming all prompts are
        displayed, all agents respond every round, and all rounds
        execute. Emits an error if the estimate exceeds the model's
        context limit, or a warning if it exceeds
        ``WARNING_THRESHOLD``.

        Returns:
            A list of ``ValidationError`` instances. An empty list
            indicates that all tasks are within safe limits.
        """
        self._errors = []
        spec = get_model_spec(self._model_name)
        max_tokens = spec.max_context_tokens

        for task in self._task_sequence:
            task_consts = self._constants.get(task, {})
            max_rounds = int(task_consts.get("MAX_NUM_ROUNDS", 1))
            players_per_group = int(
                task_consts.get("PLAYERS_PER_GROUP", self._num_agents)
            )

            task_prompts = self._prompts.get(task, [])
            if not task_prompts:
                continue

            # Estimate tokens per prompt text
            prompt_token_estimates = []
            for p in task_prompts:
                text = getattr(p, "llm_text", "") or ""
                prompt_token_estimates.append(estimate_tokens(text))
            avg_prompt_tokens = (
                sum(prompt_token_estimates) / len(prompt_token_estimates)
                if prompt_token_estimates
                else 100
            )

            # Worst case: all prompts displayed, all agents respond, all rounds run
            # Context grows as: system + sum over rounds of (prompts + agent_responses)
            turns_per_round = len(task_prompts) * players_per_group
            tokens_per_round = (
                len(task_prompts) * avg_prompt_tokens
                + turns_per_round * AVG_RESPONSE_TOKENS
            )
            total_estimate = int(
                SYSTEM_PROMPT_ESTIMATE
                + max_rounds * tokens_per_round
                + avg_prompt_tokens  # Current prompt being sent
            )

            utilization = total_estimate / max_tokens if max_tokens > 0 else 0

            if total_estimate > max_tokens:
                self._err(
                    "Settings",
                    None,
                    "MODEL_NAME",
                    f"Task '{task}': estimated worst-case context usage "
                    f"({total_estimate:,} tokens) EXCEEDS model '{self._model_name}' "
                    f"context window ({max_tokens:,} tokens). "
                    f"Consider reducing MAX_NUM_ROUNDS ({max_rounds}), "
                    f"PLAYERS_PER_GROUP ({players_per_group}), or setting "
                    f"CONTEXT_OVERFLOW_POLICY to 'truncate' or 'summarize'.",
                )
            elif utilization > WARNING_THRESHOLD:
                logger.warning(
                    "Task '%s': estimated context usage is %.0f%% of %s context window "
                    "(%s / %s tokens). Risk of overflow at runtime.",
                    task,
                    utilization * 100,
                    self._model_name,
                    f"{total_estimate:,}",
                    f"{max_tokens:,}",
                )

        return self._errors

    def get_summary(self) -> dict[str, Any]:
        """Return a summary dictionary of model context limits.

        Intended for CLI display so users can see the model's token
        constraints at a glance.

        Returns:
            A dictionary with keys ``"model_name"``,
            ``"max_context_tokens"``, and ``"max_output_tokens"``.
        """
        spec = get_model_spec(self._model_name)
        return {
            "model_name": self._model_name,
            "max_context_tokens": spec.max_context_tokens,
            "max_output_tokens": spec.max_output_tokens,
        }

    def _err(self, sheet: str, row: int | None, col: str | None, message: str) -> None:
        """Record a validation error.

        Args:
            sheet: Name of the worksheet where the error was found.
            row: Row index, or ``None`` for sheet-level errors.
            col: Column name, or ``None`` when not applicable.
            message: Human-readable error description.
        """
        self._errors.append(
            ValidationError(sheet=sheet, row=row, col=col, message=message)
        )
