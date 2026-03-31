"""Facilitator engine for executing experiment control functions.

Provides ``FacilitatorEngine``, which executes facilitator functions at
runtime. Built-in functions (e.g. ``assign_treatment``, ``assign_groups``)
delegate to the ``RandomisationEngine``, while custom (natural-language)
functions are sent to an LLM via the ``LLMRouter`` and their JSON
responses are parsed as state updates.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from jinja2.sandbox import SandboxedEnvironment

from talkingtomachines.core.models import FacilitatorFunction

if TYPE_CHECKING:
    from talkingtomachines.core.fields import ExperimentState
    from talkingtomachines.core.randomisation import RandomisationEngine
    from talkingtomachines.gateway.router import LLMRouter

logger = logging.getLogger(__name__)

_BUILTIN_NAMES = {"assign_treatment", "assign_groups", "creating_session"}


class FacilitatorEngine:
    """Executes facilitator functions at runtime.

    Dispatches built-in facilitator functions (``creating_session``,
    ``assign_treatment``, ``assign_groups``) to their respective engine
    methods and routes all other functions through an LLM call.

    Attributes:
        _state: The experiment state store for reading/writing fields.
        _rng: Randomisation engine used by built-in facilitators.
        _router: LLM router for custom facilitator calls.
        _model: Model name passed to the LLM router.
        _temperature: Sampling temperature for LLM calls.
        _jinja_env: Sandboxed Jinja2 environment for template rendering.
    """

    def __init__(
        self,
        state: "ExperimentState",
        randomisation: "RandomisationEngine",
        router: "LLMRouter",
        model_name: str,
        temperature: float = 0.0,
    ) -> None:
        """Initialise the facilitator engine.

        Args:
            state: The experiment state store.
            randomisation: Randomisation engine for treatment/group
                assignment.
            router: LLM router instance for custom facilitator calls.
            model_name: Name of the LLM model to use.
            temperature: Sampling temperature for LLM generation.
                Defaults to ``0.0``.
        """
        self._state = state
        self._rng = randomisation
        self._router = router
        self._model = model_name
        self._temperature = temperature
        self._jinja_env = SandboxedEnvironment()

    def execute(
        self,
        fn: FacilitatorFunction,
        context: dict[str, Any],
    ) -> tuple[str, str]:
        """Execute a facilitator function and return its response and rendered prompt.

        For built-in functions, delegates to the appropriate internal
        method and returns an empty string (side-effects only). For
        custom functions, makes an LLM call and returns the raw text
        response.

        Args:
            fn: The ``FacilitatorFunction`` definition to execute.
            context: Current experiment context dict (agent IDs,
                treatment labels, etc.).

        Returns:
            A tuple of ``(result, rendered_prompt)``. For custom LLM
            facilitators, *result* is the LLM response and
            *rendered_prompt* is the Jinja-rendered definition. For
            built-in facilitators, both are empty strings.
        """
        if fn.name == "creating_session":
            self._creating_session(fn, context)
            return "", ""
        if fn.name == "assign_treatment":
            self._assign_treatment(fn, context)
            return "", ""
        if fn.name == "assign_groups":
            self._assign_groups(fn, context)
            return "", ""
        return self._execute_llm_facilitator(fn, context)

    # ------------------------------------------------------------------
    # Built-in facilitators
    # ------------------------------------------------------------------

    def _creating_session(self, fn: FacilitatorFunction, context: dict) -> dict:
        """Initialise session state from the facilitator definition.

        Args:
            fn: The facilitator function definition.
            context: Current experiment context dict.

        Returns:
            An empty dict (session creation has no state updates).
        """
        logger.info("Facilitator: creating_session")
        return {}

    def _assign_treatment(self, fn: FacilitatorFunction, context: dict) -> dict:
        """Assign treatments to agents via the randomisation engine.

        Reads ``agent_ids`` and ``treatment_labels`` from the context
        and ``strategy`` / ``path`` from the function args.

        Args:
            fn: The facilitator function definition (carries ``args``
                with optional ``strategy`` and ``path`` keys).
            context: Must contain ``agent_ids`` (list of str) and
                ``treatment_labels`` (list of str).

        Returns:
            A dict with key ``treatment_assignments`` mapping agent IDs
            to treatment labels, or an empty dict if required context
            entries are missing.
        """
        logger.info("Facilitator: assign_treatment")
        # Context should contain agent_ids and treatment_labels
        agent_ids = context.get("agent_ids", [])
        treatment_labels = context.get("treatment_labels", [])
        strategy = fn.args.get("strategy", "simple_random")
        path = fn.args.get("path", "treatments")

        if not agent_ids or not treatment_labels:
            logger.warning(
                "assign_treatment: missing agent_ids or treatment_labels in context."
            )
            return {}

        assignments = self._rng.assign_treatments(
            agent_ids=agent_ids,
            treatment_labels=treatment_labels,
            strategy=strategy,
            path=path,
        )
        return {"treatment_assignments": assignments}

    def _assign_groups(self, fn: FacilitatorFunction, context: dict) -> dict:
        """Assign agents to groups via the randomisation engine.

        Reads ``agent_ids`` and ``players_per_group`` from the context
        and ``strategy`` / ``path`` from the function args.

        Args:
            fn: The facilitator function definition (carries ``args``
                with optional ``strategy`` and ``path`` keys).
            context: Must contain ``agent_ids`` (list of str). May
                contain ``players_per_group`` (int, defaults to 2).

        Returns:
            A dict with key ``group_assignments`` mapping group IDs to
            lists of agent IDs, or an empty dict if ``agent_ids`` is
            missing or empty.
        """
        logger.info("Facilitator: assign_groups")
        agent_ids = context.get("agent_ids", [])
        players_per_group = context.get("players_per_group", 2)
        strategy = fn.args.get("strategy", "random")
        path = fn.args.get("path", "groups")

        if not agent_ids:
            return {}

        groups = self._rng.assign_groups(
            agent_ids=agent_ids,
            players_per_group=players_per_group,
            strategy=strategy,
            path=path,
        )
        return {"group_assignments": groups}

    # ------------------------------------------------------------------
    # Custom (LLM) facilitator
    # ------------------------------------------------------------------

    def _render(self, template_str: str, context: dict) -> str:
        """Render a Jinja2 template string with the given context.

        Falls back to returning the raw template string if rendering
        fails.

        Args:
            template_str: A Jinja2 template string.
            context: Variable bindings for template rendering.

        Returns:
            The rendered string, or the raw template on error.
        """
        try:
            tmpl = self._jinja_env.from_string(template_str)
            return tmpl.render(**context)
        except Exception as exc:
            logger.warning(
                "Facilitator Jinja rendering error: %s — returning raw template.", exc
            )
            return template_str

    def _execute_llm_facilitator(
        self, fn: FacilitatorFunction, context: dict
    ) -> tuple[str, str]:
        """Execute a custom facilitator function via an LLM call.

        Renders the function definition as a Jinja2 template, sends it
        as the system prompt, replays the group's conversation history
        as actual turns, and appends the rendered instruction as the
        final user message.

        Args:
            fn: The custom facilitator function definition. Its
                ``definition`` field is rendered as a Jinja2 template.
            context: Current experiment context dict. If a
                ``facilitator_messages`` key is present, those messages
                are included as prior conversation turns so the
                facilitator can see the full group history and its own
                prior responses.

        Returns:
            A tuple of ``(result, rendered_definition)`` where *result*
            is the raw LLM response and *rendered_definition* is the
            Jinja-rendered definition text. Returns ``("", rendered)``
            if the LLM call fails.
        """
        rendered_definition = self._render(fn.definition, context)
        system_prompt = (
            "You are a facilitator for a social science experiment. "
            "Execute the instruction below. Your response will be stored "
            "directly as the facilitator's output.\n\n"
            f"Instruction: {rendered_definition}"
        )

        message_history: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Replay group conversation history as actual turns.
        # - Facilitator's own prior responses keep role="assistant"
        # - All other messages become role="user" prefixed with sender
        #   identity and round context so the facilitator can distinguish
        #   who said what and when.
        facilitator_messages = context.get("facilitator_messages", [])
        for msg in facilitator_messages:
            sender = msg.get("sender_agent_id", "")
            content = msg.get("content", "")
            if not content:
                continue

            if sender == "facilitator":
                # Facilitator's own prior response
                message_history.append({"role": "assistant", "content": content})
            else:
                # Agent message — prefix with sender and round context
                module = msg.get("module", "")
                round_num = msg.get("round_number", 0)
                prefix_parts = []
                if sender:
                    prefix_parts.append(sender)
                if module:
                    prefix_parts.append(f"module={module}")
                if round_num:
                    prefix_parts.append(f"round={round_num}")
                prefix = f"[{', '.join(prefix_parts)}] " if prefix_parts else ""
                message_history.append(
                    {"role": "user", "content": f"{prefix}{content}"}
                )

        # Final user turn: the current instruction prompt
        message_history.append({"role": "user", "content": rendered_definition})

        try:
            response = self._router.generate(
                message_history=message_history,
                model=self._model,
                temperature=self._temperature,
            )
            return response.content.strip(), rendered_definition
        except Exception as exc:
            logger.error("Facilitator LLM call failed: %s", exc)
            return "", rendered_definition
