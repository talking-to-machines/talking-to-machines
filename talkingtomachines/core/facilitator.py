"""Facilitator engine for executing experiment control functions.

Provides ``FacilitatorEngine``, which executes facilitator functions at
runtime. The built-in ``assign_groups`` function delegates to the
``RandomisationEngine``, while custom (natural-language) functions are
sent to an LLM via the ``LLMRouter`` and their responses are parsed as
state updates.
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

_BUILTIN_NAMES = {"assign_groups"}

_VALID_RUNTIME_GROUP_STRATEGIES = {
    "random",
    "keep",
    "swap",
    "stratified",
}


class FacilitatorEngine:
    """Executes facilitator functions at runtime.

    Dispatches the built-in ``assign_groups`` function to the
    randomisation engine and routes all other functions through an
    LLM call.

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
            randomisation: Randomisation engine for group assignment.
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
        prompt_args: dict[str, Any] | None = None,
    ) -> tuple[dict | str, str]:
        """Execute a facilitator function and return its response and rendered prompt.

        For built-in functions, delegates to the appropriate internal
        method and returns a result dict. For custom functions, makes
        an LLM call and returns the raw text response.

        Facilitator-level kwargs (from the Facilitator worksheet) are
        merged with prompt-level kwargs (from the Prompts worksheet).
        Prompt-level kwargs take precedence on conflicts.

        Args:
            fn: The ``FacilitatorFunction`` definition to execute.
            context: Current experiment context dict (agent IDs, etc.).
            prompt_args: Optional keyword arguments from the Prompts
                worksheet ``kwargs`` column.

        Returns:
            A tuple of ``(result, rendered_prompt)``. For built-in
            facilitators, *result* is a dict of state updates and
            *rendered_prompt* is an empty string. For custom LLM
            facilitators, *result* is the LLM response string and
            *rendered_prompt* is the Jinja-rendered definition.
        """
        # Merge facilitator-level kwargs with prompt-level kwargs
        merged_args = dict(fn.kwargs) if fn.kwargs else {}
        if prompt_args:
            merged_args.update(prompt_args)

        if fn.name == "assign_groups":
            result = self._assign_groups(fn, context, merged_args)
            return result, ""
        return self._execute_llm_facilitator(fn, context)

    # ------------------------------------------------------------------
    # Built-in facilitators
    # ------------------------------------------------------------------

    def _assign_groups(
        self, fn: FacilitatorFunction, context: dict, prompt_args: dict
    ) -> dict:
        """Assign agents to groups via the randomisation engine.

        Reads ``strategy`` from *prompt_args* and ``agent_ids`` /
        ``players_per_group`` from the context.

        Args:
            fn: The facilitator function definition.
            context: Must contain ``all_session_agent_ids`` or
                ``agent_ids`` (list of str). May contain
                ``players_per_group`` (int, defaults to 2).
            prompt_args: Keyword arguments from the Prompts worksheet.
                Supports ``strategy`` and ``stratify_by``.

        Returns:
            A dict with key ``group_assignments`` mapping group IDs to
            lists of agent IDs, or an empty dict if ``agent_ids`` is
            missing or empty.
        """
        logger.info("Facilitator: assign_groups")
        agent_ids = context.get("all_session_agent_ids", context.get("agent_ids", []))
        players_per_group = context.get("players_per_group", 2)
        strategy = prompt_args.get("strategy", "random")

        if strategy not in _VALID_RUNTIME_GROUP_STRATEGIES:
            logger.warning(
                "assign_groups: strategy '%s' is not valid at runtime.", strategy
            )
            return {}

        if not agent_ids:
            return {}

        # Build stratify_by from profile info if needed
        stratify_by = None
        if strategy == "stratified":
            stratify_attr = prompt_args.get("stratify_by")
            if stratify_attr:
                agents_profile_info = context.get("agents_profile_info", {})
                stratify_by = [
                    agents_profile_info.get(aid, {}).get(stratify_attr, "")
                    for aid in agent_ids
                ]

        groups = self._rng.assign_groups(
            agent_ids=agent_ids,
            players_per_group=players_per_group,
            strategy=strategy,
            path="groups",
            stratify_by=stratify_by,
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
