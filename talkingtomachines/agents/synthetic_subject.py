"""Synthetic subject agent for AI-driven experiment participation.

Provides the ``ConversationalSyntheticSubject`` class, which wraps an
``Agent``/``Player`` pair and orchestrates LLM-based response generation
within the experiment hierarchy.

Key design points:
    - Constructor accepts ``Agent`` and ``Player`` model objects rather
      than loose parameters.
    - Prompt rendering is delegated to ``ContextBuilder.build()``.
    - Response storage writes to ``Player.state``, ``Agent.state``,
      ``Group.state``, or ``Session.state`` via ``ExperimentState``.
    - All LLM calls are routed through ``LLMRouter``, supporting
      multiple providers.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Optional, TYPE_CHECKING

from talkingtomachines.core.models import (
    Agent,
    Player,
    Group,
    Session,
    PromptDefinition,
    FieldDefinition,
)
from talkingtomachines.core.fields import ExperimentState
from talkingtomachines.agents.context_builder import ContextBuilder
from talkingtomachines.agents.prompt_builder import (
    build_profile_prompt,
    build_system_message,
)

if TYPE_CHECKING:
    from talkingtomachines.gateway.router import LLMRouter

logger = logging.getLogger(__name__)

MAX_VALIDATION_RETRIES = 3


class ConversationalSyntheticSubject:
    """A synthetic (AI) participant in an experiment.

    Wraps one ``Agent`` / ``Player`` pair and provides system message
    construction, visibility-filtered context building, LLM generation
    with optional response validation and retry logic, and state
    persistence via ``ExperimentState``.

    Attributes:
        agent: The ``Agent`` model object representing this participant's
            persistent identity across rounds.
        player: The ``Player`` model object representing this participant's
            round-specific state within a group.
        group: The ``Group`` this player belongs to in the current session.
        session: The ``Session`` this player is participating in.
    """

    def __init__(
        self,
        agent: Agent,
        player: Player,
        group: Group,
        session: Session,
        state: ExperimentState,
        router: "LLMRouter",
        model_name: str,
        temperature: float = 0.0,
        profiles_meta: Optional[dict] = None,
        build_profile_qa: bool = False,
        build_profile_backstories: bool = False,
        constants: Optional[dict] = None,
        rng: Optional[random.Random] = None,
        context_guard=None,
    ):
        """Initialise a synthetic subject.

        Args:
            agent: The ``Agent`` model for this participant.
            player: The ``Player`` model for this participant.
            group: The ``Group`` the player belongs to.
            session: The current ``Session``.
            state: Shared ``ExperimentState`` for reading/writing field values.
            router: ``LLMRouter`` instance for generating LLM responses.
            model_name: Identifier of the model to use for generation.
            temperature: Sampling temperature for LLM calls.
            profiles_meta: Optional dictionary with ``"short_names"`` and
                ``"full_names"`` lists used for profile prompt construction.
            build_profile_qa: Whether to build a Q&A-format profile prompt.
            build_profile_backstories: Whether to generate a backstory via LLM.
            constants: Experiment-level constants injected into Jinja context.
            rng: Seeded ``random.Random`` instance for reproducibility.
            context_guard: Optional context-window guard that truncates or
                summarises messages before they are sent to the LLM.
        """
        self.agent = agent
        self.player = player
        self.group = group
        self.session = session
        self._state = state
        self._router = router
        self._model = model_name
        self._temperature = temperature
        self._constants = constants or {}
        self._rng = rng or random.Random()
        self._context_guard = context_guard

        # Build system message once
        short_names = (profiles_meta or {}).get("short_names", [])
        full_names = (profiles_meta or {}).get("full_names", [])
        profile_prompt = build_profile_prompt(
            profile_row=agent.profile_info,
            short_names=short_names,
            full_names=full_names,
            build_profile_qa=build_profile_qa,
            build_profile_backstories=build_profile_backstories,
            router=router if build_profile_backstories else None,
            model_name=model_name,
            temperature=temperature,
        )
        self._system_message = build_system_message(profile_prompt)
        self._context_builder = ContextBuilder(state=state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def respond(
        self,
        task: str,
        round_number: int,
        prompt: PromptDefinition,
        field_def: Optional[FieldDefinition] = None,
    ) -> str:
        """Generate a response to the given prompt.

        Builds the full message context, optionally applies a context-window
        guard, sends the messages to the LLM, validates the response if
        required, and persists the result to the appropriate state scope.

        Args:
            task: The task identifier (e.g., app name in oTree terms).
            round_number: The current round number within the task.
            prompt: The ``PromptDefinition`` containing the LLM text template
                and prompt metadata.
            field_def: Optional ``FieldDefinition`` describing the expected
                response format, validation rules, and storage scope.

        Returns:
            The raw response string from the LLM.

        Raises:
            No exceptions are raised directly; LLM and validation errors are
            logged and handled with retry logic.
        """
        messages = self._context_builder.build(
            agent=self.agent,
            player=self.player,
            group=self.group,
            session=self.session,
            task=task,
            round_number=round_number,
            prompt=prompt,
            system_message=self._system_message,
            field_def=field_def,
            constants=self._constants,
            rng=self._rng,
        )

        # Apply context window guard before sending to LLM
        if self._context_guard is not None:
            messages = self._context_guard.apply(
                messages,
                router=self._router,
                temperature=self._temperature,
            )

        validate = field_def.validate if field_def else False
        response_options = field_def.response_options if field_def else None

        # Generate with optional validation retries
        content = ""
        for attempt in range(MAX_VALIDATION_RETRIES if validate else 1):
            llm_response = self._router.generate(
                message_history=messages,
                model=self._model,
                temperature=self._temperature,
            )
            content = llm_response.content
            if not content:
                logger.error(
                    "Agent %s received empty LLM response (llm_fallback) for prompt type '%s'.",
                    self.agent.agent_id,
                    prompt.type,
                )

            if validate and response_options:
                if self._validate_response(content, response_options, field_def):
                    break
                if attempt < MAX_VALIDATION_RETRIES - 1:
                    logger.warning(
                        "Agent %s response validation failed (attempt %d/%d). Retrying.",
                        self.agent.agent_id,
                        attempt + 1,
                        MAX_VALIDATION_RETRIES,
                    )
            else:
                break

        # Store response in Player.state
        if field_def and field_def.name:
            parsed, speculation_score = self._parse_json_response(content, field_def)
            self._state.set_player(self.player.player_id, task, field_def.name, parsed)
            if speculation_score is not None:
                self._state.set_player(
                    self.player.player_id,
                    task,
                    f"{field_def.name}_speculation_score",
                    speculation_score,
                )
            if field_def.field_class == "Agent":
                self._state.set_agent(self.agent.agent_id, task, field_def.name, parsed)
            elif field_def.field_class == "Group":
                if prompt.type == "PRIVATE_QUESTION":
                    self._state.accumulate_group(
                        self.group.group_id,
                        task,
                        field_def.name,
                        self.player.player_id,
                        parsed,
                    )
                else:
                    self._state.set_group(
                        self.group.group_id, task, field_def.name, parsed
                    )
            elif field_def.field_class == "Session":
                if prompt.type == "PRIVATE_QUESTION":
                    self._state.accumulate_session(
                        task,
                        field_def.name,
                        self.player.player_id,
                        parsed,
                    )
                else:
                    self._state.set_session(task, field_def.name, parsed)

        return content

    def get_system_message(self) -> str:
        """Return the pre-built system message for this subject.

        Returns:
            The system message string containing the profile prompt.
        """
        return self._system_message

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_response(
        self,
        content: str,
        response_options: Any,
        field_def: FieldDefinition,
    ) -> bool:
        """Check if the response matches expected options or type.

        Args:
            content: The raw LLM response string.
            response_options: A list of valid response values, or another
                constraint descriptor.
            field_def: The ``FieldDefinition`` used for JSON parsing rules.

        Returns:
            True if the response is valid, False otherwise.
        """
        if not response_options:
            return True
        if isinstance(response_options, list):
            # Try JSON parse first
            parsed, _ = self._parse_json_response(content, field_def)
            valid = str(parsed) in [str(opt) for opt in response_options]
            if not valid:
                logger.warning(
                    "Agent %s: response '%s' for field '%s' not in expected options %s.",
                    self.agent.agent_id,
                    parsed,
                    field_def.name,
                    response_options,
                )
            return valid
        return True

    def _parse_json_response(
        self,
        content: str,
        field_def: Optional[FieldDefinition],
    ) -> tuple[Any, Optional[float]]:
        """Extract a structured value and optional speculation score from a JSON response.

        Handles markdown code-fence stripping and JSON parsing. If the
        parsed result is a dict and ``field_def.name`` is set, the value
        for that key is returned. A ``speculation_score`` key, if present,
        is extracted separately.

        Args:
            content: The raw LLM response string.
            field_def: Optional ``FieldDefinition`` whose ``format_response``
                flag and ``name`` control parsing behaviour.

        Returns:
            A tuple of ``(parsed_value, speculation_score)``. The score is
            ``None`` if not present or if JSON parsing fails.
        """
        if not field_def or not field_def.format_response:
            return content.strip(), None
        try:
            text = content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            speculation_score = None
            if isinstance(data, dict):
                speculation_score = data.pop("speculation_score", None)
                if field_def.name:
                    return data.get(field_def.name, content), speculation_score
            return data, speculation_score
        except json.JSONDecodeError:
            logger.warning(
                "Agent %s: JSON parse failed for field '%s' — storing raw text.",
                self.agent.agent_id,
                field_def.name if field_def else "",
            )
            return content.strip(), None
