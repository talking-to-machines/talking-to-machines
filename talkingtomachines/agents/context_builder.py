"""Context builder for constructing LLM message lists.

Assembles the full, normalised message list that is sent to
``LLMProvider.generate()`` for a single agent turn. The pipeline
proceeds through the following stages:

    1. Inject the system message (persona derived from the profile prompt).
    2. Apply the visibility filter to the group's shared turn history so
       the agent only sees messages it is permitted to observe.
    3. Render the current prompt's ``llm_text`` via Jinja using the
       experiment state context.
    3.5. Optionally prepend RAG-retrieved chunks to the rendered prompt.
    4. Optionally shuffle and inject response options into the prompt.
    5. Append formatting instructions (JSON schema, speculation score).
    5.5. Detect ``[IMAGE:]``, ``[VIDEO:]``, ``[AUDIO:]`` media tags and
         convert the prompt into provider-compatible multimodal content
         blocks.
    6. Return the normalised ``[{"role": ..., "content": ...}, ...]`` list.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from talkingtomachines.core.models import (
        Agent,
        Player,
        Group,
        Session,
        PromptDefinition,
        FieldDefinition,
    )
    from talkingtomachines.core.fields import ExperimentState

from talkingtomachines.core.visibility import filter_message_history
from talkingtomachines.gateway.media import (
    extract_all_media,
    strip_media_tags,
    build_multimodal_content,
)

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds the full message history for a single LLM call.

    Combines the system message, visibility-filtered conversation history,
    Jinja-rendered prompt text, optional RAG augmentation, response-option
    injection, formatting instructions, and multimodal media blocks into a
    normalised message list.

    Attributes:
        _state: The ``ExperimentState`` used to resolve Jinja template
            variables across all scopes (player, agent, group, session).
        _env: A Jinja2 ``SandboxedEnvironment`` for safe template rendering.
        _conjoint_designer: Optional conjoint-experiment table designer.
    """

    def __init__(
        self,
        state: "ExperimentState",
        jinja_env=None,
        conjoint_designer=None,
    ):
        """Initialise the context builder.

        Args:
            state: The shared ``ExperimentState`` for Jinja context resolution.
            jinja_env: Optional pre-configured Jinja2 environment. If None,
                a ``SandboxedEnvironment`` is created automatically.
            conjoint_designer: Optional ``ConjointDesigner`` for injecting
                conjoint-analysis tables into the Jinja context.
        """
        self._state = state
        if jinja_env is None:
            from jinja2.sandbox import SandboxedEnvironment

            self._env = SandboxedEnvironment()
        else:
            self._env = jinja_env
        self._conjoint_designer = conjoint_designer

    def build(
        self,
        agent: "Agent",
        player: "Player",
        group: "Group",
        session: "Session",
        module: str,
        round_number: int,
        prompt: "PromptDefinition",
        system_message: str,
        field_def: Optional["FieldDefinition"] = None,
        constants: Optional[dict] = None,
        rng: Optional[random.Random] = None,
    ) -> list[dict]:
        """Build the normalised message list for a single LLM turn.

        Executes the full context-building pipeline: system message,
        visibility filtering, Jinja rendering, RAG augmentation,
        response-option injection, formatting instructions, and
        multimodal content conversion.

        Args:
            agent: The ``Agent`` model for the current participant.
            player: The ``Player`` model for the current participant.
            group: The ``Group`` the player belongs to.
            session: The current ``Session``.
            module: Module identifier (e.g., oTree app name).
            round_number: Current round number within the module.
            prompt: ``PromptDefinition`` containing the Jinja template text.
            system_message: Pre-built system message (profile prompt).
            field_def: Optional ``FieldDefinition`` controlling response
                options, formatting, and validation metadata.
            constants: Experiment-level constants for the Jinja context.
            rng: Seeded ``random.Random`` for reproducible option shuffling.

        Returns:
            A list of ``{"role": ..., "content": ...}`` dicts ready for
            ``LLMProvider.generate()``.
        """
        # Step 1: System message (persona)
        messages: list[dict] = [{"role": "system", "content": system_message}]

        # Step 2: Visibility-filtered conversation history
        # Read from the agent's session-wide message history.
        visible_history = filter_message_history(
            shared_history=agent.message_history,
            requesting_agent_id=agent.agent_id,
            requesting_group_id=group.group_id,
            requesting_treatment_label=agent.treatment_label,
        )
        for msg in visible_history:
            messages.append(
                {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                }
            )

        # Step 3: Render current prompt
        jinja_ctx = self._state.build_jinja_context(
            agent=agent,
            player=player,
            group=group,
            session=session,
            module=module,
            round_number=round_number,
            constants=constants,
        )
        # Inject conjoint table if designer is configured
        if self._conjoint_designer is not None:
            try:
                jinja_ctx["conjoint_table"] = self._conjoint_designer.generate_table(
                    player_id=player.player_id,
                    round_number=round_number,
                )
            except Exception as exc:
                logger.warning("Conjoint table generation error: %s", exc)

        rendered_prompt = self._render(prompt.llm_text, jinja_ctx)

        # Step 3.5: RAG augmentation — per-prompt vector store ID
        effective_rag = None
        if hasattr(prompt, "rag_vector_store_id") and prompt.rag_vector_store_id:
            effective_rag = self._get_per_prompt_rag_tool(prompt.rag_vector_store_id)
        if effective_rag is not None:
            try:
                rag_result = effective_rag.retrieve(rendered_prompt)
                if rag_result.chunks:
                    chunks_text = "\n\n".join(rag_result.chunks)
                    rendered_prompt = (
                        f"Relevant context:\n{chunks_text}\n\n{rendered_prompt}"
                    )
                    logger.debug(
                        "RAG retrieved %d chunk(s) (%.1f ms).",
                        rag_result.chunks_retrieved,
                        rag_result.latency_ms,
                    )
                if rag_result.failures:
                    logger.warning("RAG partial failures: %s", rag_result.failures)
            except Exception as exc:
                logger.warning("RAG retrieval error: %s — continuing without RAG.", exc)

        # Step 4: Inject response options (optionally shuffled)
        if field_def and field_def.response_options:
            options = (
                list(field_def.response_options)
                if isinstance(field_def.response_options, list)
                else []
            )
            if field_def.randomise_options_order and options and rng:
                rng.shuffle(options)
            if options:
                intro = (
                    field_def.response_options_intro
                    or "Please choose one of the following options:"
                )
                options_text = "\n".join(f"  - {opt}" for opt in options)
                rendered_prompt = f"{rendered_prompt}\n\n{intro}\n{options_text}"

        # Step 5: Append formatting instructions
        if field_def:
            rendered_prompt = self._append_formatting(
                rendered_prompt,
                generate_speculation_score=field_def.generate_speculation_score,
                format_response=field_def.format_response,
                field_name=field_def.name,
                field_type=field_def.type,
            )

        # Step 5.5: Multimodal injection — convert to content blocks if media tags present
        media_items = extract_all_media(rendered_prompt)
        if media_items:
            clean_text = strip_media_tags(rendered_prompt)
            content: Any = build_multimodal_content(clean_text, media_items)
        else:
            content = rendered_prompt
        messages.append({"role": "user", "content": content})
        return messages

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_per_prompt_rag_tool(self, vector_store_id: str):
        """Lazily create a RAG tool for a specific vector store ID.

        Results are cached in ``_rag_cache`` so that repeated calls with the
        same ``vector_store_id`` reuse the same tool instance.

        Args:
            vector_store_id: The OpenAI vector store identifier to bind to.

        Returns:
            An ``OpenAIRAGTool`` instance, or None if creation fails or the
            API key is unavailable.
        """
        if not hasattr(self, "_rag_cache"):
            self._rag_cache: dict = {}
        if vector_store_id in self._rag_cache:
            return self._rag_cache[vector_store_id]
        try:
            import os

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                logger.warning("Per-prompt RAG requires OPENAI_API_KEY; skipping.")
                return None
            from talkingtomachines.gateway.rag import OpenAIRAGTool

            tool = OpenAIRAGTool(api_key=api_key, vector_store_id=vector_store_id)
            self._rag_cache[vector_store_id] = tool
            return tool
        except Exception as exc:
            logger.warning("Failed to create per-prompt RAG tool: %s", exc)
            return None

    def _render(self, template_str: str, context: dict) -> str:
        """Render a Jinja2 template string with the given context.

        Args:
            template_str: The raw Jinja2 template string.
            context: Dictionary of variables available inside the template.

        Returns:
            The rendered string, or the raw template string if rendering
            fails.
        """
        try:
            tmpl = self._env.from_string(template_str)
            return tmpl.render(**context)
        except Exception as exc:
            logger.warning("Jinja rendering error: %s — returning raw template.", exc)
            return template_str

    def _append_formatting(
        self,
        prompt: str,
        generate_speculation_score: bool,
        format_response: bool,
        field_name: str = "",
        field_type: str = "text",
    ) -> str:
        """Append JSON formatting and speculation-score instructions to the prompt.

        Args:
            prompt: The rendered prompt text to augment.
            generate_speculation_score: If True, instruct the LLM to include
                a ``speculation_score`` key in its JSON response.
            format_response: If True, instruct the LLM to return a JSON
                object keyed by ``field_name``.
            field_name: The field name to use as the JSON key.
            field_type: The expected value type (``"integer"``, ``"float"``,
                ``"boolean"``, or ``"text"``).

        Returns:
            The prompt string with formatting instructions appended, or the
            original prompt if no instructions are needed.
        """
        instructions: list[str] = []

        if format_response:
            if field_type in ("integer", "float"):
                instructions.append(
                    f'Return your answer as a JSON object with a single key "{field_name}" '
                    f'containing a {field_type} value. Example: {{"{field_name}": 5}}'
                )
            elif field_type == "boolean":
                instructions.append(
                    f'Return your answer as a JSON object: {{"{field_name}": true}} or {{"{field_name}": false}}'
                )
            else:
                instructions.append(
                    f'Return your answer as a JSON object with a single key "{field_name}". '
                    f'Example: {{"{field_name}": "your answer here"}}'
                )

        if generate_speculation_score:
            instructions.append(
                'Also include a "speculation_score" key in your JSON response with a value '
                "between 0.0 (completely certain) and 1.0 (highly speculative), reflecting "
                "your uncertainty about your answer."
            )

        if instructions:
            return f"{prompt}\n\n" + " ".join(instructions)
        return prompt
