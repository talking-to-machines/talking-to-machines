"""Runtime context window guard.

Checks the estimated token count of a message list against the model's
context window limit before each LLM call and applies the configured
overflow policy:

- **terminate** -- raise :class:`ContextWindowExceededError`.
- **truncate** -- sliding-window removal that keeps the system prompt,
  the first exchange, and the most recent messages.
- **summarize** -- use the LLM to summarise older messages, falling
  back to truncation on failure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from talkingtomachines.gateway.model_registry import (
    get_model_spec,
    OUTPUT_RESERVE_TOKENS,
    SAFETY_MARGIN,
)
from talkingtomachines.gateway.token_estimator import estimate_messages_tokens

if TYPE_CHECKING:
    from talkingtomachines.gateway.router import LLMRouter

logger = logging.getLogger(__name__)


class ContextWindowExceededError(Exception):
    """Raised when the context window is exceeded under the ``terminate`` policy."""


class ContextGuard:
    """Guards against context window overflow.

    Computes an effective token limit from the model's context window
    (applying :data:`SAFETY_MARGIN` and reserving
    :data:`OUTPUT_RESERVE_TOKENS` for the response), then enforces that
    limit using the chosen overflow policy.

    Usage::

        guard = ContextGuard(model_name="gpt-4o", policy="truncate")
        safe_messages = guard.apply(messages, router=router, temperature=0.0)

    Args:
        model_name: The model identifier used to look up the context
            window size via :func:`get_model_spec`.
        policy: Overflow policy -- one of ``"terminate"``,
            ``"truncate"``, or ``"summarize"``. Defaults to
            ``"terminate"``.
    """

    def __init__(self, model_name: str, policy: str = "terminate"):
        """Initialise the guard for a specific model and overflow policy."""
        self._model_name = model_name
        self._policy = policy.lower()
        spec = get_model_spec(model_name)
        self._effective_limit = (
            int(spec.max_context_tokens * SAFETY_MARGIN) - OUTPUT_RESERVE_TOKENS
        )

    @property
    def effective_limit(self) -> int:
        """The maximum number of input tokens allowed after safety margins.

        Returns:
            The effective token limit as an integer.
        """
        return self._effective_limit

    def apply(
        self,
        messages: list[dict],
        router: "LLMRouter | None" = None,
        temperature: float = 0.0,
    ) -> list[dict]:
        """Check token count and apply the overflow policy if needed.

        Args:
            messages: The full conversation message list to validate.
            router: An optional :class:`LLMRouter` required by the
                ``"summarize"`` policy to generate summaries.
            temperature: Sampling temperature passed to the summarisation
                call. Defaults to ``0.0``.

        Returns:
            A (possibly shortened) message list that fits within the
            model's effective context window.

        Raises:
            ContextWindowExceededError: If the policy is ``"terminate"``
                and the message list exceeds the effective limit.
            ValueError: If the policy string is not recognised.
        """
        current_tokens = estimate_messages_tokens(messages)

        if current_tokens <= self._effective_limit:
            return messages

        logger.warning(
            "Context window usage (%d tokens) exceeds effective limit (%d tokens) "
            "for model '%s'. Applying '%s' policy.",
            current_tokens,
            self._effective_limit,
            self._model_name,
            self._policy,
        )

        if self._policy == "terminate":
            raise ContextWindowExceededError(
                f"Context window exceeded for model '{self._model_name}': "
                f"{current_tokens:,} tokens > {self._effective_limit:,} effective limit. "
                f"Policy is 'terminate'. Set CONTEXT_OVERFLOW_POLICY to 'truncate' or "
                f"'summarize' in your Settings sheet to handle this automatically."
            )
        elif self._policy == "truncate":
            return self._truncate(messages)
        elif self._policy == "summarize":
            return self._summarize(messages, router, temperature)
        else:
            raise ValueError(f"Unknown overflow policy: '{self._policy}'")

    def _truncate(self, messages: list[dict]) -> list[dict]:
        """Apply sliding-window truncation to fit the context window.

        Preserves three anchored messages and fills the remaining token
        budget with the most recent middle messages:

        - ``messages[0]``: system prompt
        - ``messages[1]``: first conversation message (first-round context)
        - ``messages[-1]``: current user prompt

        Oldest middle messages are removed first. A system-role marker
        noting the number of removed messages is inserted when truncation
        occurs.

        Args:
            messages: The full conversation message list.

        Returns:
            A truncated message list that fits within the effective
            token limit.
        """
        if len(messages) <= 3:
            return messages

        system = messages[0]
        first_exchange = messages[1]
        current_prompt = messages[-1]
        middle = messages[2:-1]

        # Token budget for preserved messages
        preserved_tokens = (
            estimate_messages_tokens([system])
            + estimate_messages_tokens([first_exchange])
            + estimate_messages_tokens([current_prompt])
        )

        middle_budget = self._effective_limit - preserved_tokens

        # Walk from the END of middle (newest) backward to keep the most recent
        kept_middle: list[dict] = []
        accumulated = 0
        for msg in reversed(middle):
            msg_tokens = estimate_messages_tokens([msg])
            if accumulated + msg_tokens <= middle_budget:
                kept_middle.insert(0, msg)
                accumulated += msg_tokens
            else:
                break

        truncated_count = len(middle) - len(kept_middle)
        if truncated_count > 0:
            logger.info(
                "Truncated %d message(s) from conversation history to fit context window.",
                truncated_count,
            )
            marker = {
                "role": "system",
                "content": (
                    f"[Note: {truncated_count} earlier message(s) were removed "
                    f"from the conversation history to fit within the context window.]"
                ),
            }
            return [system, first_exchange, marker] + kept_middle + [current_prompt]

        return [system, first_exchange] + kept_middle + [current_prompt]

    def _summarize(
        self,
        messages: list[dict],
        router: "LLMRouter | None",
        temperature: float,
    ) -> list[dict]:
        """Summarise older conversation history to reduce token count.

        Splits the middle messages in half, summarises the older half
        via an LLM call through *router*, and replaces those messages
        with a single system-role summary block. Falls back to
        :meth:`_truncate` if no router is provided or if the
        summarisation call fails.

        Args:
            messages: The full conversation message list.
            router: The :class:`LLMRouter` used to generate the summary.
            temperature: Sampling temperature for the summarisation call.

        Returns:
            A shortened message list with older history replaced by a
            concise summary.
        """
        if router is None:
            logger.warning(
                "Summarize policy requires an LLMRouter but none was provided. "
                "Falling back to truncation."
            )
            return self._truncate(messages)

        if len(messages) <= 3:
            return messages

        system = messages[0]
        first_exchange = messages[1]
        current_prompt = messages[-1]
        middle = messages[2:-1]

        # Summarize the oldest half of middle messages
        split_point = max(1, len(middle) // 2)
        to_summarize = middle[:split_point]
        to_keep = middle[split_point:]

        # Build summarization prompt
        conversation_text = "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in to_summarize
        )
        summary_prompt = [
            {"role": "system", "content": "You are a concise summarizer."},
            {
                "role": "user",
                "content": (
                    "Summarize the following conversation history in a brief paragraph. "
                    "Preserve all key decisions, arguments, and facts mentioned. "
                    "Be concise but do not omit important details.\n\n"
                    f"{conversation_text}"
                ),
            },
        ]

        try:
            response = router.generate(
                message_history=summary_prompt,
                model=self._model_name,
                temperature=temperature,
            )
            summary_text = response.content
        except Exception as exc:
            logger.warning(
                "Summarization LLM call failed: %s. Falling back to truncation.", exc
            )
            return self._truncate(messages)

        summary_msg = {
            "role": "system",
            "content": (
                f"[Summary of earlier conversation ({len(to_summarize)} messages)]: "
                f"{summary_text}"
            ),
        }

        result = [system, first_exchange, summary_msg] + to_keep + [current_prompt]

        # If still over limit after summarization, fall back to truncation
        if estimate_messages_tokens(result) > self._effective_limit:
            logger.info("Post-summarization still exceeds limit; applying truncation.")
            return self._truncate(result)

        return result
