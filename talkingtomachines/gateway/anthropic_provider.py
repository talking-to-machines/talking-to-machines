"""Anthropic (Claude) gateway provider.

Implements :class:`AnthropicProvider` using the Anthropic Messages API.
System prompts are extracted from the message history and passed via the
dedicated ``system`` parameter.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from .base import LLMProvider, LLMResponse

_COST_TABLE: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (0.015, 0.075),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-haiku-4-5-20251001": (0.00025, 0.00125),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of an Anthropic API call.

    Args:
        model: The model identifier (longest prefix match against the
            cost table is used).
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        The estimated cost in US dollars.
    """
    # Strip date suffixes for lookup
    key = model
    for k in _COST_TABLE:
        if model.startswith(k):
            key = k
            break
    rates = _COST_TABLE.get(key, (0.0, 0.0))
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1000.0


class AnthropicProvider(LLMProvider):
    """Gateway for Anthropic Claude models.

    Args:
        api_key: Anthropic API key.

    Raises:
        ImportError: If the ``anthropic`` package is not installed.
    """

    provider_name = "anthropic"

    def supports_vision(self) -> bool:
        """Return ``True`` -- Claude models support image inputs."""
        return True

    def supports_structured_output(self) -> bool:
        """Return ``False`` -- tool-use JSON enforcement is not yet implemented."""
        return False

    def __init__(self, api_key: str):
        """Initialise the Anthropic client with the given API key."""
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required: pip install anthropic>=0.40"
            ) from exc
        self._client = _anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        message_history: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """Send messages to a Claude model and return a normalised response.

        Args:
            message_history: Conversation messages in OpenAI-compatible
                format. System-role messages are extracted and passed via
                the Anthropic ``system`` parameter.
            model: The Claude model identifier string.
            temperature: Sampling temperature. Defaults to ``0.0``.
            max_tokens: Maximum tokens to generate. Defaults to ``4096``.
            **kwargs: Additional keyword arguments forwarded to the
                Messages API.

        Returns:
            An :class:`LLMResponse` with generated text, token counts,
            cost, and latency.
        """
        system_prompt, messages = self._split_messages(message_history)

        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system_prompt:
            params["system"] = system_prompt
        params.update(kwargs)

        t0 = time.perf_counter()
        raw = self._client.messages.create(**params)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract text from the first text content block; gracefully skip
        # non-text blocks (e.g. tool_use) that lack a .text attribute.
        content = ""
        for block in raw.content or []:
            if hasattr(block, "text"):
                content = block.text
                break
        input_tokens = raw.usage.input_tokens
        output_tokens = raw.usage.output_tokens

        return LLMResponse(
            content=content,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model=model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            cost_usd=_estimate_cost(model, input_tokens, output_tokens),
            raw_response=raw,
        )

    # ------------------------------------------------------------------

    def _split_messages(self, message_history: list[dict]) -> tuple[str, list[dict]]:
        """Separate system-role messages from the conversation.

        Args:
            message_history: The full message list.

        Returns:
            A tuple ``(system_prompt, conversation_messages)`` where
            *system_prompt* is the concatenation of all system messages
            and *conversation_messages* contains only user/assistant
            turns.
        """
        system_parts: list[str] = []
        conversation: list[dict] = []
        for msg in message_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role in ("user", "assistant"):
                conversation.append({"role": role, "content": content})
        return "\n\n".join(system_parts), conversation
