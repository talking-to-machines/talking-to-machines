"""Mistral AI gateway provider.

Implements :class:`MistralProvider` using the ``mistralai`` SDK's chat
completion endpoint.
"""

from __future__ import annotations

import time
from typing import Any

from .base import LLMProvider, LLMResponse

_COST_TABLE: dict[str, tuple[float, float]] = {
    "mistral-large-latest": (0.002, 0.006),
    "mistral-medium-latest": (0.00270, 0.0081),
    "mistral-small-latest": (0.0002, 0.0006),
    "open-mistral-7b": (0.00025, 0.00025),
    "open-mixtral-8x7b": (0.0007, 0.0007),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a Mistral API call.

    Args:
        model: The model identifier used as a direct lookup key.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        The estimated cost in US dollars.
    """
    rates = _COST_TABLE.get(model, (0.0, 0.0))
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1000.0


class MistralProvider(LLMProvider):
    """Gateway for Mistral AI models.

    Args:
        api_key: Mistral AI API key.

    Raises:
        ImportError: If the ``mistralai`` package is not installed.
    """

    provider_name = "mistral"

    def __init__(self, api_key: str):
        """Initialise the Mistral client with the given API key."""
        try:
            from mistralai import Mistral as _Mistral
        except ImportError as exc:
            raise ImportError(
                "mistralai package is required: pip install mistralai>=1.0"
            ) from exc
        self._client = _Mistral(api_key=api_key)

    def generate(
        self,
        message_history: list[dict],
        model: str,
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        """Send messages to a Mistral model and return a normalised response.

        Args:
            message_history: Conversation messages in OpenAI-compatible
                format.
            model: The Mistral model identifier string.
            temperature: Sampling temperature. Defaults to ``0.0``.
            **kwargs: Additional keyword arguments forwarded to the
                chat completion call.

        Returns:
            An :class:`LLMResponse` with generated text, token counts,
            cost, and latency.
        """
        messages = self._format_messages(message_history)

        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        params.update(kwargs)

        t0 = time.perf_counter()
        response = self._client.chat.complete(**params)
        latency_ms = (time.perf_counter() - t0) * 1000

        content = response.choices[0].message.content if response.choices else ""
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        return LLMResponse(
            content=content,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model=model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            cost_usd=_estimate_cost(model, input_tokens, output_tokens),
            raw_response=response,
        )

    # ------------------------------------------------------------------

    def _format_messages(self, message_history: list[dict]) -> list[dict]:
        """Normalise message dicts for the Mistral API.

        Unrecognised roles are mapped to ``"user"``.

        Args:
            message_history: Raw message dicts.

        Returns:
            A list of dicts with validated ``role`` and ``content`` keys.
        """
        result = []
        for msg in message_history:
            role = msg.get("role", "user")
            if role not in ("system", "user", "assistant"):
                role = "user"
            result.append({"role": role, "content": msg.get("content", "")})
        return result
