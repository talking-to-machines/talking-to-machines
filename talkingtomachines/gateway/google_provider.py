"""Google Gemini gateway provider.

Implements :class:`GoogleProvider` using the ``google-genai`` SDK.
System instructions are extracted from the message history and passed
via the ``system_instruction`` config parameter. Assistant messages are
mapped to the Gemini ``model`` role.
"""

from __future__ import annotations

import time
from typing import Any

from .base import LLMProvider, LLMResponse

_COST_TABLE: dict[str, tuple[float, float]] = {
    "gemini-2.0-flash": (0.000075, 0.0003),
    "gemini-2.0-pro": (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "gemini-1.5-pro": (0.00125, 0.005),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a Google Gemini API call.

    Args:
        model: The model identifier (``-exp`` and ``-latest`` suffixes
            are stripped for lookup).
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        The estimated cost in US dollars.
    """
    key = model.split("-exp")[0].split("-latest")[0]
    rates = _COST_TABLE.get(key, (0.0, 0.0))
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1000.0


class GoogleProvider(LLMProvider):
    """Gateway for Google Gemini models.

    Args:
        api_key: Google AI API key.

    Raises:
        ImportError: If the ``google-genai`` package is not installed.
    """

    provider_name = "google"

    def __init__(self, api_key: str):
        """Initialise the Google Gemini client with the given API key."""
        try:
            from google import genai as _genai
        except ImportError as exc:
            raise ImportError(
                "google-genai package is required: pip install google-genai>=0.8"
            ) from exc
        self._genai = _genai
        self._client = _genai.Client(api_key=api_key)

    def generate(
        self,
        message_history: list[dict],
        model: str,
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        """Send messages to a Gemini model and return a normalised response.

        Args:
            message_history: Conversation messages in OpenAI-compatible
                format. System-role messages are extracted as the Gemini
                ``system_instruction``.
            model: The Gemini model identifier string.
            temperature: Sampling temperature. Defaults to ``0.0``.
            **kwargs: Additional keyword arguments forwarded to the
                ``GenerateContentConfig``.

        Returns:
            An :class:`LLMResponse` with generated text, token counts,
            cost, and latency.
        """
        system_instruction, contents = self._split_messages(message_history)

        config_params: dict[str, Any] = {"temperature": temperature}
        if system_instruction:
            config_params["system_instruction"] = system_instruction
        config_params.update(kwargs)

        generate_config = self._genai.types.GenerateContentConfig(**config_params)

        t0 = time.perf_counter()
        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_config,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        content = response.text or ""
        usage = response.usage_metadata
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

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

    def _split_messages(self, message_history: list[dict]) -> tuple[str, list[dict]]:
        """Separate system messages and convert roles to Gemini format.

        Args:
            message_history: The full message list in OpenAI format.

        Returns:
            A tuple ``(system_instruction, contents)`` where
            *system_instruction* is the concatenated system text and
            *contents* is a list of Gemini-formatted content dicts
            (``assistant`` mapped to ``model``).
        """
        system_parts: list[str] = []
        contents: list[dict] = []
        for msg in message_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:
                contents.append({"role": "user", "parts": [{"text": content}]})
        return "\n\n".join(system_parts), contents
