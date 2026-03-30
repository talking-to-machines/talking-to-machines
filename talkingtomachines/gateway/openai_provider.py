"""OpenAI gateway provider.

Implements :class:`OpenAIProvider`, the gateway for OpenAI models. Uses
the Responses API for native OpenAI calls and the Chat Completions API
for OpenAI-compatible endpoints. Subclasses (Grok, DeepSeek,
OpenRouter, HuggingFace) set ``_use_chat_completions = True`` to route
through the Chat Completions path.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from .base import LLMProvider, LLMResponse

# Models that do not accept a temperature parameter
NO_TEMPERATURE_MODELS = {"o1", "o1-pro", "o3-pro", "o3", "o4-mini"}

# Cost estimates per 1K tokens (input / output) USD — approximate, update as needed
_COST_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-4.1-nano": (0.0001, 0.0004),
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-5": (0.010, 0.030),
    "gpt-5-mini": (0.002, 0.008),
    "o1": (0.015, 0.060),
    "o3": (0.010, 0.040),
    "o3-pro": (0.020, 0.080),
    "o4-mini": (0.003, 0.012),
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the USD cost of an OpenAI API call.

    Args:
        model: The model identifier (date suffixes are stripped for lookup).
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens.

    Returns:
        The estimated cost in US dollars.
    """
    key = model.split("-202")[0]  # strip date suffixes
    rates = _COST_TABLE.get(key, (0.0, 0.0))
    return (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1000.0


class OpenAIProvider(LLMProvider):
    """Gateway for OpenAI models and OpenAI-compatible endpoints.

    Native OpenAI uses the Responses API. OpenAI-compatible endpoints
    (Grok/xAI, DeepSeek, HuggingFace TGI, OpenRouter) use the Chat
    Completions API -- those subclasses set ``_use_chat_completions = True``.

    Attributes:
        provider_name: ``"openai"`` for the base class; overridden by
            subclasses.
        _use_chat_completions: When ``True``, use the Chat Completions
            API instead of the Responses API. Defaults to ``False``.

    Args:
        api_key: The OpenAI (or compatible) API key.
        base_url: Optional base URL for OpenAI-compatible endpoints.
            When ``None``, the default OpenAI API URL is used.

    Raises:
        ImportError: If the ``openai`` package is not installed.
    """

    provider_name = "openai"

    # Subclasses that point at OpenAI-compatible (non-Responses) endpoints
    # must set this to True so the correct API and token field names are used.
    _use_chat_completions: bool = False

    def supports_vision(self) -> bool:
        """Return ``True`` -- OpenAI models support image inputs."""
        return True

    def supports_structured_output(self) -> bool:
        """Return ``True`` -- OpenAI supports JSON schema enforcement."""
        return True

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """Initialise the OpenAI client with the given API key and optional base URL."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required: pip install openai>=1.60"
            ) from exc

        self._client = (
            OpenAI(api_key=api_key, base_url=base_url)
            if base_url
            else OpenAI(api_key=api_key)
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        message_history: list[dict],
        model: str,
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        """Send messages to an OpenAI model and return a normalised response.

        Routes to the Responses API or the Chat Completions API based on
        :attr:`_use_chat_completions`.

        Args:
            message_history: Conversation messages in OpenAI-compatible
                format.
            model: The model identifier string.
            temperature: Sampling temperature. Defaults to ``0.0``.
            **kwargs: Additional keyword arguments forwarded to the API.

        Returns:
            An :class:`LLMResponse` with generated text, token counts,
            cost, and latency.
        """
        input_messages = self._format_messages(message_history)
        t0 = time.perf_counter()

        if self._use_chat_completions:
            raw = self._call_api_chat(input_messages, model, temperature, **kwargs)
            content = self._extract_content_chat(raw)
            usage = getattr(raw, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        else:
            raw = self._call_api(input_messages, model, temperature, **kwargs)
            content = self._extract_content(raw)
            usage = getattr(raw, "usage", None)
            prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        latency_ms = (time.perf_counter() - t0) * 1000
        total_tokens = prompt_tokens + completion_tokens

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            cost_usd=_estimate_cost(model, prompt_tokens, completion_tokens),
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    # Internal helpers — Responses API (native OpenAI only)
    # ------------------------------------------------------------------

    def _format_messages(self, message_history: list[dict]) -> list[dict]:
        """Normalise message dicts to standard role/content pairs.

        Args:
            message_history: Raw message dicts that may contain
                non-standard roles.

        Returns:
            A filtered list containing only messages with ``system``,
            ``user``, or ``assistant`` roles.
        """
        formatted = []
        for msg in message_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("system", "user", "assistant"):
                formatted.append({"role": role, "content": content})
        return formatted

    def _call_api(
        self, messages: list[dict], model: str, temperature: float, **kwargs
    ) -> Any:
        """Call the OpenAI Responses API (native OpenAI endpoint only).

        Args:
            messages: Formatted message list.
            model: Model identifier.
            temperature: Sampling temperature.
            **kwargs: Extra parameters forwarded to the API.

        Returns:
            The raw Responses API response object.
        """
        system_content = ""
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                conversation.append(msg)

        params: dict[str, Any] = {"model": model, "input": conversation}
        if system_content:
            params["instructions"] = system_content
        if model not in NO_TEMPERATURE_MODELS:
            params["temperature"] = temperature
        params.update(kwargs)

        return self._client.responses.create(**params)

    def _extract_content(self, raw: Any) -> str:
        """Extract text from an OpenAI Responses API response.

        Args:
            raw: The raw response object from the Responses API.

        Returns:
            The generated text string. Falls back to ``str(raw)`` if
            the expected attributes are absent.
        """
        try:
            return raw.output_text
        except AttributeError:
            pass
        try:
            return raw.output[0].content[0].text
        except (AttributeError, IndexError, TypeError):
            return str(raw)

    # ------------------------------------------------------------------
    # Internal helpers — Chat Completions API (OpenAI-compatible endpoints)
    # ------------------------------------------------------------------

    def _call_api_chat(
        self, messages: list[dict], model: str, temperature: float, **kwargs
    ) -> Any:
        """Call the Chat Completions API (all OpenAI-compatible endpoints).

        Args:
            messages: Formatted message list.
            model: Model identifier.
            temperature: Sampling temperature.
            **kwargs: Extra parameters forwarded to the API.

        Returns:
            The raw Chat Completions response object.
        """
        params: dict[str, Any] = {"model": model, "messages": messages}
        if model not in NO_TEMPERATURE_MODELS:
            params["temperature"] = temperature
        params.update(kwargs)
        return self._client.chat.completions.create(**params)

    def _extract_content_chat(self, raw: Any) -> str:
        """Extract text from a Chat Completions API response.

        Args:
            raw: The raw response object from the Chat Completions API.

        Returns:
            The generated text string. Falls back to ``str(raw)`` if
            the expected attributes are absent.
        """
        try:
            return raw.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            return str(raw)
