"""Abstract base class for all LLM providers.

Defines the :class:`LLMResponse` data container and the :class:`LLMProvider`
abstract base class that every gateway provider must implement. Provider
implementations inherit from ``LLMProvider`` and override its
:meth:`~LLMProvider.generate` method to call the vendor-specific SDK.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMResponse:
    """Normalised response returned by every provider.

    Attributes:
        content: The generated text content from the model.
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the generated completion.
        total_tokens: Sum of prompt and completion tokens.
        model: The model identifier used for this request.
        provider: The provider key (e.g. ``"openai"``, ``"anthropic"``).
        latency_ms: Wall-clock time for the API call in milliseconds.
        cost_usd: Estimated cost in US dollars. Defaults to ``0.0``.
        raw_response: The unprocessed SDK response object. Defaults to ``None``.
    """

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    provider: str
    latency_ms: float
    cost_usd: float = 0.0
    raw_response: Any = None


class LLMProvider(ABC):
    """Abstract base class every gateway provider must implement.

    Subclasses must set :attr:`provider_name` to a unique string and
    implement :meth:`generate`. Optional capability flags
    (:meth:`supports_vision`, :meth:`supports_audio`, etc.) default to
    ``False`` and can be overridden.

    Attributes:
        provider_name: Short identifier for the provider (e.g. ``"openai"``).
    """

    provider_name: str = "base"

    @abstractmethod
    def generate(
        self,
        message_history: list[dict],
        model: str,
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        """Send ``message_history`` to the model and return a normalised response.

        Args:
            message_history: Conversation messages in OpenAI-compatible format
                (list of dicts with ``role`` and ``content`` keys).
            model: The model identifier string to use for generation.
            temperature: Sampling temperature. Defaults to ``0.0``.
            **kwargs: Additional provider-specific keyword arguments.

        Returns:
            An :class:`LLMResponse` containing the generated text, token
            counts, cost estimate, and latency.
        """

    # ------------------------------------------------------------------
    # Capability flags — override in subclasses that support these modalities
    # ------------------------------------------------------------------

    def supports_vision(self) -> bool:
        """Return True if this provider supports image inputs."""
        return False

    def supports_audio(self) -> bool:
        """Return True if this provider supports audio inputs."""
        return False

    def supports_video(self) -> bool:
        """Return True if this provider supports video inputs."""
        return False

    def supports_structured_output(self) -> bool:
        """Return True if this provider supports structured/JSON output mode."""
        return False

    def _timed_call(self, fn, *args, **kwargs) -> tuple[Any, float]:
        """Call *fn* and return its result together with the elapsed time.

        Args:
            fn: The callable to invoke.
            *args: Positional arguments forwarded to *fn*.
            **kwargs: Keyword arguments forwarded to *fn*.

        Returns:
            A tuple ``(result, elapsed_ms)`` where *result* is the return
            value of *fn* and *elapsed_ms* is the wall-clock duration in
            milliseconds.
        """
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return result, elapsed_ms
