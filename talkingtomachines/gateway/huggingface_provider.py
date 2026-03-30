"""HuggingFace TGI gateway provider.

Implements :class:`HuggingFaceProvider` as a subclass of
:class:`~talkingtomachines.gateway.openai_provider.OpenAIProvider`,
pointing the OpenAI SDK at a user-specified HuggingFace Text Generation
Inference endpoint.
"""

from __future__ import annotations

from typing import Any

from .openai_provider import OpenAIProvider
from .base import LLMResponse


class HuggingFaceProvider(OpenAIProvider):
    """Gateway for HuggingFace Text Generation Inference (TGI) endpoints.

    TGI exposes an OpenAI-compatible Chat Completions endpoint, so
    ``_use_chat_completions`` is set to ``True``. The model identifier
    sent to TGI is always ``"tgi"``; the endpoint URL determines the
    actual model being served.

    Args:
        api_key: HuggingFace API key.
        inference_endpoint: The URL of the TGI inference endpoint.

    Raises:
        ValueError: If *inference_endpoint* is empty.
    """

    provider_name = "huggingface"
    _use_chat_completions = True

    def __init__(self, api_key: str, inference_endpoint: str):
        """Initialise the HuggingFace provider with an API key and TGI endpoint URL."""
        if not inference_endpoint:
            raise ValueError(
                "HuggingFace provider requires a non-empty inference_endpoint URL."
            )
        super().__init__(api_key=api_key, base_url=inference_endpoint)

    def generate(
        self,
        message_history: list[dict],
        model: str,
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        """Send messages to the TGI endpoint and return a normalised response.

        The model identifier is overridden to ``"tgi"`` for the API call,
        but the original *model* name is preserved in the returned
        :class:`LLMResponse`.

        Args:
            message_history: Conversation messages in OpenAI-compatible
                format.
            model: The conceptual model name reported in the response.
            temperature: Sampling temperature. Defaults to ``0.0``.
            **kwargs: Additional keyword arguments forwarded to the API.

        Returns:
            An :class:`LLMResponse` with generated text, token counts,
            and latency.
        """
        # TGI uses "tgi" as the model identifier
        result = super().generate(message_history, "tgi", temperature, **kwargs)
        result.model = model  # report the conceptual model name
        result.provider = self.provider_name
        return result
