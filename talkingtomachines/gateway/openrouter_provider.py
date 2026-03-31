"""OpenRouter gateway provider.

Implements :class:`OpenRouterProvider` as a thin subclass of
:class:`~talkingtomachines.gateway.openai_provider.OpenAIProvider`,
pointing the OpenAI SDK at the OpenRouter Chat Completions endpoint.
"""

from __future__ import annotations

from .openai_provider import OpenAIProvider

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAIProvider):
    """Gateway for OpenRouter-proxied models via OpenAI-compatible Chat Completions endpoint.

    Args:
        api_key: OpenRouter API key.
    """

    provider_name = "openrouter"
    _use_chat_completions = True

    def __init__(self, api_key: str):
        """Initialise the OpenRouter provider pointing at the OpenRouter endpoint."""
        super().__init__(api_key=api_key, base_url=_OPENROUTER_BASE_URL)
