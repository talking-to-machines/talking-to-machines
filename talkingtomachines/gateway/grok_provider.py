"""Grok / xAI gateway provider.

Implements :class:`GrokProvider` as a thin subclass of
:class:`~talkingtomachines.gateway.openai_provider.OpenAIProvider`,
pointing the OpenAI SDK at the xAI Chat Completions endpoint.
"""

from __future__ import annotations

from .openai_provider import OpenAIProvider

_XAI_BASE_URL = "https://api.x.ai/v1"


class GrokProvider(OpenAIProvider):
    """Gateway for Grok (xAI) models via OpenAI-compatible Chat Completions endpoint.

    Args:
        api_key: xAI API key.
    """

    provider_name = "grok"
    _use_chat_completions = True

    def __init__(self, api_key: str):
        """Initialise the Grok provider pointing at the xAI endpoint."""
        super().__init__(api_key=api_key, base_url=_XAI_BASE_URL)
