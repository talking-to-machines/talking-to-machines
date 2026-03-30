"""DeepSeek gateway provider.

Implements :class:`DeepSeekProvider` as a thin subclass of
:class:`~talkingtomachines.gateway.openai_provider.OpenAIProvider`,
pointing the OpenAI SDK at the DeepSeek Chat Completions endpoint.
"""

from __future__ import annotations

from .openai_provider import OpenAIProvider

_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekProvider(OpenAIProvider):
    """Gateway for DeepSeek models via OpenAI-compatible Chat Completions endpoint.

    Args:
        api_key: DeepSeek API key.
    """

    provider_name = "deepseek"
    _use_chat_completions = True

    def __init__(self, api_key: str):
        """Initialise the DeepSeek provider pointing at the DeepSeek endpoint."""
        super().__init__(api_key=api_key, base_url=_DEEPSEEK_BASE_URL)
