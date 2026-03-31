"""LLM gateway router.

Selects the correct :class:`LLMProvider` based on the model name,
constructs it with credentials from :class:`~talkingtomachines.config.Config`,
and delegates the ``generate`` call.

Key responsibilities:
    - **Provider detection**: Maps a model-name string to a provider key
      using prefix-matching rules (see :func:`_detect_provider`).
    - **Lazy provider construction**: Provider instances are built on first
      use and cached for the lifetime of the :class:`LLMRouter`.
    - **Retry logic**: Failed API calls are retried with exponential
      back-off up to :attr:`LLMRouter.MAX_RETRIES` attempts.
    - **Budget enforcement**: Each successful call is recorded via
      :class:`~talkingtomachines.gateway.cost_tracker.CostTracker`;
      a :class:`BudgetExhaustedError` halts further generation.

Attributes:
    _OPENAI_PREFIXES (tuple[str, ...]): Model-name prefixes routed to
        OpenAI.
    _ANTHROPIC_PREFIXES (tuple[str, ...]): Model-name prefixes routed to
        Anthropic.
    _GOOGLE_PREFIXES (tuple[str, ...]): Model-name prefixes routed to
        Google.
    _MISTRAL_PREFIXES (tuple[str, ...]): Model-name prefixes routed to
        Mistral AI.
    _GROK_PREFIXES (tuple[str, ...]): Model-name prefixes routed to
        xAI Grok.
    _DEEPSEEK_PREFIXES (tuple[str, ...]): Model-name prefixes routed to
        DeepSeek.
    _OPENROUTER_PREFIXES (tuple[str, ...]): Model-name prefixes routed to
        OpenRouter.
    _PROVIDER_KEY_MAP (dict[str, str]): Maps provider keys to the
        corresponding :class:`~talkingtomachines.config.Config` attribute
        name that stores the API key.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from talkingtomachines.config import Config
from .base import LLMProvider, LLMResponse
from .cost_tracker import BudgetExhaustedError, CostTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model → provider routing table
# ---------------------------------------------------------------------------

_OPENAI_PREFIXES = (
    "gpt-",
    "o1",
    "o3",
    "o4",
    "o5",
    "gpt-4",
    "gpt-5",
    "chatgpt-",
)

_ANTHROPIC_PREFIXES = ("claude-",)

_GOOGLE_PREFIXES = ("gemini-", "models/gemini")

_MISTRAL_PREFIXES = (
    "mistral-",
    "open-mistral-",
    "open-mixtral-",
    "codestral-",
)

_GROK_PREFIXES = ("grok-",)

_DEEPSEEK_PREFIXES = ("deepseek-",)

_OPENROUTER_PREFIXES = ("openrouter/",)


def _detect_provider(model: str) -> str:
    """Detect the provider key for a given model name.

    Matching is case-insensitive. OpenRouter is checked first because its
    model strings may embed other provider names
    (e.g. ``"openrouter/anthropic/claude-3"``). Unrecognised models
    default to ``"openrouter"``.

    Args:
        model: The model identifier string
            (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-20250514"``).

    Returns:
        A lowercase provider key such as ``"openai"``, ``"anthropic"``,
        ``"google"``, ``"mistral"``, ``"grok"``, ``"deepseek"``,
        ``"huggingface"``, or ``"openrouter"``.
    """
    m = model.lower()
    # OpenRouter must be checked before provider-specific prefixes because
    # OpenRouter model strings look like "openrouter/anthropic/claude-3".
    if any(m.startswith(p) for p in _OPENROUTER_PREFIXES):
        return "openrouter"
    if any(m.startswith(p) for p in _OPENAI_PREFIXES):
        return "openai"
    if any(m.startswith(p) for p in _ANTHROPIC_PREFIXES):
        return "anthropic"
    if any(m.startswith(p) for p in _GOOGLE_PREFIXES):
        return "google"
    if any(m.startswith(p) for p in _MISTRAL_PREFIXES):
        return "mistral"
    if any(m.startswith(p) for p in _GROK_PREFIXES):
        return "grok"
    if any(m.startswith(p) for p in _DEEPSEEK_PREFIXES):
        return "deepseek"
    if m == "tgi" or m.startswith("hf-"):
        return "huggingface"
    # Default to OpenRouter for unknown model names
    return "openrouter"


def _build_provider(
    provider: str,
    hf_inference_endpoint: str = "",
) -> LLMProvider:
    """Construct an :class:`LLMProvider` instance for the given provider key.

    Imports the concrete provider class lazily to avoid pulling in
    unnecessary SDK dependencies at module load time. API keys are read
    from :class:`~talkingtomachines.config.Config`.

    Args:
        provider: A provider key returned by :func:`_detect_provider`
            (e.g. ``"openai"``, ``"anthropic"``).
        hf_inference_endpoint: URL of a Hugging Face inference endpoint.
            Only used when *provider* is ``"huggingface"``.

    Returns:
        A fully initialised :class:`LLMProvider` ready for
        :meth:`~LLMProvider.generate` calls.
    """
    cfg = Config()
    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=cfg.ANTHROPIC_API_KEY)
    if provider == "google":
        from .google_provider import GoogleProvider

        return GoogleProvider(api_key=cfg.GOOGLE_API_KEY)
    if provider == "mistral":
        from .mistral_provider import MistralProvider

        return MistralProvider(api_key=cfg.MISTRAL_API_KEY)
    if provider == "grok":
        from .grok_provider import GrokProvider

        return GrokProvider(api_key=cfg.XAI_API_KEY)
    if provider == "deepseek":
        from .deepseek_provider import DeepSeekProvider

        return DeepSeekProvider(api_key=cfg.DEEPSEEK_API_KEY)
    if provider == "huggingface":
        from .huggingface_provider import HuggingFaceProvider

        return HuggingFaceProvider(
            api_key=cfg.HF_API_KEY,
            inference_endpoint=hf_inference_endpoint,
        )
    if provider == "openrouter":
        from .openrouter_provider import OpenRouterProvider

        return OpenRouterProvider(api_key=cfg.OPENROUTER_API_KEY)
    # Default: OpenAI
    from .openai_provider import OpenAIProvider

    return OpenAIProvider(api_key=cfg.OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------

_PROVIDER_KEY_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "grok": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "huggingface": "HF_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def check_provider_api_key(model_name: str) -> tuple[bool, str]:
    """Check whether the API key for the detected provider is configured.

    Detects the provider from *model_name*, looks up the expected
    environment-variable name in :data:`_PROVIDER_KEY_MAP`, and verifies
    that it is set in :class:`~talkingtomachines.config.Config`.

    Args:
        model_name: The model identifier to check
            (e.g. ``"gpt-5"``, ``"claude-sonnet-4-20250514"``).

    Returns:
        A two-element tuple ``(is_valid, error_message)``. When the key
        is present or the provider has no key requirement, *is_valid* is
        ``True`` and *error_message* is an empty string.
    """
    provider = _detect_provider(model_name)
    env_var = _PROVIDER_KEY_MAP.get(provider, "")
    if not env_var:
        return True, ""
    cfg = Config()
    key = getattr(cfg, env_var, "")
    if not key:
        return False, (
            f"Model '{model_name}' requires {env_var} but it is not set. "
            f"Set it as an environment variable or in a .env file."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class LLMRouter:
    """Route ``generate()`` calls to the appropriate LLM provider.

    Handles provider selection, retry with exponential back-off, and
    cost tracking against a configurable budget cap.

    Args:
        cost_tracker: Optional :class:`CostTracker` instance. A default
            tracker with no budget limit is created when ``None``.
        hf_inference_endpoint: URL passed through to the Hugging Face
            provider when that backend is selected.

    Attributes:
        MAX_RETRIES (int): Maximum number of attempts per ``generate``
            call (default ``5``).
        RETRY_DELAY_S (float): Base delay in seconds between retries
            (default ``5.0``). Actual delay grows as
            ``RETRY_DELAY_S * 2 ** attempt``.
    """

    MAX_RETRIES: int = 5
    RETRY_DELAY_S: float = 5.0

    def __init__(
        self,
        cost_tracker: Optional[CostTracker] = None,
        hf_inference_endpoint: str = "",
    ):
        """Initialise the router with optional cost tracking and HF endpoint."""
        self._cost_tracker = cost_tracker or CostTracker()
        self._hf_endpoint = hf_inference_endpoint
        self._provider_cache: dict[str, LLMProvider] = {}

    def generate(
        self,
        message_history: list[dict],
        model: str,
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        """Route to the correct provider and return a normalised response.

        Detects the provider, delegates to
        :meth:`~LLMProvider.generate`, records the cost, and retries on
        transient failures with exponential back-off. If all retries are
        exhausted an empty :class:`LLMResponse` is returned.

        Args:
            message_history: Conversation messages in OpenAI-compatible
                format (list of dicts with ``role`` and ``content``).
            model: Model identifier used for provider detection and
                passed to the underlying SDK.
            temperature: Sampling temperature (default ``0.0``).
            **kwargs: Additional keyword arguments forwarded to the
                provider's ``generate`` method.

        Returns:
            An :class:`LLMResponse` containing the generated text,
            token counts, cost, and latency.

        Raises:
            BudgetExhaustedError: If the cumulative cost exceeds the
                budget cap configured on the :class:`CostTracker`.
        """
        provider_name = _detect_provider(model)
        provider = self._get_provider(provider_name)

        last_exc: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = provider.generate(
                    message_history, model, temperature, **kwargs
                )
                try:
                    self._cost_tracker.add(response.cost_usd)
                except BudgetExhaustedError:
                    raise
                return response
            except BudgetExhaustedError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "LLM call failed (attempt %d/%d, model=%s): %s",
                    attempt + 1,
                    self.MAX_RETRIES,
                    model,
                    exc,
                )
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY_S * (2**attempt))

        logger.error("All %d retries exhausted for model %s", self.MAX_RETRIES, model)
        # Return empty fallback
        return LLMResponse(
            content="",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            model=model,
            provider=provider_name,
            latency_ms=0.0,
            cost_usd=0.0,
            raw_response=None,
        )

    def _get_provider(self, provider_name: str) -> LLMProvider:
        """Return a cached provider instance, building it on first access.

        Args:
            provider_name: A provider key returned by
                :func:`_detect_provider`.

        Returns:
            The :class:`LLMProvider` for *provider_name*.
        """
        if provider_name not in self._provider_cache:
            self._provider_cache[provider_name] = _build_provider(
                provider_name, self._hf_endpoint
            )
        return self._provider_cache[provider_name]
