"""Provider validator for LLM model names and modality support.

Checks that a given ``model_name`` is recognised by the built-in provider
registry and that the requested modalities (text, audio, video) and features
(RAG) are compatible with the detected provider.

Validations performed:
    - The model name matches at least one known provider prefix or is
      listed explicitly in a provider model set.
    - If the experiment uses RAG, the model must belong to an OpenAI
      provider (current limitation).
    - If the experiment includes video or audio prompts, the model must
      appear in the corresponding capability set.

Attributes:
    _OPENAI_MODELS (set[str]): Explicit set of recognised OpenAI model
        identifiers.
    _ANTHROPIC_PREFIXES (tuple[str, ...]): Name prefixes for Anthropic
        models.
    _GOOGLE_PREFIXES (tuple[str, ...]): Name prefixes for Google models.
    _MISTRAL_PREFIXES (tuple[str, ...]): Name prefixes for Mistral AI
        models.
    _GROK_PREFIXES (tuple[str, ...]): Name prefixes for xAI Grok models.
    _DEEPSEEK_PREFIXES (tuple[str, ...]): Name prefixes for DeepSeek
        models.
    _HF_PREFIXES (tuple[str, ...]): Name prefixes for Hugging Face
        models.
    _VIDEO_CAPABLE (set[str]): Model-name prefixes that support video
        inputs.
    _AUDIO_CAPABLE (set[str]): Model-name prefixes that support audio
        inputs.
"""

from __future__ import annotations

import logging
from typing import Optional

from .schema_validator import ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider / model registry
# ---------------------------------------------------------------------------

_OPENAI_MODELS = {
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-chat-latest",
    "gpt-5-codex",
    "gpt-5-pro",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini",
    "o1",
    "o1-pro",
    "o3-pro",
    "o3",
    "o4-mini",
}

_ANTHROPIC_PREFIXES = ("claude-",)
_GOOGLE_PREFIXES = ("gemini-",)
_MISTRAL_PREFIXES = ("mistral-", "open-mistral-", "open-mixtral-")
_GROK_PREFIXES = ("grok-",)
_DEEPSEEK_PREFIXES = ("deepseek-",)
_HF_PREFIXES = ("hf-", "tgi")

# Models that support video/audio inputs natively
_VIDEO_CAPABLE = {"gemini-", "gpt-4o", "gpt-4.1"}
_AUDIO_CAPABLE = {"gemini-", "gpt-4o", "claude-"}


def _detect_provider(model_name: str) -> Optional[str]:
    """Detect the provider for a given model name.

    Compares *model_name* (case-insensitive) against known provider
    prefixes and the explicit OpenAI model set.

    Args:
        model_name: The model identifier string to look up
            (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-20250514"``).

    Returns:
        A lowercase provider key such as ``"openai"``, ``"anthropic"``,
        ``"google"``, ``"mistral"``, ``"grok"``, ``"deepseek"``, or
        ``"huggingface"``. Returns ``None`` if the model is not
        recognised.
    """
    m = model_name.lower()
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
    if any(m.startswith(p) for p in _HF_PREFIXES):
        return "huggingface"
    if m in _OPENAI_MODELS or any(m.startswith(p) for p in ("gpt-", "o1", "o3", "o4")):
        return "openai"
    return None


class ProviderValidator:
    """Validate a model name against the provider registry.

    Ensures the specified model is recognised, and that any requested
    modalities or features (RAG, video, audio) are supported by the
    detected provider.

    Args:
        model_name: The LLM model identifier to validate.
        has_rag: Whether the experiment uses retrieval-augmented
            generation.
        has_video: Whether the experiment includes video prompt
            inputs.
        has_audio: Whether the experiment includes audio prompt
            inputs.

    Attributes:
        _model_name (str): The model identifier under validation.
        _has_rag (bool): RAG flag.
        _has_video (bool): Video-input flag.
        _has_audio (bool): Audio-input flag.
        _errors (list[ValidationError]): Accumulated validation errors
            populated during :meth:`validate`.
    """

    def __init__(
        self,
        model_name: str,
        has_rag: bool = False,
        has_video: bool = False,
        has_audio: bool = False,
    ):
        self._model_name = model_name
        self._has_rag = has_rag
        self._has_video = has_video
        self._has_audio = has_audio
        self._errors: list[ValidationError] = []

    def validate(self) -> list[ValidationError]:
        """Run all provider and modality checks.

        Returns:
            A list of :class:`ValidationError` instances. An empty list
            indicates the model passed all checks.
        """
        self._errors = []

        provider = _detect_provider(self._model_name)
        if provider is None:
            self._err(
                "Settings",
                None,
                "MODEL_NAME",
                f"Unknown model '{self._model_name}'. Not found in any provider registry. "
                "If this is a new model, add it to the provider registry in "
                "authoring/validators/provider_validator.py.",
            )
            return self._errors

        # RAG only supported on OpenAI this sprint
        if self._has_rag and provider != "openai":
            self._err(
                "Settings",
                None,
                "MODEL_NAME",
                f"RAG (retrieval-augmented generation) is only supported with OpenAI models "
                f"this sprint. Current model '{self._model_name}' uses provider '{provider}'.",
            )

        # Video support
        if self._has_video:
            supported = any(
                self._model_name.lower().startswith(p) for p in _VIDEO_CAPABLE
            )
            if not supported:
                self._err(
                    "Settings",
                    None,
                    "MODEL_NAME",
                    f"Model '{self._model_name}' may not support video inputs. "
                    "Check provider documentation.",
                )

        # Audio support
        if self._has_audio:
            supported = any(
                self._model_name.lower().startswith(p) for p in _AUDIO_CAPABLE
            )
            if not supported:
                self._err(
                    "Settings",
                    None,
                    "MODEL_NAME",
                    f"Model '{self._model_name}' may not support audio inputs. "
                    "Check provider documentation.",
                )

        return self._errors

    def _err(self, sheet: str, row: object, col: object, message: str) -> None:
        """Append a validation error to the internal error list.

        Args:
            sheet: Name of the worksheet where the error originated.
            row: Row index or identifier (may be ``None``).
            col: Column name or identifier (may be ``None``).
            message: Human-readable description of the validation
                failure.
        """
        self._errors.append(
            ValidationError(sheet=sheet, row=row, col=col, message=message)
        )
