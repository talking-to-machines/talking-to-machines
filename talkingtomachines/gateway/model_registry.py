"""Centralized model registry that maps model names to capabilities.

This module maintains a lookup table of supported LLM models and their
token limits. It is used at compile time by ``ContextWindowValidator``
to estimate prompt size and at runtime by ``ContextGuard`` to handle
context-window overflow.

Typical usage example::

    spec = get_model_spec("gpt-4o")
    print(spec.max_context_tokens)  # 128000
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Static specification for a supported LLM model.

    Attributes:
        max_context_tokens: Maximum number of tokens the model can accept
            as combined input and output in a single request.
        max_output_tokens: Maximum number of tokens the model can generate
            in a single response. Defaults to 4096.
    """

    max_context_tokens: int
    max_output_tokens: int = 4096


# Registry keyed by canonical model name.
_MODEL_REGISTRY: dict[str, ModelSpec] = {
    # OpenAI
    "gpt-4.1": ModelSpec(max_context_tokens=1_047_576, max_output_tokens=32_768),
    "gpt-4.1-mini": ModelSpec(max_context_tokens=1_047_576, max_output_tokens=32_768),
    "gpt-4.1-nano": ModelSpec(max_context_tokens=1_047_576, max_output_tokens=32_768),
    "gpt-4o": ModelSpec(max_context_tokens=128_000, max_output_tokens=16_384),
    "gpt-4o-mini": ModelSpec(max_context_tokens=128_000, max_output_tokens=16_384),
    "gpt-5": ModelSpec(max_context_tokens=1_047_576, max_output_tokens=32_768),
    "gpt-5-mini": ModelSpec(max_context_tokens=1_047_576, max_output_tokens=32_768),
    "o1": ModelSpec(max_context_tokens=200_000, max_output_tokens=100_000),
    "o3": ModelSpec(max_context_tokens=200_000, max_output_tokens=100_000),
    "o3-pro": ModelSpec(max_context_tokens=200_000, max_output_tokens=100_000),
    "o4-mini": ModelSpec(max_context_tokens=200_000, max_output_tokens=100_000),
    # Anthropic
    "claude-opus-4": ModelSpec(max_context_tokens=200_000, max_output_tokens=32_000),
    "claude-sonnet-4": ModelSpec(max_context_tokens=200_000, max_output_tokens=32_000),
    "claude-haiku-4": ModelSpec(max_context_tokens=200_000, max_output_tokens=8_192),
    # Google
    "gemini-2.5-pro": ModelSpec(max_context_tokens=1_048_576, max_output_tokens=65_536),
    "gemini-2.5-flash": ModelSpec(
        max_context_tokens=1_048_576, max_output_tokens=65_536
    ),
    "gemini-2.0-flash": ModelSpec(
        max_context_tokens=1_048_576, max_output_tokens=8_192
    ),
    "gemini-2.0-pro": ModelSpec(max_context_tokens=1_048_576, max_output_tokens=8_192),
    "gemini-1.5-flash": ModelSpec(
        max_context_tokens=1_048_576, max_output_tokens=8_192
    ),
    "gemini-1.5-pro": ModelSpec(max_context_tokens=2_097_152, max_output_tokens=8_192),
    # Mistral
    "mistral-large": ModelSpec(max_context_tokens=128_000, max_output_tokens=4_096),
    "mistral-medium": ModelSpec(max_context_tokens=128_000, max_output_tokens=4_096),
    "mistral-small": ModelSpec(max_context_tokens=128_000, max_output_tokens=4_096),
    "open-mistral-7b": ModelSpec(max_context_tokens=32_000, max_output_tokens=4_096),
    "open-mixtral-8x7b": ModelSpec(max_context_tokens=32_000, max_output_tokens=4_096),
    # Grok / xAI
    "grok-3": ModelSpec(max_context_tokens=131_072, max_output_tokens=16_384),
    "grok-3-mini": ModelSpec(max_context_tokens=131_072, max_output_tokens=16_384),
    # DeepSeek
    "deepseek-chat": ModelSpec(max_context_tokens=128_000, max_output_tokens=8_192),
    "deepseek-r1": ModelSpec(max_context_tokens=128_000, max_output_tokens=8_192),
}

# Fallback for unknown models
_DEFAULT_SPEC = ModelSpec(max_context_tokens=128_000, max_output_tokens=4_096)

# Context window estimation defaults (used by ContextGuard and ContextWindowValidator)
OUTPUT_RESERVE_TOKENS = 4_096  # Tokens reserved for the model's response
SAFETY_MARGIN = 0.95  # Fraction of context window used as effective limit
WARNING_THRESHOLD = 0.80  # Utilisation fraction that triggers a compile-time warning
SYSTEM_PROMPT_ESTIMATE = 500  # Conservative token estimate for the system prompt
AVG_RESPONSE_TOKENS = 200  # Conservative average tokens per LLM response turn


def get_model_spec(model_name: str) -> ModelSpec:
    """Look up the ``ModelSpec`` for a given model name.

    Resolution order:
        1. Exact match against the registry.
        2. Strip date suffixes (e.g. ``"gpt-4o-2024-05-13"`` → ``"gpt-4o"``).
        3. Strip ``"-latest"`` suffix (Mistral convention).
        4. Longest prefix match.
        5. Return a conservative default (128 k context, 4 k output).

    Args:
        model_name: The model identifier string (case-insensitive).

    Returns:
        The matching ``ModelSpec``, or ``_DEFAULT_SPEC`` if no match is
        found.
    """
    m = model_name.lower()
    # Exact match
    if m in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[m]
    # Strip date suffixes (e.g., "gpt-4o-2024-05-13" → "gpt-4o")
    base = m.split("-202")[0]
    if base in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[base]
    # Strip "-latest" suffix (Mistral convention)
    if m.endswith("-latest"):
        base_latest = m.removesuffix("-latest")
        if base_latest in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[base_latest]
    # Longest prefix match
    best_key = ""
    for key in _MODEL_REGISTRY:
        if m.startswith(key) and len(key) > len(best_key):
            best_key = key
    if best_key:
        return _MODEL_REGISTRY[best_key]
    return _DEFAULT_SPEC
