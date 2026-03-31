"""Token counting utilities for context window estimation.

Provides functions to estimate the number of tokens in text strings and
message lists. Uses the ``tiktoken`` library (``cl100k_base`` encoding)
when available; otherwise falls back to a characters-divided-by-four
heuristic.

The encoder is initialised lazily on first use so that the import cost
of ``tiktoken`` is only paid when token estimation is actually needed.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_encoder = None
_USE_TIKTOKEN = False
_INIT_DONE = False


def _init_encoder() -> None:
    """Initialise the token encoder on first use.

    Attempts to load ``tiktoken`` with the ``cl100k_base`` encoding. If
    the import fails, sets a module-level flag so that subsequent calls
    use the character-count heuristic instead. This function is
    idempotent; repeated calls after initialisation are no-ops.
    """
    global _encoder, _USE_TIKTOKEN, _INIT_DONE
    if _INIT_DONE:
        return
    try:
        import tiktoken

        _encoder = tiktoken.get_encoding("cl100k_base")
        _USE_TIKTOKEN = True
        logger.debug("Using tiktoken for token estimation.")
    except ImportError:
        _USE_TIKTOKEN = False
        logger.debug("tiktoken not available; using chars/4 heuristic.")
    _INIT_DONE = True


def estimate_tokens(text: str) -> int:
    """Estimate the token count of a text string.

    Args:
        text: The input string to tokenise.

    Returns:
        The estimated number of tokens. Always at least ``1`` when
        the heuristic path is used.
    """
    _init_encoder()
    if _USE_TIKTOKEN and _encoder is not None:
        return len(_encoder.encode(text))
    # Heuristic: ~4 characters per token for English text
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens for a list of message dicts.

    Each message contributes the tokens of its ``content`` field plus a
    fixed per-message overhead of 4 tokens (accounting for role tags and
    delimiters). For multimodal content blocks, only the ``text`` parts
    are counted.

    Args:
        messages: A list of message dictionaries, each containing at
            least a ``content`` key whose value is either a string or a
            list of content-block dicts.

    Returns:
        The estimated total token count across all messages.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # Multimodal content blocks — estimate text parts only
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += estimate_tokens(block["text"])
        total += 4  # Per-message overhead (role, delimiters)
    return total
