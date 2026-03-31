"""Prompt builder for synthetic subject profile construction.

Generates profile prompts and system messages for synthetic subjects by
transforming tabular profile data into textual representations suitable
for LLM system messages.

Strategies:
    QA strategy:
        Interview Q&A format using full question names (row-1 headers).
    Backstory strategy:
        LLM-generated first-person narrative paragraph (requires QA as base).
    Neither:
        No profile information provided (warning logged).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from talkingtomachines.gateway.router import LLMRouter

logger = logging.getLogger(__name__)


def build_profile_prompt(
    profile_row: dict[str, Any],
    short_names: list[str],
    full_names: list[str],
    build_profile_qa: bool = False,
    build_profile_backstories: bool = False,
    router: Optional["LLMRouter"] = None,
    model_name: str = "",
    temperature: float = 0.0,
) -> str:
    """Build a profile prompt string for injection into the system message.

    Constructs a textual representation of an agent's profile data. The
    output format depends on which strategy flags are enabled.

    Args:
        profile_row: Dictionary mapping short column names to values for
            one profile row (e.g., ``{"age": 30, "gender": "F"}``).
        short_names: Ordered list of short (column-header) field names.
        full_names: Ordered list of full question wordings, aligned with
            ``short_names`` by index.
        build_profile_qa: If True, produce an interview Q&A-format prompt.
        build_profile_backstories: If True, use an LLM to generate a
            first-person narrative backstory appended to the Q&A prompt.
        router: LLM router instance used for backstory generation. Required
            when ``build_profile_backstories`` is True.
        model_name: Model identifier passed to the router for backstory
            generation.
        temperature: Sampling temperature for the backstory LLM call.

    Returns:
        A string containing the formatted profile prompt, or an empty
        string if both strategy flags are False.
    """
    if not build_profile_qa and not build_profile_backstories:
        logger.warning(
            "Both BUILD_PROFILE_QA and BUILD_PROFILE_BACKSTORIES are False. "
            "Agent will not receive profile information."
        )
        return ""

    # Build base prompt (Q&A format)
    base_prompt = _build_qa_prompt(profile_row, short_names, full_names)

    if build_profile_backstories and router and model_name:
        backstory = _generate_backstory(base_prompt, router, model_name, temperature)
        if backstory:
            return f"{base_prompt}\n\nBackstory:\n{backstory}"

    return base_prompt


def build_system_message(profile_prompt: str) -> str:
    """Build the system message from the profile prompt.

    Args:
        profile_prompt: Pre-rendered profile text (Q&A and/or backstory).

    Returns:
        The system message string to prepend to the LLM conversation.
    """
    if not profile_prompt:
        return ""
    intro = (
        "Before participating in this study, you were asked to share some "
        "personal information. Your responses are provided below:"
    )
    return f"{intro}\n\n{profile_prompt}"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_qa_prompt(
    profile_row: dict[str, Any],
    short_names: list[str],
    full_names: list[str],
) -> str:
    """Build an interview Q&A-format prompt from a profile row.

    Each non-null, non-ID field is rendered as a two-line exchange with
    the full question wording as the interviewer line and the value as
    the respondent line.

    Args:
        profile_row: Dictionary mapping short column names to profile values.
        short_names: Ordered list of short (column-header) field names.
        full_names: Ordered list of full question wordings corresponding
            to ``short_names``.

    Returns:
        A multi-line string formatted as an interview Q&A transcript.
    """
    name_to_full: dict[str, str] = dict(zip(short_names, full_names))
    lines: list[str] = ["Profile (Q&A format):"]
    for short, value in profile_row.items():
        if short.upper() == "ID":
            continue
        if value is None:
            continue
        full = name_to_full.get(short, short)
        lines.append(f"  Interviewer: {full}")
        lines.append(f"  Me: {value}")
    return "\n".join(lines)


def _generate_backstory(
    system_prompt: str,
    router: "LLMRouter",
    model_name: str,
    temperature: float,
) -> str:
    """Use an LLM to generate a first-person backstory narrative.

    Sends the Q&A system prompt to the router and asks the model to
    produce a brief first-person backstory (2-4 sentences).

    Args:
        system_prompt: The Q&A profile prompt used as context for
            backstory generation.
        router: LLM router instance for generation.
        model_name: Model identifier to use for the generation call.
        temperature: Sampling temperature for the LLM call.

    Returns:
        The generated backstory string, or an empty string if
        generation fails.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Based on the profile information above, write a brief first-person backstory "
                "(2-4 sentences) that brings this person to life. Write in first person."
            ),
        },
    ]
    try:
        response = router.generate(
            message_history=messages,
            model=model_name,
            temperature=temperature,
        )
        return response.content.strip()
    except Exception as exc:
        logger.warning("Backstory generation failed: %s", exc)
        return ""
