"""Jinja2 reference resolver for compile-time constant expansion.

This module performs a partial-evaluation pass over prompt text,
expanding statically-resolvable Jinja2 references while leaving
runtime references intact for later substitution during experiment
execution.

References handled at compile time:

    - ``{{ C.<module>.<name> }}`` — Replaced with the corresponding
      value from the Constants sheet.

References left for runtime resolution:

    - ``{{ <module>.<class>.<name> }}`` — Field references.
    - ``{{ player.<field> }}`` — Profile attribute references.
    - ``{{ round_number }}``, ``{{ group_id }}``, etc. — Built-in
      runtime variables.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_CONSTANT_REF = re.compile(r"\{\{\s*C\.(\w+)\.(\w+)\s*\}\}")


def resolve_static_refs(
    text: str,
    constants: dict[str, dict[str, Any]],
) -> str:
    """Replace constant references in *text* with their compile-time values.

    Scans for ``{{ C.<module>.<name> }}`` patterns and substitutes each
    with the string representation of the corresponding constant value.
    References to unknown modules or constant names are left unchanged so
    that downstream validation can report them.

    Args:
        text: The prompt text containing Jinja2 constant references.
        constants: Nested mapping of
            ``{module_name: {constant_name: value}}`` from the Constants
            sheet.

    Returns:
        A copy of *text* with all resolvable constant references
        replaced by their values.
    """

    def _replace(match: re.Match) -> str:
        """Substitute a single constant reference match with its value.

        Args:
            match: A regex match object capturing the module name (group 1)
                and constant name (group 2) from a ``{{ C.<module>.<name> }}``
                pattern.

        Returns:
            The string representation of the constant value, or the
            original matched text if the module or constant name is not
            found in *constants*.
        """
        module, name = match.group(1), match.group(2)
        module_consts = constants.get(module)
        if module_consts is None:
            logger.debug(
                "Resolver: unknown module '%s' in constant ref — leaving as-is.", module
            )
            return match.group(0)
        if name not in module_consts:
            logger.debug(
                "Resolver: unknown constant '%s.%s' — leaving as-is.", module, name
            )
            return match.group(0)
        return str(module_consts[name])

    return _CONSTANT_REF.sub(_replace, text)


def resolve_prompts(
    prompts: dict[str, list],
    constants: dict[str, dict[str, Any]],
) -> dict[str, list]:
    """Resolve static constant references across all prompt definitions.

    Iterates over every ``PromptDefinition`` in the provided mapping and
    applies ``resolve_static_refs`` to its ``llm_text`` field, producing
    new dataclass instances with the expanded text.

    Args:
        prompts: Mapping of module names to lists of ``PromptDefinition``
            dataclass instances whose ``llm_text`` may contain
            ``{{ C.<module>.<name> }}`` references.
        constants: Nested mapping of
            ``{module_name: {constant_name: value}}`` from the Constants
            sheet.

    Returns:
        A new dictionary with the same structure as *prompts*, where
        each ``PromptDefinition`` has its ``llm_text`` field resolved.
        The original objects are not mutated.
    """
    import dataclasses

    resolved: dict[str, list] = {}
    for module, prompt_list in prompts.items():
        new_list = []
        for prompt in prompt_list:
            updated_text = resolve_static_refs(prompt.llm_text, constants)
            new_prompt = dataclasses.replace(prompt, llm_text=updated_text)
            new_list.append(new_prompt)
        resolved[module] = new_list
    return resolved


def resolve_facilitators(
    facilitators: list,
    constants: dict[str, dict[str, Any]],
) -> list:
    """Resolve static constant references in facilitator definitions.

    Iterates over every ``FacilitatorFunction`` and applies
    ``resolve_static_refs`` to its ``definition`` field, producing
    new dataclass instances with the expanded text.

    Args:
        facilitators: List of ``FacilitatorFunction`` dataclass
            instances whose ``definition`` may contain
            ``{{ C.<module>.<name> }}`` references.
        constants: Nested mapping of
            ``{module_name: {constant_name: value}}`` from the Constants
            sheet.

    Returns:
        A new list of ``FacilitatorFunction`` instances with their
        ``definition`` fields resolved. The original objects are not
        mutated.
    """
    import dataclasses

    resolved = []
    for fn in facilitators:
        updated_def = resolve_static_refs(fn.definition, constants)
        resolved.append(dataclasses.replace(fn, definition=updated_def))
    return resolved
