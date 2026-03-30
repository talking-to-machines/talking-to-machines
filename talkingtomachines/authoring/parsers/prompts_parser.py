"""Prompts worksheet parser.

Converts each row of the Prompts worksheet into a
``PromptDefinition`` dataclass, grouped by task and sorted by
``prompt_sequence``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from talkingtomachines.core.models import PromptDefinition

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"task", "prompt_sequence", "type", "llm_text"}
_VALID_TYPES = {
    "CONTEXT",
    "DISCUSSION",
    "PUBLIC_QUESTION",
    "PRIVATE_QUESTION",
    "FACILITATOR",
}
_FIELD_REQUIRED_TYPES = {"PUBLIC_QUESTION", "PRIVATE_QUESTION", "DISCUSSION"}

_BOOL_MAP = {"true": True, "false": False, "1": True, "0": False}


def _parse_bool(value: Any) -> bool:
    """Coerce a cell value to a Python boolean.

    Handles native booleans, ``NaN`` (treated as ``False``), and
    common string representations via ``_BOOL_MAP``. Defaults to
    ``False`` for unrecognised values.

    Args:
        value: The raw cell value to interpret.

    Returns:
        The boolean interpretation of *value*.
    """
    if isinstance(value, bool):
        return value
    if pd.isna(value) if not isinstance(value, str) else False:
        return False
    return _BOOL_MAP.get(str(value).strip().lower(), False)


def _parse_optional_str(value: Any) -> Optional[str]:
    """Convert a cell value to an optional string.

    Returns ``None`` for ``None``, ``NaN``, empty strings, and the
    sentinel strings ``"nan"`` and ``"none"``.

    Args:
        value: The raw cell value to convert.

    Returns:
        The stripped string, or ``None`` if the value is considered
        absent.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def parse_prompts(df: pd.DataFrame) -> dict[str, list[PromptDefinition]]:
    """Parse the Prompts worksheet into task-grouped prompt definitions.

    Each row is converted into a ``PromptDefinition``. Rows missing a
    ``task`` or ``llm_text`` value raise a ``ValueError``. Unknown
    prompt types are accepted with a warning. Results are grouped by
    task and sorted by ``prompt_sequence`` within each group.

    Args:
        df: DataFrame for the Prompts worksheet with at least the
            columns ``task``, ``prompt_sequence``, ``type``, and
            ``llm_text``.

    Returns:
        A dictionary ``{task_name: [PromptDefinition, ...]}`` where
        each list is sorted by ``prompt_sequence``.

    Raises:
        ValueError: If any of the required columns are missing from
            the DataFrame.
    """
    cols_lower = {c.strip().lower() for c in df.columns}
    missing = _REQUIRED_COLS - cols_lower
    if missing:
        raise ValueError(
            f"Prompts sheet is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    all_prompts: list[PromptDefinition] = []

    for row_num, row in df.iterrows():
        task = str(row.get("task", "")).strip()
        prompt_type = str(row.get("type", "")).strip().upper()
        llm_text = str(row.get("llm_text", "")).strip()

        if not task or not llm_text:
            missing_parts = []
            if not task:
                missing_parts.append("task")
            if not llm_text:
                missing_parts.append("llm_text")
            raise ValueError(
                f"Prompts row {row_num}: missing required value(s): {', '.join(missing_parts)}. "
                "Every row must have a task and llm_text."
            )

        try:
            seq = int(row.get("prompt_sequence", 0))
        except (ValueError, TypeError):
            seq = 0

        if prompt_type not in _VALID_TYPES:
            logger.warning(
                "Prompts row %s: unknown type '%s'. Expected one of %s.",
                row_num,
                prompt_type,
                _VALID_TYPES,
            )

        field_class = _parse_optional_str(row.get("field_class"))
        field_name = _parse_optional_str(row.get("field_name"))

        if prompt_type in _FIELD_REQUIRED_TYPES:
            missing_parts = []
            if not field_class:
                missing_parts.append("field_class")
            if not field_name:
                missing_parts.append("field_name")
            if missing_parts:
                raise ValueError(
                    f"Prompts row {row_num}: {prompt_type} prompt is missing required value(s): "
                    f"{', '.join(missing_parts)}. "
                    f"{prompt_type} prompts must have a field_class and field_name."
                )

        all_prompts.append(
            PromptDefinition(
                task=task,
                prompt_sequence=seq,
                type=prompt_type,
                is_displayed=_parse_optional_str(row.get("is_displayed")),
                is_adapted=_parse_bool(row.get("is_adapted")),
                human_text=str(row.get("human_text", "") or "").strip(),
                llm_text=llm_text,
                field_class=field_class,
                field_name=field_name,
                rag_vector_store_id=_parse_optional_str(row.get("rag_vector_store_id")),
            )
        )

    # Group by task and sort by prompt_sequence
    result: dict[str, list[PromptDefinition]] = {}
    for pd_obj in sorted(all_prompts, key=lambda p: (p.task, p.prompt_sequence)):
        result.setdefault(pd_obj.task, []).append(pd_obj)

    return result
