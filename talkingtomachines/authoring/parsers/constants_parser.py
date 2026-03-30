"""Constants (C) worksheet parser.

Converts rows of ``{task, name, value, type}`` into a nested dictionary
keyed by task and constant name::

    {task: {name: typed_value}}

Jinja access pattern in prompt templates: ``{{ C.<task>.<name> }}``
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"task", "name", "value", "type"}

_TYPE_MAP: dict[str, type] = {
    "integer": int,
    "int": int,
    "float": float,
    "str": str,
    "string": str,
    "text": str,
    "bool": bool,
    "boolean": bool,
}

_BOOL_MAP = {"true": True, "false": False, "1": True, "0": False}


def _coerce_value(raw: Any, type_str: str) -> Any:
    """Cast a raw cell value to the Python type specified in the worksheet.

    Uses ``_TYPE_MAP`` to resolve the type string and ``_BOOL_MAP`` for
    common boolean representations. Falls back to ``str`` if the cast
    fails.

    Args:
        raw: The raw cell value read from the worksheet.
        type_str: Type label from the ``type`` column (e.g.
            ``"integer"``, ``"bool"``, ``"text"``).

    Returns:
        The coerced value, or ``None`` if *raw* is ``NaN``.
    """
    t = str(type_str).strip().lower()
    dtype = _TYPE_MAP.get(t, str)
    if pd.isna(raw):
        return None
    if dtype is bool:
        return _BOOL_MAP.get(str(raw).strip().lower(), bool(raw))
    try:
        return dtype(raw)
    except (ValueError, TypeError):
        logger.warning(
            "Constants: could not coerce '%s' to %s; keeping as string.", raw, dtype
        )
        return str(raw)


def parse_constants(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Parse the Constants (C) worksheet into a nested dictionary.

    Each row is expected to contain a task name, constant name, raw
    value, and type label. The function groups constants by task and
    applies type coercion. Per-task required constants
    (``MAX_NUM_ROUNDS``, ``PLAYERS_PER_GROUP``) are validated by the
    ``FlowValidator``.

    Args:
        df: DataFrame for the C worksheet with columns ``task``,
            ``name``, ``value``, and ``type``.

    Returns:
        A nested dictionary ``{task_name: {constant_name: typed_value}}``.

    Raises:
        ValueError: If required columns are missing.
    """
    cols_lower = {c.strip().lower() for c in df.columns}
    missing = _REQUIRED_COLS - cols_lower
    if missing:
        raise ValueError(
            f"Constants sheet is missing required columns: {sorted(missing)}"
        )

    # Normalise column names to lower-case
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    result: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        task = str(row["task"]).strip() if not pd.isna(row["task"]) else ""
        name = str(row["name"]).strip() if not pd.isna(row["name"]) else ""
        type_str = str(row.get("type", "string")).strip()
        raw_value = row["value"]

        if not task or not name:
            continue

        value = _coerce_value(raw_value, type_str)
        result.setdefault(task, {})[name] = value

    return result
