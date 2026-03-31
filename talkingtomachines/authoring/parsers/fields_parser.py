"""Fields worksheet parser.

Converts each row of the Fields worksheet into a ``FieldDefinition``
dataclass and builds an O(1) lookup index keyed by
``(module, field_class, name)``.
"""

from __future__ import annotations

import ast
import logging
from typing import Any

import pandas as pd

from talkingtomachines.core.models import FieldDefinition

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"class", "module", "name", "type"}
_VALID_CLASSES = {"Session", "Subsession", "Agent", "Group", "Player"}
_VALID_TYPES = {"integer", "float", "text", "category", "boolean"}

_BOOL_MAP = {"true": True, "false": False, "1": True, "0": False}


def _parse_bool(value: Any) -> bool:
    """Coerce a cell value to a Python boolean.

    Handles native booleans, ``NaN`` (treated as ``False``), and common
    string representations via ``_BOOL_MAP``.

    Args:
        value: The raw cell value to interpret.

    Returns:
        The boolean interpretation of *value*.
    """
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return _BOOL_MAP.get(str(value).strip().lower(), bool(value))


def _parse_response_options(raw: Any) -> Any:
    """Parse the ``response_options`` cell into a Python object.

    Attempts ``ast.literal_eval`` first to support lists, tuples, and
    dicts written as Python literals. Falls back to splitting on commas
    to produce a list of strings.

    Args:
        raw: The raw cell value (may be ``None``, ``NaN``, a string,
            or an already-parsed Python object).

    Returns:
        A ``list``, ``dict``, ``tuple``, or ``None`` depending on the
        cell content.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, (list, dict, tuple)):
        return raw
    s = str(raw).strip()
    if not s:
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # Treat as comma-separated list of strings
        return [item.strip() for item in s.split(",") if item.strip()]


def parse_fields(df: pd.DataFrame) -> tuple[list[FieldDefinition], dict]:
    """Parse the Fields worksheet into definitions and a lookup index.

    Each row is converted into a ``FieldDefinition`` dataclass. Rows
    missing a ``class``, ``module``, or ``name`` value raise a
    ``ValueError``. Unknown class or type values are accepted with a
    warning.

    Args:
        df: DataFrame for the Fields worksheet with at least the
            columns ``class``, ``module``, ``name``, and ``type``.

    Returns:
        tuple: A two-element tuple containing

        - **fields** -- Ordered list of ``FieldDefinition`` instances.
        - **index** -- Dictionary keyed by ``(module, field_class, name)``
          mapping to the corresponding ``FieldDefinition``.

    Raises:
        ValueError: If any of the required columns are missing from
            the DataFrame.
    """
    cols_lower = {c.strip().lower() for c in df.columns}
    missing = _REQUIRED_COLS - cols_lower
    if missing:
        raise ValueError(f"Fields sheet is missing required columns: {sorted(missing)}")

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    fields: list[FieldDefinition] = []
    index: dict[tuple, FieldDefinition] = {}

    for row_num, row in df.iterrows():
        field_class = str(row.get("class", "")).strip()
        module = str(row.get("module", "")).strip()
        name = str(row.get("name", "")).strip()
        field_type = str(row.get("type", "text")).strip().lower()

        if not field_class or not module or not name:
            missing_parts = []
            if not field_class:
                missing_parts.append("class")
            if not module:
                missing_parts.append("module")
            if not name:
                missing_parts.append("name")
            raise ValueError(
                f"Fields row {row_num}: missing required value(s): {', '.join(missing_parts)}. "
                "Every row must have a class, module, and name."
            )

        if field_class not in _VALID_CLASSES:
            logger.warning(
                "Fields row %s: unknown class '%s'. Expected one of %s.",
                row_num,
                field_class,
                _VALID_CLASSES,
            )

        if field_type not in _VALID_TYPES:
            logger.warning(
                "Fields row %s: unknown type '%s'. Expected one of %s.",
                row_num,
                field_type,
                _VALID_TYPES,
            )

        fd = FieldDefinition(
            field_class=field_class,
            module=module,
            name=name,
            type=field_type,
            response_options=_parse_response_options(row.get("response_options")),
            response_options_intro=str(row.get("response_options_intro", "") or ""),
            randomise_options_order=_parse_bool(row.get("randomise_options_order")),
            validate=_parse_bool(row.get("validate")),
            generate_speculation_score=_parse_bool(
                row.get("generate_speculation_score")
            ),
            format_response=_parse_bool(row.get("format_response")),
        )
        fields.append(fd)
        index[(module, field_class, name)] = fd

    return fields, index
