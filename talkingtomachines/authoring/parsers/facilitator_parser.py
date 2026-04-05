"""Facilitator worksheet parser.

Converts rows of ``{name, definition, kwargs}`` into
``FacilitatorFunction`` dataclasses.

Built-in functions: ``assign_groups``. All other names are treated as
custom LLM natural-language instruction functions.
"""

from __future__ import annotations

import ast
import logging

import pandas as pd

from talkingtomachines.core.models import FacilitatorFunction

logger = logging.getLogger(__name__)

_BUILTIN_NAMES = {"assign_groups"}
_REQUIRED_COLS = {"name", "definition"}


def _parse_kwargs(raw) -> dict:
    """Parse a kwargs cell into a Python dictionary.

    Uses ``ast.literal_eval`` to interpret the cell as a dict literal.
    Returns an empty dict for ``None``, ``NaN``, empty strings, or
    values that cannot be parsed.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none"):
        return {}
    try:
        result = ast.literal_eval(s)
        return result if isinstance(result, dict) else {}
    except (ValueError, SyntaxError):
        logger.warning(
            "Facilitator: could not parse kwargs '%s' as dict; using empty dict.", s
        )
        return {}


def parse_facilitators(df: pd.DataFrame) -> list[FacilitatorFunction]:
    """Parse the Facilitator worksheet into a list of function definitions.

    Each row becomes a ``FacilitatorFunction``. Rows with an empty
    ``name`` raise a ``ValueError``. Names not in ``_BUILTIN_NAMES``
    are logged as custom LLM instruction functions.

    Args:
        df: DataFrame for the Facilitator worksheet with at least the
            columns ``name`` and ``definition``. An optional ``kwargs``
            column provides per-function keyword arguments.

    Returns:
        An ordered list of ``FacilitatorFunction`` instances.

    Raises:
        ValueError: If any of the required columns are missing from
            the DataFrame.
    """
    cols_lower = {c.strip().lower() for c in df.columns}
    missing = _REQUIRED_COLS - cols_lower
    if missing:
        raise ValueError(
            f"Facilitator sheet is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    has_kwargs = "kwargs" in df.columns

    functions: list[FacilitatorFunction] = []

    for row_num, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        definition = str(row.get("definition", "")).strip()

        if not name:
            raise ValueError(
                f"Facilitator row {row_num}: missing required 'name' value. "
                "Every row must have a function name."
            )

        if name not in _BUILTIN_NAMES:
            logger.debug(
                "Facilitator: '%s' is a custom LLM instruction function.", name
            )

        kwargs = _parse_kwargs(row.get("kwargs")) if has_kwargs else {}

        functions.append(
            FacilitatorFunction(
                name=name,
                definition=definition,
                kwargs=kwargs,
            )
        )

    return functions
