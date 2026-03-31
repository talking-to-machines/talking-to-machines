"""Manual assignment worksheet parser

Supports two modes for reading manual assignment data from Excel workbooks:

    - **Single-tab**: A sheet named exactly ``Manual_``.
    - **Multi-tab**: One or more sheets named ``Manual_*``
      (e.g. ``Manual_Treatments``).

All ``Manual_`` sheets are merged into a unified registry keyed by
``(profile_id, module, round_number, class, name)``.

The ``id`` column corresponds to the ``ID`` column in the Profiles worksheet.

Accepted ``class`` values:
    Session, Module, Subsession, Group, Agent, Player.

Attributes:
    _REQUIRED_COLS (set[str]): Column names that every ``Manual_`` sheet must
        contain (lower-cased): ``id``, ``module``, ``round_number``, ``class``,
        ``name``, and ``value``.
    _VALID_CLASSES (set[str]): The set of accepted hierarchy class names.
    ManualRegistry: Type alias ``dict[tuple[Any, str, Any, str, str], Any]``
        — the return type produced by :func:`parse_manual_sheets`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"id", "module", "round_number", "class", "name", "value"}
_VALID_CLASSES = {"Session", "Module", "Subsession", "Group", "Agent", "Player"}

ManualRegistry = dict[tuple[Any, str, Any, str, str], Any]


def _parse_manual_sheet(df: pd.DataFrame, sheet_name: str = "Manual_") -> list[dict]:
    """Parse a single ``Manual_`` sheet into a list of record dicts.

    Normalises column names to lower-case, validates that all required
    columns are present, and raises a ``ValueError`` if ``id`` or
    ``name`` is missing. At least one of ``module`` or ``class`` must
    also be provided; if both are missing, a ``ValueError`` is raised.

    Args:
        df: Raw DataFrame read from one ``Manual_`` Excel tab.
        sheet_name: Name of the sheet (used for error messages).

    Returns:
        A list of record dicts, each containing keys ``profile_id``,
        ``module``, ``round_number``, ``class``, ``name``, ``value``,
        and ``_sheet``.

    Raises:
        ValueError: If any of the required columns are missing from
            the sheet.
    """
    cols_lower = {str(c).strip().lower() for c in df.columns}
    missing = _REQUIRED_COLS - cols_lower
    if missing:
        raise ValueError(
            f"Manual sheet '{sheet_name}' is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    records: list[dict] = []
    for row_num, row in df.iterrows():
        profile_id = row.get("id")
        module = str(row.get("module", "")).strip()
        round_number = row.get("round_number")
        cls = str(row.get("class", "")).strip()
        name = str(row.get("name", "")).strip()
        value = row.get("value")

        missing_parts = []
        if pd.isna(profile_id):
            missing_parts.append("id")
        if not name:
            missing_parts.append("name")
        if missing_parts:
            raise ValueError(
                f"Manual sheet '{sheet_name}' row {row_num}: missing required value(s): "
                f"{', '.join(missing_parts)}. Every row must have an id and name."
            )
        if not module and not cls:
            raise ValueError(
                f"Manual sheet '{sheet_name}' row {row_num}: both 'module' and 'class' are empty. "
                "At least one of 'module' or 'class' must be provided."
            )

        if cls not in _VALID_CLASSES:
            logger.warning(
                "Manual sheet '%s' row %s: invalid class '%s'. Expected one of %s.",
                sheet_name,
                row_num,
                cls,
                _VALID_CLASSES,
            )

        records.append(
            {
                "profile_id": profile_id,
                "module": module,
                "round_number": round_number,
                "class": cls,
                "name": name,
                "value": value,
                "_sheet": sheet_name,
            }
        )

    return records


def parse_manual_sheets(
    sheets: dict[str, pd.DataFrame],
    profile_ids: Optional[set] = None,
) -> ManualRegistry:
    """Parse one or more ``Manual_`` sheets and merge into a unified registry.

    Iterates over all sheets whose names start with ``Manual_``, parses
    each with ``_parse_manual_sheet``, and combines the results into
    a single ``ManualRegistry``. Duplicate keys are overwritten with
    a logged warning.

    Args:
        sheets: Mapping of ``sheet_name`` to ``DataFrame``. The caller
            should pass only sheets whose names start with ``Manual_``;
            non-matching names are silently skipped.
        profile_ids: Optional set of valid profile IDs from the Profiles
            worksheet. When provided, any ``id`` value not found in this
            set causes a ``ValueError``.

    Returns:
        A :data:`ManualRegistry` dict keyed by
        ``(profile_id, module, round_number, class, name)`` with the
        corresponding ``value`` from the worksheet.

    Raises:
        ValueError: If ``profile_ids`` is supplied and a record references
            an ID not present in that set.
    """
    registry: ManualRegistry = {}

    for sheet_name, df in sheets.items():
        if not sheet_name.startswith("Manual_"):
            continue
        records = _parse_manual_sheet(df, sheet_name=sheet_name)
        for rec in records:
            # Cross-validate profile IDs if provided
            if profile_ids is not None:
                pid = rec["profile_id"]
                # Compare as strings for consistency
                if str(pid) not in {str(p) for p in profile_ids}:
                    raise ValueError(
                        f"Manual sheet '{sheet_name}': ID '{pid}' not found in Profiles worksheet."
                    )

            key = (
                rec["profile_id"],
                rec["module"],
                rec["round_number"],
                rec["class"],
                rec["name"],
            )
            if key in registry:
                logger.warning(
                    "Manual registry: duplicate key %s — overwriting with value from '%s'.",
                    key,
                    sheet_name,
                )
            registry[key] = rec["value"]

    return registry
