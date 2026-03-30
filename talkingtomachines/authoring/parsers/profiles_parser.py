"""Parser for the Profiles worksheet in experiment template workbooks.

This module reads and validates the Profiles sheet, which defines agent
characteristics used during experiment execution. The sheet follows a
fixed layout:

    - **Row 0**: Short names used as Jinja2 identifiers
      (e.g. ``{{ player.age }}``).
    - **Row 1**: Full question wording (human-readable labels for QA
      format output).
    - **Rows 2+**: One row per agent profile containing attribute values.

The parser enforces that every short name is a valid Jinja2 identifier,
that an ``ID`` column is present with unique values, and that column
counts are consistent between headers and data rows. A ``PROFILE_FIELDS``
filter controls which columns are included in the parsed output.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_JINJA_IDENTIFIER_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")


def parse_profiles(
    df: pd.DataFrame,
    profile_fields: str = "ALL",
) -> dict:
    """Parse the Profiles sheet into a structured dictionary.

    Reads header rows for short names and full names, validates column
    consistency and ID uniqueness, applies the ``PROFILE_FIELDS`` filter,
    and serializes each profile row into a dictionary keyed by short name.

    Args:
        df: Raw ``DataFrame`` of the Profiles worksheet. Row 0 must
            contain short names, row 1 full names, and rows 2+ the
            profile data.
        profile_fields: Comma-separated list of short names to include
            in the output, or ``"ALL"`` (default) to include every
            column.

    Returns:
        A dictionary with the following keys:

        - ``short_names`` – List of short-name strings from row 0.
        - ``full_names`` – List of full question-wording strings from
          row 1.
        - ``rows`` – List of dictionaries (one per agent) mapping
          included short names to their values.
        - ``id_column`` – The short name of the ID column.
        - ``included_columns`` – List of short names retained after
          applying the ``PROFILE_FIELDS`` filter.

    Raises:
        ValueError: If the sheet has fewer than 2 rows, a short name is
            empty or not a valid Jinja2 identifier, column counts are
            mismatched, the ``ID`` column is missing, or IDs are not
            unique.
    """
    if df.shape[0] < 2:
        raise ValueError(
            "Profiles sheet must have at least 2 header rows (short names + full names)."
        )

    short_names: list[str] = [str(v).strip() for v in df.iloc[0].tolist()]
    full_names: list[str] = [str(v).strip() for v in df.iloc[1].tolist()]

    # Validate short_name format for Jinja2 dot notation compatibility
    for sn in short_names:
        if not sn:
            raise ValueError("Profiles sheet: short_name (row 0) must not be empty.")
        if not _JINJA_IDENTIFIER_RE.match(sn):
            raise ValueError(
                f"Profile short_name '{sn}' is not a valid Jinja2 identifier. "
                "Use only letters, digits, and underscores. Must start with a letter."
            )

    # Remaining rows are profile data
    data_df = df.iloc[2:].copy().reset_index(drop=True)
    if data_df.shape[1] != len(short_names):
        raise ValueError(
            f"Profiles sheet: {data_df.shape[1]} data columns but "
            f"{len(short_names)} short names in header row. "
            "Ensure every column has a non-blank header."
        )
    data_df.columns = short_names  # use short names as column headers

    # Identify ID column
    id_col: Optional[str] = None
    for col in short_names:
        if col.upper() == "ID":
            id_col = col
            break

    if id_col is None:
        raise ValueError(
            "Profiles sheet is missing an 'ID' column in the short-name header row."
        )

    # Validate ID uniqueness
    ids = data_df[id_col].dropna().tolist()
    if len(ids) != len(set(str(i) for i in ids)):
        raise ValueError("Profiles sheet: ID column contains duplicate values.")

    # Apply PROFILE_FIELDS filter
    if profile_fields.strip().upper() == "ALL":
        included_cols = short_names
    else:
        requested = [f.strip() for f in profile_fields.split(",") if f.strip()]
        included_cols = [c for c in short_names if c in requested or c.upper() == "ID"]
        missing = set(requested) - set(short_names)
        if missing:
            logger.warning(
                "Profiles: PROFILE_FIELDS requested unknown columns: %s", missing
            )

    # Serialize rows
    rows: list[dict[str, Any]] = []
    for _, row in data_df.iterrows():
        row_dict: dict[str, Any] = {}
        for col in included_cols:
            val = row.get(col)
            if pd.isna(val) if not isinstance(val, str) else False:
                val = None
            row_dict[col] = val
        rows.append(row_dict)

    return {
        "short_names": short_names,
        "full_names": full_names,
        "rows": rows,
        "id_column": id_col,
        "included_columns": included_cols,
    }
