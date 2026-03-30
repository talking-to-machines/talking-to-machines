"""Schema validator for prompt template workbooks.

Performs full column, field, and enum validation for every worksheet
in a prompt template. Collects all errors and never fails fast, so
that users receive a complete diagnostic report in a single pass.

Error format:
    ``ValidationError(sheet, row, col, message)``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a single validation failure tied to a sheet location.

    Attributes:
        sheet: Name of the worksheet where the error occurred.
        row: Zero-based row index of the error, or ``None`` for
            sheet-level errors.
        col: Column name associated with the error, or ``None`` when
            not applicable.
        message: Human-readable description of the validation failure.
    """

    sheet: str
    row: Optional[int]
    col: Optional[str]
    message: str

    def __str__(self) -> str:
        """Return a bracket-delimited location string followed by the message.

        Returns:
            A formatted string such as
            ``[sheet=Settings, row=3, col=name] Required setting ...``.
        """
        loc = f"[sheet={self.sheet}"
        if self.row is not None:
            loc += f", row={self.row}"
        if self.col is not None:
            loc += f", col={self.col}"
        loc += "]"
        return f"{loc} {self.message}"


class SchemaValidator:
    """Validates the structure of each worksheet in a prompt template.

    Checks that every required sheet is present, that each sheet contains
    the expected columns, and that cell values conform to their allowed
    enumerations (e.g. field types, prompt types, constant types).

    Example:
        >>> validator = SchemaValidator(sheets)
        >>> errors = validator.validate()
        >>> if errors:
        ...     for err in errors:
        ...         print(err)

    Attributes:
        _SHEET_REQUIRED_COLS: Mapping of sheet names to their required
            column names (lower-cased).
        _VALID_FIELD_CLASSES: Allowed values for the ``class`` column
            in the Fields sheet.
        _VALID_FIELD_TYPES: Allowed values for the ``type`` column in
            the Fields sheet.
        _VALID_PROMPT_TYPES: Allowed values for the ``type`` column in
            the Prompts sheet.
        _VALID_CONST_TYPES: Allowed values for the ``type`` column in
            the C (constants) sheet.
    """

    # Required columns per sheet (lower-cased)
    _SHEET_REQUIRED_COLS: dict[str, set[str]] = {
        "Settings": {"name", "value"},
        "C": {"task", "name", "value", "type"},
        "Fields": {"class", "task", "name", "type"},
        "Facilitator": {"name", "definition"},  # optional sheet
        "Prompts": {"task", "prompt_sequence", "type", "llm_text"},
        "Profiles": set(),  # optional; validated separately (row-0 / row-1 header)
        # No Treatments sheet — treatment labels defined in C worksheet
    }

    _VALID_FIELD_CLASSES = {"Session", "Subsession", "Agent", "Group", "Player"}
    _VALID_FIELD_TYPES = {"integer", "float", "text", "category", "boolean"}
    _VALID_PROMPT_TYPES = {
        "CONTEXT",
        "DISCUSSION",
        "PUBLIC_QUESTION",
        "PRIVATE_QUESTION",
        "FACILITATOR",
    }
    _VALID_CONST_TYPES = {
        "integer",
        "int",
        "float",
        "str",
        "string",
        "text",
        "bool",
        "boolean",
    }

    def __init__(self, sheets: dict[str, pd.DataFrame]) -> None:
        """Initialise the validator with parsed worksheet data.

        Args:
            sheets: Mapping of sheet names to their ``DataFrame``
                representations, as returned by the template parser.
        """
        self._sheets = sheets
        self._errors: list[ValidationError] = []

    def validate(self) -> list[ValidationError]:
        """Run all structural validations and return accumulated errors.

        Checks required sheets, required columns, and per-sheet content
        rules. Also auto-discovers and validates any ``Manual_*`` sheets.

        Returns:
            A list of ``ValidationError`` instances. An empty list
            indicates that the template passed all checks.
        """
        self._errors = []

        for sheet_name, required_cols in self._SHEET_REQUIRED_COLS.items():
            if sheet_name not in self._sheets:
                self._err(
                    sheet_name, None, None, f"Required sheet '{sheet_name}' is missing."
                )
                continue

            df = self._sheets[sheet_name]
            self._check_required_columns(sheet_name, df, required_cols)

        # Per-sheet content checks
        if "Settings" in self._sheets:
            self._validate_settings(self._sheets["Settings"])
        if "C" in self._sheets:
            self._validate_constants(self._sheets["C"])
        if "Fields" in self._sheets:
            self._validate_fields(self._sheets["Fields"])
        if "Prompts" in self._sheets:
            self._validate_prompts(self._sheets["Prompts"])
        # Manual_ sheet auto-discovery and validation
        manual_sheets = {
            k: v for k, v in self._sheets.items() if k.startswith("Manual_")
        }
        for sheet_name, df in manual_sheets.items():
            self._validate_manual_sheet(sheet_name, df)

        return self._errors

    # ------------------------------------------------------------------
    # Per-sheet validators
    # ------------------------------------------------------------------

    def _validate_settings(self, df: pd.DataFrame) -> None:
        """Validate the Settings sheet for required experiment parameters.

        Args:
            df: DataFrame for the Settings sheet.
        """
        df = self._normalize_cols(df)
        required_settings = {
            "EXPERIMENT_ID",
            "MODEL_NAME",
            "RANDOM_SEED",
            "NUM_AGENTS_PER_SESSION",
            "TASK_SEQUENCE",
        }
        if "name" not in df.columns:
            return
        present = {str(v).strip().upper() for v in df["name"].dropna()}
        missing = required_settings - present
        for key in sorted(missing):
            self._err("Settings", None, "name", f"Required setting '{key}' not found.")

    def _validate_constants(self, df: pd.DataFrame) -> None:
        """Validate the C (constants) sheet for allowed type values.

        Args:
            df: DataFrame for the C sheet.
        """
        df = self._normalize_cols(df)
        if "type" not in df.columns:
            return
        for i, row in df.iterrows():
            type_val = str(row.get("type", "")).strip().lower()
            if type_val and type_val not in self._VALID_CONST_TYPES:
                self._err("C", i, "type", f"Invalid type '{type_val}'.")

    def _validate_fields(self, df: pd.DataFrame) -> None:
        """Validate the Fields sheet for allowed class and type values.

        Args:
            df: DataFrame for the Fields sheet.
        """
        df = self._normalize_cols(df)
        for i, row in df.iterrows():
            cls = str(row.get("class", "")).strip()
            ftype = str(row.get("type", "")).strip().lower()
            if cls and cls not in self._VALID_FIELD_CLASSES:
                self._err(
                    "Fields",
                    i,
                    "class",
                    f"Invalid class '{cls}'. Expected: {self._VALID_FIELD_CLASSES}.",
                )
            if ftype and ftype not in self._VALID_FIELD_TYPES:
                self._err(
                    "Fields",
                    i,
                    "type",
                    f"Invalid type '{ftype}'. Expected: {self._VALID_FIELD_TYPES}.",
                )

    def _validate_prompts(self, df: pd.DataFrame) -> None:
        """Validate the Prompts sheet for allowed prompt type values.

        Args:
            df: DataFrame for the Prompts sheet.
        """
        df = self._normalize_cols(df)
        for i, row in df.iterrows():
            ptype = str(row.get("type", "")).strip().upper()
            if ptype and ptype not in self._VALID_PROMPT_TYPES:
                self._err(
                    "Prompts",
                    i,
                    "type",
                    f"Invalid type '{ptype}'. Expected: {self._VALID_PROMPT_TYPES}.",
                )

    def _validate_manual_sheet(self, sheet_name: str, df: pd.DataFrame) -> None:
        """Validate a ``Manual_*`` sheet for required columns and class values.

        Args:
            sheet_name: Name of the manual override sheet (e.g.
                ``Manual_Round1``).
            df: DataFrame for the manual sheet.
        """
        from talkingtomachines.authoring.parsers.manual_parser import _VALID_CLASSES

        # "id" matches the Profiles tab ID column (design doc §5.2)
        required = {"id", "task", "round_number", "class", "name", "value"}
        self._check_required_columns(sheet_name, df, required)

        df = self._normalize_cols(df)
        if "class" not in df.columns:
            return
        for i, row in df.iterrows():
            cls = str(row.get("class", "")).strip()
            if cls and cls not in _VALID_CLASSES:
                self._err(
                    sheet_name,
                    i,
                    "class",
                    f"Invalid class '{cls}'. Expected one of {_VALID_CLASSES}.",
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_required_columns(
        self, sheet_name: str, df: pd.DataFrame, required: set[str]
    ) -> None:
        """Check that a DataFrame contains all required columns.

        Column comparison is case-insensitive and strips whitespace.

        Args:
            sheet_name: Name of the sheet being checked (for error
                reporting).
            df: DataFrame whose columns are inspected.
            required: Set of required column names (lower-cased).
        """
        actual = {str(c).strip().lower() for c in df.columns}
        missing = required - actual
        for col in sorted(missing):
            self._err(sheet_name, None, col, f"Required column '{col}' is missing.")

    def _normalize_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the DataFrame with lower-cased, stripped column names.

        Args:
            df: Input DataFrame.

        Returns:
            A shallow copy of *df* with normalised column headers.
        """
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df

    def _err(
        self, sheet: str, row: Optional[int], col: Optional[str], message: str
    ) -> None:
        """Record a validation error.

        Args:
            sheet: Name of the worksheet where the error was found.
            row: Row index, or ``None`` for sheet-level errors.
            col: Column name, or ``None`` when not applicable.
            message: Human-readable error description.
        """
        self._errors.append(
            ValidationError(sheet=sheet, row=row, col=col, message=message)
        )
