"""Cross-sheet Jinja reference validator for experiment templates.

This module validates that every Jinja2 variable reference found in the
Prompts sheet resolves to a known definition in the workbook. It
supports the following reference patterns:

    - ``{{ C.<module>.<name> }}`` — Constant references, resolved against
      the Constants sheet.
    - ``{{ <module>.<class>.<name> }}`` — Field references, resolved
      against the Fields sheet index.
    - ``{{ player.<field> }}`` / ``{{ agent.<field> }}`` — Profile
      attribute references, resolved against Profiles short names,
      Fields definitions, and ``Manual_`` variable names.
    - ``{{ group.<field> }}`` / ``{{ session.<field> }}`` — Scoped
      runtime fields, resolved against Fields definitions, ``Manual_``
      variable names, and built-in attributes.
    - ``{{ round_number }}``, ``{{ group_id }}``, etc. — Built-in
      runtime variables that are always valid.

Unresolved references are collected as ``ValidationError`` instances
and returned to the caller for reporting.
"""

from __future__ import annotations

import re
import logging
from typing import Any

import pandas as pd

from .schema_validator import ValidationError

logger = logging.getLogger(__name__)

# Matches {{ C.module.name }} or {{ module.Class.name }} or {{ player.field }}
_JINJA_VAR = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def _extract_jinja_refs(text: str) -> list[str]:
    """Extract all Jinja2 variable references from a text string.

    Args:
        text: The string to scan for ``{{ ... }}`` expressions.

    Returns:
        A list of dotted reference strings (e.g. ``['C.pgg.ENDOWMENT']``).
        Returns an empty list if *text* is falsy or not a string.
    """
    if not text or not isinstance(text, str):
        return []
    return _JINJA_VAR.findall(text)


class ReferenceValidator:
    """Validate that Jinja2 references in Prompts and Facilitator sheets resolve to known definitions.

    The validator scans the ``llm_text`` column of the Prompts sheet and
    the ``definition`` column of the Facilitator sheet for ``{{ ... }}``
    expressions and checks each one against the constants, fields,
    profile columns, and built-in runtime variables registered at
    construction time. Unresolvable references are collected as
    ``ValidationError`` instances.

    Attributes:
        _sheets: Mapping of sheet names to their raw ``DataFrame`` contents.
        _constants: Nested mapping ``{module_name: {constant_name: value}}``.
        _field_index: Set-like mapping of ``(module, class, name)`` tuples
            parsed from the Fields sheet.
        _module_names: Known module names from the workbook.
        _profile_cols: Short-name column headers from the Profiles sheet.
        _facilitator_names: Facilitator identifiers defined in the workbook.
        _errors: Accumulated validation errors from the most recent run.
    """

    def __init__(
        self,
        sheets: dict[str, pd.DataFrame],
        constants: dict[str, dict[str, Any]],
        field_index: dict[tuple, Any],
        module_names: list[str],
        profile_short_names: list[str],
        facilitator_names: list[str],
        manual_names_by_class: dict[str, set[str]] | None = None,
    ):
        """Initialise the validator with parsed workbook data.

        Args:
            sheets: Mapping of sheet names to their raw ``DataFrame``
                contents.
            constants: Nested mapping of
                ``{module_name: {constant_name: value}}`` from the
                Constants sheet.
            field_index: Mapping of ``(module, class, name)`` tuples from
                the Fields sheet.
            module_names: List of module names defined in the workbook.
            profile_short_names: Short-name column headers from the
                Profiles sheet.
            facilitator_names: Facilitator identifiers defined in the
                workbook.
            manual_names_by_class: Variable names from the ``Manual_``
                sheet(s) grouped by class (e.g. ``{"Player": {"treatment"}}``).
        """
        self._sheets = sheets
        self._constants = constants
        self._field_index = field_index
        self._module_names = set(module_names)
        self._profile_cols = set(profile_short_names)
        self._facilitator_names = set(facilitator_names)
        self._manual_names_by_class = manual_names_by_class or {}
        self._errors: list[ValidationError] = []

        # Collect all field names from the Fields worksheet keyed by scope
        # so that runtime fields are recognised during validation.
        self._field_names_by_class: dict[str, set[str]] = {}
        for module, cls, name in self._field_index:
            self._field_names_by_class.setdefault(cls, set()).add(name)

    def validate(self) -> list[ValidationError]:
        """Run validation on all Jinja2 references in the Prompts and Facilitator sheets.

        Returns:
            A list of ``ValidationError`` instances for references that
            could not be resolved. An empty list indicates all
            references are valid.
        """
        self._errors = []

        # Validate Prompts sheet — llm_text column
        prompts_df = self._sheets.get("Prompts")
        if prompts_df is not None:
            prompts_df = prompts_df.copy()
            prompts_df.columns = [c.strip().lower() for c in prompts_df.columns]
            for row_num, row in prompts_df.iterrows():
                row_type = str(row.get("type", "") or "").strip().upper()
                llm_text = str(row.get("llm_text", "") or "")

                # FACILITATOR rows: llm_text must be a defined function name
                if row_type == "FACILITATOR":
                    func_name = llm_text.strip()
                    if func_name and func_name not in self._facilitator_names:
                        self._err(
                            "Prompts",
                            row_num,
                            "llm_text",
                            f"Facilitator function '{func_name}' is not defined "
                            f"in the Facilitator worksheet.",
                        )
                    continue

                for ref in _extract_jinja_refs(llm_text):
                    self._check_ref(ref, "Prompts", row_num, "llm_text")

        # Validate Facilitator sheet — definition column
        facilitator_df = self._sheets.get("Facilitator")
        if facilitator_df is not None:
            facilitator_df = facilitator_df.copy()
            facilitator_df.columns = [c.strip().lower() for c in facilitator_df.columns]
            for row_num, row in facilitator_df.iterrows():
                definition = str(row.get("definition", "") or "")
                for ref in _extract_jinja_refs(definition):
                    self._check_ref(ref, "Facilitator", row_num, "definition")

        return self._errors

    def _check_ref(self, ref: str, sheet: str, row: int, col: str) -> None:
        """Check a single Jinja2 reference against known definitions.

        Args:
            ref: Dotted reference string (e.g. ``"C.pgg.ENDOWMENT"``).
            sheet: Name of the sheet where the reference was found.
            row: Row index within the sheet.
            col: Column name within the sheet.
        """
        parts = ref.split(".")

        # {{ C.module.name }} — constant reference
        if parts[0] == "C":
            if len(parts) < 3:
                self._err(sheet, row, col, f"Malformed constant ref '{{{{ {ref} }}}}'.")
                return
            module, name = parts[1], parts[2]
            if module not in self._constants:
                self._err(sheet, row, col, f"Constant ref: unknown module '{module}'.")
            elif name not in self._constants.get(module, {}):
                self._err(
                    sheet,
                    row,
                    col,
                    f"Constant ref: unknown constant '{module}.{name}'.",
                )
            return

        # {{ player.field }} / {{ agent.field }} — profile or runtime field
        if parts[0] in ("player", "agent"):
            if len(parts) >= 2:
                field_name = parts[1]
                # Accept profile columns, built-in attributes, field names
                # from the Fields worksheet, and Manual_ variable names.
                _BUILTIN_ATTRS = {
                    "treatment",
                    "role",
                    "agent_id",
                    "agent_instance_id",
                    "player_id",
                }
                all_field_names = (
                    set().union(*self._field_names_by_class.values())
                    if self._field_names_by_class
                    else set()
                )
                all_manual_names = (
                    set().union(*self._manual_names_by_class.values())
                    if self._manual_names_by_class
                    else set()
                )
                if (
                    field_name not in self._profile_cols
                    and field_name not in _BUILTIN_ATTRS
                    and field_name not in all_field_names
                    and field_name not in all_manual_names
                ):
                    logger.warning(
                        "Reference '{{ %s }}' not found in Profiles short_names, "
                        "Fields worksheet, Manual_ worksheet, or built-in "
                        "attributes — possible typo (known columns: %s).",
                        ref,
                        sorted(self._profile_cols),
                    )
            return

        # {{ group.field }} / {{ session.field }} — runtime scoped fields
        if parts[0] in ("group", "session"):
            if len(parts) >= 2:
                field_name = parts[1]
                _BUILTIN_GROUP_ATTRS = {"group_id", "num_players"}
                _BUILTIN_SESSION_ATTRS = {"session_id", "run_id", "experiment_id"}
                builtin_attrs = (
                    _BUILTIN_GROUP_ATTRS
                    if parts[0] == "group"
                    else _BUILTIN_SESSION_ATTRS
                )
                all_field_names = (
                    set().union(*self._field_names_by_class.values())
                    if self._field_names_by_class
                    else set()
                )
                manual_class = "Group" if parts[0] == "group" else "Session"
                manual_names = self._manual_names_by_class.get(manual_class, set())
                if (
                    field_name not in builtin_attrs
                    and field_name not in all_field_names
                    and field_name not in manual_names
                ):
                    logger.warning(
                        "Reference '{{ %s }}' not found in Fields worksheet, "
                        "Manual_ worksheet, or built-in attributes — "
                        "possible typo.",
                        ref,
                    )
            return

        # {{ round_number }} — runtime reference, always valid
        if ref in ("round_number", "group_id", "session_id", "run_id", "treatment"):
            return

        # {{ module.Class.name }} — field reference
        if len(parts) == 3:
            module, cls, name = parts
            if module not in self._module_names:
                self._err(sheet, row, col, f"Field ref: unknown module '{module}'.")
            elif (module, cls, name) not in self._field_index:
                logger.debug(
                    "Field ref '%s' not found in Fields sheet — may be valid runtime field.",
                    ref,
                )
            return

        logger.debug("Unrecognised Jinja ref pattern: '%s'", ref)

    def _err(self, sheet: str, row: int, col: str, message: str) -> None:
        """Record a validation error.

        Args:
            sheet: Name of the sheet containing the error.
            row: Row index where the error was found.
            col: Column name where the error was found.
            message: Human-readable description of the validation failure.
        """
        self._errors.append(
            ValidationError(sheet=sheet, row=row, col=col, message=message)
        )
