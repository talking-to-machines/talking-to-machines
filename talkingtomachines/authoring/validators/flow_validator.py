"""Flow validator for prompt template workbooks.

Validates module-level structural and logical constraints:

- ``is_displayed`` expression syntax (via Jinja2 sandbox).
- ``MAX_NUM_ROUNDS`` and ``PLAYERS_PER_GROUP`` present in Constants
  for each module.
- Warns when ``PRIVATE_QUESTION`` prompts write to Group/Session-scoped
  fields.
- Cross-references ``field_class``/``field_name`` in prompts against
  the Fields worksheet.
"""

from __future__ import annotations

import logging
from typing import Any

from .schema_validator import ValidationError

logger = logging.getLogger(__name__)


class FlowValidator:
    """Validates conditional display logic and module structure.

    Checks that each module in the experiment has the required constants,
    valid ``is_displayed`` Jinja expressions, and correct field
    references.

    Attributes:
        _prompts: Mapping of module names to their prompt definitions.
        _constants: Nested dictionary of constants per module.
        _module_names: Ordered list of module names from
            ``MODULE_SEQUENCE``.
        _field_index: Lookup dictionary keyed by
            ``(module, field_class, name)`` for cross-referencing.
        _errors: Accumulated validation errors.
    """

    def __init__(
        self,
        prompts: dict[str, list],
        constants: dict[str, dict[str, Any]],
        module_names: list[str],
        field_index: dict[tuple, Any] | None = None,
    ) -> None:
        """Initialise the flow validator.

        Args:
            prompts: Mapping ``{module: [PromptDefinition, ...]}`` as
                returned by the prompts parser.
            constants: Nested dictionary ``{module: {name: value}}``
                as returned by the constants parser.
            module_names: Ordered list of module names from the
                ``MODULE_SEQUENCE`` setting.
            field_index: Optional lookup dictionary keyed by
                ``(module, field_class, name)`` for cross-referencing
                prompt field references against the Fields worksheet.
        """
        self._prompts = prompts
        self._constants = constants
        self._module_names = module_names
        self._field_index = field_index or {}
        self._errors: list[ValidationError] = []

    def validate(self) -> list[ValidationError]:
        """Run all flow validations and return accumulated errors.

        Iterates over each module in ``MODULE_SEQUENCE`` and checks
        required constants, prompt availability, ``is_displayed``
        syntax, scope warnings for ``PRIVATE_QUESTION``, and field
        cross-references.

        Returns:
            A list of ``ValidationError`` instances. An empty list
            indicates that all checks passed.
        """
        self._errors = []

        for module in self._module_names:
            # Check MAX_NUM_ROUNDS and PLAYERS_PER_GROUP
            module_consts = self._constants.get(module, {})
            if "MAX_NUM_ROUNDS" not in module_consts:
                self._err(
                    "C",
                    None,
                    None,
                    f"Module '{module}': 'MAX_NUM_ROUNDS' is missing from C worksheet.",
                )
            if "PLAYERS_PER_GROUP" not in module_consts:
                self._err(
                    "C",
                    None,
                    None,
                    f"Module '{module}': 'PLAYERS_PER_GROUP' is missing from C worksheet.",
                )

            module_prompts = self._prompts.get(module, [])
            if not module_prompts:
                self._err(
                    "Prompts", None, None, f"Module '{module}': no prompts defined."
                )
                continue

            # Validate is_displayed syntax using Jinja2 sandbox
            for prompt in module_prompts:
                if prompt.is_displayed:
                    self._validate_is_displayed_syntax(prompt.is_displayed, module)

            # Warn when PRIVATE_QUESTION writes to Group/Session-scoped fields
            for prompt in module_prompts:
                if (
                    prompt.type == "PRIVATE_QUESTION"
                    and prompt.field_class in ("Group", "Session")
                    and prompt.field_name
                ):
                    logger.warning(
                        "[sheet=Prompts] Module '%s': PRIVATE_QUESTION prompt (sequence %s) "
                        "writes to %s-scoped field '%s'. When multiple players respond in "
                        "parallel, their responses will be aggregated into a dictionary "
                        "keyed by player_id (e.g., {player_1: value, player_2: value}). "
                        "If you expect a single value per player, use field_class 'Player' "
                        "instead.",
                        module,
                        prompt.prompt_sequence,
                        prompt.field_class,
                        prompt.field_name,
                    )

            # Validate field_class/field_name references against Fields worksheet
            if self._field_index:
                for prompt in module_prompts:
                    if prompt.field_class and prompt.field_name:
                        key = (module, prompt.field_class, prompt.field_name)
                        if key not in self._field_index:
                            self._err(
                                "Prompts",
                                None,
                                "field_class/field_name",
                                f"Module '{module}': prompt (sequence {prompt.prompt_sequence}) "
                                f"references field_class='{prompt.field_class}', "
                                f"field_name='{prompt.field_name}' which does not match any "
                                f"entry in the Fields worksheet.",
                            )

        return self._errors

    def _validate_is_displayed_syntax(self, expression: str, module: str) -> None:
        """Validate that an ``is_displayed`` value is a parseable Jinja expression.

        Wraps the expression in a Jinja ``{% if %}`` block inside a
        sandboxed environment to check for syntax errors.

        Args:
            expression: The raw ``is_displayed`` string from the
                Prompts worksheet.
            module: Module name, used for error reporting.
        """
        try:
            from jinja2.sandbox import SandboxedEnvironment

            env = SandboxedEnvironment()
            # Wrap in a conditional to check syntax
            env.parse(f"{{% if {expression} %}}ok{{% endif %}}")
        except Exception as exc:
            self._err(
                "Prompts",
                None,
                "is_displayed",
                f"Module '{module}': invalid is_displayed expression '{expression}': {exc}",
            )

    def _err(self, sheet: str, row: int | None, col: str | None, message: str) -> None:
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
