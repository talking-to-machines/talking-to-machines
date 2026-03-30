"""Flow validator for prompt template workbooks.

Validates task-level structural and logical constraints:

- ``is_displayed`` expression syntax (via Jinja2 sandbox).
- ``MAX_NUM_ROUNDS`` and ``PLAYERS_PER_GROUP`` present in Constants
  for each task.
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
    """Validates conditional display logic and task structure.

    Checks that each task in the experiment has the required constants,
    valid ``is_displayed`` Jinja expressions, and correct field
    references.

    Attributes:
        _prompts: Mapping of task names to their prompt definitions.
        _constants: Nested dictionary of constants per task.
        _task_names: Ordered list of task names from
            ``TASK_SEQUENCE``.
        _field_index: Lookup dictionary keyed by
            ``(task, field_class, name)`` for cross-referencing.
        _errors: Accumulated validation errors.
    """

    def __init__(
        self,
        prompts: dict[str, list],
        constants: dict[str, dict[str, Any]],
        task_names: list[str],
        field_index: dict[tuple, Any] | None = None,
    ) -> None:
        """Initialise the flow validator.

        Args:
            prompts: Mapping ``{task: [PromptDefinition, ...]}`` as
                returned by the prompts parser.
            constants: Nested dictionary ``{task: {name: value}}``
                as returned by the constants parser.
            task_names: Ordered list of task names from the
                ``TASK_SEQUENCE`` setting.
            field_index: Optional lookup dictionary keyed by
                ``(task, field_class, name)`` for cross-referencing
                prompt field references against the Fields worksheet.
        """
        self._prompts = prompts
        self._constants = constants
        self._task_names = task_names
        self._field_index = field_index or {}
        self._errors: list[ValidationError] = []

    def validate(self) -> list[ValidationError]:
        """Run all flow validations and return accumulated errors.

        Iterates over each task in ``TASK_SEQUENCE`` and checks
        required constants, prompt availability, ``is_displayed``
        syntax, scope warnings for ``PRIVATE_QUESTION``, and field
        cross-references.

        Returns:
            A list of ``ValidationError`` instances. An empty list
            indicates that all checks passed.
        """
        self._errors = []

        for task in self._task_names:
            # Check MAX_NUM_ROUNDS and PLAYERS_PER_GROUP
            task_consts = self._constants.get(task, {})
            if "MAX_NUM_ROUNDS" not in task_consts:
                self._err(
                    "C",
                    None,
                    None,
                    f"Task '{task}': 'MAX_NUM_ROUNDS' is missing from C worksheet.",
                )
            if "PLAYERS_PER_GROUP" not in task_consts:
                self._err(
                    "C",
                    None,
                    None,
                    f"Task '{task}': 'PLAYERS_PER_GROUP' is missing from C worksheet.",
                )

            task_prompts = self._prompts.get(task, [])
            if not task_prompts:
                self._err("Prompts", None, None, f"Task '{task}': no prompts defined.")
                continue

            # Validate is_displayed syntax using Jinja2 sandbox
            for prompt in task_prompts:
                if prompt.is_displayed:
                    self._validate_is_displayed_syntax(prompt.is_displayed, task)

            # Warn when PRIVATE_QUESTION writes to Group/Session-scoped fields
            for prompt in task_prompts:
                if (
                    prompt.type == "PRIVATE_QUESTION"
                    and prompt.field_class in ("Group", "Session")
                    and prompt.field_name
                ):
                    logger.warning(
                        "[sheet=Prompts] Task '%s': PRIVATE_QUESTION prompt (sequence %s) "
                        "writes to %s-scoped field '%s'. When multiple players respond in "
                        "parallel, their responses will be aggregated into a dictionary "
                        "keyed by player_id (e.g., {player_1: value, player_2: value}). "
                        "If you expect a single value per player, use field_class 'Player' "
                        "instead.",
                        task,
                        prompt.prompt_sequence,
                        prompt.field_class,
                        prompt.field_name,
                    )

            # Validate field_class/field_name references against Fields worksheet
            if self._field_index:
                for prompt in task_prompts:
                    if prompt.field_class and prompt.field_name:
                        key = (task, prompt.field_class, prompt.field_name)
                        if key not in self._field_index:
                            self._err(
                                "Prompts",
                                None,
                                "field_class/field_name",
                                f"Task '{task}': prompt (sequence {prompt.prompt_sequence}) "
                                f"references field_class='{prompt.field_class}', "
                                f"field_name='{prompt.field_name}' which does not match any "
                                f"entry in the Fields worksheet.",
                            )

        return self._errors

    def _validate_is_displayed_syntax(self, expression: str, task: str) -> None:
        """Validate that an ``is_displayed`` value is a parseable Jinja expression.

        Wraps the expression in a Jinja ``{% if %}`` block inside a
        sandboxed environment to check for syntax errors.

        Args:
            expression: The raw ``is_displayed`` string from the
                Prompts worksheet.
            task: Task name, used for error reporting.
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
                f"Task '{task}': invalid is_displayed expression '{expression}': {exc}",
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
