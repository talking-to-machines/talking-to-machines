"""
Compiled Experiment Package (CEP) schema definition (Phase 3).

The CEP is the output of the compiler and serves as the single source of truth
for runtime execution.  It is serialized to ``compiled_experiment.json`` in
the run folder.

Schema version: 1.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AssignmentPlan:
    """Pre-computed group assignments and manual variable definitions.

    Group assignments come from the ``Manual_`` sheet (entries with
    ``class="Group"`` and ``name="id_in_subsession"``). All other
    ``Manual_`` entries (including treatment) are stored in
    ``manual_variables`` and applied at runtime.

    Attributes:
        random_seed: Seed used for deterministic randomisation.
        group_assignments: Nested mapping of manual group formations
            per module and round
            (``{module: {round: {group_label: [agent_id]}}}``).
            Empty if no manual groups are defined.
        manual_variables: List of manual variable entries from the
            ``Manual_`` sheet. Each entry is a dict with keys
            ``profile_id``, ``module``, ``round_number``, ``class``,
            ``name``, and ``value``.
    """

    random_seed: int
    group_assignments: dict[str, dict] = field(default_factory=dict)
    manual_variables: list[dict] = field(default_factory=list)

    def get_manual_groups(self, module: str, round_number: int) -> dict[str, list[str]]:
        """Return manual group assignments for a specific module and round.

        Args:
            module: Module name.
            round_number: Round number (1-based).

        Returns:
            A dict ``{group_label: [agent_id, ...]}`` if manual groups
            are defined for this module/round, otherwise an empty dict.
        """
        module_dict = self.group_assignments.get(module, {})
        # Round keys may be int (in-memory) or str (after JSON round-trip)
        return module_dict.get(round_number) or module_dict.get(str(round_number), {})

    def to_dict(self) -> dict:
        """Serialise the assignment plan to a plain dictionary.

        Returns:
            A dictionary containing all assignment plan fields.
        """
        return {
            "random_seed": self.random_seed,
            "group_assignments": self.group_assignments,
            "manual_variables": self.manual_variables,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AssignmentPlan":
        """Reconstruct an ``AssignmentPlan`` from a plain dictionary.

        Args:
            data: Dictionary with assignment plan fields, typically
                loaded from a CEP JSON file.

        Returns:
            A new ``AssignmentPlan`` instance.
        """
        return cls(
            random_seed=data.get("random_seed", 0),
            group_assignments=data.get("group_assignments", {}),
            manual_variables=data.get("manual_variables", []),
        )


@dataclass
class CompiledExperiment:
    """Compiled Experiment Package (CEP).

    The CEP is the single source of truth for runtime execution,
    produced by the compiler and serialised to
    ``compiled_experiment.json``. All Jinja compile-time references
    are pre-resolved; runtime references (``round_number``,
    ``player.field``, etc.) remain as Jinja placeholders for
    evaluation at runtime.

    Attributes:
        schema_version: CEP schema version string (e.g. ``"1.0"``).
        experiment_id: Unique experiment identifier from the Settings
            worksheet.
        config_hash: Truncated SHA-256 hash of all parsed worksheet
            data.
        cep_hash: Full SHA-256 hash of the serialised CEP.
        run_id: Unique run identifier incorporating a UTC timestamp
            and random suffix.
        settings: Flat dictionary of experiment settings.
        constants: Nested dictionary of constants per module
            (``{module: {name: typed_value}}``).
        fields: List of field definition dictionaries.
        facilitator_functions: List of facilitator function
            dictionaries.
        prompts: Module-grouped prompt dictionaries
            (``{module: [prompt_dict, ...]}``).
        profiles: Agent profile data from the Profiles worksheet.
        module_sequence: Ordered list of module names.
        assignment_plan: Pre-computed treatment and group assignments.
    """

    schema_version: str
    experiment_id: str
    config_hash: str
    cep_hash: str
    run_id: str
    settings: dict
    constants: dict[str, dict[str, Any]]
    fields: list[dict]
    facilitator_functions: list[dict]
    prompts: dict[str, list[dict]]
    profiles: dict
    module_sequence: list[str]
    assignment_plan: AssignmentPlan

    def to_dict(self) -> dict:
        """Serialise the compiled experiment to a plain dictionary.

        Recursively converts the ``assignment_plan`` via its own
        ``to_dict`` method.

        Returns:
            A dictionary containing all CEP fields, suitable for
            JSON serialisation.
        """
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "cep_hash": self.cep_hash,
            "run_id": self.run_id,
            "settings": self.settings,
            "constants": self.constants,
            "fields": self.fields,
            "facilitator_functions": self.facilitator_functions,
            "prompts": self.prompts,
            "profiles": self.profiles,
            "module_sequence": self.module_sequence,
            "assignment_plan": self.assignment_plan.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the compiled experiment to a JSON string.

        Args:
            indent: Number of spaces for JSON indentation.

        Returns:
            A formatted JSON string representation of the CEP.
        """
        return json.dumps(
            self.to_dict(), indent=indent, default=str, ensure_ascii=False
        )

    def save(self, path: str | Path) -> None:
        """Write the compiled experiment to a JSON file.

        Args:
            path: Destination file path. Parent directories must
                already exist.
        """
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "CompiledExperiment":
        """Reconstruct a ``CompiledExperiment`` from a plain dictionary.

        Args:
            data: Dictionary with CEP fields, typically loaded from
                a ``compiled_experiment.json`` file.

        Returns:
            A new ``CompiledExperiment`` instance.
        """
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            experiment_id=data["experiment_id"],
            config_hash=data.get("config_hash", ""),
            cep_hash=data.get("cep_hash", ""),
            run_id=data.get("run_id", ""),
            settings=data.get("settings", {}),
            constants=data.get("constants", {}),
            fields=data.get("fields", []),
            facilitator_functions=data.get("facilitator_functions", []),
            prompts=data.get("prompts", {}),
            profiles=data.get("profiles", {}),
            module_sequence=data.get("module_sequence", []),
            assignment_plan=AssignmentPlan.from_dict(data.get("assignment_plan", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CompiledExperiment":
        """Load a ``CompiledExperiment`` from a JSON file on disk.

        Args:
            path: Path to a ``compiled_experiment.json`` file.

        Returns:
            A new ``CompiledExperiment`` instance populated from the
            file contents.
        """
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))
