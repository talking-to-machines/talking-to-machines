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
    """Pre-computed treatment and group assignments for an experiment run.

    Stores the strategies used for treatment and group allocation,
    along with the concrete assignment mappings produced by the
    compiler.

    Attributes:
        treatment_strategy: Strategy used for treatment allocation
            (``"manual"`` or ``"complete_random"``).
        group_strategy: Strategy used for group formation
            (``"manual"`` or ``"random"``).
        random_seed: Seed used for deterministic randomisation.
        treatment_assignments: When ``treatment_strategy`` is
            ``"manual"``, a nested mapping
            ``{module: {round: {agent_id: treatment_label}}}``.
            When random, a flat mapping
            ``{agent_id: treatment_label}`` applied across all
            modules and rounds.
        group_assignments: Nested mapping of group formations per module
            and round (``{module: {round: {group_id: [agent_id]}}}``).
    """

    treatment_strategy: str
    group_strategy: str
    random_seed: int
    treatment_assignments: dict = field(default_factory=dict)
    group_assignments: dict[str, dict] = field(default_factory=dict)

    def get_treatment(
        self, agent_id: str, module: str = "", round_number: int = 0
    ) -> str:
        """Resolve the treatment label for an agent.

        For random assignments (flat ``{agent_id: label}``), the label
        is the same regardless of module and round.  For manual
        assignments (nested ``{module: {round: {agent_id: label}}}``),
        the label is looked up by module and round number.

        Args:
            agent_id: The agent identifier.
            module: Module name (used only for manual assignments).
            round_number: Round number (used only for manual assignments).

        Returns:
            The treatment label, or an empty string if not found.
        """
        ta = self.treatment_assignments
        if self.treatment_strategy == "manual":
            module_dict = ta.get(module, {})
            # Round keys may be int (in-memory) or str (after JSON round-trip)
            round_dict = module_dict.get(round_number) or module_dict.get(
                str(round_number), {}
            )
            return round_dict.get(agent_id, "")
        # Flat / random: agent_id → label
        return ta.get(agent_id, "")

    def to_dict(self) -> dict:
        """Serialise the assignment plan to a plain dictionary.

        Returns:
            A dictionary containing all assignment plan fields.
        """
        return {
            "treatment_strategy": self.treatment_strategy,
            "group_strategy": self.group_strategy,
            "random_seed": self.random_seed,
            "treatment_assignments": self.treatment_assignments,
            "group_assignments": self.group_assignments,
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
            treatment_strategy=data.get("treatment_strategy", "simple_random"),
            group_strategy=data.get("group_strategy", "random"),
            random_seed=data.get("random_seed", 0),
            treatment_assignments=data.get("treatment_assignments", {}),
            group_assignments=data.get("group_assignments", {}),
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
