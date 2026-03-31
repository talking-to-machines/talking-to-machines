"""
Experiment Compiler (Phase 3).

Transforms parsed worksheet data into a validated, serializable
Compiled Experiment Package (CEP).

Pipeline:
  1. Run all four validators (schema → reference → flow → provider)
  2. Call resolver.py to expand statically-resolvable Jinja references
  3. Call id_generator.py to assign stable, deterministic IDs
  4. Emit assignment plans using the randomisation engine
  5. Build CEP and compute cep_hash
  6. Serialize CEP to compiled_experiment.json in the run folder
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import pandas as pd

from talkingtomachines.authoring.parsers.settings_parser import parse_settings
from talkingtomachines.authoring.parsers.constants_parser import parse_constants
from talkingtomachines.authoring.parsers.fields_parser import parse_fields
from talkingtomachines.authoring.parsers.facilitator_parser import parse_facilitators
from talkingtomachines.authoring.parsers.prompts_parser import parse_prompts
from talkingtomachines.authoring.parsers.profiles_parser import parse_profiles
from talkingtomachines.authoring.parsers.manual_parser import parse_manual_sheets
from talkingtomachines.authoring.validators.schema_validator import (
    SchemaValidator,
    ValidationError,
)
from talkingtomachines.authoring.validators.reference_validator import (
    ReferenceValidator,
)
from talkingtomachines.authoring.validators.flow_validator import FlowValidator
from talkingtomachines.authoring.validators.provider_validator import ProviderValidator
from talkingtomachines.compiler.resolver import resolve_prompts, resolve_facilitators
from talkingtomachines.compiler.id_generator import (
    compute_config_hash,
    compute_cep_hash,
    make_run_id,
    make_agent_id,
)
from talkingtomachines.compiler.cep_schema import CompiledExperiment, AssignmentPlan

logger = logging.getLogger(__name__)


class CompilationError(Exception):
    """Raised when compilation fails due to validation errors.

    Attributes:
        errors: List of ``ValidationError`` instances describing each failure.
    """

    def __init__(self, errors: list[ValidationError]):
        """Initialise with a list of validation errors.

        Args:
            errors: One or more ``ValidationError`` instances collected
                during the compilation pipeline.
        """
        self.errors = errors
        lines = "\n".join(f"  {e}" for e in errors)
        super().__init__(f"Compilation failed with {len(errors)} error(s):\n{lines}")


class Compiler:
    """Compiles a prompt template (Excel or dict of DataFrames) into a CEP.

    The compilation pipeline validates all worksheet data, resolves
    compile-time Jinja references, generates deterministic IDs, builds
    treatment/group assignment plans, and emits a serializable
    ``CompiledExperiment`` package.

    Usage::

        compiler = Compiler.from_excel("path/to/template.xlsx")
        cep = compiler.compile(output_dir="experiment_results/my_exp/run_001")

    Attributes:
        _sheets: Dictionary mapping sheet names to their parsed DataFrames.
    """

    def __init__(self, sheets: dict[str, pd.DataFrame]):
        """Initialise the compiler with parsed worksheet data.

        Args:
            sheets: Dictionary mapping sheet names (e.g. ``"Settings"``,
                ``"Prompts"``) to pandas DataFrames.
        """
        self._sheets = sheets

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_excel(cls, path: str | Path) -> "Compiler":
        """Load all sheets from an Excel file.

        Args:
            path: Path to the ``.xlsx`` template file.

        Returns:
            A new ``Compiler`` instance with all sheets parsed.
        """
        xl = pd.ExcelFile(path, engine="openpyxl")
        sheets = {
            name: xl.parse(name, header=None if name == "Profiles" else 0)
            for name in xl.sheet_names
        }
        return cls(sheets)

    @classmethod
    def from_csv_dir(cls, directory: str | Path) -> "Compiler":
        """Load sheets from a directory of CSV files.

        Each CSV filename (without extension) is used as the sheet name.

        Args:
            directory: Path to the directory containing CSV files.

        Returns:
            A new ``Compiler`` instance with all CSVs parsed as sheets.
        """
        d = Path(directory)
        sheets: dict[str, pd.DataFrame] = {}
        for csv_file in d.glob("*.csv"):
            header = None if csv_file.stem == "Profiles" else 0
            sheets[csv_file.stem] = pd.read_csv(csv_file, header=header)
        return cls(sheets)

    # ------------------------------------------------------------------
    # Main compile entry point
    # ------------------------------------------------------------------

    def compile(
        self,
        output_dir: Optional[str | Path] = None,
        raise_on_error: bool = True,
    ) -> CompiledExperiment:
        """Run the full compilation pipeline.

        Executes validation, Jinja resolution, ID generation, assignment
        planning, and CEP construction. Optionally persists the result
        to disk.

        Args:
            output_dir: If provided, saves ``compiled_experiment.json``
                to this directory.
            raise_on_error: If ``True`` (default), raises
                ``CompilationError`` on any validation error. If ``False``,
                logs warnings and continues.

        Returns:
            A ``CompiledExperiment`` instance.

        Raises:
            CompilationError: If validation errors are found and
                ``raise_on_error`` is ``True``.
        """
        all_errors: list[ValidationError] = []

        # ------------------------------------------------------------------
        # Step 1: Parse all sheets
        # ------------------------------------------------------------------
        settings_df = self._sheets.get("Settings", pd.DataFrame())
        settings = parse_settings(settings_df)

        constants_df = self._sheets.get("C", pd.DataFrame())
        constants = parse_constants(constants_df) if not constants_df.empty else {}

        fields_df = self._sheets.get("Fields", pd.DataFrame())
        fields, field_index = (
            parse_fields(fields_df) if not fields_df.empty else ([], {})
        )

        facilitator_df = self._sheets.get("Facilitator", pd.DataFrame())
        facilitators = (
            parse_facilitators(facilitator_df) if not facilitator_df.empty else []
        )

        prompts_df = self._sheets.get("Prompts", pd.DataFrame())
        prompts = parse_prompts(prompts_df) if not prompts_df.empty else {}

        profiles_df = self._sheets.get("Profiles", pd.DataFrame())
        profiles = (
            parse_profiles(profiles_df, settings.profile_fields)
            if not profiles_df.empty
            else {}
        )

        manual_sheets = {
            k: v for k, v in self._sheets.items() if k.startswith("Manual_")
        }
        # Extract profile IDs for cross-validation
        profile_ids = None
        if profiles and profiles.get("rows"):
            pid_col = profiles.get("id_column", "ID")
            profile_ids = {
                row.get(pid_col)
                for row in profiles["rows"]
                if row.get(pid_col) is not None
            }
        manual_registry = parse_manual_sheets(manual_sheets, profile_ids=profile_ids)

        # ------------------------------------------------------------------
        # Step 2: Run validators (collect all errors)
        # ------------------------------------------------------------------
        schema_errors = SchemaValidator(self._sheets).validate()
        all_errors.extend(schema_errors)

        profile_short_names = profiles.get("short_names", []) if profiles else []
        facilitator_names = [f.name for f in facilitators]

        ref_errors = ReferenceValidator(
            sheets=self._sheets,
            constants=constants,
            field_index=field_index,
            module_names=settings.module_sequence,
            profile_short_names=profile_short_names,
            facilitator_names=facilitator_names,
        ).validate()
        all_errors.extend(ref_errors)

        flow_errors = FlowValidator(
            prompts=prompts,
            constants=constants,
            module_names=settings.module_sequence,
            field_index=field_index,
        ).validate()
        all_errors.extend(flow_errors)

        # Detect multimodal and RAG usage from prompt text and settings
        all_prompt_text = " ".join(
            p.llm_text
            for module_prompts in prompts.values()
            for p in module_prompts
            if hasattr(p, "llm_text") and p.llm_text
        )
        from talkingtomachines.gateway.media import has_video, has_audio

        _has_video = has_video(all_prompt_text)
        _has_audio = has_audio(all_prompt_text)
        # RAG is enabled if any prompt has a rag_vector_store_id
        _has_rag = any(
            hasattr(p, "rag_vector_store_id") and p.rag_vector_store_id
            for module_prompts in prompts.values()
            for p in module_prompts
        )

        # API key validation
        from talkingtomachines.gateway.router import check_provider_api_key

        key_ok, key_msg = check_provider_api_key(settings.model_name)
        if not key_ok:
            all_errors.append(
                ValidationError(
                    sheet="Settings",
                    row=None,
                    col="MODEL_NAME",
                    message=key_msg,
                )
            )

        provider_errors = ProviderValidator(
            model_name=settings.model_name,
            has_rag=_has_rag,
            has_video=_has_video,
            has_audio=_has_audio,
        ).validate()
        all_errors.extend(provider_errors)

        from talkingtomachines.authoring.validators.context_window_validator import (
            ContextWindowValidator,
        )

        ctx_errors = ContextWindowValidator(
            model_name=settings.model_name,
            prompts=prompts,
            constants=constants,
            module_sequence=settings.module_sequence,
            num_agents_per_session=settings.num_agents_per_session,
        ).validate()
        all_errors.extend(ctx_errors)

        if all_errors and raise_on_error:
            raise CompilationError(all_errors)
        if all_errors:
            for err in all_errors:
                logger.warning("Validation warning: %s", err)

        # ------------------------------------------------------------------
        # Step 3: Resolve compile-time Jinja refs
        # ------------------------------------------------------------------
        resolved_prompts = resolve_prompts(prompts, constants)
        resolved_facilitators = resolve_facilitators(facilitators, constants)

        # ------------------------------------------------------------------
        # Step 4: Compute config hash and run ID
        # ------------------------------------------------------------------
        parsed_canonical = {
            "settings": settings.to_dict(),
            "constants": constants,
            "fields": [f.to_dict() for f in fields],
            "prompts": {
                t: [p.to_dict() for p in ps] for t, ps in resolved_prompts.items()
            },
        }
        config_hash = compute_config_hash(parsed_canonical)
        rng = random.Random(settings.random_seed)
        run_id = make_run_id(settings.experiment_id, rng)

        # ------------------------------------------------------------------
        # Step 5: Emit assignment plan
        # ------------------------------------------------------------------
        agent_ids = []
        if profiles and profiles.get("rows"):
            id_col = profiles.get("id_column", "ID")
            for row in profiles["rows"]:
                pid = row.get(id_col)
                if pid is not None:
                    agent_ids.append(make_agent_id(settings.experiment_id, pid))

        assignment_plan = self._build_assignment_plan(
            settings=settings,
            agent_ids=agent_ids,
            constants=constants,
            rng=rng,
            prompts=prompts,
            facilitators=facilitators,
        )

        # ------------------------------------------------------------------
        # Step 6: Build CEP
        # ------------------------------------------------------------------
        cep_dict: dict = {
            "schema_version": "1.0",
            "experiment_id": settings.experiment_id,
            "config_hash": config_hash,
            "cep_hash": "",  # filled after
            "run_id": run_id,
            "settings": settings.to_dict(),
            "constants": constants,
            "fields": [f.to_dict() for f in fields],
            "facilitator_functions": [f.to_dict() for f in resolved_facilitators],
            "prompts": {
                t: [p.to_dict() for p in ps] for t, ps in resolved_prompts.items()
            },
            "profiles": profiles,
            "module_sequence": settings.module_sequence,
            "assignment_plan": assignment_plan.to_dict(),
        }
        cep_hash = compute_cep_hash(cep_dict)
        cep_dict["cep_hash"] = cep_hash

        cep = CompiledExperiment.from_dict(cep_dict)

        # ------------------------------------------------------------------
        # Step 7: Save to disk if output_dir given
        # ------------------------------------------------------------------
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            cep.save(out / "compiled_experiment.json")
            logger.info("CEP saved to %s", out / "compiled_experiment.json")

        return cep

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_assignment_plan(
        self,
        settings,
        agent_ids: list[str],
        constants: dict,
        rng: random.Random,
        prompts: dict | None = None,
        facilitators: list | None = None,
    ) -> AssignmentPlan:
        """Build a treatment and group assignment plan.

        Uses ``ASSIGN_MANUALLY`` to determine which assignments come
        from Manual_ sheets (``"Treatment"``, ``"Group"``, or both).
        Assignments not listed are random at session level.

        Args:
            settings: Parsed experiment settings object.
            agent_ids: List of deterministic agent IDs derived from profiles.
            constants: Parsed constants dictionary keyed by module name.
            rng: Seeded random number generator for reproducibility.
            prompts: Parsed prompts dictionary keyed by module name, used
                to validate that ``TREATMENT_LABELS`` is defined when
                prompts reference ``{{ treatment }}``.
            facilitators: Parsed facilitator function list, used to
                validate treatment references in facilitator definitions.

        Returns:
            An ``AssignmentPlan`` containing treatment and group assignments.

        Raises:
            CompilationError: If prompts reference ``{{ treatment }}`` but
                no treatment labels are defined and treatment assignment
                is not manual.
        """
        from talkingtomachines.core.randomisation import RandomisationEngine

        engine = RandomisationEngine(global_seed=settings.random_seed)

        # Parse ASSIGN_MANUALLY into a set of normalised tokens
        manual_modes: set[str] = set()
        if settings.assign_manually:
            manual_modes = {
                t.strip().lower()
                for t in settings.assign_manually.split(",")
                if t.strip()
            }

        # Parse Manual_ sheets once (used by both treatment and group)
        manual_sheets = {
            k: v for k, v in self._sheets.items() if k.startswith("Manual_")
        }
        manual_registry = parse_manual_sheets(manual_sheets)

        # ------------------------------------------------------------------
        # Treatment assignment
        # ------------------------------------------------------------------
        treatment_assignments: dict = {}
        treatment_strategy = "complete_random"
        treatment_labels = self._parse_treatment_labels()

        if "treatment" in manual_modes:
            # Manual treatment: {module: {round: {agent_id: treatment_label}}}
            # Reads entries where name == "treatment" from Manual_ registry.
            # Missing module → apply to all modules; missing round → all rounds.
            treatment_strategy = "manual"
            for (pid, module, round_num, cls, name), value in manual_registry.items():
                if name == "treatment":
                    agent_id = make_agent_id(settings.experiment_id, pid)
                    label = str(value)

                    # Determine target modules
                    if module:
                        target_modules = [module]
                    else:
                        target_modules = list(settings.module_sequence)

                    for t in target_modules:
                        module_consts = constants.get(t, {})
                        max_rounds = int(module_consts.get("MAX_NUM_ROUNDS", 1))

                        # Determine target rounds
                        if pd.notna(round_num) and round_num != "":
                            target_rounds = [int(round_num)]
                        else:
                            target_rounds = list(range(1, max_rounds + 1))

                        for r in target_rounds:
                            module_dict = treatment_assignments.setdefault(t, {})
                            round_dict = module_dict.setdefault(r, {})
                            round_dict[agent_id] = label
        elif treatment_labels and agent_ids:
            # Random assignment at session level: {agent_id: treatment_label}
            treatment_assignments = engine.assign_treatments(
                agent_ids=agent_ids,
                treatment_labels=treatment_labels,
                strategy=treatment_strategy,
                path="treatments",
            )

        # Validate: if prompts or facilitator definitions reference
        # {{ treatment }} but no labels are defined and assignment is not
        # manual, raise an error early.
        if not treatment_assignments:
            all_template_text = ""
            if prompts:
                all_template_text += " ".join(
                    p.llm_text
                    for module_prompts in prompts.values()
                    for p in module_prompts
                    if hasattr(p, "llm_text") and p.llm_text
                )
            if facilitators:
                all_template_text += " " + " ".join(
                    f.definition
                    for f in facilitators
                    if hasattr(f, "definition") and f.definition
                )
            if "treatment" in all_template_text:
                raise CompilationError(
                    [
                        "Templates reference '{{ treatment }}' but no treatment labels are defined. "
                        "Either add TREATMENT_LABELS to the C (Constants) worksheet, "
                        "or set ASSIGN_MANUALLY to 'Treatment' and define treatments "
                        "in a Manual_ worksheet."
                    ]
                )

        # ------------------------------------------------------------------
        # Group assignment plan: {module: {round: {group_id: [agent_ids]}}}
        # ------------------------------------------------------------------
        group_assignments: dict[str, dict] = {}
        group_strategy = "random"

        if "group" in manual_modes:
            # Manual groups: read from Manual_ registry.
            # Entries have class="Group", name="id_in_subsession", value=group label.
            # Missing module → apply to all modules; missing round → all rounds.
            group_strategy = "manual"
            manual_group_plan: dict[str, dict] = {}
            for (pid, module, round_num, cls, name), value in manual_registry.items():
                if cls == "Group" and name == "id_in_subsession":
                    agent_id = make_agent_id(settings.experiment_id, pid)
                    group_label = str(value)

                    # Determine target modules
                    if module:
                        target_modules = [module]
                    else:
                        target_modules = list(settings.module_sequence)

                    for t in target_modules:
                        module_consts = constants.get(t, {})
                        max_rounds = int(module_consts.get("MAX_NUM_ROUNDS", 1))

                        # Determine target rounds
                        if pd.notna(round_num) and round_num != "":
                            target_rounds = [int(round_num)]
                        else:
                            target_rounds = list(range(1, max_rounds + 1))

                        for r in target_rounds:
                            t_dict = manual_group_plan.setdefault(t, {})
                            r_dict = t_dict.setdefault(r, {})
                            r_dict.setdefault(group_label, []).append(agent_id)

            for mod in settings.module_sequence:
                module_consts = constants.get(mod, {})
                max_rounds = int(module_consts.get("MAX_NUM_ROUNDS", 1))
                group_assignments[mod] = {}
                for round_num in range(1, max_rounds + 1):
                    manual = manual_group_plan.get(mod, {}).get(round_num)
                    if manual:
                        groups = engine.assign_groups(
                            agent_ids=agent_ids,
                            players_per_group=len(agent_ids) or 1,
                            strategy="manual",
                            path=f"{mod}.round_{round_num}",
                            manual_groups=manual,
                        )
                    else:
                        groups = engine.assign_groups(
                            agent_ids=agent_ids,
                            players_per_group=len(agent_ids) or 1,
                            strategy="random",
                            path=f"{mod}.round_{round_num}",
                        )
                    group_assignments[mod][round_num] = groups
        else:
            # Random groups at session level — assign once, reuse for all modules/rounds
            players_per_group = len(agent_ids) or 1
            # Use the first module's PLAYERS_PER_GROUP if available
            if settings.module_sequence:
                first_module_consts = constants.get(settings.module_sequence[0], {})
                players_per_group = int(
                    first_module_consts.get("PLAYERS_PER_GROUP", players_per_group)
                )

            session_groups = engine.assign_groups(
                agent_ids=agent_ids,
                players_per_group=players_per_group,
                strategy="random",
                path="session_groups",
            )
            for mod in settings.module_sequence:
                module_consts = constants.get(mod, {})
                max_rounds = int(module_consts.get("MAX_NUM_ROUNDS", 1))
                group_assignments[mod] = {}
                for round_num in range(1, max_rounds + 1):
                    group_assignments[mod][round_num] = session_groups

        return AssignmentPlan(
            treatment_strategy=treatment_strategy,
            group_strategy=group_strategy,
            random_seed=settings.random_seed,
            treatment_assignments=treatment_assignments,
            group_assignments=group_assignments,
        )

    def _parse_treatment_labels(self) -> list[str]:
        """Extract treatment labels from the C (Constants) worksheet.

        Looks for a constant named ``TREATMENT_LABELS`` (under ``global``
        or any module) containing a comma-separated string of labels.

        Returns:
            A list of treatment label strings, or an empty list if
            ``TREATMENT_LABELS`` is not defined.
        """
        constants_df = self._sheets.get("C")
        if constants_df is None or constants_df.empty:
            return []

        constants = parse_constants(constants_df)
        # Check global constants first, then per-module
        for module_consts in constants.values():
            raw = module_consts.get("TREATMENT_LABELS")
            if raw:
                labels = [s.strip() for s in str(raw).split(",") if s.strip()]
                return labels
        return []
