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

        # Extract Manual_ variable names organised by class for validation
        manual_names_by_class: dict[str, set[str]] = {}
        for _pid, _mod, _rnd, cls, name in manual_registry:
            if cls:
                manual_names_by_class.setdefault(cls, set()).add(name)

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
            manual_names_by_class=manual_names_by_class,
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
        # RAG is enabled if any prompt has rag_vector_store_id in kwargs
        _has_rag = any(
            hasattr(p, "kwargs") and p.kwargs.get("rag_vector_store_id")
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
            profiles=profiles,
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
        profiles: dict | None = None,
    ) -> AssignmentPlan:
        """Build assignment plan from ``Manual_`` sheet entries.

        Parses ``Manual_`` entries into group assignments and general
        manual variables. Treatment is no longer a special concept —
        it is just another variable set via the ``Manual_`` sheet.

        Args:
            settings: Parsed experiment settings object.
            agent_ids: List of deterministic agent IDs derived from profiles.
            constants: Parsed constants dictionary keyed by module name.
            rng: Seeded random number generator for reproducibility.
            prompts: Unused, kept for signature compatibility.
            facilitators: Unused, kept for signature compatibility.
            profiles: Unused, kept for signature compatibility.

        Returns:
            An ``AssignmentPlan`` containing group assignments and
            manual variable definitions.
        """
        # Parse Manual_ sheets once
        manual_sheets = {
            k: v for k, v in self._sheets.items() if k.startswith("Manual_")
        }
        manual_registry = parse_manual_sheets(manual_sheets)

        # ------------------------------------------------------------------
        # Separate group assignments from other manual variables
        # ------------------------------------------------------------------
        group_assignments: dict[str, dict] = {}
        manual_variables: list[dict] = []

        for (pid, module, round_num, cls, name), value in manual_registry.items():
            if cls == "Group" and name == "id_in_subsession":
                # Group assignment entry
                agent_id = make_agent_id(settings.experiment_id, pid)
                group_label = str(value)

                # Determine target modules
                target_modules = [module] if module else list(settings.module_sequence)

                for t in target_modules:
                    module_consts = constants.get(t, {})
                    max_rounds = int(module_consts.get("MAX_NUM_ROUNDS", 1))

                    # Determine target rounds
                    if pd.notna(round_num) and round_num != "":
                        target_rounds = [int(round_num)]
                    else:
                        target_rounds = list(range(1, max_rounds + 1))

                    for r in target_rounds:
                        t_dict = group_assignments.setdefault(t, {})
                        r_dict = t_dict.setdefault(r, {})
                        r_dict.setdefault(group_label, []).append(agent_id)
            else:
                # All other entries (including treatment) go into
                # manual_variables for runtime application
                manual_variables.append(
                    {
                        "profile_id": pid,
                        "module": module if module else "",
                        "round_number": (
                            int(round_num)
                            if pd.notna(round_num) and round_num != ""
                            else ""
                        ),
                        "class": cls,
                        "name": name,
                        "value": value,
                    }
                )

        return AssignmentPlan(
            random_seed=settings.random_seed,
            group_assignments=group_assignments,
            manual_variables=manual_variables,
        )
