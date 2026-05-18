"""Command-line interface for the Talking to Machines platform.

This module provides the main CLI entry point that dispatches to
subcommands (init, validate, run) via Click.
A legacy entry point is also supported for backward compatibility
with older template-file invocations.
"""

from __future__ import annotations

import sys


def main():
    """Main CLI entry point that dispatches to subcommands via Click.

    Subcommands:
        init: Create a blank project folder with template files.
        validate: Run all validators on a template file without execution.
        run: Compile and execute an experiment from a template file.

    Raises:
        SystemExit: If a subcommand encounters an error or if Click
            is not installed.
    """
    try:
        import click
    except ImportError:
        print(
            "click is required. Install it with: pip install click>=8.0.0",
            file=sys.stderr,
        )
        sys.exit(1)

    @click.group()
    @click.version_option(package_name="talkingtomachines")
    def cli():
        """Talking to Machines — AI-to-AI experiment platform."""

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------

    @cli.command()
    @click.argument("project_name", default="simple_pgg")
    @click.option(
        "--format",
        "fmt",
        type=click.Choice(["xlsx", "csv"]),
        default="xlsx",
        help="Template format (default: xlsx).",
    )
    @click.option(
        "--output", "-o", default=".", help="Output directory (default: current dir)."
    )
    def init(project_name: str, fmt: str, output: str):
        """Create a blank project folder with template files.

        Args:
            project_name: Name of the new project. Defaults to ``"simple_pgg"``.
            fmt: Template format, either ``"xlsx"`` or ``"csv"``.
            output: Output directory for the generated project folder.
        """
        from talkingtomachines.authoring.template_generator import generate_template

        created = generate_template(
            output_dir=output, project_name=project_name, fmt=fmt
        )
        click.echo(f"Created project '{project_name}':")
        for path in created:
            click.echo(f"  {path}")

    # ------------------------------------------------------------------
    # validate
    # ------------------------------------------------------------------

    def _load_compiler(template_path: str):
        """Load a ``Compiler`` from an Excel file or a directory of CSVs.

        Args:
            template_path: Path to an ``.xlsx`` template file or a directory
                containing CSV template files.

        Returns:
            A ``Compiler`` instance initialised from the given template source.
        """
        from pathlib import Path
        from talkingtomachines.compiler.compiler import Compiler

        p = Path(template_path)
        if p.is_dir():
            return Compiler.from_csv_dir(template_path)
        return Compiler.from_excel(template_path)

    @cli.command()
    @click.argument("template_path")
    def validate(template_path: str):
        """Run all validators on a template file or CSV directory without execution.

        Compiles the template and reports experiment metadata on success,
        or lists validation errors on failure.

        Args:
            template_path: Path to an Excel template file or a directory
                containing CSV template files.

        Raises:
            SystemExit: If validation fails or an unexpected error occurs.
        """
        from talkingtomachines.compiler.compiler import CompilationError

        click.echo(f"Validating: {template_path}")
        try:
            compiler = _load_compiler(template_path)
            cep = compiler.compile(raise_on_error=True)
            click.echo(click.style("Validation passed.", fg="green"))
            click.echo(f"  Experiment ID : {cep.experiment_id}")
            click.echo(f"  Config hash   : {cep.config_hash}")
            click.echo(f"  Modules       : {', '.join(cep.module_sequence)}")
            from talkingtomachines.gateway.model_registry import get_model_spec

            spec = get_model_spec(cep.settings.get("model_name", ""))
            click.echo(
                f"  Context window: {spec.max_context_tokens:,} tokens ({cep.settings.get('model_name', '')})"
            )
            click.echo(
                f"  Overflow policy: {cep.settings.get('context_overflow_policy', 'terminate')}"
            )
        except CompilationError as exc:
            click.echo(
                click.style(
                    f"Validation FAILED ({len(exc.errors)} error(s)):", fg="red"
                )
            )
            for err in exc.errors:
                click.echo(f"  {err}")
            sys.exit(1)
        except Exception as exc:
            click.echo(click.style(f"Error: {exc}", fg="red"))
            sys.exit(1)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    @cli.command()
    @click.argument("template_path")
    @click.option(
        "--test",
        "mode",
        flag_value="test",
        default=True,
        help="Test mode (default): run one group per module, sequential.",
    )
    @click.option(
        "--full-run",
        "mode",
        flag_value="full",
        help="Full run: execute all groups in parallel.",
    )
    @click.option(
        "--budget", type=float, default=0.0, help="Budget cap in USD (0 = no cap)."
    )
    @click.option(
        "--output",
        "-o",
        default=None,
        help="Output base directory (default: same directory as the template).",
    )
    def run(template_path: str, mode: str, budget: float, output: str):
        """Compile and execute an experiment from a template file or CSV directory.

        By default runs in test mode (``--test``), executing only one group
        per module sequentially.  Pass ``--full-run`` to run all groups in
        parallel within a single session.

        Args:
            template_path: Path to an Excel template file or a directory
                containing CSV template files.
            mode: ``"test"`` (default) runs one group per module
                sequentially. ``"full"`` runs all groups in parallel.
            budget: Budget cap in USD. ``0`` means no cap.
            output: Base directory for experiment output.

        Raises:
            SystemExit: If compilation or runtime fails.
        """
        import logging
        from datetime import datetime, timezone

        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
        )

        from pathlib import Path
        from talkingtomachines.compiler.compiler import CompilationError
        from talkingtomachines.orchestrator.runtime import ExperimentRuntime
        from talkingtomachines.storage.artifact_manager import ArtifactManager

        # Default output to the same directory as the template
        if output is None:
            tp = Path(template_path)
            output = str(tp.parent if tp.is_file() else tp)

        click.echo(f"Compiling: {template_path}")
        try:
            compiler = _load_compiler(template_path)
            cep = compiler.compile(output_dir=f"{output}/_cep_tmp", raise_on_error=True)
        except CompilationError as exc:
            click.echo(click.style(f"Compilation failed: {exc}", fg="red"))
            sys.exit(1)

        run_dir = f"{output}/results/{cep.run_id}"
        test_mode = mode == "test"
        click.echo(
            f"Mode: {'Test Mode (1 group per module)' if test_mode else 'Full Run (All groups)'}"
        )
        click.echo(f"Run ID: {cep.run_id}")
        click.echo(f"Output: {run_dir}")

        artifact_mgr = ArtifactManager(
            base_dir=output,
            experiment_id=cep.experiment_id,
            run_id=cep.run_id,
        )
        artifact_mgr.save_config(cep)

        runtime = ExperimentRuntime(cep, output_dir=run_dir, budget_cap_usd=budget)
        start_time = datetime.now(timezone.utc).isoformat()

        num_modules = len(cep.module_sequence)
        progress_bar = click.progressbar(
            length=num_modules,
            label="Running modules",
            show_pos=True,
            item_show_func=lambda t: t or "",
        )

        try:
            with progress_bar as bar:

                def _on_module_complete(_module_idx, module_name, _total):
                    bar.update(1, module_name)

                session = runtime.run(
                    session_number=1,
                    test_mode=test_mode,
                    on_module_complete=_on_module_complete,
                )
        except Exception as exc:
            click.echo(click.style(f"\nRuntime error: {exc}", fg="red"))
            sys.exit(1)

        end_time = datetime.now(timezone.utc).isoformat()

        # Export artifacts
        paths = artifact_mgr.save_all_artifacts(
            session=session,
            cep=cep,
            state=runtime.state,
            start_time=start_time,
            end_time=end_time,
            total_cost_usd=runtime.total_cost_usd,
        )

        click.echo(click.style("Run complete!", fg="green"))
        click.echo(f"  Total cost: ${runtime.total_cost_usd:.4f} USD")
        for name, path in paths.items():
            click.echo(f"  {name}: {path}")

    # ------------------------------------------------------------------
    # Backward-compatible legacy entry point
    # ------------------------------------------------------------------

    # If called with no subcommand args and a .xlsx argument, run legacy path
    if len(sys.argv) > 1 and sys.argv[1].endswith((".xlsx", ".xls", ".csv")):
        # Legacy: talkingtomachines <template_file>
        from talkingtomachines.interface.prompt_template import main as legacy_main

        legacy_main()
        return

    cli()


if __name__ == "__main__":
    main()
