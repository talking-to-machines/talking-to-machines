"""Template generator for blank experiment workbooks.

Creates a blank ``prompt_template.xlsx`` workbook with seven worksheets
(Settings, C, Fields, Facilitator, Prompts, Profiles, Manual\_) containing
bold headers and example rows for use with ``talkingtomachines init``.

Supports two output formats:

    - **xlsx** (default): A single Excel workbook with one tab per sheet.
    - **csv**: Individual CSV files, one per sheet, written into a project
      subdirectory.

Attributes:
    _SHEETS (dict[str, dict]): Registry of sheet definitions. Each entry
        maps a sheet name to a dict with ``columns`` (list of header
        strings), ``rows`` (list of example data rows), and optional
        Profiles-specific keys (``header_row0``, ``header_row1``).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Literal

import pandas as pd


# ---------------------------------------------------------------------------
# Sheet definitions  — (headers, example_rows)
# ---------------------------------------------------------------------------

_SHEETS: dict[str, dict] = {
    "Settings": {
        "columns": ["name", "value"],
        "rows": [
            ["EXPERIMENT_ID", "my_experiment"],
            ["MODEL_NAME", "gpt-4.1-mini"],
            ["HF_INFERENCE_ENDPOINT", ""],
            ["TEMPERATURE", 0.0],
            ["RANDOM_SEED", 42],
            ["PROFILE_FIELDS", "ALL"],
            ["BUILD_PROFILE_QA", False],
            ["BUILD_PROFILE_BACKSTORIES", False],
            ["ASSIGN_MANUALLY", ""],
            ["NUM_AGENTS_PER_SESSION", 4],
            ["CONTEXT_OVERFLOW_POLICY", "terminate"],
            ["TASK_SEQUENCE", "task1"],
        ],
    },
    "C": {
        "columns": ["task", "name", "value", "type"],
        "rows": [
            ["task1", "MAX_NUM_ROUNDS", 3, "integer"],
            ["task1", "PLAYERS_PER_GROUP", 2, "integer"],
            ["task1", "ENDOWMENT", 10, "integer"],
            ["global", "TREATMENT_LABELS", "control,treatment_A", "string"],
        ],
    },
    "Fields": {
        "columns": [
            "task",
            "class",
            "name",
            "type",
            "response_options",
            "response_options_intro",
            "randomise_options_order",
            "validate",
            "generate_speculation_score",
            "format_response",
        ],
        "rows": [
            [
                "task1",
                "Player",
                "decision",
                "integer",
                "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
                "Please choose one of the following:",
                False,
                True,
                False,
                True,
            ],
        ],
    },
    "Facilitator": {
        "columns": ["name", "definition", "args"],
        "rows": [
            ["creating_session", "Initialize the session state.", "{}"],
            ["assign_treatment", "Randomly assign treatment conditions.", "{}"],
            ["assign_groups", "Form groups for this round.", "{}"],
        ],
    },
    "Prompts": {
        "columns": [
            "task",
            "prompt_sequence",
            "type",
            "is_displayed",
            "is_adapted",
            "human_text",
            "llm_text",
            "rag_vector_store_id",
            "field_class",
            "field_name",
        ],
        "rows": [
            [
                "task1",
                0,
                "CONTEXT",
                None,
                False,
                "Introduction context.",
                "You are participating in a decision-making experiment. "
                "Your endowment is {{ C.task1.ENDOWMENT }} tokens.",
                None,
                None,
                None,
            ],
            [
                "task1",
                1,
                "PUBLIC_QUESTION",
                None,
                False,
                "How much will you contribute?",
                "How many tokens (0-{{ C.task1.ENDOWMENT }}) do you contribute to the group fund?",
                None,
                "Player",
                "decision",
            ],
        ],
    },
    "Profiles": {
        "columns": None,  # Special: row-0 = short names, row-1 = full names, rows 2+ = data
        "header_row0": ["ID", "age", "gender"],
        "header_row1": ["ID", "What is your age?", "What is your gender?"],
        "rows": [
            [1, 25, "Male"],
            [2, 32, "Female"],
            [3, 28, "Male"],
            [4, 41, "Female"],
        ],
    },
    "Manual_": {
        "columns": ["ID", "task", "round_number", "class", "name", "value"],
        "rows": [
            # Examples (uncomment and set ASSIGN_MANUALLY in Settings):
            # Manual treatment: [1, "task1", 1, "Agent", "treatment", "treatment_A"],
            # Manual group:     [1, "task1", 1, "Group", "id_in_subsession", "G1"],
        ],
    },
}


def generate_template(
    output_dir: str = ".",
    project_name: str = "my_experiment",
    fmt: Literal["xlsx", "csv"] = "xlsx",
) -> list[str]:
    """Create template files in a project subdirectory.

    Generates a blank experiment template with seven worksheets
    (Settings, C, Fields, Facilitator, Prompts, Profiles, Manual\_)
    containing example rows and bold headers.

    Args:
        output_dir: Parent directory where the project folder is created.
        project_name: Name of the project subfolder and template file prefix.
        fmt: Output format — ``"xlsx"`` for a single Excel workbook or
            ``"csv"`` for individual CSV files per sheet.

    Returns:
        A list of file paths that were created.
    """
    out = Path(output_dir) / project_name
    out.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    if fmt == "csv":
        created.extend(_write_csvs(out))
    else:
        created.append(_write_xlsx(out, project_name))

    return created


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_df(sheet_name: str) -> pd.DataFrame:
    """Build a DataFrame for a single sheet definition.

    Args:
        sheet_name: Key into ``_SHEETS`` identifying the sheet to build.

    Returns:
        A DataFrame with the columns and example rows defined in ``_SHEETS``.
    """
    spec = _SHEETS[sheet_name]
    if sheet_name == "Profiles":
        rows = [spec["header_row0"], spec["header_row1"]] + spec["rows"]
        return pd.DataFrame(rows)

    cols = spec["columns"]
    rows = spec["rows"]
    if rows:
        return pd.DataFrame(rows, columns=cols)
    return pd.DataFrame(columns=cols)


def _write_xlsx(out: Path, project_name: str) -> str:
    """Write all sheets to a single Excel workbook with bold headers.

    Args:
        out: Project directory to write the file into.
        project_name: Used as the filename prefix (``<project_name>_template.xlsx``).

    Returns:
        The absolute path to the created Excel file.
    """
    file_path = str(out / f"{project_name}_template.xlsx")
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        for sheet_name in _SHEETS:
            df = _make_df(sheet_name)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Bold the header row
            ws = writer.sheets[sheet_name]
            try:
                from openpyxl.styles import Font

                for cell in ws[1]:
                    cell.font = Font(bold=True)
            except ImportError:
                pass

    return file_path


def _write_csvs(out: Path) -> list[str]:
    """Write each sheet as an individual CSV file.

    Args:
        out: Project directory to write the CSV files into.

    Returns:
        A list of file paths for the created CSV files.
    """
    created = []
    for sheet_name in _SHEETS:
        df = _make_df(sheet_name)
        safe_name = sheet_name.replace(" ", "_").replace("/", "_")
        file_path = str(out / f"{safe_name}.csv")
        df.to_csv(file_path, index=False)
        created.append(file_path)
    return created
