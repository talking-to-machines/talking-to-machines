"""Template generator for blank experiment workbooks.

Creates a blank ``prompt_template.xlsx`` workbook with seven worksheets
(Settings, C, Fields, Facilitator, Prompts, Profiles, ``Manual_``) containing
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
            ["EXPERIMENT_ID", "simple_pgg"],
            ["MODEL_NAME", "gpt-4.1-mini"],
            ["HF_INFERENCE_ENDPOINT", ""],
            ["TEMPERATURE", 0.0],
            ["RANDOM_SEED", 42],
            ["PROFILE_FIELDS", "ALL"],
            ["BUILD_PROFILE_QA", True],
            ["BUILD_PROFILE_BACKSTORIES", False],
            ["NUM_AGENTS_PER_SESSION", 4],
            ["CONTEXT_OVERFLOW_POLICY", "terminate"],
            ["MODULE_SEQUENCE", "task1"],
        ],
    },
    "C": {
        "columns": ["module", "name", "value", "type"],
        "rows": [
            ["task1", "MAX_NUM_ROUNDS", 3, "integer"],
            ["task1", "PLAYERS_PER_GROUP", 2, "integer"],
            ["task1", "MAX_ENDOWMENT", 10, "integer"],
        ],
    },
    "Fields": {
        "columns": [
            "module",
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
                True,
                True,
            ],
            [
                "task1",
                "Group",
                "total_contribution",
                "integer",
                "",
                "",
                False,
                False,
                False,
                True,
            ],
            [
                "task1",
                "Player",
                "public_account_earnings",
                "integer",
                "",
                "",
                False,
                False,
                False,
                True,
            ],
            [
                "task1",
                "Player",
                "private_account_remaining",
                "integer",
                "",
                "",
                False,
                False,
                False,
                True,
            ],
            [
                "task1",
                "Player",
                "total_earnings",
                "integer",
                "",
                "",
                False,
                False,
                False,
                True,
            ],
        ],
    },
    "Facilitator": {
        "columns": ["name", "definition", "kwargs"],
        "rows": [
            ["assign_groups", "", None],
            [
                "calculate_total_contribution",
                """Calculate the total number of tokens contributed to the public account in this round using the following formula:
Player 1's contribution to public account in this round + Player 2's contribution to public account in this round
Respond with only the resulting nunber without providing any explanations.""",
                None,
            ],
            [
                "calculate_public_account_earnings",
                """Calculate Player ID {{ player.agent_id }}'s earnings from the public account this round using the following formula:
({{ group.total_contribution }} × 2) ÷ {{ group.num_players }}
Respond with only the resulting number without providing any explanations.""",
                None,
            ],
            [
                "calculate_private_account_remaining",
                """Calculate Player ID {{ player.agent_id }}'s remaining private tokens this round using the following formula:
{{ C.task1.MAX_ENDOWMENT }} − {{ player.decision }}
Respond with only the resulting number without any explanations.""",
                None,
            ],
            [
                "calculate_total_earnings",
                """Calculate Player ID {{ player.agent_id }}'s total earnings this round using the following formula:
{{ player.public_account_earnings }} + {{ player.private_account_remaining }}
Respond with only the resulting number without any explanations.""",
                None,
            ],
        ],
    },
    "Prompts": {
        "columns": [
            "module",
            "prompt_sequence",
            "type",
            "is_displayed",
            "is_adapted",
            "human_text",
            "llm_text",
            "kwargs",
            "field_class",
            "field_name",
        ],
        "rows": [
            [
                "task1",
                0,
                "CONTEXT",
                "round_number == 1",
                True,
                "Introduction context.",
                "You are participating in a decision-making experiment. Your endowment is {{ C.task1.MAX_ENDOWMENT }} tokens.",
                None,
                None,
                None,
            ],
            [
                "task1",
                1,
                "PRIVATE_QUESTION",
                None,
                True,
                "How much will you contribute?",
                "How many tokens would you contribute to the group fund? Tokens in the public account will be multiplied by a factor of 2 and distributed equally among all players. Tokens kept in your private account will not be shared and remain yours. Choose a value between 0 and {{ C.task1.MAX_ENDOWMENT }}.",
                None,
                "Player",
                "decision",
            ],
            [
                "task1",
                2,
                "FACILITATOR",
                None,
                False,
                "",
                "calculate_total_contribution",
                None,
                "Group",
                "total_contribution",
            ],
            [
                "task1",
                3,
                "FACILITATOR",
                None,
                False,
                "",
                "calculate_public_account_earnings",
                None,
                "Player",
                "public_account_earnings",
            ],
            [
                "task1",
                4,
                "FACILITATOR",
                None,
                False,
                "",
                "calculate_private_account_remaining",
                None,
                "Player",
                "private_account_remaining",
            ],
            [
                "task1",
                5,
                "FACILITATOR",
                None,
                False,
                "",
                "calculate_total_earnings",
                None,
                "Player",
                "total_earnings",
            ],
            [
                "task1",
                6,
                "CONTEXT",
                None,
                False,
                "",
                "The total contribution for this round is {{ group.total_contribution }} tokens. Your total earnings for this round is {{ player.total_earnings }} tokens.",
                None,
                "",
                "",
            ],
        ],
    },
    "Profiles": {
        "columns": [
            "ID",
            "age",
            "gender",
        ],  # Special: row-0 = short names, row-1 = full names, rows 2+ = data
        "rows": [
            ["ID", "What is your age?", "What is your gender?"],
            [1, 25, "Male"],
            [2, 32, "Female"],
            [3, 28, "Male"],
            [4, 41, "Female"],
        ],
    },
    "Manual_": {
        "columns": ["ID", "module", "round_number", "class", "name", "value"],
        "rows": [
            # Treatment assignments (just another variable)
            [1, "task1", 1, "Player", "treatment", "control"],
            [1, "task1", 2, "Player", "treatment", "control"],
            [1, "task1", 3, "Player", "treatment", "control"],
            [2, "task1", 1, "Player", "treatment", "treatment_A"],
            [2, "task1", 2, "Player", "treatment", "treatment_A"],
            [2, "task1", 3, "Player", "treatment", "treatment_A"],
            [3, "task1", 1, "Player", "treatment", "control"],
            [3, "task1", 2, "Player", "treatment", "control"],
            [3, "task1", 3, "Player", "treatment", "control"],
            [4, "task1", 1, "Player", "treatment", "treatment_A"],
            [4, "task1", 2, "Player", "treatment", "treatment_A"],
            [4, "task1", 3, "Player", "treatment", "treatment_A"],
            # Manual group assignments
            [1, "task1", None, "Group", "id_in_subsession", "group1"],
            [2, "task1", None, "Group", "id_in_subsession", "group2"],
            [3, "task1", None, "Group", "id_in_subsession", "group1"],
            [4, "task1", None, "Group", "id_in_subsession", "group2"],
        ],
    },
}


def generate_template(
    output_dir: str = ".",
    project_name: str = "simple_pgg",
    fmt: Literal["xlsx", "csv"] = "xlsx",
) -> list[str]:
    """Create template files in a project subdirectory.

    Generates a blank experiment template with seven worksheets
    (Settings, C, Fields, Facilitator, Prompts, Profiles, ``Manual_``)
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
        created.extend(_write_csvs(out, project_name=project_name))
    else:
        created.append(_write_xlsx(out, project_name))

    return created


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_df(sheet_name: str, project_name: str = "simple_pgg") -> pd.DataFrame:
    """Build a DataFrame for a single sheet definition.

    Args:
        sheet_name: Key into ``_SHEETS`` identifying the sheet to build.
        project_name: Project name used to set the default ``EXPERIMENT_ID``
            in the Settings sheet.

    Returns:
        A DataFrame with the columns and example rows defined in ``_SHEETS``.
    """
    spec = _SHEETS[sheet_name]
    cols = spec["columns"]
    rows = [list(row) for row in spec["rows"]]
    if sheet_name == "Settings":
        for row in rows:
            if row[0] == "EXPERIMENT_ID":
                row[1] = project_name
                break
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
            df = _make_df(sheet_name, project_name=project_name)
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


def _write_csvs(out: Path, project_name: str = "simple_pgg") -> list[str]:
    """Write each sheet as an individual CSV file.

    Args:
        out: Project directory to write the CSV files into.
        project_name: Project name used to set the default ``EXPERIMENT_ID``.

    Returns:
        A list of file paths for the created CSV files.
    """
    created = []
    for sheet_name in _SHEETS:
        df = _make_df(sheet_name, project_name=project_name)
        safe_name = sheet_name.replace(" ", "_").replace("/", "_")
        file_path = str(out / f"{safe_name}.csv")
        df.to_csv(file_path, index=False)
        created.append(file_path)
    return created
