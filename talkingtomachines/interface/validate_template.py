import os
import pandas as pd


def validate_prompt_template_path(file_path: str) -> None:
    """Validates the provided file path to ensure it points to an existing Excel prompt template file.

    Args:
        file_path (str): The path to the prompt template file that needs to be validated.

    Raises:
        ValueError: If the file does not exist at the provided path or is not an Excel file.
    """
    # Validate the provided directory path
    if not os.path.isfile(file_path):
        raise ValueError(
            f"The prompt template cannot be found in the path you provided: {file_path}"
        )

    # Check if the file has a valid Excel extension
    valid_extensions = [".xlsx", ".xls"]
    if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
        raise ValueError(
            f"The file provided is not a valid Excel file. Expected extensions: {', '.join(valid_extensions)}"
        )


def validate_prompt_template_sheets(
    excel_file: pd.ExcelFile, required_sheet_list: list
) -> None:
    """Validates that all required sheets are present in the given prompt template file.

    Args:
        excel_file (pd.ExcelFile): The Excel file to validate.
        required_sheet_list (list[str]): A list of sheet names that are required to be present in the Excel file.

    Raises:
        ValueError: If any of the required sheets are missing from the Excel file.
    """
    # Validate that all required sheets are present in the Excel file
    missing_sheets = [
        sheet for sheet in required_sheet_list if sheet not in excel_file.sheet_names
    ]
    if missing_sheets:
        raise ValueError(
            f"The following sheets are missing from the prompt template: {', '.join(missing_sheets)}"
        )


def validate_settings_sheet(settings: pd.DataFrame) -> None:
    """Validates the experimental settings worksheet to ensure it has the correct structure and required settings.

    Args:
        settings (pd.DataFrame): A DataFrame containing the experimental settings.
                                It should have columns "settings_label" and "value".

    Raises:
        AssertionError: If the columns of the DataFrame do not match the expected columns.
        AssertionError: If any of the required settings are missing from the "settings_label" column.
    """
    # Validate the column headers
    expected_columns = ["settings_label", "value"]
    assert (
        list(settings.columns) == expected_columns
    ), f"Invalid columns in settings sheet. Expected {expected_columns}, got {list(settings.columns)}"

    # Validate the experimental settings field
    valid_settings = [
        "session_id",
        "model_info",
        "hf_inference_endpoint",
        "temperature",
        "num_subjects_per_group",
        "num_groups",
        "max_num_rounds",
        "treatment_assignment_strategy",
        "treatment_column",
        "group_assignment_strategy",
        "group_column",
        "role_assignment_strategy",
        "role_column",
        "random_seed",
        "build_profile_qna",
        "build_profile_backstories",
    ]
    for setting in valid_settings:
        assert (
            setting in settings["settings_label"].tolist()
        ), f"{setting} not found in the settings worksheet."


def validate_treatment_sheet(treatments: pd.DataFrame) -> None:
    """Validates the structure of the treatments worksheet.

    This function checks if the treatments DataFrame has the expected column headers.
    It raises an assertion error if the columns do not match the expected columns.

    Args:
        treatments (pd.DataFrame): The DataFrame containing treatment data to be validated.

    Raises:
        AssertionError: If the columns of the treatments DataFrame do not match the expected columns.
    """
    # Validate the column headers
    expected_columns = ["treatment_label", "value"]
    assert (
        list(treatments.columns) == expected_columns
    ), f"Invalid columns in treatments sheet. Expected {expected_columns}, got {list(treatments.columns)}"


def validate_role_sheet(roles: pd.DataFrame) -> None:
    """Validates the structure of the role worksheet.

    This function checks if the provided DataFrame has the expected column headers.
    It raises an assertion error if the columns do not match the expected structure.

    Args:
        roles (pd.DataFrame): The DataFrame containing roles to be validated.

    Raises:
        AssertionError: If the columns of the DataFrame do not match the expected columns.
    """
    # Validate the column headers
    expected_columns = ["role_label", "value"]
    assert (
        list(roles.columns) == expected_columns
    ), f"Invalid columns in role sheet. Expected {expected_columns}, got {list(roles.columns)}"


def validate_prompt_sheet(prompts: pd.DataFrame) -> None:
    """Validates the structure of a prompts_template worksheet containing prompt data.

    This function checks if the DataFrame has the expected column headers.
    If the columns do not match the expected headers, an assertion error is raised.

    Args:
        prompts (pd.DataFrame): The DataFrame containing the prompt data to be validated.

    Raises:
        AssertionError: If the columns of the DataFrame do not match the expected columns.
    """
    # Validate the column headers
    expected_columns = [
        "round_id",
        "type",
        "round_order",
        "is_adapted",
        "human_text",
        "llm_text",
        "response_name",
        "response_type",
        "response_options",
        "randomize_response_order",
        "validate_response",
        "generate_speculation_score",
        "format_response",
    ]
    assert (
        list(prompts.columns) == expected_columns
    ), f"Invalid columns in prompts_template sheet. Expected {expected_columns}, got {list(prompts.columns)}"


def validate_constant_sheet(constants: pd.DataFrame) -> None:
    """Validates the structure of the constants worksheet.

    This function checks that the DataFrame has the expected column headers: "constant_label" and "value".
    If the columns do not match the expected headers, an assertion error is raised.

    Args:
        constants (pd.DataFrame): The DataFrame to validate.

    Raises:
        AssertionError: If the DataFrame does not have the expected columns.
    """
    # Validate the column headers
    expected_columns = ["constant_label", "value"]
    assert (
        list(constants.columns) == expected_columns
    ), f"Invalid columns in constants sheet. Expected {expected_columns}, got {list(constants.columns)}"
