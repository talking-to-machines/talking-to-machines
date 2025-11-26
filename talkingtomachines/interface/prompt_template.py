import argparse, warnings, ast, concurrent.futures, json
import pandas as pd
from tqdm import tqdm
from importlib.metadata import version, PackageNotFoundError
from talkingtomachines.interface.validate_template import (
    validate_prompt_template_path,
    validate_settings_sheet,
    validate_treatment_sheet,
    validate_role_sheet,
    validate_prompt_sheet,
    validate_constant_sheet,
    validate_prompt_template_sheets,
)
from talkingtomachines.management.initialize_experiment import initialize_experiment
from talkingtomachines.management.experiment import (
    AItoAIInterviewExperiment,
    Treatment,
    Role,
)


PROMPT_TEMPLATE_SHEETS = [
    "settings",
    "treatment",
    "role",
    "prompt",
    "profile",
    "constant",
]
SPECIAL_ROLES = [
    "facilitator",
]
SUPPORTED_PROMPT_TYPES = [
    "context",
    "discussion",
    "public_question",
    "repeat_public_question",
    "private_question",
    "repeat_private_question",
]
BOOLEAN_KEYS = [
    "build_profile_qna",
    "build_profile_backstories",
    "randomize_response_order",
    "validate_response",
    "generate_speculation_score",
    "format_response",
]


def extract_settings(file_path: str, sheet_name: str) -> dict:
    """Extracts the experimental settings from a specified worksheet in the prompt template.

    Args:
        file_path (str): The file path to the prompt template.
        sheet_name (str): The name of the worksheet containing the experimental settings.

    Returns:
        dict: A dictionary representation of the experimental settings, where the keys are the
              values from the first column and the values are the corresponding values from the
              second column.

    Raises:
        ValueError: If the mandatory fields are not present in the settings worksheet.
    """
    # Read the settings worksheet into a DataFrame
    settings_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Validate the presence of mandatory fields in the settings worksheet
    validate_settings_sheet(settings_df)

    # Convert the experimental setting worksheet to a dictionary
    settings_dict = settings_df.set_index(settings_df.columns[0]).to_dict()[
        settings_df.columns[1]
    ]

    # Convert string booleans to actual booleans
    for key in BOOLEAN_KEYS:
        if key in settings_dict:
            val = settings_dict[key]
            if isinstance(val, str):
                v = val.strip().lower()
                if v == "true":
                    settings_dict[key] = True
                elif v == "false":
                    settings_dict[key] = False
            elif isinstance(val, (int, float)):
                # treat 0 as False, non-zero as True
                settings_dict[key] = bool(val)

    return settings_dict


def extract_treatments(file_path: str, sheet_name: str) -> dict:
    """
    Extracts treatment information from an Excel worksheet and returns it as a dictionary.

    Args:
        file_path (str): The file path to the Excel workbook.
        sheet_name (str): The name of the worksheet containing treatment data.

    Returns:
        dict: A dictionary with a single key "treatments", where the value is another dictionary
              mapping treatment labels to `Treatment` objects. Each `Treatment` object is created
              based on the data in the worksheet.

    The function performs the following steps:
        1. Reads the specified worksheet into a pandas DataFrame.
        2. Validates the presence of mandatory fields in the worksheet.
        3. Iterates through each row of the DataFrame to extract treatment labels and their
           corresponding attributes.
        4. Parses the "value" field as a JSON-style string if possible, or treats it as plain text.
        5. Creates a `Treatment` object for each treatment label and adds it to the resulting dictionary.

    Raises:
        ValueError: If the treatment worksheet fails validation.
        json.JSONDecodeError: If the "value" field contains invalid JSON and cannot be parsed.
    """
    # Read the treatment worksheet into a DataFrame
    treatments_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Validate the presence of mandatory fields in the treatment worksheet
    validate_treatment_sheet(treatments_df)

    # Convert the treatment worksheet to a dictionary
    treatments = {}
    for _, row in treatments_df.iterrows():
        label = str(row.get("treatment_label", "")).strip()
        raw_value = row.get("value", "")

        # Skip empty labels
        if not label:
            continue

        attr_dict = {}
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()

            # Try to parse JSON-style string
            try:
                attr_dict = json.loads(raw_value)
                if not isinstance(attr_dict, dict):
                    attr_dict = {"description": str(attr_dict)}
            except json.JSONDecodeError:
                warnings.warn(
                    f"Error parsing treatment label '{label}' as a Python dictionary. The treatment value will be treated as a plain string and assigned to the description field."
                )
                attr_dict = {"description": raw_value}

        elif pd.isna(raw_value):
            attr_dict = {"description": ""}
        else:
            # For non-string (e.g., numeric or object), just cast to str
            attr_dict = {"description": str(raw_value)}

        # Create Treatment object for each treatment arm
        treatments[label] = Treatment(**attr_dict)

    return {"treatments": treatments}


def extract_roles(file_path: str, sheet_name: str) -> dict:
    """
    Extract roles from an Excel worksheet and convert them into a dictionary.

    Args:
        file_path (str): The path to the Excel file containing the role worksheet.
        sheet_name (str): The name of the worksheet to read roles from.

    Returns:
        dict: A dictionary with a single key "roles", where the value is another
              dictionary mapping role labels to Role objects. Each Role object
              contains attributes parsed from the worksheet.

    Raises:
        ValueError: If the role worksheet does not contain mandatory fields or
                    if the data format is invalid.
    """
    # Read the role worksheet into a DataFrame
    roles_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Validate the presence of mandatory fields in the role worksheet
    validate_role_sheet(roles_df)

    # Convert the role worksheet to a dictionary
    roles = {}
    for _, row in roles_df.iterrows():
        label = str(row.get("role_label", "")).strip()
        raw_value = row.get("value", "")

        # Skip empty labels
        if not label:
            continue

        attr_dict = {}
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()

            # Try to parse JSON-style string
            try:
                attr_dict = json.loads(raw_value)
                if not isinstance(attr_dict, dict):
                    attr_dict = {"description": str(attr_dict)}
            except json.JSONDecodeError:
                warnings.warn(
                    f"Error parsing role label '{label}' as a Python dictionary. The role value will be treated as a plain string and assigned to the description field."
                )
                # Treat as plain description text
                attr_dict = {"description": raw_value}

        elif pd.isna(raw_value):
            attr_dict = {"description": ""}
        else:
            # For non-string (e.g., numeric or object), just cast to str
            attr_dict = {"description": str(raw_value)}

        # Create Role object for each role defined
        roles[label] = Role(**attr_dict)

    return {"roles": roles}


def parse_prompt_text_field(
    text_field: str,
    role_list: list,
    round_id: str,
    prompt_type: str,
    is_response_options: bool,
) -> dict:
    """Parses the prompt text for the llm_text and response_options fields, converting them into Python objects and assigning to roles.

    If text_field is a plain string (i.e. not a dictionary literal), this function
    creates a dictionary where the keys are the roles in role_list (excluding those in SPECIAL_ROLES)
    and the value is the plain string.

    Otherwise, if text_field is a string representation of a dictionary, it is parsed as a Python dictionary.

    Args:
        text_field (str): The text field to parse.
        role_list (list): A list of role names.
        round_id (str): The round ID associated with that round.
        prompt_type (str): The type of the prompt.
        is_response_options (bool): Boolean indicator on whether the function is parsing the text for the llm_text field or response_options field.

    Returns:
        dict: A dictionary mapping roles to the prompt text field.
    """
    # Create list of user-defined roles, excluding the special roles
    user_defined_roles = [role for role in role_list if role not in SPECIAL_ROLES]
    if isinstance(text_field, str):
        text_field = text_field.strip()
    else:
        try:
            text_field = str(text_field).strip()
        except Exception as e:
            text_field = ""

    if prompt_type in [
        "public_question",
        "private_question",
        "repeat_public_question",
        "repeat_private_question",
    ]:
        # If the text starts with "{" and ends with "}", assume it's a dictionary literal.
        if text_field.startswith("{") and text_field.endswith("}"):
            try:
                prompt_dict = ast.literal_eval(text_field)
            except (ValueError, SyntaxError) as e:
                warnings.warn(
                    f"Error parsing text field ({text_field}) in Round ID {round_id} as dictionary: {e}. The prompt will be treated as a plain string and assigned to all user-defined roles."
                )
                prompt_dict = {role: text_field for role in user_defined_roles}

            return prompt_dict

        # Otherwise, it's a plain string: build a dictionary mapping each user-defined role to that string.
        else:
            if is_response_options:
                try:
                    return {
                        role: ast.literal_eval(text_field)
                        for role in user_defined_roles
                    }
                except:
                    return {role: text_field for role in user_defined_roles}
            else:
                return {role: text_field for role in user_defined_roles}

    elif prompt_type in ["context", "discussion"]:
        if is_response_options:
            try:
                return {"facilitator": ast.literal_eval(text_field)}
            except:
                return {"facilitator": text_field}
        else:
            return {"facilitator": text_field}

    else:
        raise ValueError(
            f"Invalid prompt type: {prompt_type} in Round ID {round_id}: Supported prompt types include: {SUPPORTED_PROMPT_TYPES}"
        )


def parse_range_response_options(response_options: dict) -> dict:
    """Parses a dictionary of response options, converting any tuple values into a range object.

    Args:
        response_options (dict): A dictionary where keys represent roles and values are either
                                 tuples (to be converted into range objects) or other types
                                 (which are left unchanged).

    Returns:
        dict: A dictionary with the same keys as the input, where tuple values are replaced
              with range objects, and other values remain unchanged.
    """
    parsed_response_options = {
        role: (
            range(*response_option)
            if isinstance(response_option, tuple)
            else response_option
        )
        for role, response_option in response_options.items()
    }
    return parsed_response_options


def extract_prompts(file_path: str, sheet_name: str, role_list: list) -> dict:
    """Extracts prompts from the prompt worksheet in the prompt template and returns them as a dictionary.

    This function reads the prompt worksheet from the prompt template file, validates the presence
    of mandatory fields, processes the prompts, and returns them in a structured format.

    Args:
        file_path (str): The file path to the prompt template file.
        sheet_name (str): The name of the worksheet to read from the prompt template file.
        role_list (list): A list of roles to be used in the experiment.

    Returns:
        dict: A dictionary containing a list of prompts. Each prompt is represented as a dictionary
              with relevant fields and their corresponding values.

    Raises:
        ValueError: If mandatory fields are not present in the prompts_template worksheet.
    """
    # Read the prompt template worksheet into a DataFrame
    prompts_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Validate the presence of mandatory fields in the prompt template worksheet
    validate_prompt_sheet(prompts_df)

    # Convert the prompts to a list of dictionaries
    prompts_list = prompts_df[
        [
            "round_id",
            "type",
            "round_order",
            "llm_text",
            "response_name",
            "response_type",
            "response_options",
            "randomize_response_order",
            "validate_response",
            "generate_speculation_score",
            "format_response",
        ]
    ].to_dict(orient="records")

    # Parse llm_text column from string format to appropriate Python format
    for prompt_dict in prompts_list:
        prompt_dict["llm_text"] = parse_prompt_text_field(
            text_field=prompt_dict["llm_text"],
            role_list=role_list,
            round_id=prompt_dict["round_id"],
            prompt_type=prompt_dict["type"],
            is_response_options=False,
        )
        prompt_dict["response_options"] = parse_prompt_text_field(
            text_field=prompt_dict["response_options"],
            role_list=role_list,
            round_id=prompt_dict["round_id"],
            prompt_type=prompt_dict["type"],
            is_response_options=True,
        )
        prompt_dict["response_options"] = parse_range_response_options(
            prompt_dict["response_options"]
        )

        for key, val in prompt_dict.items():
            if key in BOOLEAN_KEYS:
                if isinstance(val, str):
                    v = val.strip().lower()
                    if v == "true":
                        prompt_dict[key] = True
                    elif v == "false":
                        prompt_dict[key] = False
                elif isinstance(val, (int, float)):
                    # treat 0 as False, non-zero as True
                    prompt_dict[key] = bool(val)

    return {"prompts": prompts_list}


def extract_profiles(file_path: str, sheet_name: str) -> dict:
    """Extracts subject profiles from the profile worksheet in the prompt template.

    Args:
        file_path (str): The file path to the prompt template.
        sheet_name (str): The name of the sheet to read from the prompt template.

    Returns:
        dict: A dictionary containing:
            - "profiles_mapping" (dict): A dictionary mapping of the first row of the sheet.
            - "profiles" (pd.DataFrame): A DataFrame containing the remaining data with new column headers.

    Raises:
        ValueError: If mandatory fields are not present in the profile worksheet.
    """
    # Read the specified sheet into a DataFrame
    profiles_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Extract the first row and convert them to a dictionary
    profiles_mapping = profiles_df.iloc[0].to_dict()

    # Use the first row as the new column headers for the remaining data
    profiles = profiles_df.iloc[1:].reset_index(drop=True)
    profiles.columns = profiles_df.iloc[0]

    return {
        "profiles_mapping": profiles_mapping,
        "profiles": profiles,
    }


def extract_constants(file_path: str, sheet_name: str) -> dict:
    """
    Extracts constants from a specified Excel worksheet and returns them as a dictionary.

    Args:
        file_path (str): The path to the Excel file containing the constants.
        sheet_name (str): The name of the worksheet to extract constants from.

    Returns:
        dict: A dictionary with a single key "constants", where the value is another dictionary
              mapping the first column values to the second column values. If the second column
              contains string representations of lists, they are converted to actual lists.

    Raises:
        ValueError: If the worksheet does not contain the required fields or if the data format
                    is invalid.
    """
    # Read the constant worksheet into a DataFrame
    constants_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Validate the presence of mandatory fields in the constant worksheet
    validate_constant_sheet(constants_df)

    # Convert the roles worksheet to a dictionary
    constants_dict = constants_df.set_index(constants_df.columns[0]).to_dict()[
        constants_df.columns[1]
    ]

    # Convert string representations of lists to actual lists
    for key, value in constants_dict.items():
        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
            constants_dict[key] = [str(item) for item in ast.literal_eval(value)]
        else:
            constants_dict[key] = [value]

    return {"constants": constants_dict}


def print_session_settings(
    session: AItoAIInterviewExperiment, constant_permutation: dict
) -> None:
    """
    Prints the experimental settings and configuration details for an AI-to-AI interview experiment.

    Args:
        session (AItoAIInterviewExperiment): An object containing all relevant session parameters and settings.
        constant_permutation (dict): A dictionary representing constant permutations used in the session.

    Returns:
        None

    The function outputs a formatted summary of the session's configuration, including model information,
    API keys, session and role details, treatment and assignment strategies, random seed, roles,
    and interview prompts.
    """

    def _format_dict_to_str(d: dict, indent: int = 10) -> str:
        if not d:
            return " " * indent + "{}"

        def _format_val(val, ind):
            if isinstance(val, dict):
                sub_lines = []
                for kk, vv in val.items():
                    if isinstance(vv, dict):
                        sub_lines.append(f"{' ' * ind}{kk}:")
                        sub_lines.append(_format_val(vv, ind + 4))
                    else:
                        sub_lines.append(f"{' ' * ind}{kk}: {vv}")
                return "\n".join(sub_lines)
            elif isinstance(val, (list, tuple)):
                return ", ".join(str(x) for x in val)
            else:
                return str(val)

        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"\n{' ' * indent}{k}:")
                lines.append(_format_val(v, indent + 4))
            else:
                lines.append(f"\n{' ' * indent}{k}: { _format_val(v, indent + 4) }")
        return "\n".join(lines) + "\n"

    print(
        """
        Experiment Settings for {session_id}:
        {line_separator}
        Model Info: {model_info}
        HF Inference Endpoint (Only applicable when using Hugging Face Models): {hf_inference_endpoint}
        Temperature: {temperature}
        Number of Subjects per Group (Excluding Special Roles like 'facilitator'): {num_subjects_per_group}
        Number of Groups: {num_groups}
        Maximum Number of Rounds: {max_num_rounds}
        Treatments: {treatments}
        Treatment Assignment Strategy: {treatment_assignment_strategy}
        Treatment Column (Only valid when using manual treatment assignment strategy): {treatment_column}
        Group Assignment Strategy: {group_assignment_strategy}
        Group Column (Only valid when using manual group assignment strategy): {group_column}
        Role Assignment Strategy: {role_assignment_strategy}
        Role Column (Only valid when using manual role assignment strategy): {role_column}
        Random Seed: {random_seed}
        Build Profiles using Q&A Format: {build_profile_qna}
        Build Profiles using Backstories: {build_profile_backstories}
        Constant Permutation: {constant_permutation}

        Roles: {roles}
        Prompts: 
        {prompts}

        Constants: {constants}

        """.format(
            session_id=session.session_id,
            line_separator="=" * (25 + len(session.session_id)),
            model_info=session.model_info,
            hf_inference_endpoint=session.hf_inference_endpoint,
            temperature=session.temperature,
            num_subjects_per_group=session.num_subjects_per_group,
            num_groups=session.num_groups,
            max_num_rounds=session.max_num_rounds,
            treatments=_format_dict_to_str(session.treatments),
            treatment_assignment_strategy=session.treatment_assignment_strategy,
            treatment_column=session.treatment_column,
            group_assignment_strategy=session.group_assignment_strategy,
            group_column=session.group_column,
            role_assignment_strategy=session.role_assignment_strategy,
            role_column=session.role_column,
            random_seed=session.random_seed,
            build_profile_qna=session.build_profile_qna,
            build_profile_backstories=session.build_profile_backstories,
            constant_permutation=constant_permutation,
            roles=_format_dict_to_str(session.roles),
            prompts=session.prompts,
            constants=_format_dict_to_str(session.constants),
        )
    )


def run_session_wrapper(args: tuple) -> None:
    """Wrapper function to execute a session with the provided arguments.

    Args:
        args (tuple): A tuple containing the following elements:
            - session: An object with a `run_session` method to execute the session.
            - test_mode (bool): A flag indicating whether the session should run in test mode.
            - version (str): The version identifier for the session.

    Returns:
        None
    """
    session, test_mode, version = args
    session.run_session(test_mode=test_mode, version=version, save_results_as_csv=True)


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Parse the prompt template provided by the user and initialise the experiment in the Talking to Machines Platform."
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Prints the current package version.",
    )
    parser.add_argument(
        "prompt_template_file_path",
        nargs="?",
        type=str,
        default=None,
        help="Path to the prompt template Excel file.",
    )
    args = parser.parse_args()

    # Handle --version flag and exit
    if args.version:
        try:
            print(f"talkingtomachines {version('talkingtomachines')}")
        except PackageNotFoundError:
            print("talkingtomachines (version unknown)")
        return

    # Require a valid path to the prompt template if not using --version
    if not args.prompt_template_file_path:
        parser.error(
            "A valid file path to the prompt template needs to be provided as an argument."
        )

    prompt_template_file_path = args.prompt_template_file_path

    # Validate the provided directory path to the prompt template
    validate_prompt_template_path(file_path=prompt_template_file_path)

    # Read the prompt template in Excel format
    prompt_template = pd.ExcelFile(prompt_template_file_path)

    # Validate the required sheets in the prompt template
    validate_prompt_template_sheets(
        excel_file=prompt_template, required_sheet_list=PROMPT_TEMPLATE_SHEETS
    )

    # Dictionary to store data from each worksheet
    prompt_template_dict = {}

    # Iterate through each worksheet
    for sheet_name in prompt_template.sheet_names:
        if sheet_name == "settings":
            prompt_template_dict.update(
                extract_settings(
                    file_path=prompt_template_file_path, sheet_name=sheet_name
                )
            )
        elif sheet_name == "treatment":
            prompt_template_dict.update(
                extract_treatments(
                    file_path=prompt_template_file_path, sheet_name=sheet_name
                )
            )
        elif sheet_name == "role":
            prompt_template_dict.update(
                extract_roles(
                    file_path=prompt_template_file_path, sheet_name=sheet_name
                )
            )
        elif sheet_name == "prompt":
            prompt_template_dict.update(
                extract_prompts(
                    file_path=prompt_template_file_path,
                    sheet_name=sheet_name,
                    role_list=list(prompt_template_dict["roles"].keys()),
                )
            )

        elif sheet_name == "profile":
            prompt_template_dict.update(
                extract_profiles(
                    file_path=prompt_template_file_path, sheet_name=sheet_name
                )
            )
        elif sheet_name == "constant":
            prompt_template_dict.update(
                extract_constants(
                    file_path=prompt_template_file_path, sheet_name=sheet_name
                )
            )
        else:
            warnings.warn(
                f"{sheet_name} will be ignored as it is not one of the required prompt template sheets: {', '.join(PROMPT_TEMPLATE_SHEETS)}"
            )

    # Initialize experiment based on prompt template
    session_list, constant_permutations = initialize_experiment(
        prompt_template_dict=prompt_template_dict
    )

    # Print out experimental settings for each session for user verification
    for session, constant_permutation in zip(session_list, constant_permutations):
        print_session_settings(session, constant_permutation)

    # Ask for user confirmation to run the experiment
    user_input = (
        input(
            "Verify the experiment settings above and choose a run mode:\n"
            "  • Type 'test'  → Runs the session in TEST mode (one randomly selected group per treatment)\n"
            "  • Type 'full'  → Runs the FULL session\n"
            "  • Anything else → Terminates the session immediately\n"
            "Your choice: "
        )
        .strip()
        .lower()
    )
    if user_input == "test":
        print("Experiment has started running in TEST mode.")

        session_version = 1
        for session in tqdm(session_list):
            session.run_session(
                test_mode=True,
                version=session_version,
                save_results_as_csv=True,
            )
            session_version += 1
        print("Experiment is completed successfully in TEST mode.")

    elif user_input == "full":
        print("Experiment has started running in FULL mode.")

        # Prepare a list of arguments for each session
        experiment_args = [
            (session, False, idx + 1) for idx, session in enumerate(session_list)
        ]

        # Run session in parallel using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit all sessions
            futures = [
                executor.submit(run_session_wrapper, arg) for arg in experiment_args
            ]

            # Update progress bar as each session completes
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                pass
        print("Experiment is completed successfully in FULL mode.")

    else:
        print("Experiment is terminated by the user.")


if __name__ == "__main__":
    main()
