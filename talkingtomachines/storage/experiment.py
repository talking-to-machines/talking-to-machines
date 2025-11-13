import os, json
import pandas as pd
import numpy as np
from typing import Any
from datetime import date, datetime
from talkingtomachines.management.experiment import Role, Treatment, Constant


def _json_serializer(obj):
    """
    Serialize various object types into JSON-compatible formats.

    This function is designed to handle specific object types and convert them
    into formats that can be serialized into JSON. The supported object types
    and their conversions are as follows:

    - `datetime` and `date`: Converted to ISO 8601 string format using `isoformat()`.
    - `numpy.ndarray`: Converted to a Python list using `tolist()`.
    - `numpy.generic`: Converted to a native Python scalar using `item()`.
    - `Role`, `Treatment`, `Constant`: Converted to a dictionary representation using `to_dict()`.
    - Other types: Converted to their string representation using `str()`.

    Args:
        obj: The object to be serialized.

    Returns:
        A JSON-compatible representation of the input object.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (Role, Treatment, Constant)):
        return obj.to_dict()

    return str(obj)


def save_session(session: dict[str, Any], save_results_as_csv: bool = False) -> None:
    """Save the outputs of a session to a local JSON file in the experiment_results folder at the root directory.

    Args:
        session (dict[str, Any]): The session to be saved.
        save_results_as_csv (bool, optional): Indicates whether the results of the session will be saved as CSV format.
                Defaults to False

    Returns:
        None
    """
    os.makedirs("experiment_results", exist_ok=True)
    json_file_path = f"experiment_results/{session['session_id']}.json"
    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(session, file, default=_json_serializer, ensure_ascii=False, indent=2)

    if save_results_as_csv:
        save_session_as_csv(json_file_path)


def parse_json_field(json_field: str):
    """Parses a JSON string field, removing optional Markdown-style JSON code block delimiters
    (e.g., ```json ... ```) if present, and attempts to decode the JSON string into a Python object.

    Args:
        json_field (str): The JSON string to parse. It may optionally include Markdown-style
                          code block delimiters.

    Returns:
        dict or list or str: The parsed JSON object (e.g., a dictionary or list) if the input
                             is valid JSON. If the input is not valid JSON, the original string
                             is returned.
    """
    try:
        if json_field.startswith("```json"):
            json_field = json_field[len("```json") :].strip()
        if json_field.endswith("```"):
            json_field = json_field[:-3].strip()

        return json.loads(json_field)

    except json.JSONDecodeError:
        return json_field


def save_session_as_csv(file_name: str) -> None:
    """Reads a JSON file containing the outputs of a session, processes the data to extract relevant information,
    and saves the result as a CSV file.

    Args:
        file_name (str): The path to the JSON file containing the session data.
    """
    with open(file_name, "r", encoding="utf-8", errors="ignore") as file:
        json_output = json.load(file)

    output_dict = {}
    for _, group_info in json_output["groups"].items():

        for role, subject in group_info["subjects"].items():
            if role == "facilitator":
                subject_id = f"{role}_{group_info['session_id']}"

            else:
                subject_id = subject["profile_info"]["ID"]

            output_dict[subject_id] = {
                "session_id": subject["session_id"],
                "group_id": subject["group_id"],
                "model_info": subject["model_info"],
                "temperature": subject["temperature"],
                "role": role,
                "treatment_label": group_info["treatment_label"],
                "experiment_context": subject["experiment_context"],
                "system_message": subject["system_message"],
                "build_profile_qna": subject["build_profile_qna"],
                "build_profile_backstories": subject["build_profile_backstories"],
                "constants": group_info["constants"],
            }

        for message in group_info["message_history"]:
            role = list(message.keys())[0]
            if role == "facilitator":
                subject_id = f"facilitator_{group_info['session_id']}"
            else:
                subject_id = message["subject_id"]

            round_num = message.get("round_num", "")

            if (
                message.get("round_id", "") == ""
                and message.get("response_name", "") == ""
            ):
                continue

            elif message.get("response_name", "") != "":
                parsed_response = parse_json_field(json_field=message[role])
                output_dict[subject_id][
                    f"{message['response_name']}.round{round_num}"
                ] = parsed_response

                if isinstance(parsed_response, dict):
                    if parsed_response.get("response", "") != "":
                        output_dict[subject_id][
                            f"{message['response_name']}.response.round{round_num}"
                        ] = parsed_response.get("response")
                    if parsed_response.get("reasoning", "") != "":
                        output_dict[subject_id][
                            f"{message['response_name']}.reasoning.round{round_num}"
                        ] = parsed_response.get("reasoning")
                    if parsed_response.get("speculation_score", "") != "":
                        output_dict[subject_id][
                            f"{message['response_name']}.speculation_score.round{round_num}"
                        ] = parsed_response.get("speculation_score")

            else:
                parsed_response = parse_json_field(json_field=message[role])
                output_dict[subject_id][
                    f"{message['round_id']}.round{round_num}"
                ] = parsed_response

                if isinstance(parsed_response, dict):
                    if parsed_response.get("response", "") != "":
                        output_dict[subject_id][
                            f"{message['round_id']}.response.round{round_num}"
                        ] = parsed_response.get("response")
                    if parsed_response.get("reasoning", "") != "":
                        output_dict[subject_id][
                            f"{message['round_id']}.reasoning.round{round_num}"
                        ] = parsed_response.get("reasoning")
                    if parsed_response.get("speculation_score", "") != "":
                        output_dict[subject_id][
                            f"{message['round_id']}.speculation_score.round{round_num}"
                        ] = parsed_response.get("speculation_score")

    output_df = pd.DataFrame.from_dict(output_dict, orient="index")
    output_df.reset_index(drop=False, inplace=True)
    output_df.rename(columns={"index": "ID"}, inplace=True)
    output_df.sort_values(by="group_id", ascending=True, inplace=True)
    output_df.to_csv(file_name[:-5] + ".csv", index=False)
