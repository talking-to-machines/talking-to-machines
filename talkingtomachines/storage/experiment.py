import os
from typing import Any
import json


def save_experiment(experiment: dict[int, Any]) -> None:
    """Save an experiment to a local JSON file in the storage/experiment folder at the root directory.

    Args:
        experiment (dict[int, Any]): The experiment to be saved.

    Returns:
        None
    """
    os.makedirs("storage/experiment", exist_ok=True)
    with open(f"storage/experiment/{experiment['experiment_id']}.json", "w") as file:
        json.dump(experiment, file)
