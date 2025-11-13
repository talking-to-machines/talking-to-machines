import random
import pandas as pd
from typing import List, Any


def simple_random_assignment_session(
    treatment_labels: List[str], group_id_list: List[Any], random_seed: int
) -> dict[int, str]:
    """Assigns treatment labels randomly to each group using a simple random assignment strategy.

    Args:
        treatment_labels (List[str]): A list of treatment labels.
        group_id_list (List[Any]): The list of group IDs for assignment.
        random_seed (int): The random seed for reproducibility.

    Returns:
        dict[int, str]: A dictionary where the keys represent group IDs and the values represent the assigned treatment labels.
    """
    # Set the seed for reproducibility
    random.seed(random_seed)

    treatment_assignment = {}
    for group_id in group_id_list:
        if not treatment_labels:
            treatment_assignment[group_id] = ""
        else:
            treatment_assignment[group_id] = random.choice(treatment_labels)

    return treatment_assignment


def complete_random_assignment_session(
    treatment_labels: List[Any], group_id_list: List[Any], random_seed: int
) -> dict[int, str]:
    """Assigns treatment labels randomly to a specified number of groups using a complete random assignment strategy.

    Args:
        treatment_labels (List[str]): A list of treatment labels.
        group_id_list (List[Any]): The list of group IDs for assignment.
        random_seed (int): The random seed for reproducibility.

    Returns:
        dict[int, str]: A dictionary where the keys represent group IDs and the values represent the assigned treatment labels.
    """
    # Set the seed for reproducibility
    random.seed(random_seed)

    # Randomize the order of the group IDs
    randomised_groups = group_id_list.copy()
    random.shuffle(randomised_groups)

    num_treatments = len(treatment_labels)
    treatment_assignment = {}
    for i, group_id in enumerate(randomised_groups):
        if not treatment_labels:
            treatment_assignment[group_id] = ""
        else:
            treatment_assignment[group_id] = treatment_labels[i % num_treatments]
    return treatment_assignment


def manual_assignment_session(
    profiles: pd.DataFrame,
    treatment_column: str,
    group_column: str,
    group_id_list: List[Any],
) -> dict[Any, str]:
    """Extract the group treatment dictionary pairs provided by the user.

    Args:
        profiles (pd.DataFrame): A dataframe containing the manually assigned treatment arms for each subject profile.
        treatment_column (str): The column containing the assigned treatments.
        group_column (str): The column containing the group information.
        group_id_list (List[Any]): The list of group IDs for assignment.

    Returns:
        dict[int, str]: A dictionary where the keys represent group IDs and the values represent the assigned treatment labels.
    """
    group_treatment_dict = {}
    for group_id in group_id_list:
        group_treatment_set = set(
            profiles[profiles[group_column] == group_id][treatment_column].tolist()
        )

        if len(group_treatment_set) == 1:
            group_treatment_dict[group_id] = group_treatment_set.pop()
        else:
            raise ValueError(
                f"Group {group_id} is assigned different treatments: {group_treatment_set}"
            )

    return group_treatment_dict
