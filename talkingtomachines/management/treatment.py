import random
import pandas as pd
from talkingtomachines.generative.synthetic_agent import DemographicInfo
from typing import List, Any, Tuple
from itertools import product


def simple_random_assignment_session(
    treatment_labels: List[str], num_sessions: int
) -> dict[int, str]:
    """Assigns treatment labels randomly to each session using a simepl random assignment strategy.

    Args:
        treatment_labels (List[str]): A list of treatment labels.
        num_sessions (int): The number of sessions to assign treatment labels to.

    Returns:
        dict[int, str]: A dictionary where the keys represent session numbers and the values represent the assigned treatment labels.
    """
    treatment_assignment = {}
    for i in range(num_sessions):
        if not treatment_labels:
            treatment_assignment[i] = ""
        else:
            treatment_assignment[i] = random.choice(treatment_labels)

    return treatment_assignment


def complete_random_assignment_session(
    treatment_labels: List[Any], num_sessions: int
) -> dict[int, str]:
    """Assigns treatment labels randomly to a specified number of sessions using a complete random assignment strategy.

    Args:
        treatment_labels (List[str]): A list of treatment labels.
        num_sessions (int): The number of sessions to assign treatment labels to.

    Returns:
        dict[int, str]: A dictionary where the keys represent session numbers and the values represent the assigned treatment labels.
    """
    num_treatments = len(treatment_labels)

    treatment_assignment = {}
    for i in range(num_sessions):
        if not treatment_labels:
            treatment_assignment[i] = ""
        else:
            treatment_assignment[i] = treatment_labels[i % num_treatments]

    return treatment_assignment


def full_factorial_assignment_session(
    treatment_labels: List[List[str]], num_sessions: int
) -> dict[int, str]:
    """Assigns treatment labels to sessions using a full factorial design assignment strategy.

    Args:
        treatment_labels (List[List[str]]): A list of lists containing the treatment labels.
            Each inner list represents the possible labels for a specific treatment factor.
        num_sessions (int): The number of sessions to assign treatment labels to.

    Returns:
        dict[int, str]: A dictionary where the keys represent the session numbers and the values
            represent the assigned treatment labels.
    """
    if not treatment_labels:
        treatment_label_combinations = []
    else:
        treatment_label_combinations = list(product(*treatment_labels))

    return complete_random_assignment_session(
        treatment_label_combinations, num_sessions
    )


def simple_random_assignment_individual(
    treatment_labels: List[str], agent_demographics: pd.DataFrame
) -> Tuple[pd.DataFrame, dict[str, List]]:
    """Randomly assigns agents to different treatments based on their demographics using a simple random assignment strategy.

    Args:
        treatment_labels (List[str]): A list containing the treatment labels.
        agent_demographics (pd.DataFrame): A pandas DataFrame containing the demographic information of the agents.

    Returns:
        Tuple[pd.DataFrame, dict[str, List]]: A tuple containing the updated agent demographics DataFrame and a dictionary
        mapping treatment labels to lists of agent IDs assigned to each treatment.
    """
    num_agents = len(agent_demographics)
    agent_demographics["treatment"] = [
        random.choice(treatment_labels) for _ in range(num_agents)
    ]

    treatment_assignment = {}
    for label in treatment_labels:
        treatment_assignment[label] = agent_demographics[
            agent_demographics["treatment"] == label
        ]["ID"].tolist()

    return agent_demographics, treatment_assignment


def complete_random_assignment_individual(
    treatment_labels: List[Any],
    agent_demographics: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict[Any, List]]:
    """Randomly assigns agents to different treatments using a complete random assignment strategy.

    Args:
        treatment_labels (List[Any]): A list of treatment labels.
        agent_demographics (pd.DataFrame): A DataFrame containing agent demographics.

    Returns:
        Tuple[pd.DataFrame, dict[Any, List]]: A tuple containing the updated agent demographics DataFrame and a dictionary
        mapping treatment labels to lists of agent IDs assigned to each treatment.
    """
    num_agents = len(agent_demographics)
    num_treatments = len(treatment_labels)
    agent_demographics["treatment"] = [
        treatment_labels[i % num_treatments] for i in range(num_agents)
    ]

    treatment_assignment = {}
    for label in treatment_labels:
        treatment_assignment[label] = agent_demographics[
            agent_demographics["treatment"] == label
        ]["ID"].tolist()

    return agent_demographics, treatment_assignment


def full_factorial_assignment_individual(
    treatment_labels: List[List[str]],
    agent_demographics: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict[str, List]]:
    """Assigns treatments to agents using a full factorial design assignment strategy.

    This function takes a list of treatment labels and a DataFrame containing agent demographics,
    and assigns treatments to agents using a full factorial design. It generates all possible combinations
    of treatment labels and assigns agents to each combination randomly.

    Args:
        treatment_labels (List[List[str]]): A list of treatment labels. Each inner list represents the possible
            treatment options for a specific factor.
        agent_demographics (pd.DataFrame): A DataFrame containing agent demographics.

    Returns:
        Tuple[pd.DataFrame, dict[str, List]]: A tuple containing the updated agent demographics DataFrame and a dictionary
        mapping treatment labels to lists of agent IDs assigned to each treatment.
    """
    treatment_label_combinations = list(product(*treatment_labels))

    return complete_random_assignment_individual(
        treatment_label_combinations, agent_demographics
    )


def block_random_assignment_individual(
    treatment_labels: List[str], agent_demographics: pd.DataFrame, block_size: int
) -> Tuple[pd.DataFrame, dict[str, List]]:
    """Assigns treatments randomly to agents using block random assignment strategy.

    Args:
        treatment_labels (List[str]): A list of treatment labels.
        agent_demographics (pd.DataFrame): A DataFrame containing agent demographics.
        block_size (int): The size of each block.

    Returns:
        Tuple[pd.DataFrame, dict[str, List]]: A tuple containing the updated agent demographics DataFrame and a dictionary mapping block-treatment assignments to agent IDs.
    """
    num_agents = len(agent_demographics)
    num_treatments = len(treatment_labels)
    num_blocks = num_agents // block_size

    # Assign block to subjects
    block_assignments = [i % num_blocks for i in range(num_agents)]
    random.shuffle(block_assignment)
    agent_demographics["block"] = block_assignments

    # Assign treatments within each block
    agent_demographics.sort_values(by="block", inplace=True)
    agent_demographics["treatment"] = [
        treatment_labels[i % num_treatments] for i in range(num_agents)
    ]

    block_treatment_assignment = {}
    for block_assignment, treatment_assignment in list(
        product(set(block_assignments), treatment_labels)
    ):
        block_treatment_assignment[(block_assignment, treatment_assignment)] = (
            agent_demographics[
                (agent_demographics["block"] == block_assignment)
                & (agent_demographics["treatment"] == treatment_assignment)
            ]["ID"].tolist()
        )

    return agent_demographics, block_treatment_assignment


def cluster_random_assignment_individual(
    treatment_labels: List[str], agent_demographics: pd.DataFrame, cluster_criteria: str
) -> Tuple[pd.DataFrame, dict[str, List]]:
    """Assigns treatments randomly using a cluster random assignment strategy.

    Args:
        treatment_labels (List[str]): A list of treatment labels.
        agent_demographics (pd.DataFrame): A pandas DataFrame containing agent demographics.
        cluster_criteria (str): The column name in `agent_demographics` DataFrame to use for clustering.

    Returns:
        Tuple[pd.DataFrame, dict[str, List]]: A tuple containing the updated `agent_demographics` DataFrame
        with assigned treatments, and a dictionary mapping clusters to assigned treatments.
    """
    clusters = list(agent_demographics[cluster_criteria].unique())
    random.shuffle(clusters)

    num_clusters = len(clusters)
    num_treatments = len(treatment_labels)

    agent_demographics["treatment"] = None
    cluster_treatment_assignment = {}
    assigned_cluster_treatment = [
        (clusters[i], treatment_labels[i % num_treatments]) for i in range(num_clusters)
    ]
    for cluster, treatment in assigned_cluster_treatment:
        agent_demographics.loc[
            agent_demographics[cluster_criteria] == cluster, "treatment"
        ] = treatment
        cluster_treatment_assignment[(cluster, treatment)] = agent_demographics[
            agent_demographics[cluster_criteria] == cluster
        ]["ID"].tolist()

    # Check for None values in treatment column
    if agent_demographics["treatment"].isnull().any():
        raise ValueError(
            "Some agents have not been assigned a treatment when performing cluster random assignment."
        )

    return agent_demographics, cluster_treatment_assignment
