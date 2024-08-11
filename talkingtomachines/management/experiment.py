from typing import Any
import pandas as pd
import datetime
import base64
import random
from talkingtomachines.generative.synthetic_agent import (
    ConversationalSyntheticAgent,
    DemographicInfo,
)
from talkingtomachines.management.treatment import (
    simple_random_assignment_session,
    complete_random_assignment_session,
    full_factorial_assignment_session,
)
from talkingtomachines.storage.experiment import store_experiment

SUPPORTED_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
SUPPORTED_ASSIGNMENT_STRATEGIES = [
    "simple_random",
    "complete_random",
    "full_factorial",
    "block_randomisation",
    "cluster_randomisation",
    "manual",
]


class Experiment:
    """A class for constructing the base experiment management class.

    Attributes:
        experiment_id (str): The unique ID of the experiment.
    """

    def __init__(self):
        self.experiment_id = self.generate_experiment_id()

    def generate_experiment_id(self) -> str:
        """Generates a unique ID for the experiment by concatenating the date and time information and encoding it.

        Returns:
            str: Unique ID for the experiment as a base64 encoded string.
        """
        current_datetime = datetime.datetime.now()
        current_datetime_str = current_datetime.strftime("%Y%m%d%H%M%S%f")
        encoded_bytes = base64.urlsafe_b64encode(current_datetime_str.encode("utf-8"))
        experiment_id = encoded_bytes.decode("utf-8")

        return experiment_id

    def get_experiment_id(self) -> str:
        """Return the experiment ID of the synthetic agent.

        Returns:
            str: The experiment ID of the synthetic agent.
        """
        return self.experiment_id


class AIConversationalExperiment(Experiment):
    """A class representing an AI conversational experiment. Inherits from the Experiment base class.

    This class extends the base `Experiment` class and provides additional functionality
    specific to AI conversational experiments.

    Args:
        model_info (str): The information about the AI model used in the experiment.
        experiment_context (str): The context or purpose of the experiment.
        agent_demographics (pd.DataFrame): The demographic information of the agents participating in the experiment.
        treatments (dict[str, Any], optional): The treatments for the experiment. Defaults to an empty dictionary.
        treatment_assignment_strategy (str, optional): The strategy used for assigning treatments to agents.
            Defaults to "simple_random".

    Raises:
        ValueError: If the provided model_info is not supported.
        ValueError: If the provided treatment_assignment_strategy is not supported.
        ValueError: If the provided agent_demographics is an empty DataFrame.

    Attributes:
        model_info (str): The information about the AI model used in the experiment.
        experiment_context (str): The context or purpose of the experiment.
        agent_demographics (pd.DataFrame): The demographic information of the agents participating in the experiment.
        treatments (dict[str, Any]): The treatments for the experiment.
        treatment_assignment_strategy (str): The strategy used for assigning treatments to agents.
    """

    def __init__(
        self,
        model_info: str,
        experiment_context: str,
        agent_demographics: pd.DataFrame,
        treatments: dict[str, Any] = {},
        treatment_assignment_strategy: str = "simple_random",
    ):
        super().__init__()

        self.model_info = self.check_model_info(model_info)
        self.treatment_assignment_strategy = self.check_treatment_assignment_strategy(
            treatment_assignment_strategy
        )
        self.agent_demographics = self.check_agent_demographics(agent_demographics)
        self.treatments = self.check_treatments(treatments)
        self.experiment_context = experiment_context

    def check_model_info(self, model_info: str) -> str:
        """Checks if the provided model_info is supported.

        Args:
            model_info (str): The model_info to be checked.

        Returns:
            str: The validated model_info.

        Raises:
            ValueError: If the provided model_info is not supported.
        """
        if model_info not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_info: {model_info}. Supported models are: {SUPPORTED_MODELS}."
            )
        else:
            return model_info

    def check_treatment_assignment_strategy(
        self, treatment_assignment_strategy: str
    ) -> str:
        """Checks if the provided treatment_assignment_strategy is supported.

        Args:
            treatment_assignment_strategy (str): The treatment_assignment_strategy to be checked.

        Returns:
            str: The validated treatment_assignment_strategy.

        Raises:
            ValueError: If the provided treatment_assignment_strategy is not supported.
        """

        if treatment_assignment_strategy not in SUPPORTED_ASSIGNMENT_STRATEGIES:
            raise ValueError(
                f"Unsupported treatment_assignment_strategy: {treatment_assignment_strategy}. Supported strategies are: {SUPPORTED_ASSIGNMENT_STRATEGIES}."
            )
        else:
            return treatment_assignment_strategy

    def check_agent_demographics(
        self, agent_demographics: pd.DataFrame
    ) -> pd.DataFrame:
        """Checks to ensure that provided agent_demographics is not empty and contains a ID column.

        Args:
            agent_demographics (str): The agent_demographics to be checked.

        Returns:
            str: The validated agent_demographics.

        Raises:
            ValueError: If the provided agent_demographics is an empty dataframe or if it does not contain an ID column.
        """
        if agent_demographics.empty:
            raise ValueError("agent_demographics DataFrame cannot be empty.")
        elif "ID" not in agent_demographics.columns:
            raise ValueError(
                "agent_demographics DataFrame should contain an 'ID' column."
            )
        else:
            return agent_demographics

    def check_treatments(self, treatments: dict[str, Any]) -> dict[str, Any]:
        """Checks if the provided treatments is valid.

        Args:
            treatments (dict[str, Any]): The treatments to be checked.

        Returns:
            dict[str, Any]: The validated treatments.

        Raises:
            ValueError: If the provided treatments is not in the correct format.
        """
        if self.check_treatment_assignment_strategy == "full_factorial":
            for _, subtreatments in treatments.items():
                if not isinstance(subtreatments, dict):
                    raise ValueError(
                        "Invalid treatment format for full factorial assignment strategy. Please store the treatments in a nested dictionary structure."
                    )

        return treatments

    def get_model_info(self) -> str:
        """Return the model used in this experiment.

        Returns:
            str: The model information.
        """
        return self.model_info

    def get_experiment_context(self) -> str:
        """Return the experiment context.

        Returns:
            str: The experiment context.
        """
        return self.experiment_context

    def get_agent_demographics(self) -> pd.DataFrame:
        """Return the agents' demographic information for this experiment.

        Returns:
            pd.DataFrame: The agents' demographic information.
        """
        return self.agent_demographics

    def get_treatments(self) -> dict[str, Any]:
        """Return the treatments for this experiment.

        Returns:
            dict[str, Any]: The treatments for this experiment.
        """
        return self.treatments

    def get_treatment_assignment_strategy(self) -> str:
        """Return the treatment assignment strategy for this experiment.

        Returns:
            str: The treatments assignment strategy.
        """
        return self.treatment_assignment_strategy


class AItoAIConversationalExperiment(AIConversationalExperiment):
    """A class representing an AI-to-AI conversational experiment. Inherits from the AIConversationalExperiment class."""

    def __init__(
        self,
        model_info: str,
        experiment_context: str,
        agent_demographics: pd.DataFrame,
        agent_roles: dict[str, str],
        num_agents_per_session: int = 2,
        num_sessions: int = 10,
        treatments: dict[str, Any] = {},
        treatment_assignment_strategy: str = "simple_random",
    ):
        super().__init__(
            model_info,
            experiment_context,
            agent_demographics,
            treatments,
            treatment_assignment_strategy,
        )

        self.num_sessions = self.check_num_sessions(num_sessions)
        self.num_agents_per_session = self.check_num_agents_per_session(
            num_agents_per_session
        )
        self.agent_roles = self.check_agent_roles(agent_roles)
        self.treatment_assignment = self.assign_treatment()
        self.session_id_list = list(self.treatment_assignment.keys())
        self.agent_assignment = self.assign_agents_to_session()

    def check_num_sessions(self, num_sessions: int) -> int:
        """Checks if the provided num_sessions is valid.

        Args:
            num_sessions (int): The num_sessions to be checked.

        Returns:
            int: The validated num_sessions.

        Raises:
            ValueError: If the provided check_num_sessions is not valid.
        """
        if num_sessions < 1:
            raise ValueError(
                f"Unsupported valid for num_sessions: {num_sessions}. num_sessions should be an integer that is equal to or greater than 1."
            )
        else:
            return num_sessions

    def check_num_agents_per_session(self, num_agents_per_session: int) -> int:
        """Checks if the provided num_agents_per_session is valid.

        Args:
            num_agents_per_session (int): The num_agents_per_session to be checked.

        Returns:
            int: The validated num_agents_per_session.

        Raises:
            ValueError: If the provided num_agents_per_session is not valid.
        """
        if num_agents_per_session < 2:
            raise ValueError(
                f"Unsupported num_agents_per_session: {num_agents_per_session}. For AI-AI conversation-based experiments, num_agents_per_session should be an integer that is equal to or greater than 2."
            )
        elif self.num_sessions * num_agents_per_session > len(self.agent_demographics):
            raise ValueError(
                f"Total number of agents required for experiment ({self.num_sessions * num_agents_per_session}) exceed the number of profiles provided in agent_demographics ({len(self.agent_demographics)})."
            )
        else:
            return num_agents_per_session

    def check_agent_roles(self, agent_roles: dict[str, str]) -> dict[str, str]:
        """Checks if the provided agent_roles is valid.

        Args:
            agent_roles (dict[str, str]): The agent_roles to be checked.

        Returns:
            dict[str, str]: The validated agent_roles.

        Raises:
            ValueError: If the provided agent_roles is not valid.
        """
        if len(agent_roles) != self.num_agents_per_session:
            raise ValueError(
                f"Number of roles defined ({len(agent_roles)}) does not match the number of agents assigned to each session ({self.num_agents_per_session})."
            )
        else:
            return agent_roles

    def get_num_sessions(self) -> int:
        """Return the num_sessions defined this experiment.

        Returns:
            int: The num_sessions information.
        """
        return self.num_sessions

    def get_num_agents_per_session(self) -> int:
        """Return the num_agents_per_session defined this experiment.

        Returns:
            int: The num_agents_per_session information.
        """
        return self.num_agents_per_session

    def get_agent_roles(self) -> dict[str, str]:
        """Return the agent_roles defined this experiment.

        Returns:
            dict[str, str]: The agent_roles information.
        """
        return self.agent_roles

    def assign_treatment(self) -> dict[int, str]:
        """Assign treatments to agents based on the specified treatment assignment strategy.

        Returns:
            dict: A dictionary where keys are treatment names and values are lists of agents assigned to those treatments.
        """
        if self.treatment_assignment_strategy == "simple_random":
            treatment_labels = list(self.treatment.keys())
            return simple_random_assignment_session(treatment_labels, self.num_sessions)

        elif self.treatment_assignment_strategy == "complete_random":
            treatment_labels = list(self.treatment.keys())
            return complete_random_assignment_session(
                treatment_labels, self.num_sessions
            )

        elif self.treatment_assignment_strategy == "full_factorial":
            treatment_labels = []
            for _, inner_treatment_dict in self.treatment.items():
                inner_treatment_labels = list(inner_treatment_dict.keys())
                treatment_labels.append(inner_treatment_labels)
            return full_factorial_assignment_session(
                treatment_labels, self.num_sessions
            )

        else:
            raise ValueError(
                f"Unsupported treatment_assignment_strategy: {self.treatment_assignment_strategy}. Supported strategies are: {self.SUPPORTED_ASSIGNMENT_STRATEGIES}."
            )

    def assign_agents_to_session(self) -> dict[int, list[DemographicInfo]]:
        """Randomly assigns agents' demographics to each session based on the given number of agents per session.

        Returns:
            dict[int, list[DemographicInfo]]: A dictionary mapping session IDs to a list of agent demographic information.
        """
        randomised_agent_demographics = self.agent_demographics.sample(
            frac=1
        ).reset_index(drop=True)

        agent_to_session_assignment = {}
        for i, session_id in enumerate(self.session_id_list):
            agent_to_session_assignment[session_id] = (
                randomised_agent_demographics.iloc[
                    i : i + self.num_agents_per_session + 1
                ].to_dict(orient="records")
            )

        return agent_to_session_assignment

    def run_experiment(self, test_mode: bool = True) -> dict[str, Any]:
        """Runs an experiment based on the experimental settings defined during class initialisation. If test_mode is set to True, one of the predefined sessions will be randomly selected and run.

        Args:
            test_mode (bool, optional): Indicates whether the experiment is in test mode or not.
                Defaults to True.

        Returns:
            dict[str, Any]: A dictionary containing the experiment ID and session information.
        """

        if test_mode:
            session_id_list = random.choice(self.session_id_list)
        else:
            session_id_list = self.session_id_list

        experiment = {"experiment_id": self.experiment_id, "sessions": {}}
        for session_id in session_id_list:
            session_info = {}
            session_info["session_id"] = session_id
            treatment_label = self.treatment_assignment[session_id]
            session_info["treatment"] = self.treatments[treatment_label]
            session_info["agents_demographic"] = self.agent_assignment[session_id]
            session_info["agents"] = self.initialize_agents(session_info)
            session_info = self.run_session(session_info)
            experiment["sessions"][session_id] = session_info

        self.save_experiment(experiment)

        return experiment

    def initialize_agents(
        self, session_info: dict[str, Any]
    ) -> list[ConversationalSyntheticAgent]:
        """Initializes and returns a list of ConversationalSyntheticAgent objects based on the provided session information.

        Args:
            session_info (dict[str, Any]): A dictionary containing session information, including agents' demographics, session ID, treatment, etc.

        Returns:
            list[ConversationalSyntheticAgent]: A list of initialized ConversationalSyntheticAgent objects.

        Raises:
            AssertionError: If the number of agents' demographics does not match the number of agent roles when initializing agents.
        """
        assert len(session_info["agents_demographic"]) == len(
            self.agent_roles
        ), "Number of agents' demographics does not match the number of agent roles when initialising agents."
        agent_list = []
        for i in range(len(session_info["agents_demographic"])):
            agent_demographic = session_info["agents_demographic"][i]
            agent_list.append(
                ConversationalSyntheticAgent(
                    experiment_id=self.experiment_id,
                    experiment_context=self.experiment_context,
                    session_id=session_info["session_id"],
                    agent_id=agent_demographic["ID"],
                    demographic_info=agent_demographic,
                    role=list(self.agent_roles.values())[i],
                    model_info=self.model_info,
                    treatment=session_info["treatment"],
                )
            )

        return agent_list

    def run_session(self):
        return None

    def save_experiment(self, experiment: dict[int, Any]):
        """Save the experiment data.

        Args:
            experiment (dict[int, Any]): The experiment data to be saved.

        Returns:
            None
        """
        store_experiment(experiment)
