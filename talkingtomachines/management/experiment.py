from typing import Any
import pandas as pd
import datetime
import base64
from typing import List, Dict
from talkingtomachines.generative.synthetic_agent import ConversationalSyntheticAgent
from talkingtomachines.management.treatment import (
    simple_random_assignment,
    complete_random_assignment,
    full_factorial_assignment,
    block_random_assignment,
    cluster_random_assignment,
)

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
        self.experiment_context = experiment_context
        self.treatments = treatments

        # agent_demographics_with_treatment, treatment_assignment = self.assign_treatment()
        # self.agent_demographics_with_treatment = agent_demographics_with_treatment
        # self.treatment_assignment = treatment_assignment

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
        # if treatment_assignment_stratey == "manual" and "treatment" not in self.agent_demographics.columns:
        #     raise ValueError("Column 'treatment' not found in agent_demographics for manual treatment assignment.")
        else:
            return treatment_assignment_strategy

    def check_agent_demographics(self, agent_demographics: str) -> str:
        """Checks if the provided agent_demographics is empty.

        Args:
            agent_demographics (str): The agent_demographics to be checked.

        Returns:
            str: The validated agent_demographics.

        Raises:
            ValueError: If the provided agent_demographics is an empty dataframe.
        """
        if agent_demographics.empty:
            raise ValueError("agent_demographics DataFrame cannot be empty.")
        else:
            return agent_demographics

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

    # def assign_treatment(self) -> Dict[str, List[ConversationalSyntheticAgent]]:
    #     """Assign treatments to agents based on the specified treatment assignment strategy.

    #     Returns:
    #         dict: A dictionary where keys are treatment names and values are lists of agents assigned to those treatments.
    #     """
    #     if self.treatment_assignment_strategy == "simple_random":
    #         return simple_random_assignment()
    #     elif self.treatment_assignment_strategy == "complete_random":
    #         return complete_random_assignment()
    #     elif self.treatment_assignment_strategy == "full_factorial":
    #         return full_factorial_assignment()
    #     elif self.treatment_assignment_strategy == "block_randomisation":
    #         return block_random_assignment()
    #     elif self.treatment_assignment_strategy == "cluster_randomisation":
    #         return cluster_random_assignment()
    #     else:
    #         raise ValueError(f"Unsupported treatment_assignment_strategy: {self.treatment_assignment_strategy}. Supported strategies are: {self.SUPPORTED_ASSIGNMENT_STRATEGIES}.")

    # def generate_agent(self, session_id) -> ConversationalSyntheticAgent:
    #     """Generates a synthetic conversational agent based on the demographic information provided.

    #     Returns:
    #         ConversationalSyntheticAgent: A ConversationalSyntheticAgent object generated for this experiment.
    #     """
    #     if self.get_treatment_assignment == "random":
    #         treatment = self.random_treatment_assignment()
    #     else:
    #         treatment = ""

    #     agent = ConversationalSyntheticAgent(
    #         experiment_id=self.get_experiment_id(),
    #         experiment_context=self.get_experiment_context(),
    #         session_id=self.generate_session_id(),
    #         demographic_info=self.sample_agent_demographic(),
    #         model_info=self.get_model_info(),
    #         assigned_treatment=treatment,
    #     )

    #     return agent

    # def run_experiment(experiment_type: str, experiment_data: dict) -> dict:
    #     """Run an experiment of the specified type."""
    #     try:
    #         # Implement experiment module functionality
    #         pass
    #     except Exception as e:
    #         # Log the exception
    #         print(f"Error during experiment execution: {e}")
    #         return {}
