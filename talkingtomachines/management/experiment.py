import random
import pandas as pd
from typing import List
from talkingtomachines.generative.synthetic_agent import (
    ConversationalSyntheticAgent,
    DemographicInfo,
)


class Experiment:
    """A class for constructing the base experiment management class.

    Attributes:
        experiment_id (str): The unique ID of the experiment.
    """

    def __init__(self):
        self.experiment_id = self.generate_experiment_id()

    def generate_experiment_id(self) -> int:
        """Generates a unique ID for the experiment

        Returns:
            int: Unique ID for the experiment.
        """
        # TODO
        return 0

    def get_experiment_id(self) -> str:
        """Return the experiment ID of the synthetic agent.

        Returns:
            str: The experiment ID of the synthetic agent.
        """
        return self.experiment_id


class AIConversationalExperiment(Experiment):
    """A class for constructing AI-based conversational experiments. Inherits from the Experiment base class.

    Args:
        model_info (str): The information about the model used in this conversational experiment.
        experiment_context (str): The context of the experiment.

    Attributes:
        model_info (str): The information about the model used in this conversational experiment.
        experiment_context (str): The context of the experiment.
        experiment_id (str): The unique ID of the experiment.
        synthetic_agents (List[ConversationalSyntheticAgent]): The list of synthetic agents generated for this experiment.
    """

    def __init__(
        self,
        model_info: str,
        experiment_context: str,
        agent_demographics: pd.DataFrame,
        treatments: List[str] = [],
        treatment_assignment_strategy: str = "random",
        num_sessions: int = 10,
    ):
        super().__init__()
        # TODO need to ensure that model_info is supported
        self.model_info = model_info
        self.experiment_context = experiment_context

        # TODO need to ensure that num_session is >0
        self.num_sessions = num_sessions
        self.treatments = treatments

        # TODO need to ensure that treatment assignment strategy is supported: random
        self.treatment_assignment_strategy = treatment_assignment_strategy

        # TODO need to ensure that agent demographics is not empty
        self.agent_demographics = agent_demographics

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

    def get_num_sessions(self) -> int:
        """Return the number of sessions that will be generated for this experiment.

        Returns:
            str: The number of experimental sessions.
        """
        return self.num_sessions

    def get_treatments(self) -> List[str]:
        """Return the treatments for this experiment.

        Returns:
            str: The treatments assigned.
        """
        return self.treatments

    def get_treatment_assignment_strategy(self) -> str:
        """Return the treatment assignment strategy for this experiment.

        Returns:
            str: The treatments assignment strategy.
        """
        return self.treatment_assignment_strategy

    def get_agent_demographics(self) -> pd.DataFrame:
        """Return the agents' demographic information for this experiment.

        Returns:
            str: The agents' demographic information.
        """
        return self.agent_demographics

    def generate_session_id(self) -> int:
        """Generates a unique session ID to be assigned to a synthetic agent.

        Returns:
            int: Session ID for a synthetic agent.
        """
        # TODO
        return 0

    def random_treatment_assignment(self) -> str:
        """Randomly selects one of the treatments.

        Returns:
            str: Selected treatment arm.
        """
        treatment_list = self.get_treatments()

        if len(treatment_list) == 0:
            return ""
        else:
            return random.choice(treatment_list)

    def sample_agent_demographic(self) -> DemographicInfo:
        """Randomly samples a demographic profile for synthetic agent.

        Returns:
            DemographicInfo: Randomly sampled demographic profile for synthetic agent.
        """
        agent_demographics = self.get_agent_demographics()
        random_demographic = agent_demographics.sample(n=1)
        return random_demographic.to_dict(orient="records")[0]

    def generate_conversational_agent(self, session_id) -> ConversationalSyntheticAgent:
        """Generates a synthetic conversational agent based on the demographic information provided.

        Returns:
            ConversationalSyntheticAgent: A ConversationalSyntheticAgent object generated for this experiment.
        """
        if self.get_treatment_assignment == "random":
            treatment = self.random_treatment_assignment()
        else:
            treatment = ""

        agent = ConversationalSyntheticAgent(
            experiment_id=self.get_experiment_id(),
            experiment_context=self.get_experiment_context(),
            session_id=self.generate_session_id(),
            demographic_info=self.sample_agent_demographic(),
            model_info=self.get_model_info(),
            assigned_treatment=treatment,
        )

        return agent

    # def run_experiment(experiment_type: str, experiment_data: dict) -> dict:
    #     """Run an experiment of the specified type."""
    #     try:
    #         # Implement experiment module functionality
    #         pass
    #     except Exception as e:
    #         # Log the exception
    #         print(f"Error during experiment execution: {e}")
    #         return {}
