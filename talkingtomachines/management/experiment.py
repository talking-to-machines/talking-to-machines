from typing import Any, List
import pandas as pd
import datetime
from tqdm import tqdm
from talkingtomachines.generative.synthetic_agent import (
    ConversationalSyntheticAgent,
    DemographicInfo,
)
from talkingtomachines.management.treatment import (
    simple_random_assignment_session,
    complete_random_assignment_session,
    full_factorial_assignment_session,
    manual_assignment_session,
)
from talkingtomachines.generative.prompt import (
    generate_conversational_session_system_message,
)
from talkingtomachines.storage.experiment import save_experiment

SUPPORTED_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
SUPPORTED_TREATMENT_ASSIGNMENT_STRATEGIES = [
    "simple_random",
    "complete_random",
    "full_factorial",
    "block_randomisation",
    "cluster_randomisation",
    "manual",
]
SUPPORTED_AGENT_ASSIGNMENT_STRATEGIES = [
    "random",
    "manual",
]


class Experiment:
    """A class for constructing the base experiment class.

    Args:
        experiment_id (str): The unique ID of the experiment.

    Attributes:
        experiment_id (str): The unique ID of the experiment.
    """

    def __init__(self, experiment_id: str = ""):
        if experiment_id == "":
            self.experiment_id = self.generate_experiment_id()
        else:
            self.experiment_id = experiment_id

    def generate_experiment_id(self) -> str:
        """Generates a unique ID for the experiment by concatenating the date and time information.

        Returns:
            str: Unique ID for the experiment as a base64 encoded string.
        """
        current_datetime = datetime.datetime.now()
        experiment_id = current_datetime.strftime("%Y%m%d_%H%M%S")

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
        experiment_id (str, optional): The unique ID of the experiment. Defaults to an empty string.
        max_conversation_length (int, optional): The maximum length of a conversation. Defaults to 10.
        treatments (dict[str, Any], optional): The treatments for the experiment. Defaults to an empty dictionary.
        treatment_assignment_strategy (str, optional): The strategy used for assigning treatments to agents. Defaults to "simple_random".
        agent_assignment_strategy (str, optional): The strategy used for assigning agents to sessions. Defaults to "random".
        treatment_column (str, optional): The column in agent_demographics that contains the manually assigned treatments. Defaults to an empty string.
        session_column (str, optional): The column in agent_demographics that contains the manually assigned sessions. Defaults to an empty string.

    Raises:
        ValueError: If the provided model_info is not supported.
        ValueError: If the provided treatment_assignment_strategy is not supported.
        ValueError: If the provided agent_demographics is an empty DataFrame or does not contain a 'ID' column.
        ValueError: If the provided max_conversation_length is lesser than 5.
        ValueError: If the provided treatment is not in the nested dictionary structure when treatment_assignment_strategy is 'full_factorial'.

    Attributes:
        model_info (str): The information about the AI model used in the experiment.
        experiment_context (str): The context or purpose of the experiment.
        agent_demographics (pd.DataFrame): The demographic information of the agents participating in the experiment.
        experiment_id (str): The unique ID of the experiment.
        max_conversation_length (int): The maximum length of a conversation.
        treatments (dict[str, Any]): The treatments for the experiment.
        treatment_assignment_strategy (str): The strategy used for assigning treatments to agents.
        agent_assignment_strategy (str, optional): The strategy used for assigning agents to sessions.
        treatment_column (str, optional): The column in agent_demographics that contains the manually assigned treatments.
        session_column (str, optional): The column in agent_demographics that contains the manually assigned sessions.
    """

    def __init__(
        self,
        model_info: str,
        experiment_context: str,
        agent_demographics: pd.DataFrame,
        experiment_id: str = "",
        max_conversation_length: int = 10,
        treatments: dict[str, Any] = {},
        treatment_assignment_strategy: str = "simple_random",
        agent_assignment_strategy: str = "random",
        treatment_column: str = "",
        session_column: str = "",
    ):
        super().__init__(
            experiment_id,
        )

        self.model_info = self.check_model_info(model_info)
        self.agent_demographics = self.check_agent_demographics(agent_demographics)
        self.treatment_assignment_strategy = self.check_treatment_assignment_strategy(
            treatment_assignment_strategy, treatment_column, session_column
        )
        self.agent_assignment_strategy = self.check_agent_assignment_strategy(
            agent_assignment_strategy, session_column
        )
        self.treatment_column = treatment_column
        self.session_column = session_column
        self.max_conversation_length = self.check_max_conversation_length(
            max_conversation_length
        )
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

        return model_info

    def check_treatment_assignment_strategy(
        self,
        treatment_assignment_strategy: str,
        treatment_column: str,
        session_column: str,
    ) -> str:
        """Checks if the provided treatment_assignment_strategy is supported.

        Args:
            treatment_assignment_strategy (str): The treatment_assignment_strategy to be checked.
            treatment_column (str): The column name containing information about the manually assigned treatments.
            session_column (str): The column name containing the session information.

        Returns:
            str: The validated treatment_assignment_strategy.

        Raises:
            ValueError: If the provided treatment_assignment_strategy is not supported.
            ValueError: If treatment_column is an empty string or not one of the columns in agent_demographics when using the manual treatment assignment strategy.
            ValueError: If session_column is an empty string or not one of the columns in agent_demographics when using the manual treatment assignment strategy.
        """
        if (
            treatment_assignment_strategy
            not in SUPPORTED_TREATMENT_ASSIGNMENT_STRATEGIES
        ):
            raise ValueError(
                f"Unsupported treatment_assignment_strategy: {treatment_assignment_strategy}. Supported strategies are: {SUPPORTED_TREATMENT_ASSIGNMENT_STRATEGIES}."
            )

        # Check that treatment_column and session_column can be found in agent_demographics when using manual treatment assignment
        if treatment_assignment_strategy == "manual":
            if (
                treatment_column == ""
                or treatment_column not in self.agent_demographics.columns
            ):
                raise ValueError(
                    f"The argument 'treatment_column' cannot be an empty string and must be one of the columns in agent_demographics when performing manual treatment assignment."
                )
            if (
                session_column == ""
                or session_column not in self.agent_demographics.columns
            ):
                raise ValueError(
                    f"The argument 'session_column' cannot be an empty string and must be one of the columns in agent_demographics when performing manual treatment assignment."
                )

        return treatment_assignment_strategy

    def check_agent_assignment_strategy(
        self, agent_assignment_strategy: str, session_column: str
    ) -> str:
        """Checks if the provided agent_assignment_strategy is supported.

        Args:
            agent_assignment_strategy (str): The agent_assignment_strategy to be checked.
            session_column (str): The column name containing the session information.

        Returns:
            str: The validated agent_assignment_strategy.

        Raises:
            ValueError: If the provided agent_assignment_strategy is not supported.
            ValueError: If session_column is an empty string or not one of the columns in agent_demographics when using the manual agent assignment strategy.
        """
        if agent_assignment_strategy not in SUPPORTED_AGENT_ASSIGNMENT_STRATEGIES:
            raise ValueError(
                f"Unsupported agent_assignment_strategy: {agent_assignment_strategy}. Supported strategies are: {SUPPORTED_AGENT_ASSIGNMENT_STRATEGIES}."
            )

        # Check that session_column can be found in agent_demographics when using manual agent assignment
        if agent_assignment_strategy == "manual":
            if (
                session_column == ""
                or session_column not in self.agent_demographics.columns
            ):
                raise ValueError(
                    f"The argument 'session_column' cannot be an empty string and must be one of the columns in agent_demographics when performing manual agent assignment."
                )

        return agent_assignment_strategy

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

        if "ID" not in agent_demographics.columns:
            raise ValueError(
                "agent_demographics DataFrame should contain an 'ID' column."
            )

        return agent_demographics

    def check_max_conversation_length(self, max_conversation_length: int) -> int:
        """Checks if the provided max_conversation is an integer greater than or equal to 5.

        Args:
            max_conversation_length (int): The max_conversation_length to be checked.

        Returns:
            int: The validated max_conversation_length.

        Raises:
            ValueError: If the provided treatments is less than 5.
        """
        if max_conversation_length < 5:
            raise ValueError(
                "Invalid value for max_conversation_length. Please ensure that max_conversation_length is an integer greater than or equal to 5."
            )

        return max_conversation_length

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

    def get_max_conversation_length(self) -> int:
        """Return the maximum length of a conversation for this experiment.

        Returns:
            int: The maximum length of a conversation.
        """
        return self.max_conversation_length

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
    """A class representing an AI-to-AI conversational experiment. Inherits from the AIConversationalExperiment class.

    This class extends the `AIConversationalExperiment` class and provides additional functionality
    specific to AI-to-AI conversational experiments.

    Args:
        model_info (str): The information about the AI model used in the experiment.
        experiment_context (str): The context or purpose of the experiment.
        agent_demographics (pd.DataFrame): The demographic information of the agents participating in the experiment.
        agent_roles (dict[str, str]): Dictionary mapping agent roles to their descriptions.
        experiment_id (str, optional): The unique ID of the experiment. Defaults to an empty string.
        num_agents_per_session (int, optional): Number of agents per session. Defaults to 2.
        num_sessions (int, optional): Number of sessions. Defaults to 10.
        max_conversation_length (int, optional): Maximum length of a conversation. Defaults to 10.
        treatments (dict[str, Any], optional): The treatments for the experiment. Defaults to an empty dictionary.
        treatment_assignment_strategy (str, optional): The strategy used for assigning treatments to sessions. Defaults to "simple_random".
        agent_assignment_strategy (str, optional): The strategy used for assigning agents to sessions. Defaults to "random".
        treatment_column (str, optional): The column in agent_demographics that contains the manually assigned treatments. Defaults to an empty string.
        session_column (str, optional): The column in agent_demographics that contains the manually assigned sessions. Defaults to an empty string.

    Raises:
        ValueError: If the provided num_sessions is not valid.
        ValueError: If the provided num_agents_per_session is less than 2 or will exceed the total number of demographic information.
        ValueError: If the provided number of agent_roles is not equal to num_agents_per_session.
        ValueError: If the number of roles defined does not match the number of agents assigned to each session.

    Attributes:
        model_info (str): The information about the AI model used in the experiment.
        experiment_context (str): The context or purpose of the experiment.
        agent_demographics (pd.DataFrame): The demographic information of the agents participating in the experiment.
        agent_roles (dict[str, str]): The roles assigned to agents.
        experiment_id (str): The unique ID of the experiment.
        num_agents_per_session (int): The number of agents per session.
        num_sessions (int): The number of sessions in the experiment.
        max_conversation_length (int): The maximum length of a conversation.
        treatments (dict[str, Any]): The treatments for the experiment.
        treatment_assignment_strategy (str, optional): The strategy used for assigning treatments to agents.
        agent_assignment_strategy (str, optional): The strategy used for assigning agents to sessions.
        treatment_column (str, optional): The column in agent_demographics that contains the manually assigned treatments.
        session_column (str, optional): The column in agent_demographics that contains the manually assigned sessions.
        session_id_list (List[int]): List of session IDs.
        treatment_assignment (dict[int, str]): The assignment of treatments to agents.
        agent_assignment (dict[int, List[DemographicInfo]]): The assignment of agents to sessions.
    """

    def __init__(
        self,
        model_info: str,
        experiment_context: str,
        agent_demographics: pd.DataFrame,
        agent_roles: dict[str, str],
        experiment_id: str = "",
        num_agents_per_session: int = 2,
        num_sessions: int = 10,
        max_conversation_length: int = 10,
        treatments: dict[str, Any] = {},
        treatment_assignment_strategy: str = "simple_random",
        agent_assignment_strategy: str = "random",
        treatment_column: str = "",
        session_column: str = "",
    ):
        super().__init__(
            model_info,
            experiment_context,
            agent_demographics,
            experiment_id,
            max_conversation_length,
            treatments,
            treatment_assignment_strategy,
            agent_assignment_strategy,
            treatment_column,
            session_column,
        )

        self.num_sessions = self.check_num_sessions(num_sessions)
        self.num_agents_per_session = self.check_num_agents_per_session(
            num_agents_per_session
        )
        self.agent_roles = self.check_agent_roles(agent_roles)

        self.session_id_list = self.generate_session_id_list()

        self.treatment_assignment = self.assign_treatment()
        if self.treatment_assignment_strategy == "manual":
            self.check_manually_assigned_treatments()

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

        if self.num_sessions * num_agents_per_session > len(self.agent_demographics):
            raise ValueError(
                f"Total number of agents required for experiment ({self.num_sessions * num_agents_per_session}) exceed the number of profiles provided in agent_demographics ({len(self.agent_demographics)})."
            )

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

        return agent_roles

    def generate_session_id_list(self) -> List[Any]:
        """Generates a list of session IDs.

        If either the treatment assignment strategy or the agent assignment strategy is set to 'manual',
        the function returns a list of unique session IDs from the agent demographics DataFrame.
        Otherwise, it returns a list of sequential integers starting from 0 up to the number of sessions.

        Returns:
            List[Any]: A list of session IDs. If the assignment strategies are manual, the list contains unique session IDs
                from the agent demographics DataFrame. Otherwise, it contains sequential integers starting from 0.
        """
        if (
            self.treatment_assignment_strategy == "manual"
            or self.agent_assignment_strategy == "manual"
        ):
            return list(self.agent_demographics[self.session_column].unique())
        else:
            return list(range(self.num_sessions))

    def check_manually_assigned_treatments(self) -> None:
        """Checks if the manually defined treatments align with the treatment labels provided in self.treatments.

        Raises:
            ValueError: If the manually defined treatments do not align with the treatment labels provided in self.treatments.
        """
        treatment_label_set = set(self.treatments.keys())
        manual_defined_treatments = set(self.treatment_assignment.values())

        if not treatment_label_set.issuperset(manual_defined_treatments):
            raise ValueError(
                f"The treatment labels defined in 'treatments' ({list[treatment_label_set]}) is not a superset of the manually defined treatments in agent_demographics ({list[manual_defined_treatments]})."
            )
        else:
            pass

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

    def get_treatment_assignment(self) -> dict[int, str]:
        """Return the treatment_assignment defined this experiment.

        Returns:
            dict[int, str]: The treatment_assignment information.
        """
        return self.treatment_assignment

    def get_session_id_list(self) -> List[int]:
        """Return the session_id_list of this experiment.

        Returns:
            list[int]: The session_id_list information.
        """
        return self.session_id_list

    def get_agent_assignment(self) -> dict[int, list[DemographicInfo]]:
        """Return the agent_assignment for this experiment.

        Returns:
            dict[int, list[DemographicInfo]]: The agent_assignment information.
        """
        return self.agent_assignment

    def assign_treatment(self) -> dict[int, str]:
        """Assign treatments to sessions based on the specified treatment assignment strategy.

        Returns:
            dict[int, str]: A dictionary where the keys represent session numbers and the values represent the assigned treatment labels.
        """
        if self.treatment_assignment_strategy == "simple_random":
            treatment_labels = list(self.treatments.keys())
            return simple_random_assignment_session(
                treatment_labels, self.session_id_list
            )

        elif self.treatment_assignment_strategy == "complete_random":
            treatment_labels = list(self.treatments.keys())
            return complete_random_assignment_session(
                treatment_labels, self.session_id_list
            )

        elif self.treatment_assignment_strategy == "full_factorial":
            treatment_labels = []
            for _, inner_treatment_dict in self.treatments.items():
                inner_treatment_labels = list(inner_treatment_dict.keys())
                treatment_labels.append(inner_treatment_labels)
            return full_factorial_assignment_session(
                treatment_labels, self.session_id_list
            )

        elif self.treatment_assignment_strategy == "manual":
            return manual_assignment_session(
                self.agent_demographics,
                self.treatment_column,
                self.session_column,
                self.session_id_list,
            )

        else:
            raise ValueError(
                f"Unsupported treatment_assignment_strategy: {self.treatment_assignment_strategy}. Supported strategies are: {SUPPORTED_TREATMENT_ASSIGNMENT_STRATEGIES}."
            )

    def assign_agents_to_session(self) -> dict[int, List[DemographicInfo]]:
        """Assigns agents' demographics to each session based on the given number of agents per session and agent assignment strategy.
        However, if the agent_assignment_strategy is 'manual', then assign the agents to their respective sessions based on the
        assignment defined in agent_demographics.

        Returns:
            dict[int, List[DemographicInfo]]: A dictionary mapping session IDs to a list of agent demographic information.
        """
        if self.agent_assignment_strategy == "manual":
            agent_to_session_assignment = {}
            for i, session_id in enumerate(self.session_id_list):
                session_participants = self.agent_demographics[
                    self.agent_demographics[self.session_column] == session_id
                ]

                if self.treatment_assignment_strategy == "manual":
                    session_participants_filtered = session_participants.drop(
                        columns=[self.session_column, self.treatment_column]
                    )
                else:
                    session_participants_filtered = session_participants.drop(
                        columns=[self.session_column]
                    )

                num_session_participants = len(session_participants_filtered)
                if num_session_participants != self.num_agents_per_session:
                    raise ValueError(
                        f"Session {session_id} contains {num_session_participants} participants while the number of participants per session is supposed to be {self.num_agents_per_session}"
                    )

                agent_to_session_assignment[session_id] = (
                    session_participants_filtered.to_dict(orient="records")
                )

        else:
            randomised_agent_demographics = self.agent_demographics.sample(
                frac=1
            ).reset_index(drop=True)

            agent_to_session_assignment = {}
            for i, session_id in enumerate(self.session_id_list):
                agent_to_session_assignment[session_id] = (
                    randomised_agent_demographics.iloc[
                        i
                        * self.num_agents_per_session : (i + 1)
                        * self.num_agents_per_session
                    ].to_dict(orient="records")
                )

        return agent_to_session_assignment

    def run_experiment(self, test_mode: bool = True) -> dict[str, Any]:
        """Runs an experiment based on the experimental settings defined during class initialisation. If test_mode is set to True, the first session will be selected and run.

        Args:
            test_mode (bool, optional): Indicates whether the experiment is in test mode or not.
                Defaults to True.

        Returns:
            dict[str, Any]: A dictionary containing the experiment ID and session information.
        """

        if test_mode:
            session_id_list = [self.session_id_list[0]]
        else:
            session_id_list = self.session_id_list

        experiment = {"experiment_id": self.experiment_id, "sessions": {}}
        for session_id in tqdm(session_id_list):
            session_info = {}
            session_info["session_id"] = session_id
            treatment_label = self.treatment_assignment[session_id]
            session_info["treatment"] = self.treatments[treatment_label]
            session_info["session_system_message"] = (
                generate_conversational_session_system_message(
                    experiment_context=self.experiment_context,
                    treatment=session_info["treatment"],
                )
            )
            session_info["agents_demographic"] = self.agent_assignment[session_id]
            session_info["agents"] = self.initialize_agents(session_info)
            session_info = self.run_session(session_info, test_mode=test_mode)
            session_info["agents"] = [
                agent.to_dict() for agent in session_info["agents"]
            ]
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
                    demographic_info=agent_demographic,
                    role=list(self.agent_roles.keys())[i],
                    role_description=list(self.agent_roles.values())[i],
                    model_info=self.model_info,
                    treatment=session_info["treatment"],
                )
            )

        return agent_list

    def run_session(
        self, session_info: dict[str, Any], test_mode: bool = False
    ) -> dict[str, Any]:
        """Runs a session involving a conversation between multiple AI agents.

        Args:
            session_info (dict[str, Any]): A dictionary containing session information.
            test_mode (bool, optional): A boolean indicating if the session is executed under test mode. In test mode, only the first session is executed and all responses are printed out for easy reference.

        Returns:
            dict[str, Any]: A dictionary containing the updated session information at the end of the session.
        """
        message_history = []
        conversation_length = 0
        num_agents = len(session_info["agents"])
        response = session_info["session_system_message"]
        agent_role = "system"
        while (
            "Thank you for the conversation" not in response
            and conversation_length < self.max_conversation_length
        ):
            message_history.append({agent_role: response})
            if test_mode:
                print({agent_role: response})
                print()
            agent = session_info["agents"][conversation_length % num_agents]

            if conversation_length == 0:
                response = agent.respond(question="Start")
            else:
                response = agent.respond(question=response)
            agent_role = agent.demographic_info["ID"]
            conversation_length += 1
        message_history.append({agent_role: response})
        message_history.append({"system": "End"})
        if test_mode:
            print({agent_role: response})
            print()
            print({"system": "End"})

        session_info["message_history"] = message_history
        return session_info

    def save_experiment(self, experiment: dict[int, Any]) -> None:
        """Save the experimental data.

        Args:
            experiment (dict[int, Any]): The experiment data to be saved.

        Returns:
            None
        """
        save_experiment(experiment)


class AItoAIInterviewExperiment(AItoAIConversationalExperiment):
    """A class representing an AI-to-AI interview experiment. Inherits from the AItoAIConversationalExperiment class.

    This class extends the `AItoAIConversationalExperiment` class and provides additional functionality
    specific to AI-to-AI interview experiments. More specifically, one of the agents in the session will serve
    as the interviewer and will not be given a demographic profile.

    Args:
        model_info (str): The information about the AI model used in the experiment.
        experiment_context (str): The context or purpose of the experiment.
        agent_demographics (pd.DataFrame): The demographic information of the agents participating in the experiment.
        agent_roles (dict[str, str]): Dictionary mapping agent roles to their descriptions.
        experiment_id (str, optional): The unique ID of the experiment. Defaults to an empty string.
        num_agents_per_session (int, optional): Number of agents per session. Defaults to 2.
        num_sessions (int, optional): Number of sessions. Defaults to 10.
        max_conversation_length (int, optional): Maximum length of a conversation. Defaults to 10.
        treatments (dict[str, Any], optional): The treatments for the experiment. Defaults to an empty dictionary.
        treatment_assignment_strategy (str, optional): The strategy used for assigning treatments to agents. Defaults to "simple_random".
        agent_assignment_strategy (str, optional): The strategy used for assigning agents to sessions. Defaults to "random".
        treatment_column (str, optional): The column in agent_demographics that contains the manually assigned treatments. Defaults to an empty string.
        session_column (str, optional): The column in agent_demographics that contains the manually assigned sessions. Defaults to an empty string.
        interview_script (dict[int, Any], optional): An optional dictionary containing the interview script that the interviewer agent has to follow. Defaults to an empty dictionary.

    Raises:
        ValueError: If the provided num_sessions is not valid.
        ValueError: If the provided num_agents_per_session is less than 2 or will exceed the total number of demographic information.
        ValueError: If the provided number of agent_roles is not equal to num_agents_per_session or if the first role is not Interviewer.
        ValueError: If the number of roles defined does not match the number of agents assigned to each session. Also if the first role is not Interviewer.
        ValueError: If the format of the interview script does not fit with the expected format.

    Attributes:
        model_info (str): The information about the AI model used in the experiment.
        experiment_context (str): The context or purpose of the experiment.
        agent_demographics (pd.DataFrame): The demographic information of the agents participating in the experiment.
        agent_roles (dict[str, str]): Dictionary mapping agent roles to their descriptions.
        experiment_id (str): The unique ID of the experiment.
        num_agents_per_session (int, optional): Number of agents per session.
        num_sessions (int, optional): Number of sessions.
        max_conversation_length (int, optional): Maximum length of a conversation.
        treatments (dict[str, Any], optional): The treatments for the experiment.
        treatment_assignment_strategy (str, optional): The strategy used for assigning treatments to agents.
        agent_assignment_strategy (str, optional): The strategy used for assigning agents to sessions.
        treatment_column (str, optional): The column in agent_demographics that contains the manually assigned treatments.
        session_column (str, optional): The column in agent_demographics that contains the manually assigned sessions.
        interview_script (dict[int, Any], optional): A dictionary containing the interview script that the interviewer agent has to follow.
        treatment_assignment (dict[int, str]): The assignment of treatments to agents.
        session_id_list (List[Any]): List of session IDs.
        agent_assignment (dict[int, list[DemographicInfo]]): The assignment of agents to sessions.
    """

    def __init__(
        self,
        model_info: str,
        experiment_context: str,
        agent_demographics: pd.DataFrame,
        agent_roles: dict[str, str],
        experiment_id: str = "",
        num_agents_per_session: int = 2,
        num_sessions: int = 10,
        max_conversation_length: int = 10,
        treatments: dict[str, Any] = {},
        treatment_assignment_strategy: str = "simple_random",
        agent_assignment_strategy: str = "random",
        treatment_column: str = "",
        session_column: str = "",
        interview_script: dict[int, Any] = {},
    ):
        super().__init__(
            model_info,
            experiment_context,
            agent_demographics,
            agent_roles,
            experiment_id,
            num_agents_per_session,
            num_sessions,
            max_conversation_length,
            treatments,
            treatment_assignment_strategy,
            agent_assignment_strategy,
            treatment_column,
            session_column,
        )

        self.num_agents_per_session = self.check_num_agents_per_session(
            num_agents_per_session
        )
        self.agent_roles = self.check_agent_roles(agent_roles)
        self.agent_assignment = self.assign_agents_to_session()
        self.interview_script = self.check_interview_script(interview_script)

    def check_interview_script(
        self, interview_script: dict[int, Any]
    ) -> dict[int, Any]:
        """If the interview script is provided, checks if 1) each dictionary key is an integer and starts with 0, 1) the value is either a string or list of strings, and 3) the maxmimum length of the conversation is equal to self.max_conversation_length.

        Args:
            interview_script (dict[int, Any]): A list containing the interview script that the interviewer agent has to follow.

        Raises:
            ValueError: If the length of the interview script multiplied by the number of agents is not equal to the defined maximum conversation length.

        Returns:
            List[str]: A list containing the interview script that the interviewer agent has to follow.
        """

        if (
            interview_script
            and len(interview_script) * self.num_agents_per_session
            != self.max_conversation_length
        ):
            raise ValueError(
                f"Based on the length of the interview script ({len(interview_script)}) and number of agents ({self.num_agents_per_session}), the maximum length of the conversation should be {len(interview_script) * self.num_agents_per_session} and not {self.max_conversation_length}."
            )

        return interview_script

    def check_interview_script(
        self, interview_script: dict[int, Any]
    ) -> dict[int, Any]:
        """Checks the interview script.

        Checks if the values are either a string or a list of strings.

        Args:
            interview_script (dict[int, Any]): A dictionary containing the interview script that the interviewer agent has to follow.

        Raises:
            ValueError: If any value is not a string or a list of strings.

        Returns:
            List[str]: A dictionary containing the interview script that the interviewer agent has to follow.
        """
        if interview_script:
            # Check if values are either a string or a list of strings
            for key, value in interview_script.items():
                if not isinstance(value, (str, list)):
                    raise ValueError(
                        f"Value in interview_script {key} must be a string or a list of strings. Found: {type(value).__name__}"
                    )
                if isinstance(value, list) and not all(
                    isinstance(item, str) for item in value
                ):
                    raise ValueError(
                        f"All items in the list in interview_script {key} must be strings."
                    )
                if (
                    isinstance(value, list)
                    and len(value) != self.num_agents_per_session - 1
                ):
                    raise ValueError(
                        f"The number of items in the list in interview_script {key} must be equal to the number of interview subjects ({self.num_agents_per_session - 1})"
                    )

        return interview_script

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

        if self.num_sessions * (num_agents_per_session - 1) > len(
            self.agent_demographics
        ):
            raise ValueError(
                f"Total number of agents required for experiment ({self.num_sessions * (num_agents_per_session-1)}) exceed the number of profiles provided in agent_demographics ({len(self.agent_demographics)})."
            )

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

        if list(agent_roles.keys())[0] != "Interviewer":
            raise ValueError("The first role in agent_roles should be 'Interviewer'.")

        return agent_roles

    def assign_agents_to_session(self) -> dict[int, List[DemographicInfo]]:
        """Assigns agents' demographics to each session based on the given number of agents per session (minus the Interviewer agent) and agent assignment strategy.
        However, if the agent_assignment_strategy is 'manual', then assign the agents to their respective sessions based on the
        assignment defined in agent_demographics.

        Returns:
            dict[int, List[DemographicInfo]]: A dictionary mapping session IDs to a list of agent demographic information.
        """
        if self.agent_assignment_strategy == "manual":
            agent_to_session_assignment = {}
            for i, session_id in enumerate(self.session_id_list):
                session_participants = self.agent_demographics[
                    self.agent_demographics[self.session_column] == session_id
                ]

                if self.treatment_assignment_strategy == "manual":
                    session_participants_filtered = session_participants.drop(
                        columns=[self.session_column, self.treatment_column]
                    )
                else:
                    session_participants_filtered = session_participants.drop(
                        columns=[self.session_column]
                    )

                num_session_participants = len(session_participants_filtered)
                if num_session_participants != self.num_agents_per_session - 1:
                    raise ValueError(
                        f"Session {session_id} contains {num_session_participants} participants while the number of participants per session is supposed to be {self.num_agents_per_session - 1}, excluding the Interviewer."
                    )

                agent_to_session_assignment[session_id] = (
                    session_participants_filtered.to_dict(orient="records")
                )

        else:
            randomised_agent_demographics = self.agent_demographics.sample(
                frac=1
            ).reset_index(drop=True)

            num_interview_subjects_per_session = self.num_agents_per_session - 1
            agent_to_session_assignment = {}
            for i, session_id in enumerate(self.session_id_list):
                agent_to_session_assignment[session_id] = (
                    randomised_agent_demographics.iloc[
                        i
                        * num_interview_subjects_per_session : (i + 1)
                        * num_interview_subjects_per_session
                    ].to_dict(orient="records")
                )

        return agent_to_session_assignment

    def run_experiment(self, test_mode: bool = True) -> dict[str, Any]:
        """Runs an experiment based on the experimental settings defined during class initialisation. If test_mode is set to True, the first session will be selected and run.

        Args:
            test_mode (bool, optional): Indicates whether the experiment is in test mode or not.
                Defaults to True.

        Returns:
            dict[str, Any]: A dictionary containing the experiment ID and session information.
        """

        if test_mode:
            session_id_list = [self.session_id_list[0]]
        else:
            session_id_list = self.session_id_list

        experiment = {"experiment_id": self.experiment_id, "sessions": {}}
        for session_id in tqdm(session_id_list):
            session_info = {}
            session_info["session_id"] = session_id
            treatment_label = self.treatment_assignment[session_id]
            session_info["treatment"] = self.treatments[treatment_label]
            session_info["session_system_message"] = (
                generate_conversational_session_system_message(
                    experiment_context=self.experiment_context,
                    treatment=session_info["treatment"],
                )
            )
            session_info["agents_demographic"] = self.agent_assignment[session_id]
            session_info["agents"] = self.initialize_agents(session_info)
            session_info = self.run_session(
                session_info, self.interview_script, test_mode=test_mode
            )
            session_info["agents"] = [
                agent.to_dict() for agent in session_info["agents"]
            ]
            experiment["sessions"][session_id] = session_info

        self.save_experiment(experiment)

        return experiment

    def initialize_agents(
        self, session_info: dict[str, Any]
    ) -> List[ConversationalSyntheticAgent]:
        """Initializes and returns a list of ConversationalSyntheticAgent objects based on the provided session information.

        Args:
            session_info (dict[str, Any]): A dictionary containing session information, including agents' demographics, session ID, treatment, etc.

        Returns:
            List[ConversationalSyntheticAgent]: A list of initialized ConversationalSyntheticAgent objects.

        Raises:
            AssertionError: If the number of agents' demographics does not match the number of agent roles (minus Interviewer role) when initializing agents.
        """
        assert (
            len(session_info["agents_demographic"]) == len(self.agent_roles) - 1
        ), f"Number of agents' demographics ({len(session_info['agents_demographic'])}) does not match the number of agent roles ({len(self.agent_roles)-1}) when initialising agents. The number of agent demographics should be one less than the number of roles (minus the Interviewer)."

        agent_list = []
        for i in range(len(session_info["agents_demographic"]) + 1):
            if i == 0:
                agent_demographic = {}  # No demographic profile for Interviewer role
            else:
                agent_demographic = session_info["agents_demographic"][i - 1]

            agent_list.append(
                ConversationalSyntheticAgent(
                    experiment_id=self.experiment_id,
                    experiment_context=self.experiment_context,
                    session_id=session_info["session_id"],
                    demographic_info=agent_demographic,
                    role=list(self.agent_roles.keys())[i],
                    role_description=list(self.agent_roles.values())[i],
                    model_info=self.model_info,
                    treatment=session_info["treatment"],
                )
            )

        return agent_list

    def run_session(
        self,
        session_info: dict[str, Any],
        interview_script: List[str],
        test_mode: bool = False,
    ) -> dict[str, Any]:
        """Runs a session involving a conversation between multiple AI agents.

        Args:
            session_info (dict[str, Any]): A dictionary containing session information.
            interview_script (List[str]): An list containing the interview script that the interviewer agent has to follow.
            test_mode (bool, optional): A boolean indicating if the session is executed under test mode. In test mode, only the first session is executed and all responses are printed out for easy reference.

        Returns:
            dict[str, Any]: A dictionary containing the updated session information at the end of the session.
        """
        message_history = []
        conversation_length = 0
        num_agents = len(session_info["agents"])
        response = session_info["session_system_message"]
        agent_role = "system"
        message_history.append({agent_role: response})
        if test_mode:
            print({agent_role: response})

        if interview_script:
            for _, script in interview_script.items():
                if (
                    script == ""
                ):  # Interviewer is keeping silent to facilitate discussion among subjects
                    for agent in session_info["agents"][1:]:
                        response = agent.respond(question=response)
                        agent_role = agent.get_role()
                        message_history.append({agent_role: response})
                        if test_mode:
                            print({agent_role: response})

                elif isinstance(
                    script, str
                ):  # Interviewer is providing instructions for all subjects and allowing the subjects to continue the conversation
                    response = script
                    message_history.append({"Interviewer": response})
                    if test_mode:
                        print({"Interviewer": response})

                    if "Thank you for the conversation" in response:
                        break

                    for agent in session_info["agents"][1:]:
                        response = agent.respond(question=response)
                        agent_role = agent.get_role()
                        message_history.append({agent_role: response})
                        if test_mode:
                            print({agent_role: response})

                elif isinstance(
                    script, list
                ):  # Interview is providing specific instructions to each subject
                    for i, subject_script in enumerate(script):
                        message_history.append({"Interviewer": subject_script})
                        if test_mode:
                            print({"Interviewer": subject_script})

                        agent = session_info["agents"][i + 1]
                        response = agent.respond(question=subject_script)
                        agent_role = agent.get_role()
                        message_history.append({agent_role: response})
                        if test_mode:
                            print({agent_role: response})

                else:
                    raise ValueError(
                        f"The script ({script}) for each interview round should either be a string or list of strings that is equal to the number of interview subjects ({self.num_agents_per_session-1})."
                    )

            message_history.append({"system": "End"})
            if test_mode:
                print({"system": "End"})

        else:
            while (
                "Thank you for the conversation" not in response
                and conversation_length < self.max_conversation_length
            ):
                message_history.append({agent_role: response})
                if test_mode:
                    print({agent_role: response})
                    print()
                agent = session_info["agents"][conversation_length % num_agents]

                if conversation_length == 0:
                    response = agent.respond(question="Start")
                else:
                    response = agent.respond(question=response)
                agent_role = agent.get_role()
                conversation_length += 1

        message_history.append({agent_role: response})
        message_history.append({"system": "End"})
        if test_mode:
            print({agent_role: response})
            print()
            print({"system": "End"})

        session_info["message_history"] = message_history
        return session_info
