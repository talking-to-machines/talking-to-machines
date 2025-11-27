import datetime, random, warnings, concurrent.futures, re, math, copy
import pandas as pd
from collections import defaultdict
from typing import Any, List
from tqdm import tqdm
from jinja2 import Template
from talkingtomachines.generative.synthetic_subject import (
    ConversationalSyntheticSubject,
    ProfileInfo,
)
from talkingtomachines.management.treatment import (
    simple_random_assignment_session,
    complete_random_assignment_session,
    manual_assignment_session,
)
from talkingtomachines.storage.experiment import save_session

SUPPORTED_MODELS = [
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-chat-latest",
    "gpt-5-codex",
    "gpt-5-pro",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini",
    "o1",
    "o1-pro",
    "o3-pro",
    "o3",
    "o4-mini",
    "hf-inference",
]
SUPPORTED_TREATMENT_ASSIGNMENT_STRATEGIES = [
    "simple_random",
    "complete_random",
    "manual",
]
SUPPORTED_GROUP_ASSIGNMENT_STRATEGIES = [
    "random",
    "manual",
]
SUPPORTED_ROLE_ASSIGNMENT_STRATEGIES = [
    "random",
    "manual",
]
SPECIAL_ROLES = ["facilitator"]
SUPPORTED_PROMPT_TYPES = [
    "context",
    "discussion",
    "public_question",
    "repeat_public_question",
    "private_question",
    "repeat_private_question",
]


class Treatment:
    """
    A class representing a treatment with dynamically assigned attributes.

    Attributes:
        description (str): A description of the treatment. Defaults to an empty string if not provided.

    Methods:
        __init__(**kwargs):
            Initializes the Treatment instance with dynamically assigned attributes.
            If a 'description' attribute is not provided, it defaults to an empty string.
        __repr__():
            Returns a string representation of the Treatment instance, including all its attributes.
        to_dict() -> dict[str, Any]:
            Converts the Treatment object's attributes into a dictionary.
    """

    def __init__(self, **kwargs):
        # Store any attributes provided dynamically
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensure description always exists
        if not hasattr(self, "description"):
            self.description = ""

    def __repr__(self):
        return f"Treatment({self.__dict__})"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Role:
    """
    Role is a class that represents a dynamic object with attributes that can be
    set at runtime. It ensures that a `description` attribute always exists, even
    if not explicitly provided during initialization.

    Attributes:
        description (str): A string attribute that defaults to an empty string if
            not provided during initialization. Represents a description of the role.
        **kwargs: Additional attributes can be dynamically added to the instance
            during initialization.

    Methods:
        __init__(**kwargs):
            Initializes the Role instance with dynamically provided attributes.
            Ensures the `description` attribute is always present.
        __repr__():
            Returns a string representation of the Role instance, including all
            its attributes.
        to_dict() -> dict[str, Any]:
            Converts the Role object's attributes into a dictionary.
    """

    def __init__(self, **kwargs):
        # Store any attributes provided dynamically
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensure description always exists
        if not hasattr(self, "description"):
            self.description = ""

    def __repr__(self):
        return f"Role({self.__dict__})"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Constant:
    """
    A class that dynamically stores attributes provided during initialization.
    This class allows for the creation of objects with arbitrary attributes
    that are passed as keyword arguments during instantiation.

    Methods:
        __repr__():
            Returns a string representation of the object, including its attributes.
        to_dict() -> dict[str, Any]:
            Converts the object's attributes into a dictionary.
    """

    def __init__(self, **kwargs):
        # Store any attributes provided dynamically
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"Constant({self.__dict__})"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Experiment:
    """A class for constructing the base experiment class.

    Args:
        session_id (str): The unique ID of the experiment session.

    Attributes:
        session_id (str): The unique ID of the experiment session.
    """

    def __init__(self, session_id: str = ""):
        if session_id == "":
            self.session_id = self._generate_session_id()
        else:
            self.session_id = session_id

    def _generate_session_id(self) -> str:
        """Generates a unique ID for the experiment by concatenating the date and time information.

        Returns:
            str: Unique ID for the session as a base64 encoded string.
        """
        current_datetime = datetime.datetime.now()
        session_id = current_datetime.strftime("%Y%m%d_%H%M%S")

        return session_id


class AIConversationalExperiment(Experiment):
    """A class representing an AI conversational experiment. Inherits from the Experiment base class.

    This class extends the base `Experiment` class and provides additional functionality
    specific to AI conversational experiments.

    Args:
        model_info (str): The information about the LLM used in the experiment.
        temperature (float): The temperature setting that will be applied to the LLM.
        profiles (pd.DataFrame): The profile information of the subjects participating in the experiment.
        experiment_context (str, optional): The context or purpose of the experiment. Defaults to an empty string
        session_id (str, optional): The unique ID of the session. Defaults to an empty string.
        hf_inference_endpoint (str, optional): The API inference endpoint for the HuggingFace model. Defaults to an empty string.
        max_num_rounds (int, optional): The maximum number of expected rounds conducted in a session. Defaults to 10.
        treatments (dict[str, Treatment], optional): The treatments for the experiment. Defaults to an empty dictionary.
        treatment_assignment_strategy (str, optional): The strategy used for assigning treatments to subjects. Defaults to "simple_random".
        treatment_column (str, optional): The column in profiles that contains the manually assigned treatments. Defaults to an empty string.
        group_assignment_strategy (str, optional): The strategy used for assigning subjects to groups. Defaults to "random".
        group_column (str, optional): The column in profiles that contains the manually assigned groups. Defaults to an empty string.
        role_assignment_strategy (str, optional): The strategy used for assigning subjects to groups. Defaults to "random".
        role_column (str, optional): The column in profiles that contains the manually assigned roles. Defaults to an empty string.
        random_seed (int, optional): The random seed for reproducibility. Defaults to 42.
        build_profile_qna (bool, optional): Whether to build the subject profiles using a Q&A format. Defaults to True.
        build_profile_backstories (bool, optional): Whether to include backstories when building the subject profiles. Defaults to False.

    Raises:
        ValueError: If the provided model_info is not supported.
        ValueError: If the provided temperature information is not supported.
        ValueError: If the provided treatment_assignment_strategy is not supported.
        ValueError: If the provided group_assignment_strategy is not supported.
        ValueError: If the provided role_assignment_strategy is not supported.
        ValueError: If the provided profiles is an empty DataFrame or does not contain a 'ID' column.
        ValueError: If the provided max_num_rounds is lesser than 1.
        ValueError: If build_profile_qna and build_profile_backstories are both set to False.
        ValueError: If the provided random_seed is not an integer value.

    Attributes:
        model_info (str): The information about the LLM used in the experiment.
        temperature (float): The temperature setting that will be applied to the LLM.
        profiles (pd.DataFrame): The profile information of the subjects participating in the experiment.
        experiment_context (str): The context or purpose of the experiment.
        session_id (str): The unique session ID of the experiment.
        hf_inference_endpoint (str, optional): The API inference endpoint for the HuggingFace model.
        max_num_rounds (int): The maximum number of expected rounds conducted in a session.
        treatments (dict[str, Treatment]): The treatment arms for the experiment.
        treatment_assignment_strategy (str): The strategy used for assigning treatments to subjects.
        treatment_column (str, optional): The column in profiles that contains the manually assigned treatments.
        group_assignment_strategy (str, optional): The strategy used for assigning subjects to groups.
        group_column (str, optional): The column in profiles that contains the manually assigned groups.
        role_assignment_strategy (str, optional): The strategy used for assigning subjects to each role.
        role_column (str, optional): The column in profiles that contains the manually assigned roles.
        random_seed (int, optional): The random seed for reproducibility.
        build_profile_qna (bool, optional): Whether to build the subject profiles using a Q&A format.
        build_profile_backstories (bool, optional): Whether to include backstories in the subjects profiles.
    """

    def __init__(
        self,
        model_info: str,
        temperature: float,
        profiles: pd.DataFrame,
        experiment_context: str = "",
        session_id: str = "",
        hf_inference_endpoint: str = "",
        max_num_rounds: int = 10,
        treatments: dict[str, Treatment] = {},
        treatment_assignment_strategy: str = "simple_random",
        treatment_column: str = "",
        group_assignment_strategy: str = "random",
        group_column: str = "",
        role_assignment_strategy: str = "random",
        role_column: str = "",
        random_seed: int = 42,
        build_profile_qna: bool = True,
        build_profile_backstories: bool = False,
    ):
        super().__init__(
            session_id,
        )

        self.hf_inference_endpoint = hf_inference_endpoint
        self.model_info = self._check_model_info(model_info=model_info)
        self.temperature = self._check_temperature(temperature=temperature)
        self.experiment_context = experiment_context
        self.profiles = self._check_profiles(profiles=profiles)
        self.max_num_rounds = self._check_max_num_rounds(max_num_rounds=max_num_rounds)
        self.treatments = self._check_treatments(treatments=treatments)
        self.treatment_assignment_strategy = self._check_treatment_assignment_strategy(
            treatment_assignment_strategy=treatment_assignment_strategy,
            treatment_column=treatment_column,
            group_assignment_strategy=group_assignment_strategy,
        )
        self.treatment_column = treatment_column
        self.group_assignment_strategy = self._check_group_assignment_strategy(
            group_assignment_strategy=group_assignment_strategy,
            group_column=group_column,
        )
        self.group_column = group_column
        self.role_assignment_strategy = self._check_role_assignment_strategy(
            role_assignment_strategy=role_assignment_strategy, role_column=role_column
        )
        self.role_column = role_column
        self.random_seed = self._check_random_seed(random_seed)
        self.build_profile_qna = build_profile_qna
        self.build_profile_backstories = self._check_build_profile_backstories(
            build_profile_backstories, build_profile_qna
        )

    def _check_model_info(self, model_info: str) -> str:
        """Checks if the provided model_info is supported.

        Args:
            model_info (str): The model_info to be checked.

        Returns:
            str: The validated model_info.

        Raises:
            ValueError: If the provided model_info is not supported based on SUPPORTED_MODELS.
        """
        if model_info not in SUPPORTED_MODELS:
            warnings.warn(
                f"Since {model_info} is not 'hf-inference' and not one of the OpenAI instruct models ({SUPPORTED_MODELS}), '{model_info}' is assumed to be an OpenRouter.ai supported model."
            )

        if model_info == "hf-inference" and (
            self.hf_inference_endpoint == ""
            or self.hf_inference_endpoint is None
            or pd.isna(self.hf_inference_endpoint)
        ):
            raise ValueError(
                "When setting 'model_info' as 'hf-inference', a valid hf_inference_endpoint must be provided."
            )

        return model_info

    def _check_temperature(self, temperature: float) -> float:
        """
        Validates and adjusts the provided temperature value.

        This method ensures that the temperature value is within the acceptable range
        of 0 to 2 (inclusive). If the value is None, NaN, below 0, or above 2, it will
        be adjusted to a default value with a warning. If the value is not a float or
        integer, a ValueError is raised.

        Args:
            temperature (float): The temperature value to validate.

        Returns:
            float: A valid temperature value within the range [0, 2].

        Raises:
            ValueError: If the temperature is not a float or integer.

        Warnings:
            - If the temperature is None, it defaults to 0.
            - If the temperature is NaN, it defaults to 0.
            - If the temperature is below 0, it is set to 0.
            - If the temperature is above 2, it is set to 2.
        """
        if temperature is None:
            warnings.warn(
                "The temperature field contains a None value; defaulting to 0."
            )
            return 0

        # Ensure that temperature is a number (float or int)
        if isinstance(temperature, (int, float)):
            if math.isnan(temperature):
                warnings.warn(
                    "The temperature field contains a NaN value; defaulting to 0."
                )
                return 0

            # If temperature is below 0, warn and set to 0
            if temperature < 0:
                warnings.warn(
                    f"Provided temperature {temperature} is below 0. Setting temperature to 0..."
                )
                return 0

            # If temperature is above 2, warn and set to 2
            if temperature > 2:
                warnings.warn(
                    f"Provided temperature {temperature} is greater than 2. Setting temperature to 2..."
                )
                return 2

            # Otherwise, return the provided temperature as is
            return temperature

        else:
            raise ValueError("The temperature field must be a float or integer value.")

    def _check_profiles(self, profiles: pd.DataFrame) -> pd.DataFrame:
        """Checks to ensure that provided profiles is not empty and contains a ID column.

        Args:
            profiles (pd.DataFrame): The subject profiles to be checked.

        Returns:
            pd.DataFrame: The validated profiles.

        Raises:
            ValueError: If the provided profiles is an empty dataframe or if it does not contain an ID column.
        """
        if profiles.empty:
            raise ValueError("profiles DataFrame cannot be empty.")

        if "ID" not in profiles.columns:
            raise ValueError("profiles DataFrame should contain an 'ID' column.")

        return profiles

    def _check_max_num_rounds(self, max_num_rounds: int) -> int:
        """
        Validates and processes the `max_num_rounds` parameter.
        This method ensures that the `max_num_rounds` parameter is a valid integer
        greater than or equal to 1. If the parameter is `None`, it defaults to 10
        with a warning. If the parameter is a NaN value, it also defaults to 10
        with a warning. If the parameter is invalid, a `ValueError` is raised.

        Args:
            max_num_rounds (int): The maximum number of rounds to validate.

        Returns:
            int: A valid integer value for `max_num_rounds`.

        Raises:
            ValueError: If `max_num_rounds` is less than 1 or not a numeric type.
        """
        if max_num_rounds is None:
            warnings.warn(
                "The max_num_rounds field contains a None value; defaulting to 10."
            )
            return 10

        if isinstance(max_num_rounds, (int, float)):
            if math.isnan(max_num_rounds):
                warnings.warn(
                    "The max_num_rounds field contains a NaN value; defaulting to 10."
                )
                return 10

            if max_num_rounds < 1:
                raise ValueError(
                    "Invalid value for max_num_rounds. Please ensure that max_num_rounds is an integer greater than or equal to 1."
                )

            return max_num_rounds

        else:
            raise ValueError(
                "The max_num_rounds field must be either an integer or float value."
            )

    def _check_treatments(
        self, treatments: dict[str, Treatment]
    ) -> dict[str, Treatment]:
        """
        Validates the treatments dictionary to ensure each Treatment object has a valid 'description' attribute.

        Args:
            treatments (dict[str, Treatment]): A dictionary where keys are treatment labels (strings) and values
                are Treatment objects.

        Returns:
            dict[str, Treatment]: The validated treatments dictionary.

        Raises:
            ValueError: If a Treatment object is missing the 'description' attribute or if the 'description'
                is not a string.
        """
        for label, treatment in treatments.items():
            if not hasattr(treatment, "description"):
                raise ValueError(
                    f"Treatment '{label}' is missing a required attribute 'description'."
                )

            if not isinstance(treatment.description, str):
                raise ValueError(
                    f"Invalid treatment description: {treatment.description}. Treatment descriptions should be strings."
                )

        return treatments

    def _check_treatment_assignment_strategy(
        self,
        treatment_assignment_strategy: str,
        treatment_column: str,
        group_assignment_strategy: str,
    ) -> str:
        if (
            treatment_assignment_strategy
            not in SUPPORTED_TREATMENT_ASSIGNMENT_STRATEGIES
        ):
            raise ValueError(
                f"Unsupported treatment_assignment_strategy: {treatment_assignment_strategy}. Supported strategies are: {SUPPORTED_TREATMENT_ASSIGNMENT_STRATEGIES}."
            )

        # Check that treatment_column and group_column can be found in profiles when using manual treatment assignment
        if treatment_assignment_strategy == "manual":
            if treatment_column == "" or treatment_column not in self.profiles.columns:
                raise ValueError(
                    f"The argument 'treatment_column' cannot be an empty string and must be one of the columns in profiles when using manual treatment assignment."
                )

            if group_assignment_strategy != "manual":
                raise ValueError(
                    f"When using manual treatment assignment, group assignment strategy must also be 'manual' to ensure that subjects in the same group experienced the same treatment arm."
                )

        return treatment_assignment_strategy

    def _check_group_assignment_strategy(
        self, group_assignment_strategy: str, group_column: str
    ) -> str:
        """Checks if the provided group_assignment_strategy is supported.

        Args:
            group_assignment_strategy (str): The group_assignment_strategy to be checked.
            group_column (str): The column name containing the group information when using manual assignment strategy.

        Returns:
            str: The validated group_assignment_strategy.

        Raises:
            ValueError: If the provided group_assignment_strategy is not supported.
            ValueError: If group_column is an empty string or not one of the columns in profiles when using the manual group assignment strategy.
        """
        if group_assignment_strategy not in SUPPORTED_GROUP_ASSIGNMENT_STRATEGIES:
            raise ValueError(
                f"Unsupported group_assignment_strategy: {group_assignment_strategy}. Supported strategies are: {SUPPORTED_GROUP_ASSIGNMENT_STRATEGIES}."
            )

        # Check that group_column can be found in profiles when using manual group assignment
        if group_assignment_strategy == "manual":
            if group_column == "" or group_column not in self.profiles.columns:
                raise ValueError(
                    f"The argument 'group_column' cannot be an empty string and must be one of the columns in profiles when performing manual group assignment."
                )

        return group_assignment_strategy

    def _check_role_assignment_strategy(
        self, role_assignment_strategy: str, role_column: str
    ) -> str:
        """Checks if the provided role_assignment_strategy is supported.

        Args:
            role_assignment_strategy (str): The role_assignment_strategy to be checked.
            role_column (str): The column name containing the role information when using manual assignment strategy.

        Returns:
            str: The validated role_assignment_strategy.

        Raises:
            ValueError: If the provided role_assignment_strategy is not supported.
            ValueError: If role_column is an empty string or not one of the columns in profiles when using the manual role assignment strategy.
        """
        if role_assignment_strategy not in SUPPORTED_ROLE_ASSIGNMENT_STRATEGIES:
            raise ValueError(
                f"Unsupported role_assignment_strategy: {role_assignment_strategy}. Supported strategies are: {SUPPORTED_ROLE_ASSIGNMENT_STRATEGIES}."
            )

        # Check that role_column can be found in profiles when using manual role assignment
        if role_assignment_strategy == "manual":
            if role_column == "" or role_column not in self.profiles.columns:
                raise ValueError(
                    f"The argument 'role_column' cannot be an empty string and must be one of the columns in profiles when performing manual role assignment."
                )

        return role_assignment_strategy

    def _check_random_seed(self, random_seed: int) -> int:
        """
        Validates and returns a random seed value.
        This method checks the provided `random_seed` value and ensures it is a valid
        integer or float. If the value is `None` or `NaN`, it defaults to 42 and issues
        a warning. If the value is not an integer or float, it raises a ValueError.

        Args:
            random_seed (int): The random seed value to validate.

        Returns:
            int: A valid random seed value.

        Raises:
            ValueError: If `random_seed` is not an integer or float.

        Warnings:
            UserWarning: If `random_seed` is `None` or `NaN`, a warning is issued and
            the value defaults to 42.
        """
        if random_seed is None:
            warnings.warn(
                "The random_seed field contains a None value; defaulting to 42."
            )
            return 42

        if isinstance(random_seed, (int, float)):
            if math.isnan(random_seed):
                warnings.warn(
                    "The random_seed field contains a NaN value; defaulting to 42."
                )
                return 42

            return random_seed

        else:
            raise ValueError(
                "The random_seed field must be either an integer or float value."
            )

    def _check_build_profile_backstories(
        self, build_profile_backstories: bool, build_profile_qna: bool
    ) -> bool:
        """
        Checks the configuration for building profile backstories and warns if both
        `build_profile_backstories` and `build_profile_qna` are set to False.

        Args:
            build_profile_backstories (bool): Indicates whether to build profile backstories.
            build_profile_qna (bool): Indicates whether to build profile Q&A.

        Returns:
            bool: The value of `build_profile_backstories`.

        Warns:
            UserWarning: If both `build_profile_backstories` and `build_profile_qna`
            are set to False, indicating that subjects in the experiment will not
            have any profile information.
        """
        if not build_profile_qna and not build_profile_backstories:
            warnings.warn(
                "Both build_profile_qna and build_profile_backstories are set to False. The subjects in the experiment will not have any profile information."
            )

        return build_profile_backstories


class AItoAIConversationalExperiment(AIConversationalExperiment):
    """A class representing an AI-to-AI conversational experiment. Inherits from the AIConversationalExperiment class.

    This class extends the `AIConversationalExperiment` class and provides additional functionality
    specific to AI-to-AI conversational experiments.

    Args:
        model_info (str): The information about the LLM used in the experiment.
        temperature (float): The temperature setting that will be applied to the LLM.
        profiles (pd.DataFrame): The profile information of the subjects participating in the experiment.
        roles (dict[str, Role]): Dictionary mapping roles to their Role objects.
        num_subjects_per_group (int, optional): Number of subjects per group. Defaults to 2.
        num_groups (int, optional): Number of groups. Defaults to 1.
        experiment_context (str, optional): The context or purpose of the experiment. Defaults to an empty string.
        session_id (str, optional): The unique session ID of the experiment. Defaults to an empty string.
        hf_inference_endpoint (str, optional): The API inference endpoint for the HuggingFace model. Defaults to an empty string.
        max_num_rounds (int, optional): The maximum number of expected rounds conducted in a session. Defaults to 10.
        treatments (dict[str, Treatment], optional): The treatments for the experiment. Defaults to an empty dictionary.
        treatment_assignment_strategy (str, optional): The strategy used for assigning treatments to subjects. Defaults to "simple_random".
        treatment_column (str, optional): The column in profiles that contains the manually assigned treatments. Defaults to an empty string.
        group_assignment_strategy (str, optional): The strategy used for assigning subjects to groups. Defaults to "random".
        group_column (str, optional): The column in profiles that contains the manually assigned groups. Defaults to an empty string.
        role_assignment_strategy (str, optional): The strategy used for assigning subjects to different roles. Defaults to "random".
        role_column (str, optional): The column in profiles that contains the manually assigned role. Defaults to an empty string.
        random_seed (int, optional): The random seed for reproducibility. Defaults to 42.
        build_profile_qna (bool, optional): Whether to build the subject profiles using a Q&A format. Defaults to True.
        build_profile_backstories (bool, optional): Whether to include backstories when building the subject profiles. Defaults to False.

    Raises:
        ValueError: If the provided model_info is not supported.
        ValueError: If the provided temperature information is not supported.
        ValueError: If the provided treatment_assignment_strategy is not supported.
        ValueError: If the provided group_assignment_strategy is not supported.
        ValueError: If the provided role_assignment_strategy is not supported.
        ValueError: If the provided profiles is an empty DataFrame or does not contain a 'ID' column.
        ValueError: If the provided max_num_rounds is lesser than 1.
        ValueError: If the provided num_groups is not valid.
        ValueError: If the provided num_subjects_per_group is less than 2 or will exceed the total number of profiles provided.
        ValueError: If the provided number of roles is not equal to num_subjects_per_group.
        ValueError: If the number of roles defined does not match the number of subjects assigned to each group.

    Attributes:
        model_info (str): The information about the LLM used in the experiment.
        temperature (float): The temperature setting that will be applied to the LLM.
        profiles (pd.DataFrame): The profile information of the subjects participating in the experiment.
        roles (dict[str, Role]): The roles assigned to subjects.
        num_subjects_per_group (int): The number of subjects per group.
        num_groups (int): The number of groups in the experiment.
        experiment_context (str): The context or purpose of the experiment.
        session_id (str): The unique session ID of the experiment.
        hf_inference_endpoint (str, optional): The API inference endpoint for the HuggingFace model.
        max_num_rounds (int): The maximum number of expected rounds conducted in a session.
        treatments (dict[str, Treatment]): The treatments for the experiment.
        treatment_assignment_strategy (str): The strategy used for assigning treatments to subjects.
        treatment_column (str, optional): The column in profiles that contains the manually assigned treatments.
        group_assignment_strategy (str, optional): The strategy used for assigning subjects to groups.
        group_column (str, optional): The column in profiles that contains the manually assigned groups.
        role_assignment_strategy (str, optional): The strategy used for assigning subjects to each role.
        role_column (str, optional): The column in profiles that contains the manually assigned roles.
        random_seed (int, optional): The random seed for reproducibility.
        build_profile_qna (bool): Whether to build the subject profiles using a Q&A format. Defaults to True.
        build_profile_backstories (bool): Whether to include backstories when building the subject profiles. Defaults to False.
        group_id_list (list): A list of group IDs generated based on the number of groups in each session.
        treatment_assignment (dict[Any, str]): A dictionary mapping group IDs to treatment arms.
        group_assignment (dict[Any, list[ProfileInfo]]): A dictionary mapping group IDs to a list of profile information.
        role_assignment (dict[Any, str]): A dictionary mapping subject IDs to a specified role.
    """

    def __init__(
        self,
        model_info: str,
        temperature: float,
        profiles: pd.DataFrame,
        roles: dict[str, Role],
        num_subjects_per_group: int = 1,
        num_groups: int = 1,
        experiment_context: str = "",
        session_id: str = "",
        hf_inference_endpoint: str = "",
        max_num_rounds: int = 10,
        treatments: dict[str, Treatment] = {},
        treatment_assignment_strategy: str = "simple_random",
        treatment_column: str = "",
        group_assignment_strategy: str = "random",
        group_column: str = "",
        role_assignment_strategy: str = "random",
        role_column: str = "",
        random_seed: int = 42,
        build_profile_qna: bool = True,
        build_profile_backstories: bool = False,
    ):
        super().__init__(
            model_info,
            temperature,
            profiles,
            experiment_context,
            session_id,
            hf_inference_endpoint,
            max_num_rounds,
            treatments,
            treatment_assignment_strategy,
            treatment_column,
            group_assignment_strategy,
            group_column,
            role_assignment_strategy,
            role_column,
            random_seed,
            build_profile_qna,
            build_profile_backstories,
        )

        self.roles = roles
        self.num_groups = self._check_num_groups(num_groups=num_groups)
        self.num_subjects_per_group = self._check_num_subjects_per_group(
            num_subjects_per_group=num_subjects_per_group
        )
        self.group_id_list = self._generate_group_id_list()
        self.treatment_assignment = self._assign_treatment(random_seed=self.random_seed)
        if self.treatment_assignment_strategy == "manual":
            self._check_manually_assigned_treatments()
        self.group_assignment = self._assign_group(random_seed=self.random_seed)
        self.role_assignment = self._assign_role(random_seed=self.random_seed)
        if self.role_assignment_strategy == "manual":
            self._check_manually_assigned_roles()

    def _check_num_subjects_per_group(self, num_subjects_per_group: int) -> int:
        """
        Validates and processes the `num_subjects_per_group` parameter for an experiment.

        Args:
            num_subjects_per_group (int): The number of subjects per group. Can be an integer or float.
                If None, defaults to 2. If NaN, also defaults to 2.

        Returns:
            int: The validated and processed number of subjects per group.

        Raises:
            ValueError: If `num_subjects_per_group` is less than 2.
            ValueError: If the total number of subjects required for the experiment
                (calculated as `self.num_groups * num_subjects_per_group`) does not match
                the number of profiles provided (`len(self.profiles)`).
            ValueError: If `num_subjects_per_group` is not a float or integer.

        Warnings:
            - If `num_subjects_per_group` is None, a warning is issued, and the value defaults to 2.
            - If `num_subjects_per_group` is NaN, a warning is issued, and the value defaults to 2.
        """
        if num_subjects_per_group is None:
            warnings.warn(
                "The num_subjects_per_group field contains a None value; defaulting to 2."
            )
            return 2

        # Ensure that num_subjects_per_group is a number (float or int)
        if isinstance(num_subjects_per_group, (int, float)):
            if math.isnan(num_subjects_per_group):
                warnings.warn(
                    "The num_subjects_per_group field contains a NaN value; defaulting to 2."
                )
                return 2

            # Check if number of subjects per group is 2 or more
            if num_subjects_per_group < 2:
                raise ValueError(
                    f"Invalid num_subjects_per_group: {num_subjects_per_group}. For AI-AI conversation-based experiments, num_subjects_per_group should be an integer that is equal to or greater than 2."
                )

            # Check if number of subjects per group multipled by the number of groups is less than the number of profiles provided
            if self.num_groups * num_subjects_per_group != len(self.profiles):
                raise ValueError(
                    f"Total number of subjects required for experiment ({self.num_groups * num_subjects_per_group}) does not match with the number of profiles provided ({len(self.profiles)})."
                )

            # Otherwise, return the provided num_subjects_per_group as is
            return num_subjects_per_group

        else:
            raise ValueError(
                "The num_subjects_per_group field must be a float or integer value."
            )

    def _check_num_groups(self, num_groups: int) -> int:
        """
        Validates and processes the `num_groups` parameter.
        This method ensures that the `num_groups` parameter is a valid number
        (integer or float) and meets the required conditions. If `num_groups`
        is `None` or `NaN`, it defaults to 1. If `num_groups` is less than 1,
        a `ValueError` is raised. If the input is valid, the method returns
        the provided value.

        Args:
            num_groups (int): The number of groups to validate.

        Returns:
            int: The validated number of groups.

        Raises:
            ValueError: If `num_groups` is not a float or integer, or if it is
                        less than 1.
        """
        if num_groups is None:
            warnings.warn(
                "The num_groups field contains a None value; defaulting to 1."
            )
            return 1

        # Ensure that num_groups is a number (float or int)
        if isinstance(num_groups, (int, float)):
            if math.isnan(num_groups):
                warnings.warn(
                    "The num_groups field contains a NaN value; defaulting to 1."
                )
                return 1

            # Check if number of groups is 1 or more
            if num_groups < 1:
                raise ValueError(
                    f"Invalid value for num_groups: {num_groups}. num_groups should be an integer that is equal to or greater than 1."
                )

            # Otherwise, return the provided num_subjects_per_group as is
            return num_groups

        else:
            raise ValueError("The num_groups field must be a float or integer value.")

    def _generate_group_id_list(self) -> List[Any]:
        """Generates a list of group IDs.

        If the group assignment strategy is set to 'manual',
        the function returns a list of unique group IDs from the profiles DataFrame.
        Otherwise, it returns a list of sequential integers starting from 0 up to the number of sessions - 1.

        Returns:
            List[Any]: A list of group IDs. If the assignment strategies are manual, the list contains unique group IDs
                from the group_column in the profiles DataFrame. Otherwise, it contains sequential integers starting from 0.
        """
        if self.group_assignment_strategy == "manual":
            return list(self.profiles[self.group_column].unique())
        else:
            return list(range(self.num_groups))

    def _assign_treatment(self, random_seed: int) -> dict[int, str]:
        """Assign treatments to groups based on the specified treatment assignment strategy.

        Args:
            random_seed (int): The random seed for reproducibility.

        Returns:
            dict[int, str]: A dictionary where the keys represent group IDs and the values represent the assigned treatment labels.
        """
        if self.treatment_assignment_strategy == "simple_random":
            treatment_labels = list(self.treatments.keys())
            return simple_random_assignment_session(
                treatment_labels=treatment_labels,
                group_id_list=self.group_id_list,
                random_seed=random_seed,
            )

        elif self.treatment_assignment_strategy == "complete_random":
            treatment_labels = list(self.treatments.keys())
            return complete_random_assignment_session(
                treatment_labels=treatment_labels,
                group_id_list=self.group_id_list,
                random_seed=random_seed,
            )

        elif self.treatment_assignment_strategy == "manual":
            return manual_assignment_session(
                profiles=self.profiles,
                treatment_column=self.treatment_column,
                group_column=self.group_column,
                group_id_list=self.group_id_list,
            )

        else:
            raise ValueError(
                f"Invalid treatment_assignment_strategy: {self.treatment_assignment_strategy}. Supported strategies are: {SUPPORTED_TREATMENT_ASSIGNMENT_STRATEGIES}."
            )

    def _check_manually_assigned_treatments(self) -> None:
        """Checks if the manually defined treatments align with the treatment labels provided in self.treatments.

        Raises:
            ValueError: If the manually defined treatments do not align with the treatment labels provided in self.treatments.
        """
        treatment_label_set = set(self.treatments.keys())
        manual_defined_treatments = set(self.treatment_assignment.values())

        if not treatment_label_set.issuperset(manual_defined_treatments):
            raise ValueError(
                f"The treatment labels defined in the treatments worksheet ({list[treatment_label_set]}) is not a superset of the manually defined treatments in the profiles worksheet ({list[manual_defined_treatments]})."
            )
        else:
            pass

    def _assign_group(self, random_seed: int) -> dict[int, List[ProfileInfo]]:
        """Assigns profiles to each group based on the given number of subjects per group and group assignment strategy.
        However, if the group_assignment_strategy is 'manual', then assign the subjects to their respective groups based on the
        assignment defined in profiles.

        Args:
            random_seed (int): The random seed for reproducibility.

        Returns:
            dict[int, List[ProfileInfo]]: A dictionary mapping group IDs to a list of profile information.
        """
        if self.group_assignment_strategy == "manual":
            group_assignment = {}
            for i, group_id in enumerate(self.group_id_list):
                group_subjects = self.profiles[
                    self.profiles[self.group_column] == group_id
                ]

                num_subjects_in_group = len(group_subjects)
                if num_subjects_in_group != self.num_subjects_per_group:
                    raise ValueError(
                        f"Group {group_id} contains {num_subjects_in_group} subjects while the number of subjects per group is supposed to be {self.num_subjects_per_group}"
                    )

                group_assignment[group_id] = group_subjects.to_dict(orient="records")

        else:
            randomised_profiles = self.profiles.sample(
                frac=1, random_state=random_seed
            ).reset_index(drop=True)

            group_assignment = {}
            for i, group_id in enumerate(self.group_id_list):
                group_assignment[group_id] = randomised_profiles.iloc[
                    i
                    * self.num_subjects_per_group : (i + 1)
                    * self.num_subjects_per_group
                ].to_dict(orient="records")

        return group_assignment

    def _assign_role(self, random_seed: int) -> dict[int, str]:
        """Assigns roles to subjects based on the specified role assignment strategy.

        Args:
            random_seed (int): The seed value for randomization when using the "random" role assignment strategy.

        Returns:
            dict[int, str]: A dictionary mapping subject IDs to their assigned roles.

        Raises:
            ValueError: If the number of defined roles does not match the number of subjects
                        assigned to a group when using the "random" role assignment strategy.
        """
        if self.role_assignment_strategy == "manual":
            role_assignment = self.profiles.set_index("ID")[self.role_column].to_dict()

        else:
            random.seed(random_seed)
            role_assignment = {}
            role_labels = list(self.roles.keys())
            for group_id, group_subjects in self.group_assignment.items():
                num_subjects_in_group = len(group_subjects)

                if len(role_labels) == num_subjects_in_group:
                    randomized_roles = random.sample(role_labels, num_subjects_in_group)

                else:
                    raise ValueError(
                        f"Number of roles defined ({len(role_labels)}) does not match the number of subjects ({num_subjects_in_group}) assigned to Group {group_id}."
                    )

                role_assignment.update(
                    {
                        subject["ID"]: role
                        for subject, role in zip(group_subjects, randomized_roles)
                    }
                )

        return role_assignment

    def _check_manually_assigned_roles(self) -> None:
        """Validates that all manually assigned roles are defined in the roles worksheet.

        This method checks whether the roles manually assigned in the `role_assignment`
        dictionary are a subset of the roles defined in the `roles` dictionary.
        If any manually assigned role is not present in the defined roles, a
        `ValueError` is raised.

        Raises:
            ValueError: If the roles defined in the `roles` worksheet are not a
            superset of the manually defined roles in the `profiles` worksheet.
        """
        role_label_set = set(self.roles.keys())
        manual_defined_roles = set(self.role_assignment.values())

        if not role_label_set.issuperset(manual_defined_roles):
            raise ValueError(
                f"The roles defined in the roles worksheet ({list[role_label_set]}) is not a superset of the manually defined roles in the profiles worksheet ({list[manual_defined_roles]})."
            )
        else:
            pass

    def run_session(
        self,
        test_mode: bool = True,
        version: int = 1,
        save_results_as_csv: bool = False,
    ) -> dict[str, Any]:
        """Runs a session based on the experimental settings defined during class initialisation.
        If test_mode is set to True, only a random group for each treatment arm will be selected and run sequentially; otherwise, groups are run in parallel.

        Args:
            test_mode (bool, optional): Indicates whether the session is run in test mode or not.
                Defaults to True.
            version (int, optional): Indicates the version of the session.
                Defaults to 1.
            save_results_as_csv (bool, optional): Indicates whether the results of the session will be saved as CSV format.
                Defaults to False

        Returns:
            dict[str, Any]: A dictionary containing the group ID and group information.
        """
        if test_mode:  # Run one random group from each treatment group
            group_id_list = []
            for treatment in list(self.treatments.keys()):
                matching_groups = [
                    group_id
                    for group_id, assigned_treatment in self.treatment_assignment.items()
                    if assigned_treatment == treatment
                ]
                if matching_groups:
                    group_id_list.append(random.choice(matching_groups))

        else:
            group_id_list = self.group_id_list

        session = {
            "session_id": f"{self.session_id}_{version}",
            "groups": {},
        }

        # Helper function to process a single group.
        def process_group(group_id: Any) -> tuple[Any, dict]:
            group_info = {}
            group_info["group_id"] = group_id
            group_info["random_seed"] = self.random_seed
            group_info["treatment_label"] = self.treatment_assignment[group_id]
            group_info["experiment_context"] = self.experiment_context
            group_info["profiles"] = self.group_assignment[group_id]
            group_subject_ids = [profile["ID"] for profile in group_info["profiles"]]
            group_info["roles"] = {
                subject_id: assigned_role
                for subject_id, assigned_role in self.role_assignment.items()
                if subject_id in group_subject_ids
            }
            group_info["subjects"] = self._initialize_subjects(group_info)
            group_info = self._run_group(group_info, test_mode=test_mode)
            updated_group_subjects = {}
            for subject_role, subject in group_info["subjects"].items():
                updated_group_subjects[subject_role] = subject.to_dict()
            group_info["subjects"] = updated_group_subjects

            return group_id, group_info

        if test_mode:
            # Sequentially process sessions in test mode.
            for group_id in tqdm(group_id_list):
                group_id, group_info = process_group(group_id)
                session["groups"][group_id] = group_info
        else:
            # Process sessions in parallel using ThreadPoolExecutor.
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_sid = {
                    executor.submit(process_group, group_id): group_id
                    for group_id in group_id_list
                }
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_sid),
                    total=len(future_to_sid),
                ):
                    group_id, group_info = future.result()
                    session["groups"][group_id] = group_info

        self._save_session(session, save_results_as_csv=save_results_as_csv)

        return session

    def _initialize_subjects(
        self, group_info: dict[str, Any]
    ) -> dict[str, ConversationalSyntheticSubject]:
        """Initializes and returns a dictionary of ConversationalSyntheticSubject objects based on the provided group information.

        Args:
            group_info (dict[str, Any]): A dictionary containing group information, including subjects' profile, role, group ID, treatment, etc.

        Returns:
            dict[str, ConversationalSyntheticSubject]: A dictionary where the key indicates the role and the value is an initialized ConversationalSyntheticSubject objects.

        Raises:
            AssertionError: If the number of profiles does not match the number of roles when initializing subjects.
        """
        assert len(group_info["profiles"]) == len(
            group_info["roles"]
        ), "Number of profiles does not match the number of roles when initialising subjects."
        subject_dict = {}
        for i in range(len(group_info["profiles"])):
            subject_id = group_info["profiles"][i]["ID"]
            role_label = group_info["roles"][subject_id]
            subject_dict[role_label] = ConversationalSyntheticSubject(
                session_id=self.session_id,
                experiment_context=self.experiment_context,
                group_id=group_info["group_id"],
                profile_info={
                    k: v
                    for k, v in group_info["profiles"][i].items()
                    if k
                    not in [self.treatment_column, self.role_column, self.group_column]
                },
                model_info=self.model_info,
                temperature=self.temperature,
                build_profile_qna=self.build_profile_qna,
                build_profile_backstories=self.build_profile_backstories,
                hf_inference_endpoint=self.hf_inference_endpoint,
                role_label=role_label,
                role=self.roles[role_label],
                treatment=self.treatments[group_info["treatment_label"]],
            )

        return subject_dict

    def _run_group(
        self, group_info: dict[str, Any], test_mode: bool = False
    ) -> dict[str, Any]:
        """
        Executes a conversation session for a group of subjects, simulating a dialogue
        between a system and multiple subjects. The session continues until either the
        "end_session" condition is met or the maximum number of rounds is reached.

        Args:
            group_info (dict[str, Any]): A dictionary containing information about the group,
                including subjects and the initial system message. Expected keys:
                - "subjects": A dictionary of subject objects, where each subject has a
                  `role`, `role_label`, and `profile_info`.
                - "experiment_context": The initial message about the experimental context.
            test_mode (bool, optional): If True, prints the message history for debugging
                purposes. Defaults to False.

        Returns:
            dict[str, Any]: The updated `group_info` dictionary with the conversation
            message history added under the key "message_history".
        """
        session_message_history = []
        subject_message_history = {}
        round_num = 0
        num_subjects = len(group_info["subjects"])
        subject_list = list(group_info["subjects"].values())
        response = group_info["experiment_context"]
        role = "system"

        while (
            not re.compile(r"(?<!\w)end_session(?!\w)", re.IGNORECASE).search(response)
            and round_num < self.max_num_rounds
        ):
            if role == "system" and round_num == 0:
                message_dict = {
                    role: response,
                    "round_id": round_num,
                }
                for subject in subject_list:
                    subject_message_history[subject.role_label] = [message_dict]

            else:
                message_dict = {
                    role: response,
                    "subject_id": subject_id,
                    "round_id": round_num,
                }
                for subject in subject_list:
                    subject_message_history[subject.role_label].append(message_dict)

            session_message_history.append(message_dict)

            if test_mode:
                print(message_dict)
                print()

            # If no interview script is provided, the sequence of conversation will follow the sequence of subjects defined in self._initialize_subjects
            subject = subject_list[round_num % num_subjects]
            subject_id = subject.profile_info.get("ID", "")
            role = subject.role_label
            response = subject.respond(
                latest_message_history=subject_message_history[role]
            )
            subject_message_history[role] = []
            round_num += 1

        message_dict = {
            role: response,
            "subject_id": subject_id,
            "round_id": round_num,
        }
        session_message_history.append(message_dict)
        session_message_history.append({"system": "end_session"})
        if test_mode:
            print(message_dict)
            print()
            print({"system": "end_session"})

        group_info["message_history"] = session_message_history
        return group_info

    def _save_session(
        self, session: dict[int, Any], save_results_as_csv: bool = False
    ) -> None:
        """Save the session data.

        Args:
            session (dict[int, Any]): The session data to be saved.
            save_results_as_csv (bool, optional): Indicates whether the results of the session will be saved as CSV format.
                Defaults to False

        Returns:
            None
        """
        save_session(session, save_results_as_csv)


class AItoAIInterviewExperiment(AItoAIConversationalExperiment):
    """A class representing an AI-to-AI interview experiment. Inherits from the AItoAIConversationalExperiment class.

    This class extends the `AItoAIConversationalExperiment` class and provides additional functionality
    specific to AI-to-AI interview experiments.

    Args:
        model_info (str): The information about the LLM used in the experiment.
        temperature (float): The temperature setting that will be applied to the LLM.
        profiles (pd.DataFrame): The profile information of the subjects participating in the experiment.
        roles (dict[str, Role]): Dictionary mapping of roles to their Role objects.
        num_subjects_per_group (int, optional): Number of subjects per group. Defaults to 1.
        num_groups (int, optional): Number of groups. Defaults to 1.
        experiment_context (str, optional): The context or purpose of the experiment. Defaults to an empty string.
        session_id (str, optional): The unique session ID of the experiment. Defaults to an empty string.
        hf_inference_endpoint (str, optional): The API inference endpoint for the HuggingFace model. Defaults to an empty string.
        max_num_rounds (int, optional): The maximum number of expected rounds conducted in a session. Defaults to 10.
        treatments (dict[str, Treatment], optional): The treatments for the experiment. Defaults to an empty dictionary.
        treatment_assignment_strategy (str, optional): The strategy used for assigning treatments to subjects. Defaults to "simple_random".
        treatment_column (str, optional): The column in profiles that contains the manually assigned treatments. Defaults to an empty string.
        group_assignment_strategy (str, optional): The strategy used for assigning subjects to groups. Defaults to "random".
        group_column (str, optional): The column in profiles that contains the manually assigned groups. Defaults to an empty string.
        role_assignment_strategy (str, optional): The strategy used for assigning subjects to differen roles. Defaults to "random".
        role_column (str, optional): The column in profiles that contains the manually assigned role. Defaults to an empty string.
        random_seed (int, optional): The random seed for reproducibility. Defaults to 42.
        build_profile_qna (bool, optional): Whether to build the subject profiles using a Q&A format. Defaults to True.
        build_profile_backstories (bool, optional): Whether to include backstories when building the subject profiles. Defaults to False.
        prompts (List[dict[str, str]], optional): An optional dictionary containing the interview script that the facilitator has to follow.
        constants (dict, optional): An optional dictionary containing constants to populate the interview prompts.

    Raises:
        ValueError: If the provided model_info is not supported.
        ValueError: If the provided temperature information is not supported.
        ValueError: If the provided treatment_assignment_strategy is not supported.
        ValueError: If the provided group_assignment_strategy is not supported.
        ValueError: If the provided role_assignment_strategy is not supported.
        ValueError: If the provided profiles is an empty DataFrame or does not contain a 'ID' column.
        ValueError: If the provided max_num_rounds is lesser than 1.
        ValueError: If the provided num_groups is not valid.
        ValueError: If the provided num_subjects_per_group is less than 1 or will exceed the total number of profile information provided.
        ValueError: If the provided number of user-defined roles is not equal to num_subjects_per_group.
        ValueError: If the number of user-defined roles does not match the number of subjects assigned to each group.
        ValueError: If the format of the prompts does not fit with the expected format.

    Attributes:
        model_info (str): The information about the LLM used in the experiment.
        temperature (float): The temperature setting that will be applied to the LLM.
        profiles (pd.DataFrame): The profile information of the subjects participating in the experiment.
        roles (dict[str, Role]): The roles assigned to subjects.
        num_subjects_per_group (int): The number of subjects per group.
        num_groups (int): The number of groups in the experiment.
        experiment_context (str): The context or purpose of the experiment.
        session_id (str): The unique session ID of the experiment.
        hf_inference_endpoint (str, optional): The API inference endpoint for the HuggingFace model.
        max_num_rounds (int): The maximum number of expected rounds conducted in a session.
        treatments (dict[str, Treatment]): The treatments for the experiment.
        treatment_assignment_strategy (str): The strategy used for assigning treatments to subjects.
        treatment_column (str, optional): The column in profiles that contains the manually assigned treatments.
        group_assignment_strategy (str, optional): The strategy used for assigning subjects to groups.
        group_column (str, optional): The column in profiles that contains the manually assigned groups.
        role_assignment_strategy (str, optional): The strategy used for assigning subjects to different roles.
        role_column (str, optional): The column in profiles that contains the manually assigned roles.
        random_seed (int, optional): The random seed for reproducibility.
        build_profile_qna (bool): Whether to build the subject profiles using a Q&A format. Defaults to True.
        build_profile_backstories (bool): Whether to include backstories when building the subject profiles. Defaults to False.
        group_id_list (list): A list of group IDs generated based on the number of groups in each session.
        treatment_assignment (dict[Any, str]): A dictionary mapping group IDs to treatment labels.
        group_assignment (dict[Any, list[ProfileInfo]]): A dictionary mapping group IDs to a list of profile information.
        role_assignment (dict[Any, str]): A dictionary mapping subject IDs to a specified role.
        prompts (List[dict[str, str]], optional): An optional dictionary containing the interview prompts that the facilitator has to follow.
        constants (dict): An optional dictionary containing constants to populate the interview prompts.
    """

    def __init__(
        self,
        model_info: str,
        temperature: float,
        profiles: pd.DataFrame,
        roles: dict[str, Role],
        num_subjects_per_group: int = 1,
        num_groups: int = 1,
        experiment_context: str = "",
        session_id: str = "",
        hf_inference_endpoint: str = "",
        max_num_rounds: int = 10,
        treatments: dict[str, Treatment] = {},
        treatment_assignment_strategy: str = "simple_random",
        treatment_column: str = "",
        group_assignment_strategy: str = "random",
        group_column: str = "",
        role_assignment_strategy: str = "random",
        role_column: str = "",
        random_seed: int = 42,
        build_profile_qna: bool = True,
        build_profile_backstories: bool = False,
        prompts: List[dict[str, str]] = [],
        constants: dict = {},
    ):
        super().__init__(
            model_info,
            temperature,
            profiles,
            roles,
            num_subjects_per_group,
            num_groups,
            experiment_context,
            session_id,
            hf_inference_endpoint,
            max_num_rounds,
            treatments,
            treatment_assignment_strategy,
            treatment_column,
            group_assignment_strategy,
            group_column,
            role_assignment_strategy,
            role_column,
            random_seed,
            build_profile_qna,
            build_profile_backstories,
        )

        self.roles = self._check_roles(roles=roles)
        self.num_subjects_per_group = self._check_num_subjects_per_group(
            num_subjects_per_group=num_subjects_per_group
        )
        self.group_assignment = self._assign_group(random_seed=self.random_seed)
        self.role_assignment = self._assign_role(random_seed=self.random_seed)
        if self.role_assignment_strategy == "manual":
            self._check_manually_assigned_roles()
        self.prompts = self._check_prompts(prompts=prompts)
        self.constants = constants

    def _check_roles(self, roles: dict[str, Role]) -> dict[str, Role]:
        """Checks if the provided roles are valid.

        Args:
            roles (dict[str, Role]): The roles to be checked.

        Returns:
            dict[str, Role]: The validated roles.

        Raises:
            ValueError: If the provided roles is not valid.
        """
        if "facilitator" not in list(roles.keys()):
            raise ValueError(
                "For an AI-to-AI interview-based experiment, one of the roles must be 'facilitator'."
            )

        return roles

    def _check_num_subjects_per_group(self, num_subjects_per_group: int) -> int:
        """
        Validates and processes the `num_subjects_per_group` parameter.
        This method ensures that the number of subjects per group is valid and consistent
        with the experiment's configuration, including the number of user-defined roles
        and the total number of profiles provided.

        Args:
            num_subjects_per_group (int): The number of subjects assigned to each group.
                This value must be an integer or float that is equal to or greater than 1.

        Returns:
            int: The validated number of subjects per group. If the input is `None` or `NaN`,
            the method defaults to 1.

        Raises:
            ValueError: If `num_subjects_per_group` is less than 1, not a number, or if it
            does not align with the number of user-defined roles and profiles provided.

        Warnings:
            - If `num_subjects_per_group` is `None`, a warning is issued, and the value defaults to 1.
            - If `num_subjects_per_group` is `NaN`, a warning is issued, and the value defaults to 1.
        """
        if num_subjects_per_group is None:
            warnings.warn(
                "The num_subjects_per_group field contains a None value; defaulting to 1."
            )
            return 1

        # Ensure that num_subjects_per_group is a number (float or int)
        if isinstance(num_subjects_per_group, (int, float)):
            if math.isnan(num_subjects_per_group):
                warnings.warn(
                    "The num_subjects_per_group field contains a NaN value; defaulting to 1."
                )
                return 1

            # Check if number of subjects per group is 1 or more
            if num_subjects_per_group < 1:
                raise ValueError(
                    f"Invalid num_subjects_per_group: {num_subjects_per_group}. For AI-AI interview-based experiments, num_subjects_per_group should be an integer that is equal to or greater than 1."
                )

            # Ensure that number of subjects per group matches with the number of profiles provided
            user_defined_roles = [
                role for role in list(self.roles.keys()) if role not in SPECIAL_ROLES
            ]
            if len(user_defined_roles) != num_subjects_per_group:
                raise ValueError(
                    f"Number of user-defined roles ({len(user_defined_roles)}) does not match the number of subjects assigned to each group ({num_subjects_per_group})."
                )

            # Ensure that number of user-defined roles multiplied by the number of groups is less than or equal to the number of profiles provided
            if self.num_groups * len(user_defined_roles) != len(self.profiles):
                raise ValueError(
                    f"Total number of subjects required for session ({self.num_groups * len(user_defined_roles)}) does not match the number of profiles provided ({len(self.profiles)})."
                )

            # Otherwise, return the provided num_subjects_per_group as is
            return num_subjects_per_group

        else:
            raise ValueError(
                "The num_subjects_per_group field must be a float or integer value."
            )

    def _assign_group(self, random_seed: int) -> dict[int, List[ProfileInfo]]:
        """Assigns profiles to each group based on the given number of subjects per group (excluding the special roles) and group assignment strategy.
        However, if the group_assignment_strategy is 'manual', then assign the subjects to their respective groups based on the
        assignment defined in profiles.

        Args:
            random_seed (int): The random seed for reproducibility.

        Returns:
            dict[int, List[ProfileInfo]]: A dictionary mapping group IDs to a list of profile information.
        """
        num_user_defined_roles = len(
            [role for role in list(self.roles.keys()) if role not in SPECIAL_ROLES]
        )

        if self.group_assignment_strategy == "manual":
            group_assignment = {}
            for i, group_id in enumerate(self.group_id_list):
                group_subjects = self.profiles[
                    self.profiles[self.group_column] == group_id
                ].reset_index(drop=True)

                num_group_subjects = len(group_subjects)
                if num_group_subjects != num_user_defined_roles:
                    raise ValueError(
                        f"Group {group_id} contains {num_group_subjects} subjects while the number of user-defined roles per group is supposed to be {num_user_defined_roles}"
                    )

                group_assignment[group_id] = group_subjects.to_dict(orient="records")

        else:
            randomised_profiles = self.profiles.sample(
                frac=1, random_state=random_seed
            ).reset_index(drop=True)

            group_assignment = {}
            for i, group_id in enumerate(self.group_id_list):
                group_assignment[group_id] = randomised_profiles.iloc[
                    i * num_user_defined_roles : (i + 1) * num_user_defined_roles
                ].to_dict(orient="records")

        return group_assignment

    def _assign_role(self, random_seed: int) -> dict[int, str]:
        """Assigns roles to subjects based on the specified role assignment strategy.

        Args:
            random_seed (int): The seed value for randomization when using the "random" role assignment strategy.

        Returns:
            dict[int, str]: A dictionary mapping subject IDs to their assigned roles.

        Raises:
            ValueError: If the number of user-defined roles does not match the number of subjects
                        assigned to a group when using the "random" role assignment strategy.
        """
        if self.role_assignment_strategy == "manual":
            role_assignment = self.profiles.set_index("ID")[self.role_column].to_dict()

        else:
            random.seed(random_seed)
            role_assignment = {}
            user_defined_role_labels = [
                role for role in list(self.roles.keys()) if role not in SPECIAL_ROLES
            ]
            for group_id, group_subjects in self.group_assignment.items():
                num_group_subjects = len(group_subjects)

                if len(user_defined_role_labels) == num_group_subjects:
                    randomized_roles = random.sample(
                        user_defined_role_labels, num_group_subjects
                    )

                else:
                    raise ValueError(
                        f"Number of user-defined roles ({len(user_defined_role_labels)}) does not match the number of subjects ({num_group_subjects}) assigned to Group {group_id}."
                    )

                role_assignment.update(
                    {
                        subject["ID"]: role
                        for subject, role in zip(group_subjects, randomized_roles)
                    }
                )

        return role_assignment

    def _check_manually_assigned_roles(self) -> None:
        """Validates that all manually assigned roles are defined in roles, excluding special roles like "facilitator".

        This method checks whether the roles manually assigned in the `role_assignment`
        dictionary are a subset of the roles defined in the `roles` dictionary.
        If any manually assigned role is not present in the defined roles, a
        `ValueError` is raised.

        Raises:
            ValueError: If the roles defined in the `role` worksheet are not a
            superset of the manually defined roles in the `profile` worksheet.
        """
        user_defined_roles = set(
            [role for role in list(self.roles.keys()) if role not in SPECIAL_ROLES]
        )
        manual_defined_roles = set(self.role_assignment.values())

        if not user_defined_roles.issuperset(manual_defined_roles):
            raise ValueError(
                f"The user-defined roles in the role worksheet ({list[user_defined_roles]}) is not a superset of the manually defined roles in the profiles worksheet ({list[manual_defined_roles]})."
            )
        else:
            pass

    def _check_prompts(self, prompts: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """
        Validates a list of prompt dictionaries to ensure they conform to the expected structure
        and contain valid values.

        Args:
            prompts (List[dict[str, Any]]): A list of dictionaries where each dictionary represents
                a prompt with specific fields and values.

        Returns:
            List[dict[str, Any]]: The validated list of prompts.

        Raises:
            ValueError: If any of the following conditions are not met:
                - The `prompts` list is not empty.
                - The first item in `prompts` contains a "type" field with the value "context".
                - Each prompt's "type" field contains only approved prompt types (defined in SUPPORTED_PROMPT_TYPES).
                - Each prompt's "response_name" field is unique across all prompts.
                - Each prompt's "randomize_response_order" field contains only approved values (True or False).
                - Each prompt's "validate_response" field contains only approved values (True or False).
                - Each prompt's "generate_speculation_score" field contains only approved values (True or False).
                - Each prompt's "format_response" field contains only approved values (True or False).
        """
        # Check if prompts is not an empty list
        if not prompts:
            raise ValueError("The prompts list should not be an empty list.")

        # Check if the first item in prompts contains "type": "context"
        if "type" not in prompts[0] or prompts[0]["type"] != "context":
            raise ValueError(
                'The first item in prompts must contain "type": "context" to provide context for the experiment/interview.'
            )

        unique_response_names = []
        for prompt in prompts:
            # Check if the type field contains only approved prompt types
            if prompt["type"] not in SUPPORTED_PROMPT_TYPES:
                raise ValueError(
                    f"Round ID {prompt['round_id']} contains an invalid prompt type: {prompt['type']}. Supported prompt types include: {SUPPORTED_PROMPT_TYPES}."
                )

            # Check if the response_name column contains unique response names
            if prompt["response_name"] in unique_response_names:
                raise ValueError(
                    f"Round ID {prompt['round_id']} contains a non-unique response name: {prompt['response_name']}."
                )
            else:
                unique_response_names.append(prompt["response_name"])

            # Check if the randomize_response_order column contains only approved values (True or False)
            if prompt["randomize_response_order"] not in [True, False]:
                raise ValueError(
                    f"Round ID {prompt['round_id']} contains an invalid value in randomize_response_order field: {prompt['randomize_response_order']}. Supported options include: True or False."
                )

            # Check if the validate_response column contains only approved values (True or False)
            if prompt["validate_response"] not in [True, False]:
                raise ValueError(
                    f"Round ID {prompt['round_id']} contains an invalid value in validate_response field: {prompt['validate_response']}. Supported options include: True or False."
                )

            # Check if the generate_speculation_score column contains only approved values (True or False)
            if prompt["generate_speculation_score"] not in [True, False]:
                raise ValueError(
                    f"Round ID {prompt['round_id']} contains an invalid value in generate_speculation_score field: {prompt['generate_speculation_score']}. Supported options include: True or False."
                )

            # Check if the format_response column contains only approved values (True or False)
            if prompt["format_response"] not in [True, False]:
                raise ValueError(
                    f"Round ID {prompt['round_id']} contains an invalid value in format_response field: {prompt['format_response']}. Supported options include: True or False."
                )

        return prompts

    def run_session(
        self,
        test_mode: bool = True,
        version: int = 1,
        save_results_as_csv: bool = False,
    ) -> dict[str, Any]:
        """Runs a session based on the experimental settings defined during class initialisation.
        If test_mode is set to True, a random group from each treatment arm will be selected and run; otherwise, groups are run in parallel.

        Args:
            test_mode (bool, optional): Indicates whether the session is run in test mode or not.
                Defaults to True.
            version (int, optional): Indicates the version of the session.
                Defaults to 1.
            save_results_as_csv (bool, optional): Indicates whether the results of the experiment will be saved as CSV format.
                Defaults to False

        Returns:
            dict[str, Any]: A dictionary containing the session ID and session information.
        """
        if test_mode:  # Run one session from each treatment arm
            random.seed(self.random_seed)
            group_id_list = []
            for treatment in list(self.treatments.keys()):
                matching_sessions = [
                    sid
                    for sid, assigned_treatment in self.treatment_assignment.items()
                    if assigned_treatment == treatment
                ]
                if matching_sessions:
                    group_id_list.append(random.choice(matching_sessions))

        else:
            group_id_list = self.group_id_list

        session = {
            "session_id": f"{self.session_id}_{version}",
            "groups": {},
        }
        self.experiment_context = self.prompts.pop(0)["llm_text"]

        # Helper function to process a single group.
        def process_group(group_id: Any) -> tuple[Any, dict]:
            group_info = {}
            group_info["group_id"] = group_id
            group_info["random_seed"] = self.random_seed
            group_info["constants"] = self.constants
            group_info["treatment_label"] = self.treatment_assignment[group_id]
            group_info["treatment"] = self.treatments[group_info["treatment_label"]]
            group_info["experiment_context"] = self.experiment_context
            group_info["profiles"] = self.group_assignment[group_id]
            group_subject_ids = [profile["ID"] for profile in group_info["profiles"]]
            group_info["roles"] = {
                subject_id: assigned_role
                for subject_id, assigned_role in self.role_assignment.items()
                if subject_id in group_subject_ids
            }
            group_info["subjects"] = self._initialize_subjects(group_info)
            group_info = self._run_group(
                group_info=group_info,
                prompts=self.prompts,
                test_mode=test_mode,
            )
            updated_group_subjects = {}
            for subject_role, subject in group_info["subjects"].items():
                updated_group_subjects[subject_role] = subject.to_dict()
            group_info["subjects"] = updated_group_subjects

            return group_id, group_info

        if test_mode:
            # Sequentially process sessions in test mode.
            for group_id in tqdm(group_id_list):
                group_id, group_info = process_group(group_id)
                session["groups"][group_id] = group_info
        else:
            # Process groups in parallel using ThreadPoolExecutor.
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_group_id = {
                    executor.submit(process_group, group_id): group_id
                    for group_id in group_id_list
                }
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_group_id),
                    total=len(future_to_group_id),
                ):
                    group_id, group_info = future.result()
                    session["groups"][group_id] = group_info

        self._save_session(session, save_results_as_csv=save_results_as_csv)

        return session

    def _initialize_subjects(
        self, group_info: dict[str, Any]
    ) -> dict[str, ConversationalSyntheticSubject]:
        """Initializes and returns a dictionary of ConversationalSyntheticSubject objects for both special and user-defined roles based on the provided group information.

        Args:
            group_info (dict[str, Any]): A dictionary containing group information, including subject profiles, roles, group ID, treatment, etc.

        Returns:
            dict[str, ConversationalSyntheticSubject]: A dictionary where the key indicates the role and the value is an initialized ConversationalSyntheticSubject objects.

        Raises:
            AssertionError: If the number of profiles does not match the number of user-defined roles when initializing subjects.
        """
        subject_dict = {}

        # First, initialise facilitator
        subject_dict["facilitator"] = ConversationalSyntheticSubject(
            session_id=self.session_id,
            experiment_context="",
            group_id=group_info["group_id"],
            profile_info={},
            model_info=self.model_info,
            temperature=self.temperature,
            build_profile_qna=False,
            build_profile_backstories=False,
            hf_inference_endpoint=self.hf_inference_endpoint,
            role_label="facilitator",
            role=self.roles["facilitator"],
            treatment=group_info["treatment"],
            constants=group_info["constants"],
        )

        # Initialise user-defined subjects based on sequence defined in roles
        user_defined_roles = [
            role for role in list(self.roles.keys()) if role not in SPECIAL_ROLES
        ]
        assert len(group_info["profiles"]) == len(
            user_defined_roles
        ), f"Number of profiles ({len(group_info['profiles'])}) does not match the number of user-defined roles ({len(user_defined_roles)}) when initialising subjects. The number of profiles should be equal the number of user-defined roles (excluding special roles like facilitator)."

        for i in range(len(group_info["profiles"])):
            subject_id = group_info["profiles"][i]["ID"]
            role_label = group_info["roles"][subject_id]
            subject_dict[role_label] = ConversationalSyntheticSubject(
                session_id=self.session_id,
                experiment_context=self.experiment_context[role_label],
                group_id=group_info["group_id"],
                profile_info={
                    k: v
                    for k, v in group_info["profiles"][i].items()
                    if k
                    not in [self.treatment_column, self.role_column, self.group_column]
                },
                model_info=self.model_info,
                temperature=self.temperature,
                build_profile_qna=self.build_profile_qna,
                build_profile_backstories=self.build_profile_backstories,
                hf_inference_endpoint=self.hf_inference_endpoint,
                role_label=role_label,
                role=self.roles[role_label],
                treatment=group_info["treatment"],
                constants=group_info["constants"],
            )

        return subject_dict

    def _sort_tasks(self, prompts: list[dict]) -> list[dict]:
        """Sorts and shuffles a list of prompts based on their "round_order" value.
        This method groups the input prompts by their "round_order" value, sorts the prompts
        in ascending order of "round_order", and shuffles the prompts within each group
        randomly. The shuffled groups are then concatenated to form the final sorted list.

        Args:
            prompts (list[dict]): A list of dictionaries, where each dictionary represents
                a prompt and contains a "round_order" key.

        Returns:
            list[dict]: A list of prompts sorted by "round_order" and shuffled within each
                "round_order" group.
        """
        random.seed(self.random_seed)

        # Group prompts by their round_order value.
        groups = defaultdict(list)
        for prompt in prompts:
            groups[prompt["round_order"]].append(prompt)

        # For prompts with the same round_order, shuffle the group randomly.
        sorted_prompts = []
        for order in sorted(groups.keys()):  # Ascending order on round_order
            group = groups[order]
            random.shuffle(group)  # Randomly shuffle prompts with the same round_order
            sorted_prompts.extend(group)

        return sorted_prompts

    def _format_response_options(
        self, response_options: Any, randomize_response_order: bool = False
    ) -> str:
        """Formats the given response options into a human-readable string.

        Args:
            response_options (Any): The response options to format. This can be a range,
                                    a list, or any other type.
            randomize_response_order (bool, optional): Indicates whether the response options should be randomized.
                Defaults to False.

        Returns:
            str: A formatted string representation of the response options.
                 - If `response_options` is a range, it is formatted as "from X to Y".
                 - If `response_options` is a list, its elements are joined with commas.
                 - For other types, it falls back to a simple string conversion.
        """
        random.seed(self.random_seed)
        # If response_options is a range, format it as "from X to Y"
        if isinstance(response_options, range):
            return f"Respond with a numerical value ranging from '{response_options.start}' to '{response_options.stop}' (inclusive):"

        # If it's a list, join the elements with commas.
        elif isinstance(response_options, list):
            if len(response_options) == 0:
                formatted = ""
            elif len(response_options) == 1:
                formatted = f"'{response_options[0]}'"
            else:
                if randomize_response_order:
                    random.shuffle(response_options)
                formatted = (
                    ", ".join(f"'{opt}'" for opt in response_options[:-1])
                    + f" or '{response_options[-1]}'"
                )
            return f"Respond with one option from {formatted}:"

        # Otherwise, fallback to a simple string conversion.
        else:
            return str(response_options)

    def _run_group(
        self,
        group_info: dict[str, Any],
        prompts: List[dict[str, str]],
        test_mode: bool = False,
    ) -> dict[str, Any]:
        session_message_history = []
        subject_message_history = {}

        # Set initial experiment context for each user-defined role
        for role, experiment_context in group_info["experiment_context"].items():
            experiment_context_template = Template(experiment_context)
            rendered_experiment_context = experiment_context_template.render(
                treatment=group_info["subjects"][role].treatment.to_dict(),
                role=group_info["subjects"][role].role.to_dict(),
                constant=group_info["subjects"][role].constants,
                response_options={},
            )
            message_dict = {"system": f"{role}: {rendered_experiment_context}"}
            subject_message_history[role] = [message_dict]
            session_message_history.append(message_dict)

            if test_mode:
                print(message_dict)
                print()

        subject_message_history["facilitator"] = []

        # Sort the order of rounds based on the round_order field. If round order is repeated, then it is expected that the round order are randomised
        prompts = self._sort_tasks(prompts)

        subject_list = list(group_info["subjects"].values())
        round_num = 0
        for round in prompts:
            # facilitator is providing instructions/information to all subjects.
            if round["type"] == "context":
                # Format context for each role
                for role, context_prompt in round["llm_text"].items():
                    context_prompt_template = Template(context_prompt)
                    formatted_response_options = self._format_response_options(
                        response_options=round["response_options"].get(role, ""),
                        randomize_response_order=round["randomize_response_order"],
                    )
                    rendered_context_prompt = context_prompt_template.render(
                        treatment=group_info["subjects"][role].treatment.to_dict(),
                        role=group_info["subjects"][role].role.to_dict(),
                        constant=group_info["subjects"][role].constants,
                        response_options=formatted_response_options,
                    )

                    message_dict = {
                        "facilitator": rendered_context_prompt,
                    }

                    subject_message_history[role].append(message_dict)
                    session_message_history.append(message_dict)

                    if test_mode:
                        print(message_dict)
                        print()

                # Context setting only, no response required from subjects
                round_num += 1
                if round_num >= self.max_num_rounds:
                    warnings.warn(
                        "Maximum number of rounds reached. Terminating session prematurely."
                    )
                    group_info["message_history"] = session_message_history
                    return group_info

            elif round["type"] == "discussion":
                # Format discussion question
                discussion_prompt = round["llm_text"]["facilitator"]
                discussion_prompt_template = Template(discussion_prompt)
                formatted_response_options = self._format_response_options(
                    response_options=round["response_options"].get("facilitator", ""),
                    randomize_response_order=round["randomize_response_order"],
                )
                rendered_discussion_prompt = discussion_prompt_template.render(
                    treatment=group_info["subjects"]["facilitator"].treatment.to_dict(),
                    role=group_info["subjects"]["facilitator"].role.to_dict(),
                    constant=group_info["subjects"]["facilitator"].constants,
                    response_options=formatted_response_options,
                )

                message_dict = {
                    "facilitator": rendered_discussion_prompt,
                }

                for subject in subject_list:
                    subject_message_history[subject.role_label].append(message_dict)
                session_message_history.append(message_dict)

                if test_mode:
                    print(message_dict)
                    print()

                # Loop through each subject back-to-back and get their response during a discussion round
                for role, subject in group_info["subjects"].items():
                    if role == "facilitator":
                        continue
                    response = subject.respond(
                        latest_message_history=subject_message_history[role],
                        generate_speculation_score=round["generate_speculation_score"],
                        format_response=round["format_response"],
                    )
                    subject_message_history[role] = []

                    message_dict = {
                        role: response,
                        "subject_id": subject.profile_info.get("ID", ""),
                        "round_id": round.get("round_id", None),
                        "response_name": round.get("response_name", None),
                    }
                    for subject in subject_list:
                        subject_message_history[subject.role_label].append(message_dict)
                    session_message_history.append(message_dict)

                    if test_mode:
                        print(message_dict)
                        print()

                round_num += 1
                if round_num >= self.max_num_rounds:
                    warnings.warn(
                        "Maximum number of rounds reached. Terminating session prematurely."
                    )
                    group_info["message_history"] = session_message_history
                    return group_info

            elif round["type"] in ["public_question", "repeat_public_question"]:
                # facilitator is posing the same question to each subject and the subjects' responses are shown to all subjects during the round.
                response = ""
                num_repeated = 0
                while not re.compile(r"(?<!\w)end_round(?!\w)", re.IGNORECASE).search(
                    response
                ):
                    for role, question in round["llm_text"].items():
                        question_template = Template(question)
                        formatted_response_options = self._format_response_options(
                            response_options=round["response_options"].get(role, ""),
                            randomize_response_order=round["randomize_response_order"],
                        )
                        rendered_question = question_template.render(
                            treatment=group_info["subjects"][role].treatment.to_dict(),
                            role=group_info["subjects"][role].role.to_dict(),
                            constant=group_info["subjects"][role].constants,
                            response_options=formatted_response_options,
                        )

                        if round["type"] == "repeat_public_question":
                            rendered_question = (
                                f"Round {num_repeated + 1}: {rendered_question}"
                            )

                        message_dict = {
                            "facilitator": rendered_question,
                        }
                        for subject in subject_list:
                            subject_message_history[subject.role_label].append(
                                message_dict
                            )
                        session_message_history.append(message_dict)

                        if test_mode:
                            print(message_dict)
                            print()

                        subject = group_info["subjects"][role]
                        response = subject.respond(
                            latest_message_history=subject_message_history[role],
                            validate_response=round["validate_response"],
                            response_options=round["response_options"].get(role, ""),
                            generate_speculation_score=round[
                                "generate_speculation_score"
                            ],
                            format_response=round["format_response"],
                        )
                        subject_message_history[role] = []

                        message_dict = {
                            role: response,
                            "subject_id": subject.profile_info.get("ID", ""),
                            "round_id": round.get("round_id", None),
                            "response_name": round.get("response_name", None),
                        }
                        if round["type"] == "repeat_public_question":
                            message_dict["round_num"] = num_repeated + 1

                        for subject in subject_list:
                            subject_message_history[subject.role_label].append(
                                message_dict
                            )
                        session_message_history.append(message_dict)

                        if test_mode:
                            print(message_dict)
                            print()

                    num_repeated += 1
                    round_num += 1
                    if round_num >= self.max_num_rounds:
                        warnings.warn(
                            "Maximum number of rounds reached. Terminating session prematurely."
                        )
                        group_info["message_history"] = session_message_history
                        return group_info

                    if round["type"] == "public_question":
                        break  # exit while loop after one full round of public_question

            elif round["type"] in ["private_question", "repeat_private_question"]:
                # facilitator is posing the same question to each subject but the subjects' responses are not shown to other subjects during the round
                response = ""
                num_repeated = 0
                while not re.compile(r"(?<!\w)end_round(?!\w)", re.IGNORECASE).search(
                    response
                ):
                    for role, question in round["llm_text"].items():
                        question_template = Template(question)
                        formatted_response_options = self._format_response_options(
                            response_options=round["response_options"].get(role, ""),
                            randomize_response_order=round["randomize_response_order"],
                        )
                        rendered_question = question_template.render(
                            treatment=group_info["subjects"][role].treatment.to_dict(),
                            role=group_info["subjects"][role].role.to_dict(),
                            constant=group_info["subjects"][role].constants,
                            response_options=formatted_response_options,
                        )

                        if round["type"] == "repeat_private_question":
                            rendered_question = (
                                f"Round {num_repeated + 1}: {rendered_question}"
                            )

                        message_dict = {
                            "facilitator": rendered_question,
                        }

                        session_message_history.append(message_dict)
                        if role != "facilitator":
                            subject_message_history[role].append(message_dict)
                            subject_message_history["facilitator"].append(message_dict)
                        else:
                            subject_message_history["facilitator"].append(message_dict)

                        if test_mode:
                            print(message_dict)
                            print()

                        subject = group_info["subjects"][role]
                        if role != "facilitator":
                            response = subject.respond(
                                latest_message_history=subject_message_history[role],
                                validate_response=round["validate_response"],
                                response_options=round["response_options"].get(
                                    role, ""
                                ),
                                generate_speculation_score=round[
                                    "generate_speculation_score"
                                ],
                                format_response=round["format_response"],
                            )
                        else:
                            response = subject.respond(
                                latest_message_history=subject_message_history[role],
                                validate_response=False,
                                response_options=[],
                                generate_speculation_score=False,
                                format_response=True,
                            )
                        subject_message_history[role] = []

                        message_dict = {
                            role: response,
                            "subject_id": subject.profile_info.get("ID", ""),
                            "round_id": round.get("round_id", None),
                            "response_name": round.get("response_name", None),
                        }
                        if round["type"] == "repeat_private_question":
                            message_dict["round_num"] = num_repeated + 1

                        session_message_history.append(message_dict)
                        if role != "facilitator":
                            subject_message_history[role].append(message_dict)
                            subject_message_history["facilitator"].append(message_dict)

                        else:
                            for subject in subject_list:
                                subject_message_history[subject.role_label].append(
                                    message_dict
                                )

                        if test_mode:
                            print(message_dict)
                            print()

                    num_repeated += 1
                    round_num += 1
                    if round_num >= self.max_num_rounds:
                        warnings.warn(
                            "Maximum number of rounds reached. Terminating session prematurely."
                        )
                        group_info["message_history"] = session_message_history
                        return group_info

                    if round["type"] == "private_question":
                        break  # exit while loop after one full round of private_question

            else:
                raise ValueError(
                    f"Invalid prompt type: {round['type']}. The type of prompt for each round should one of these options: {SUPPORTED_PROMPT_TYPES}"
                )

        for subject in subject_list:
            subject.update_message_history(
                latest_message_history=subject_message_history[subject.role_label]
            )

        session_message_history.append({"system": "end_session"})
        if test_mode:
            print({"system": "end_session"})

        group_info["message_history"] = session_message_history
        return group_info
