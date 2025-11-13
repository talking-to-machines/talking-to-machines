import re, warnings, openai, json
from typing import Any, Callable
from talkingtomachines.generative.prompt import (
    generate_subject_system_message,
    generate_profile_prompt,
)
from talkingtomachines.generative.llm import query_llm
from talkingtomachines.config import DevelopmentConfig
from talkingtomachines.management.experiment import Role, Treatment

ProfileInfo = dict[str, Any]
NUM_RETRY = 3
OPENAI_MODELS = [
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
]


class SyntheticSubject:
    """A class for constructing the base synthetic subject.

    Args:
        session_id (str): The session ID of the experiment.
        experiment_context (str): The context of the experiment.
        group_id (Any): The group ID of the session.
        profile_info (ProfileInfo): The profile information of the subject.
        model_info (str): The information about the model used by the subject.
        temperature (float): The model temperature setting for the subject.
        build_profile_qna (bool): Whether to build the subject profiles using Q&A format.
        build_profile_backstories (bool): Whether to build the subject profiles using backstories.
        hf_inference_endpoint (str, optional): API inference endpoint to the LLM model hosted externally in HuggingFace.
        profile_prompt_generator (Callable[[ProfileInfo, bool, bool, openai.OpenAI, str, float], str], optional):
            A function that generates a profile prompt based on the profile information.
            Defaults to generate_profile_prompt.

    Attributes:
        session_id (str): The session ID of the experiment.
        experiment_context (str): The context of the experiment.
        group_id (Any): The group ID of the session.
        profile_info (ProfileInfo): The profile information of the subject.
        profile_prompt (str): A prompt string containing the profile information of the subject.
        model_info (str): The information about the model used by the subject.
        temperature (float): The model temperature setting for the subject.
        build_profile_qna (bool): Whether to build the subject profiles using Q&A format.
        build_profile_backstories (bool): Whether to build the subject profiles using backstories.
        hf_inference_endpoint (str): API inference endpoint to the LLM model hosted externally in HuggingFace.
        llm_client (openai.OpenAI): The LLM client.
    """

    def __init__(
        self,
        session_id: str,
        experiment_context: str,
        group_id: Any,
        profile_info: ProfileInfo,
        model_info: str,
        temperature: float,
        build_profile_qna: bool,
        build_profile_backstories: bool,
        hf_inference_endpoint: str = "",
        profile_prompt_generator: Callable[
            [ProfileInfo, bool, bool, openai.OpenAI, str, float], str
        ] = generate_profile_prompt,
    ):
        self.session_id = session_id
        self.experiment_context = experiment_context
        self.group_id = group_id
        self.profile_info = profile_info
        self.model_info = model_info
        self.temperature = temperature
        self.build_profile_qna = build_profile_qna
        self.build_profile_backstories = build_profile_backstories
        self.hf_inference_endpoint = hf_inference_endpoint
        self.llm_client = self._initialise_llm_client()
        self.profile_prompt = profile_prompt_generator(
            self.profile_info,
            self.build_profile_qna,
            self.build_profile_backstories,
            self.llm_client,
            self.model_info,
            self.temperature,
        )

    def _initialise_llm_client(self):
        """
        Initializes and returns an instance of the OpenAI client based on the specified model information.
        This method determines the appropriate API client configuration based on the `model_info` attribute.
        It supports OpenAI models, Hugging Face inference models, and OpenRouter-supported models.

        Returns:
            openai.OpenAI: An instance of the OpenAI client configured for the specified model.

        Raises:
            None
        """
        if self.model_info in OPENAI_MODELS:
            return openai.OpenAI(api_key=DevelopmentConfig.OPENAI_API_KEY)

        elif self.model_info in ["hf-inference"]:
            return openai.OpenAI(
                base_url=self.hf_inference_endpoint,
                api_key=DevelopmentConfig.HF_API_KEY,
            )

        else:
            warnings.warn(
                f"Since {self.model_info} is not 'hf-inference' and not one of the openai.OpenAI instruct models ({OPENAI_MODELS}), {self.model_info} is assumed to be an Openrouter.ai supported model."
            )
            return openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=DevelopmentConfig.OPENROUTER_API_KEY,
            )

    def to_dict(self) -> dict[str, Any]:
        """Converts the SyntheticSubject object to a dictionary.

        Returns:
            dict[str, Any]: A dictionary representation of the SyntheticSubject object.
        """
        return {
            "session_id": self.session_id,
            "experiment_context": self.experiment_context,
            "group_id": self.group_id,
            "profile_info": self.profile_info,
            "model_info": self.model_info,
            "temperature": self.temperature,
            "build_profile_qna": self.build_profile_qna,
            "build_profile_backstories": self.build_profile_backstories,
            "hf_inference_endpoint": self.hf_inference_endpoint,
            "profile_prompt": self.profile_prompt,
        }

    def respond(self) -> str:
        """Generate a response based on the synthetic subject's model.

        Returns:
            str: The response generated by the synthetic subject.
        """
        try:
            return ""
        except Exception as e:
            # Log the exception
            print(f"Error during response generation in SyntheticSubject object: {e}")
            return None


class ConversationalSyntheticSubject(SyntheticSubject):
    """A synthetic subject that interacts with users in a conversational system. Inherits from the SyntheticSubject base class.

    Args:
        session_id (str): The session ID of the experiment.
        experiment_context (str): The context of the experiment.
        group_id (Any): The group ID of the session.
        profile_info (ProfileInfo): The profile information of the subject.
        model_info (str): The information about the model used by the subject.
        temperature (float): The model temperature setting for the subject.
        build_profile_qna (bool): Whether to build the subject profiles using Q&A format.
        build_profile_backstories (bool): Whether to build the subject profiles using backstories.
        hf_inference_endpoint (str, optional): API inference endpoint to the LLM model hosted externally in HuggingFace.
        role_label (str): The name of the role assigned to the subject.
        role (Role): The Role object assigned to the subject.
        treatment (Treatment): The Treatment object assigned to the subject.
        profile_prompt_generator (Callable[[ProfileInfo, bool, bool, openai.OpenAI, str, float], str], optional):
            A function that generates a profile prompt based on the profile information.
            Defaults to generate_profile_prompt.

    Attributes:
        session_id (str): The session ID of the experiment.
        experiment_context (str): The context of the experiment.
        group_id (Any): The group ID of the session.
        profile_info (ProfileInfo): The profile information of the subject.
        profile_prompt (str): A prompt string containing the profile information of the subject.
        model_info (str): The information about the model used by the subject.
        temperature (float): The model temperature setting for the subject.
        build_profile_qna (bool): Whether to build the subject profiles using Q&A format.
        build_profile_backstories (bool): Whether to build the subject profiles using backstories.
        hf_inference_endpoint (str): API inference endpoint to the LLM model hosted externally in HuggingFace.
        role_label (str): The Role object assigned to the subject.
        role (Role): The description of the role assigned to the subject.
        treatment (Treatment): The Treatment object assigned to the subject.
        system_message (str): The system message generated for the conversation.
        llm_client (openai.OpenAI): The LLM client.
        message_history (List[dict]): The history of the conversation with the synthetic subject.
    """

    def __init__(
        self,
        session_id: str,
        experiment_context: str,
        group_id: Any,
        profile_info: ProfileInfo,
        model_info: str,
        temperature: float,
        build_profile_qna: bool,
        build_profile_backstories: bool,
        hf_inference_endpoint: str,
        role_label: str,
        role: Role,
        treatment: Treatment,
        profile_prompt_generator: Callable[
            [ProfileInfo, bool, bool, openai.OpenAI, str, float], str
        ] = generate_profile_prompt,
    ):
        super().__init__(
            session_id,
            experiment_context,
            group_id,
            profile_info,
            model_info,
            temperature,
            build_profile_qna,
            build_profile_backstories,
            hf_inference_endpoint,
            profile_prompt_generator,
        )
        self.role_label = role_label
        self.role = role
        self.treatment = treatment
        self.system_message = generate_subject_system_message(
            role_description=self.role.description,
            profile_prompt=self.profile_prompt,
        )
        self.message_history = [{"role": "system", "content": self.system_message}]

    def to_dict(self) -> dict[str, Any]:
        """Converts the ConversationalSyntheticSubject object to a dictionary.

        Returns:
            dict[str, Any]: A dictionary representation of the ConversationalSyntheticSubject object.
        """
        return {
            "session_id": self.session_id,
            "experiment_context": self.experiment_context,
            "group_id": self.group_id,
            "profile_info": self.profile_info,
            "profile_prompt": self.profile_prompt,
            "model_info": self.model_info,
            "temperature": self.temperature,
            "build_profile_qna": self.build_profile_qna,
            "build_profile_backstories": self.build_profile_backstories,
            "hf_inference_endpoint": self.hf_inference_endpoint,
            "role_label": self.role_label,
            "role": self.role.to_dict(),
            "treatment": self.treatment.to_dict(),
            "system_message": self.system_message,
            "message_history": self.message_history,
        }

    def _build_message_history(self, message_history: list[dict]) -> list[dict]:
        """
        Builds and formats a message history for a conversational AI system.

        Args:
            message_history (list[dict]): A list of dictionaries representing the message history.
                Each dictionary contains a single key-value pair where the key is the role
                (e.g., "system", "assistant", or a user-defined role) and the value is the
                corresponding message content.

        Returns:
            list[dict]: A list of formatted message dictionaries. Each dictionary contains:
                - "role": The role of the message sender (e.g., "system", "assistant", or "user").
                - "content": The formatted message content, which may include rendered templates
                  based on the treatment and role information.
        """
        formatted_message_history = []

        for message in message_history:
            role = list(message.keys())[0]
            role_response = list(message.values())[0]

            if role == "system":
                formatted_message_history.append(
                    {"role": "system", "content": role_response}
                )

            elif role == self.role:
                formatted_message_history.append(
                    {"role": "assistant", "content": role_response}
                )

            else:  # Other conversations in the same session
                formatted_message_history.append(
                    {"role": "user", "content": f"{role}: {role_response}"}
                )

        return formatted_message_history

    def _validate_response(self, response: Any, response_options: Any) -> bool:
        """Validates whether a given response contains any of the valid response options as whole words.

        Args:
            response (Any): The response to validate. This could be a string or a JSON dictionary
            response_options (Any): The valid response options. Can be a range, a list, or a string value.

        Returns:
            bool: True if the response contains any of the valid options as whole words, False otherwise.
        """
        try:
            response = json.loads(response)
            response = response["response"]
        except json.JSONDecodeError:
            pass

        # Build a list of valid options as strings.
        if isinstance(response_options, range) or isinstance(response_options, list):
            options = [str(opt) for opt in response_options]
        else:
            return True

        # Check if any valid option appears as a whole word inside the response.
        for opt in options:
            pattern = r"\b" + re.escape(opt) + r"\b"
            if re.search(pattern, response):
                return True
        return False

    def _insert_formatting_instruction(
        self, generate_speculation_score: bool, format_response: bool
    ) -> None:
        """
        Appends a formatting instruction to the most recent facilitator message in the message history,
        based on the specified flags.

        If the last message in `self.message_history` starts with "facilitator", this method modifies its
        content by appending a formatting instruction according to the following logic:

        - If both `generate_speculation_score` and `format_response` are True, instructs to reply with a
          JSON object containing `response`, `reasoning`, and `speculation_score` keys.
        - If only `generate_speculation_score` is True, instructs to reply in prose and append a
          speculation score on a new line.
        - If only `format_response` is True, instructs to reply with a JSON object containing `response`
          and `reasoning` keys.
        - If neither flag is True, does nothing.

        Parameters:
            generate_speculation_score (bool): Whether to include instructions for a speculation score.
            format_response (bool): Whether to require the response in a specific JSON format.

        Returns:
            None
        """
        # Only alter the most recent facilitator message.
        if not self.message_history[-1]["content"].startswith("facilitator"):
            return None

        if generate_speculation_score and format_response:
            formatting_instruction = (
                "\n\n---\n"
                "✱ **Formatting Instruction** ✱\n"
                "Reply **only** with a valid JSON object containing exactly three keys:\n"
                "  • `response` (string)  – your main answer.\n"
                "  • `reasoning` (string) – concise explanation of how you arrived at the answer in first person narration.\n"
                "  • `speculation_score` (integer 0‑100) – how speculative the answer is, "
                "where 0 = not speculative at all and 100 = entirely speculative.\n"
            )

        elif generate_speculation_score:
            formatting_instruction = (
                "\n\n---\n"
                "✱ **Formatting Instruction** ✱\n"
                "Write your response in normal prose, then on a new line append:\n"
                "`Speculation Score: <integer 0‑100>`\n"
            )

        elif format_response:
            formatting_instruction = (
                "\n\n---\n"
                "✱ **Formatting Instruction** ✱\n"
                "Reply **only** with a valid JSON object containing exactly two keys:\n"
                "  • `response`  (string) – your main answer.\n"
                "  • `reasoning` (string) – concise explanation of how you arrived at the answer in first person narration.\n"
            )

        else:
            return None

        # Update the facilitator’s instruction with additional formatting instructions.
        self.message_history[-1]["content"] += formatting_instruction

    def respond(
        self,
        latest_message_history: list[dict],
        validate_response: bool = False,
        response_options: Any = [],
        generate_speculation_score: bool = False,
        format_response: bool = False,
    ) -> str:
        """
        Generates a response from the conversational synthetic subject based on the provided message history.

        Args:
            latest_message_history (list[dict]): The latest history of messages exchanged in the conversation.
            validate_response (bool, optional): If True, validates the generated response against the provided response_options. Defaults to False.
            response_options (Any, optional): Options to validate the response against. Defaults to [].
            generate_speculation_score (bool, optional): If True, includes instructions to generate a speculation score in the response. Defaults to False.
            format_response (bool, optional): If True, formats the response according to specific instructions. Defaults to False.
            is_full_message_history (bool, optional): If True, treats the input from message_history as the full message history; otherwise, appends the input from message_history to
            its own message_history. Defaults to False.

        Returns:
            str: The generated response from the subject. Returns an empty string if an exception occurs during response generation.

        Raises:
            Exception: Logs any exception that occurs during response generation and returns an empty string.
        """
        latest_message_history = self._build_message_history(
            message_history=latest_message_history
        )
        self.message_history.extend(latest_message_history)

        try:
            self._insert_formatting_instruction(
                generate_speculation_score=generate_speculation_score,
                format_response=format_response,
            )

            if validate_response:  # Validate response based on response_options
                for _ in range(NUM_RETRY):
                    response = query_llm(
                        llm_client=self.llm_client,
                        model_info=self.model_info,
                        message_history=self.message_history,
                        temperature=self.temperature,
                    )
                    if response_options and self._validate_response(
                        response=response, response_options=response_options
                    ):
                        break

            else:  # Skip validation
                response = query_llm(
                    llm_client=self.llm_client,
                    model_info=self.model_info,
                    message_history=self.message_history,
                    temperature=self.temperature,
                )

            return response

        except Exception as e:
            # Log the exception
            print(
                f"Error during response generation by ConversationalSyntheticSubject object: {e}"
            )
            return ""
