import openai
import pandas as pd
from talkingtomachines.generative.llm import query_llm


def generate_profile_prompt(
    profile_info: dict,
    build_profile_qna: bool,
    build_profile_backstories: bool,
    llm_client: openai.OpenAI = None,
    model_info: str = None,
    temperature: float = None,
) -> str:
    """
    Generates a profile prompt based on the provided profile information and options.
    This function constructs a prompt string that can include a Q&A section, a backstory section,
    or both, depending on the specified parameters. It uses an optional language model client
    to generate backstories if required.

    Args:
        profile_info (dict): A dictionary containing profile questions as keys and their
            corresponding responses as values. Questions with the key "ID" or empty/NaN
            responses are ignored.
        build_profile_qna (bool): If True, includes a Q&A section in the generated prompt.
        build_profile_backstories (bool): If True, generates and includes a backstory section
            in the prompt using the language model client.
        llm_client (openai.OpenAI, optional): The language model client used to generate
            backstories. Required if `build_profile_backstories` is True.
        model_info (str, optional): The model information for the language model client.
            Required if `build_profile_backstories` is True.
        temperature (float, optional): The temperature setting for the language model client.
            Required if `build_profile_backstories` is True.

    Returns:
        str: The generated profile prompt, which may include a Q&A section, a backstory section,
            or both, depending on the specified parameters. Returns an empty string if no
            sections are requested or if an error occurs.

    Raises:
        ValueError: If `build_profile_backstories` is True but `llm_client`, `model_info`, or
            `temperature` is not provided.
    """
    try:
        qna_profile_prompt = "Prior to this study, you are asked to complete some interview questions about your profile, which are provided below. Each question starts with 'Interviewer:', and your response is preceded by 'Me:'\n"
        counter = 1
        for question, response in profile_info.items():
            if question == "ID" or pd.isna(response) or not response:
                continue
            qna_profile_prompt += f"{counter}) Interviewer: {question} Me: {response} "
            counter += 1

        if build_profile_backstories:
            if llm_client is None or model_info is None or temperature is None:
                raise ValueError(
                    "llm_client, model_info, and temperature must be provided when build_profile_backstories is set to True."
                )

            backstory_profile_prompt = generate_backstories(
                system_prompt=qna_profile_prompt,
                llm_client=llm_client,
                model_info=model_info,
                temperature=temperature,
            )

        if build_profile_qna and build_profile_backstories:
            return f"{qna_profile_prompt}\n\nYou are also asked to provide a detailed backstory about yourself, which is provided below:\n{backstory_profile_prompt}"

        elif build_profile_qna:
            return qna_profile_prompt

        elif build_profile_backstories:
            return f"Prior to this study, you are asked to provide a detailed backstory about yourself, which is provided below:\n{backstory_profile_prompt}"

        else:
            return ""

    except Exception as e:
        print(
            f"Error encountered when generating the subject's profile prompt: {e}. Returning an empty string."
        )
        return ""


def generate_backstories(
    system_prompt: str, llm_client: openai.OpenAI, model_info: str, temperature: float
) -> str:
    """
    Generate a backstory using a language model based on the provided system prompt.

    Args:
        system_prompt (str): The initial system-level prompt to guide the language model's behavior.
        llm_client (openai.OpenAI): The OpenAI client instance used to interact with the language model.
        model_info (str): The identifier or configuration of the language model to be used.
        temperature (float): The sampling temperature to control the randomness of the model's output.

    Returns:
        str: The generated backstory in first-person narration.
    """
    message_history = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Create a backstory based on the information provided and describe it in detail in first person narration. Start generating the backstory immediately starting with 'I...'.",
        },
    ]

    backstory_prompt = query_llm(
        llm_client=llm_client,
        model_info=model_info,
        message_history=message_history,
        temperature=temperature,
    )

    return backstory_prompt


def generate_subject_system_message(
    role_description: str,
    profile_prompt: str,
) -> str:
    """Constructs system message for subjects by combining role description and profile_prompt, in that order.

    Args:
        role_description (str): A description of the subject's role.
        profile_prompt (str): The profile information of the synthetic subject generated by the generate_profile_prompt function.

    Returns:
        str: The constructed conversational system message for the synthetic subject.
    """
    return f"{role_description}\n\n{profile_prompt}"


def generate_session_system_message(experiment_context: str) -> str:
    """Constructs system message for sessions by providing the experiment context.

    Args:
        experiment_context (str): The context of the experiment.

    Returns:
        str: The constructed conversational system message.
    """
    return f"{experiment_context}"
