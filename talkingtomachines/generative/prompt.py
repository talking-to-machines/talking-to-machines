def generate_demographic_prompt(demographic_info: dict) -> str:
    """Formats the demographic information of a synthetic subject into a prompt.

    Args:
        demographic_info (dict): A dictionary containing the demographic information of the synthetic subject.

    Returns:
        str: The formatted demographic information as a prompt.
    """
    try:
        demographic_prompt = "Your demographic profile: "
        counter = 1
        for question, response in demographic_info.items():
            if question == "ID":
                continue
            demographic_prompt += f"{counter}) Interviewer: {question} Me: {response} "
            counter += 1
        return demographic_prompt

    except Exception as e:
        # Log the exception
        print(
            f"Error encountered when generating demographic prompt: {e}. Returning empty string."
        )
        return ""


def generate_conversational_agent_system_message(
    experiment_context: str,
    treatment: str,
    role_description: str,
    demographic_info: str,
) -> str:
    """Constructs system message for conversational agents by combining experiment_context, treatment, role description, and demographic_info.

    Args:
        experiment_context (str): The context of the experiment.
        treatment (str): The treatment that is assigned to the session.
        role_description (str): A description of the agent's role.
        demographic_info (str): The demographic information of the synthetic subject.

    Returns:
        str: The constructed conversational system message.
    """
    return f"{experiment_context}\n\n{role_description}\n\n{demographic_info}\n\n{treatment}"


def generate_conversational_session_system_message(
    experiment_context: str, treatment: str
) -> str:
    """Constructs system message for conversational sessions by combining experiment_context and treatment.

    Args:
        experiment_context (str): The context of the experiment.
        treatment (str): The treatment that is assigned to the session.

    Returns:
        str: The constructed conversational system message.
    """
    return f"{experiment_context}\n\n{treatment}"
