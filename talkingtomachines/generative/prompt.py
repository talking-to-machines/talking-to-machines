def generate_demographic_prompt(demographic_info: dict) -> str:
    """Formats the demographic information of a synthetic subject into a prompt.

    Args:
        demographic_info (dict): A dictionary containing the demographic information of the synthetic subject.

    Returns:
        str: The formatted demographic information as a prompt.
    """
    try:
        demographic_prompt = ""
        for question, response in demographic_info.items():
            demographic_prompt += f"Interviewer: {question} Me: {response} "
        return demographic_prompt

    except Exception as e:
        # Log the exception
        print(f"Error encountered when generating demographic prompt: {e}")
        return ""


def generate_conversational_system_message(
    experiment_context: str, role: str, treatment: str, demographic_info: str
) -> str:
    """Constructs system message for conversational-based experiment by combining experiment_context, role, demographic_info and treatment.

    Args:
        experiment_context (str): The context of the experiment.
        role (str): The agent's role.
        treatment (str): The treatment assigned to the synthetic subject.
        demographic_info (str): The demographic information of the synthetic subject.

    Returns:
        str: The constructed conversational system message.
    """
    try:
        return f"{experiment_context}\n\n{role}\n\n{treatment}\n\n{demographic_info}"

    except Exception as e:
        # Log the exception
        print(f"Error encountered when generating conversational system message: {e}")
        return ""
