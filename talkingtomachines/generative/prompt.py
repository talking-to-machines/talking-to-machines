def generate_demographic_prompt(demographic_info: dict) -> str:
    """
    Formats the demographic information of a synthetic subject into a prompt.

    Parameters:
        demographic_info (dict): A pandas Series containing the demographic information of the synthetic subject.

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
    experiment_context: str, demographic_info: str, assigned_treatment: str
) -> str:
    """
    Constructs system message for conversational-based experiment by combining experiment_context, demographic_info and assigned_treatment.

    Parameters:
        experiment_context (str): The context of the experiment.
        demographic_info (str): The demographic information of the synthetic subject.
        assigned_treatment (str): The treatment assigned to the synthetic subject.

    Returns:
        str: The constructed conversational system message.
    """
    try:
        return f"{experiment_context}\n\n{demographic_info}\n\n{assigned_treatment}"

    except Exception as e:
        # Log the exception
        print(f"Error encountered when generating conversational system message: {e}")
        return ""
