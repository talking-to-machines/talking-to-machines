from typing import List
from openai import OpenAI

openai_client = OpenAI()


def query_llm(model_info: str, message_history: List[dict]) -> str:
    """ 
    Queries a LLM for a response based on the latest message history.

    Parameters:
        model_info (str): Information about the model.
        message_history (List[dict]): Contains the history of message exchanged between user and assistant.

    Returns:
        str: Response from the LLM.
    """
    if model_info in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]:
        return query_open_ai(model_info=model_info, message_history=message_history)
    else:
        # Log the exception
        print(f"Model type {model_info} is not supported.")
        return None


def query_open_ai(model_info: str, messsage_history: List[dict]) -> str:
    """
    Query OpenAI API with the provided prompt.
    
    Parameters:
        model_info (str): Information about the model.
        message_history (List[dict]): Contains the history of message exchanged between user and assistant.

    Returns:
        str: Response from the LLM.
    """
    try:
        response = openai_client.chat.completions.create(
            model=model_info, messages=messsage_history
        )
        return response.choices[0].message.content

    except Exception as e:
        # Log the exception
        print(f"Error during OpenAI API call: {e}")
        return ""


def query_anthropic(model_info: str, messsage_history: List[dict], prompt: str) -> str:
    """Query Anthropic for the provided prompt."""
    try:
        # Implement integration with Anthropic
        pass
    except Exception as e:
        # Log the exception
        print(f"Error during Anthropic integration: {e}")
        return ""


def integrate_with_mistral(
    model_info: str, messsage_history: List[dict], prompt: str
) -> str:
    """Query Mistral for the provided prompt."""
    try:
        # Implement integration with Mistral
        pass
    except Exception as e:
        # Log the exception
        print(f"Error during Mistral integration: {e}")
        return ""


def integrate_with_meta(
    model_info: str, messsage_history: List[dict], prompt: str
) -> str:
    """Query Meta for the provided prompt."""
    try:
        # Implement integration with Meta
        pass
    except Exception as e:
        # Log the exception
        print(f"Error during Meta integration: {e}")
        return ""
