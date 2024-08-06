# TODO:
#  - Backend functionalities:
#  -- agents generation
#  -- timeout handling
#  -- conversations retrieval
#  -- HTML generation (to pass conversations to templates)
#  -- data export?
#  - Frontend functionalities:
#  -- CSS rules for conversations
#  -- JS methods for conversations handling
#  - Chat interface:
#  -- AI-Humans
#  -- AI-AI
#  -- Humans-Humans (default option already built-in in oTree)


def get_chat_interface(experiment: dict) -> str:
    pass


def save(experiment: dict, model_to_link, conversation, extra_model) -> bool:
    # TODO:
    #  - move to relevant class
    #  - handle compatibility issue with model_to_link and ExtraModel (check oTree API)
    models_supported = [
        'Session', 'Participant', 'Player', 'Group', 'Subsession'
    ]
    model_to_link_name = type(self).__name__
    if model_to_link_name not in models_supported:
        raise TypeError(
            f"""The model to link must be of one of the following types: {', '.join(models_supported)}"""
        )

    model_to_link = {type(self).__name__: model_to_link}
    extra_model.create(**model_to_link, **conversation)


def integrate_with_otree(experiment: dict) -> bool:
    """Integrate with oTree for the given experiment."""
    try:
        # Implement oTree integration functionality
        pass
    except Exception as e:
        # Log the exception
        print(f"Error during oTree integration: {e}")
        return False
