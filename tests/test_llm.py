from talkingtomachines.generative.llm import query_llm, query_open_ai, openai_client
from unittest.mock import patch, MagicMock


def test_query_llm_supported_models(mocker):
    # Mock the query_open_ai function
    mocker.patch(
        "talkingtomachines.generative.llm.query_open_ai", return_value="Mock response"
    )

    supported_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    ]
    message_history = [{"role": "user", "content": "Hello!"}]

    for model in supported_models:
        response = query_llm(model_info=model, message_history=message_history)
        assert response == "Mock response"


def test_query_llm_unsupported_model():
    unsupported_model = "unsupported-model"
    message_history = [{"role": "user", "content": "Hello!"}]
    response = query_llm(model_info=unsupported_model, message_history=message_history)
    assert response == ""


def test_query_llm_empty_message_history(mocker):
    # Mock the query_open_ai function
    mocker.patch(
        "talkingtomachines.generative.llm.query_open_ai", return_value="Mock response"
    )

    model_info = "gpt-4"
    message_history = []
    response = query_llm(model_info=model_info, message_history=message_history)
    assert response == "Mock response"


def test_query_llm_various_message_histories(mocker):
    # Mock the query_open_ai function
    mocker.patch(
        "talkingtomachines.generative.llm.query_open_ai", return_value="Mock response"
    )

    model_info = "gpt-3.5-turbo"
    message_histories = [
        [{"role": "user", "content": "Hello!"}],
        [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        [
            {"role": "user", "content": "What is the weather today?"},
            {"role": "assistant", "content": "It's sunny."},
            {"role": "user", "content": "Great!"},
        ],
    ]

    for message_history in message_histories:
        response = query_llm(model_info=model_info, message_history=message_history)
        assert response == "Mock response"


def test_query_open_ai_success():
    model_info = "gpt-4"
    message_history = [{"role": "user", "content": "Hello, how are you?"}]

    # Mocking the response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "I am fine, thank you."

    with patch.object(
        openai_client.chat.completions, "create", return_value=mock_response
    ):
        result = query_open_ai(model_info, message_history)
        assert result == "I am fine, thank you."


def test_query_open_ai_invalid_model():
    model_info = "invalid-model"
    message_history = [{"role": "user", "content": "Hello, how are you?"}]

    with patch.object(openai_client.chat.completions, "create") as mock_create:
        mock_create.side_effect = Exception("Model not found")
        result = query_open_ai(model_info, message_history)
        assert result == ""
        mock_create.assert_called_once_with(model=model_info, messages=message_history)


def test_query_open_ai_exception():
    model_info = "gpt-4"
    message_history = [{"role": "user", "content": "Hello, how are you?"}]

    with patch.object(openai_client.chat.completions, "create") as mock_create:
        mock_create.side_effect = Exception("API call failed")
        result = query_open_ai(model_info, message_history)
        assert result == ""
        mock_create.assert_called_once_with(model=model_info, messages=message_history)
