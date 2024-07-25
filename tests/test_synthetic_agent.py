import pytest
from unittest.mock import Mock, patch
from talkingtomachines.generative.synthetic_agent import (
    SyntheticAgent,
    DemographicInfo,
    ConversationalSyntheticAgent,
)


@pytest.fixture
def demographic_info():
    # Replace with actual initialization of DemographicInfo
    return DemographicInfo(name="John Doe", age=30, gender="Male")


@pytest.fixture
def mock_demographic_prompt_generator():
    return Mock(return_value="Mock demographic prompt")


@pytest.fixture
def base_synthetic_agent(demographic_info, mock_demographic_prompt_generator):
    return SyntheticAgent(
        experiment_id="exp123",
        experiment_context="Test Context",
        session_id="sess456",
        demographic_info=demographic_info,
        model_info="model789",
        demographic_prompt_generator=mock_demographic_prompt_generator,
    )


def test_initialization_base(
    base_synthetic_agent, demographic_info, mock_demographic_prompt_generator
):
    assert base_synthetic_agent.experiment_id == "exp123"
    assert base_synthetic_agent.experiment_context == "Test Context"
    assert base_synthetic_agent.session_id == "sess456"
    assert base_synthetic_agent.demographic_info == "Mock demographic prompt"
    assert base_synthetic_agent.model_info == "model789"
    mock_demographic_prompt_generator.assert_called_once_with(demographic_info)


def test_get_experiment_id_base(base_synthetic_agent):
    assert base_synthetic_agent.get_experiment_id() == "exp123"


def test_get_session_id_base(base_synthetic_agent):
    assert base_synthetic_agent.get_session_id() == "exp123"


def test_get_experiment_context_base(base_synthetic_agent):
    assert base_synthetic_agent.get_experiment_context() == "Test Context"


def test_get_demographic_info_base(base_synthetic_agent):
    assert base_synthetic_agent.get_demographic_info() == "Mock demographic prompt"


def test_get_model_info_base(base_synthetic_agent):
    assert base_synthetic_agent.get_model_info() == "model789"


def test_respond_base(base_synthetic_agent):
    assert base_synthetic_agent.respond("What is your name?") == ""


@pytest.fixture
def mock_generate_conversational_system_message():
    return Mock(return_value="Mock system message")


@pytest.fixture
def conversational_synthetic_agent(
    demographic_info,
    mock_demographic_prompt_generator,
    mock_generate_conversational_system_message,
):
    with patch(
        "talkingtomachines.generative.prompt.generate_conversational_system_message",
        mock_generate_conversational_system_message,
    ):
        return ConversationalSyntheticAgent(
            experiment_id="exp123",
            experiment_context="Test Context",
            session_id="sess456",
            demographic_info=demographic_info,
            model_info="model789",
            assigned_treatment="treatmentA",
            demographic_prompt_generator=mock_demographic_prompt_generator,
        )


def test_initialization_conversational(conversational_synthetic_agent):
    assert conversational_synthetic_agent.experiment_id == "exp123"
    assert conversational_synthetic_agent.experiment_context == "Test Context"
    assert conversational_synthetic_agent.session_id == "sess456"
    assert conversational_synthetic_agent.demographic_info == "Mock demographic prompt"
    assert conversational_synthetic_agent.model_info == "model789"
    assert conversational_synthetic_agent.assigned_treatment == "treatmentA"
    assert (
        conversational_synthetic_agent.system_message
        == "Test Context\n\nMock demographic prompt\n\ntreatmentA"
    )
    assert conversational_synthetic_agent.message_history == [
        {
            "role": "system",
            "content": "Test Context\n\nMock demographic prompt\n\ntreatmentA",
        }
    ]


def test_get_assigned_treatment_conversational(conversational_synthetic_agent):
    assert conversational_synthetic_agent.get_assigned_treatment() == "treatmentA"


def test_get_system_message_conversational(conversational_synthetic_agent):
    assert (
        conversational_synthetic_agent.get_system_message()
        == "Test Context\n\nMock demographic prompt\n\ntreatmentA"
    )


def test_get_message_history_conversational(conversational_synthetic_agent):
    assert conversational_synthetic_agent.get_message_history() == [
        {
            "role": "system",
            "content": "Test Context\n\nMock demographic prompt\n\ntreatmentA",
        }
    ]


def test_update_message_history_conversational(conversational_synthetic_agent):
    conversational_synthetic_agent.update_message_history(
        message="Hello", message_type="user"
    )
    assert conversational_synthetic_agent.get_message_history()[-1] == {
        "role": "user",
        "content": "Hello",
    }
    conversational_synthetic_agent.update_message_history(
        message="Hi there", message_type="assistant"
    )
    assert conversational_synthetic_agent.get_message_history()[-1] == {
        "role": "assistant",
        "content": "Hi there",
    }


def test_update_message_history_invalid_type_conversational(
    conversational_synthetic_agent,
):
    assert (
        conversational_synthetic_agent.update_message_history(
            message="Hello", message_type="invalid"
        )
        is None
    )


def test_respond_conversational(conversational_synthetic_agent):
    with patch(
        "talkingtomachines.generative.llm.query_llm", return_value="Mock response"
    ):
        response = conversational_synthetic_agent.respond("What is your name?")
        assert response == ""


def test_respond_exception_handling_conversational(conversational_synthetic_agent):
    with patch(
        "talkingtomachines.generative.llm.query_llm", side_effect=Exception("API error")
    ):
        response = conversational_synthetic_agent.respond("What is your name?")
        assert response == ""
