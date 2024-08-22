import pytest
from unittest.mock import Mock, patch
from talkingtomachines.generative.synthetic_agent import (
    SyntheticAgent,
    DemographicInfo,
    ConversationalSyntheticAgent,
)


def test_synthetic_agent():
    # Create a sample demographic info
    demographic_info = {"age": 30, "gender": "male", "occupation": "engineer"}

    # Create a synthetic agent instance
    agent = SyntheticAgent(
        experiment_id="123",
        experiment_context="context",
        session_id=1,
        demographic_info=demographic_info,
        model_info="model",
    )

    # Test the getter methods
    assert agent.get_experiment_id() == "123"
    assert agent.get_experiment_context() == "context"
    assert agent.get_session_id() == 1
    assert (
        agent.get_demographic_info()
        == "1) Interviewer: age Me: 30 2) Interviewer: gender Me: male 3) Interviewer: occupation Me: engineer "
    )
    assert agent.get_model_info() == "model"

    # Test the to_dict() method
    agent_dict = agent.to_dict()
    assert agent_dict["experiment_id"] == "123"
    assert agent_dict["experiment_context"] == "context"
    assert agent_dict["session_id"] == 1
    assert (
        agent_dict["demographic_info"]
        == "1) Interviewer: age Me: 30 2) Interviewer: gender Me: male 3) Interviewer: occupation Me: engineer "
    )
    assert agent_dict["model_info"] == "model"

    # Test the respond() method
    response = agent.respond()
    assert isinstance(response, str)


def test_conversational_synthetic_agent():
    # Create a sample demographic info
    demographic_info = {"age": 30, "gender": "male", "occupation": "engineer"}

    # Create a conversational synthetic agent instance
    agent = ConversationalSyntheticAgent(
        experiment_id="123",
        experiment_context="context",
        session_id=1,
        demographic_info=demographic_info,
        role="assistant",
        role_description="AI assistant",
        model_info="model",
        treatment="treatment",
    )

    # Test the getter methods
    assert agent.get_experiment_id() == "123"
    assert agent.get_experiment_context() == "context"
    assert agent.get_session_id() == 1
    assert (
        agent.get_demographic_info()
        == "1) Interviewer: age Me: 30 2) Interviewer: gender Me: male 3) Interviewer: occupation Me: engineer "
    )
    assert agent.get_model_info() == "model"
    assert agent.get_role() == "assistant"
    assert agent.get_role_description() == "AI assistant"
    assert agent.get_treatment() == "treatment"

    # Test the to_dict() method
    agent_dict = agent.to_dict()
    assert agent_dict["experiment_id"] == "123"
    assert agent_dict["experiment_context"] == "context"
    assert agent_dict["session_id"] == 1
    assert (
        agent_dict["demographic_info"]
        == "1) Interviewer: age Me: 30 2) Interviewer: gender Me: male 3) Interviewer: occupation Me: engineer "
    )
    assert agent_dict["model_info"] == "model"
    assert agent_dict["role"] == "assistant"
    assert agent_dict["role_description"] == "AI assistant"
    assert agent_dict["treatment"] == "treatment"

    # Test the respond() method
    response = agent.respond("How can I assist you?")
    assert isinstance(response, str)
