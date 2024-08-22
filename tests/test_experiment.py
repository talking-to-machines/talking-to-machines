import pytest
import pandas as pd
from talkingtomachines.management.experiment import (
    Experiment,
    AIConversationalExperiment,
    AItoAIConversationalExperiment,
    AItoAIInterviewExperiment,
)


@pytest.fixture
def experiment():
    return Experiment()


def test_generate_experiment_id(experiment):
    experiment_id = experiment.generate_experiment_id()
    assert isinstance(experiment_id, str)
    assert len(experiment_id) == 15


def test_get_experiment_id(experiment):
    experiment_id = experiment.get_experiment_id()
    assert isinstance(experiment_id, str)
    assert len(experiment_id) == 15


def test_experiment_initialization():
    experiment = Experiment()
    assert experiment is not None


def test_ai_conversational_experiment_initialization():
    agent_demographics = pd.DataFrame({"ID": [1, 2, 3], "Age": [25, 30, 35]})
    experiment = AIConversationalExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    assert experiment is not None
    assert experiment.get_model_info() == "gpt-4o"
    assert experiment.get_experiment_context() == "Testing"
    assert experiment.get_agent_demographics().equals(agent_demographics)
    assert experiment.get_max_conversation_length() == 10
    assert experiment.get_treatments() == {
        "treatment1": "value1",
        "treatment2": "value2",
    }
    assert experiment.get_treatment_assignment_strategy() == "simple_random"


def test_ai_conversational_experiment_check_model_info():
    agent_demographics = pd.DataFrame({"ID": [1, 2, 3], "Age": [25, 30, 35]})
    experiment = AIConversationalExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    assert experiment.check_model_info("gpt-4o") == "gpt-4o"
    assert experiment.check_model_info("gpt-3.5-turbo") == "gpt-3.5-turbo"
    with pytest.raises(ValueError):
        experiment.check_model_info("invalid_model")


def test_ai_conversational_experiment_check_treatment_assignment_strategy():
    agent_demographics = pd.DataFrame({"ID": [1, 2, 3], "Age": [25, 30, 35]})
    experiment = AIConversationalExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    assert (
        experiment.check_treatment_assignment_strategy("simple_random")
        == "simple_random"
    )
    assert (
        experiment.check_treatment_assignment_strategy("full_factorial")
        == "full_factorial"
    )
    with pytest.raises(ValueError):
        experiment.check_treatment_assignment_strategy("invalid_strategy")


def test_ai_conversational_experiment_check_agent_demographics():
    agent_demographics = pd.DataFrame({"ID": [1, 2, 3], "Age": [25, 30, 35]})
    experiment = AIConversationalExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    assert experiment.check_agent_demographics(agent_demographics).equals(
        agent_demographics
    )
    with pytest.raises(ValueError):
        experiment.check_agent_demographics(pd.DataFrame())


def test_ai_conversational_experiment_check_max_conversation_length():
    agent_demographics = pd.DataFrame({"ID": [1, 2, 3], "Age": [25, 30, 35]})
    experiment = AIConversationalExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    assert experiment.check_max_conversation_length(10) == 10
    assert experiment.check_max_conversation_length(5) == 5
    with pytest.raises(ValueError):
        experiment.check_max_conversation_length(3)


def test_ai_conversational_experiment_check_treatments():
    agent_demographics = pd.DataFrame({"ID": [1, 2, 3], "Age": [25, 30, 35]})
    experiment = AIConversationalExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    treatments = {"treatment1": "value1", "treatment2": "value2"}
    assert experiment.check_treatments(treatments) == treatments
    assert experiment.check_treatments({}) == {}


def test_ai_conversational_experiment_getters():
    agent_demographics = pd.DataFrame({"ID": [1, 2, 3], "Age": [25, 30, 35]})
    experiment = AIConversationalExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    assert experiment.get_model_info() == "gpt-4o"
    assert experiment.get_experiment_context() == "Testing"
    assert experiment.get_agent_demographics().equals(agent_demographics)
    assert experiment.get_max_conversation_length() == 10
    assert experiment.get_treatments() == {
        "treatment1": "value1",
        "treatment2": "value2",
    }
    assert experiment.get_treatment_assignment_strategy() == "simple_random"


def test_ai_to_ai_conversational_experiment_initialization():
    agent_demographics = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        }
    )
    agent_roles = {"agent1": "Role 1", "agent2": "Role 2"}
    experiment = AItoAIConversationalExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        agent_roles=agent_roles,
        num_agents_per_session=2,
        num_sessions=5,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    assert experiment is not None
    assert experiment.get_model_info() == "gpt-4o"
    assert experiment.get_experiment_context() == "Testing"
    assert experiment.get_agent_demographics().equals(agent_demographics)
    assert experiment.get_max_conversation_length() == 10
    assert experiment.get_treatments() == {
        "treatment1": "value1",
        "treatment2": "value2",
    }
    assert experiment.get_treatment_assignment_strategy() == "simple_random"
    assert experiment.get_num_sessions() == 5
    assert experiment.get_num_agents_per_session() == 2
    assert experiment.get_agent_roles() == agent_roles


def test_ai_to_ai_conversational_experiment_check_num_sessions():
    agent_demographics = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        }
    )
    agent_roles = {"agent1": "Role 1", "agent2": "Role 2"}
    experiment = AItoAIConversationalExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        agent_roles=agent_roles,
        num_agents_per_session=2,
        num_sessions=5,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    assert experiment.check_num_sessions(10) == 10
    assert experiment.check_num_sessions(5) == 5
    with pytest.raises(ValueError):
        experiment.check_num_sessions(0)


def test_ai_to_ai_conversational_experiment_check_num_agents_per_session():
    agent_demographics = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        }
    )
    agent_roles = {"agent1": "Role 1", "agent2": "Role 2"}
    experiment = AItoAIConversationalExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        agent_roles=agent_roles,
        num_agents_per_session=2,
        num_sessions=5,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    assert experiment.check_num_agents_per_session(2) == 2
    with pytest.raises(ValueError):
        experiment.check_num_agents_per_session(3)


def test_ai_to_ai_conversational_experiment_check_agent_roles():
    agent_demographics = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        }
    )
    agent_roles = {"agent1": "Role 1", "agent2": "Role 2"}
    experiment = AItoAIConversationalExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        agent_roles=agent_roles,
        num_agents_per_session=2,
        num_sessions=5,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    assert experiment.check_agent_roles(agent_roles) == agent_roles
    with pytest.raises(ValueError):
        experiment.check_agent_roles({"agent1": "Role 1"})


def test_ai_to_ai_conversational_experiment_assign_treatment():
    agent_demographics = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        }
    )
    agent_roles = {"agent1": "Role 1", "agent2": "Role 2"}
    experiment = AItoAIConversationalExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        agent_roles=agent_roles,
        num_agents_per_session=2,
        num_sessions=5,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    treatment_assignment = experiment.assign_treatment()
    assert isinstance(treatment_assignment, dict)
    assert len(treatment_assignment) == 5


def test_ai_to_ai_conversational_experiment_assign_agents_to_session():
    agent_demographics = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        }
    )
    agent_roles = {"agent1": "Role 1", "agent2": "Role 2"}
    experiment = AItoAIConversationalExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        agent_roles=agent_roles,
        num_agents_per_session=2,
        num_sessions=5,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    agent_assignment = experiment.assign_agents_to_session()
    assert isinstance(agent_assignment, dict)
    assert len(agent_assignment) == 5


def test_ai_to_ai_conversational_experiment_initialize_agents():
    agent_demographics = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        }
    )
    agent_roles = {"agent1": "Role 1", "agent2": "Role 2"}
    experiment = AItoAIConversationalExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        agent_roles=agent_roles,
        num_agents_per_session=2,
        num_sessions=5,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    session_info = {
        "session_id": 1,
        "treatment": "treatment1",
        "agents": [1, 2],
        "agents_demographic": [{"ID": 1, "Age": 25}, {"ID": 2, "Age": 30}],
    }
    agents = experiment.initialize_agents(session_info)
    assert isinstance(agents, list)
    assert len(agents) == 2


def test_ai_to_ai_interview_experiment_initialization():
    agent_demographics = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        }
    )
    agent_roles = {"Interviewer": "Role 1", "agent2": "Role 2"}
    experiment = AItoAIInterviewExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        agent_roles=agent_roles,
        num_agents_per_session=2,
        num_sessions=5,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    assert experiment is not None
    assert experiment.get_model_info() == "gpt-4o"
    assert experiment.get_experiment_context() == "Testing"
    assert experiment.get_agent_demographics().equals(agent_demographics)
    assert experiment.get_max_conversation_length() == 10
    assert experiment.get_treatments() == {
        "treatment1": "value1",
        "treatment2": "value2",
    }
    assert experiment.get_treatment_assignment_strategy() == "simple_random"
    assert experiment.get_num_sessions() == 5
    assert experiment.get_num_agents_per_session() == 2
    assert experiment.get_agent_roles() == agent_roles


def test_ai_to_ai_interview_experiment_check_num_sessions():
    agent_demographics = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        }
    )
    agent_roles = {"Interviewer": "Role 1", "agent2": "Role 2"}
    experiment = AItoAIInterviewExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        agent_roles=agent_roles,
        num_agents_per_session=2,
        num_sessions=5,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    assert experiment.check_num_sessions(10) == 10
    assert experiment.check_num_sessions(5) == 5
    with pytest.raises(ValueError):
        experiment.check_num_sessions(0)


def test_ai_to_ai_interview_experiment_check_num_agents_per_session():
    agent_demographics = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        }
    )
    agent_roles = {"Interviewer": "Role 1", "agent2": "Role 2"}
    experiment = AItoAIInterviewExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        agent_roles=agent_roles,
        num_agents_per_session=2,
        num_sessions=10,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    assert experiment.check_num_agents_per_session(2) == 2
    with pytest.raises(ValueError):
        experiment.check_num_agents_per_session(1)
    with pytest.raises(ValueError):
        experiment.check_num_agents_per_session(3)


def test_ai_to_ai_interview_experiment_check_agent_roles():
    agent_demographics = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        }
    )
    agent_roles = {"Interviewer": "Role 1", "agent2": "Role 2"}
    experiment = AItoAIInterviewExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        agent_roles=agent_roles,
        num_agents_per_session=2,
        num_sessions=5,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    assert experiment.check_agent_roles(agent_roles) == agent_roles
    with pytest.raises(ValueError):
        experiment.check_agent_roles({"agent1": "Role 1"})


def test_ai_to_ai_interview_experiment_assign_treatment():
    agent_demographics = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        }
    )
    agent_roles = {"Interviewer": "Role 1", "agent2": "Role 2"}
    experiment = AItoAIInterviewExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        agent_roles=agent_roles,
        num_agents_per_session=2,
        num_sessions=5,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    treatment_assignment = experiment.assign_treatment()
    assert isinstance(treatment_assignment, dict)
    assert len(treatment_assignment) == 5


def test_ai_to_ai_interview_experiment_assign_agents_to_session():
    agent_demographics = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        }
    )
    agent_roles = {"Interviewer": "Role 1", "agent2": "Role 2"}
    experiment = AItoAIInterviewExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        agent_roles=agent_roles,
        num_agents_per_session=2,
        num_sessions=5,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    agent_assignment = experiment.assign_agents_to_session()
    assert isinstance(agent_assignment, dict)
    assert len(agent_assignment) == 5


def test_ai_to_ai_interview_experiment_initialize_agents():
    agent_demographics = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        }
    )
    agent_roles = {"Interviewer": "Role 1", "agent2": "Role 2"}
    experiment = AItoAIInterviewExperiment(
        model_info="gpt-4o",
        experiment_context="Testing",
        agent_demographics=agent_demographics,
        agent_roles=agent_roles,
        num_agents_per_session=2,
        num_sessions=5,
        max_conversation_length=10,
        treatments={"treatment1": "value1", "treatment2": "value2"},
        treatment_assignment_strategy="simple_random",
    )
    session_info = {
        "session_id": 1,
        "treatment": "treatment1",
        "agents": [1],
        "agents_demographic": [{"ID": 1, "Age": 25}],
    }
    agents = experiment.initialize_agents(session_info)
    assert isinstance(agents, list)
    assert len(agents) == len(session_info["agents"]) + 1
