from talkingtomachines.generative.prompt import (
    generate_demographic_prompt,
    generate_conversational_agent_system_message,
    generate_conversational_session_system_message,
)


def test_generate_demographic_prompt_valid_input():
    demographic_info = {
        "What is your name?": "Alice",
        "How old are you?": "30",
        "Where do you live?": "Wonderland",
    }
    expected_output = "1) Interviewer: What is your name? Me: Alice 2) Interviewer: How old are you? Me: 30 3) Interviewer: Where do you live? Me: Wonderland "
    assert generate_demographic_prompt(demographic_info) == expected_output


def test_generate_demographic_prompt_empty_input():
    demographic_info = {}
    expected_output = ""
    assert generate_demographic_prompt(demographic_info) == expected_output


def test_generate_demographic_prompt_none_input():
    demographic_info = None
    expected_output = ""
    assert generate_demographic_prompt(demographic_info) == expected_output


def test_generate_demographic_prompt_invalid_input():
    demographic_info = "invalid input"
    expected_output = ""
    assert generate_demographic_prompt(demographic_info) == expected_output


def test_generate_conversational_agent_system_message_valid_input():
    experiment_context = "Experiment A"
    treatment = "Treatment 1"
    role_description = "Interviewer"
    demographic_info = "Alice"
    expected_output = "Experiment A\n\nTreatment 1\n\nInterviewer\n\nAlice"
    assert (
        generate_conversational_agent_system_message(
            experiment_context, treatment, role_description, demographic_info
        )
        == expected_output
    )


def test_generate_conversational_agent_system_message_empty_input():
    experiment_context = ""
    treatment = ""
    role_description = ""
    demographic_info = ""
    expected_output = "\n\n\n\n\n\n"
    assert (
        generate_conversational_agent_system_message(
            experiment_context, treatment, role_description, demographic_info
        )
        == expected_output
    )


def test_generate_conversational_agent_system_message_none_input():
    experiment_context = None
    treatment = None
    role_description = None
    demographic_info = None
    expected_output = "None\n\nNone\n\nNone\n\nNone"
    assert (
        generate_conversational_agent_system_message(
            experiment_context, treatment, role_description, demographic_info
        )
        == expected_output
    )


def test_generate_conversational_session_system_message_valid_input():
    experiment_context = "Experiment A"
    treatment = "Treatment 1"
    expected_output = "Experiment A\n\nTreatment 1"
    assert (
        generate_conversational_session_system_message(experiment_context, treatment)
        == expected_output
    )


def test_generate_conversational_session_system_message_empty_input():
    experiment_context = ""
    treatment = ""
    expected_output = "\n\n"
    assert (
        generate_conversational_session_system_message(experiment_context, treatment)
        == expected_output
    )


def test_generate_conversational_session_system_message_none_input():
    experiment_context = None
    treatment = None
    expected_output = "None\n\nNone"
    assert (
        generate_conversational_session_system_message(experiment_context, treatment)
        == expected_output
    )
