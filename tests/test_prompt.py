from talkingtomachines.generative.prompt import (
    generate_demographic_prompt,
    generate_conversational_system_message,
)


def test_generate_demographic_prompt_valid_input():
    demographic_info = {
        "What is your name?": "Alice",
        "How old are you?": "30",
        "Where do you live?": "Wonderland",
    }
    expected_output = "Interviewer: What is your name? Me: Alice Interviewer: How old are you? Me: 30 Interviewer: Where do you live? Me: Wonderland "
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


def test_generate_conversational_system_message_valid_input():
    # Test with typical input
    experiment_context = "This is the experiment context."
    demographic_info = "Demographic info: Age 25, Gender Female."
    assigned_treatment = "Assigned treatment: Control group."
    expected_output = "This is the experiment context.\n\nDemographic info: Age 25, Gender Female.\n\nAssigned treatment: Control group."
    assert (
        generate_conversational_system_message(
            experiment_context, demographic_info, assigned_treatment
        )
        == expected_output
    )


def test_generate_conversational_system_message_empty_string():
    # Test with empty strings
    experiment_context = ""
    demographic_info = ""
    assigned_treatment = ""
    expected_output = "\n\n\n\n"
    assert (
        generate_conversational_system_message(
            experiment_context, demographic_info, assigned_treatment
        )
        == expected_output
    )


def test_generate_conversational_system_message_long_string():
    # Test with long strings
    experiment_context = "A" * 1000
    demographic_info = "B" * 1000
    assigned_treatment = "C" * 1000
    expected_output = f"{'A' * 1000}\n\n{'B' * 1000}\n\n{'C' * 1000}"
    assert (
        generate_conversational_system_message(
            experiment_context, demographic_info, assigned_treatment
        )
        == expected_output
    )


def test_generate_conversational_system_message_special_characters():
    # Test with special characters
    experiment_context = "Experiment context with special chars: @#$%^&*()"
    demographic_info = "Demographic info with special chars: []{}|;:'\",.<>/?"
    assigned_treatment = "Treatment with special chars: ~`!"
    expected_output = "Experiment context with special chars: @#$%^&*()\n\nDemographic info with special chars: []{}|;:'\",.<>/?\n\nTreatment with special chars: ~`!"
    assert (
        generate_conversational_system_message(
            experiment_context, demographic_info, assigned_treatment
        )
        == expected_output
    )
