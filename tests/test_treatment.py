from talkingtomachines.management.treatment import (
    simple_random_assignment_session,
    complete_random_assignment_session,
    full_factorial_assignment_session,
)
from itertools import product


def test_simple_random_assignment_session():
    treatment_labels = ["A", "B", "C"]
    num_sessions = 10

    assignments = simple_random_assignment_session(treatment_labels, num_sessions)

    # Check if the number of sessions matches the expected number
    assert len(assignments) == num_sessions

    # Check if all sessions have valid treatment labels
    for session, treatment in assignments.items():
        assert treatment in treatment_labels

    # Check if the assignments are random
    assert len(set(assignments.values())) > 1


def test_simple_random_assignment_session_empty_labels():
    treatment_labels = []
    num_sessions = 10

    assignments = simple_random_assignment_session(treatment_labels, num_sessions)
    assert assignments[0] == ""


def test_complete_random_assignment_session():
    treatment_labels = ["A", "B", "C"]
    num_sessions = 10

    assignments = complete_random_assignment_session(treatment_labels, num_sessions)

    # Check if the number of sessions matches the expected number
    assert len(assignments) == num_sessions

    # Check if all sessions have valid treatment labels
    for session, treatment in assignments.items():
        assert treatment in treatment_labels
        assert treatment == treatment_labels[session % len(treatment_labels)]


def test_complete_random_assignment_session_empty_labels():
    treatment_labels = []
    num_sessions = 10

    assignments = complete_random_assignment_session(treatment_labels, num_sessions)
    for session, treatment in assignments.items():
        assert treatment == ""


def test_full_factorial_assignment_session():
    treatment_labels = [["A", "B"], ["X", "Y", "Z"], ["1", "2", "3"]]
    num_sessions = (
        len(treatment_labels[0]) * len(treatment_labels[1]) * len(treatment_labels[2])
    )

    assignments = full_factorial_assignment_session(treatment_labels, num_sessions)

    # Check if the number of sessions matches the expected number
    assert len(assignments) == num_sessions

    # Check if all sessions have valid treatment labels
    for session, treatment in assignments.items():
        assert len(treatment) == len(treatment_labels)
        for i, label in enumerate(treatment_labels):
            assert treatment[i] in label

    # Check if the assignments cover all possible combinations
    combinations = list(product(*treatment_labels))
    for combination in combinations:
        assert combination in assignments.values()


def test_full_factorial_assignment_session_empty_labels():
    treatment_labels = []
    num_sessions = 10

    assignments = full_factorial_assignment_session(treatment_labels, num_sessions)
    for session, treatment in assignments.items():
        assert treatment == ""
