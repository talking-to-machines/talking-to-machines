"""Experiment initialisation from prompt template dictionaries.

.. deprecated::
    This module is deprecated.  Use
    ``talkingtomachines.compiler.compiler.Compiler`` instead.

Generates all constant permutations and creates one
``AItoAIInterviewExperiment`` instance per permutation.
"""

import warnings

warnings.warn(
    "talkingtomachines.management.initialize_experiment is deprecated. "
    "Use talkingtomachines.compiler.compiler.Compiler instead.",
    DeprecationWarning,
    stacklevel=2,
)

import itertools
from talkingtomachines.management.experiment import (
    AItoAIInterviewExperiment,
    Constant,
)

__all__ = ["generate_permutations", "initialize_experiment"]


def generate_permutations(constants: dict) -> list:
    """
    Generate all possible permutations of a dictionary's values.

    This function takes a dictionary where the keys are constant names and the values
    are iterables of possible values for those constants. It generates all possible
    combinations (Cartesian product) of the values and returns a list of `Constant`
    objects initialized with each combination.

    Args:
        constants (dict): A dictionary where keys are strings representing constant names
                          and values are iterables of possible values for those constants.

    Returns:
        list: A list of `Constant` objects, each initialized with a unique combination
              of the input dictionary's values. If the input dictionary is empty, an
              empty list is returned.
    """
    if constants:
        keys, values = zip(*constants.items())
        constant_permutations = [
            Constant(**permutation)
            for permutation in [dict(zip(keys, v)) for v in itertools.product(*values)]
        ]
        return constant_permutations
    else:
        return [Constant()]


def initialize_experiment(prompt_template_dict: dict) -> tuple[list, list]:
    """Initialise AI-to-AI interview experiments from prompt template data.

    Creates one ``AItoAIInterviewExperiment`` for every constant
    permutation derived from the ``"constants"`` key in the template
    dictionary.

    Args:
        prompt_template_dict: A dictionary containing all prompt template
            data (settings, treatments, roles, prompts, profiles, and
            constants) as extracted from the Excel workbook.

    Returns:
        tuple[list, list]: A two-element tuple containing a list of
            initialised ``AItoAIInterviewExperiment`` objects (one per
            constant permutation) and a list of ``Constant`` objects
            representing each permutation.
    """
    # Define all constant permutations
    constant_permutations = generate_permutations(prompt_template_dict["constants"])

    experiments = []
    for constant_permutation in constant_permutations:
        # Initialise experiment based on rendered prompt template
        experiment = AItoAIInterviewExperiment(
            model_info=prompt_template_dict["model_info"],
            temperature=prompt_template_dict["temperature"],
            profiles=prompt_template_dict["profiles"],
            roles=prompt_template_dict["roles"],
            num_subjects_per_group=prompt_template_dict["num_subjects_per_group"],
            num_groups=prompt_template_dict["num_groups"],
            session_id=prompt_template_dict["session_id"],
            hf_inference_endpoint=prompt_template_dict["hf_inference_endpoint"],
            max_num_rounds=prompt_template_dict["max_num_rounds"],
            treatments=prompt_template_dict["treatments"],
            treatment_assignment_strategy=prompt_template_dict[
                "treatment_assignment_strategy"
            ],
            treatment_column=prompt_template_dict["treatment_column"],
            group_assignment_strategy=prompt_template_dict["group_assignment_strategy"],
            group_column=prompt_template_dict["group_column"],
            role_assignment_strategy=prompt_template_dict["role_assignment_strategy"],
            role_column=prompt_template_dict["role_column"],
            random_seed=prompt_template_dict["random_seed"],
            build_profile_qna=prompt_template_dict["build_profile_qna"],
            build_profile_backstories=prompt_template_dict["build_profile_backstories"],
            prompts=prompt_template_dict["prompts"],
            constants=constant_permutation.to_dict(),
        )

        experiments.append(experiment)

    return experiments, constant_permutations
