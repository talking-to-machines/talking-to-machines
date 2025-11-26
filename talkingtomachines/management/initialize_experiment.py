import itertools
from talkingtomachines.management.experiment import (
    AItoAIInterviewExperiment,
    Constant,
    # Treatment,
    # Role,
)

# from jinja2 import Template


# def render_dict_with_template(
#     prompt_template_dict: dict, constant_permutation: Constant
# ) -> dict:
#     rendered_dict = {}
#     for key, value in prompt_template_dict.items():
#         if key == "llm_text":
#             # Skip rendering for 'llm_text' key as they will be rendered later during the session
#             rendered_dict[key] = value
#             continue

#         if isinstance(value, dict):
#             # Recursively process nested dictionaries
#             rendered_dict[key] = render_dict_with_template(value, constant_permutation)

#         elif isinstance(value, str):
#             # Render template for each string value
#             template = Template(value)
#             rendered_value = template.render(constant=constant_permutation.to_dict())
#             rendered_dict[key] = rendered_value

#         elif isinstance(value, Treatment):
#             rendered_value = Treatment(
#                 **render_dict_with_template(value.to_dict(), constant_permutation)
#             )
#             rendered_dict[key] = rendered_value

#         elif isinstance(value, Role):
#             rendered_value = Role(
#                 **render_dict_with_template(value.to_dict(), constant_permutation)
#             )
#             rendered_dict[key] = rendered_value

#         elif isinstance(value, list):
#             # Render template for each item in the list
#             rendered_list = []
#             for item in value:
#                 if isinstance(item, dict):
#                     # Recursively process nested dictionaries
#                     rendered_list.append(
#                         render_dict_with_template(item, constant_permutation)
#                     )

#                 elif isinstance(item, str):
#                     # Render template for each string value
#                     template = Template(item)
#                     rendered_item = template.render(
#                         constant=constant_permutation.to_dict()
#                     )
#                     rendered_list.append(rendered_item)

#                 elif isinstance(item, Treatment):
#                     rendered_item = Treatment(
#                         **render_dict_with_template(
#                             item.to_dict(), constant_permutation
#                         )
#                     )
#                     rendered_list.append(rendered_item)

#                 elif isinstance(item, Role):
#                     rendered_item = Role(
#                         **render_dict_with_template(
#                             item.to_dict(), constant_permutation
#                         )
#                     )
#                     rendered_list.append(rendered_item)

#                 else:
#                     rendered_list.append(item)

#             rendered_dict[key] = rendered_list

#         else:
#             rendered_dict[key] = value

#     return rendered_dict


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


def initialize_experiment(prompt_template_dict: dict) -> list:
    """Initializes a list of AI-to-AI interview experiments based on the provided prompt template data.

    Args:
        prompt_template_dict (dict): A dictionary containing the prompt template data.

    Returns:
        list: A list of initialized AItoAIInterviewExperiment objects.
    """
    # Define all constant permutations
    constant_permutations = generate_permutations(prompt_template_dict["constants"])

    experiments = []
    for constant_permutation in constant_permutations:
        # # For each permutation, apply constants to prompt template using Jjanja
        # rendered_prompt_template_dict = render_dict_with_template(
        #     prompt_template_dict=prompt_template_dict,
        #     constant_permutation=constant_permutation,
        # )

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
