# Prompt Template

## Prompt Template Workbook (`.xlsx`) – Worksheet Overview


| Worksheet Name | Description |
| - | - |
| settings | Contains the user-defined settings and configurations for the experiment. |
| treatment | Contains the treatment arms and their descriptions. |
| role | Contains the list of user-defined and special (e.g., facilitator) roles and their functions. |
| prompt | Contains information about the experiment flow and prompts. |
| profile | Contains the synthetic subjects' profile in tabular format. |
| constant | Contains the string/numerical constants that can be dynamically injected into the treatment, agent_roles, interview_prompts worksheets using Jinja2 templates. |

*Every sheet name is **case‑sensitive** and **mandatory**. Any additional or missing worksheets will trigger a validation failure.*

---

## 1.  `settings`

| Key | **Required** | Description/Expected Value |
| - | - | - |
| `settings_label` | **Yes** | Serves as the header for the canonical keys listed below. Expected value: `value`. |
| `session_id` | **Yes** | The unique session identifier that will be assigned to the experiment. This information will also be used to name the output files after the experiment completes (e.g., `<session_id>.json` and `<session_id>.csv`). |
| `model_info` | **Yes** | The LLM that will be used in the experiment. The platform currently supports most LLMs from OpenAI (`gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-chat-latest`, `gpt-5-codex`, `gpt-5-pro`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-2024-05-13`, `gpt-4o-mini`, `o1`, `o1-pro`, `o3-pro`, `o3`, `o4-mini`), Hugging Face Inference APIs (`hf-inference`), and [OpenRouter.ai](https://openrouter.ai/models). Currently, only the LLMs from OpenAI can accept visual inputs. |
| `hf_inference_endpoint` | **Optional** | Refers to the base URL generated when deploying a Hugging Face Inference Endpoint. This field is only required when choosing `hf-inference` in `model_info`. Deploy a HF inference endpoint by navigating to the model of your choice on the Hugging Face website → Select `Deploy` and `HF Inference Endpoint` → Select your cloud provider and define your endpoint's configuration, then select `Create Endpoint` → Wait for the API endpoint to be successfully deployed and you can obtain the `hf_inference_endpoint` URL by clicking on the `API` tab under `Playground` and copying the URL in `base_url` |
| `temperature` | **Yes** | The temperature setting that will be applied to the LLM. Expected values: Any value between `0-2` (inclusive). This setting is ignored for certain thinking models (`gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-codex`, `gpt-5-pro`, `o1`, `o1-pro`, `o3-pro`, `o3`, `o4-mini`). |
| `num_subjects_per_group` | **Yes** | The number of user-defined subjects assigned to each group. Special roles like `facilitator` are excluded from this value. |
| `num_groups` | **Yes** | The number of groups that will be participating in a particular session. |
| `max_num_rounds` | **Yes** | Sets the maximum expected number of rounds in each session; the session terminates prematurely if this limit is exceeded. Particularly useful in preventing infinite loops when using `repeat_private_question` or `repeat_public_question` type questions. |
| `treatment_assignment_strategy` | **Yes** | The strategy used for assigning treatments to each group. Expected values: `simple_random`, `complete_random`, `manual`. |
| `treatment_column` | **Optional** | In the case that the treatment assignment strategy is `manual`, provide the column name from the `profile` worksheet that contains the assigned treatments. |
| `group_assignment_strategy` | **Yes** | The strategy used for assigning subjects to groups. Expected values: `random`, `manual`. If the treatment assignment strategy is set as `manual`, the group_assignment_strategy must also be set as `manual` to ensure that all subjects in the same group is assigned the same treatment. |
| `group_column` | **Optional** | In the case that the group assignment strategy is `manual`, provide the column name from the `profile` worksheet that contains the assigned groups. |
| `role_assignment_strategy` | **Yes** | The strategy used for assigning roles to subjects. Expected values: `random`, `manual`. |
| `role_column` | **Optional** | In the case that the role assignment strategy is `manual`, provide the column name from the `profile` worksheet that contains the assigned roles. |
| `random_seed` | **Optional** | The random seed for reproducibility. Defaults to `42` if not provided. |
| `build_profile_qna` | **Yes** | A boolean flag for representing the subject's profile information in Q&A format in the system message. Expected values: `True` or `False`. |
| `build_profile_backstories` | **Yes** | A boolean flag for representing the subject's profile information as first-person backstories in the system message. Expected values: `True` or `False`. |

---

## 2.  `treatment`

| Column | **Required** | Description |
| - | - | - |
| `treatment_label` | **Yes (Unique)** | A short, concise label for each treatment arm. In the case that the treatment assignment strategy is `manual`, the treatment labels in this worksheet should be a superset of the treatment labels provided in the `profile` worksheet.|
| `value` | **Yes** | A full description of the treatment arm. Users can define the treatment arm as a Python dictionary with any attributes needed (e.g., description, other_treatment_attribute). Example: ```{"description":"Description of the treatment arm", "other_treatment_attribute":"Description of another attribute related to treatment arm."}```. In the `prompt` worksheet, you can reference these attributes with Jinja dot notation to control when/where the treatment is introduced in your experiment, e.g. ```{{ treatment.description }}```. If a plain string is provided instead of a Python dictionary, it will automatically be placed into the `description` field. The `description` field is a compulsary field. |

*Extra columns will be rejected. Each row refers to a unique treatment arm.*

---

## 3.  `role`

| Column | **Required** | Description |
| - | - | - |
| `role_label` | **Yes (Unique)** | A short, concise label for each role. If the role assignment strategy is `manual`, the role labels in the `role` worksheet should be a superset of the role labels provided in the `profile` worksheet. A special role, `facilitator`, is required for every experiment and must be defined in the `role` worksheet. This role is used to orchestrate the flow of the experiment, and can perform other user-defined functions, such as performing intermediate payoff calculations during interactive experiments and evaluating terminating conditions for `repeat_private_question` or `repeat_public_question` type questions. The [`public goods experiment demo example`](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/public_good_experiment) provides a useful reference on how the `facilitator` role can be leveraged to perform intermediate payoff calculations and evaluate terminating conditions for `repeat_private_question`-type questions. |
| `value` | **Yes** | A full description of the role. Users can define the role as a Python dictionary with any attributes needed (e.g., description, other_role_attribute). Example: ```{"description":"Description of the role", "other_role_attribute":"Description of another attribute related to role."}```. The role's description is automatically included as part of the LLM-powered subject's system message. Other than that, these role attributes can also be referenced in the `prompt` worksheet using Jinja dot notation, e.g. ```{{ role.description }}```. If a plain string is provided instead of a Python dictionary, it will automatically be placed into the `description` field. The `description` field is a compulsary field. |

*Extra columns will be rejected. Each row refers to a unique agent role.*

---

## 4.  `prompt`

| Column | **Required** | Description/Expected value |
| - | - | - |
| `round_id` | **Yes (Unique)** | A unique identifier for each round in the experiment. This identifier will be tagged to the LLM's response when generating the output JSON and CSV file if `response_name` is not defined. |
| `type` | **Yes** | The task type that will be conducted during a particular experiment round. Expected values: `context`, `discussion`, `public_question`, `private_question`, , `repeat_public_question`, `repeat_private_question`. `context` tasks are used to provide the LLM-powered subjects with contextual information about the experiment, and must be defined at the beginning so that it can be incorporated as part of the session's system prompt. `discussion` tasks are meant to facilitate a group discussion/conversation where a question is posed by the `facilitator` to the group at the beginning of the round and the subjects will respond sequentially, having visibility of other subjects' responses (i.e., facilitator → Participant 1 → Participant 2 → Participant 3). `public_question` and `private_question` tasks are 1-on-1 type questions that are posed separately to each subject (i.e., facilitator → Participant 1 → facilitator → Participant 2 → facilitator → Participant 3). However, `public_question` tasks are chosen when you want the subjects to have visibility over their peers' responses within the same round. On the other hand, `private_question` tasks are chosen when you want to hide the subjects' responses from other subjects. `repeat_public_question` and `repeat_private_question` are extensions of `public_question` and `private_question` where the round will repeat until a terminating condition is met. For such questions, ensure that you define a terminating condition and instruct the `facilitator` role to return the keyword `end_round` to move on to the next round when the condition is met. Otherwise, the session will be stuck in an infinite loop and only terminated when it hits the `max_num_rounds` condition. The [`public goods experiment demo example`](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/public_good_experiment) provides a useful reference on how the `facilitator` role can be leveraged to evaluate terminating conditions for such questions. |
| `round_order` | **Yes** | An integer value indicating the order in which the rounds defined will be executed. If the round order value is duplicated across multiple rounds, then the order of these rounds will be randomized. |
| `is_adapted` | **Yes** | A boolean field indicating if the text from the actual experiment has been adapted. Expected values: `True` or `False`. This field is only used for documentation purposes and does not affect the operation of the platform. |
| `human_text` | **Optional** | The original instructions used in the actual experiment before adaptation. This field is only used for documentation purposes and does not affect the operation of the platform. |
| `llm_text` | **Yes** | The prompt presented to the LLM-powered subjects during each experiment round. This could be adapted from the original instructions used in the actual experiment to improve the LLM's performance. The prompt can be defined as a plain string; in that case the same prompt will be automatically presented to each user-defined role listed in the `role` worksheet. Alternatively, you can define a Python dictionary, where the keys are the role labels (matching those in the `role` worksheet) and values are the prompt that will be presented to the role. When presenting your prompt as a Python dictionary, you can also customise the role order (based on the order in the dictionary) and also the roles that will participate in this experiment round (i.e., you can exclude certain roles from participating in specific rounds). Additionally, you can reference different role, treatment, or constant attributes in your prompts by using Jinja dot notation (e.g., ```{{ treatment.description }}```, ```{{ role.description }}```, ```{{ constant.label }}```). Lastly, you can pass visual inputs to the LLM-powered subjects by including the URL of a public image in the prompt. Note: Links to images uploaded to Google Drive are currently not supported by the platform as they cannot be accessed by OpenAI's visual models. |
| `response_name` | **Yes (Unique)** | The response name that will be tagged to the LLM's response when generating the output JSON and CSV files. All response names should be unique. |
| `response_type` | **Yes** | The expected response type. Expected values: `context`, `category`, `integer`, `free-text`. |
| `response_options` | **Optional** | The response options that will be used to validate the LLM's generated response during each experiment round. The response options can be defined either as a plain string `Enter a number between 0 and 5`, a Python list `[0,1,2,3,4,5]`, or a Python tuple `(0,5)`. In that case, the same response options will be automatically assigned to every user-defined role listed in the `role` worksheet. Alternatively, you can define a Python dictionary, where the keys are the role labels (matching those in the `role` worksheet) and values are the response options for that specific role. Similarly, the response options can be a plain string, a Python list, or a Python tuple. When presenting your response options as a Python dictionary, you can also customise different action spaces for each user-defined role in that experiment round. |
| `randomize_response_order` | **Yes** | A boolean field indicating if the order of the response options should be randomized before presenting it to the LLM. Expected values: `True` or `False`. |
| `validate_response` | **Yes** | A boolean field indicating if the LLM responses should be validated against the values in the `response_options` field. If the LLM response does not match with any of the options in the `response_options` field, the LLM will be queried again for a maximum of 5 times before proceeding with the last response. Expected values: `True` or `False`. |
| `generate_speculation_score` | **Yes** | A boolean field indicating if the LLM should generate a speculation score (where 0 = not speculative at all and 100 = entirely speculative.). This is used to guard against LLM hallucination. Expected values: `True` or `False`. |
| `format_response` | **Yes** | A boolean field indicating if the LLM response should be formatted as a JSON string or plain text string. Expected values: `True` or `False`. |

*Extra columns will be rejected. Each row refers to a new round in the experiment.*

---

## 5.  `profile`

* **Row 1:** Shorten name for the profile-related question. *Must be non‑blank & unique.*
* **Row 2:** The actual wording used when asking the profile-related question. *Must be non‑blank and human-readable.*
* **Row 3 … n:** The subjects' profile data, where each row represent the profile of a unique subject and each column refers to the response provided by the subject for each profile-related question.
* There must be a column named 'ID' representing a unique identifier for each subject that will be participanting in the experiment. This must be satisfied even if you do not intend to provide any profile information for your subjects.
* If `build_profile_qna` is set to `True`, the subject’s profile is formatted as a Q&A snippet, where each profile-related question is prefixed with “Interviewer:” and the subject's response with “Me:”. This snippet is inserted into the LLM-powered subject’s system message to give the LLM context about the subject's profile.
* If `build_profile_backstories` is set to `True`, a first-person narrated backstory will be generated based on the subject's responses and inserted into the LLM-powered subject’s system message to give the LLM context about the subject's profile.
* If both `build_profile_qna` and `build_profile_backstories` is set to `False`, the LLM will not be provided any profile-related information about the subject.

---

## 6.  `constant`

| Column | **Required** | Description |
| - | - | - |
| `constant_label` | **Yes (Unique)** | The template that will be used by Jinja to identify and replace the constant placeholders in the `treatment`, `role`, `prompt` worksheets (e.g., ```{{ constant.<insert constant label> }}```). |
| `value` | **Yes** | Expects a list containing different permutations that should be applied to the constant placeholders. |

*Extra columns will be rejected. Each row refers to a new constant permutation. If more than one row is defined, the package will perform a cartesian product over all rows to create a list of all possible permutations. Each permutation will spin off a separate session.*

---

## Demo Examples

You may also explore these example experimental designs and their accompanying prompt templates prepared by the development team:

* **Public Goods Experiment**: A public goods experiment demo example with a populated prompt template workbook and description of its experimental design: [Public Goods Experiment Demo](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/public_good_experiment)

* **Randomized Controlled Trial (RCT)**: A RCT experiment demo example with a populated prompt template workbook: [RCT Demo](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/rct_experiment)

* **Prompt Template**: A unpopulated version of the prompt template has been provided to serve as a starting point for creating new synthetic experiments: [Prompt Template](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/prompt_template.xlsx)

---

## 📹 Video Walkthrough
A video walkthrough on how to populate the prompt template workbook based on a simple public goods experiment can be found here: [Video Walkthrough](https://www.loom.com/share/ba5c913979344fd384fd769c64c01cf4?sid=7e2a8981-4826-4230-858d-e1fd63894157)