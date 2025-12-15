# Prompt Template

## Prompt Template Workbook (`.xlsx`) – Worksheet Overview


| Worksheet Name | Description |
| - | - |
| settings | Contains the user-defined settings and configurations for the experiment. |
| treatment | Contains the treatment arms and their descriptions. |
| role | Contains the list of user-defined and special (e.g., facilitator) roles and their functions. |
| prompt | Contains information about the experiment flow and prompts. |
| profile | Contains the synthetic subjects' profile information in tabular format. |
| constant | Contains the string/numerical constants that can be dynamically injected into the `treatment`, `role`, and `prompt` worksheets using Jinja2 templates. |

*Every sheet name is **case‑sensitive** and **mandatory**. Any additional or missing worksheets will trigger a validation failure.*

---

## 1.  `settings`

| Key | **Required** | Description/Expected Value |
| - | - | - |
| `settings_label` | **Yes** | Serves as the header for the canonical keys listed below. Expected value: `value`. |
| `session_id` | **Yes** | The unique session identifier that will be assigned to the experiment. This information will also be used to name the output files (with UTC date time) after the experiment completes (e.g., `<session_id>_YYYYMMDDTHHMMSSZ.json` and `<session_id>_YYYYMMDDTHHMMSSZ.csv`). |
| `model_info` | **Yes** | The LLM that will be used in the experiment. The platform currently supports most LLMs from OpenAI (`gpt-5.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-chat-latest`, `gpt-5-codex`, `gpt-5-pro`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-2024-05-13`, `gpt-4o-mini`, `o1`, `o1-pro`, `o3-pro`, `o3`, `o4-mini`), Hugging Face Inference APIs (`hf-inference`), and [OpenRouter.ai](https://openrouter.ai/models) by default. Currently, only the LLMs from OpenAI can accept visual inputs. |
| `hf_inference_endpoint` | **Optional** | Refers to the base URL generated when deploying a Hugging Face Inference Endpoint. This field is only required when choosing `hf-inference` in `model_info`. Deploy a HF inference endpoint by navigating to the model of your choice on the Hugging Face website → Select `Deploy` and `HF Inference Endpoint` → Select your cloud provider and define your endpoint's configuration, then select `Create Endpoint` → Wait for the API endpoint to be successfully deployed and you can obtain the `hf_inference_endpoint` URL by clicking on the `API` tab under `Playground` and copying the URL in `base_url` |
| `temperature` | **Yes** | The temperature setting that will be applied to the LLM. Expected values: Any value between `0-2` (inclusive). This setting is ignored for certain thinking models (`gpt-5.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-codex`, `gpt-5-pro`, `o1`, `o1-pro`, `o3-pro`, `o3`, `o4-mini`). If this value is not provided, the platform will default to a temperature value of `0`. |
| `num_subjects_per_group` | **Yes** | The number of user-defined subjects assigned to each group. Special roles like `facilitator` are excluded from this value. If this value is not provided, the platform will default to a value of `1`. |
| `num_groups` | **Yes** | The number of groups that will be participating in a particular session. If this value is not provided, the platform will default to a value of `1`. |
| `max_num_rounds` | **Yes** | Sets the maximum expected number of rounds in each session; the session terminates prematurely if this limit is exceeded. Particularly useful in preventing infinite loops when using `repeat_private_question` or `repeat_public_question` type questions. If this value is not provided, the platform will default to a value of `10`. |
| `treatment_assignment_strategy` | **Yes** | The strategy used for assigning treatments to each group. Expected values: `simple_random`, `complete_random`, `manual`. |
| `treatment_column` | **Optional** | In the case that the treatment assignment strategy is `manual`, provide the column name from the `profile` worksheet that contains the assigned treatments. |
| `group_assignment_strategy` | **Yes** | The strategy used for assigning subjects to groups. Expected values: `random`, `manual`. If the treatment assignment strategy is set as `manual`, the group_assignment_strategy must also be set as `manual` to ensure that all subjects in the same group is assigned the same treatment. |
| `group_column` | **Optional** | In the case that the group assignment strategy is `manual`, provide the column name from the `profile` worksheet that contains the assigned groups. |
| `role_assignment_strategy` | **Yes** | The strategy used for assigning roles to subjects. Expected values: `random`, `manual`. |
| `role_column` | **Optional** | In the case that the role assignment strategy is `manual`, provide the column name from the `profile` worksheet that contains the assigned roles. |
| `random_seed` | **Optional** | The random seed for reproducibility. If this value is not provided, the platform will default to a value of `42`. |
| `build_profile_qna` | **Yes** | A boolean flag for representing the subject's profile information in Q&A format in the system message. Expected values: `True` or `False`. If `build_profile_qna` is set to `True`, the subject’s profile is formatted as a Q&A snippet, where each profile-related question is prefixed with “Interviewer:” and the subject's response with “Me:”. This snippet is inserted into the LLM-powered subject’s system message to give the LLM context about the subject's profile. |
| `build_profile_backstories` | **Yes** | A boolean flag for representing the subject's profile information as first-person backstories in the system message. Expected values: `True` or `False`. If `build_profile_backstories` is set to `True`, a first-person narrated backstory will be generated based on the subject's responses and inserted into the LLM-powered subject’s system message to give the LLM context about the subject's profile. If both `build_profile_qna` and `build_profile_backstories` is set to `False`, the LLM will not be provided any profile-related information about the subject. |

---

## 2.  `treatment`

| Column | **Required** | Description |
| - | - | - |
| `treatment_label` | **Yes (Unique)** | A short, concise label for each treatment arm. In the case that the treatment assignment strategy is `manual`, the treatment labels in this worksheet should be a superset of the treatment labels provided in the `profile` worksheet.|
| `value` | **Yes** | A full description of the treatment arm. Users can define the treatment arm as a Python dictionary with any attributes needed (e.g., description, other_treatment_attribute). Example: ```{"description":"Description of the treatment arm", "other_treatment_attribute":"Description of another attribute related to treatment arm."}```. In the `prompt` worksheet, you can reference these attributes with Jinja dot notation to control when/where the treatment is introduced in your experiment under the `llm_text` field, e.g. ```{{ treatment.description }}```. If a plain string is provided instead of a Python dictionary, it will automatically be placed into the `description` field. The `description` field is a compulsory field. Additionally, you can use Jinja dot notation to reference specific constant attributes in this field (e.g., `{{ constant.label }}`). **However, you should not use the Jinja dot notation to reference specific role attributes in this field to avoid creating an infinite referencing loop.** |

*Extra columns will be rejected. Each row refers to a unique treatment arm.*

---

## 3.  `role`

| Column | **Required** | Description |
| - | - | - |
| `role_label` | **Yes (Unique)** | A short, concise label for each role. If the role assignment strategy is `manual`, the role labels in the `role` worksheet should match with the role labels provided in the `profile` worksheet. A special role, `facilitator`, is required for every experiment and must be defined in the `role` worksheet. This role is used to orchestrate the flow of the experiment, and can perform other user-defined functions, such as performing intermediate payoff calculations during interactive experiments and evaluating terminating conditions for `repeat_private_question` or `repeat_public_question` type questions. The [`public goods experiment demo example`](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/public_good_experiment) provides a useful reference on how the `facilitator` role can be leveraged to perform intermediate payoff calculations and evaluate terminating conditions for `repeat_private_question`-type questions. |
| `value` | **Yes** | A full description of the role. Users can define the role as a Python dictionary with any attributes needed (e.g., description, other_role_attribute). Example: ```{"description":"Description of the role", "other_role_attribute":"Description of another attribute related to role."}```. The role's description is automatically included as part of the LLM-powered subject's system message. Other than that, these role attributes can also be referenced in the `prompt` worksheet under the `llm_text` field using Jinja dot notation, e.g. ```{{ role.description }}```. If a plain string is provided instead of a Python dictionary, it will automatically be placed into the `description` field. The `description` field is a compulsory field. Additionally, you can use Jinja dot notation to reference specific treatment and constant attributes in this field (e.g.,`{{ treatment.description }}`, `{{ constant.label }}`) |

*Extra columns will be rejected. Each row refers to a unique agent role.*

---

## 4.  `prompt`

| Column | **Required** | Description/Expected value |
| - | - | - |
| `round_id` | **Yes (Unique)** | A unique identifier for each round in the experiment. This identifier will be tagged to the LLM's response when generating the output JSON and CSV file if `response_name` is not defined. |
| `type` | **Yes** | The prompt type that will be conducted during a particular experiment round. Expected values: `context`, `discussion`, `public_question`, `private_question`, , `repeat_public_question`, `repeat_private_question`. `context` prompts are used to provide the LLM-powered subjects with contextual information about the experiment, and must be defined at the beginning so that it can be incorporated as part of the session's system prompt. `discussion` prompts are meant to facilitate a group discussion/conversation where a question is posed by the `facilitator` to the group at the beginning of the round and the subjects will respond sequentially, having visibility of other subjects' responses (i.e., facilitator → Participant 1 → Participant 2 → Participant 3). `public_question` and `private_question` prompts are 1-on-1 type questions that are posed separately to each subject (i.e., facilitator → Participant 1 → facilitator → Participant 2 → facilitator → Participant 3). However, `public_question` prompts are chosen when you want the subjects to have visibility over their peers' responses within the same round. On the other hand, `private_question` prompts are chosen when you want to hide the subjects' responses from other subjects. `repeat_public_question` and `repeat_private_question` are extensions of `public_question` and `private_question` where the round will repeat until a terminating condition is met. For such questions, ensure that you define a terminating condition and instruct the `facilitator` role to return the keyword `end_round` to move on to the next round when the condition is met. Otherwise, the session will be stuck in an infinite loop and only terminated when it hits the `max_num_rounds` condition. The [`public goods experiment demo example`](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/public_good_experiment) provides a useful reference on how the `facilitator` role can be leveraged to evaluate terminating conditions for such questions. |
| `round_order` | **Yes** | An integer value indicating the order in which the rounds defined will be executed. If the round order value is duplicated across multiple rounds, then the order of these rounds will be randomized. |
| `is_adapted` | **Yes** | A boolean field indicating if the text from the actual experiment has been adapted. Expected values: `True` or `False`. This field is only used for documentation purposes and does not affect the operation of the platform. |
| `human_text` | **Optional** | The original instructions used in the actual experiment before adaptation. This field is only used for documentation purposes and does not affect the operation of the platform. |
| `llm_text` | **Yes** | The prompt presented to the LLM-powered subjects during each experiment round. This could be adapted from the original instructions used in the actual experiment to improve the LLM's performance. The prompt can be defined as a plain string; in that case the same prompt will be automatically presented to each user-defined role listed in the `role` worksheet. Alternatively, you can define a Python dictionary, where the keys are the role labels (matching those in the `role` worksheet) and values are the prompt that will be presented to that role. When presenting your prompt as a Python dictionary, you can also customise the role order (based on the order in the dictionary) and also the roles that will participate in this experiment round (i.e., you can exclude certain roles from participating in specific rounds). Additionally, you can reference different role attributes, treatment attributes, or constants in your prompts by using Jinja dot notation (e.g., `{{ treatment.description }}`, `{{ role.description }}`, `{{ constant.label }}`). Lastly, you can pass visual inputs to the LLM-powered subjects by including the URL of a public image in the prompt. **Note: Links to images uploaded to Google Drive are currently not supported by the platform as they cannot be accessed by OpenAI's visual models.** |
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
* Depending on whether `build_profile_qna` and `build_profile_backstories` in the `settings` worksheet is set to `True` or `False`, the subject's responses will be formatted accordingly and passed into the system message to provide the LLM context about the subject’s profile.

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
A video walkthrough on how to populate the prompt template workbook based on a simple public goods experiment can be found here: [Video Walkthrough](https://www.loom.com/share/2a9c02bfb9094afcbe7767d168179dfd)

---

## FAQ

This section contains questions and answers we’ve compiled from past workshops, and we’ll continue to expand this section as more questions arise:

---

**Question**: How can I use Jinja dot notation across the `treatment`, `role`, `prompt`, and `constant` worksheets?

**Response**: You can safely use Jinja dot notation to:
* Reference treatment attributes inside the `role` worksheet (e.g. `{{ treatment.description }}`),
* Reference constants inside both the `treatment` and `role` worksheets (e.g. `{{ constant.label }}`), and
* Reference role, treatment, and constant attributes in the `llm_text` field of the `prompt` worksheet (e.g. `{{ role.description }}`, `{{ treatment.description }}`, `{{ constant.label }}`). 

However, you should not use Jinja dot notation in the `treatment` worksheet to reference role attributes, as this can potentially create infinite reference loops (role → treatment → role → …). You also shouldn't use Jinja dot notation inside the `response_options` field of the `prompt` worksheet. If you need to provide instructions/prompts with dynamic bounds, leave the `response_options` field empty and express the constraint directly in `llm_text`. For example:

Respond with a numerical value between `{{ treatment.start_value }}` and `{{ treatment.end_value }}`.

---

**Question**: What is treated as private information vs public information for the LLM-powered subjects during an experiment?

**Response**: Each LLM-powered subject is provided with certain “private” and “public” information:
* Private to each subject:
  * Role descriptions and role-related attributes from the `role` worksheet
  * Profile-related information from the `profile` worksheet
  * `private_question` and `repeat_private_question` prompts
  * `context` prompts (when addressed to specific subjects)
* Public to all subjects in a group:
  * `public_question` and `repeat_public_question` prompts
  * `discussion` prompts

We rely on these prompt types to control visibility within a group. 

---

**Question**: Can I define subgroups so that only a subset of subjects receives certain information or participates in certain rounds of the experiment?

**Response**: Yes. You can define subgroups by providing a Python dictionary in the `llm_text` field where the keys are the subjects' roles and the values are the prompts that will be presented to those subjects. For example:
```json
{
  "Participant 1": "Question or context only for Participant 1",
  "Participant 2": "Question or context only for Participant 2",
  "Participant 4": "Question or context only for Participant 4"
}
```

Only those subjects will participate in that round.
* Set `type = "context"` if you are only providing contextual information to those subjects and not expecting a response.
* Set `type = "private_question"` or `type = "repeat_private_question"` if only that subset of subjects need to participate in this round.

This effectively creates “subgroups” within a larger group in a single round. However, the platform does not yet support randomising which subjects participate in each experiment round. This functionality is planned and will be introduced in future iterations of the platform.

---

**Question**: I’m not very familiar with LLMs. How should I think about key parameters like model choice, temperature, and speculation score?

**Response**: At a high level:
* `model_info` affects model's capability and cost. Larger or more advanced models (e.g. `gpt-5.1`, `gpt-5-pro`) tend to be more reliable and robust but are slower and more expensive. Smaller or “mini/nano” models are cheaper and faster but may make more logical errors. The platform supports a range of OpenAI models, Hugging Face Inference (`hf-inference`), and OpenRouter models via `model_info` to fit your experiment needs.

* `temperature` controls the randomness and creativity of the LLM's output:
  * Low temperature (0–0.3): Responses that are deterministic, more stable, less variation.
  * Medium (0.4–0.8): Responses with balanced variety vs stability.
  * High (0.9–2): Responses that are diverse, exploratory, more variance and sometimes noisy. For many synthetic experiments, a moderate–high temperature (e.g. ~1.0) is helpful if you want a richer distribution of behaviors.

* `generate_speculation_score` tells the LLM to self-assess how speculative its answer is using a value between 0 and 100. A score of 0 means that the LLM is very certain of its response while a score of 100 means the response is purely speculative. This is useful to flag potentially hallucinated content and can be used later in analysis to filter or weight responses. 

We recommend running small pilot sessions, where you can try varying temperature and model choice on a subset of prompts and inspect how response variability, stability, and realism change.

---

**Question**: How can I implement complex rematching schemes, where participants are flexibly rematched multiple times and information from earlier stages must be carried forward?

**Response**: The current version of the Python package has limited support for fully flexible multi-round matching and direct use of `profile.*` in prompts. For such designs, we recommend treating the current implementation as a simplified single-round approximation and preparing your materials so they can be upgraded later:

1. Import a grouping indicator from the original human data for one iteration only (e.g. group IDs linking the two stakeholders and the observer). Store this in the `profile` worksheet and reference it in `settings.group_column` when using manual group assignment. 
2. Implement only a single iteration of the later-stage interaction.
3. Avoid heavy “offline” matching logic for the synthetic replication; treat that as an edge case until the package supports richer matching.

This makes your template forward-compatible with a future version of the package that will handle multi-round, constraint-based rematching more flexibly. In the meantime, this functionality is planned and will be introduced in future iterations of the platform.

---

**Question**: How do I handle experiments that require groups of different sizes (e.g. 3 and 6 participants) whose decisions interact, like in “Coordination in the Presence of Asset Markets”?

**Response**: The current platform expects fixed group sizes per session via `num_subjects_per_group` and `num_groups` in the `settings` sheet. It does not yet support a single session where some groups have 3 members and others have 6, all sharing one template. 

Recommended workarounds:
1. You can define subgroups to participate in certain rounds by providing a Python dictionary in the `llm_text` field where the keys are the subjects' roles and the values are the prompts that will be presented to those subjects. For example, for the round that only involves 3 subjects (`Participant 1`, `Participant 2`, `Participant 4`), you can define the following dictionary in the `llm_text` field:
```json
{
  "Participant 1": "Question or context only for Participant 1",
  "Participant 2": "Question or context only for Participant 2",
  "Participant 4": "Question or context only for Participant 4"
}
```

2. Use two separate templates: one for groups of 3, another for groups of 6. Once the package supports more flexible grouping and timing, these separate templates can be merged into a more faithful single design.

---

**Question**: Will additional experiment templates be made available (e.g. on GitHub), beyond the current demos?

**Response**: Yes. The goal is to gradually build an archive of synthetic replication packages and example templates to support different experimental designs. At present, the repository includes a Public Goods Experiment, an RCT demo, and a blank prompt template workbook as starting points. More templates from other studies will be added over time as they are cleaned, documented, and made suitable for reuse. 

---

**Question**: What is the difference between the `prompt` and `profile` worksheets in the prompt template workbook?

**Response**:
* The `profile` worksheet stores static subject information: each row is an unique subject, each column is a profile-related question and the subjects' responses, and there must be an `ID` column. These values can be turned into Q&A snippets or backstories and fed into the LLM-powered subject's system message, depending on `build_profile_qna` and `build_profile_backstories` in `settings`. 
* The `prompt` worksheet defines the experiment flow: rounds (`round_id`, `round_order`), prompt `type`, the text shown to LLM-powered subjects (`llm_text`), and how responses are collected and validated (`response_type`, `response_options`, etc.).

In short: `profile` = *who the subject is*; `prompt` = *what happens each round and what the subjects are asked to do*.

---

**Question**: If the original experiment asked participants about age or gender, should these questions appear in the `prompt` worksheet even if the answers already exist in `profile`?

**Response**: Yes. To faithfully replicate the original human protocol, any questions that participants actually saw (such as age or gender) should appear as prompts in the `prompt` worksheet, even if the same information is also present in the `profile` worksheet. This preserves both the data and the interaction flow, which can matter for behavioral outcomes.

---

**Question**: Can I adapt original questions to clearer multiple-choice formats and use `response_options`, even if the original data are coded differently?

**Response**: Yes. It is often helpful to adapt human-facing questions to a clearer multiple-choice format for LLMs. For example, you can:
* Keep the original wording in `human_text`,
* Provide a labelled MCQ version in `llm_text`, e.g. A/B/C/D options,
* Define `response_options = ["A", "B", "C", "D"]`.

During preprocessing or analysis, you can then map each option back to the format presented in the original study. This is a reasonable and recommended adaptation, as long as it’s documented via `is_adapted = True`. 

---

**Question**: How exactly does the `response_options` field work, and how does it interact with `validate_response` and `llm_text`?

**Response**: The `response_options` field defines the allowed response space for each round. It can be:
* A plain string, e.g. `"Enter a number between 0 and 5"`;
* A Python list, e.g. `[0, 1, 2, 3, 4, 5]` or `["A", "B", "C", "D"]`;
* A Python tuple representing a numeric range, e.g. `(0, 5)`;
* A Python dictionary mapping role labels to any of the above so different roles can have different action spaces. 

It is used in two ways:
1. When `validate_response = True`, the platform checks whether the LLM’s answer matches with any of the permissible values. If not, it retries up to 5 times before moving on.
2. By including `{{ response_options }}` in the `llm_text` field, the platform injects standard formatting instructions based on the response options (e.g. “Answer with one of: A, B, C, D”), making it more likely the LLM chooses a valid option.

Note that you cannot use Jinja dot notation inside the `response_options` field.

---

**Question**: The original experiment uses pictures as stimuli. How can I include images (e.g. with `gpt-5-nano`) in my prompts?

**Response**: Only OpenAI models support visual inputs in the current platform. If your `model_info` uses one of the supported OpenAI models, you can:
* Host the image at a publicly accessible URL (such as in Dropbox), and
* Reference that URL in the `llm_text` field as part of the instructions (e.g. “Look at this image: <URL> and then answer…”). 

If you cannot use images (e.g. because you’re on `hf-inference` or OpenRouter), the recommended approach is to replace each image with a detailed textual description that conveys the same information. Also, note that images hosted on Google Drive cannot be accessed by OpenAI’s models, even if the link is public. Instead, you can consider hosting the image in a publicly accessible Dropbox folder.

---

**Question**: Our paper has two separate experiments with different variables and populations. Should we model them in a single template or as separate templates?

**Response**: In general, it is cleaner to treat experiments with different designs or populations as separate templates (i.e. separate prompt workbooks or sessions), even if they are reported in one paper. This makes:
* Treatment and role definitions clearer
* Group assignment strategies easier to reason about
* Replication and debugging simpler

You can still keep both templates in the same repository or project, but the platform will handle them as distinct experiments.

---

**Question**: Does the platform support alternative backends like OpenRouter and local LLMs (e.g. Ollama)?

**Response**: Yes, to an extent. The platform already supports:
* OpenAI models (via `model_info` such as `gpt-5.1`, `gpt-4.1`, `gpt-4o`, etc.)
* Hugging Face Inference when `model_info = "hf-inference"` and `hf_inference_endpoint` is provided
* OpenRouter.ai models when `model_info = <provider/model_name, i.e., mistralai/devstral-2512>`. 

We are also considering providing support for local LLMs (e.g. via an Ollama-like client); however, it is likely to be deprioritized due to the low demand.

---

**Question**: Our replication shows very little variation in the responses generated by the LLM-powered subjects during the experiment. Is this a problem?

**Response**: Modest variation or clustering in LLM outputs is not unusual and is unlikely a bug due to the platform. First check whether:
* The prompt itself is anchoring on particular values (e.g. giving 20 and 40 as examples), or
* The temperature is set too low (e.g. 0–0.3), which makes the model more deterministic.

If prompts are neutral and temperature is reasonably high (e.g. around 1.0) yet variation is still limited, treat this as part of the broader issue of distribution alignment between human and LLM responses. One of the goals of building a library of synthetic replications is precisely to explore and improve this alignment across many studies.

---

**Question**: What should I do if the model repeatedly produces incorrect or malformed responses?

**Response**: Use a two-step debugging strategy:
1. Enable reasoning output by setting `format_response = True`. This instructs the platform to provide a structured response containing a “reasoning” field so you can inspect the model's thinking process. 
2. Based on that reasoning:
   * If the model misunderstood the instructions or the structure of the task, revise your prompts (especially in the `llm_text` field) to be clearer, ensure reference information are correctly rendered by the facilitator, and remove ambiguity.
   * If the instructions are clear and the model *still* makes logical errors, consider switching to a more capable thinking model (e.g. `gpt-5.1`), but treat this as a second step because larger models tend to be more expensive.

Most issues are resolved by clarifying prompts and validation logic before needing a larger model.

---

**Question**: Is there an `end_experiment` command, and how can I stop a repeated round when some terminating condition is met (e.g., participants choose to stop)?

**Response**: There is no dedicated `end_experiment` command. For rounds using `repeat_public_question` or `repeat_private_question`, these rounds will repeat until the facilitator role returns a response with the special command: `end_round`. 

To implement participant-driven stopping:
1. Allow participants to indicate their desire to stop.
2. Have the facilitator evaluate this condition and, when it is met, return `end_round` so the platform proceeds to the next round or terminates when no more rounds are defined.
3. Use `max_num_rounds` in `settings` as a safety cap to prevent infinite loops.

---

**Question**: When should I set `is_adapted = True` in the `prompt` worksheet?

**Response**: Set `is_adapted = True` whenever the text shown to the LLM (`llm_text`) differs from the original human instructions, even if the change is minor (e.g. small rephrasings, graph caption changes, labelled options). This field is only for documentation and does not affect how the platform runs the experiment. It simply records that the LLM saw an adapted version rather than the exact original wording seen by the human subject. 

---

**Question**: How can I represent experimental designs where groups are reshuffled every period with constraints in the `profile` sheet?

**Response**: The platform currently has only a partial workaround for fully dynamic group reshuffling. If your human data include a group ID per round, we recommend:
1. Using the group composition from the first round only as the grouping structure for the synthetic replication.
2. Adding a dedicated column in the `profile` sheet (e.g. `group_round1`) to store this first-round group ID for each subject.
3. Setting `group_assignment_strategy = "manual"` and pointing `group_column` in the `settings` worksheet to that newly created column. 

Document this as an adaptation so that once the package supports more flexible, round-by-round grouping, your design can be updated accordingly. In the meantime, this functionality is planned and will be introduced in future iterations of the platform.

---

**Question**: For a purely individual decision-making experiment, is there a better setup than putting all participants into a single group?

**Response**: Yes. For individual decision-making tasks where subjects do not interact, the recommended strategy is to assign one subject per group (e.g. `num_subjects_per_group = 1` and `num_groups` equal to the number of subjects). Putting all subjects into a single group is not advisable, especially in test mode, because the platform would treat them as interacting participants within that group.

---

**Question**: How should we treat answers from a Big Five questionnaire that served both as a distraction and as a personality measure in the original experiment?

**Response**: The recommended approach is to use the Big Five responses as profile information in the `profile` worksheet, and document this adaptation. This preserves the original participants’ personality traits as part of the synthetic persona, which may be behaviourally relevant. You can still note the “distraction” role of the questionnaire in your design documentation.

---

**Question**: The dataset includes both birth year and age at the time of the experiment. Which should we use for profiling?

**Response**: You should use the age at the time of the original experiment (in years), not the birth year plus today’s date. The goal is to replicate the participant’s persona at the time of data collection, which is better captured by the original age variable than recalculating age relative to the present.

---

**Question**: The original design of my experiment requires the AI to choose three items simultaneously, but there is no `list` response type. How can I implement this?

**Response**: You can simulate a multi-item choice as structured free text:
1. Set `response_type = "free-text"`.
2. Leave `response_options` empty.
3. In `llm_text`, give explicit formatting instructions, e.g.:

“In your response, return a JSON list containing your three chosen items, like: `["item1", "item2", "item3"]`.”

This way, the model outputs a single response that encodes all three choices, and you can parse the returned list during preprocessing or analysis.
