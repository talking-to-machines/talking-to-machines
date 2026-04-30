# Prompt Template

This document describes how to populate the prompt template workbook (`.xlsx`) for each version of the `talkingtomachines` platform.

- [v0.3.1 Prompt Template](#v031-prompt-template)
- [Demo Examples](#demo-examples)
- [Video Walkthrough](#-video-walkthrough)

---

# v0.3.1 Prompt Template

## Worksheet Overview

v0.3.1 uses **7 worksheets** that map to the oTree-inspired hierarchy (Session → Module → Subsession → Group → Agent/Player). Use `talkingtomachines init` to generate a blank template with example rows.

| Worksheet Name | Description |
| - | - |
| Settings | Global experiment settings and configurations. |
| C | Module-specific and global constants injected into prompts via Jinja2. |
| Fields | Data variables (fields) that agents write to during the experiment. |
| Facilitator | Built-in and custom facilitator functions that control experiment flow. |
| Prompts | The prompt sequence defining what happens in each module and round. |
| Profiles | Agent profile attributes in tabular format (demographics, characteristics). |
| Manual_ | Optional manual overrides for field values and group assignments. |

*Every sheet name is **case-sensitive**. The `Manual_` worksheet is optional; all others are mandatory. Use `talkingtomachines validate` to check your template before running.*

---

## 1. `Settings`

Two-column format: the first column contains the setting key (`name`), the second column contains the setting value (`value`).

| Key | **Required** | Default | Description/Expected Value |
| - | - | - | - |
| `EXPERIMENT_ID` | **Yes** | (none) | Unique experiment identifier. Used in run ID generation. |
| `MODEL_NAME` | **Yes** | (none) | The LLM model identifier to use for agent inference. The platform auto-detects the provider from the model name. Supported providers: OpenAI (`gpt-*`, `o1`, `o3`, `o4`, `o5`), Anthropic (`claude-*`), Google (`gemini-*`), Mistral (`mistral-*`, `codestral-*`), xAI (`grok-*`), DeepSeek (`deepseek-*`), Hugging Face (`hf-*`), and [OpenRouter.ai](https://openrouter.ai/models) (`openrouter/*`). Unrecognised model names default to OpenRouter. The model is validated against available API keys during `talkingtomachines validate`. |
| `HF_INFERENCE_ENDPOINT` | No | `""` | The base URL of a deployed Hugging Face Inference Endpoint. Only required when `MODEL_NAME` starts with `hf-`. |
| `TEMPERATURE` | No | `0.0` | Sampling temperature controlling response randomness. Expected values: `0.0` to `2.0`. Higher values produce more diverse responses. A value of `0.0` produces near-deterministic output. This setting is automatically omitted for models that do not support it (e.g., `o1`, `o3`, `gpt-5-mini`); the platform detects unsupported-temperature errors and retries without the parameter. |
| `RANDOM_SEED` | **Yes** | (none) | A non-negative integer seed for reproducible randomisation. Used by the `RandomisationEngine` for group assignment, option shuffling, and run ID generation. Two runs with the same seed and template produce identical assignments. |
| `NUM_AGENTS_PER_SESSION` | **Yes** | (none) | The number of agents (synthetic subjects) participating in each session. Must be a positive integer. This determines how many profile rows from the `Profiles` worksheet are used. |
| `MODULE_SEQUENCE` | **Yes** | (none) | A comma-separated, ordered list of module names to execute in the session (e.g., `module1,module2,module3`). Each module name must correspond to entries in the `C`, `Fields`, `Prompts`, and `Facilitator` worksheets. Cannot be empty. |
| `PROFILE_FIELDS` | No | `"ALL"` | Controls which profile columns are included. Set to `ALL` to include every column, or provide a comma-separated list of short names (e.g., `age,gender,education`). The `ID` column is always included regardless of this setting. |
| `BUILD_PROFILE_QA` | No | `False` | If `True`, each agent's profile is formatted as a Q&A snippet (prefixed with "Interviewer:" and "Me:") and inserted into the agent's LLM system message. Accepts: `True`, `False`, `1`, `0`, `yes`, `no`. |
| `BUILD_PROFILE_BACKSTORIES` | No | `False` | If `True`, a first-person narrated backstory is generated from the agent's profile via an additional LLM call and inserted into the agent's system message. If both `BUILD_PROFILE_QA` and `BUILD_PROFILE_BACKSTORIES` are `False`, no profile information is passed to the LLM. Accepts: `True`, `False`, `1`, `0`, `yes`, `no`. |
| `CONTEXT_OVERFLOW_POLICY` | No | `"terminate"` | Strategy for handling LLM context window overflow. Expected values: `terminate` (raise error and stop), `summarize` (compress prior messages via LLM summarisation), or `truncate` (drop oldest messages to fit). Case-insensitive. |

**Important notes:**
- Unknown keys are logged as warnings but do not prevent compilation.
- Boolean values accept multiple representations: `"true"`/`"false"`, `"1"`/`"0"`, `"yes"`/`"no"`.
- `MODULE_SEQUENCE` is parsed from a comma-separated string into an ordered list at compile time. Whitespace around module names is trimmed.

---

## 2. `C` (Constants)

Defines module-specific and global numeric/string/boolean constants that can be dynamically injected into prompts and facilitator definitions using Jinja2 templates.

| Column | **Required** | Description |
| - | - | - |
| `module` | **Yes** | The module name this constant belongs to (must match a name in `MODULE_SEQUENCE`). Use `global` for experiment-wide constants that apply across all modules. |
| `name` | **Yes** | The constant identifier (e.g., `ENDOWMENT`, `MAX_NUM_ROUNDS`, `PLAYERS_PER_GROUP`). |
| `value` | **Yes** | The constant value. Rows with empty values are silently skipped. |
| `type` | No (default: `string`) | Controls how `value` is coerced to a Python type. Expected values: `integer` (or `int`), `float`, `string` (or `str`, `text`), `boolean` (or `bool`). If coercion fails, the value is kept as a string with a warning logged. |

**Jinja access pattern:** `{{ C.<module>.<name> }}` (e.g., `{{ C.module1.ENDOWMENT }}`)

**Special constants (expected by the platform):**

| Name | Scope | Type | Purpose |
| - | - | - | - |
| `MAX_NUM_ROUNDS` | Per module | integer | Maximum number of rounds in the module. Required by the flow validator for each module. The session terminates the module prematurely if this limit is exceeded. Useful for preventing infinite loops with repeated prompts. |
| `PLAYERS_PER_GROUP` | Per module | integer | Number of agents assigned to each group. If omitted, defaults to `NUM_AGENTS_PER_SESSION` (i.e., all agents in one group). |

**Important notes:**
- Rows with empty `module` or `name` are silently skipped.
- Constants are resolved at **compile time** — they are baked into the compiled experiment and not re-evaluated at runtime. This means `{{ C.module1.ENDOWMENT }}` in a prompt is replaced with the literal value before execution begins.

---

## 3. `Fields`

Defines all data variables (fields) that agents write to during the experiment. Each field has a scope (class), a data type, and optional validation constraints.

| Column | **Required** | Default | Description |
| - | - | - | - |
| `module` | **Yes** | (none) | Module name from `MODULE_SEQUENCE`. |
| `class` | **Yes** | (none) | The hierarchy scope at which the field is tracked. Expected values: `Session`, `Subsession`, `Agent`, `Group`, `Player`. See scope semantics below. |
| `name` | **Yes** | (none) | Field identifier (variable name). Used in Jinja templates and state lookups. Must be unique within the same `(module, class)` combination. |
| `type` | **Yes** | `"text"` | Data type for validation and response formatting. Expected values: `integer`, `float`, `text`, `category`, `boolean`. |
| `response_options` | No | `None` | Allowed response values. Can be a Python list literal (`[0, 1, 2, 3, 4, 5]`), a Python dict literal (`{"a": "Option A", "b": "Option B"}`), or a comma-separated string (`opt1,opt2,opt3`). If empty, the field accepts free-form responses. |
| `response_options_intro` | No | `""` | Introductory text displayed before presenting response options to the agent (e.g., `"Please choose one of the following:"`). |
| `randomise_options_order` | No | `False` | If `True`, response options are shuffled before presentation to the agent. |
| `validate` | No | `False` | If `True`, agent responses are validated against `response_options`. Invalid responses trigger re-prompting (up to 3 retries). |
| `generate_speculation_score` | No | `False` | If `True`, the LLM is asked to self-assess how speculative its answer is (0–100). The score is extracted and stored separately as `{field_name}_speculation_score`. |
| `format_response` | No | `False` | If `True`, the platform instructs the LLM to return a structured JSON response and parses the result accordingly. |

**Field scope semantics:**

| Class | Persistence | Description |
| - | - | - |
| `Session` | Entire session | Shared across all agents, modules, and rounds. |
| `Subsession` | One round | Shared across all groups within a single round of a module. |
| `Agent` | Across rounds | Persists across all rounds for a single agent, indexed by module and round number. Values from any round are accessible via `{{ agent.<module>[<round_number>].<field_name> }}`. Player-scoped field values are automatically archived to the agent scope after each round for cross-round access. |
| `Group` | One round, one group | Scoped to a specific group in a specific round. |
| `Player` | One round, one agent | Per-round, per-group instance of an agent. The most granular scope. |

**Response options parsing:** The platform first attempts `ast.literal_eval()` to parse Python literals (lists, dicts, tuples). If that fails, it falls back to comma-separated string splitting (e.g., `"a,b,c"` → `["a", "b", "c"]`). Whitespace is stripped from each element.

**Important notes:**
- Boolean columns (`randomise_options_order`, `validate`, `generate_speculation_score`, `format_response`) treat empty/`NaN` values as `False`.
- Fields are looked up at runtime by the composite key `(module, class, name)`.
- `PUBLIC_QUESTION`, `PRIVATE_QUESTION`, and `DISCUSSION` prompts in the Prompts worksheet must link to a field defined here via `field_class` and `field_name`.

---

## 4. `Prompts`

Defines the prompt sequence — what happens in each module and round, what text is presented to the LLM, and how responses are collected.

| Column | **Required** | Default | Description |
| - | - | - | - |
| `module` | **Yes** | (none) | Module name from `MODULE_SEQUENCE`. |
| `prompt_sequence` | No | `0` | Non-negative integer controlling execution order within the module. Prompts with the same sequence value are executed in definition order. |
| `type` | **Yes** | (none) | Prompt type. Expected values: `CONTEXT`, `DISCUSSION`, `PUBLIC_QUESTION`, `PRIVATE_QUESTION`, `FACILITATOR`. See type descriptions below. |
| `is_displayed` | No | `None` (always display) | A Jinja2 boolean expression evaluated at runtime to conditionally display the prompt. Examples: `round_number == 1`, `player.treatment == 'T1'`, `round_number > 1`. If empty or `None`, the prompt always displays. Expressions referencing `player.` or `agent.` attributes are evaluated **per player** within the group, so each agent is individually included or excluded. Non-player expressions (e.g., `round_number == 1`) are evaluated once for the group. If the expression references an undefined variable, a `ValueError` is raised. Other evaluation errors default to displaying the prompt. |
| `is_adapted` | No | `False` | Documentation flag indicating whether the text has been adapted from the original human experiment. Does not affect platform operation. |
| `human_text` | No | `""` | The original instructions from the human experiment (for documentation). Does not affect platform operation. |
| `llm_text` | **Yes** | (none) | The Jinja2 template text sent to the LLM. Supports all variable references (see below). Must be non-empty. |
| `kwargs` | No | `{}` | A Python dict literal of per-prompt keyword arguments, parsed via `ast.literal_eval()`. For FACILITATOR prompts, these kwargs are merged with facilitator-level kwargs (from the Facilitator worksheet); prompt-level kwargs take precedence. Example: `{"strategy": "complete_random", "scope": "group"}`. For other prompt types, supports `{"rag_vector_store_id": "<id>"}` for retrieval-augmented generation. Empty string or `NaN` → empty dict. |
| `field_class` | **Required for question/discussion types** | `None` | The field scope for response storage. Expected values: `Session`, `Subsession`, `Agent`, `Group`, `Player`. Must match a field defined in the `Fields` worksheet. |
| `field_name` | **Required for question/discussion types** | `None` | The field name for response storage. Must match the `name` column in the `Fields` worksheet. Combined with `field_class` to identify the target field. |

**Prompt type descriptions:**

| Type | Purpose | Agents respond? | Visibility | Execution order |
| - | - | - | - | - |
| `CONTEXT` | Provides background information, instructions, or game rules to agents. | No | Each agent receives their own rendered context (per-player Jinja rendering). | Rendered and appended to each agent's message history. |
| `DISCUSSION` | Sequential group discussion. A question is posed to the group; agents respond one by one, each seeing all previous agents' responses. | Yes | All responses visible to group members immediately as they are generated (role `assistant` for the responding agent, role `user` for others). | Facilitator → Agent 1 → Agent 2 → Agent 3 (sequential). |
| `PUBLIC_QUESTION` | Independent public question. Each agent answers the same question without seeing other agents' answers during that round. After all agents have responded, all responses become visible to the group. | Yes | Responses are collected first, then made visible to the entire group. | All agents answer independently; responses appended to histories afterward. |
| `PRIVATE_QUESTION` | Private question. Each agent answers independently. Responses are only visible to the sender. | Yes | Only the responding agent can see their own response. Other agents cannot see it. | Agents answer in parallel (or sequentially, depending on implementation); responses remain private. |
| `FACILITATOR` | Executes a facilitator function (built-in or custom LLM instruction). Not shown to agents. | No (facilitator responds) | Facilitator response appended to agent histories with `facilitator` visibility scope. | Executed by the `FacilitatorEngine`. |

**Important notes:**
- `PUBLIC_QUESTION`, `PRIVATE_QUESTION`, and `DISCUSSION` types **must** specify both `field_class` and `field_name`. Omitting these raises a `ValueError` during compilation.
- `CONTEXT` and `FACILITATOR` types do not require field references.
- Compile-time Jinja references (e.g., `{{ C.module1.ENDOWMENT }}`) are resolved during compilation and baked into the prompt. Runtime references (e.g., `{{ player.age }}`, `{{ round_number }}`) remain as placeholders and are evaluated during execution.

**Jinja variables available in `llm_text` (Prompts worksheet) and `definition` (Facilitator worksheet):**

| Variable | Source | Description |
| - | - | - |
| `{{ C.<module>.<constant_name> }}` | Constants sheet | Constants resolved at compile time (e.g., `{{ C.module1.ENDOWMENT }}`). |
| `{{ player.<profile_field> }}` | Profiles sheet | Profile attributes (e.g., `{{ player.age }}`, `{{ player.gender }}`). |
| `{{ player.<field_name> }}` | Player-scoped `Manual_` / runtime | Player-scoped field values for the current round, including values set via `Manual_` with `class=Player`. |
| `{{ player.player_id }}` | Built-in | Player identifier for the current round. |
| `{{ player.agent_id }}` | Built-in | Agent identifier associated with this player. |
| `{{ player.agent_instance_id }}` | Built-in | Agent instance identifier associated with this player. |
| `{{ agent.<profile_field> }}` | Profiles sheet | Profile attributes as flat agent attributes (e.g., `{{ agent.age }}`, `{{ agent.gender }}`). |
| `{{ agent.<field_name> }}` | Agent-scoped `Manual_` / runtime | Agent-scoped field values for the current round as flat attributes (e.g., `{{ agent.treatment }}`). |
| `{{ agent.<module>[<round>].<field_name> }}` | Agent scope (cross-round) | Agent-scoped field values indexed by module and round number for cross-round access (e.g., `{{ agent.pgg[1].contribution }}`). Player-scoped values are automatically archived here after each round. |
| `{{ agent.agent_id }}` | Built-in | Agent identifier (the raw profile `ID` value). |
| `{{ agent.agent_instance_id }}` | Built-in | Run-specific agent instance identifier. |
| `{{ group.<field_name> }}` | Group-scoped `Manual_` / runtime | Group-scoped field values for the current round, including values set via `Manual_` with `class=Group`. |
| `{{ group.group_id }}` | Built-in | Current group identifier. |
| `{{ group.num_players }}` | Built-in | Number of players in the current group. |
| `{{ session.<field_name> }}` | Session-scoped `Manual_` / runtime | Session-scoped field values, including values set via `Manual_` with `class=Session`. |
| `{{ session.session_id }}` | Built-in | Current session identifier. |
| `{{ session.run_id }}` | Built-in | Unique experiment run identifier. |
| `{{ session.experiment_id }}` | Built-in | Experiment identifier from `Settings`. |
| `{{ round_number }}` | Built-in | Current round number (1-based). |
| `{{ group_id }}` | Built-in | Current group identifier (shorthand). |
| `{{ session_id }}` | Built-in | Current session identifier (shorthand). |
| `{{ run_id }}` | Built-in | Unique experiment run identifier (shorthand). |

*Note:* Values set via the `Manual_` sheet are accessible in the scope matching their `class` column: `Session` → `{{ session.<field> }}`, `Agent` → `{{ agent.<field> }}`, `Group` → `{{ group.<field> }}`, `Player` → `{{ player.<field> }}`. Player-scoped values are automatically archived to the agent scope after each round, enabling cross-round access via `{{ agent.<module>[<round_number>].<field_name> }}`.

---

## 5. `Facilitator`

Defines facilitator functions that control experiment flow. The platform includes a built-in function; all other names are treated as custom LLM-powered facilitator instructions.

| Column | **Required** | Default | Description |
| - | - | - | - |
| `name` | **Yes** | (none) | Function identifier. Built-in names: `assign_groups`. Any other name is treated as a custom LLM facilitator. |
| `definition` | **Yes** | (none) | For built-in functions: leave empty (not used at runtime). For custom functions: a natural-language instruction sent to the LLM as a prompt. Supports Jinja2 templates (e.g., `{{ C.module1.ENDOWMENT }}`). |

**Built-in facilitator functions:**

| Name | Purpose | Runtime Args (via Prompts `kwargs` column) |
| - | - | - |
| `assign_groups` | Rebuild groups at runtime. Uses the `RandomisationEngine`. | `{"strategy": "random"}`, `{"strategy": "swap"}`, `{"strategy": "stratified", "stratify_by": "gender"}` |

**`assign_groups` runtime args:**
- `strategy` (str): One of `random`, `keep`, `swap`, `stratified`. **`manual` is not allowed at runtime.** Defaults to `"random"`.
- `stratify_by` (str, optional): Profile attribute name for stratified grouping. Required when `strategy="stratified"`.

**Custom facilitator functions:**
- Any function name not in the built-in list is treated as a custom LLM instruction.
- The `definition` is rendered as a Jinja2 template and sent to the LLM as the system prompt.
- The facilitator sees the full group conversation history (all agents' messages) formatted with sender identity and round context.
- Custom facilitators can return **stop signals** to control flow:
  - `end_round` — stops the current round loop for the current group.
  - `end_session` — stops the entire session across all groups.
  - `continue` — explicitly continues (default behaviour).

**Important notes:**
- Arguments for built-in facilitators are specified per-prompt in the Prompts worksheet `kwargs` column.
- Facilitator-level kwargs can also be specified in a `kwargs` column on the Facilitator worksheet. They are merged with prompt-level kwargs (prompt-level takes precedence).
- Custom facilitator responses are appended to agent message histories with `facilitator` visibility scope.

---

## 6. `Profiles`

Defines agent persona characteristics (demographics, attributes, profile data). Uses a special two-row header format.

**Structure:**
```
Row 0:    Short Names (Jinja identifiers)     → ID, age, gender, education, ...
Row 1:    Full Question Wording (labels)       → ID, "What is your age?", "What is your gender?", ...
Rows 2+:  Profile Data (one row per agent)    → 1, 25, "Male", "University", ...
                                               → 2, 32, "Female", "High School", ...
```

| Component | Required | Constraints |
| - | - | - |
| **Row 0** (Short Names) | **Yes** | Each must be a valid Jinja2 identifier: letters, digits, and underscores only; must start with a letter (matches `^[a-zA-Z][a-zA-Z0-9_]*$`). Must be non-blank and unique. |
| **Row 1** (Full Names) | **Yes** | Human-readable question wording. Must be non-blank. Column count must match Row 0. |
| **Data Rows** | **Yes** | One row per agent profile. Column count must match headers. |

**ID column requirements:**
- There **must** be a column with the short name `ID` (case-insensitive match). This is required even if you do not intend to provide any other profile information.
- ID values must be **unique** across all profile rows. Duplicates raise a `ValueError`.
- IDs are used as stable agent identifiers (the raw profile `ID` value).

**Jinja access pattern:** Profile fields are accessible as both `{{ agent.<short_name> }}` and `{{ player.<short_name> }}` in prompts (e.g., `{{ agent.age }}` or `{{ player.age }}`).

**Important notes:**
- The `PROFILE_FIELDS` setting in `Settings` controls which columns are included. If set to `ALL` (default), all columns are passed through. If set to a comma-separated list, only those columns (plus `ID`) are included.
- Depending on `BUILD_PROFILE_QA` and `BUILD_PROFILE_BACKSTORIES` in `Settings`, profile data is formatted and inserted into each agent's LLM system message. If both are `False`, no profile information is passed to the LLM (though profile fields are still available via `{{ agent.<field> }}` or `{{ player.<field> }}` in prompts).
- The number of profile data rows must be at least `NUM_AGENTS_PER_SESSION`.

---

## 7. `Manual_` (Optional)

A general-purpose mechanism for populating any field on any class (Session, Agent, Group, Player). Values set here are automatically accessible via Jinja dot notation in both `llm_text` (Prompts worksheet) and `definition` (Facilitator worksheet).

You can use a single worksheet named `Manual_` or multiple worksheets with the prefix `Manual_` (e.g., `Manual_Treatments`, `Manual_Groups`). All are merged into a single registry.

| Column | **Required** | Description |
| - | - | - |
| `ID` | **Yes** | Must correspond to a value in the `ID` column of the `Profiles` worksheet. Identifies which agent receives the assignment. |
| `module` | Conditional | Module name. If empty, the assignment applies to all modules in `MODULE_SEQUENCE`. At least one of `module` or `class` must be non-empty. |
| `round_number` | No | If empty or `NaN`, the assignment applies to all rounds. If specified, it applies only to that round. |
| `class` | Conditional | Hierarchy level of the assignment. Expected values: `Session`, `Module`, `Subsession`, `Group`, `Agent`, `Player`. At least one of `module` or `class` must be non-empty. |
| `name` | **Yes** | The field name to populate. For group assignments: `id_in_subsession`. For other field values: any defined field name. |
| `value` | **Yes** | The assigned value. For groups: a group identifier. For fields: the field value. |

**Group assignment rows:**
- Set `name = "id_in_subsession"` and `class = "Group"`.
- The `value` is a group identifier. Agents with the same group value are placed in the same group.

**Scoping behaviour:**

| `module` value | `round_number` value | Behaviour |
| - | - | - |
| (blank) | (blank) | Applied to every module in `MODULE_SEQUENCE` and every round. Each module/round combination stores a separate value (no overwriting). |
| Specific module | (blank) | Applied to every round of that module. Each round stores a separate value. |
| Specific module | Specific round | Applied to that specific module and round only. |

**Cross-round access:**
- Player-scoped field values are automatically archived to the agent scope at the end of each round. This means you can access a player's response from a previous round via the agent scope: `{{ agent.<module>[<round_number>].<field_name> }}`.
- Agent-scoped values set via `Manual_` with blank module/round are applied to each module/round as separate entries, so a facilitator modifying a value within a round only affects that specific round's copy.

**Important notes:**
- Profile IDs referenced in `Manual_` sheets are cross-validated against the `Profiles` worksheet. If a referenced ID does not exist, a `ValueError` is raised.
- Duplicate keys (same `ID`, `module`, `round_number`, `class`, `name`) result in the later value overwriting the earlier one with a warning logged.
- Rows missing `ID` or `name` raise a `ValueError`.

---

## Validation Pipeline (v0.3.1)

When you run `talkingtomachines validate`, the platform executes the following validators in order:

1. **Schema Validator** — Checks worksheet structure, column presence, required fields.
2. **Reference Validator** — Validates Jinja references (constants, fields, profiles, facilitators) to ensure all referenced variables exist.
3. **Flow Validator** — Validates prompt ordering, field linkage, and confirms `MAX_NUM_ROUNDS` and `PLAYERS_PER_GROUP` are defined for each module.
4. **Provider Validator** — Checks that the LLM model is valid, the corresponding API key is set, and the model supports any features used (e.g., RAG, visual inputs).
5. **Context Window Validator** — Estimates total token usage to ensure the experiment fits within the model's context window. Reports warnings at 80% utilisation and errors if the limit would be exceeded.

---

## Demo Examples

You may also explore these example experimental designs and their accompanying prompt templates prepared by the development team:

* **Simple Public Goods Experiment**: [Simple Public Goods Experiment Demo](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/simple_pgg)

* **Complex Public Goods Experiment**: [Complex Public Goods Experiment Demo](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/pgg)

* **Manual Conjoint Experiment**: [Manual Conjoint Experiment Demo](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/conjoint_manual)

* **Randomized Conjoint Experiment**: [Randomized Conjoint Experiment Demo](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/conjoint_auto)

---

## 📹 Video Walkthrough
A video walkthrough on how to populate the prompt template workbook based on a simple public goods experiment can be found here: [Video Walkthrough](https://www.loom.com/share/8ecad62ff7b947efb90d338a303262e1)

