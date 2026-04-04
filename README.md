# Talking to Machines Platform (Beta Release)

The `talkingtomachines` platform facilitates the design, conduct, and analysis of large-scale experimental trials and treatments with LLM-powered agents.

---

## ✨ Features

* **CLI-first Workflow** – Build and run large-scale experimental trials straight from your terminal.

* **Python Package** – Use the same engine as a Python package in Jupyter notebooks and Python pipelines.

* **Multi-Model Providers** – Built-in support for **OpenAI**, **Anthropic (Claude)**, **Google (Gemini)**, **Mistral**, **xAI (Grok)**, **DeepSeek**, **Hugging Face** Inference API, and **OpenRouter** models.

* **Reproducible** – Every run is JSON-logged for auditable and reproducible results.

* **Excel-based Configuration** – Researchers configure experiments entirely through an Excel workbook with no programming required.

---

## Version History

| Version | Architecture | Status |
|---------|-------------|--------|
| **v0.3.0** | oTree-inspired hierarchy (Session → Module → Subsession → Group → Agent/Player). Compiler-based pipeline with validation, checkpointing, and multi-format export. | Development (current branch) |

---

## ⚙️ Requirements

* Python >= 3.10

* macOS or Windows 10/11

* Network access and API keys to your chosen model providers

---

## 🔧 Installation

### 1. Install `Miniconda` (recommended)

* Download `Miniconda` for your OS (Windows, macOS):
  [https://www.anaconda.com/docs/getting-started/miniconda/install](https://www.anaconda.com/docs/getting-started/miniconda/install)
* Run the installer and accept the default configurations.

  * **Windows**: Add `conda` to your PATH (ignore the installer's recommendation).
  * **macOS**: You may need to restart your terminal after installation.

Verify your installation by running the following command in your terminal (for macOS) or Windows Powershell (for Windows):

```bash
conda --version
```

### 2. Create and activate a new `conda` environment

Create a fresh `conda` environment (with Python 3.12):
```bash
conda create -n your-env-name python=3.12
```

Activate your newly created `conda` environment:
```bash
conda activate your-env-name
```

Upgrade the `pip` package inside the `conda` environment (optional step but highly recommended):
```bash
python -m pip install --upgrade pip
```

Verify that `python` and `pip` are properly installed:
```bash
python --version
pip --version
```

### 3. Install the `talkingtomachines` package

**v0.3.0:**
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple talkingtomachines==0.3.0
```

Verify that the `talkingtomachines` package is properly installed:
```bash
talkingtomachines --version
talkingtomachines --help
```

---

## API Key Setup

Set the API keys for your chosen model provider(s) as environment variables. You only need to set the key(s) for the provider(s) you plan to use. An error will be raised during validation if a model is selected but its corresponding API key is not set.

### macOS / Linux

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
export MISTRAL_API_KEY=...
export XAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export HF_API_KEY=hf_...
export OPENROUTER_API_KEY=sk-or-...
```

### Windows (PowerShell)

```powershell
$env:OPENAI_API_KEY='sk-...'
$env:ANTHROPIC_API_KEY='sk-ant-...'
$env:GOOGLE_API_KEY='...'
$env:MISTRAL_API_KEY='...'
$env:XAI_API_KEY='...'
$env:DEEPSEEK_API_KEY='...'
$env:HF_API_KEY='hf_...'
$env:OPENROUTER_API_KEY='sk-or-...'
```

### Using a `.env` file

Alternatively, you can create a `.env` file in your project directory with your API keys. The platform uses `python-dotenv` to automatically load environment variables from a `.env` file at startup:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 🚀 CLI Usage

### v0.3.0

v0.3.0 uses a subcommand-based CLI built on [Click](https://click.palletsprojects.com/).

#### `init` — Create a new experiment project

```bash
talkingtomachines init my_experiment
talkingtomachines init my_experiment --format csv
talkingtomachines init my_experiment --output ./projects
```

Creates a project folder with a blank prompt template (Excel workbook or CSV files) containing all required worksheets with example rows.

| Flag | Description |
|------|-------------|
| `--format, -fmt` | Template format: `xlsx` (default) or `csv` |
| `--output, -o` | Output directory (default: current directory) |

#### `validate` — Validate a template without running

```bash
talkingtomachines validate path/to/template.xlsx
talkingtomachines validate path/to/csv_directory/
```

Runs all validators (schema, reference, flow, provider, context window) and reports any errors. On success, displays the experiment ID, config hash, module sequence, context window size, and model name.

#### `run` — Compile and execute an experiment

```bash
# Run in test mode (default): one group per module, sequential execution
talkingtomachines run path/to/template.xlsx 
# Alternatively,
talkingtomachines run path/to/template.xlsx --test

# Perform full run: all groups, parallel execution
talkingtomachines run path/to/template.xlsx --full-run

# Set a budget cap (in USD)
talkingtomachines run path/to/template.xlsx --budget 10.0

# Define custom output directory
talkingtomachines run path/to/template.xlsx --output my_results
```

| Flag | Description |
|------|-------------|
| `--test` / `--full-run` | Test mode (default) runs one group per module; full run executes all groups in parallel |
| `--budget` | Budget cap in USD (default: 0 = no cap) |
| `--output, -o` | Output base directory (default: same directory as the template) |

#### Output Files

After a run completes, the following files are produced in the run directory (`<template_dir>/results/<run_id>/`):

| File | Description |
|------|-------------|
| `compiled_experiment.json` | The full Compiled Experiment Package (CEP) containing all settings, field definitions, prompts, assignment plans, and hashes. Created before execution begins. |
| `config.json` | Lightweight copy of experiment settings for quick reference. Created before execution begins. |
| `codebook.json` | Data dictionary documenting all field definitions, response options, data types, prompt definitions, and profile columns. Organised by table (session, agent, group, player). |
| `session_table.csv` | One row per session with run identifiers, timing, cumulative API cost (USD), and stop reason. |
| `agent_table.csv` | One row per agent per session. Includes profile fields, agent-scoped field values for each module, system message, and full message history (JSON). |
| `group_table.csv` | One row per group per subsession. Includes group metadata, round number, and group-scoped field values. |
| `player.csv` | One row per player per subsession. Includes player/agent/group identifiers, round number, and player-scoped field values. |
| `assignments.csv` | One row per manual assignment or group assignment. Documents all entries from the `Manual_` worksheet and group membership for each agent, module, and round. |
| `metrics.csv` | Aggregate experiment metrics: agent count, module count, total rounds, groups, players, messages, API cost, and timing. |
| `traces.jsonl` | Complete event log written live during execution. Each line is a JSON record covering LLM calls, retries, facilitator calls, validation outcomes, checkpoint events, randomisation, and assignments. |
| `events.csv` | Filtered view of `traces.jsonl` containing non-routine events (retries, errors, validation, checkpoints, assignments). Only generated if `traces.jsonl` exists. |
| `checkpoints/<session_id>.json` | Session checkpoint saved at subsession boundaries for fault-tolerant resumption. Contains the serialised Session object hierarchy. |
| `checkpoints/<session_id>_state.json` | Experiment state checkpoint saved alongside the session checkpoint. Contains all recorded field values across all scopes. |

---

## Supported Models

The provider is auto-detected from the model name.

| Provider | Model prefix | Example models | Environment variable |
|----------|-------------|----------------|----------------------|
| OpenAI | `gpt-*`, `o1`, `o3`, `o4`, `o5`, `chatgpt-*` | `gpt-4.1`, `gpt-5`, `o4-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-*` | `claude-opus-4-6`, `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| Google | `gemini-*` | `gemini-2.5-pro`, `gemini-2.0-flash` | `GOOGLE_API_KEY` |
| Mistral | `mistral-*`, `codestral-*` | `mistral-large`, `codestral` | `MISTRAL_API_KEY` |
| xAI | `grok-*` | `grok-3`, `grok-3-mini` | `XAI_API_KEY` |
| DeepSeek | `deepseek-*` | `deepseek-chat`, `deepseek-r1` | `DEEPSEEK_API_KEY` |
| HuggingFace | `hf-*` | Custom inference endpoints | `HF_API_KEY` |
| OpenRouter | `openrouter/*` | `openrouter/anthropic/claude-3` | `OPENROUTER_API_KEY` |

Unrecognised model names default to OpenRouter.

---

## 📄 Prompt Template Setup

Detailed instructions on how to populate the prompt template can be found here: [`Prompt Template Instructions`](https://github.com/talking-to-machines/talking-to-machines/tree/main/talkingtomachines/interface/README.md). The experiment is configured through an Excel workbook. Use `talkingtomachines init` (v0.3.0) to generate a blank template with example rows.

### v0.3.0 Worksheets

v0.3.0 uses 7 worksheets that map to the oTree-inspired hierarchy.

| Worksheet | Purpose |
|-----------|---------|
| **Settings** | Global experiment settings: model name, temperature, random seed, module sequence, context window overflow policy. |
| **C** | Constants defined per module (e.g., `ENDOWMENT`, `MAX_NUM_ROUNDS`, `PLAYERS_PER_GROUP`). Columns: `module`, `name`, `value`, `type`. Accessed via `{{ C.module_name.constant_name }}` in prompts. |
| **Fields** | Data variables that agents write to during the experiment. Columns: `module`, `class`, `name`, `type`, `response_options`, `response_options_intro`, `randomise_options_order`, `validate`, `generate_speculation_score`, `format_response`. |
| **Facilitator** | Built-in and custom facilitator functions. Columns: `name`, `definition`, `kwargs`. Built-in functions: `assign_groups`. Custom functions use natural-language LLM instructions. |
| **Prompts** | The prompt sequence for each module. Columns: `module`, `prompt_sequence`, `type`, `is_displayed`, `is_adapted`, `human_text`, `llm_text`, `rag_vector_store_id`, `field_class`, `field_name`. |
| **Profiles** | Agent profile attributes. Row 0 = short names (Jinja2 identifiers), Row 1 = full question wording, Rows 2+ = profile data. First column must be `ID` with unique values. |
| **Manual_** | Optional manual overrides for treatment assignments, group memberships, and field values. Columns: `ID`, `module`, `round_number`, `class`, `name`, `value`. |

---

## 📹 Video Walkthrough
A video walkthrough on how to set up the `talkingtomachines` platform for both macOS and Windows devices can be found here:

### v0.3.0
macOS: [Video Walkthrough](https://www.loom.com/share/a2c15f1258d5436eaeca197998286cd9?sid=7958bdbe-2f34-4d5f-8d47-49fd49cb315c)

Windows: [Video Walkthrough](https://www.loom.com/share/79969b38be6d4c2387d19ecc3e54ae4d?sid=d372a5e7-6dd5-4923-b838-69eba7dce20a)


---

## Demo Experiments
You may also want to explore these example experimental designs and their accompanying prompt templates prepared by the development team:

* **Public Goods Experiment**: [Public Goods Experiment Demo](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/public_good_experiment)

* **Randomized Controlled Trial (RCT)**: [RCT Demo](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/rct_experiment)

---

## License

MIT License – see [`LICENSE`](https://github.com/talking-to-machines/talking-to-machines/blob/main/LICENSE) for details.
