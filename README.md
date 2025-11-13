# Talking to Machines Platform (Beta Release)


The `talkingtomachines` platform is developed to facilitate the design, conduct, and analysis of large-scale experimental trials and treatments with LLM-powered agents. 

---

## ✨ Features

* **CLI‑first Workflow** – Build and run large-scale experimental trials straight from your terminal.

* **Python Package** – Use the same engine as a Python package in Juypter notebooks and Python pipelines.

* **Multi‑Model Providers** – Built-in support for **OpenAI** chat models, **Hugging Face** Inference API, and OpenRouter.ai models.

* **Reproducible** – Every run is JSON‑logged for auditable and reproducible results.

---

## ⚙️ Requirements

* Python 3.10+

* macOS or Windows 10/11

* Network access and API keys to your chosen model providers (OpenAI, Hugging Face, OpenRouter.ai)

---

## 🔧 Installation


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

Create a fresh `conda` environment (with Python 3.10 or newer):
```bash
conda create -n your-env-name python=3.10
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
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple talkingtomachines
```

Verify that the `talkingtomachines` package is properly installed:
```bash
talkingtomachines --version
talkingtomachines --help
```

---

### 📹 Video Walkthrough
A video walkthrough on how to set up the `talkingtomachines` platform for both macOS and Windows devices can be found here:

macOS: [Video Walkthrough](https://www.loom.com/share/a2c15f1258d5436eaeca197998286cd9?sid=7958bdbe-2f34-4d5f-8d47-49fd49cb315c)

Windows: [Video Walkthrough](https://www.loom.com/share/79969b38be6d4c2387d19ecc3e54ae4d?sid=d372a5e7-6dd5-4923-b838-69eba7dce20a)


---

## 🚀 Usage

### CLI Tool
The `talkingtomachines` platform can be used as a CLI tool for non-technical users or users who are not familiar with Python. To use the CLI tool, you will need to populate a prompt template Excel workbook to define your experimental setup. Detailed instructions on how to properly set up a prompt template for your experiment can be found here: [`Prompt Template Instructions`](https://github.com/talking-to-machines/talking-to-machines/tree/main/talkingtomachines/interface/README.md)

1. Set the API keys for OpenAI, Hugging Face, and OpenRouter.ai as environment variables on your terminal (for macOS) or Windows Powershell (for Windows):

For macOS: 
```bash
export OPENAI_API_KEY=sk-...
export HF_API_KEY=hf_...
export OPENROUTER_API_KEY=sk-...
```

For Windows:
```bash
$env:OPENAI_API_KEY='sk-...'
$env:HF_API_KEY='hf-...'
$env:OPENROUTER_API_KEY='sk-...'
```

2. Provide the file path to the prompt template to allow the `talkingtomachines` platform parse your experimental setup
```bash
talkingtomachines path/to/prompt/template.xlsx
```

You will see a summary of your experimental setup parsed from the prompt template for your verification and the following prompt:
```
Verify the experiment settings above and choose a run mode:
  • Type 'test'  → Runs the session in TEST mode (one randomly selected group per treatment)
  • Type 'full'  → Runs the FULL session
  • Anything else → Terminates the session immediately
Your choice: 
```

Experimental results are saved to your current directory under the `experiment_results` folder with a JSON file containing the raw outputs (`experiment_results/<session_id>.json`) and a CSV file containing the formatted outputs (`experiment_results/<session_id>.csv`).


### Python Package
The `talkingtomachines` platform can also be imported as a Python package for power users/developers who are interested in creating more advanced experimental designs that are currently not supported by the prompt template.

```python
import talkingtomachines
```

---

## 📄 Prompt Template Setup

Detailed instructions on how to populate the prompt template can be found here: [`Prompt Template Instructions`](https://github.com/talking-to-machines/talking-to-machines/tree/main/talkingtomachines/interface/README.md)

You may also explore these example experimental designs and their accompanying prompt templates prepared by the development team:

* **Public Goods Experiment**: A public goods experiment demo example with a populated prompt template workbook and description of its experimental design: [Public Goods Experiment Demo](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/public_good_experiment)

* **Randomized Controlled Trial (RCT)**: A RCT experiment demo example with a populated prompt template workbook: [RCT Demo](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/rct_experiment)

* **Prompt Template**: A unpopulated version of the prompt template has been provided to serve as a starting point for creating new synthetic experiments: [Prompt Template](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/prompt_template.xlsx)

---

## 📜 License

MIT License – see [`LICENSE`](https://github.com/talking-to-machines/talking-to-machines/blob/main/LICENSE) for details.
