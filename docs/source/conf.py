# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "talking-to-machines"
copyright = "2025, Raymond Low, Tommaso Batistoni, Ray Duch"
author = "Raymond Low, Tommaso Batistoni, Ray Duch"
release = "0.3.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]
exclude_patterns = ["build", ".DS_Store", "talkingtomachines.rst"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Nitpicky exceptions --------------------------------------------------
nitpick_ignore = [
    ("py:class", "openai.OpenAI"),
    ("py:class", "pandas.core.frame.DataFrame"),
    ("py:class", "pandas.io.excel._base.ExcelFile"),
    ("py:class", "pd.ExcelFile"),
    ("py:class", "pd.DataFrame"),
    ("py:class", "optional"),
    ("py:exc", "json.JSONDecodeError"),
    ("py:class", "pathlib.Path"),
    ("py:class", "random.Random"),
    # Project-internal unqualified references used in docstrings
    ("py:class", "AnthropicProvider"),
    ("py:class", "BudgetExhaustedError"),
    ("py:class", "ContextWindowExceededError"),
    ("py:class", "CostTracker"),
    ("py:class", "DeepSeekProvider"),
    ("py:class", "EventLogger"),
    ("py:class", "GoogleProvider"),
    ("py:class", "GrokProvider"),
    ("py:class", "HuggingFaceProvider"),
    ("py:class", "LLMProvider"),
    ("py:class", "LLMResponse"),
    ("py:class", "LLMRouter"),
    ("py:class", "MistralProvider"),
    ("py:class", "OpenAIProvider"),
    ("py:class", "OpenAIRAGTool"),
    ("py:class", "OpenRouterProvider"),
    ("py:class", "ProfileInfo"),
    ("py:class", "RAGResult"),
    ("py:class", "RAGTool"),
    ("py:class", "ValidationError"),
    ("py:data", "ManualRegistry"),
    ("py:data", "OUTPUT_RESERVE_TOKENS"),
    ("py:data", "SAFETY_MARGIN"),
    ("py:data", "_PROVIDER_KEY_MAP"),
    ("py:func", "_detect_provider"),
    ("py:func", "get_model_spec"),
    ("py:func", "parse_manual_sheets"),
    ("py:func", "serialize_session"),
    ("py:func", "to_json_safe"),
    ("py:meth", "_truncate"),
    ("py:meth", "generate"),
    ("py:meth", "retrieve"),
    ("py:meth", "supports_audio"),
    ("py:meth", "supports_vision"),
    ("py:meth", "to_dict"),
    ("py:meth", "validate"),
    ("py:meth", "~LLMProvider.generate"),
    ("py:meth", "LLMProvider.generate"),
    ("py:attr", "LLMRouter.MAX_RETRIES"),
    ("py:attr", "_use_chat_completions"),
    ("py:attr", "budget_cap_usd"),
    ("py:attr", "provider_name"),
    ("py:class", "Role"),
    ("py:class", "T"),
    ("py:class", "callable"),
    ("py:class", "dict[str"),
]

nitpick_ignore_regex = [
    # Fully-qualified internal references
    (r"py:.*", r"talkingtomachines\..*"),
    (r"py:.*", r"exporters\..*"),
    # Third-party libraries without intersphinx inventories
    (r"py:.*", r"openai\..*"),
    (r"py:.*", r"pd\..*"),
    (r"py:.*", r"pandas\..*"),
]

# Suppress duplicate object warnings from autodoc discovering the same objects
# through both submodule pages and parent package namespaces.
import logging


class _DuplicateFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return (
            "duplicate object description" not in msg
            and "more than one target found" not in msg
        )


logging.getLogger("sphinx").addFilter(_DuplicateFilter())
