# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../talkingtomachines/"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "talking-to-machines"
copyright = "2025, Raymond Low, Tommaso Batistoni, Ray Duch"
author = "Raymond Low, Tommaso Batistoni, Ray Duch"
release = "0.2.7"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon", "sphinx.ext.autodoc", "sphinx.ext.viewcode"]

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
    ("py:class", "ProfileInfo"),
    ("py:class", "optional"),
    ("py:exc", "json.JSONDecodeError"),
    ("py:class", "pathlib.Path"),
]

nitpick_ignore_regex = [
    (r"py:.*", r"talkingtomachines\.core\..*"),
    (r"py:.*", r"talkingtomachines\.compiler\..*"),
    (r"py:.*", r"talkingtomachines\.gateway\..*"),
    (r"py:.*", r"talkingtomachines\.orchestrator\..*"),
    (r"py:.*", r"talkingtomachines\.agents\..*"),
    (r"py:.*", r"talkingtomachines\.authoring\..*"),
    (r"py:.*", r"exporters\..*"),
]
