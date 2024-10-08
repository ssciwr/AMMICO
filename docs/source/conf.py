# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../ammico/"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "AMMICO"
copyright = "2022, Scientific Software Center, Heidelberg University"
author = "Scientific Software Center, Heidelberg University"
release = "0.2.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "myst_parser", "nbsphinx"]
nbsphinx_allow_errors = True
nbsphinx_execute = "never"
napoleon_custom_sections = [("Returns", "params_style")]
myst_heading_anchors = 3

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "ssciwr",  # Username
    "github_repo": "AMMICO",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/source/",  # Path in the checkout to the docs root
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
