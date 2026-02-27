"""Sphinx configuration for MineLab documentation."""

project = "MineLab"
copyright = "2026, MineLab Contributors"
author = "MineLab Contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
