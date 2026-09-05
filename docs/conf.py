"""Sphinx configuration for csv-toolkit API docs."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

project = "csv-toolkit"
author = "Wesley Cialek"
version = release = "0.1.0"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.viewcode"]
templates_path = []
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
