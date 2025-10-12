# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "Neural Circuit Policies for Apple MLX"
copyright = "2025, Sydney Renee (MLX port). Original work: 2023, Mathias Lechner"
author = "Sydney Renee"

# The short X.Y version
version = "1.0"
# The full version, including alpha/beta/rc tags
release = "1.0.0"

html_favicon = "img/ncp_32.ico"
needs_sphinx = "4.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"
autodoc_typehints = "description"
autoclass_content = "init"
autodoc_inherit_docstrings = False
autodoc_default_options = {
    "undoc-members": False,
    "member-order": "bysource",
    "show-inheritance": False,
}
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_logo = "img/banner.png"
html_theme_options = {
    "sidebar_hide_name": True,
}
