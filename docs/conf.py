# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'NCPS'
copyright = '2024, NCPS Team'
author = 'NCPS Team'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Automatically include docstrings
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',     # Add links to highlighted source code
    'sphinx.ext.intersphinx',  # Link to other project's documentation
    'sphinx.ext.mathjax',      # Render math via MathJax
    'sphinx.ext.todo',         # Support for todo items
    'sphinx_rtd_theme',        # Read the Docs theme
    'myst_parser',            # Support for Markdown
]

# -- Options for Markdown support -------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# -- Extension configuration ------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# -- Intersphinx configuration ---------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'tensorflow': ('https://www.tensorflow.org/api_docs/python', None),
}

# -- Todo configuration ---------------------------------------------------
todo_include_todos = True

# -- Custom configuration ------------------------------------------------
# Add any custom configuration specific to NCPS here

# Directory structure for documentation
html_additional_pages = {
    'architecture': 'architecture.html',
    'api': 'api.html',
    'guides': 'guides.html',
}

# Sidebar navigation depth
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

# Custom roles and directives
rst_epilog = """
.. |project| replace:: NCPS
.. |version| replace:: 1.0.0
"""

# -- Setup function for custom processing ---------------------------------
def setup(app):
    # Add custom setup here if needed
    pass