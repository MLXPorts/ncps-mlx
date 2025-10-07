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
    'sphinx_gallery.gen_gallery',  # Support for code examples gallery
    'sphinx.ext.autosummary',  # Generate summaries automatically
    'sphinx.ext.doctest',      # Test code examples in documentation
    'sphinx.ext.coverage',     # Check documentation coverage
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

# Template configuration
templates_path = ['_templates']

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
}

# -- Todo configuration ---------------------------------------------------
todo_include_todos = True

# -- Sphinx Gallery configuration -----------------------------------------
sphinx_gallery_conf = {
    'examples_dirs': '../examples',   # path to example scripts
    'gallery_dirs': 'auto_examples',  # path to save generated output
    'filename_pattern': '/plot_',     # pattern to match example files
    'ignore_pattern': '__init__\\.py',
    'plot_gallery': True,
    'download_all_examples': True,
    'within_subsection_order': lambda folder: sorted(folder),
    'show_memory': True,
    'thumbnail_size': (400, 400),
    'remove_config_comments': True,
    'min_reported_time': 0,
    'show_signature': True,
    'backreferences_dir': 'gen_modules/backreferences',
    'doc_module': ('sphinx_gallery', ),
    'reference_url': {
        'sphinx_gallery': None,
    },
    'inspect_global_variables': True,
    'remove_config_comments': True,
    'thumbnail_size': (320, 224),
    'first_notebook_cell': None,
    'pypandoc': False,
}

# -- Custom configuration ------------------------------------------------
# Add any custom configuration specific to NCPS here

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
    # Register architecture template for RST files in architecture directory
    app.add_html_theme('architecture', os.path.join(os.path.dirname(__file__), '_templates'))
    pass