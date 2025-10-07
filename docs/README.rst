NCPS Documentation
==================

This directory contains the NCPS documentation, built using Sphinx.

Quick Start
-----------

1. Set up documentation environment:

.. code:: bash

# Make setup script executable
chmod +x tools/setup.sh

# Run setup
./tools/setup.sh

# Activate documentation environment
source activate_docs

2. Use documentation tools:

.. code:: bash

# Show available commands
doctools help

# Build documentation
doctools build

# Preview in browser
doctools preview

Directory Structure
-------------------

::

docs/
├── _static/            # Static files (CSS, images)
├── _templates/         # Custom templates
├── architecture/       # Architecture documentation
│   ├── abstractions/  # Abstraction layer designs
│   ├── implementation/# Implementation plans
│   ├── knowledge/     # Architecture knowledge base
│   └── design/        # Core design documentation
├── api/               # API documentation
├── guides/            # User guides
├── examples/          # Code examples
└── tools/            # Documentation tools
    ├── doctools      # Main documentation management script
    ├── setup.sh      # Environment setup script
    └── README.md     # Tools documentation

Documentation Tools
-------------------

The ``doctools`` command provides a unified interface for documentation
management:

.. code:: bash

# Convert markdown to RST
doctools convert

# Run documentation tests
doctools test

# Create new documentation
doctools create "Title" --template guide

# Build documentation
doctools build

# Preview in browser
doctools preview

# Clean build directory
doctools clean

See :doc:`tools/README.md  </tools/README.md>`                 _ for detailed documentation on
available tools and templates.

Building Documentation
----------------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.8 or later
- pip
- virtualenv
- pandoc (installed automatically by setup.sh on macOS/Linux)

Build Commands
~~~~~~~~~~~~~~

1. HTML Documentation:

.. code:: bash

doctools build

2. Live Preview (auto-rebuilds on changes):

.. code:: bash

doctools preview

3. PDF Documentation:

.. code:: bash

doctools build pdf

4. Check Documentation:

.. code:: bash

doctools test

Contributing
------------

1. Set up the documentation environment:

.. code:: bash

./tools/setup.sh
source activate_docs

2. Create or modify documentation:

.. code:: bash

# Create new documentation
doctools create "My Guide" --template guide

# Convert existing markdown
doctools convert

3. Test your changes:

.. code:: bash

doctools test

4. Preview the results:

.. code:: bash

doctools preview

Documentation Standards
-----------------------

1. File Format

- Use RST format for documentation
- Follow Sphinx conventions
- Include code examples where appropriate

2. Organization

- Place files in appropriate directories
- Update index files
- Maintain logical structure

3. Style

- Clear, concise writing
- Proper formatting
- Consistent terminology

4. Code Examples

- Tested and working
- Well-commented
- Follow style guide

Getting Help
------------

1. Tool Help:

.. code:: bash

doctools help

2. Documentation:

- Check :doc:`tools/README.md  </tools/README.md>`                 _ for tool documentation
- Review existing documentation for examples
- Consult Sphinx documentation: https://www.sphinx-doc.org/

3. Issues:

- Run ``doctools test`` for validation
- Check error messages
- Review build output

Maintenance
-----------

1. Regular Tasks:

- Run tests
- Update dependencies
- Check links
- Verify examples

2. Updates:

- Keep content current
- Update templates
- Maintain tools
- Review organization

This documentation system is designed to be maintainable, extensible,
and user-friendly. Use the provided tools to ensure consistency and
quality.
