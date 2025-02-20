Documentation Tools
===================

This directory contains tools for managing NCPS documentation.

Quick Start
-----------

1. Set up documentation environment:

.. code:: bash

# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh

# Activate documentation environment
source ../activate_docs

2. Use documentation tools:

.. code:: bash

# Show available commands
doctools help

# Build documentation
doctools build

# Preview in browser
doctools preview

Available Tools
---------------

1. convert_docs.py
~~~~~~~~~~~~~~~~~~

Converts markdown files to RST and organizes them into the Sphinx
structure.

.. code:: bash

# Convert and organize all documentation
doctools convert

# The script will:
# 1. Convert MD to RST
# 2. Organize files into appropriate directories
# 3. Create index files

2. test_docs.py
~~~~~~~~~~~~~~~

Tests documentation builds and checks for common issues.

.. code:: bash

# Run all documentation tests
doctools test

# Tests include:
# - RST syntax validation
# - Documentation build
# - Link checking
# - Code block validation
# - Cross-reference verification

3. create_doc.py
~~~~~~~~~~~~~~~~

Creates new documentation files using templates.

.. code:: bash

# Create a guide
doctools create "My Guide Title" --template guide --directory guides

# Create API documentation
doctools create "MyClass API" --template api --directory api \
    --module-path ncps.module --class-name MyClass

# Create architecture documentation
doctools create "Design Document" --template architecture \
    --directory architecture/design

# Create research documentation
doctools create "Research Topic" --template research \
    --directory architecture/research

4. fix_rst.py
~~~~~~~~~~~~~

Fixes common RST syntax issues and code block formatting.

.. code:: bash

# Fix RST syntax in all documentation
doctools fix

Templates
---------

Guide Template
~~~~~~~~~~~~~~

- Overview
- Prerequisites
- Getting Started
- Detailed Instructions
- Advanced Usage
- Troubleshooting

API Template
~~~~~~~~~~~~

- Overview
- Classes
- Examples
- Functions
- See Also

Architecture Template
~~~~~~~~~~~~~~~~~~~~~

- Overview
- Design Goals
- Implementation Details
- Integration Points
- Performance Considerations
- Future Considerations

Research Template
~~~~~~~~~~~~~~~~~

- Overview
- Background
- Methodology
- Results
- Conclusions
- Future Work
- References

Usage Examples
--------------

Converting Documentation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

# Convert all markdown files
doctools convert

# Test the conversion
doctools test

Creating New Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

# Create a new guide
doctools create "Installation Guide" \
    --template guide \
    --directory guides

# Create API documentation
doctools create "TensorAbstraction API" \
    --template api \
    --directory api \
    --module-path ncps.abstractions \
    --class-name TensorAbstraction

Testing Documentation
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

# Run all tests
doctools test

# Build documentation after changes
doctools build

Best Practices
--------------

1. Documentation Creation

- Use appropriate templates
- Follow RST syntax guidelines
- Include practical examples
- Add proper cross-references

2. Testing

- Run tests before committing
- Fix all reported issues
- Verify links work
- Check code examples

3. Organization

- Use correct directories
- Update index files
- Maintain proper hierarchy
- Follow naming conventions

Maintenance
-----------

1. Regular Tasks

- Run tests periodically
- Update broken links
- Verify code examples
- Check cross-references

2. Updates

- Keep templates current
- Update test scripts
- Maintain conversion tools
- Add new features as needed

Getting Help
------------

1. Tool Help:

.. code:: bash

doctools help

2. Documentation:

- Check Sphinx documentation: https://www.sphinx-doc.org/
- Review existing docs for examples
- Use test output for debugging

3. Issues:

- Run ``doctools test`` for validation
- Check error messages
- Review build output

These tools are designed to make documentation management easier and
more consistent. Use them regularly to maintain high-quality
documentation.
