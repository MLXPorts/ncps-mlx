#!/usr/bin/env python3
"""
Documentation creation tool for NCPS.
Creates new documentation files with proper templates and structure.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

class DocCreator:
    TEMPLATES = {
        'guide': """
{title}
{title_underline}

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------
[Provide a brief overview of this guide]

Prerequisites
------------
* Requirement 1
* Requirement 2

Getting Started
--------------
[Initial steps or basic concepts]

.. code-block:: python

    # Example code here
    print("Hello, World!")

Detailed Instructions
-------------------
[Step-by-step instructions or detailed explanations]

Advanced Usage
-------------
[Advanced features or configurations]

Troubleshooting
--------------
Common issues and their solutions:

.. note::
   Important information here.

.. warning::
   Critical warnings here.

See Also
--------
* :doc:`/related/document1`
* :doc:`/related/document2`
""",
        
        'api': """
{title}
{title_underline}

.. currentmodule:: ncps

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------
[Brief overview of this API component]

Classes
-------

.. autoclass:: {module_path}.{class_name}
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------
.. code-block:: python

    # Example usage
    from ncps import {class_name}

Functions
---------
[Document any standalone functions]

See Also
--------
* :doc:`/related/document1`
* :doc:`/related/document2`
""",
        
        'architecture': """
{title}
{title_underline}

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------
[Architectural overview]

Design Goals
-----------
* Goal 1
* Goal 2

Implementation Details
--------------------
[Technical implementation details]

.. code-block:: python

    # Implementation example
    class Example:
        pass

Integration Points
----------------
[How this component integrates with others]

Performance Considerations
------------------------
[Performance characteristics and optimizations]

Future Considerations
-------------------
[Planned improvements or potential changes]

See Also
--------
* :doc:`/related/document1`
* :doc:`/related/document2`
""",
        
        'research': """
{title}
{title_underline}

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------
[Research overview]

Background
---------
[Context and previous work]

Methodology
----------
[Research approach]

Results
-------
[Findings and analysis]

.. code-block:: python

    # Example implementation
    def research_function():
        pass

Conclusions
----------
[Key takeaways]

Future Work
----------
[Next steps and future research directions]

References
----------
.. [1] Reference 1
.. [2] Reference 2
"""
    }
    
    def __init__(self, docs_dir: str = "."):
        self.docs_dir = Path(docs_dir).resolve()
        
    def create_doc(
        self,
        title: str,
        template: str,
        directory: Optional[str] = None,
        module_path: Optional[str] = None,
        class_name: Optional[str] = None
    ) -> Path:
        """Create a new documentation file."""
        # Validate template
        if template not in self.TEMPLATES:
            raise ValueError(
                f"Unknown template: {template}. "
                f"Available templates: {', '.join(self.TEMPLATES.keys())}"
            )
            
        # Process title and filename
        filename = title.lower().replace(' ', '_') + '.rst'
        if directory:
            doc_path = self.docs_dir / directory / filename
        else:
            doc_path = self.docs_dir / filename
            
        # Create directory if needed
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate content
        content = self.TEMPLATES[template].format(
            title=title,
            title_underline='=' * len(title),
            module_path=module_path or 'module.path',
            class_name=class_name or 'ClassName'
        )
        
        # Write file
        with open(doc_path, 'w') as f:
            f.write(content.lstrip())
            
        print(f"Created documentation file: {doc_path}")
        return doc_path
        
    def update_index(self, doc_path: Path):
        """Update the index.rst in the document's directory."""
        index_path = doc_path.parent / 'index.rst'
        
        if not index_path.exists():
            # Create new index if it doesn't exist
            title = doc_path.parent.name.title()
            content = [
                f"{title}",
                "=" * len(title),
                "",
                ".. toctree::",
                "   :maxdepth: 2",
                "   :caption: Contents:",
                "",
                f"   {doc_path.stem}"
            ]
            with open(index_path, 'w') as f:
                f.write('\n'.join(content))
        else:
            # Update existing index
            with open(index_path, 'r') as f:
                content = f.read()
                
            # Add new document if not already listed
            if doc_path.stem not in content:
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    if line.strip() == ':caption: Contents:':
                        lines.insert(i + 2, f"   {doc_path.stem}")
                        break
                        
                with open(index_path, 'w') as f:
                    f.write('\n'.join(lines))
                    
def main():
    parser = argparse.ArgumentParser(description="Create new documentation files")
    parser.add_argument('title', help="Document title")
    parser.add_argument(
        '--template',
        choices=DocCreator.TEMPLATES.keys(),
        default='guide',
        help="Template to use"
    )
    parser.add_argument(
        '--directory',
        help="Subdirectory for the document"
    )
    parser.add_argument(
        '--module-path',
        help="Module path for API documentation"
    )
    parser.add_argument(
        '--class-name',
        help="Class name for API documentation"
    )
    
    args = parser.parse_args()
    
    creator = DocCreator()
    doc_path = creator.create_doc(
        args.title,
        args.template,
        args.directory,
        args.module_path,
        args.class_name
    )
    creator.update_index(doc_path)
    
if __name__ == '__main__':
    main()