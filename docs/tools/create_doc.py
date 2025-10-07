#!/usr/bin/env python3
"""
Documentation creation tool for NCPS.
Creates new documentation files with proper templates and structure.
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

class DocCreator:
    def __init__(self, docs_dir: str = "."):
        self.docs_dir = Path(docs_dir).resolve()
        self.logs_dir = self.docs_dir / "logs"
        self.templates_dir = Path(__file__).parent / "templates"
        
        # Create necessary directories
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load templates
        self.templates = self._load_templates()
        
    def setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = logging.getLogger("doc_creator")
        self.logger.setLevel(logging.DEBUG)
        
        # Create log file
        log_file = self.logs_dir / f"doc_create_{timestamp}.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def _load_templates(self) -> Dict[str, str]:
        """Load templates from files or use defaults."""
        templates = {}
        
        # Default templates
        default_templates = {
            'guide': 'guide_template.rst',
            'api': 'api_template.rst',
            'example': 'example_template.rst',
            'visualization': 'visualization_template.rst',
            'research': 'research_template.rst',
            'architecture': 'architecture_template.rst',
            'design': 'design_template.rst',
            'implementation': 'implementation_template.rst'
        }
        
        # Try to load from template files first
        for template_name, filename in default_templates.items():
            template_path = self.templates_dir / filename
            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    templates[template_name] = f.read()
            else:
                # Use built-in templates as fallback
                templates[template_name] = self._get_builtin_template(template_name)
                
        return templates
        
    def _get_builtin_template(self, template_name: str) -> str:
        """Get built-in template content."""
        templates = {
            'guide': self._get_guide_template(),
            'api': self._get_api_template(),
            'example': self._get_example_template(),
            'visualization': self._get_visualization_template(),
            'research': self._get_research_template(),
            'architecture': self._get_architecture_template(),
            'design': self._get_design_template(),
            'implementation': self._get_implementation_template()
        }
        return templates.get(template_name, templates['guide'])
        
    def create_doc(
        self,
        title: str,
        template: str,
        directory: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Path:
        """Create a new documentation file."""
        self.logger.info(f"Creating documentation: {title} ({template})")
        
        try:
            # Validate template
            if template not in self.templates:
                raise ValueError(
                    f"Unknown template: {template}. "
                    f"Available templates: {', '.join(self.templates.keys())}"
                )
                
            # Process title and filename
            filename = self._sanitize_filename(title) + '.rst'
            if directory:
                doc_path = self.docs_dir / directory / filename
            else:
                doc_path = self.docs_dir / filename
                
            # Create directory if needed
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare metadata
            meta = {
                'title': title,
                'created_at': datetime.now().isoformat(),
                'template': template,
                **kwargs
            }
            if metadata:
                meta.update(metadata)
                
            # Generate content
            content = self._generate_content(template, meta)
            
            # Write file
            with open(doc_path, 'w', encoding='utf-8') as f:
                if metadata:
                    # Add metadata as YAML front matter
                    f.write('---\n')
                    yaml.dump(meta, f, default_flow_style=False)
                    f.write('---\n\n')
                f.write(content)
                
            self.logger.info(f"Created file: {doc_path}")
            
            # Update index
            self.update_index(doc_path)
            
            return doc_path
            
        except Exception as e:
            self.logger.error(f"Error creating documentation: {str(e)}", exc_info=True)
            raise
            
    def _generate_content(self, template: str, metadata: Dict[str, Any]) -> str:
        """Generate content from template and metadata."""
        content = self.templates[template]
        
        # Basic replacements
        replacements = {
            '{title}': metadata['title'],
            '{title_underline}': '=' * len(metadata['title']),
            '{module_path}': metadata.get('module_path', 'module.path'),
            '{class_name}': metadata.get('class_name', 'ClassName'),
            '{purpose}': metadata.get('purpose', 'specific task')
        }
        
        # Add any additional metadata as replacements
        for key, value in metadata.items():
            if isinstance(value, (str, int, float)):
                replacements[f'{{{key}}}'] = str(value)
                
        # Apply replacements
        for key, value in replacements.items():
            content = content.replace(key, value)
            
        return content.lstrip()
        
    def update_index(self, doc_path: Path):
        """Update the index.rst in the document's directory."""
        self.logger.info(f"Updating index for: {doc_path}")
        
        try:
            index_path = doc_path.parent / 'index.rst'
            
            if not index_path.exists():
                # Create new index
                self._create_new_index(index_path, doc_path)
            else:
                # Update existing index
                self._update_existing_index(index_path, doc_path)
                
        except Exception as e:
            self.logger.error(f"Error updating index: {str(e)}", exc_info=True)
            raise
            
    def _create_new_index(self, index_path: Path, doc_path: Path):
        """Create a new index.rst file."""
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
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
            
        self.logger.info(f"Created new index: {index_path}")
        
    def _update_existing_index(self, index_path: Path, doc_path: Path):
        """Update an existing index.rst file."""
        with open(index_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if doc_path.stem not in content:
            lines = content.splitlines()
            toctree_end = -1
            
            # Find the end of the toctree directive
            for i, line in enumerate(lines):
                if line.strip() == ':caption: Contents:':
                    toctree_end = i + 1
                    # Skip any existing entries
                    while toctree_end < len(lines) and (not lines[toctree_end].strip() or lines[toctree_end].startswith(' ')):
                        toctree_end += 1
                    break
                    
            if toctree_end != -1:
                # Insert new entry
                lines.insert(toctree_end, f"   {doc_path.stem}")
                
                with open(index_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                    
                self.logger.info(f"Updated index: {index_path}")
                
    def _sanitize_filename(self, title: str) -> str:
        """Convert title to a valid filename."""
        # Replace spaces and special characters
        filename = title.lower()
        filename = ''.join(c if c.isalnum() else '_' for c in filename)
        # Remove multiple underscores
        filename = '_'.join(filter(None, filename.split('_')))
        return filename
        
    # Template getters
    def _get_guide_template(self) -> str:
        return """
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

    # Example code
    from ncps.mlx import CfC
    
    model = CfC(units=64)
    model.compile(optimizer='adam', loss='mse')

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
"""
        
    def _get_api_template(self) -> str:
        return """
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

    # Basic usage
    from ncps import {class_name}
    
    model = {class_name}()
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, epochs=10)

Functions
---------
[Document any standalone functions]

See Also
--------
* :doc:`/related/document1`
* :doc:`/related/document2`
"""
        
    def _get_example_template(self) -> str:
        return """
{title}
{title_underline}

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------
This example demonstrates how to use {class_name} for {purpose}.

Prerequisites
------------
Before running this example, make sure you have:

* NCPS installed
* Required dependencies: numpy, matplotlib

Code Example
-----------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from ncps.{module_path} import {class_name}
    
    # Create model
    model = {class_name}(
        units=64,
        activation='tanh'
    )
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse'
    )
    
    # Generate data
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    
    # Train model
    history = model.fit(
        x, y,
        epochs=100,
        batch_size=32
    )
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

Explanation
----------
[Explain how the code works and key concepts]

Results
-------
[Show and explain the results]

See Also
--------
* :doc:`/api/{module_path}`
* :doc:`/guides/related_guide`
"""
        
    def _get_visualization_template(self) -> str:
        return """
{title}
{title_underline}

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------
[Visualization overview]

Basic Usage
----------

.. code-block:: python

    from ncps.visualization import {class_name}
    
    # Create visualizer
    visualizer = {class_name}()
    
    # Configure options
    visualizer.set_options(
        figsize=(10, 6),
        style='dark'
    )
    
    # Generate visualization
    visualizer.plot()

Advanced Features
---------------

1. Custom Styling
~~~~~~~~~~~~~~~~

.. code-block:: python

    visualizer.set_style(
        colors=['#1f77b4', '#ff7f0e'],
        markers=['o', 's'],
        line_styles=['-', '--']
    )

2. Interactive Elements
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    visualizer.add_interactive_elements(
        tooltips=True,
        zoom=True,
        pan=True
    )

3. Export Options
~~~~~~~~~~~~~~~

.. code-block:: python

    visualizer.save(
        'visualization.png',
        dpi=300,
        transparent=True
    )

Best Practices
-------------
* Keep visualizations simple and focused
* Use consistent styling
* Add proper labels and legends
* Consider colorblind-friendly palettes

See Also
--------
* :doc:`/guides/visualization`
* :doc:`/api/visualization`
"""
        
    def _get_research_template(self) -> str:
        return """
{title}
{title_underline}

.. contents:: Table of Contents
   :local:
   :depth: 2

Abstract
--------
[Brief summary of the research]

Background
----------
[Context and previous work]

Methodology
----------
[Research approach and methods]

Results
-------
[Findings and analysis]

Discussion
----------
[Interpretation and implications]

Future Work
----------
[Next steps and open questions]

References
----------
.. [1] Reference 1
.. [2] Reference 2
"""
        
    def _get_architecture_template(self) -> str:
        return """
{title}
{title_underline}

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------
[High-level architecture overview]

Design Goals
-----------
* Goal 1
* Goal 2

System Components
---------------
[Major system components]

Integration Points
----------------
[How components interact]

Performance Considerations
------------------------
[Performance requirements and optimizations]

Security Considerations
---------------------
[Security requirements and measures]

Deployment Strategy
-----------------
[Deployment and scaling approach]

Future Considerations
-------------------
[Planned improvements and scalability]

See Also
--------
* :doc:`/architecture/related_doc`
"""
        
    def _get_design_template(self) -> str:
        return """
{title}
{title_underline}

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------
[Design overview]

Requirements
-----------
[Functional and non-functional requirements]

Design Details
-------------
[Detailed design specifications]

Implementation Notes
------------------
[Implementation guidelines]

Testing Strategy
--------------
[Testing approach and requirements]

See Also
--------
* :doc:`/design/related_doc`
"""
        
    def _get_implementation_template(self) -> str:
        return """
{title}
{title_underline}

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------
[Implementation overview]

Prerequisites
------------
[Required setup and dependencies]

Implementation Details
--------------------
[Detailed implementation steps]

Code Examples
------------
[Example code and usage]

Testing
-------
[Testing procedures and validation]

Deployment
---------
[Deployment instructions]

See Also
--------
* :doc:`/implementation/related_doc`
"""
        
def main():
    parser = argparse.ArgumentParser(description="Create new documentation files")
    parser.add_argument('title', help="Document title")
    parser.add_argument(
        '--template',
        help="Template to use",
        default='guide'
    )
    parser.add_argument(
        '--directory',
        help="Subdirectory for the document"
    )
    parser.add_argument(
        '--metadata',
        help="JSON string of additional metadata",
        type=json.loads,
        default='{}'
    )
    parser.add_argument(
        '--module-path',
        help="Module path for API documentation"
    )
    parser.add_argument(
        '--class-name',
        help="Class name for API documentation"
    )
    parser.add_argument(
        '--purpose',
        help="Purpose description for example documentation"
    )
    
    args = parser.parse_args()
    
    try:
        creator = DocCreator()
        doc_path = creator.create_doc(
            args.title,
            args.template,
            args.directory,
            args.metadata,
            module_path=args.module_path,
            class_name=args.class_name,
            purpose=args.purpose
        )
        print(f"Successfully created: {doc_path}")
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
        
if __name__ == '__main__':
    sys.exit(main())