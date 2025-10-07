# Sphinx Integration Plan

## Overview
Plan to integrate our new architecture documentation into the Sphinx documentation system, leveraging its features while maintaining our logical organization.

## Current Structure vs Sphinx Structure

### Current (Markdown):
```
docs/architecture/
├── abstractions/
│   ├── tensor_abstraction.md
│   ├── layer_abstraction.md
│   ├── gpu_abstraction.md
│   └── liquid_integration.md
├── implementation/
├── knowledge/
├── design/
└── research/
```

### Proposed Sphinx Structure:
```
docs/
├── index.rst                    # Main documentation entry
├── architecture/
│   ├── index.rst               # Architecture overview
│   ├── abstractions/
│   │   ├── index.rst          # Abstractions overview
│   │   ├── tensor.rst         # TensorAbstraction
│   │   ├── layer.rst          # LayerAbstraction
│   │   ├── gpu.rst            # GPUAbstraction
│   │   └── liquid.rst         # Liquid integration
│   ├── implementation/
│   │   ├── index.rst          # Implementation overview
│   │   ├── phase1.rst         # Core abstractions
│   │   └── phase2.rst         # Liquid integration
│   ├── knowledge/
│   │   ├── index.rst          # Knowledge base overview
│   │   ├── insights.rst       # Architectural insights
│   │   └── frameworks.rst     # Framework knowledge
│   └── design/
       ├── index.rst           # Design overview
       ├── core.rst            # Core design
       └── components.rst      # Component design
```

## Integration Steps

1. Directory Structure
   ```bash
   # Create new RST directories
   mkdir -p docs/architecture/{abstractions,implementation,knowledge,design}
   ```

2. Index Files
   ```rst
   # docs/architecture/index.rst
   Architecture
   ============

   .. toctree::
      :maxdepth: 2
      :caption: Contents:

      abstractions/index
      implementation/index
      knowledge/index
      design/index
   ```

3. Content Migration
   - Convert Markdown to RST
   - Maintain all code examples
   - Preserve diagrams and images
   - Add Sphinx directives for better navigation

## Sphinx Features to Use

1. Cross-References
   ```rst
   :ref:`tensor-abstraction`
   :doc:`/architecture/abstractions/tensor`
   ```

2. Code Examples
   ```rst
   .. code-block:: python
      :emphasize-lines: 3,5
      :caption: Example Code
      
      class TensorAbstraction:
          def __init__(self):
              self.backend = get_optimal_backend()
   ```

3. Admonitions
   ```rst
   .. note::
      Important implementation details.

   .. warning::
      Critical considerations.
   ```

4. Auto-Documentation
   ```rst
   .. autoclass:: ncps.abstractions.TensorAbstraction
      :members:
      :undoc-members:
      :show-inheritance:
   ```

## Benefits

1. Documentation Features
   - Automatic table of contents
   - Cross-referencing between docs
   - Code syntax highlighting
   - API documentation integration
   - Search functionality

2. Multiple Output Formats
   - HTML (default)
   - PDF
   - ePub
   - Man pages

3. Development Benefits
   - Version control friendly
   - Easy to maintain
   - Consistent formatting
   - Better collaboration

## Implementation Plan

### Phase 1: Setup
1. Update conf.py configuration
2. Create directory structure
3. Create index.rst files

### Phase 2: Content Migration
1. Convert Markdown to RST
2. Add Sphinx directives
3. Update cross-references

### Phase 3: Enhancement
1. Add API documentation
2. Improve navigation
3. Add search functionality

### Phase 4: Validation
1. Build documentation
2. Check cross-references
3. Validate formatting
4. Test search functionality

## Next Steps

1. Technical Setup
   - Update conf.py
   - Install required extensions
   - Configure theme

2. Content Migration
   - Convert existing MD files
   - Add Sphinx features
   - Update references

3. Documentation
   - Update build instructions
   - Add contribution guidelines
   - Document RST conventions

This plan provides a path to integrate our architecture documentation into Sphinx while maintaining our logical organization and leveraging Sphinx's powerful features.