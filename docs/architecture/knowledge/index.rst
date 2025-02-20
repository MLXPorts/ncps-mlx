Architecture Knowledge Base
===========================

This directory contains architectural insights, framework-specific
knowledge, and deep-dive documentation that informs our design decisions
but isn’t directly part of the implementation specifications.

Directory Structure
-------------------

::

knowledge/
├── insights/           # General architectural insights
│   ├── architectural_insights.md
│   ├── exploration_plan.md
│   ├── mlx_differences.md
│   └── ncp_layer_impact.md
│││││││││││││││││││││││││││
└── frameworks/        # Framework-specific knowledge
    ├── keras_compatible_design.md
    ├── keras_core_neurons.md
    ├── keras3_layer_system.md
    ├── mlx_integration.md
    └── mlx_tasks.md

Purpose
-------

1. Insights

- Document architectural decisions and their rationale
- Capture exploration findings
- Track impact analysis
- Record technical differences between approaches

2. Framework Knowledge

- Framework-specific design considerations
- Integration patterns
- Compatibility requirements
- Framework-specific optimizations

Usage
-----

This knowledge base should be consulted when: 1. Making architectural
decisions 2. Planning framework integrations 3. Understanding design
rationale 4. Exploring optimization opportunities

Relationship to Implementation
------------------------------

While these documents inform our design decisions, they are not direct
implementation specifications. For implementation details, refer to: -
``/architecture/abstractions/`` - Abstraction system design -
``/architecture/implementation/`` - Implementation plans -
``/architecture/design/`` - Core design specifications
