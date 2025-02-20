# NCPS Refactoring Plan

## Overview
Following MLX's example, we'll create a lightweight, focused implementation for liquid neurons that mirrors Keras's interface while maintaining our own infrastructure.

## Core Components

### 1. Base Classes
- `BaseCell`: Core liquid neuron functionality
  - Wiring configuration
  - Time-based state updates
  - Backbone network support
  - Modular activation functions

- `BaseRNN`: Sequence processing wrapper
  - Time delta handling
  - State management
  - Bidirectional support
  - Sequence return options

### 2. ODE Solvers
Implement efficient solvers in `ode_solvers.py`:
- Euler method
- RK4 method
- Semi-implicit method
- Focus on neural ODE applications

### 3. Cell Implementations
Each cell type inherits from BaseCell:

#### CfC (Closed-form Continuous-time)
- Pure mode with direct ODE solution
- Gated mode with time interpolation
- Backbone network support
- Time-scaled state updates

#### LTC (Liquid Time-Constant)
- Time-based decay rates
- Adaptive time constants
- State interpolation

#### CTRNN (Continuous-Time RNN)
- Basic continuous-time updates
- Simple but efficient implementation

### 4. Utility Modules
- `liquid_utils.py`: Common functions
- `typing.py`: Type definitions
- `wiring.py`: Connection patterns

## Implementation Steps

1. Base Infrastructure
   - Create base classes
   - Implement ODE solvers
   - Set up utility modules

2. Cell Implementations
   - Port CfC implementation
   - Adapt LTC implementation
   - Add CTRNN implementation

3. Testing & Validation
   - Unit tests for each component
   - Integration tests
   - Performance benchmarks

## Key Differences from Previous Implementation

1. Architecture
   - More focused scope
   - Cleaner inheritance
   - Better separation of concerns

2. Features
   - Simplified backbone networks
   - More efficient ODE solving
   - Better time handling

3. Integration
   - Cleaner Keras interface
   - More consistent API
   - Better error handling

## Migration Path

1. Phase 1: Core Infrastructure
   - Base classes
   - ODE solvers
   - Utility modules

2. Phase 2: Cell Implementations
   - One cell type at a time
   - Full test coverage
   - Documentation updates

3. Phase 3: API Refinement
   - Clean up interfaces
   - Add convenience methods
   - Improve error messages

## Benefits

1. Maintainability
   - Clearer code structure
   - Better separation of concerns
   - Easier to extend

2. Performance
   - More efficient implementations
   - Better memory usage
   - Faster training

3. Usability
   - Cleaner API
   - Better documentation
   - More intuitive interfaces