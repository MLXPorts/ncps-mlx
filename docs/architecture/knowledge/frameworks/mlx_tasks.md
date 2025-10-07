# MLX Integration Tasks

## Phase 1: Core Operations

### 1. MLX Ops Implementation
- [ ] Set up ncps/mlx/ops directory structure
- [ ] Implement array_ops.py
  - [ ] Basic array creation (zeros, ones)
  - [ ] Shape manipulation (reshape, transpose)
  - [ ] Array operations (concatenate, split)
- [ ] Implement math_ops.py
  - [ ] Basic arithmetic (add, multiply)
  - [ ] Matrix operations (matmul)
  - [ ] Reduction operations (mean, sum)
- [ ] Implement nn_ops.py
  - [ ] Activation functions (sigmoid, tanh)
  - [ ] Training ops (dropout)
  - [ ] Normalization (batch_norm, layer_norm)
- [ ] Implement random_ops.py
  - [ ] Distribution sampling (normal, uniform)
  - [ ] Random state management
- [ ] Implement state_ops.py
  - [ ] Variable management
  - [ ] State updates

### 2. MLX Ops Testing
- [ ] Create test suite mirroring NumPy tests
- [ ] Add MLX-specific test cases
- [ ] Implement performance benchmarks
- [ ] Test device placement and optimization

## Phase 2: Layer System

### 1. Base Layer Implementation
- [x] Use MLX's native modules (`mlx.nn`) directly — no custom layer directory
- [x] Delegate parameter management, device placement, and initialisation to MLX
- [x] Rely on `nn.Linear`, `nn.Sequential`, etc. for dense layers and composition

### 2. RNN Implementation
- [ ] Implement RNNCell base class
  - [ ] State management
  - [ ] Time handling
- [ ] Implement RNN layer
  - [ ] Sequence processing
  - [ ] State updates
  - [ ] Return sequences/states

### 3. Layer Testing
- [ ] Base layer tests
- [ ] Dense layer tests
- [ ] RNN tests
- [ ] Integration tests

## Phase 3: Liquid Neurons

### 1. LiquidCell Implementation
- [ ] Port LiquidCell to MLX
  - [ ] Backbone network
  - [ ] State management
  - [ ] Feature processing
- [ ] Port CfCCell to MLX
  - [ ] Mode-specific updates
  - [ ] Time handling
  - [ ] State updates

### 2. Liquid Testing
- [ ] LiquidCell tests
- [ ] CfCCell tests
- [ ] Integration tests
- [ ] Performance benchmarks

## Phase 4: Optimization

### 1. Performance Tuning
- [ ] Profile MLX operations
- [ ] Identify bottlenecks
- [ ] Implement optimizations
- [ ] Benchmark improvements

### 2. Memory Management
- [ ] Analyze memory usage
- [ ] Optimize memory patterns
- [ ] Handle device transfers
- [ ] Test memory efficiency

### 3. Device Support
- [ ] CPU implementation
- [ ] GPU support
- [ ] Device placement strategy
- [ ] Cross-device operations

## Phase 5: Documentation

### 1. API Documentation
- [ ] Document MLX ops
- [ ] Document MLX layers
- [ ] Document liquid neurons
- [ ] Add usage examples

### 2. Migration Guide
- [ ] Write migration overview
- [ ] Create step-by-step guide
- [ ] Document common patterns
- [ ] Add troubleshooting tips

### 3. Performance Guide
- [ ] Document optimization tips
- [ ] Provide benchmarks
- [ ] Explain device usage
- [ ] Share best practices

## Phase 6: Examples

### 1. Basic Examples
- [ ] MLX ops usage
- [ ] Layer creation
- [ ] RNN examples
- [ ] Liquid neuron demos

### 2. Advanced Examples
- [ ] Complex architectures
- [ ] Performance optimization
- [ ] Device management
- [ ] Real-world applications

## Dependencies

- Phase 1 must complete before Phase 2
- Phase 2 must complete before Phase 3
- Phases 4-6 can run partially in parallel
- Documentation should be updated throughout

## Timeline

1. Phase 1: 2 weeks
2. Phase 2: 2 weeks
3. Phase 3: 2 weeks
4. Phase 4: 1 week
5. Phase 5: 1 week
6. Phase 6: 1 week

Total estimated time: 9 weeks

## Success Criteria

1. All tests pass on both CPU and GPU
2. Performance matches or exceeds NumPy version
3. Memory usage is optimized
4. Documentation is complete and clear
5. Examples demonstrate practical usage
6. Migration path is well-documented

## Risk Mitigation

1. Regular testing throughout development
2. Continuous performance monitoring
3. Frequent documentation updates
4. User feedback incorporation
5. Compatibility testing
6. Performance regression checks
