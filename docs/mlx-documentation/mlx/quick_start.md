Curator's note: Prefer the human-authored MLX guides for clarity. - ../docs\_curated/README.md - ../docs\_curated/PYTORCH\_DISSONANCE.md - ../docs\_curated/NUMPY\_USERS.md - ../docs\_curated/COMMON\_PITFALLS.md

[](https://github.com/ml-explore/mlx "Source repository")\- [.rst](https://ml-explore.github.io/mlx/build/html/_sources/usage/quick_start.rst "Download source file") - .pdf

# Quick Start Guide

## Contents

\- [Basics](https://ml-explore.github.io/mlx/build/html/#basics) - [Function and Graph Transformations](https://ml-explore.github.io/mlx/build/html/#function-and-graph-transformations)

# Quick Start Guide

## Basics Import `\`mlx.core\`` and make an [``array``](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array "mlx.core.array"):

```python

import mlx.core as mx
a = mx.array([1, 2, 3, 4])
a.shape
[4]
a.dtype
int32
b = mx.array([1.0, 2.0, 3.0, 4.0])
b.dtype
float32
```

 Operations in MLX are lazy. The outputs of MLX operations are not computed until they are needed. To force an array to be evaluated use [``eval()``](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html#mlx.core.eval "mlx.core.eval"). Arrays will automatically be evaluated in a few cases. For example, inspecting a scalar with [``array.item()``](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.item.html#mlx.core.array.item "mlx.core.array.item"), printing an array, or converting an array from [``array``](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array "mlx.core.array") to [``numpy.ndarray``](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.2)") all automatically evaluate the array.

```python

c = a + b    # c not yet evaluated
mx.eval(c)  # evaluates c
c = a + b
print(c)     # Also evaluates c
array([2, 4, 6, 8], dtype=float32)
c = a + b
import numpy as np
np.array(c)   # Also evaluates c
array([2., 4., 6., 8.], dtype=float32)
```

 See the page on [Lazy Evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html#lazy-eval) for more details.

## Function and Graph Transformations MLX has standard function transformations like [``grad()``](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.grad.html#mlx.core.grad "mlx.core.grad") and [``vmap()``](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vmap.html#mlx.core.vmap "mlx.core.vmap"). Transformations can be composed arbitrarily. For example `\`grad(vmap(grad(fn)))\`` (or any other composition) is allowed.

```python

x = mx.array(0.0)
mx.sin(x)
array(0, dtype=float32)
mx.grad(mx.sin)(x)
array(1, dtype=float32)
mx.grad(mx.grad(mx.sin))(x)
array(-0, dtype=float32)
```

 Other gradient transformations include [``vjp()``](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vjp.html#mlx.core.vjp "mlx.core.vjp") for vector-Jacobian products and [``jvp()``](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.jvp.html#mlx.core.jvp "mlx.core.jvp") for Jacobian-vector products. Use [``value_and_grad()``](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.value_and_grad.html#mlx.core.value_and_grad "mlx.core.value_and_grad") to efficiently compute both a function’s output and gradient with respect to the function’s input.

[](https://ml-explore.github.io/mlx/build/html/install.html "previous page")

previous Build and Install

[](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html "next page")

next Lazy Evaluation

Contents

\- [Basics](https://ml-explore.github.io/mlx/build/html/#basics) - [Function and Graph Transformations](https://ml-explore.github.io/mlx/build/html/#function-and-graph-transformations)

By MLX Contributors

© Copyright 2023, MLX Contributors.
