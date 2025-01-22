import builtins
import contextlib
import functools
import warnings

import mlx.core as mx
import mlx.nn as nn

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import slice_along_axis
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.symbolic_scope import SymbolicScope

SUPPORTS_SPARSE_TENSORS = False
SUPPORTS_RAGGED_TENSORS = False
IS_THREAD_SAFE = True


class Variable(KerasVariable):
    def _initialize(self, value):
        self._value = mx.convert_to_tensor(value)

    def _direct_assign(self, value):
        self._value = mx.convert_to_tensor(value, dtype=self._dtype)

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    # Overload native accessor.
    def __array__(self):
        return self.value.numpy()

def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if sparse:
        raise ValueError("`sparse=True` is not supported with mlx backend")
    if ragged:
        raise ValueError("`ragged=True` is not supported with mlx backend")
    if dtype is not None:
        dtype = standardize_dtype(dtype)
    if isinstance(x, Variable):
        if dtype and dtype != x.dtype:
            return mx.cast(x.value, dtype)
        return x.value
    if not is_tensor(x) and standardize_dtype(dtype) == "bfloat16":
        # Can't create bfloat16 arrays on the fly (e.g. from a h5 Dataset).
        # Instead we convert "as is" (to stored dtype) and cast.
        return mx.cast(mx.convert_to_tensor(x), dtype)
    if dtype is None:
        dtype = result_type(
            *[getattr(item, "dtype", type(item)) for item in tree.flatten(x)]
        )
    return mx.convert_to_tensor(x, dtype=dtype)

def convert_to_numpy(x):
    return x.numpy()

def is_tensor(x):
    return isinstance(x, mx.Tensor)

def shape(x):
    return mx.shape(x)

def cast(x, dtype):
    return mx.cast(x, dtype=dtype)

def cond(pred, true_fn, false_fn):
    return true_fn() if pred else false_fn()

def vectorized_map(function, elements):
    if not isinstance(elements, (list, tuple)):
        return mx.stack([function(x) for x in elements])
    else:
        batch_size = elements[0].shape[0]
        output_store = []
        for index in range(batch_size):
            output_store.append(function([x[index] for x in elements]))
        return mx.stack(output_store)

def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope(), SymbolicScope():

        def has_none_shape(x):
            if isinstance(x, KerasTensor):
                return None in x.shape
            return False

        none_in_shape = any(
            builtins.map(has_none_shape, tree.flatten((args, kwargs)))
        )

        def convert_keras_tensor_to_mlx(x, fill_value=None):
            if isinstance(x, KerasTensor):
                shape = list(x.shape)
                if fill_value:
                    for i, e in enumerate(shape):
                        if e is None:
                            shape[i] = fill_value
                return mx.zeros(shape=shape, dtype=x.dtype)
            return x

        args_1, kwargs_1 = tree.map_structure(
            lambda x: convert_keras_tensor_to_mlx(x, fill_value=83),
            (args, kwargs),
        )
        outputs_1 = fn(*args_1, **kwargs_1)

        outputs = outputs_1

        if none_in_shape:
            args_2, kwargs_2 = tree.map_structure(
                lambda x: convert_keras_tensor_to_mlx(x, fill_value=89),
                (args, kwargs),
            )
            outputs_2 = fn(*args_2, **kwargs_2)

            flat_out_1 = tree.flatten(outputs_1)
            flat_out_2 = tree.flatten(outputs_2)

            flat_out = []
            for x1, x2 in zip(flat_out_1, flat_out_2):
                shape = list(x1.shape)
                for i, e in enumerate(x2.shape):
                    if e != shape[i]:
                        shape[i] = None
                flat_out.append(KerasTensor(shape, standardize_dtype(x1.dtype)))
            outputs = tree.pack_sequence_as(outputs_1, flat_out)

        def convert_mlx_to_keras_tensor(x):
            if is_tensor(x):
                return KerasTensor(x.shape, standardize_dtype(x.dtype))
            return x

        output_spec = tree.map_structure(convert_mlx_to_keras_tensor, outputs)
    return output_spec

def map(f, xs):
    def g(_, x):
        return (), f(x)

    _, ys = scan(g, (), xs)
    return ys

def _interleave(a, b, axis):
    """Given two Tensors of static shape, interleave them along axis."""
    assert (
        a.shape[axis] == b.shape[axis] or a.shape[axis] == b.shape[axis] + 1
    )

    # we want to get a: [a1, a2], b: [b1, b2]
    # to a: [a1, 0, a2, 0], b: [0, b1, 0, b2]
    a_shape = list(mx.shape(a))
    a_shape[axis] = a.shape[axis] * 2 - 1

    b_shape = list(mx.shape(b))
    b_shape[axis] = b.shape[axis] * 2 - 1

    a_dil = mx.zeros(a_shape, dtype=a.dtype)
    mx.copyto(slice_along_axis(a_dil, 0, None, 2, axis), a)
    b_dil = mx.zeros(b_shape, dtype=b.dtype)
    mx.copyto(slice_along_axis(b_dil, 0, None, 2, axis), b)

    a_pad = [[0, 0] for _ in range(a.ndim)]
    a_pad[axis][-1] = 1 if a.shape[axis] == b.shape[axis] else 0

    b_pad = [[0, 0] for _ in range(b.ndim)]
    b_pad[axis] = [1, 0] if a.shape[axis] == b.shape[axis] else [1, 1]

    op = mx.bitwise_or if a.dtype == mx.bool_ else mx.add
    return op(
        mx.pad(a_dil, a_pad),
        mx.pad(b_dil, b_pad),
    )

def scan(f, init, xs=None, length=None, reverse=False, unroll=1):
    # Ref: jax.lax.scan
    if not callable(f):
        raise TypeError(f"`f` should be a callable. Received: f={f}")
    if not isinstance(unroll, bool):
        if not isinstance(unroll, int) or unroll < 1:
            raise ValueError(
                "`unroll` must be an positive integer or boolean. "
                f"Received: unroll={unroll}"
            )
    if xs is None and length is None:
        raise ValueError("Got no `xs` to scan over and `length` not provided.")

    input_is_sequence = tree.is_nested(xs)
    output_is_sequence = tree.is_nested(init)

    def pack_input(x):
        return tree.pack_sequence_as(xs, x) if input_is_sequence else x[0]

    def pack_output(x):
        return tree.pack_sequence_as(init, x) if output_is_sequence else x[0]

    if xs is None:
        xs_flat = []
        n = int(length)
    else:
        xs_flat = tree.flatten(xs)
        xs_flat = [convert_to_tensor(elem) for elem in xs_flat]
        n = int(length) if length is not None else shape(xs_flat[0])[0]

    init_flat = tree.flatten(init)
    init_flat = [convert_to_tensor(init) for init in init_flat]
    init = pack_output(init_flat)
    dummy_y = [mx.zeros_like(init) for init in init_flat]

    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(n)):
        xs_slice = [x[i] for x in xs_flat]
        packed_xs = pack_input(xs_slice) if len(xs_slice) > 0 else None
        carry, y = f(carry, packed_xs)
        ys.append(y if y is not None else dummy_y)
    stacked_y = tree.map_structure(
        lambda *ys: mx.stack(ys), *maybe_reversed(ys)
    )
    return carry, stacked_y
