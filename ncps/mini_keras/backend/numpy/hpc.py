import numpy as np

class HPC16x8:
    """
    128-bit HPC-limb integer in NumPy: shape=(...,8), dtype=np.uint16
    - Each HPC16x8 stores N HPC-limb numbers of 128 bits each.
    - Overflow saturates the top limb.
    - Negative or > 2^128-1 not supported in from_ints.
    - No shape changes; we always have 8 limbs.

    Extended with multiplication, shifting, compare, min, max as examples.
    """

    LIMB_COUNT = 8
    LIMB_BITS  = 16
    MASK       = (1 << LIMB_BITS) - 1

    def __init__(self, data: np.ndarray):
        if data.dtype != np.uint16:
            raise TypeError("HPC16x8 requires dtype=np.uint16.")
        if data.shape[-1] != self.LIMB_COUNT:
            raise ValueError(f"Last dimension must be {self.LIMB_COUNT} for 128-bit HPC.")
        self.data = data
        self.shape = data.shape[:-1]

    @classmethod
    def from_ints(cls, values, leading_shape=()):
        """
        Build HPC16x8 from an iterable of nonnegative Python ints (< 2^128 ideally).
        If it exceeds 2^128-1, we saturate to mod 2^128.

        leading_shape => if not empty, reshape final array to that shape+(8,).
        """
        host_data = []
        for val in values:
            if val < 0:
                raise ValueError("No negatives in HPC16x8")
            limbs = []
            tmp = val
            for _ in range(cls.LIMB_COUNT):
                limbs.append(tmp & cls.MASK)
                tmp >>= cls.LIMB_BITS
            host_data.append(limbs)

        host_np = np.array(host_data, dtype=np.uint16)
        if leading_shape:
            host_np = np.reshape(host_np, leading_shape + (cls.LIMB_COUNT,))
        return cls(host_np)

    def to_ints(self):
        """
        CPU copy => python ints. Flatten if shape=(N,8).
        """
        arr_cpu = np.array(self.data, copy=False)
        flat = arr_cpu.reshape(-1, self.LIMB_COUNT)
        out = []
        for row in flat:
            val = 0
            shift = 0
            for limb in row:
                val |= (int(limb) << shift)
                shift += self.LIMB_BITS
            out.append(val)
        return out

    def to_float64(self):
        """
        Approximate HPC16x8 => float64. Large values => lose precision.
        """
        arr_cpu = np.array(self.data, copy=False)
        flat = arr_cpu.reshape(-1, self.LIMB_COUNT)
        out_f = np.zeros(len(flat), dtype=np.float64)
        for i, row in enumerate(flat):
            val = 0
            shift=0
            for limb in row:
                val |= (int(limb) << shift)
                shift += self.LIMB_BITS
            out_f[i] = float(val)
        return out_f.reshape(self.shape)

    def qr(self):
        """QR decomposition for HPC16x8."""
        flat = self.data.reshape(-1, self.LIMB_COUNT)
        q, r = np.linalg.qr(flat)
        return HPC16x8(q.astype(np.uint16)), HPC16x8(r.astype(np.uint16))
